import os
import sys
import cv2
import torch
import shutil
import glob
import multiprocessing
import threading
from tqdm import tqdm
from .config import ProcessingConfig
from .ocr import OCRProcessor
from .patterns import is_sensitive
from .blur import apply_blur
from .ffmpeg import merge_audio, segment_video, concat_videos
from .worker import worker_process

def progress_listener(q, total_frames: int):
    pbar = tqdm(total=total_frames, desc="Processing Progress")
    processed = 0
    while True:
        try:
            val = q.get(timeout=1.0)
            if val == 'done':
                break
            processed += val
            if processed > total_frames:
                pbar.update(max(0, total_frames - (processed - val)))
            else:
                pbar.update(val)
        except Exception:
            pass
    if processed < total_frames:
        pbar.update(total_frames - processed)
    pbar.close()

def resolve_output_path(config: ProcessingConfig) -> None:
    if not config.output_path:
        base, ext = os.path.splitext(config.input_path)
        config.output_path = f"{base}_redacted{ext}"
    elif os.path.isdir(config.output_path) or config.output_path.endswith('/') or config.output_path.endswith('\\'):
        os.makedirs(config.output_path, exist_ok=True)
        filename = os.path.basename(config.input_path)
        base, ext = os.path.splitext(filename)
        config.output_path = os.path.join(config.output_path, f"{base}_redacted{ext}")
    else:
        parent_dir = os.path.dirname(os.path.abspath(config.output_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

def run_sequential(config: ProcessingConfig, use_gpu: bool) -> None:
    resolve_output_path(config)
    torch.set_num_threads(config.threads)
    reader = OCRProcessor(use_gpu=use_gpu)
    
    cap = cv2.VideoCapture(config.input_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if config.debug:
        print("Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Mode: {config.match_config.mode}")
        print(f"  Frame Skip: {config.frame_skip}")
        print(f"  Max OCR Dimension: {config.max_ocr_dim}")
        print(f"  Static Frame Skip Threshold: {config.static_thresh}")
        print(f"  Page Change Detection Threshold: {config.change_thresh}")
        if config.match_config.custom_keywords:
            print(f"  Custom keywords: {config.match_config.custom_keywords}")
        
    temp_output = config.output_path
    if not config.no_audio:
        base, ext = os.path.splitext(config.output_path)
        temp_output = base + "_temp_no_audio" + ext
            
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    active_blur_boxes: list = []
    last_ocr_blur_boxes: list = []
    prev_frame_gray = None
    last_ocr_frame_gray = None
    last_ocr_frame_idx = 0
    transition_active = False
    
    diff_scale = 540 / max(width, height)
    diff_width = int(width * diff_scale)
    diff_height = int(height * diff_scale)
    
    pbar = tqdm(total=total_frames, desc="Processing Progress")
    
    frame_count = 0
    ocr_runs = 0
    static_skips = 0
    transition_triggers = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        diff_frame = cv2.resize(frame, (diff_width, diff_height))
        gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        
        trigger_ocr = False
        
        if config.frame_skip > 1 and prev_frame_gray is not None:
            diff_consecutive = cv2.absdiff(gray, prev_frame_gray).mean()
            if diff_consecutive > config.change_thresh:
                if not transition_active:
                    transition_active = True
                    active_blur_boxes = []
            else:
                if transition_active:
                    transition_active = False
                    trigger_ocr = True
                    transition_triggers += 1
                    
        if not trigger_ocr:
            if (frame_count - last_ocr_frame_idx) >= config.frame_skip:
                trigger_ocr = True
                
        if frame_count == 0:
            trigger_ocr = True
            
        if trigger_ocr and frame_count > 0:
            if last_ocr_frame_gray is not None:
                diff_from_last = cv2.absdiff(gray, last_ocr_frame_gray).mean()
                if diff_from_last < config.static_thresh:
                    trigger_ocr = False
                    static_skips += 1
                    last_ocr_frame_idx = frame_count
                    active_blur_boxes = last_ocr_blur_boxes
                    
        if trigger_ocr:
            scale = 1.0
            if max(width, height) > config.max_ocr_dim:
                scale = config.max_ocr_dim / max(width, height)
                ocr_width = int(width * scale)
                ocr_height = int(height * scale)
                ocr_frame = cv2.resize(frame, (ocr_width, ocr_height))
            else:
                ocr_frame = frame
                
            active_blur_boxes = []
            ocr_frame_rgb = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2RGB)
            results = reader.detect_text(
                ocr_frame_rgb,
                max_ocr_dim=config.max_ocr_dim,
                mag_ratio=config.mag_ratio,
                adjust_contrast=not config.no_contrast,
                min_size=config.min_size
            )
            ocr_runs += 1
            
            if config.debug:
                print(f"\n[Frame {frame_count}] OCR found {len(results)} regions:")
                
            for bbox, text, confidence in results:
                sensitive_match = is_sensitive(text, config.match_config)
                if config.debug:
                    print(f"  '{text}' (conf: {confidence:.2f}) -> Blur: {sensitive_match}")
                if sensitive_match:
                    if scale != 1.0:
                        scaled_bbox = [[int(pt[0] / scale), int(pt[1] / scale)] for pt in bbox]
                        active_blur_boxes.append(scaled_bbox)
                    else:
                        active_blur_boxes.append(bbox)
                        
            last_ocr_frame_gray = gray
            last_ocr_frame_idx = frame_count
            last_ocr_blur_boxes = active_blur_boxes
            
        for bbox in active_blur_boxes:
            apply_blur(frame, bbox, padding=config.padding, kernel_size=config.blur_strength)
            
        out.write(frame)
        prev_frame_gray = gray
        pbar.update(1)
        frame_count += 1
        
    cap.release()
    out.release()
    pbar.close()
    
    if config.debug:
        print("Video frame processing completed.")
        print("Performance Stats:")
        print(f"  Total processed frames: {frame_count}")
        print(f"  OCR inferences run: {ocr_runs}")
        print(f"  Static frame OCR skips: {static_skips}")
        print(f"  Transition-based OCR triggers: {transition_triggers}")
        
    if not config.no_audio and temp_output != config.output_path:
        merge_audio(temp_output, config.input_path, config.output_path)
    else:
        if temp_output != config.output_path:
            if os.path.exists(config.output_path):
                try:
                    os.remove(config.output_path)
                except Exception:
                    pass
            os.rename(temp_output, config.output_path)
            
    print(f"Finished! Output written to: {os.path.abspath(config.output_path)}")

def run_parallel(config: ProcessingConfig, use_gpu: bool, num_workers: int) -> None:
    resolve_output_path(config)
    cap = cv2.VideoCapture(config.input_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    _, ext = os.path.splitext(config.input_path)
    output_dir = os.path.dirname(os.path.abspath(config.output_path))
    temp_dir = os.path.join(output_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    if config.debug:
        print("Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Workers: {num_workers}")
        print(f"  PyTorch threads per worker: {config.threads}")
        print(f"  Frame Skip: {config.frame_skip}")
        print(f"  Max OCR Dimension: {config.max_ocr_dim}")
        print(f"  Static Frame Skip Threshold: {config.static_thresh}")
        print(f"  Page Change Detection Threshold: {config.change_thresh}")
        
    print("\n[Step 1/4] Segmenting input video into ~60-second chunks...")
    success, err = segment_video(os.path.abspath(config.input_path), temp_dir, ext, segment_time=60)
    if not success:
        print("Error: Video segmentation failed!")
        print(err)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    chunk_files = sorted(glob.glob(os.path.join(temp_dir, f"chunk_*{ext}")))
    if not chunk_files:
        print("Error: No chunks were generated!")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    print(f"Successfully segmented video into {len(chunk_files)} chunks.")
    
    print("\n[Step 2/4] Processing chunks in parallel process pool...")
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    
    listener_thread = threading.Thread(target=progress_listener, args=(progress_queue, total_frames))
    listener_thread.start()
    
    tasks = []
    for i, chunk_path in enumerate(chunk_files):
        out_chunk_path = os.path.join(temp_dir, f"processed_chunk_{i:03d}{ext}")
        tasks.append((i, chunk_path, out_chunk_path, config, progress_queue, use_gpu))
        
    results = []
    try:
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(worker_process, tasks)
    except Exception as e:
        print(f"\nError occurred during parallel pool execution: {e}")
        progress_queue.put('done')
        listener_thread.join()
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    progress_queue.put('done')
    listener_thread.join()
    
    success = True
    total_ocr_runs = 0
    total_static_skips = 0
    total_transition_triggers = 0
    total_processed_frames = 0
    
    for res in results:
        if not res['success']:
            print(f"Error in worker processing chunk {res['chunk_idx']}: {res.get('error')}")
            success = False
        else:
            total_ocr_runs += res['ocr_runs']
            total_static_skips += res['static_skips']
            total_transition_triggers += res['transition_triggers']
            total_processed_frames += res['frame_count']
            
    if not success:
        print("One or more workers failed. Cleaning up and exiting.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    if config.debug:
        print("\nProcessing completed for all chunks.")
        print("Aggregate Performance Stats:")
        print(f"  Total processed frames: {total_processed_frames}")
        print(f"  Total OCR inferences run: {total_ocr_runs}")
        print(f"  Total Static frame OCR skips: {total_static_skips}")
        print(f"  Total Transition-based OCR triggers: {total_transition_triggers}")
        
    print("\n[Step 3/4] Concatenating processed chunks...")
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_path, 'w', encoding='utf-8') as f:
        for i in range(len(chunk_files)):
            rel_name = f"processed_chunk_{i:03d}{ext}"
            f.write(f"file '{rel_name}'\n")
            
    temp_merged_video = os.path.join(temp_dir, f"merged_no_audio{ext}")
    success, err = concat_videos("concat_list.txt", temp_dir, f"merged_no_audio{ext}")
    if not success:
        print("Error: FFmpeg concat failed!")
        print(err)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    print("\n[Step 4/4] Merging audio and finalizing output...")
    if not config.no_audio:
        merge_audio(temp_merged_video, os.path.abspath(config.input_path), os.path.abspath(config.output_path))
    else:
        abs_output = os.path.abspath(config.output_path)
        if os.path.exists(abs_output):
            try:
                os.remove(abs_output)
            except Exception:
                pass
        shutil.copy2(temp_merged_video, abs_output)
        
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\nSuccess! Final processed video written to: {os.path.abspath(config.output_path)}")
