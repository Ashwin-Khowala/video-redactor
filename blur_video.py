import sys
# Set console encoding to UTF-8 to prevent encoding issues with EasyOCR's console output on Windows
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import re
import argparse
import subprocess
import cv2
import torch
import easyocr
import numpy as np
from tqdm import tqdm
import multiprocessing
import threading
import shutil
import glob

# Default patterns
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
# Generic phone regex: matches digits separated by spaces, dashes, parentheses, or dots
PHONE_REGEX = re.compile(r'(\+?\d[\d\s\-().]{6,20}\d)')

def matches_sensitive(text, mode, custom_keywords=None):
    """
    Check if the text matches any sensitive pattern.
    """
    text_clean = text.strip()
    
    if mode == 'all':
        return True
        
    # Check custom keywords if provided
    if custom_keywords:
        for kw in custom_keywords:
            if kw.lower() in text_clean.lower():
                return True
                
    lower_text = text_clean.lower()
    
    # --- Robust Email Heuristic ---
    # 1. Contains @ symbol (almost always an email in screen recordings)
    if "@" in lower_text:
        return True
        
    # 2. Contains common email domain keywords or common OCR misreadings of them
    email_keywords = [
        'gmail', 'gmall', 'gmai', 'gma1', 'outlook', 'yahoo', 'hotmail', 'icloud', 
        'protonmail', 'kiit.ac', 'kiitacin', 'clubfyndr', 'digilocker'
    ]
    for kw in email_keywords:
        if kw in lower_text:
            return True
            
    # 3. Strict regex fallback
    if EMAIL_REGEX.search(text_clean):
        return True
        
    # --- Robust Phone Heuristic ---
    digits_only = "".join(c for c in text_clean if c.isdigit())
    if 7 <= len(digits_only) <= 15:
        # Check standard phone patterns or generic digit groupings
        if PHONE_REGEX.search(text_clean):
            # Avoid matching timestamps like "2026-06-10" or "22:02:35"
            if ":" in text_clean or ("-" in text_clean and len(digits_only) == 8):
                # likely date/time
                return False
            return True
            
    return False

def apply_blur(frame, bbox, padding=10, kernel_size=35):
    """
    Applies Gaussian blur to a bounding box ROI in the frame.
    """
    h, w, _ = frame.shape
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    
    min_x = max(0, int(min(x_coords)) - padding)
    max_x = min(w, int(max(x_coords)) + padding)
    min_y = max(0, int(min(y_coords)) - padding)
    max_y = min(h, int(max(y_coords)) + padding)
    
    if max_x > min_x and max_y > min_y:
        roi = frame[min_y:max_y, min_x:max_x]
        # Kernel size must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        frame[min_y:max_y, min_x:max_x] = blurred_roi

def merge_audio(video_no_audio, video_with_audio, final_output):
    print("Merging audio using FFmpeg...")
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_no_audio,
            '-i', video_with_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',  # copy video, encode audio to standard aac
            '-map', '0:v:0',
            '-map', '1:a:0?',
            final_output
        ]
        print(f"Running command: {' '.join(ffmpeg_cmd)}")
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print("Audio merge successful!")
            try:
                if os.path.exists(video_no_audio):
                    os.remove(video_no_audio)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {video_no_audio}: {e}")
        else:
            print("Warning: FFmpeg audio merge failed with return code", res.returncode)
            print("FFmpeg error output:")
            print(res.stderr)
            print("Fallback: Keeping the video without audio.")
            try:
                if os.path.exists(final_output):
                    os.remove(final_output)
            except Exception:
                pass
            try:
                os.rename(video_no_audio, final_output)
            except Exception as e:
                print(f"Error renaming fallback video: {e}")
    except Exception as e:
        print(f"Error merging audio: {e}")
        if not os.path.exists(final_output) and os.path.exists(video_no_audio):
            try:
                os.rename(video_no_audio, final_output)
            except Exception:
                pass

def worker_process(chunk_idx, input_path, output_path, args, progress_queue):
    # Cap PyTorch CPU execution threads for this worker
    torch.set_num_threads(args.threads)
    
    # Initialize reader inside worker process
    reader = easyocr.Reader(['en'])
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {
            'chunk_idx': chunk_idx,
            'success': False,
            'error': f"Failed to open chunk video {input_path}"
        }
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    active_blur_boxes = []
    last_ocr_blur_boxes = []
    prev_frame_gray = None
    last_ocr_frame_gray = None
    last_ocr_frame_idx = 0
    transition_active = False
    
    diff_scale = 540 / max(width, height)
    diff_width = int(width * diff_scale)
    diff_height = int(height * diff_scale)
    
    custom_keywords = [k.strip() for k in args.keywords.split(',')] if args.keywords else []
    
    frame_count = 0
    ocr_runs = 0
    static_skips = 0
    transition_triggers = 0
    
    progress_interval = 10
    accumulated_progress = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        diff_frame = cv2.resize(frame, (diff_width, diff_height))
        gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        
        trigger_ocr = False
        
        # 1. Page transition check
        if prev_frame_gray is not None:
            diff_consecutive = cv2.absdiff(gray, prev_frame_gray).mean()
            if diff_consecutive > args.change_thresh:
                if not transition_active:
                    transition_active = True
                    active_blur_boxes = []
            else:
                if transition_active:
                    transition_active = False
                    trigger_ocr = True
                    transition_triggers += 1
                    
        # 2. Periodic check
        if not trigger_ocr:
            if (frame_count - last_ocr_frame_idx) >= args.frame_skip:
                trigger_ocr = True
                
        # 3. First frame check
        if frame_count == 0:
            trigger_ocr = True
            
        # 4. Static check
        if trigger_ocr and frame_count > 0:
            if last_ocr_frame_gray is not None:
                diff_from_last = cv2.absdiff(gray, last_ocr_frame_gray).mean()
                if diff_from_last < args.static_thresh:
                    trigger_ocr = False
                    static_skips += 1
                    last_ocr_frame_idx = frame_count
                    active_blur_boxes = last_ocr_blur_boxes  # Restore cached boxes!
                    
        if trigger_ocr:
            scale = 1.0
            if max(width, height) > args.max_ocr_dim:
                scale = args.max_ocr_dim / max(width, height)
                ocr_width = int(width * scale)
                ocr_height = int(height * scale)
                ocr_frame = cv2.resize(frame, (ocr_width, ocr_height))
            else:
                ocr_frame = frame
                
            active_blur_boxes = []
            results = reader.readtext(
                ocr_frame, 
                canvas_size=args.max_ocr_dim, 
                mag_ratio=args.mag_ratio, 
                adjust_contrast=not args.no_contrast,
                min_size=args.min_size
            )
            ocr_runs += 1
            
            for bbox, text, confidence in results:
                sensitive_match = matches_sensitive(text, args.mode, custom_keywords)
                if sensitive_match:
                    if scale != 1.0:
                        scaled_bbox = []
                        for pt in bbox:
                            scaled_bbox.append([int(pt[0] / scale), int(pt[1] / scale)])
                        active_blur_boxes.append(scaled_bbox)
                    else:
                        active_blur_boxes.append(bbox)
                        
            last_ocr_frame_gray = gray
            last_ocr_frame_idx = frame_count
            last_ocr_blur_boxes = active_blur_boxes  # Update cache
            
        for bbox in active_blur_boxes:
            apply_blur(frame, bbox, padding=args.padding, kernel_size=args.blur_strength)
            
        out.write(frame)
        prev_frame_gray = gray
        frame_count += 1
        
        accumulated_progress += 1
        if accumulated_progress >= progress_interval:
            progress_queue.put(accumulated_progress)
            accumulated_progress = 0
            
    if accumulated_progress > 0:
        progress_queue.put(accumulated_progress)
        
    cap.release()
    out.release()
    
    return {
        'chunk_idx': chunk_idx,
        'success': True,
        'ocr_runs': ocr_runs,
        'static_skips': static_skips,
        'transition_triggers': transition_triggers,
        'frame_count': frame_count
    }

def worker_process_helper(args_tuple):
    return worker_process(*args_tuple)

def progress_listener(q, total_frames):
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

def run_sequential(args):
    custom_keywords = [k.strip() for k in args.keywords.split(',')] if args.keywords else []
    
    print(f"Setting PyTorch CPU threads to {args.threads}...")
    torch.set_num_threads(args.threads)
    
    print(f"Initializing EasyOCR for English...")
    reader = easyocr.Reader(['en'])
    
    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        sys.exit(1)
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Mode: {args.mode}")
    print(f"  Frame Skip: {args.frame_skip}")
    print(f"  Max OCR Dimension: {args.max_ocr_dim}")
    print(f"  Static Frame Skip Threshold: {args.static_thresh}")
    print(f"  Page Change Detection Threshold: {args.change_thresh}")
    if custom_keywords:
        print(f"  Custom keywords to blur: {custom_keywords}")
        
    temp_output = args.output
    if not args.no_audio:
        base, ext = os.path.splitext(args.output)
        temp_output = base + "_temp_no_audio" + ext
            
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    active_blur_boxes = []
    last_ocr_blur_boxes = []
    prev_frame_gray = None
    last_ocr_frame_gray = None
    last_ocr_frame_idx = 0
    transition_active = False
    
    diff_scale = 540 / max(width, height)
    diff_width = int(width * diff_scale)
    diff_height = int(height * diff_scale)
    
    print("Processing video frames...")
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
        
        # 1. Page transition check
        if prev_frame_gray is not None:
            diff_consecutive = cv2.absdiff(gray, prev_frame_gray).mean()
            if diff_consecutive > args.change_thresh:
                if not transition_active:
                    transition_active = True
                    active_blur_boxes = []
            else:
                if transition_active:
                    transition_active = False
                    trigger_ocr = True
                    transition_triggers += 1
                    
        # 2. Periodic check
        if not trigger_ocr:
            if (frame_count - last_ocr_frame_idx) >= args.frame_skip:
                trigger_ocr = True
                
        # 3. First frame check
        if frame_count == 0:
            trigger_ocr = True
            
        # 4. Static check
        if trigger_ocr and frame_count > 0:
            if last_ocr_frame_gray is not None:
                diff_from_last = cv2.absdiff(gray, last_ocr_frame_gray).mean()
                if diff_from_last < args.static_thresh:
                    trigger_ocr = False
                    static_skips += 1
                    last_ocr_frame_idx = frame_count
                    active_blur_boxes = last_ocr_blur_boxes  # Restore cached boxes!
                    
        if trigger_ocr:
            scale = 1.0
            if max(width, height) > args.max_ocr_dim:
                scale = args.max_ocr_dim / max(width, height)
                ocr_width = int(width * scale)
                ocr_height = int(height * scale)
                ocr_frame = cv2.resize(frame, (ocr_width, ocr_height))
            else:
                ocr_frame = frame
                
            active_blur_boxes = []
            results = reader.readtext(
                ocr_frame, 
                canvas_size=args.max_ocr_dim, 
                mag_ratio=args.mag_ratio, 
                adjust_contrast=not args.no_contrast,
                min_size=args.min_size
            )
            ocr_runs += 1
            
            if args.debug:
                print(f"\n[Frame {frame_count}] OCR found {len(results)} text regions:")
                
            for bbox, text, confidence in results:
                sensitive_match = matches_sensitive(text, args.mode, custom_keywords)
                if args.debug:
                    print(f"  '{text}' (conf: {confidence:.2f}) -> Blur: {sensitive_match}")
                if sensitive_match:
                    if scale != 1.0:
                        scaled_bbox = []
                        for pt in bbox:
                            scaled_bbox.append([int(pt[0] / scale), int(pt[1] / scale)])
                        active_blur_boxes.append(scaled_bbox)
                    else:
                        active_blur_boxes.append(bbox)
                        
            last_ocr_frame_gray = gray
            last_ocr_frame_idx = frame_count
            last_ocr_blur_boxes = active_blur_boxes  # Update cache
            
        for bbox in active_blur_boxes:
            apply_blur(frame, bbox, padding=args.padding, kernel_size=args.blur_strength)
            
        out.write(frame)
        prev_frame_gray = gray
        pbar.update(1)
        frame_count += 1
        
    cap.release()
    out.release()
    pbar.close()
    
    print("Video frame processing completed.")
    print(f"Performance Stats:")
    print(f"  Total processed frames: {frame_count}")
    print(f"  OCR inferences run: {ocr_runs}")
    print(f"  Static frame OCR skips: {static_skips}")
    print(f"  Transition-based OCR triggers: {transition_triggers}")
    
    # Merge audio back if required and available
    if not args.no_audio and temp_output != args.output:
        merge_audio(temp_output, args.input, args.output)
    else:
        if temp_output != args.output:
            try:
                if os.path.exists(args.output):
                    os.remove(args.output)
            except Exception:
                pass
            try:
                os.rename(temp_output, args.output)
            except Exception as e:
                print(f"Error moving file to final output: {e}")
                
    print(f"Finished! Output written to: {os.path.abspath(args.output)}")

def main():
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Blur sensitive info (emails, mobile numbers) from videos using EasyOCR.")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", required=True, help="Path to output video file")
    parser.add_argument("--mode", "-m", choices=['patterns', 'all'], default='patterns', 
                        help="all: blur all text; patterns: blur emails and phone numbers (default)")
    parser.add_argument("--frame-skip", "-f", type=int, default=12, 
                        help="Run OCR every N frames. Skipped frames reuse detections to speed up processing (default: 12)")
    parser.add_argument("--padding", "-p", type=int, default=25, 
                        help="Pixels of padding to expand the blur box around detected text (default: 25)")
    parser.add_argument("--blur-strength", "-s", type=int, default=35, 
                        help="Gaussian blur kernel size (must be odd, default: 35)")
    parser.add_argument("--keywords", "-k", type=str, 
                        help="Comma-separated custom words/names to blur")
    parser.add_argument("--max-ocr-dim", "-d", type=int, default=1080,
                        help="Maximum width or height of the frame when passed to EasyOCR. (default: 1080)")
    parser.add_argument("--static-thresh", "-t", type=float, default=0.8,
                        help="Grayscale mean pixel difference threshold to classify a frame as static and skip OCR (default: 0.8)")
    parser.add_argument("--change-thresh", "-c", type=float, default=1.0,
                        help="Consecutive frame difference threshold to trigger OCR on scene/page changes (default: 1.0)")
    parser.add_argument("--no-contrast", action="store_true",
                        help="Disable EasyOCR contrast adjustment for speed (may reduce detection accuracy)")
    parser.add_argument("--mag-ratio", type=float, default=1.0,
                        help="EasyOCR detection magnification ratio (default: 1.0)")
    parser.add_argument("--min-size", type=int, default=10,
                        help="Minimum size of text to detect (default: 10)")
    parser.add_argument("--threads", "-th", type=int, default=4,
                        help="Number of threads for PyTorch CPU execution per worker (default: 4)")
    parser.add_argument("--no-audio", action="store_true", 
                        help="Skip merging audio from original video at the end")
    parser.add_argument("--workers", default="1",
                        help="Number of parallel worker processes. Use 'auto' to automatically choose based on CPU cores, or 1 to run sequentially (default: 1)")
    parser.add_argument("--debug", action="store_true",
                        help="Print OCR detections to console for debugging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
        
    # Resolve absolute path for inputs
    abs_input = os.path.abspath(args.input)
    abs_output = os.path.abspath(args.output)
    
    # Parse workers parameter
    if args.workers.lower() == 'auto':
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, max(1, cpu_count // 3))
    else:
        try:
            num_workers = int(args.workers)
        except ValueError:
            print(f"Error: Invalid value for --workers: '{args.workers}'. Must be an integer or 'auto'.")
            sys.exit(1)
            
    if num_workers <= 1:
        print("Running sequentially in a single process...")
        run_sequential(args)
        sys.exit(0)
        
    print(f"Running in parallel mode with {num_workers} worker processes...")
    
    # Open input video to check properties
    cap = cv2.VideoCapture(abs_input)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    _, ext = os.path.splitext(abs_input)
    
    # Setup temporary directory for chunks
    output_dir = os.path.dirname(abs_output)
    temp_dir = os.path.join(output_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Workers: {num_workers}")
    print(f"  PyTorch threads per worker: {args.threads}")
    print(f"  Frame Skip: {args.frame_skip}")
    print(f"  Max OCR Dimension: {args.max_ocr_dim}")
    print(f"  Static Frame Skip Threshold: {args.static_thresh}")
    print(f"  Page Change Detection Threshold: {args.change_thresh}")
    
    # 1. Segment video into 60-second chunks using FFmpeg copy
    print("\n[Step 1/4] Segmenting input video into ~60-second chunks...")
    ffmpeg_split_cmd = [
        'ffmpeg', '-y',
        '-i', abs_input,
        '-f', 'segment',
        '-segment_time', '60',
        '-reset_timestamps', '1',
        '-c', 'copy',
        '-map', '0',
        f"chunk_%03d{ext}"
    ]
    print(f"Running command: {' '.join(ffmpeg_split_cmd)}")
    split_res = subprocess.run(ffmpeg_split_cmd, cwd=temp_dir, capture_output=True, text=True)
    if split_res.returncode != 0:
        print("Error: Video segmentation failed!")
        print(split_res.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    # Find all chunks generated
    chunk_files = sorted(glob.glob(os.path.join(temp_dir, f"chunk_*{ext}")))
    if not chunk_files:
        print("Error: No chunks were generated!")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    print(f"Successfully segmented video into {len(chunk_files)} chunks.")
    
    # 2. Parallel processing of chunks
    print("\n[Step 2/4] Processing chunks in parallel process pool...")
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    
    # Start progress listener thread
    listener_thread = threading.Thread(target=progress_listener, args=(progress_queue, total_frames))
    listener_thread.start()
    
    # Prepare task tuples
    tasks = []
    for i, chunk_path in enumerate(chunk_files):
        out_chunk_path = os.path.join(temp_dir, f"processed_chunk_{i:03d}{ext}")
        tasks.append((i, chunk_path, out_chunk_path, args, progress_queue))
        
    # Execute workers
    results = []
    try:
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(worker_process, tasks)
    except Exception as e:
        print(f"\nError occurred during parallel pool execution: {e}")
        # Stop listener
        progress_queue.put('done')
        listener_thread.join()
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    # Stop progress listener
    progress_queue.put('done')
    listener_thread.join()
    
    # Check results
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
        
    print("\nProcessing completed for all chunks.")
    print("Aggregate Performance Stats:")
    print(f"  Total processed frames: {total_processed_frames}")
    print(f"  Total OCR inferences run: {total_ocr_runs}")
    print(f"  Total Static frame OCR skips: {total_static_skips}")
    print(f"  Total Transition-based OCR triggers: {total_transition_triggers}")
    
    # 3. Concatenate processed chunks back
    print("\n[Step 3/4] Concatenating processed chunks...")
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_path, 'w', encoding='utf-8') as f:
        for i in range(len(chunk_files)):
            # Use relative filenames in the list
            rel_name = f"processed_chunk_{i:03d}{ext}"
            f.write(f"file '{rel_name}'\n")
            
    temp_merged_video = os.path.join(temp_dir, f"merged_no_audio{ext}")
    ffmpeg_concat_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'concat_list.txt',
        '-c', 'copy',
        f"merged_no_audio{ext}"
    ]
    print(f"Running command: {' '.join(ffmpeg_concat_cmd)}")
    concat_res = subprocess.run(ffmpeg_concat_cmd, cwd=temp_dir, capture_output=True, text=True)
    if concat_res.returncode != 0:
        print("Error: FFmpeg concat failed!")
        print(concat_res.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
        
    # 4. Merge original audio and clean up
    print("\n[Step 4/4] Merging audio and finalizing output...")
    if not args.no_audio:
        merge_audio(temp_merged_video, abs_input, abs_output)
    else:
        # Move final merged chunk to output path
        if os.path.exists(abs_output):
            try:
                os.remove(abs_output)
            except Exception:
                pass
        shutil.copy2(temp_merged_video, abs_output)
        
    # Final cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\nSuccess! Final processed video written to: {abs_output}")

if __name__ == "__main__":
    main()
