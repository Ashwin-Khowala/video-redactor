import cv2
import torch
from .config import ProcessingConfig
from .ocr import OCRProcessor
from .patterns import is_sensitive
from .blur import apply_blur

def worker_process(
    chunk_idx: int,
    input_path: str,
    output_path: str,
    config: ProcessingConfig,
    progress_queue,
    use_gpu: bool
) -> dict:
    torch.set_num_threads(config.threads)
    
    ocr = OCRProcessor(use_gpu=use_gpu)
    
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    active_blur_boxes: list = []
    last_ocr_blur_boxes: list = []
    prev_frame_gray = None
    last_ocr_frame_gray = None
    last_ocr_frame_idx = 0
    transition_active = False
    
    diff_scale = 540 / max(width, height)
    diff_width = int(width * diff_scale)
    diff_height = int(height * diff_scale)
    
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
            results = ocr.detect_text(
                ocr_frame_rgb,
                max_ocr_dim=config.max_ocr_dim,
                mag_ratio=config.mag_ratio,
                adjust_contrast=not config.no_contrast,
                min_size=config.min_size
            )
            ocr_runs += 1
            
            for bbox, text, confidence in results:
                if is_sensitive(text, config.match_config):
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
