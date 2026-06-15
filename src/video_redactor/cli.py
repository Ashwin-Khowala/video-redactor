import os
import sys
import argparse
import torch
import multiprocessing
from .config import MatchConfig, ProcessingConfig
from .processor import run_sequential, run_parallel

def main() -> None:
    multiprocessing.freeze_support()
    
    if sys.stdout is not None:
        try:
            reconfigure = getattr(sys.stdout, 'reconfigure', None)
            if reconfigure is not None:
                reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Blur sensitive info (emails, mobile numbers) from videos using EasyOCR.")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default="", help="Path to output video file (defaults to input_redacted.mp4)")
    parser.add_argument("--redact-types", type=str, default="email,phone,keywords",
                        help="Comma-separated target types to redact: email, phone, keywords, all (default: email,phone,keywords)")
    parser.add_argument("--mode", "-m", choices=['patterns', 'all'], default='patterns', 
                        help="all: blur all text; patterns: blur emails and phone numbers (default)")
    parser.add_argument("--frame-skip", "-f", type=int, default=1, 
                        help="Run OCR every N frames. Skipped frames reuse detections to speed up processing (default: 1)")
    parser.add_argument("--padding", "-p", type=int, default=25, 
                        help="Pixels of padding to expand the blur box around detected text (default: 25)")
    parser.add_argument("--blur-strength", "-s", type=int, default=35, 
                        help="Gaussian blur kernel size (must be odd, default: 35)")
    parser.add_argument("--keywords", "-k", type=str, 
                        help="Comma-separated custom words/names to blur")
    parser.add_argument("--max-ocr-dim", "-d", type=int, default=1920,
                        help="Maximum width or height of the frame when passed to EasyOCR. (default: 1920)")
    parser.add_argument("--static-thresh", "-t", type=float, default=0.1,
                        help="Grayscale mean pixel difference threshold to classify a frame as static and skip OCR (default: 0.1)")
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
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution for EasyOCR (disables GPU)")
    parser.add_argument("--debug", action="store_true",
                        help="Print OCR detections to console for debugging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
        
    abs_input = os.path.abspath(args.input)
    abs_output = os.path.abspath(args.output) if args.output else ""
    
    custom_keywords = [k.strip() for k in args.keywords.split(',')] if args.keywords else []
    redact_types = [t.strip().lower() for t in args.redact_types.split(',')] if args.redact_types else ["email", "phone", "keywords"]
    match_config = MatchConfig(
        mode=args.mode,
        redact_types=redact_types,
        custom_keywords=custom_keywords
    )
    
    config = ProcessingConfig(
        input_path=abs_input,
        output_path=abs_output,
        match_config=match_config,
        frame_skip=args.frame_skip,
        padding=args.padding,
        blur_strength=args.blur_strength,
        max_ocr_dim=args.max_ocr_dim,
        static_thresh=args.static_thresh,
        change_thresh=args.change_thresh,
        mag_ratio=args.mag_ratio,
        min_size=args.min_size,
        threads=args.threads,
        no_contrast=args.no_contrast,
        no_audio=args.no_audio,
        workers=args.workers,
        cpu=args.cpu,
        debug=args.debug
    )
    
    gpu_available = torch.cuda.is_available()
    use_gpu = gpu_available and not args.cpu
    
    cpu_count = os.cpu_count() or 1
    if args.workers.lower() == 'auto':
        if use_gpu:
            num_workers = max(1, min(3, cpu_count - 1))
        else:
            num_workers = max(1, min(4, cpu_count - 1))
    else:
        try:
            num_workers = int(args.workers)
        except ValueError:
            print(f"Error: Invalid value for --workers: '{args.workers}'. Must be an integer or 'auto'.")
            sys.exit(1)
            
    if num_workers <= 1:
        if args.debug:
            print(f"Running sequentially in a single process (GPU={use_gpu})...")
        run_sequential(config, use_gpu=use_gpu)
    else:
        if args.debug:
            print(f"Running in parallel mode with {num_workers} worker processes (GPU={use_gpu})...")
        run_parallel(config, use_gpu=use_gpu, num_workers=num_workers)

if __name__ == "__main__":
    main()
