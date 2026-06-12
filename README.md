# Video Redactor

Automatically detect and blur sensitive information (emails, phone numbers, and custom keywords) in videos using OCR.

## Features

- **Sensitive Data Detection**: Blurs emails, phone numbers, and custom keywords/names.
- **Selective Redaction**: Target specific types of data (e.g., only blur emails, only blur phone numbers) using the `--redact-types` option.
- **Dynamic Optimization**:
  - **Frame Skipping**: Reuses detections on consecutive frames.
  - **Static Skip**: Skips OCR processing on static frames by checking pixel difference.
  - **Transition-based OCR**: Automatically triggers OCR when a scene or page transition is detected.
- **Multi-Process Parallelism**: Segments videos into chunks and processes them in parallel across multiple CPU/GPU workers using FFmpeg.
- **Audio Preservation**: Merges the original audio track back to the redacted video at the end of the run.
- **Docker Ready**: Run the application instantly inside a pre-configured CPU or GPU Docker container without installing heavy dependencies locally.

## Installation

### Prerequisites

1. **Python**: Python 3.9 or higher.
2. **FFmpeg**: Must be installed and available on your system path (on Windows, standard Gyan FFmpeg builds from winget are supported).

### Install Package (Local Setup)

Clone the repository and install it locally:

```bash
pip install -e .
```

To install development dependencies (for testing and linting):

```bash
pip install -e .[dev]
```

---

## Docker Setup (Instant Run)

Docker eliminates the need to manually configure heavy dependencies (like PyTorch, OpenCV, and CUDA).

### 1. Build Docker Images

**CPU Version (Universal):**
```bash
docker build -t vidredact:cpu -f Dockerfile .
```

**GPU Version (CUDA Accelerated):**
```bash
docker build -t vidredact:gpu -f Dockerfile.gpu .
```

### 2. Run Container

Mount your local directory containing the videos to `/data` in the container.

**Using CPU:**
```bash
docker run -v "$(pwd):/data" vidredact:cpu -i /data/input.mp4 -o /data/output.mp4
```

**Using GPU (requires NVIDIA Container Toolkit installed):**
```bash
docker run --gpus all -v "$(pwd):/data" vidredact:gpu -i /data/input.mp4 -o /data/output.mp4
```

### 3. Using Docker Compose

Alternatively, use `docker-compose.yml` to run containerized workflows:

**Run with CPU:**
```bash
docker compose run vidredact -i /data/input.mp4 -o /data/output.mp4
```

**Run with GPU:**
```bash
docker compose run vidredact-gpu -i /data/input.mp4 -o /data/output.mp4
```

---

## Quick Start

Run the CLI command with an input video file path. The output file path is optional and will default to `[input_filename]_redacted[extension]` if omitted:

```bash
video-redactor -i input_video.mp4
```

### Advanced Usage

Process a video, blurring **only** phone numbers, using 2 parallel workers and debug logging:

```bash
video-redactor -i input_video.mp4 --redact-types phone --workers 2 --debug
```

Redact only emails and custom names:

```bash
video-redactor -i input_video.mp4 --redact-types email,keywords --keywords "John Doe, secret_info"
```

## Command Line Options

| Argument | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *Required* | Path to the input video file. |
| `--output` | `-o` | `""` | Path to the output video file (defaults to `[input]_redacted.[ext]` or folder-relative equivalent). |
| `--redact-types` | | `email,phone,keywords` | Comma-separated target types to redact: `email`, `phone`, `keywords`, `all`. |
| `--mode` | `-m` | `patterns` | Redaction mode: `patterns` (guided by targets) or `all` (blur all text). |
| `--frame-skip` | `-f` | `1` | Perform OCR every N frames (skipped frames reuse prior detections). |
| `--padding` | `-p` | `25` | Pixels of padding to expand the blur box around detected text. |
| `--blur-strength`| `-s` | `35` | Gaussian blur kernel size (must be odd). |
| `--keywords` | `-k` | `None` | Comma-separated custom words or names to blur. |
| `--max-ocr-dim` | `-d` | `1080` | Maximum width or height of the frame resized for OCR. |
| `--static-thresh`| `-t` | `0.8` | Difference threshold below which a frame is skipped as static. |
| `--change-thresh`| `-c` | `1.0` | Difference threshold above which page/scene change triggers OCR. |
| `--no-contrast` | | `False` | Disable EasyOCR contrast adjustment for extra speed. |
| `--mag-ratio` | | `1.0` | EasyOCR magnification ratio. |
| `--min-size` | | `10` | Minimum size of text in pixels to detect. |
| `--threads` | `-th`| `4` | PyTorch CPU execution threads per worker. |
| `--no-audio` | | `False` | Skip merging the original audio track at the end. |
| `--workers` | | `1` | Parallel worker processes (`1`, integer, or `auto`). |
| `--cpu` | | `False` | Force CPU execution (disables GPU/CUDA). |
| `--debug` | | `False` | Enable verbose debug logs and OCR output. |

## Running Tests

Verify your installation by running the test suite:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
