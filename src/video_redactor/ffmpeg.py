import os
import shutil
import subprocess

def get_ffmpeg_path() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_pkg_dir = os.path.join(local_appdata, r"Microsoft\WinGet\Packages")
        if os.path.exists(winget_pkg_dir):
            for folder in os.listdir(winget_pkg_dir):
                if folder.startswith("Gyan.FFmpeg"):
                    search_path = os.path.join(winget_pkg_dir, folder)
                    for root, _, files in os.walk(search_path):
                        if "ffmpeg.exe" in files:
                            return os.path.join(root, "ffmpeg.exe")
                            
    return "ffmpeg"

def merge_audio(video_no_audio: str, video_with_audio: str, final_output: str) -> None:
    try:
        ffmpeg_cmd = [
            get_ffmpeg_path(), '-y',
            '-i', video_no_audio,
            '-i', video_with_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            final_output
        ]
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode == 0:
            if os.path.exists(video_no_audio):
                try:
                    os.remove(video_no_audio)
                except Exception:
                    pass
        else:
            if os.path.exists(final_output):
                try:
                    os.remove(final_output)
                except Exception:
                    pass
            os.rename(video_no_audio, final_output)
    except Exception:
        if not os.path.exists(final_output) and os.path.exists(video_no_audio):
            try:
                os.rename(video_no_audio, final_output)
            except Exception:
                pass

def segment_video(input_path: str, output_dir: str, file_extension: str, segment_time: int = 60) -> tuple:
    ffmpeg_split_cmd = [
        get_ffmpeg_path(), '-y',
        '-i', input_path,
        '-f', 'segment',
        '-segment_time', str(segment_time),
        '-reset_timestamps', '1',
        '-c', 'copy',
        '-map', '0',
        f"chunk_%03d{file_extension}"
    ]
    res = subprocess.run(ffmpeg_split_cmd, cwd=output_dir, capture_output=True, text=True)
    return res.returncode == 0, res.stderr

def concat_videos(concat_list_filename: str, output_dir: str, output_filename: str) -> tuple:
    ffmpeg_concat_cmd = [
        get_ffmpeg_path(), '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_filename,
        '-c', 'copy',
        output_filename
    ]
    res = subprocess.run(ffmpeg_concat_cmd, cwd=output_dir, capture_output=True, text=True)
    return res.returncode == 0, res.stderr
