import os
import subprocess
import tempfile


def extract_audio_to_wav(video_path: str, sr: int = 22050) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", tmp.name], check=True)
    except Exception:
        os.unlink(tmp.name)
        raise
    return tmp.name
