import os
from pathlib import Path
import imageio_ffmpeg

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
FRAMES_DIR = STORAGE_DIR / "frames"
SUSPICIOUS_FRAMES_DIR = STORAGE_DIR / "suspicious_frames"
RESULTS_DIR = STORAGE_DIR / "results"

# Ensure storage directories exist
for directory in [STORAGE_DIR, UPLOADS_DIR, FRAMES_DIR, SUSPICIOUS_FRAMES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Pillar 2 resource references
PILLAR2_DIR = BASE_DIR.parent
CASCADE_XML = PILLAR2_DIR / "haarcascade_frontalface_default.xml"

# Upload and video limits
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500
MAX_ANALYSIS_FRAMES = 120
SAMPLE_FRAME_INTERVAL_SEC = 0.5  # Sample frame every 0.5s of video

# FFmpeg binary
try:
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = "ffmpeg"

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "*"
]
