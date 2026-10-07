import os
from pathlib import Path
import imageio_ffmpeg

# Base directories
API_DIR = Path(__file__).resolve().parent
BACKEND_DIR = API_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

# Storage directories
STORAGE_DIR = BACKEND_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
FRAMES_DIR = STORAGE_DIR / "frames"
SUSPICIOUS_FRAMES_DIR = STORAGE_DIR / "suspicious_frames"
RESULTS_DIR = STORAGE_DIR / "results"
STATUS_DIR = STORAGE_DIR / "status"
XAI_DIR = STORAGE_DIR / "xai"

# Ensure all storage directories exist
for directory in [STORAGE_DIR, UPLOADS_DIR, FRAMES_DIR, SUSPICIOUS_FRAMES_DIR, RESULTS_DIR, STATUS_DIR, XAI_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Central models directory
MODELS_DIR = ROOT_DIR / "models"
CENTRAL_CASCADE = MODELS_DIR / "pillar2" / "haarcascade_frontalface_default.xml"
CASCADE_XML = CENTRAL_CASCADE if CENTRAL_CASCADE.exists() else BACKEND_DIR / "haarcascade_frontalface_default.xml"

# Upload and video limits
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500
MAX_ANALYSIS_FRAMES = 120
SAMPLE_FRAME_INTERVAL_SEC = 0.5  # Sample frame every 0.5s of video
VISUAL_INFERENCE_BATCH_SIZE = int(os.environ.get("VISUAL_INFERENCE_BATCH_SIZE", "16"))

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
