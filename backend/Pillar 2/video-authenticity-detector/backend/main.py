import backend.runtime_env  # noqa: F401 — must load before torch/opencv
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))

from backend.config import STORAGE_DIR, CORS_ORIGINS
from backend.ml_device import describe_compute_device, device_summary
from backend.routes import upload, analysis, history, pillars_integrated

app = FastAPI(
    title="Video Authenticity Detector API",
    description="Multi-Pillar Deepfake & Video Forgery Detection Engine (USMFE Pillar 2)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount storage directory for static file access (video playback & suspicious frames)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# Register routers
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(history.router)
app.include_router(pillars_integrated.router)


@app.on_event("startup")
async def log_compute_backend():
    kind, name = device_summary()
    label = describe_compute_device()
    print(f"[Backend] Inference compute: {label}")
    if kind == "cpu":
        print(
            "[Backend] Tip: install PyTorch with CUDA and restart to offload ViT/consensus to GPU "
            "(pip install torch --index-url https://download.pytorch.org/whl/cu124). "
            "Use FORCE_CPU=1 only if you need to debug on CPU."
        )


@app.get("/")
async def root():
    kind, gpu_name = device_summary()
    return {
        "system": "Video Authenticity Detection System",
        "pillar": "Pillar 2 - Hybrid Visual, Biological & Temporal Forensics",
        "status": "online",
        "compute_device": kind,
        "gpu_name": gpu_name,
        "storage": "JSON File System",
        "endpoints": {
            "upload": "POST /api/upload",
            "analyze": "POST /api/analyze/{video_id}",
            "status": "GET /api/status/{video_id}",
            "result": "GET /api/result/{video_id}",
            "history": "GET /api/history",
            "storage_files": "/storage/{uploads|suspicious_frames|results}/..."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
