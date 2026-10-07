import backend.api.runtime_env  # noqa: F401 — must load before torch/opencv
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
api_dir = Path(__file__).resolve().parent
backend_dir = api_dir.parent
project_root = backend_dir.parent
for p in [str(project_root), str(backend_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.api.config import STORAGE_DIR, CORS_ORIGINS
from backend.api.ml_device import describe_compute_device, device_summary
from backend.api.routes import upload, analysis, history, pillars

app = FastAPI(
    title="Universal Synthetic Media Forensics Engine API",
    description="Multi-Pillar Deepfake & Synthetic Media Detection Engine (USMFE)",
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

# Mount storage directory for static file access (video playback, audio, suspicious frames)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# Register routers
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(history.router)
app.include_router(pillars.router)


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
        "system": "Universal Synthetic Media Forensics Engine (USMFE)",
        "architecture": "Multi-Pillar Forensic Engine (Visual, Video, Audio, Document, Physics)",
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
            "universal": "POST /api/pillars/universal",
            "image": "POST /api/pillars/image",
            "audio": "POST /api/pillars/audio",
            "document": "POST /api/pillars/document",
            "storage_files": "/storage/{uploads|suspicious_frames|results}/..."
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
