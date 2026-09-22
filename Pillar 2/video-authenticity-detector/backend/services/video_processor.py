import os
import uuid
import time
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple
from fastapi import UploadFile, HTTPException

from backend.config import UPLOADS_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

def generate_video_id() -> str:
    """Generates a clean, unique ID in the format VID_XXXXXX."""
    random_suffix = uuid.uuid4().hex[:6].upper()
    return f"VID_{random_suffix}"

def validate_video_file(file: UploadFile) -> str:
    """
    Validates file extension and basic properties.
    Returns the lowercase file extension.
    """
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext

async def save_uploaded_video(file: UploadFile) -> Tuple[str, str, float]:
    """
    Saves an uploaded file to storage/uploads/ with a unique video_id.
    Returns (video_id, saved_filepath, file_size_mb).
    """
    ext = validate_video_file(file)
    video_id = generate_video_id()
    saved_filename = f"{video_id}{ext}"
    saved_path = UPLOADS_DIR / saved_filename

    total_bytes = 0
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    try:
        with open(saved_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB."
                    )
                f.write(chunk)
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        raise e

    size_mb = round(total_bytes / (1024 * 1024), 2)
    return video_id, str(saved_path), size_mb

def inspect_video(video_path: str) -> Dict[str, Any]:
    """
    Opens video with OpenCV and extracts fundamental video parameters:
    fps, frame count, resolution, and duration.
    """
    if not os.path.isfile(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open or decode video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration safely
        duration_seconds = round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0.0

        if width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError("Video file contains zero frames or invalid resolution dimensions.")

        return {
            "duration_seconds": duration_seconds,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count
        }
    finally:
        cap.release()
