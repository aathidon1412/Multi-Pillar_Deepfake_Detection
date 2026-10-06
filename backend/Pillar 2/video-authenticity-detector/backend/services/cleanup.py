import os
import shutil
from pathlib import Path
from backend.config import FRAMES_DIR

def cleanup_temporary_frames(video_id: str):
    """
    Safely removes temporary raw frames extracted in storage/frames/{video_id}/
    while keeping the original uploaded video and saved suspicious frames intact.
    """
    target_dir = FRAMES_DIR / video_id
    if target_dir.exists() and target_dir.is_dir():
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
            print(f"[CLEANUP] Successfully removed temporary frames directory: {target_dir}")
        except Exception as e:
            print(f"[CLEANUP WARNING] Failed to remove temporary frames directory {target_dir}: {e}")
