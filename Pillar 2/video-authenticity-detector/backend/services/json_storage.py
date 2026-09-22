import json
import os
import tempfile
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from backend.config import RESULTS_DIR, UPLOADS_DIR, SUSPICIOUS_FRAMES_DIR, FRAMES_DIR

# Active progress tracking state
_PROGRESS_REGISTRY: Dict[str, Dict[str, Any]] = {}

def get_result_file_path(video_id: str) -> Path:
    return RESULTS_DIR / f"{video_id}.json"

def save_result(video_id: str, data: Dict[str, Any]) -> str:
    """Safely and atomically writes JSON result to storage/results/{video_id}.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = get_result_file_path(video_id)
    
    # Write to temp file in same directory first to ensure atomic replace on Windows
    temp_fd, temp_path = tempfile.mkstemp(dir=RESULTS_DIR, prefix=f"tmp_{video_id}_", suffix=".json")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, target_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
    
    return str(target_path)

def create_result(video_id: str, data: Dict[str, Any]) -> str:
    return save_result(video_id, data)

def load_result(video_id: str) -> Optional[Dict[str, Any]]:
    """Loads a JSON result file if it exists."""
    target_path = get_result_file_path(video_id)
    if not target_path.exists():
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def list_results() -> List[Dict[str, Any]]:
    """Scans storage/results/*.json and returns summary list sorted latest first."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    
    for file_path in RESULTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            video_id = data.get("video_id", file_path.stem)
            file_info = data.get("file", {})
            classification = data.get("classification", {})
            processing = data.get("processing", {})
            
            summaries.append({
                "video_id": video_id,
                "filename": file_info.get("original_filename", "video.mp4"),
                "prediction": classification.get("prediction", "UNKNOWN"),
                "confidence": classification.get("confidence", 0.0),
                "scores": classification.get("scores", {}),
                "date": processing.get("processed_at", ""),
                "duration_seconds": data.get("video_details", {}).get("duration_seconds", 0),
                "status": processing.get("status", "completed"),
                "size_mb": file_info.get("size_mb", 0.0)
            })
        except Exception:
            continue
            
    # Sort by processed_at descending
    summaries.sort(key=lambda x: x.get("date", ""), reverse=True)
    return summaries

def delete_result(video_id: str) -> bool:
    """Deletes JSON result and associated files."""
    deleted_any = False
    result_path = get_result_file_path(video_id)
    if result_path.exists():
        result_path.unlink()
        deleted_any = True
        
    # Clean uploaded video if exists
    for upload_file in UPLOADS_DIR.glob(f"{video_id}.*"):
        if upload_file.exists():
            upload_file.unlink()
            deleted_any = True
            
    # Clean suspicious frames
    for s_frame in SUSPICIOUS_FRAMES_DIR.glob(f"{video_id}_*"):
        if s_frame.exists():
            s_frame.unlink()
            
    # Clean temporary frame folder if any
    temp_frames = FRAMES_DIR / video_id
    if temp_frames.exists() and temp_frames.is_dir():
        import shutil
        shutil.rmtree(temp_frames, ignore_errors=True)
        
    if video_id in _PROGRESS_REGISTRY:
        del _PROGRESS_REGISTRY[video_id]
        
    return deleted_any

def update_status(video_id: str, stage: str, progress: int, status: str = "processing", error: Optional[str] = None):
    """Updates real-time status of analysis."""
    _PROGRESS_REGISTRY[video_id] = {
        "video_id": video_id,
        "status": status,
        "progress": min(100, max(0, progress)),
        "current_stage": stage,
        "error": error,
        "updated_at": time.time()
    }

def get_status(video_id: str) -> Dict[str, Any]:
    """Retrieves current processing status or checks completed result."""
    if video_id in _PROGRESS_REGISTRY:
        reg = _PROGRESS_REGISTRY[video_id]
        return {
            "video_id": video_id,
            "status": reg.get("status", "processing"),
            "progress": reg.get("progress", 0),
            "current_stage": reg.get("current_stage", "Initializing"),
            "error": reg.get("error")
        }
    
    # Check if completed result file exists
    result = load_result(video_id)
    if result:
        proc = result.get("processing", {})
        return {
            "video_id": video_id,
            "status": proc.get("status", "completed"),
            "progress": 100,
            "current_stage": "Completed",
            "error": None
        }
        
    return {
        "video_id": video_id,
        "status": "not_found",
        "progress": 0,
        "current_stage": "Unknown",
        "error": "Video ID not found"
    }
