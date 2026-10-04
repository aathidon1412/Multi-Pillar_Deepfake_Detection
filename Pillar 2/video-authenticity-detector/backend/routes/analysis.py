import os
import subprocess
import sys
from fastapi import APIRouter, HTTPException

from backend.config import UPLOADS_DIR, BASE_DIR
from backend.services.json_storage import load_result, get_status, update_status

router = APIRouter(prefix="/api", tags=["Analysis"])


def _spawn_analysis_process(video_id: str, video_path: str, original_filename: str) -> None:
    """
    Run heavy ML pipeline in a separate Python process so a native crash cannot kill Uvicorn.
    """
    cmd = [
        sys.executable,
        "-m",
        "backend.workers.run_analysis_cli",
        "--video-id",
        video_id,
        "--video-path",
        video_path,
        "--original-filename",
        original_filename,
    ]
    env = os.environ.copy()
    app_root = str(BASE_DIR)
    repo_root = str(BASE_DIR.parent.parent)
    env["PYTHONPATH"] = os.path.pathsep.join(
        p for p in (env.get("PYTHONPATH", ""), app_root, repo_root) if p
    )
    subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


@router.post("/analyze/{video_id}")
async def start_analysis(video_id: str):
    """
    Triggers the video authenticity detection pipeline for a previously uploaded video.
    """
    video_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail=f"No uploaded video found with ID {video_id}")

    video_path = str(video_files[0])
    original_filename = video_files[0].name

    update_status(video_id, "Initializing", 5, "processing")

    try:
        _spawn_analysis_process(video_id, video_path, original_filename)
    except Exception as e:
        update_status(video_id, "Failed", 0, "failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Could not start analysis worker: {e}") from e

    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Analysis started in background worker",
    }


@router.get("/status/{video_id}")
async def check_status(video_id: str):
    """Returns current analysis stage, progress percentage, and status."""
    return get_status(video_id)


@router.get("/result/{video_id}")
async def get_result(video_id: str):
    """Fetches full JSON analysis report for a completed video."""
    result = load_result(video_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis result for video ID '{video_id}' not found or still processing.",
        )
    return result
