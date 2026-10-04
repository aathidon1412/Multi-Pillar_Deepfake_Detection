import backend.runtime_env  # noqa: F401 — configure threads before ML imports

from backend.services.video_analysis_pipeline import execute_video_analysis_pipeline


def run_video_analysis_job(video_id: str, video_path: str, original_filename: str) -> None:
    execute_video_analysis_pipeline(video_id, video_path, original_filename)
