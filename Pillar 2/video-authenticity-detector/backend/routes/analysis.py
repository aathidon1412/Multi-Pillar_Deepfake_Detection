import os
import time
import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.config import UPLOADS_DIR
from backend.services.video_processor import inspect_video
from backend.services.frame_extractor import extract_sampled_frames
from backend.services.face_detector import detect_faces_in_frames
from backend.services.visual_analyzer import run_visual_analysis
from backend.services.temporal_analyzer import run_temporal_analysis
from backend.services.audio_analyzer import extract_audio_track, run_audio_analysis
from backend.services.lip_sync_analyzer import run_lip_sync_analysis
from backend.services.metadata_analyzer import run_metadata_analysis
from backend.services.classifier import run_feature_fusion_and_classification
from backend.services.explainability import extract_and_annotate_suspicious_frames
from backend.services.cleanup import cleanup_temporary_frames
from backend.services.json_storage import (
    save_result,
    load_result,
    update_status,
    get_status
)

router = APIRouter(prefix="/api", tags=["Analysis"])

def execute_video_analysis_pipeline(video_id: str, video_path: str, original_filename: str):
    """
    Executes the comprehensive multi-pillar video authenticity detection pipeline synchronously
    inside a FastAPI background task thread.
    """
    start_time = time.time()
    try:
        # Step 1: Preprocessing & Metadata
        update_status(video_id, "Preprocessing", 10, "processing")
        video_details = inspect_video(video_path)
        file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)
        stored_filename = os.path.basename(video_path)
        format_ext = os.path.splitext(stored_filename)[1].lstrip(".").lower()

        # Step 2: Metadata Analysis
        update_status(video_id, "Metadata Analysis", 20, "processing")
        metadata_analysis = run_metadata_analysis(video_path, video_details)

        # Step 3: Frame Extraction
        update_status(video_id, "Extracting Frames", 30, "processing")
        extracted_frames = extract_sampled_frames(video_path, video_id)

        # Step 4: Face Detection
        update_status(video_id, "Detecting Faces", 45, "processing")
        face_summary = detect_faces_in_frames(extracted_frames)

        # Step 5: Visual Analysis
        update_status(video_id, "Visual Analysis", 60, "processing")
        visual_analysis = run_visual_analysis(extracted_frames)

        # Step 6: Temporal Analysis
        update_status(video_id, "Temporal Analysis", 72, "processing")
        temporal_analysis = run_temporal_analysis(extracted_frames)

        # Step 7: Audio Extraction & Analysis
        update_status(video_id, "Audio Analysis", 80, "processing")
        audio_wav_path = extract_audio_track(video_path, video_id)
        audio_analysis = run_audio_analysis(audio_wav_path)

        # Step 8: Lip-Sync Analysis
        update_status(video_id, "Lip-Sync Analysis", 86, "processing")
        lip_sync_analysis = run_lip_sync_analysis(extracted_frames, audio_wav_path)

        # Step 9: Classification / Feature Fusion
        update_status(video_id, "Classification", 92, "processing")
        classification = run_feature_fusion_and_classification(
            visual_analysis,
            temporal_analysis,
            audio_analysis,
            lip_sync_analysis,
            metadata_analysis
        )

        # Step 10: Suspicious Frame Extraction
        update_status(video_id, "Generating Report", 96, "processing")
        suspicious_frames = extract_and_annotate_suspicious_frames(extracted_frames, video_id)

        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)
        now_iso = datetime.now().isoformat()

        # Assemble Final JSON Report strictly matching the required schema
        final_report = {
            "video_id": video_id,
            "file": {
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "format": format_ext,
                "size_mb": file_size_mb
            },
            "video_details": {
                "duration_seconds": video_details["duration_seconds"],
                "resolution": video_details["resolution"],
                "fps": video_details["fps"],
                "frame_count": video_details["frame_count"]
            },
            "metadata": {
                "codec": metadata_analysis["codec"],
                "encoder": metadata_analysis["encoder"],
                "metadata_status": metadata_analysis["metadata_status"]
            },
            "analysis": {
                "face_detection": {
                    "faces_detected": face_summary["faces_detected"],
                    "face_count": face_summary["face_count"]
                },
                "visual": {
                    "score": visual_analysis["score"],
                    "status": visual_analysis["status"],
                    "anomalies": visual_analysis["anomalies"]
                },
                "temporal": {
                    "score": temporal_analysis["score"],
                    "status": temporal_analysis["status"],
                    "flickering_detected": temporal_analysis["flickering_detected"],
                    "motion_inconsistency": temporal_analysis["motion_inconsistency"]
                },
                "audio": {
                    "available": audio_analysis["available"],
                    "score": audio_analysis["score"],
                    "status": audio_analysis["status"]
                },
                "lip_sync": {
                    "available": lip_sync_analysis["available"],
                    "score": lip_sync_analysis["score"],
                    "status": lip_sync_analysis["status"]
                }
            },
            "classification": {
                "prediction": classification["prediction"],
                "scores": classification["scores"],
                "confidence": classification["confidence"]
            },
            "suspicious_frames": suspicious_frames,
            "processing": {
                "status": "completed",
                "processed_at": now_iso,
                "processing_time_seconds": processing_time
            }
        }

        # Save JSON result permanently
        save_result(video_id, final_report)

        # Cleanup temporary frames
        cleanup_temporary_frames(video_id)

        # Mark completed
        update_status(video_id, "Completed", 100, "completed")
        print(f"[ANALYSIS SUCCESS] Video {video_id} analyzed in {processing_time}s -> {classification['prediction']}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        cleanup_temporary_frames(video_id)
        update_status(video_id, "Failed", 0, "failed", error=str(e))
        print(f"[ANALYSIS ERROR] Pipeline failed for {video_id}: {e}")

@router.post("/analyze/{video_id}")
async def start_analysis(video_id: str, background_tasks: BackgroundTasks):
    """
    Triggers the video authenticity detection pipeline for a previously uploaded video.
    """
    # Find uploaded video file
    video_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail=f"No uploaded video found with ID {video_id}")

    video_path = str(video_files[0])
    original_filename = video_files[0].name

    # Set initial processing status
    update_status(video_id, "Initializing", 5, "processing")

    # Dispatch to background task
    background_tasks.add_task(
        execute_video_analysis_pipeline,
        video_id=video_id,
        video_path=video_path,
        original_filename=original_filename
    )

    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Analysis started in background"
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
            detail=f"Analysis result for video ID '{video_id}' not found or still processing."
        )
    return result
