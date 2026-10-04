import gc
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

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
from backend.services.json_storage import save_result, update_status


def _release_extracted_frame_memory(extracted_frames: Optional[List[Dict[str, Any]]]) -> None:
    """Drop large in-memory pixel buffers after disk artifacts are written."""
    if not extracted_frames:
        return
    for frame in extracted_frames:
        for key in ("image_bgr", "image_rgb"):
            frame.pop(key, None)
        faces = frame.get("faces")
        if faces:
            for face in faces:
                face.pop("face_crop", None)
                face.pop("mouth_crop", None)


def _release_ml_runtime_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def execute_video_analysis_pipeline(video_id: str, video_path: str, original_filename: str):
    """
    Executes the comprehensive multi-pillar video authenticity detection pipeline.
    Intended to run in a worker subprocess so native ML crashes cannot take down Uvicorn.
    """
    extracted_frames = None
    start_time = time.time()
    try:
        update_status(video_id, "Preprocessing", 10, "processing")
        video_details = inspect_video(video_path)
        file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)
        stored_filename = os.path.basename(video_path)
        format_ext = os.path.splitext(stored_filename)[1].lstrip(".").lower()

        update_status(video_id, "Metadata Analysis", 20, "processing")
        metadata_analysis = run_metadata_analysis(video_path, video_details)

        update_status(video_id, "Extracting Frames", 30, "processing")
        extracted_frames = extract_sampled_frames(video_path, video_id)

        update_status(video_id, "Detecting Faces", 45, "processing")
        face_summary = detect_faces_in_frames(extracted_frames)

        update_status(video_id, "Visual Analysis", 60, "processing")
        visual_analysis = run_visual_analysis(extracted_frames)

        update_status(video_id, "Temporal Analysis", 72, "processing")
        temporal_analysis = run_temporal_analysis(extracted_frames)

        update_status(video_id, "Audio Analysis", 80, "processing")
        audio_wav_path = extract_audio_track(video_path, video_id)
        audio_analysis = run_audio_analysis(audio_wav_path)

        update_status(video_id, "Lip-Sync Analysis", 86, "processing")
        lip_sync_analysis = run_lip_sync_analysis(extracted_frames, audio_wav_path)

        update_status(video_id, "Classification", 92, "processing")
        classification = run_feature_fusion_and_classification(
            visual_analysis,
            temporal_analysis,
            audio_analysis,
            lip_sync_analysis,
            metadata_analysis,
        )

        update_status(video_id, "Generating Report", 96, "processing")
        suspicious_frames = extract_and_annotate_suspicious_frames(extracted_frames, video_id)
        _release_extracted_frame_memory(extracted_frames)

        processing_time = round(time.time() - start_time, 2)
        now_iso = datetime.now().isoformat()

        final_report = {
            "video_id": video_id,
            "file": {
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "format": format_ext,
                "size_mb": file_size_mb,
            },
            "video_details": {
                "duration_seconds": video_details["duration_seconds"],
                "resolution": video_details["resolution"],
                "fps": video_details["fps"],
                "frame_count": video_details["frame_count"],
            },
            "metadata": {
                "codec": metadata_analysis["codec"],
                "encoder": metadata_analysis["encoder"],
                "metadata_status": metadata_analysis["metadata_status"],
            },
            "analysis": {
                "face_detection": {
                    "faces_detected": face_summary["faces_detected"],
                    "face_count": face_summary["face_count"],
                },
                "visual": {
                    "score": visual_analysis["score"],
                    "status": visual_analysis["status"],
                    "anomalies": visual_analysis["anomalies"],
                },
                "temporal": {
                    "score": temporal_analysis["score"],
                    "status": temporal_analysis["status"],
                    "flickering_detected": temporal_analysis["flickering_detected"],
                    "motion_inconsistency": temporal_analysis["motion_inconsistency"],
                },
                "audio": {
                    "available": audio_analysis["available"],
                    "score": audio_analysis["score"],
                    "status": audio_analysis["status"],
                },
                "lip_sync": {
                    "available": lip_sync_analysis["available"],
                    "score": lip_sync_analysis["score"],
                    "status": lip_sync_analysis["status"],
                },
            },
            "classification": {
                "prediction": classification["prediction"],
                "scores": classification["scores"],
                "confidence": classification["confidence"],
            },
            "suspicious_frames": suspicious_frames,
            "processing": {
                "status": "completed",
                "processed_at": now_iso,
                "processing_time_seconds": processing_time,
            },
        }

        try:
            from core.xai import generate_pillar2_xai, synthesize_multi_pillar_xai

            p2_xai = generate_pillar2_xai(
                final_report["analysis"],
                final_report["classification"],
                suspicious_frames,
            )
            xai_bundle = synthesize_multi_pillar_xai(pillar2=p2_xai)
            final_report["xai"] = xai_bundle
            final_report["pillar2"] = {
                "verdict": classification["prediction"],
                "confidence": round(
                    float(
                        classification["confidence"] * 100.0
                        if classification["confidence"] <= 1.0
                        else classification["confidence"]
                    ),
                    2,
                ),
                "xai": p2_xai,
            }
        except Exception as xe:
            print(f"[Pillar 2 XAI Synthesis Warning]: {xe}")
            try:
                from core.xai import create_fallback_xai_response, synthesize_multi_pillar_xai

                fallback_xai = create_fallback_xai_response(
                    "Pillar 2: Video & Biological Forensics",
                    classification["prediction"],
                    classification["confidence"] * 100.0,
                )
                final_report["xai"] = synthesize_multi_pillar_xai(pillar2=fallback_xai)
                final_report["pillar2"] = {"xai": fallback_xai}
            except Exception:
                pass

        save_result(video_id, final_report)
        update_status(video_id, "Completed", 100, "completed")
        print(
            f"[ANALYSIS SUCCESS] Video {video_id} analyzed in {processing_time}s -> {classification['prediction']}"
        )

        cleanup_temporary_frames(video_id)
        extracted_frames = None
        _release_ml_runtime_memory()

    except Exception as e:
        import traceback

        traceback.print_exc()
        cleanup_temporary_frames(video_id)
        _release_extracted_frame_memory(extracted_frames)
        extracted_frames = None
        _release_ml_runtime_memory()
        update_status(video_id, "Failed", 0, "failed", error=str(e))
        print(f"[ANALYSIS ERROR] Pipeline failed for {video_id}: {e}")
