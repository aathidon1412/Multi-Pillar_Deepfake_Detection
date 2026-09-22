"""
================================================================================
Pillar 2 Core Engine: Video Authenticity & Multi-Modal Deepfake Forensics
================================================================================
Exposes video inspection, face detection, visual seams, temporal flickering,
audio acoustics, lip-sync coherence, multi-pillar fusion, and JSON storage.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2_BACKEND_DIR = os.path.join(BASE_DIR, "Pillar 2", "video-authenticity-detector")
if P2_BACKEND_DIR not in sys.path:
    sys.path.insert(0, P2_BACKEND_DIR)

try:
    from backend.services.video_processor import inspect_video  # type: ignore
    from backend.services.frame_extractor import extract_sampled_frames  # type: ignore
    from backend.services.face_detector import detect_faces_in_frames  # type: ignore
    from backend.services.visual_analyzer import run_visual_analysis  # type: ignore
    from backend.services.temporal_analyzer import run_temporal_analysis  # type: ignore
    from backend.services.audio_analyzer import extract_audio_track, run_audio_analysis  # type: ignore
    from backend.services.lip_sync_analyzer import run_lip_sync_analysis  # type: ignore
    from backend.services.metadata_analyzer import run_metadata_analysis  # type: ignore
    from backend.services.classifier import run_feature_fusion_and_classification  # type: ignore
    from backend.services.explainability import extract_and_annotate_suspicious_frames  # type: ignore
    from backend.services.cleanup import cleanup_temporary_frames  # type: ignore
    from backend.services.json_storage import save_result, load_result, list_results  # type: ignore
    PILLAR2_AVAILABLE = True
except Exception as e:
    print(f"[Pillar 2 Engine Warning]: Could not import video detector backend: {e}")
    inspect_video = None
    extract_sampled_frames = None
    detect_faces_in_frames = None
    run_visual_analysis = None
    run_temporal_analysis = None
    extract_audio_track = None
    run_audio_analysis = None
    run_lip_sync_analysis = None
    run_metadata_analysis = None
    run_feature_fusion_and_classification = None
    extract_and_annotate_suspicious_frames = None
    cleanup_temporary_frames = None
    save_result = None
    load_result = None
    list_results = None
    PILLAR2_AVAILABLE = False

__all__ = [
    "PILLAR2_AVAILABLE",
    "inspect_video",
    "extract_sampled_frames",
    "detect_faces_in_frames",
    "run_visual_analysis",
    "run_temporal_analysis",
    "extract_audio_track",
    "run_audio_analysis",
    "run_lip_sync_analysis",
    "run_metadata_analysis",
    "run_feature_fusion_and_classification",
    "extract_and_annotate_suspicious_frames",
    "cleanup_temporary_frames",
    "save_result",
    "load_result",
    "list_results",
]
