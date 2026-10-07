"""
================================================================================
Pillar 2 Core Engine: Video Authenticity & Multi-Modal Deepfake Forensics
================================================================================
Exposes video inspection, face detection, visual seams, temporal flickering,
audio acoustics, lip-sync coherence, multi-pillar fusion, and storage.
"""

import os
import sys

try:
    from backend.services.video_processor import inspect_video
    from backend.services.frame_extractor import extract_sampled_frames
    from backend.services.face_detector import detect_faces_in_frames
    from backend.services.visual_analyzer import run_visual_analysis
    from backend.services.temporal_analyzer import run_temporal_analysis
    from backend.services.audio_analyzer import extract_audio_track, run_audio_analysis
    from backend.services.lip_sync_analyzer import run_lip_sync_analysis
    from backend.services.metadata_analyzer import run_metadata_analysis
    from backend.services.classifier import run_feature_fusion_and_classification
    from backend.services.rppg_analyzer import run_rppg_analysis
    from backend.services.explainability import extract_and_annotate_suspicious_frames
    from backend.services.cleanup import cleanup_temporary_frames
    from backend.services.storage import save_result, load_result, list_results
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
    run_rppg_analysis = None
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
    "run_rppg_analysis",
    "run_feature_fusion_and_classification",
    "extract_and_annotate_suspicious_frames",
    "cleanup_temporary_frames",
    "save_result",
    "load_result",
    "list_results",
]
