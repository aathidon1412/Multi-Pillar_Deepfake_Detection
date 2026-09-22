"""
================================================================================
Core Engine Package Initialization
Universal Synthetic Media Forensics Engine (USMFE)
================================================================================
Exposes centralized forensic engines across all 5 analytical pillars:
- Pillar 1: Vision Transformer (ViT) Spectral Artifacts
- Pillar 2: Video Authenticity & Deepfake Forensics
- Pillar 3: Acoustic Spectrogram Transformers & HPSS Demixing
- Pillar 4: Statistical Document, Invoice & Benford's Law OCR
- Pillar 5: Hybrid Perspective Geometry & Multi-Generator ML Ensemble
- Consensus: Domain-Aware Calibrated Decision Fusion
"""

from .pillar1_engine import load_pillar1_vit, run_pillar1_inference
from .pillar2_engine import (
    PILLAR2_AVAILABLE,
    inspect_video,
    extract_sampled_frames,
    detect_faces_in_frames,
    run_visual_analysis,
    run_temporal_analysis,
    extract_audio_track,
    run_audio_analysis,
    run_lip_sync_analysis,
    run_metadata_analysis,
    run_feature_fusion_and_classification,
    extract_and_annotate_suspicious_frames,
    cleanup_temporary_frames,
    save_result,
    load_result,
    list_results,
)
from .pillar3_engine import classify_audio
from .pillar4_engine import run_pillar4_inference, analyze_benford_law, extract_digits_from_text
from .pillar5_engine import load_pillar5_ml_bundle, load_pillar5_deep_backbone, run_pillar5_inference
from .consensus import fuse_multi_pillar_verdict
from .ui_components import (
    apply_custom_theme,
    render_header,
    render_verdict_banner,
    plot_benford_distribution
)

__all__ = [
    "load_pillar1_vit",
    "run_pillar1_inference",
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
    "classify_audio",
    "run_pillar4_inference",
    "analyze_benford_law",
    "extract_digits_from_text",
    "load_pillar5_ml_bundle",
    "load_pillar5_deep_backbone",
    "run_pillar5_inference",
    "fuse_multi_pillar_verdict",
    "apply_custom_theme",
    "render_header",
    "render_verdict_banner",
    "plot_benford_distribution"
]
