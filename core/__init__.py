"""
================================================================================
Core Engine Package Initialization
Universal Synthetic Media Forensics Engine (USMFE)
================================================================================
Exposes centralized forensic engines across all 5 analytical pillars:
- Pillar 1: Vision Transformer (ViT) Spectral Artifacts
- Pillar 3: Acoustic Spectrogram Transformers & HPSS Demixing
- Pillar 4: Statistical Document, Invoice & Benford's Law OCR
- Pillar 5: Hybrid Perspective Geometry & Multi-Generator ML Ensemble
- Consensus: Domain-Aware Calibrated Decision Fusion
"""

from .pillar1_engine import load_pillar1_vit, run_pillar1_inference
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
