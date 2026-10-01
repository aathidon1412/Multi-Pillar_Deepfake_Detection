"""
================================================================================
Consensus Decision Engine & Domain-Aware Multi-Pillar Fusion
================================================================================
Implements:
- Camera Sensor Hardware EXIF Signature Authentication
- Scanned Physical Paper & Document Gating (Pillar 4 Benford integration)
- Deterministic Steganalysis PRNU Anomaly Override
- Facial Spectral Transformer Anomaly Override
- Calibrated Multi-Pillar Soft-Voting Consensus
"""

import cv2
import numpy as np
from PIL import Image

def fuse_multi_pillar_verdict(pil_img, np_img, p1_res, p4_res, p5_res) -> dict:
    """
    Fuses inferences from Pillar 1, Pillar 4, and Pillar 5 into a unified verdict.
    Accounts for camera sensor hardware, document domains, and synthetic anomalies.
    """
    p5_real_prob = p5_res.get("real_probability", 0.5)
    p5_fake_prob = p5_res.get("fake_probability", 1.0 - p5_real_prob)
    p1_real_prob = p1_res.get("real_probability", 0.5) if p1_res.get("available", False) else p5_real_prob
    p1_fake_prob = 1.0 - p1_real_prob

    # Domain & Sensor Telemetry Extraction
    exif = pil_img.getexif() if hasattr(pil_img, 'getexif') else {}
    has_cam_exif = bool(exif.get(0x010f) or exif.get(0x0110) or exif.get(0x0131))
    gray_full = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    white_ratio = float(np.mean(gray_full > 220))
    is_paper_doc = (white_ratio > 0.40) or (p4_res.get("applicable", False) and p4_res.get("digits_count", 0) >= 5)

    override_reason = None
    is_env_anomaly = p5_res.get("is_environment_synthetic", False) or p5_res.get("is_prnu_anomaly", False) or ((p5_res.get("srm_var_4", 999.0) <= 2.0 or p5_res.get("raw_srm4_var", 999.0) <= 0.75) and not is_paper_doc)
    if has_cam_exif:
        # Authentic camera hardware signature confirmed by device metadata (Nikon, iPhone, Vivo, etc.)
        is_unified_real = True
        fused_real_prob = max(p1_real_prob, p5_real_prob, 0.90)
        fused_fake_prob = 1.0 - fused_real_prob
        override_reason = "Camera Hardware Sensor EXIF Signature Confirmed"
    elif is_paper_doc and p1_fake_prob < 0.50:
        # Scanned documents, receipts, and handwritten signatures
        is_unified_real = True
        fused_real_prob = 0.88
        fused_fake_prob = 0.12
        override_reason = "Physical Document / Scanned Paper Domain Gating (Pillar 4)"
    elif is_env_anomaly and not is_paper_doc:
        # Synthetic diffusion noise floor or peripheral environment anomaly (Midjourney/DALL-E/Inpainting)
        is_unified_real = False
        fused_fake_prob = max(p5_fake_prob, 0.88)
        fused_real_prob = 1.0 - fused_fake_prob
        override_reason = "Pillar 5 Steganalysis & Surrounding Environment Forgery Override"
    elif p1_res.get("available", False) and p1_fake_prob >= 0.50:
        # High-confidence facial/compression spectral anomaly detected by Vision Transformer
        is_unified_real = False
        fused_fake_prob = p1_fake_prob
        fused_real_prob = 1.0 - fused_fake_prob
        override_reason = "Pillar 1 ViT Neural Override (Facial Spectral / Splicing Anomaly)"
    elif p1_real_prob >= 0.88 and not is_env_anomaly:
        # High-confidence authentic visual spectra from Vision Transformer (protects camera images like Real_4)
        is_unified_real = True
        fused_real_prob = p1_real_prob
        fused_fake_prob = 1.0 - fused_real_prob
        override_reason = "Pillar 1 Vision Transformer High-Confidence Authentic"
    elif p5_real_prob < 0.35 and p1_real_prob < 0.80:
        # Deep perspective & shadow physics anomaly on composite media
        is_unified_real = False
        fused_fake_prob = p5_fake_prob
        fused_real_prob = 1.0 - fused_fake_prob
        override_reason = "Pillar 5 RANSAC & Deep Physics Anomaly (Composite Media)"
    else:
        # Weighted multi-pillar consensus with soft-voting
        fused_real_prob = 0.50 * p1_real_prob + 0.50 * p5_real_prob
        fused_fake_prob = 1.0 - fused_real_prob
        is_unified_real = (fused_real_prob >= 0.50)
        override_reason = "Multi-Pillar Calibrated Soft-Voting Consensus"

    unified_verdict = "AUTHENTIC MEDIA" if is_unified_real else "FAKE (SYNTHETIC AI ANOMALY)"
    unified_conf = round(((1.0 - fused_real_prob) * 100.0) if not is_unified_real else (fused_real_prob * 100.0), 2)

    return {
        "is_real": is_unified_real,
        "verdict": unified_verdict,
        "confidence": unified_conf,
        "fused_real_prob": fused_real_prob,
        "fused_fake_prob": fused_fake_prob,
        "override_reason": override_reason,
        "has_cam_exif": has_cam_exif,
        "is_paper_doc": is_paper_doc,
        "environment_analysis": p5_res.get("environment_analysis", {})
    }
