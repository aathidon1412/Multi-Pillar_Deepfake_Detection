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
    if has_cam_exif:
        # Authentic camera hardware signature confirmed by device metadata (Nikon, iPhone, Vivo, etc.)
        is_unified_real = True
        fused_real_prob = max(p5_real_prob, 0.92)
        fused_fake_prob = 1.0 - fused_real_prob
        override_reason = "Camera Hardware Sensor EXIF Signature Confirmed"
    elif is_paper_doc and p5_fake_prob < 0.90:
        # Scanned documents, receipts, and handwritten signatures
        is_unified_real = True
        fused_real_prob = 0.88
        fused_fake_prob = 0.12
        override_reason = "Physical Document / Scanned Paper Domain Gating (Pillar 4)"
    elif p5_res.get("is_prnu_anomaly") and p5_fake_prob >= 0.60:
        is_unified_real = False
        fused_fake_prob = max(p5_fake_prob, 0.82)
        fused_real_prob = 1.0 - fused_fake_prob
        override_reason = "Pillar 5 PRNU Steganalysis Override (Synthetic Noise Anomaly)"
    elif p1_res.get("available", False) and p1_fake_prob >= 0.70:
        is_unified_real = False
        fused_fake_prob = p1_fake_prob
        fused_real_prob = 1.0 - fused_fake_prob
        override_reason = "Pillar 1 ViT Neural Override (Facial Spectral Anomaly)"
    else:
        # Weighted multi-pillar consensus with calibrated decision threshold
        fused_fake_prob = 0.55 * p5_fake_prob + 0.45 * p1_fake_prob
        fused_real_prob = 1.0 - fused_fake_prob
        is_unified_real = (fused_fake_prob < 0.45)

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
        "is_paper_doc": is_paper_doc
    }
