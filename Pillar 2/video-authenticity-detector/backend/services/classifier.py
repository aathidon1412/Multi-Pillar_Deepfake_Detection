import numpy as np
from typing import Dict, Any

def run_feature_fusion_and_classification(
    visual_analysis: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    audio_analysis: Dict[str, Any],
    lip_sync_analysis: Dict[str, Any],
    metadata_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fuses multi-pillar forensic signals and computes probabilistic 3-way classification:
    - REAL: genuine authentic video with natural textures and physics
    - AI_GENERATED: generative AI, deepfake, face swap, or synthetic voice artifacts
    - FORGED: non-generative digital editing, splicing, frame cutting, or motion disruption
    """
    visual_score = visual_analysis.get("score", 0.10)
    temporal_score = temporal_analysis.get("score", 0.10)
    
    audio_available = audio_analysis.get("available", False)
    audio_score = audio_analysis.get("score", 0.10) if audio_available else 0.0

    lip_sync_available = lip_sync_analysis.get("available", False)
    lip_sync_score = lip_sync_analysis.get("score", 0.10) if lip_sync_available else 0.0

    meta_status = metadata_analysis.get("metadata_status", "inconclusive")
    meta_score = 0.60 if meta_status == "suspicious" else (0.15 if meta_status == "normal" else 0.30)

    # Dynamic Pillar Weighting
    # Default: Visual 35%, Temporal 25%, Audio 15%, LipSync 15%, Metadata 10%
    weights = {"visual": 0.35, "temporal": 0.25, "audio": 0.15, "lip_sync": 0.15, "metadata": 0.10}

    # Redistribute weights if modalities are missing
    if not audio_available:
        weights["visual"] += 0.08
        weights["temporal"] += 0.07
        weights["audio"] = 0.0

    if not lip_sync_available:
        weights["visual"] += 0.08
        weights["temporal"] += 0.07
        weights["lip_sync"] = 0.0

    # Normalize weights
    total_w = sum(weights.values())
    for k in weights:
        weights[k] /= total_w

    fused_score = (
        visual_score * weights["visual"] +
        temporal_score * weights["temporal"] +
        audio_score * weights["audio"] +
        lip_sync_score * weights["lip_sync"] +
        meta_score * weights["metadata"]
    )

    # Distribute into REAL, AI_GENERATED, FORGED
    # Key differentiators:
    # High visual texture / boundary / lip sync anomalies strongly indicate AI_GENERATED.
    # High temporal / splicing anomalies with normal visual textures indicate FORGED (cut/paste).
    # Low overall anomalies indicate REAL.

    flickering = temporal_analysis.get("flickering_detected", False)
    motion_incon = temporal_analysis.get("motion_inconsistency", False)
    is_generative_flow = temporal_analysis.get("is_generative_flow", False)
    is_splicing = temporal_analysis.get("is_splicing", False)
    sensor_noise_absence = visual_analysis.get("sensor_noise_absence", False)
    is_generative_texture = visual_analysis.get("is_generative_texture", False)

    # 1. AI GENERATED Indicators:
    # a) Generative AI Video (Adobe Firefly, Sora, Runway, Kling, Stable Video Diffusion):
    #    - Continuous generative diffusion flow across frames (is_generative_flow)
    #    - Absence of physical camera sensor PRNU noise (sensor_noise_absence)
    #    - Generative diffusion texture & gradient smoothness (is_generative_texture)
    # b) Facial Deepfakes / Face Swaps:
    #    - High face visual anomaly score (visual_score >= 0.50)
    #    - Severe lip-sync desynchronization (lip_sync_score >= 0.60)
    # c) AI Synthetic Speech:
    #    - Audio synthetic anomaly score >= 0.60
    has_generative_video_signature = bool(
        is_generative_flow or
        (sensor_noise_absence and is_generative_texture and meta_status == "suspicious")
    )
    has_deepfake_signature = bool(
        visual_score >= 0.55 or
        (lip_sync_available and lip_sync_score >= 0.65) or
        (audio_available and audio_score >= 0.70)
    )
    is_ai_generated = has_generative_video_signature or has_deepfake_signature

    # 2. FORGED Indicators (Traditional digital editing / splicing / frame manipulation):
    # - Isolated sharp cut spike between shots (is_splicing) without continuous generative diffusion flow
    is_digital_forgery = bool(is_splicing and not is_generative_flow)

    # 3. Probabilistic 3-way distribution
    if is_ai_generated and not is_digital_forgery:
        prediction = "AI_GENERATED"
        # Calibrate confidence based on multi-pillar evidence strength
        strength = 0.65
        if is_generative_flow:
            strength += 0.15
        if sensor_noise_absence:
            strength += 0.10
        if visual_score >= 0.52:
            strength += 0.10
        if meta_status == "suspicious":
            strength += 0.05
        prob_ai = min(0.96, max(0.68, strength))
        prob_forged = round((1.0 - prob_ai) * (0.25 if is_splicing else 0.10), 2)
        prob_real = round(1.0 - prob_ai - prob_forged, 2)

    elif is_digital_forgery:
        prediction = "FORGED"
        strength = 0.70 + (0.15 if flickering else 0.0) + min(0.10, temporal_score * 0.2)
        prob_forged = min(0.94, max(0.65, strength))
        prob_ai = round((1.0 - prob_forged) * 0.25, 2)
        prob_real = round(1.0 - prob_forged - prob_ai, 2)

    else:
        # Authentic Genuine Video (Camera recordings, handheld pans, real people)
        prediction = "REAL"
        prob_real = min(0.95, max(0.72, 1.0 - fused_score * 0.85))
        prob_ai = round((1.0 - prob_real) * (0.55 if visual_score > 0.45 else 0.35), 2)
        prob_forged = round(1.0 - prob_real - prob_ai, 2)

    # Normalize probabilities to sum cleanly to 1.0
    scores_raw = [max(0.01, prob_real), max(0.01, prob_ai), max(0.01, prob_forged)]
    scores_norm = [s / sum(scores_raw) for s in scores_raw]

    score_dict = {
        "real": round(float(scores_norm[0]), 2),
        "ai_generated": round(float(scores_norm[1]), 2),
        "forged": round(float(scores_norm[2]), 2)
    }

    # Ensure probabilities sum to 1.00 exactly after rounding
    diff = round(1.0 - sum(score_dict.values()), 2)
    if diff != 0:
        pred_key = prediction.lower()
        score_dict[pred_key] = round(score_dict[pred_key] + diff, 2)

    confidence = score_dict[prediction.lower()]

    return {
        "prediction": prediction,
        "scores": score_dict,
        "confidence": confidence
    }
