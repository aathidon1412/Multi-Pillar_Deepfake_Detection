import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from backend.core.consensus_engine import ConsensusEngine

# Load trained Bi-LSTM / MLP Consensus Head if available
_consensus_model: Optional["ConsensusEngine"] = None
_consensus_attempted: bool = False

def get_trained_consensus_engine() -> Optional["ConsensusEngine"]:
    global _consensus_model, _consensus_attempted
    if not _consensus_attempted:
        _consensus_attempted = True
        try:
            from backend.core.consensus_engine import ConsensusEngine
            root_dir = Path(__file__).resolve().parent.parent.parent
            central_weights = root_dir / "models" / "pillar2" / "consensus_head.pth"
            legacy_weights = root_dir / "consensus_head.pth"
            weights_path = central_weights if central_weights.exists() else legacy_weights
            if weights_path.exists():
                _consensus_model = ConsensusEngine(model_weights_path=str(weights_path))
            else:
                _consensus_model = ConsensusEngine()
        except Exception as e:
            print(f"[Pillar 2 Classifier] Notice: Using rule-based fallback, consensus head not loaded: {e}")
            _consensus_model = None
    return _consensus_model


def run_feature_fusion_and_classification(
    visual_analysis: Dict[str, Any],
    temporal_analysis: Dict[str, Any],
    audio_analysis: Dict[str, Any],
    lip_sync_analysis: Dict[str, Any],
    metadata_analysis: Dict[str, Any],
    rppg_analysis: Optional[Dict[str, Any]] = None,
    video_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fuses multi-pillar forensic signals and computes probabilistic 3-way classification:
    - REAL: genuine authentic video with natural textures and physics (or verified biological pulse)
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

    # 1. Check if trained Bi-LSTM / MLP consensus engine is available
    engine = get_trained_consensus_engine()
    model_prob = None
    if engine is not None:
        try:
            dev = getattr(engine, "device", torch.device("cpu"))
            sp_res = {
                "mean_spatial_score": visual_score,
                "spatial_embeddings": torch.zeros((16, 768), dtype=torch.float32, device=dev),
            }
            tp_res = {"motion_vectors": [temporal_score], "velocity_variance": temporal_analysis.get("variance", 0.0), "temporal_anomaly_score": temporal_score}
            au_res = {"has_audio": audio_available, "audio_anomaly_score": audio_score}
            fusion_out = engine.fuse(sp_res, tp_res, au_res)
            model_prob = fusion_out.get("deepfake_probability")
        except Exception:
            model_prob = None

    # Dynamic Pillar Weighting
    if model_prob is not None:
        weights = {"model": 0.35, "visual": 0.20, "temporal": 0.15, "lip_sync": 0.15, "audio": 0.08, "metadata": 0.07}
        if not audio_available:
            weights["visual"] += 0.04
            weights["temporal"] += 0.04
            weights["audio"] = 0.0
        if not lip_sync_available:
            weights["visual"] += 0.08
            weights["temporal"] += 0.07
            weights["lip_sync"] = 0.0
        total_w = sum(weights.values())
        for k in weights:
            weights[k] /= total_w

        fused_score = (
            model_prob * weights["model"] +
            visual_score * weights["visual"] +
            temporal_score * weights["temporal"] +
            lip_sync_score * weights.get("lip_sync", 0.0) +
            audio_score * weights["audio"] +
            meta_score * weights["metadata"]
        )
    else:
        weights = {"visual": 0.35, "temporal": 0.25, "audio": 0.15, "lip_sync": 0.15, "metadata": 0.10}
        if not audio_available:
            weights["visual"] += 0.08
            weights["temporal"] += 0.07
            weights["audio"] = 0.0
        if not lip_sync_available:
            weights["visual"] += 0.08
            weights["temporal"] += 0.07
            weights["lip_sync"] = 0.0
        total_w = sum(weights.values())
        for k in weights:
            weights[k] /= total_w

        fused_score = (
            visual_score * weights["visual"] +
            temporal_score * weights["temporal"] +
            audio_score * weights["audio"] +
            lip_sync_score * weights.get("lip_sync", 0.0) +
            meta_score * weights["metadata"]
        )

    flickering = temporal_analysis.get("flickering_detected", False)
    is_generative_flow = temporal_analysis.get("is_generative_flow", False)
    is_splicing = temporal_analysis.get("is_splicing", False)
    sensor_noise_absence = visual_analysis.get("sensor_noise_absence", False)
    is_generative_texture = visual_analysis.get("is_generative_texture", False)

    # Biological rPPG Pulse verification
    rppg_res = rppg_analysis or {}
    rppg_verdict = rppg_res.get("verdict", "INCONCLUSIVE")
    rppg_bpm = rppg_res.get("bpm")
    rppg_snr = rppg_res.get("snr", 0.0)

    is_digital_forgery = bool(is_splicing and not is_generative_flow)
    override_reason = None

    # DETERMINISTIC OVERRIDE 1: Biological Pulse Confirmed (Pillar 2 Core Innovation)
    if rppg_verdict == "REAL" and visual_score <= 0.75:
        prediction = "REAL"
        strength = min(0.96, max(0.85, 0.72 + (float(rppg_snr or 2.2) / 10.0)))
        prob_real = strength
        prob_ai = round((1.0 - prob_real) * 0.60, 2)
        prob_forged = round(1.0 - prob_real - prob_ai, 2)
        override_reason = f"Pillar 2 Biological Pulse Confirmed ({rppg_bpm} BPM, SNR={rppg_snr})"

    # DETERMINISTIC OVERRIDE 2: Generative Diffusion Video Disruption (Firefly, Sora, Kling)
    elif is_generative_flow or (sensor_noise_absence and is_generative_texture and temporal_score >= 0.45):
        prediction = "AI_GENERATED"
        strength = min(0.96, max(0.82, temporal_score + 0.15))
        prob_ai = strength
        prob_forged = round((1.0 - prob_ai) * 0.20, 2)
        prob_real = round(1.0 - prob_ai - prob_forged, 2)
        override_reason = "Pillar 2 Generative Diffusion Flow & Texture Signature"

    # DETERMINISTIC OVERRIDE 3: Visual ViT Deepfake / Synthetic Avatar Face
    elif visual_score >= 0.52:
        prediction = "AI_GENERATED"
        strength = min(0.96, max(0.80, visual_score))
        prob_ai = strength
        prob_forged = round((1.0 - prob_ai) * (0.25 if is_splicing else 0.10), 2)
        prob_real = round(1.0 - prob_ai - prob_forged, 2)
        override_reason = f"Pillar 2 Visual Neural Artifact Anomaly ({visual_score*100:.1f}%)"

    # DETERMINISTIC OVERRIDE 4: Lip-Sync Acoustic Incoherence (Dubbed / Reenacted Deepfake)
    elif lip_sync_available and lip_sync_score >= 0.72:
        prediction = "AI_GENERATED"
        strength = min(0.95, max(0.78, lip_sync_score))
        prob_ai = strength
        prob_forged = round((1.0 - prob_ai) * 0.25, 2)
        prob_real = round(1.0 - prob_ai - prob_forged, 2)
        override_reason = f"Pillar 2 Lip-Sync Asynchrony & Mouth Motion Disparity ({lip_sync_score*100:.1f}%)"

    # DETERMINISTIC OVERRIDE 5: Synthetic Biological Pulse Anomaly (rPPG Missing Liveness)
    elif rppg_verdict == "SYNTHETIC" and (lip_sync_score >= 0.55 or audio_score >= 0.50 or visual_score >= 0.35):
        prediction = "AI_GENERATED"
        strength = min(0.92, max(0.75, 0.60 + (lip_sync_score * 0.25)))
        prob_ai = strength
        prob_forged = round((1.0 - prob_ai) * 0.20, 2)
        prob_real = round(1.0 - prob_ai - prob_forged, 2)
        override_reason = f"Pillar 2 Synthetic rPPG Liveness Anomaly & Acoustic-Visual Desync"

    # DETERMINISTIC OVERRIDE 6: Digital Splicing / Frame Cutting
    elif is_digital_forgery:
        prediction = "FORGED"
        strength = 0.70 + (0.15 if flickering else 0.0) + min(0.10, temporal_score * 0.2)
        prob_forged = min(0.94, max(0.65, strength))
        prob_ai = round((1.0 - prob_forged) * 0.25, 2)
        prob_real = round(1.0 - prob_forged - prob_ai, 2)
        override_reason = "Pillar 2 Digital Splicing / Cut Anomaly"

    # DEFAULT WEIGHTED CONSENSUS
    else:
        if model_prob is not None:
            if model_prob < 0.35:
                prediction = "REAL"
                prob_real = min(0.96, max(0.75, 1.0 - model_prob))
                prob_ai = round((1.0 - prob_real) * 0.40, 2)
                prob_forged = round(1.0 - prob_real - prob_ai, 2)
                override_reason = f"Trained Bi-LSTM Fusion Head: Organic Coherence (Fake Prob: {model_prob*100:.1f}%)"
            elif model_prob > 0.65:
                prediction = "AI_GENERATED"
                prob_ai = min(0.96, max(0.72, model_prob))
                prob_forged = round((1.0 - prob_ai) * 0.15, 2)
                prob_real = round(1.0 - prob_ai - prob_forged, 2)
                override_reason = f"Trained Bi-LSTM Fusion Head: Generative Artifacts Flagged ({model_prob*100:.1f}%)"
            else:
                if fused_score >= 0.50:
                    prediction = "AI_GENERATED"
                    prob_ai = min(0.85, max(0.60, model_prob))
                    prob_forged = round((1.0 - prob_ai) * 0.20, 2)
                    prob_real = round(1.0 - prob_ai - prob_forged, 2)
                else:
                    prediction = "REAL"
                    prob_real = min(0.85, max(0.60, 1.0 - model_prob))
                    prob_ai = round((1.0 - prob_real) * 0.40, 2)
                    prob_forged = round(1.0 - prob_real - prob_ai, 2)
                override_reason = f"Trained Bi-LSTM Consensus: Soft Multi-Pillar Fusion ({model_prob*100:.1f}%)"
        elif visual_score < 0.50:
            prediction = "REAL"
            prob_real = min(0.95, max(0.75, 1.0 - fused_score * 0.75))
            prob_ai = round((1.0 - prob_real) * (0.50 if visual_score > 0.40 else 0.30), 2)
            prob_forged = round(1.0 - prob_real - prob_ai, 2)
            override_reason = f"Multi-Pillar Video Consensus (Natural Coherence, {visual_score*100:.1f}%)"
        else:
            prediction = "AI_GENERATED"
            prob_ai = min(0.92, max(0.70, visual_score))
            prob_forged = round((1.0 - prob_ai) * 0.15, 2)
            prob_real = round(1.0 - prob_ai - prob_forged, 2)
            override_reason = f"Multi-Pillar Video Consensus Anomaly ({visual_score*100:.1f}%)"

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
        "confidence": confidence,
        "override_reason": override_reason,
        "rppg_verdict": rppg_verdict,
        "rppg_bpm": rppg_bpm,
        "rppg_snr": rppg_snr
    }
