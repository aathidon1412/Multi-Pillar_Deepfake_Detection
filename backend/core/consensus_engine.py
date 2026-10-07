"""
backend/core/consensus_engine.py
================================
Trained Bi-LSTM / Sequence Consensus Fusion Engine for Video Deepfake Forensics.
Fuses frame spatial embeddings, motion metrics, and audio scores using the
trained consensus_head.pth checkpoint.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


class BiLSTMFusionHead(nn.Module):
    """Lightweight Bi-LSTM sequence fusion head for temporal-spatial artifact sequences."""

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(768, feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, spatial_emb: torch.Tensor, motion_feats: torch.Tensor) -> torch.Tensor:
        proj_emb = torch.relu(self.proj(spatial_emb))
        lstm_out, _ = self.lstm(proj_emb)
        seq_pooled = lstm_out.mean(dim=1)
        fused = torch.cat([seq_pooled, motion_feats], dim=-1)
        return self.classifier(fused)


class ConsensusEngine:
    """Fuses module outputs and assigns calibrated verdict and confidence intervals."""

    def __init__(self, model_weights_path: Optional[str] = None):
        force_cpu = os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes")
        self.device = torch.device(
            "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"
        )
        self.model = BiLSTMFusionHead()
        if model_weights_path and os.path.exists(model_weights_path):
            try:
                self.model.load_state_dict(
                    torch.load(model_weights_path, map_location=self.device, weights_only=True)
                )
            except Exception as e:
                print(f"[ConsensusEngine] Warning loading weights: {e}")
        self.model.to(self.device)
        self.model.eval()

    def fuse(
        self,
        spatial_results: Dict[str, Any],
        temporal_results: Dict[str, Any],
        audio_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate multimodal features and output calibrated confidence."""
        spatial_emb = spatial_results.get("spatial_embeddings")
        if not isinstance(spatial_emb, torch.Tensor):
            spatial_emb = torch.zeros((16, 768), dtype=torch.float32)
        if spatial_emb.dim() == 2:
            spatial_emb = spatial_emb.unsqueeze(0)

        mean_motion = float(np.mean(temporal_results.get("motion_vectors", [0.0])))
        variance = float(temporal_results.get("velocity_variance", 0.0))
        motion_tensor = torch.tensor([[mean_motion, variance]], dtype=torch.float32)

        dev = next(self.model.parameters()).device
        spatial_emb = spatial_emb.to(dev)
        motion_tensor = motion_tensor.to(dev)

        with torch.no_grad():
            logits = self.model(spatial_emb, motion_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            model_fake_prob = float(probs[1])

        spatial_mean = float(spatial_results.get("mean_spatial_score", 0.3))
        temporal_score = float(temporal_results.get("temporal_anomaly_score", 0.3))
        has_audio = bool(audio_results.get("has_audio", False))
        audio_score = float(audio_results.get("audio_anomaly_score", 0.5))

        if has_audio:
            calibrated_prob = (
                0.40 * model_fake_prob +
                0.25 * spatial_mean +
                0.20 * temporal_score +
                0.15 * audio_score
            )
        else:
            calibrated_prob = (
                0.45 * model_fake_prob +
                0.35 * spatial_mean +
                0.20 * temporal_score
            )

        calibrated_prob = float(np.clip(calibrated_prob, 0.02, 0.98))

        if calibrated_prob < 0.35:
            verdict = "REAL"
            confidence_level = "HIGH" if calibrated_prob < 0.20 else "MODERATE"
            description = "Video exhibits organic spatial textures and smooth temporal motion continuity."
        elif calibrated_prob <= 0.65:
            verdict = "INCONCLUSIVE"
            confidence_level = "LOW"
            description = "Borderline forensic signals; manual forensic verification recommended."
        else:
            verdict = "DEEPFAKE"
            confidence_level = "HIGH" if calibrated_prob > 0.80 else "MODERATE"
            description = "High density of generative artifacts, unnatural motion variance, or spectral anomalies."

        return {
            "verdict": verdict,
            "deepfake_probability": round(calibrated_prob, 4),
            "authenticity_probability": round(1.0 - calibrated_prob, 4),
            "confidence_level": confidence_level,
            "description": description,
            "pillar_contributions": {
                "spatial": round(spatial_mean, 4),
                "temporal": round(temporal_score, 4),
                "audio": round(audio_score, 4) if has_audio else None,
                "fusion_model_raw": round(model_fake_prob, 4),
            },
        }

__all__ = ["BiLSTMFusionHead", "ConsensusEngine"]
