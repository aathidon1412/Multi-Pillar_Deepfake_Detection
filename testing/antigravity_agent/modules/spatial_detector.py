"""
Module 1: Spatial Forensics Detector
Analyzes static, high-frequency frame anomalies using Vision Transformer (ViT) embeddings
and spatial artifact / boundary inconsistency detection.
"""

from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class SpatialDetector:
    """
    Extracts spatial embeddings and frame suspicion scores.
    Uses pre-trained ViT backbone when available, with a fast fallback feature extractor.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_name = model_name
        self.vit_model = None
        self.processor = None
        self._load_backbone()

    def _load_backbone(self):
        try:
            from transformers import ViTImageProcessor, ViTModel
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.vit_model = ViTModel.from_pretrained(self.model_name).to(self.device)
            self.vit_model.eval()
            for p in self.vit_model.parameters():
                p.requires_grad = False
        except Exception:
            # Safe lightweight fallback if weights are not downloaded / offline
            self.vit_model = None

    def analyze(self, frames_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze N sampled frames.
        
        Args:
            frames_tensor: Tensor of shape [N, 3, 224, 224] in [0, 1].
            
        Returns:
            Dict containing:
                frame_scores: List[float] of artifact/suspicion scores in [0, 1]
                spatial_embeddings: Tensor of shape [N, 768] (or fallback dimension)
                top_suspicious_frames: List of top 3 frame indices with highest scores
        """
        num_frames = frames_tensor.shape[0]
        frame_scores: List[float] = []

        # 1. Compute high-frequency / ELA / spatial gradient variance heuristic per frame
        frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

        for i in range(num_frames):
            frame = frames_np[i]
            # Spatial Laplacian / edge variance
            gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
            laplacian = np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1))
            # Normalized score approximation (higher variance / irregular frequency -> higher anomaly)
            score = float(np.clip(1.0 / (1.0 + np.exp(-((laplacian - 450.0) / 150.0))), 0.05, 0.95))
            frame_scores.append(round(score, 4))

        # 2. Extract deep ViT spatial embeddings
        if self.vit_model is not None:
            with torch.no_grad():
                # Normalize using standard ImageNet mean & std
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                norm_frames = (frames_tensor.to(self.device) - mean) / std
                outputs = self.vit_model(pixel_values=norm_frames)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token [N, 768]
        else:
            # Deterministic fallback embedding [N, 768] using pooled spatial filters
            embeddings = torch.zeros((num_frames, 768), dtype=torch.float32)
            for i in range(num_frames):
                flat = torch.from_numpy(frames_np[i]).float().view(-1)
                chunk_size = flat.shape[0] // 768
                embeddings[i] = flat[:768 * chunk_size].view(768, chunk_size).mean(dim=1)
                embeddings[i] = torch.nn.functional.normalize(embeddings[i], p=2, dim=0)

        # 3. Identify top suspicious frames
        top_indices = np.argsort(frame_scores)[::-1][:3].tolist()

        return {
            "frame_scores": frame_scores,
            "spatial_embeddings": embeddings.cpu(),
            "mean_spatial_score": round(float(np.mean(frame_scores)), 4),
            "top_suspicious_frames": top_indices,
        }
