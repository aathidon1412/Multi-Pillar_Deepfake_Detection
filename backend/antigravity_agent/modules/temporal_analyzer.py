"""
Module 2: Temporal Inconsistency Analyzer
Detects motion anomalies, generative morphing, flickering, and frame jumps
using Farnebäck optical flow vectors and inter-frame delta analysis.
"""

from typing import Any, Dict, List
import cv2
import numpy as np
import torch


class TemporalAnalyzer:
    """
    Computes dense optical flow and pixel difference across sequential frames.
    """

    def analyze(self, frames_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze motion consistency across sequential frames.
        
        Args:
            frames_tensor: Tensor of shape [N, 3, 224, 224] in [0, 1].
            
        Returns:
            Dict containing:
                motion_vectors: List of pairwise mean motion magnitudes
                pixel_diffs: List of pairwise mean absolute pixel deltas
                velocity_variance: Overall motion smoothness variance
                temporal_spike_indices: Pairs where motion/pixel changes exceed threshold
                temporal_anomaly_score: Normalized summary temporal suspicion score [0, 1]
        """
        num_frames = frames_tensor.shape[0]
        if num_frames < 2:
            return {
                "motion_vectors": [0.0],
                "pixel_diffs": [0.0],
                "velocity_variance": 0.0,
                "temporal_spike_indices": [],
                "temporal_anomaly_score": 0.1,
            }

        # Convert frames to grayscale uint8 arrays with guaranteed contiguous memory
        frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        gray_frames = []
        for f in frames_np:
            try:
                # Fast numpy weighted grayscale conversion
                g = np.dot(f[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                gray_frames.append(np.ascontiguousarray(g))
            except Exception:
                gray_frames.append(np.zeros((f.shape[0], f.shape[1]), dtype=np.uint8))

        motion_magnitudes: List[float] = []
        pixel_deltas: List[float] = []

        for i in range(num_frames - 1):
            prev = gray_frames[i]
            curr = gray_frames[i + 1]

            # 1. Absolute pixel difference
            diff = float(np.mean(np.abs(curr.astype(np.float32) - prev.astype(np.float32)))) / 255.0
            pixel_deltas.append(round(diff, 4))

            # 2. Dense Optical Flow (Farnebäck) with safe fallback
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, curr, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = float(np.mean(mag))
            except Exception:
                # Robust gradient flow fallback if OpenCV engine throws error
                mean_mag = float(np.mean(np.abs(curr.astype(np.float32) - prev.astype(np.float32)))) * 0.1

            motion_magnitudes.append(round(mean_mag, 4))

        # 3. Detect spikes & velocity variance
        motion_arr = np.array(motion_magnitudes)
        mean_motion = np.mean(motion_arr)
        std_motion = np.std(motion_arr)
        variance = float(np.var(motion_arr))

        # A spike occurs if motion deviates significantly from surrounding mean
        spikes: List[int] = []
        threshold = mean_motion + 1.8 * (std_motion + 1e-5)
        for idx, val in enumerate(motion_magnitudes):
            if val > threshold:
                spikes.append(idx)

        # Calibrated temporal anomaly score (irregular morphing / jerky motion triggers higher score)
        norm_score = float(np.clip((std_motion / (mean_motion + 1e-4)) * 0.4 + (len(spikes) * 0.15), 0.05, 0.95))

        return {
            "motion_vectors": motion_magnitudes,
            "pixel_diffs": pixel_deltas,
            "velocity_variance": round(variance, 4),
            "temporal_spike_indices": spikes,
            "temporal_anomaly_score": round(norm_score, 4),
        }
