"""
Visualization utilities for antigravity_agent.
Generates timeline anomaly graphs and forensic comparison plots.
"""

from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def generate_anomaly_timeline(
    timestamps: List[float],
    spatial_scores: List[float],
    motion_vectors: List[float],
    output_path: str = "anomaly_timeline.png",
) -> str:
    """
    Generate dual-axis timeline graph showing spatial artifact score and motion divergence.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=150)

    # Align lengths: motion_vectors has N-1 entries
    t_spatial = timestamps[: len(spatial_scores)]
    t_motion = timestamps[: len(motion_vectors)]

    color = "#e63946"
    ax1.set_xlabel("Video Timestamp (seconds)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Spatial Artifact Suspicion", color=color, fontsize=11, fontweight="bold")
    line1 = ax1.plot(t_spatial, spatial_scores, color=color, marker="o", linewidth=2.5, label="Spatial Anomaly")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Secondary axis for motion
    ax2 = ax1.twinx()
    color2 = "#1d3557"
    ax2.set_ylabel("Optical Flow Motion Magnitude", color=color2, fontsize=11, fontweight="bold")
    line2 = ax2.plot(t_motion, motion_vectors, color=color2, marker="s", linestyle="--", linewidth=2.0, label="Motion Vector")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Horizontal decision threshold
    ax1.axhline(0.65, color="red", linestyle=":", alpha=0.7, label="Deepfake Threshold (0.65)")
    ax1.axhline(0.35, color="green", linestyle=":", alpha=0.7, label="Real Threshold (0.35)")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)

    plt.title("Antigravity Agent: Multimodal Temporal-Spatial Anomaly Timeline", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path
