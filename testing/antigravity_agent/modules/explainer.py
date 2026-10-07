"""
Module 5: Explainability & Verdict Reporting
Synthesizes natural language reasoning, triggers, and visual evidence timeline.
"""

from typing import Any, Dict, List
from antigravity_agent.utils.visualization import generate_anomaly_timeline


class Explainer:
    """
    Translates raw numbers into natural language evidence and itemized findings.
    """

    def generate_report(
        self,
        spatial_results: Dict[str, Any],
        temporal_results: Dict[str, Any],
        audio_results: Dict[str, Any],
        consensus: Dict[str, Any],
        timestamps: List[float],
        timeline_img_path: str = "anomaly_timeline.png",
    ) -> Dict[str, Any]:
        """
        Builds the itemized report and renders the timeline chart.
        """
        findings: List[str] = []

        # 1. Spatial explanations
        top_frames = spatial_results.get("top_suspicious_frames", [])
        frame_scores = spatial_results.get("frame_scores", [])
        for f_idx in top_frames:
            if f_idx < len(frame_scores) and f_idx < len(timestamps):
                score = frame_scores[f_idx]
                if score > 0.55:
                    t_str = f"{timestamps[f_idx]:.2f}s"
                    findings.append(f"Spatial Module: Elevated texture/boundary artifact at {t_str} (Frame #{f_idx}, suspicion: {score:.2f}).")

        # 2. Temporal explanations
        spikes = temporal_results.get("temporal_spike_indices", [])
        for s_idx in spikes:
            if s_idx < len(timestamps):
                t_str = f"{timestamps[s_idx]:.2f}s"
                findings.append(f"Temporal Module: Abrupt motion velocity spike detected near {t_str} (possible morphing/flicker).")

        # 3. Audio explanations
        for irreg in audio_results.get("spectral_irregularities", []):
            findings.append(f"Audio Module: {irreg}")

        if not findings:
            findings.append("No critical individual module anomalies breached elevated thresholds.")

        # 4. Generate visual timeline graph
        chart_path = generate_anomaly_timeline(
            timestamps=timestamps,
            spatial_scores=spatial_results.get("frame_scores", []),
            motion_vectors=temporal_results.get("motion_vectors", []),
            output_path=timeline_img_path,
        )

        return {
            "verdict": consensus.get("verdict"),
            "probability": consensus.get("deepfake_probability"),
            "summary": consensus.get("description"),
            "itemized_findings": findings,
            "timeline_chart": chart_path,
        }
