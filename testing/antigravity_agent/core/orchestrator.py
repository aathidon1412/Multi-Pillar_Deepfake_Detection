"""
Core Orchestrator Agent (pipeline_manager.py)
Coordinates end-to-end ingestion, multi-module execution, consensus fusion, and explainability reporting.
"""

import os
from typing import Any, Dict, Optional
from antigravity_agent.utils.video_loader import load_video_frames, extract_audio_stream
from antigravity_agent.modules.spatial_detector import SpatialDetector
from antigravity_agent.modules.temporal_analyzer import TemporalAnalyzer
from antigravity_agent.modules.audio_detector import AudioDetector
from antigravity_agent.modules.consensus_engine import ConsensusEngine
from antigravity_agent.modules.explainer import Explainer


class OrchestratorAgent:
    """
    Main pipeline coordinator for the Antigravity Deepfake Detection system.
    """

    def __init__(self, device: str = "cpu"):
        print("[OrchestratorAgent] Initializing multi-pillar forensic modules...")
        self.device = device
        self.spatial_detector = SpatialDetector(device=device)
        self.temporal_analyzer = TemporalAnalyzer()
        self.audio_detector = AudioDetector()
        self.consensus_engine = ConsensusEngine()
        self.explainer = Explainer()
        print("[OrchestratorAgent] All modules initialized successfully.")

    def run(self, video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute full forensic analysis pipeline on input video.
        
        Args:
            video_path: Path to target .mp4 video.
            output_dir: Directory to save generated artifacts (e.g. timeline chart).
            
        Returns:
            Structured dictionary with metadata, module scores, final verdict, and report.
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(video_path))
        os.makedirs(output_dir, exist_ok=True)

        timeline_chart_path = os.path.join(output_dir, "anomaly_timeline.png")

        print(f"[OrchestratorAgent] Ingesting video: {video_path}")
        # 1. Uniform frame sampling and metadata reader
        frames_tensor, timestamps, metadata = load_video_frames(video_path, num_frames=16)

        # 2. Audio extraction
        audio_data, sr = extract_audio_stream(video_path)

        # 3. Execute Module 1: Spatial Forensics
        print("[OrchestratorAgent] Running Module 1 (Spatial Forensics)...")
        spatial_res = self.spatial_detector.analyze(frames_tensor)

        # 4. Execute Module 2: Temporal Inconsistency
        print("[OrchestratorAgent] Running Module 2 (Temporal Inconsistency)...")
        temporal_res = self.temporal_analyzer.analyze(frames_tensor)

        # 5. Execute Module 3: Audio & Track Integrity
        print("[OrchestratorAgent] Running Module 3 (Audio Analysis)...")
        audio_res = self.audio_detector.analyze(audio_data, sr, metadata)

        # 6. Execute Module 4: Consensus Aggregator
        print("[OrchestratorAgent] Running Module 4 (Consensus Aggregator)...")
        consensus_res = self.consensus_engine.fuse(spatial_res, temporal_res, audio_res)

        # 7. Execute Module 5: Explainability & Reporting
        print("[OrchestratorAgent] Running Module 5 (Explainability & Report)...")
        report_res = self.explainer.generate_report(
            spatial_results=spatial_res,
            temporal_results=temporal_res,
            audio_results=audio_res,
            consensus=consensus_res,
            timestamps=timestamps,
            timeline_img_path=timeline_chart_path,
        )

        return {
            "metadata": metadata,
            "timestamps": timestamps,
            "spatial_results": spatial_res,
            "temporal_results": temporal_res,
            "audio_results": audio_res,
            "consensus": consensus_res,
            "report": report_res,
        }
