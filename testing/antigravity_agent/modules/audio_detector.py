"""
Module 3: Audio & Track Integrity Detector
Inspects container metadata, audio track presence, and acoustic/spectral consistency.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np


class AudioDetector:
    """
    Analyzes audio spectrogram irregularities and track container metadata.
    """

    def analyze(self, audio_data: Optional[np.ndarray], sample_rate: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            audio_data: 1D numpy array of audio samples (or None if video has no audio).
            sample_rate: Sampling rate (e.g. 16000).
            metadata: Container video metadata from video_loader.
            
        Returns:
            Dict containing:
                has_audio: bool
                audio_anomaly_score: float [0, 1]
                codec_flags: List[str]
                spectral_irregularities: List[str]
        """
        codec_flags: List[str] = []
        irregularities: List[str] = []

        if audio_data is None or len(audio_data) == 0:
            codec_flags.append("NO_AUDIO_STREAM_FOUND")
            return {
                "has_audio": False,
                "audio_anomaly_score": 0.50,  # Neutral / inconclusive default
                "codec_flags": codec_flags,
                "spectral_irregularities": ["Audio stream is absent; skipped acoustic deepfake inspection."],
            }

        # Check audio length vs video duration
        audio_dur = len(audio_data) / float(sample_rate)
        video_dur = float(metadata.get("duration_sec", 0.0))
        if video_dur > 0 and abs(audio_dur - video_dur) > 1.5:
            codec_flags.append("AV_DURATION_MISMATCH")
            irregularities.append(f"A/V track duration mismatch: audio={audio_dur:.2f}s vs video={video_dur:.2f}s.")

        # Compute simple spectral flatness & high-frequency energy ratio
        # using Fast Fourier Transform
        fft_vals = np.abs(np.fft.rfft(audio_data[: min(len(audio_data), sample_rate * 5)]))
        fft_vals = fft_vals + 1e-10

        # Geometric mean / Arithmetic mean = Spectral Flatness Measure (SFM)
        geom_mean = np.exp(np.mean(np.log(fft_vals)))
        arith_mean = np.mean(fft_vals)
        sfm = float(geom_mean / arith_mean)

        # Synthetic/robotic voices frequently exhibit robotic tone gaps or extreme unnatural flatness
        if sfm > 0.45:
            irregularities.append("High spectral flatness indicating potential vocoder or robotic synthesis.")
            audio_score = 0.72
        elif sfm < 0.001:
            irregularities.append("Extremely peaked harmonic resonances detected.")
            audio_score = 0.65
        else:
            audio_score = 0.28

        if "AV_DURATION_MISMATCH" in codec_flags:
            audio_score = min(1.0, audio_score + 0.15)

        return {
            "has_audio": True,
            "audio_anomaly_score": round(float(audio_score), 4),
            "codec_flags": codec_flags,
            "spectral_irregularities": irregularities,
            "spectral_flatness": round(sfm, 4),
        }
