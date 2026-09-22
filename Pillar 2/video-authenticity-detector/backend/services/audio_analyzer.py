import os
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from backend.config import FFMPEG_PATH, FRAMES_DIR
from backend.models.model_interface import audio_model

def extract_audio_track(video_path: str, video_id: str) -> Optional[str]:
    """
    Extracts the audio track from a video as a 16kHz mono WAV file using FFmpeg.
    Returns the path to the wav file if audio exists, or None if no audio track.
    """
    target_dir = FRAMES_DIR / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    wav_path = target_dir / "extracted_audio.wav"

    cmd = [
        str(FFMPEG_PATH),
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path)
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        if proc.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
            return str(wav_path)
        else:
            return None
    except Exception as e:
        print(f"[AUDIO EXTRACTION] FFmpeg audio extraction error: {e}")
        return None

def run_audio_analysis(audio_wav_path: Optional[str]) -> Dict[str, Any]:
    """
    Runs audio authenticity detection on extracted WAV file.
    Does not penalize video if audio is missing.
    """
    if not audio_wav_path or not os.path.isfile(audio_wav_path):
        return {
            "available": False,
            "score": 0.0,
            "status": "not_applicable"
        }

    # Use model interface audio detector
    result = audio_model.predict_audio(audio_wav_path)
    if not result.get("available", False):
        return {
            "available": False,
            "score": 0.0,
            "status": "not_applicable"
        }

    return {
        "available": True,
        "score": result.get("score", 0.25),
        "status": result.get("status", "normal"),
        "rms_energy": result.get("rms_energy", 0.0),
        "zcr": result.get("zcr", 0.0),
        "hf_ratio": result.get("hf_ratio", 0.0)
    }
