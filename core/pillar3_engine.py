"""
================================================================================
Pillar 3 Core Engine: Acoustic & Voice Synthetic Speech Forensics
================================================================================
Wraps Hugging Face Wav2Vec2/Audio classification and Librosa HPSS vocal separation.
"""

import sys
import os

# Link to Pillar 3 directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3_DIR = os.path.join(BASE_DIR, "Pillar 3")
if P3_DIR not in sys.path:
    sys.path.insert(0, P3_DIR)

try:
    from detect import classify_audio
except ImportError:
    def classify_audio(audio_path, mode="spoken"):
        raise ImportError("Could not import classify_audio from Pillar 3/detect.py")

__all__ = ["classify_audio"]
