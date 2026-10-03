"""
================================================================================
Pillar 3 Core Engine: Acoustic & Voice Synthetic Speech Forensics
================================================================================
Wraps Hugging Face Wav2Vec2/Audio classification and Librosa HPSS vocal separation.
"""

import sys
import os
import importlib.util

# Link to Pillar 3 directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3_DIR = os.path.join(BASE_DIR, "Pillar 3")
P3_DETECT_PATH = os.path.join(P3_DIR, "detect.py")

if P3_DIR not in sys.path:
    sys.path.insert(0, P3_DIR)

try:
    if os.path.exists(P3_DETECT_PATH):
        spec = importlib.util.spec_from_file_location("pillar3_detect_module", P3_DETECT_PATH)
        p3_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p3_mod)
        classify_audio = p3_mod.classify_audio
        get_model = p3_mod.get_model
    else:
        from detect import classify_audio, get_model
except Exception as e:
    def classify_audio(audio_path, mode="spoken"):
        raise ImportError(f"Could not load classify_audio from Pillar 3/detect.py: {e}")
    def get_model():
        raise ImportError(f"Could not load get_model from Pillar 3/detect.py: {e}")

__all__ = ["classify_audio", "get_model"]
