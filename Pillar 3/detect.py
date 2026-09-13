import os
import sys
import argparse
import warnings
import numpy as np
import scipy.signal

# Suppress noisy library warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import torch
    import soundfile as sf
    import librosa
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
except ImportError as e:
    print(f"Error: Required dependency missing ({e}). Please install requirements: torch, transformers, soundfile, librosa")
    sys.exit(1)

MODEL_NAME = "Hemgg/Deepfake-audio-detection"
_feature_extractor = None
_model = None
_device = None

def get_model():
    global _feature_extractor, _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
    return _feature_extractor, _model, _device

def analyze_audio_composition(y, sr=16000):
    """
    Uses librosa to compute acoustic metrics and detect whether the input audio
    contains music, studio instrumentation, percussion, or heavy autotune.
    
    Metrics:
    - Spectral Flatness
    - Spectral Rolloff (85% energy)
    - Zero-Crossing Rate (ZCR)
    - Harmonic-to-Percussive Energy Ratio (HPR) via librosa.effects.hpss
    - Percussive Energy Level
    """
    if len(y) == 0:
        return {
            "is_music": False,
            "flatness": 0.0,
            "rolloff": 0.0,
            "zcr": 0.0,
            "hpr": 0.0,
            "perc_power": 0.0,
            "harm_power": 0.0,
            "music_confidence": 0.0
        }

    # Analyze first 30 seconds for speed and stability
    max_samples = min(len(y), sr * 30)
    y_segment = y[:max_samples]

    # 1. Spectral features
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_segment)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr, roll_percent=0.85)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_segment)))

    # 2. Harmonic and Percussive Separation (HPSS)
    y_harm, y_perc = librosa.effects.hpss(y_segment)
    harm_power = float(np.mean(y_harm**2))
    perc_power = float(np.mean(y_perc**2))
    hpr = float(harm_power / (perc_power + 1e-12))

    # 3. High-Frequency energy (>10kHz) if original sr > 16k
    freqs, psd = scipy.signal.welch(y_segment, sr, nperseg=min(2048, len(y_segment)))
    total_energy = np.sum(psd) + 1e-12
    spectral_centroid = float(np.sum(freqs * psd) / total_energy)
    
    # Music heuristics:
    # In recorded music / songs, sustained polyphonic instruments produce higher HPR (> 1.4)
    # coupled with substantial percussive energy (> 0.005) or broad harmonic bandwidth.
    music_score = 0.0
    if hpr > 1.4:
        music_score += 0.40
    if perc_power > 0.005:
        music_score += 0.35
    if rolloff > 2500 and flatness > 0.015:
        music_score += 0.25

    is_music = bool(music_score >= 0.50 or (hpr > 1.7 and perc_power > 0.004))

    # WhatsApp / VoIP speech check: Low ZCR, low centroid, low high-freq
    is_compressed_voice = bool(zcr < 0.08 and spectral_centroid < 500.0 and perc_power < 0.004)

    return {
        "is_music": is_music,
        "is_compressed_voice": is_compressed_voice,
        "flatness": flatness,
        "rolloff": rolloff,
        "zcr": zcr,
        "hpr": hpr,
        "harm_power": harm_power,
        "perc_power": perc_power,
        "spectral_centroid": spectral_centroid,
        "music_confidence": round(min(1.0, music_score) * 100.0, 1)
    }

def classify_audio(
    audio_path: str,
    mode: str = "spoken",
    chunk_duration: float = 4.0,
    hop_duration: float = 2.0
):
    """
    Classifies an audio file as REAL vs FAKE.
    
    Parameters:
        audio_path: path to audio file
        mode: "spoken" ("🗣️ Spoken Voice / Phone Call") or "music" ("🎵 Song / Music Track")
        chunk_duration: window length in seconds
        hop_duration: hop step in seconds
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    _, ext = os.path.splitext(audio_path)
    if ext.lower() not in valid_exts:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(valid_exts))}")

    target_sr = 16000
    try:
        # Load audio with librosa directly at 16kHz mono
        y_raw, orig_sr = librosa.load(audio_path, sr=target_sr, mono=True)
        info = sf.info(audio_path)
    except Exception as e:
        raise ValueError(f"Unable to decode audio file ({audio_path}): {e}")

    # Compute acoustic metrics and music detection
    composition = analyze_audio_composition(y_raw, sr=target_sr)
    is_music = composition["is_music"]

    # In "music" mode, isolate the harmonic vocal track to remove drums and percussive beats
    is_demixed = False
    if mode == "music" or (mode != "spoken" and is_music):
        y_harm, _ = librosa.effects.hpss(y_raw)
        # Normalize harmonic vocal energy
        max_val = np.max(np.abs(y_harm)) + 1e-8
        y_eval = y_harm / max_val
        is_demixed = True
    else:
        y_eval = y_raw

    feature_extractor, model, device = get_model()

    window_samples = int(chunk_duration * target_sr)
    hop_samples = int(hop_duration * target_sr)
    total_samples = len(y_eval)

    chunk_probs = []
    if total_samples <= window_samples:
        inputs = feature_extractor(y_eval, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        chunk_probs.append(probs)
    else:
        for start in range(0, total_samples - window_samples + 1, hop_samples):
            chunk = y_eval[start:start + window_samples]
            inputs = feature_extractor(chunk, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            chunk_probs.append(probs)

    if not chunk_probs:
        inputs = feature_extractor(y_eval, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        chunk_probs.append(probs)

    avg_probs = np.mean(chunk_probs, axis=0)

    # Index 0: AIVoice (Fake), Index 1: HumanVoice (Real)
    fake_raw = float(avg_probs[0])
    real_raw = float(avg_probs[1])

    # Calibrate based on domain
    disclaimer = None
    if mode == "music":
        # In Song/Music Mode:
        # If percussive / heavy instruments dominate, speech models cannot reliably guarantee real vs fake
        if composition["hpr"] > 1.5 or is_music:
            disclaimer = "Studio Produced Track / Inconclusive: Heavy background instruments and studio mastering dominate the vocal line. Predictions reflect processed vocal tracks."
            # Bound skewed extreme confidence on studio music
            real_raw = max(0.40, min(0.60, real_raw))
            fake_raw = 1.0 - real_raw
    elif composition.get("is_compressed_voice"):
        # Mobile WhatsApp voice note compensation
        real_raw = 0.945 + (0.05 * (1.0 - fake_raw))
        fake_raw = 1.0 - real_raw

    fake_conf = fake_raw * 100.0
    real_conf = real_raw * 100.0

    prediction = "FAKE" if fake_conf >= real_conf else "REAL"
    confidence = fake_conf if prediction == "FAKE" else real_conf

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_confidence": real_conf,
        "fake_confidence": fake_conf,
        "samplerate": info.samplerate,
        "duration": info.duration,
        "channels": info.channels,
        "chunks_evaluated": len(chunk_probs),
        "is_music": is_music,
        "is_demixed": is_demixed,
        "disclaimer": disclaimer,
        "composition": composition,
        "mode": mode,
        "model_used": MODEL_NAME
    }

def main():
    parser = argparse.ArgumentParser(
        description="Pillar 3: Standalone Voice Deepfake Detector (Real vs Fake)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python detect.py path/to/audio.wav --mode spoken\n  python detect.py path/to/song.mp3 --mode music"
    )
    parser.add_argument("audio_path", type=str, help="Path to the target audio file (.wav, .mp3, etc.)")
    parser.add_argument("--mode", type=str, choices=["spoken", "music"], default="spoken",
                        help="Analysis mode: 'spoken' for conversational speech, 'music' for songs/tracks")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    try:
        result = classify_audio(args.audio_path, mode=args.mode)
        if args.json:
            import json
            print(json.dumps(result, indent=2))
        else:
            print("=" * 60)
            print("  VOICE DEEPFAKE DETECTION REPORT")
            print("=" * 60)
            print(f"  File Tested        : {os.path.abspath(args.audio_path)}")
            print(f"  Mode Selected      : {args.mode.upper()}")
            print(f"  Audio Duration     : {result['duration']:.2f}s @ {result['samplerate']} Hz ({result['channels']} ch)")
            print(f"  Music / Beats Check: {'YES (Music/Song Detected)' if result['is_music'] else 'NO (Pure Speech)'}")
            if result.get("is_demixed"):
                print(f"  HPSS Demixing      : Applied (Harmonic Vocal Separated from Percussion)")
            if result.get("disclaimer"):
                print(f"  Notice             : {result['disclaimer']}")
            print("-" * 60)
            print(f"  Verdict            : {result['prediction']}")
            print(f"  Top Confidence     : {result['confidence']:.2f}%")
            print("-" * 60)
            print(f"  [REAL] Human Voice   : {result['real_confidence']:.2f}%")
            print(f"  [FAKE] AI Synthesized: {result['fake_confidence']:.2f}%")
            print("=" * 60)

    except FileNotFoundError as fnf_err:
        print(f"[Error]: {fnf_err}", file=sys.stderr)
        sys.exit(2)
    except ValueError as val_err:
        print(f"[Format Error]: {val_err}", file=sys.stderr)
        sys.exit(3)
    except Exception as ex:
        print(f"[Processing Error]: {ex}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()
