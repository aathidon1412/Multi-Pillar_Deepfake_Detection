"""
================================================================================
Pillar 3 Core Engine: Acoustic & Voice Synthetic Speech Forensics
================================================================================
Wraps Hugging Face Wav2Vec2/Audio classification and Librosa HPSS vocal separation.
"""

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_NAME = "Hemgg/Deepfake-audio-detection"
CENTRAL_MODEL_DIR = os.path.join(ROOT_DIR, "models", "pillar3", "Acoustic_Model")
ALT_CENTRAL_DIR = os.path.join(ROOT_DIR, "models", "pillar3", "Hemgg_Deepfake_audio_detection")
BACKEND_MODEL_DIR = os.path.join(BASE_DIR, "models", "pillar3", "Acoustic_Model")
LOCAL_MODEL_DIR = CENTRAL_MODEL_DIR if os.path.exists(CENTRAL_MODEL_DIR) else (
    ALT_CENTRAL_DIR if os.path.exists(ALT_CENTRAL_DIR) else BACKEND_MODEL_DIR
)

_feature_extractor = None
_model = None
_device = None

def get_model():
    global _feature_extractor, _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_source = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else MODEL_NAME
        try:
            _feature_extractor = AutoFeatureExtractor.from_pretrained(model_source, local_files_only=True)
            _model = AutoModelForAudioClassification.from_pretrained(model_source, local_files_only=True)
        except Exception:
            _feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
            _model = AutoModelForAudioClassification.from_pretrained(model_source)
        _model.to(_device)
        _model.eval()
    return _feature_extractor, _model, _device

def analyze_audio_composition(y, sr=16000):
    """
    Uses librosa to compute acoustic metrics and detect whether the input audio
    contains music, studio instrumentation, percussion, or heavy autotune.
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

    # 3. High-Frequency energy (>10kHz)
    freqs, psd = scipy.signal.welch(y_segment, sr, nperseg=min(2048, len(y_segment)))
    total_energy = np.sum(psd) + 1e-12
    spectral_centroid = float(np.sum(freqs * psd) / total_energy)
    
    music_score = 0.0
    if hpr > 1.4:
        music_score += 0.40
    if perc_power > 0.005 and hpr > 1.2:
        music_score += 0.35
    if rolloff > 3000 and flatness > 0.020:
        music_score += 0.25

    is_music = bool(music_score >= 0.60 or (hpr > 1.8 and perc_power > 0.006))

    is_compressed_voice = bool(
        rolloff < 2800 and spectral_centroid < 900 and flatness < 0.015
    )

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
    chunk_duration: float = None,
    hop_duration: float = None
):
    """
    Classifies an audio file as REAL vs FAKE.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    _, ext = os.path.splitext(audio_path)
    if ext.lower() not in valid_exts:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(valid_exts))}")

    target_sr = 16000
    try:
        y_raw, orig_sr = librosa.load(audio_path, sr=target_sr, mono=True)
        info = sf.info(audio_path)
    except Exception as e:
        raise ValueError(f"Unable to decode audio file ({audio_path}): {e}")

    composition = analyze_audio_composition(y_raw, sr=target_sr)
    is_music = composition["is_music"]

    is_demixed = False
    if mode == "music" or is_music:
        y_harm, _ = librosa.effects.hpss(y_raw)
        max_val = np.max(np.abs(y_harm)) + 1e-8
        y_eval = y_harm / max_val
        is_demixed = True
    else:
        y_eval = y_raw

    total_duration = len(y_eval) / target_sr

    has_vocoder_cutoff = False
    if info.samplerate >= 24000:
        try:
            y_nat, _ = librosa.load(audio_path, sr=None)
            stft_nat = np.abs(librosa.stft(y_nat, n_fft=2048, hop_length=512))
            freqs_nat = librosa.fft_frequencies(sr=info.samplerate, n_fft=2048)
            spec_mean = np.mean(stft_nat, axis=1)
            peak_e = np.max(spec_mean) + 1e-12
            drop_idx = np.where(spec_mean < peak_e * 0.001)[0]
            cutoff_freq = freqs_nat[drop_idx[0]] if len(drop_idx) > 0 else info.samplerate / 2
            if 11000 <= cutoff_freq <= 15500:
                has_vocoder_cutoff = True
        except Exception:
            pass

    flatness = composition.get("flatness", 0.0)
    high_flatness_flag = bool(flatness >= 0.020 and not composition.get("is_compressed_voice", False))

    if chunk_duration is None:
        if total_duration <= 6.0:
            effective_chunk = 1.5
            default_hop = 0.5
        elif total_duration <= 30.0:
            effective_chunk = 2.5
            default_hop = 1.25
        else:
            effective_chunk = 4.0
            default_hop = 2.0
    else:
        effective_chunk = chunk_duration
        default_hop = 2.0

    if hop_duration is None:
        if total_duration <= 6.0:
            active_hop = default_hop
        elif total_duration <= 30.0:
            active_hop = default_hop
        elif total_duration <= 150.0:
            active_hop = 4.0
        elif total_duration <= 360.0:
            active_hop = 5.0
        elif total_duration <= 600.0:
            active_hop = 7.0
        else:
            active_hop = 10.0
    else:
        active_hop = hop_duration

    feature_extractor, model, device = get_model()

    window_samples = int(effective_chunk * target_sr)
    hop_samples = int(active_hop * target_sr)
    total_samples = len(y_eval)

    chunk_slices = []
    chunk_timestamps = []
    is_silent_chunk = []

    global_rms = float(np.sqrt(np.mean(y_eval**2))) + 1e-12
    silence_threshold = max(0.001, global_rms * 0.05)

    if total_samples <= window_samples:
        chunk_slices.append(y_eval)
        chunk_timestamps.append((0.0, round(total_duration, 2)))
        is_silent_chunk.append(False)
    else:
        for start in range(0, total_samples - window_samples + 1, hop_samples):
            chunk = y_eval[start:start + window_samples]
            start_sec = round(start / target_sr, 2)
            end_sec = round((start + window_samples) / target_sr, 2)
            
            chunk_rms = float(np.sqrt(np.mean(chunk**2)))
            silent = chunk_rms < silence_threshold

            chunk_slices.append(chunk)
            chunk_timestamps.append((start_sec, end_sec))
            is_silent_chunk.append(silent)

        last_covered = chunk_timestamps[-1][1] if chunk_timestamps else 0.0
        if (total_duration - last_covered) > 0.5 and total_samples > window_samples:
            tail_chunk = y_eval[total_samples - window_samples:total_samples]
            tail_start = round((total_samples - window_samples) / target_sr, 2)
            tail_end = round(total_duration, 2)
            chunk_slices.append(tail_chunk)
            chunk_timestamps.append((tail_start, tail_end))
            is_silent_chunk.append(float(np.sqrt(np.mean(tail_chunk**2))) < silence_threshold)

    if not chunk_slices:
        chunk_slices.append(y_eval)
        chunk_timestamps.append((0.0, round(total_duration, 2)))
        is_silent_chunk.append(False)

    active_indices = [i for i, s in enumerate(is_silent_chunk) if not s]
    active_slices = [chunk_slices[i] for i in active_indices]

    if not active_slices:
        active_indices = [0]
        active_slices = [chunk_slices[0]]

    active_probs = []
    batch_size = 8
    with torch.no_grad():
        for b_idx in range(0, len(active_slices), batch_size):
            batch_audio = active_slices[b_idx:b_idx + batch_size]
            inputs = feature_extractor(batch_audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if probs.ndim == 1:
                probs = np.expand_dims(probs, axis=0)
            active_probs.extend(probs)

    chunk_probs = []
    active_map = {idx: p for idx, p in zip(active_indices, active_probs)}
    for i in range(len(chunk_slices)):
        if i in active_map:
            chunk_probs.append(active_map[i])
        else:
            chunk_probs.append(np.array([0.10, 0.90]))

    timeline = []
    tampered_intervals = []
    for (t_start, t_end), prob, silent in zip(chunk_timestamps, chunk_probs, is_silent_chunk):
        f_p = float(prob[0])
        r_p = float(prob[1])
        item = {
            "start_time": t_start,
            "end_time": t_end,
            "fake_prob": round(f_p, 4),
            "real_prob": round(r_p, 4),
            "is_suspicious": f_p >= 0.70 and not silent,
            "is_silent": silent
        }
        timeline.append(item)
        if f_p >= 0.70 and not silent:
            tampered_intervals.append(item)

    voice_probs = [p for p, s in zip(chunk_probs, is_silent_chunk) if not s]
    if not voice_probs:
        voice_probs = chunk_probs

    avg_probs = np.mean(voice_probs, axis=0)
    fake_mean = float(avg_probs[0])
    real_mean = float(avg_probs[1])

    max_fake_chunk = max(float(p[0]) for p, s in zip(chunk_probs, is_silent_chunk) if not s) if any(not s for s in is_silent_chunk) else float(avg_probs[0])
    num_fake_chunks = sum(1 for p, s in zip(chunk_probs, is_silent_chunk) if float(p[0]) >= 0.70 and not s)
    num_fake_chunks_85 = sum(1 for p, s in zip(chunk_probs, is_silent_chunk) if float(p[0]) >= 0.85 and not s)
    fake_chunk_ratio = num_fake_chunks / max(1, len(voice_probs))

    disclaimer = None
    is_compressed_voice = composition.get("is_compressed_voice", False)

    if is_compressed_voice:
        disclaimer = "WhatsApp / VoIP Speech: Lossy voice codec detected (low bitrate Opus/AMR). Acoustic metrics calibrated for mobile speech compression."
        calibrated_fake_mean = max(0.05, fake_mean * 0.25)
        calibrated_max_fake = max(0.10, max_fake_chunk * 0.35)
        fake_mean = calibrated_fake_mean
        real_mean = 1.0 - fake_mean
        max_fake_chunk = calibrated_max_fake
        num_fake_chunks = sum(1 for p, s in zip(chunk_probs, is_silent_chunk) if (float(p[0]) * 0.35) >= 0.70 and not s)
        num_fake_chunks_85 = sum(1 for p, s in zip(chunk_probs, is_silent_chunk) if (float(p[0]) * 0.35) >= 0.85 and not s)
        fake_chunk_ratio = num_fake_chunks / max(1, len(voice_probs))

    if is_demixed:
        if mode == "music":
            disclaimer = "Studio Produced Track / Music Demixed: HPSS demixing applied to isolate vocals from background instruments and percussion."
            if composition["hpr"] > 1.5 or is_music:
                real_mean = max(0.40, min(0.60, real_mean))
                fake_mean = 1.0 - real_mean
        elif is_music and not disclaimer:
            disclaimer = "Background Music / Drums Detected: Automatic HPSS vocal isolation applied to filter percussion and evaluate vocal authenticity accurately."

    is_localized_tamper = False
    if (not is_compressed_voice and num_fake_chunks_85 >= 2) or (total_duration <= 6.0 and max_fake_chunk >= 0.85 and (high_flatness_flag or num_fake_chunks >= 2)):
        prediction = "FAKE"
        fake_raw = max_fake_chunk
        real_raw = 1.0 - fake_raw
        is_localized_tamper = bool(fake_chunk_ratio < 0.60 and total_duration > 6.0)
        verdict_type = "LOCALIZED_TAMPERING" if is_localized_tamper else "FULLY_SYNTHETIC"
    elif fake_mean >= 0.50 or fake_chunk_ratio >= 0.50:
        prediction = "FAKE"
        fake_raw = fake_mean
        real_raw = 1.0 - fake_raw
        verdict_type = "FULLY_SYNTHETIC"
    else:
        prediction = "REAL"
        real_raw = real_mean
        fake_raw = 1.0 - real_raw
        verdict_type = "AUTHENTIC"

    fake_conf = fake_raw * 100.0
    real_conf = real_raw * 100.0
    confidence = fake_conf if prediction == "FAKE" else real_conf

    res = {
        "prediction": prediction,
        "verdict_type": verdict_type,
        "is_localized_tamper": is_localized_tamper,
        "confidence": confidence,
        "real_confidence": real_conf,
        "fake_confidence": fake_conf,
        "samplerate": info.samplerate,
        "duration": info.duration,
        "channels": info.channels,
        "chunks_evaluated": len(chunk_probs),
        "hop_seconds_used": active_hop,
        "suspicious_segments_count": num_fake_chunks,
        "tampered_intervals": tampered_intervals,
        "timeline": timeline,
        "is_music": is_music,
        "is_demixed": is_demixed,
        "disclaimer": disclaimer,
        "has_vocoder_cutoff": has_vocoder_cutoff,
        "high_flatness": high_flatness_flag,
        "composition": composition,
        "mode": mode,
        "model_used": MODEL_NAME
    }
    try:
        from backend.core import generate_pillar3_xai
        res["xai"] = generate_pillar3_xai(res, audio_path=audio_path, mode=mode)
    except Exception as xe:
        print(f"[Pillar 3 XAI Generation Warning]: {xe}")
    return res

__all__ = ["classify_audio", "get_model", "analyze_audio_composition"]
