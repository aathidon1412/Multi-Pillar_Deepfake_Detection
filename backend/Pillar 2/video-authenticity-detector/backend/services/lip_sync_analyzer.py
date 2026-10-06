import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

def compute_mouth_motion_series(extracted_frames: List[Dict[str, Any]]) -> List[float]:
    """
    Computes frame-by-frame mouth region optical motion / variance.
    """
    motions = []
    prev_mouth = None

    for frame in extracted_frames:
        faces = frame.get("faces", [])
        if faces and "mouth_crop" in faces[0] and faces[0]["mouth_crop"].size > 0:
            m_crop = faces[0]["mouth_crop"]
            gray_mouth = cv2.cvtColor(m_crop, cv2.COLOR_RGB2GRAY)
            # Resize mouth to standard 64x64 for optical delta
            gray_mouth = cv2.resize(gray_mouth, (64, 64))

            if prev_mouth is not None:
                delta = float(np.mean(cv2.absdiff(prev_mouth, gray_mouth))) / 255.0
                motions.append(delta)
            else:
                motions.append(0.0)
            prev_mouth = gray_mouth
        else:
            motions.append(0.0)

    return motions

def compute_audio_energy_at_timestamps(audio_wav_path: str, timestamps: List[float], window_sec: float = 0.25) -> List[float]:
    """
    Extracts RMS speech energy from WAV file corresponding to video frame timestamps.
    """
    import scipy.io.wavfile as wavfile
    try:
        sample_rate, data = wavfile.read(audio_wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(float)
        max_val = np.max(np.abs(data)) + 1e-6
        data = data / max_val

        energies = []
        win_samples = int(sample_rate * window_sec)

        for t in timestamps:
            center_idx = int(t * sample_rate)
            start_idx = max(0, center_idx - win_samples // 2)
            end_idx = min(len(data), center_idx + win_samples // 2)

            if end_idx > start_idx:
                segment = data[start_idx:end_idx]
                rms = float(np.sqrt(np.mean(segment ** 2)))
            else:
                rms = 0.0
            energies.append(rms)

        return energies
    except Exception as e:
        print(f"[LIP SYNC] Audio energy extraction error: {e}")
        return [0.0] * len(timestamps)

def run_lip_sync_analysis(extracted_frames: List[Dict[str, Any]], audio_wav_path: Optional[str]) -> Dict[str, Any]:
    """
    Cross-correlates mouth movement series with audio speech activity series.
    Returns anomaly score (high score = out of sync or mismatched speech/mouth).
    """
    if not audio_wav_path or not os.path.isfile(audio_wav_path):
        return {
            "available": False,
            "score": 0.0,
            "status": "not_applicable",
            "reason": "Audio track not available"
        }

    # Check if faces are detected in at least 20% of frames
    frames_with_faces = sum(1 for f in extracted_frames if f.get("faces"))
    if frames_with_faces < max(3, len(extracted_frames) * 0.20):
        return {
            "available": False,
            "score": 0.0,
            "status": "not_applicable",
            "reason": "No continuous human face detected"
        }

    timestamps = [f["timestamp_seconds"] for f in extracted_frames]
    mouth_motions = compute_mouth_motion_series(extracted_frames)
    audio_energies = compute_audio_energy_at_timestamps(audio_wav_path, timestamps)

    # Normalize vectors
    m_arr = np.array(mouth_motions)
    a_arr = np.array(audio_energies)

    m_std = np.std(m_arr)
    a_std = np.std(a_arr)

    # If both have variation, measure correlation
    if m_std > 1e-4 and a_std > 1e-4:
        # Correlation coefficient
        corr = float(np.corrcoef(m_arr, a_arr)[0, 1])
        if np.isnan(corr):
            corr = 0.5
        # Low or negative correlation indicates desynchronization (common in wav2lip or audio dubs)
        # corr ~ 0.5+ is normal, corr < 0.1 is suspicious
        lip_sync_anomaly = max(0.0, min(1.0, (0.60 - corr) / 0.80))
    else:
        # Low speech activity or static mouth
        lip_sync_anomaly = 0.25

    lip_sync_score = round(float(np.clip(lip_sync_anomaly, 0.05, 0.95)), 3)
    status = "suspicious" if lip_sync_score >= 0.65 else ("slight_anomaly" if lip_sync_score >= 0.45 else "normal")

    return {
        "available": True,
        "score": lip_sync_score,
        "status": status,
        "correlation": round(float(corr) if 'corr' in locals() and not np.isnan(corr) else 0.0, 3)
    }
