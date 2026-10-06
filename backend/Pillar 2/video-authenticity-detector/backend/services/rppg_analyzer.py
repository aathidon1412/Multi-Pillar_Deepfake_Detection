"""
backend/services/rppg_analyzer.py
==================================
Pillar 2 Biological rPPG & Blood Volume Pulse Liveness Analyzer.

Extracts optical capillary blood volume pulse (BVP) from the subject's forehead
region using the green color spectrum (530-600 nm peak hemoglobin absorption).
Stabilizes tracking via exponential moving average (EMA) bounding box coordinates
and velocity-gated jitter rejection. Isolates physiological pulse frequencies
(0.75 - 2.50 Hz = 45 - 150 BPM) via a 4th-order Butterworth bandpass filter.
Performs 4x zero-padded Fast Fourier Transform (FFT) to measure power spectral density
and calculate the Signal-to-Noise Ratio (SNR).
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from scipy.signal import butter, filtfilt
from backend.services.face_detector import face_detector

def butterworth_bandpass(signal: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Applies zero-phase 4th-order Butterworth bandpass filtering."""
    nyq = fs / 2.0
    low_norm = max(0.01, min(0.99, low / nyq))
    high_norm = max(0.01, min(0.99, high / nyq))
    if low_norm >= high_norm:
        return signal
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)

def run_rppg_analysis(video_path: str, max_frames: int = 80, fps_target: float = 30.0) -> Dict[str, Any]:
    """
    Analyzes video for biological blood volume pulse (rPPG liveness).
    Requires at least 5 seconds of continuous physiological tracking to confirm cardiac pulse rhythmicity.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "available": False,
            "bpm": None,
            "snr": None,
            "samples": 0,
            "verdict": "INCONCLUSIVE",
            "reason": "Unable to open video stream",
            "confidence": 0.50
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or fps_target
    if fps <= 0 or np.isnan(fps):
        fps = fps_target

    sig = []
    ema_face = None
    alpha = 0.70
    prev_c = None

    while cap.isOpened() and len(sig) < max_frames * 2:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            if ema_face is not None:
                x, y, w, h = [int(round(v)) for v in ema_face]
            else:
                continue
        else:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces[0]

        c = (x + w / 2.0, y + h / 2.0)
        vel = float(np.sqrt((c[0] - prev_c[0]) ** 2 + (c[1] - prev_c[1]) ** 2)) if prev_c else 0.0
        prev_c = c

        # Reject sudden camera jerks or extreme subject head turns
        if vel > 12.0:
            continue

        if ema_face is None:
            ema_face = np.array([x, y, w, h], dtype=float)
        else:
            ema_face = alpha * np.array([x, y, w, h], dtype=float) + (1.0 - alpha) * ema_face

        ex, ey, ew, eh = [int(round(v)) for v in ema_face]

        # Target upper forehead patch (5% to 32% of face height, 25% to 75% of face width)
        fy1 = ey + int(eh * 0.05)
        fy2 = ey + int(eh * 0.32)
        fx1 = ex + int(ew * 0.25)
        fx2 = ex + int(ew * 0.75)

        roi = frame[fy1:fy2, fx1:fx2]
        if roi.size > 0:
            sig.append(float(np.mean(roi[:, :, 1])))

    cap.release()

    # Require minimum 5 seconds (fps * 5) of continuous physiological tracking
    min_samples = int(fps * 5)
    if len(sig) < min_samples:
        return {
            "available": len(sig) > 0,
            "bpm": None,
            "snr": None,
            "samples": len(sig),
            "verdict": "INCONCLUSIVE",
            "reason": f"Insufficient continuous tracking duration: {len(sig)} frames (need >= {min_samples} for biological cardiac rhythm)",
            "confidence": 0.50
        }

    try:
        flt = butterworth_bandpass(sig, 0.75, 2.50, fps, order=4)
        N = len(flt)
        fft_res = np.abs(np.fft.rfft(flt, n=N * 4))
        freqs = np.fft.rfftfreq(N * 4, d=1.0 / fps)

        valid = (freqs >= 0.75) & (freqs <= 2.50)
        v_freqs = freqs[valid]
        v_mag = fft_res[valid]

        if len(v_mag) == 0:
            return {
                "available": True,
                "bpm": None,
                "snr": None,
                "samples": len(sig),
                "verdict": "INCONCLUSIVE",
                "reason": "Empty spectral band after filtering",
                "confidence": 0.50
            }

        peak_idx = int(np.argmax(v_mag))
        peak_bpm = float(v_freqs[peak_idx] * 60.0)
        peak_pwr = float(v_mag[peak_idx] ** 2)

        noise_pwr = float(np.mean(v_mag[np.abs(v_freqs - v_freqs[peak_idx]) > 0.15] ** 2) + 1e-8)
        snr = float(peak_pwr / noise_pwr)

        if snr >= 2.2 and 50.0 <= peak_bpm <= 135.0:
            confidence = min(0.96, max(0.85, 0.70 + (snr / 12.0)))
            return {
                "available": True,
                "bpm": round(peak_bpm, 1),
                "snr": round(snr, 2),
                "samples": len(sig),
                "verdict": "REAL",
                "reason": f"Authentic biological capillary pulse confirmed: {peak_bpm:.1f} BPM (SNR={snr:.2f})",
                "confidence": round(confidence, 2)
            }
        else:
            return {
                "available": True,
                "bpm": round(peak_bpm, 1),
                "snr": round(snr, 2),
                "samples": len(sig),
                "verdict": "SYNTHETIC",
                "reason": f"No physiological pulse detected: peak {peak_bpm:.1f} BPM (SNR={snr:.2f}) outside human biometrics",
                "confidence": 0.85
            }

    except Exception as e:
        return {
            "available": True,
            "bpm": None,
            "snr": None,
            "samples": len(sig),
            "verdict": "INCONCLUSIVE",
            "reason": f"Signal processing error: {str(e)}",
            "confidence": 0.50
        }
