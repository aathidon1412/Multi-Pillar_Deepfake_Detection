"""
pillar2_hybrid.py  -  USMFE Pillar 2 Hybrid Deepfake Detector
=================================================================
Combines two independent forensic channels:
  1. Spatial / Visual AI (dima806/deepfake_vs_real_image_detection)
  2. Biological rPPG (remote photoplethysmography via green-channel FFT)

Usage:
    python pillar2_hybrid.py <path_to_video.mp4>

Output:
    * Structured JSON verdict printed to stdout
    * pillar2_evidence.png  - 2-panel forensic evidence plot
"""

import cv2
import sys
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import butter, filtfilt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

_local_cascade = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
_cv2_cascade   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_PATH   = _cv2_cascade if os.path.isfile(_cv2_cascade) else _local_cascade
HF_MODEL           = "dima806/deepfake_vs_real_image_detection"
MAX_FRAMES         = 90
PROCESS_EVERY_N    = 3
FPS_DEFAULT        = 30.0
RPPG_BPF_LOW       = 0.75
RPPG_BPF_HIGH      = 2.5
RPPG_SNR_THRESH    = 2.2
RPPG_BPM_MIN       = 50
RPPG_BPM_MAX       = 130
MAX_JITTER_PX      = 12.0
EMA_ALPHA          = 0.70
VISUAL_FAKE_THRESH = 0.70
EVIDENCE_IMG       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pillar2_evidence.png")


def load_detector():
    from transformers import pipeline as hf_pipeline
    print("  Loading visual deepfake model ...")
    return hf_pipeline("image-classification", model=HF_MODEL)


def get_fake_score(predictions):
    for pred in predictions:
        if "FAKE" in pred["label"].upper():
            return float(pred["score"])
    for pred in predictions:
        if "REAL" in pred["label"].upper():
            return 1.0 - float(pred["score"])
    return 0.5


def butterworth_bandpass(signal, low, high, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_rppg_snr(fft_freqs, fft_mag, peak_freq, band=0.15):
    signal_mask = np.abs(fft_freqs - peak_freq) <= band
    noise_mask  = (fft_freqs >= RPPG_BPF_LOW) & (fft_freqs <= RPPG_BPF_HIGH) & ~signal_mask
    sig_power   = np.mean(fft_mag[signal_mask] ** 2) if signal_mask.any() else 0.0
    noise_power = np.mean(fft_mag[noise_mask]  ** 2) if noise_mask.any()  else 1e-9
    return float(sig_power / (noise_power + 1e-9))


def analyze_video(video_path):
    if not os.path.isfile(video_path):
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {os.path.basename(video_path)}  |  FPS={fps:.1f}  |  Frames={total_frames}")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print("[WARNING] haarcascade XML not found - face detection may fail.")

    detector = load_detector()

    frame_count      = 0
    processed_visual = 0
    fake_scores      = []
    frame_indices    = []
    green_signal     = []
    velocities       = []
    discarded_rppg   = 0
    accepted_rppg    = 0
    prev_centroid    = None
    ema_face         = None
    prev_face        = None

    print("  Processing frames ...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw     = frame.shape[:2]

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        if len(faces) == 0:
            if ema_face is not None:
                x, y, w, h = [int(round(v)) for v in ema_face]
            elif prev_face is not None:
                (x, y, w, h) = prev_face
            else:
                continue
        else:
            # Sort by area descending so we always pick the primary face, avoiding tiny background artifacts
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            (x, y, w, h) = faces[0]
            prev_face = (x, y, w, h)

        # Centroid of detected face
        cx, cy = x + w / 2.0, y + h / 2.0

        # Frame-to-frame velocity (jitter)
        if prev_centroid is not None:
            velocity = float(np.sqrt((cx - prev_centroid[0])**2 + (cy - prev_centroid[1])**2))
        else:
            velocity = 0.0
        prev_centroid = (cx, cy)
        velocities.append(velocity)

        # EMA smoothing for face bounding box (alpha=0.7)
        if ema_face is None:
            ema_face = np.array([x, y, w, h], dtype=float)
        else:
            ema_face = EMA_ALPHA * np.array([x, y, w, h], dtype=float) + (1.0 - EMA_ALPHA) * ema_face

        ex, ey, ew, eh = [int(round(v)) for v in ema_face]

        # Biological rPPG green-channel sampling:
        # Only discard individual frames where instant velocity > 12 px (head turns / jerks)
        if velocity > MAX_JITTER_PX:
            discarded_rppg += 1
        else:
            # Forehead patch anchored using EMA-smoothed face box
            fh_y1 = max(0, ey + int(eh * 0.05))
            fh_y2 = max(0, ey + int(eh * 0.32))
            fh_x1 = max(0, ex + int(ew * 0.25))
            fh_x2 = min(fw, ex + int(ew * 0.75))
            forehead_roi = frame_rgb[fh_y1:fh_y2, fh_x1:fh_x2]
            if forehead_roi.size > 0:
                green_signal.append(float(np.mean(forehead_roi[:, :, 1])))
                accepted_rppg += 1

        # Visual AI Inference (1:1 square crop with proportional padding, resized to 224x224 with LANCZOS)
        if processed_visual < MAX_FRAMES and frame_count % PROCESS_EVERY_N == 0:
            # Proportional 1:1 square centered on face
            # Close-up portrait shots (face fills >35% frame) need 1.65x to cover chin/hair; medium shots use 1.40x
            face_ratio = max(ew, eh) / max(1, min(fw, fh))
            scale = 1.65 if face_ratio >= 0.35 else 1.40
            side = max(40, int(max(ew, eh) * scale))
            ecx, ecy = ex + ew / 2.0, ey + eh / 2.0
            x1 = int(round(ecx - side / 2.0))
            y1 = int(round(ecy - side / 2.0))

            # Border reflection padding guarantees exact 1:1 square without black artificial edges
            pad = side
            frame_padded = cv2.copyMakeBorder(frame_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            crop_square = frame_padded[y1 + pad : y1 + pad + side, x1 + pad : x1 + pad + side]

            try:
                pil_img = Image.fromarray(crop_square).resize((224, 224), Image.Resampling.LANCZOS)
                preds   = detector(pil_img)
                fscore  = get_fake_score(preds)
                fake_scores.append(fscore)
                frame_indices.append(frame_count)
                processed_visual += 1
                if processed_visual % 15 == 0:
                    print(f"    Visual AI: {processed_visual}/{MAX_FRAMES} frames ...")
            except Exception as exc:
                print(f"    [WARN] Frame {frame_count} inference error: {exc}")

        if processed_visual >= MAX_FRAMES and len(green_signal) > int(fps * 15):
            break

    cap.release()
    print(f"  Done. Visual frames: {len(fake_scores)} | rPPG stable samples: {len(green_signal)} (discarded {discarded_rppg} jitter frames)")

    # --- CHANNEL 1: VISUAL AI VERDICT ---
    if len(fake_scores) == 0:
        visual_score   = 0.0
        visual_verdict = "INCONCLUSIVE (No faces detected)"
    else:
        scores_arr = np.array(fake_scores)
        if len(scores_arr) >= 5:
            # Drop top and bottom 10% outliers, then take median of remaining frames
            p10 = np.percentile(scores_arr, 10)
            p90 = np.percentile(scores_arr, 90)
            trimmed = scores_arr[(scores_arr >= p10) & (scores_arr <= p90)]
            if len(trimmed) == 0:
                trimmed = scores_arr
            visual_score = float(np.median(trimmed))
        else:
            visual_score = float(np.median(scores_arr))

        visual_verdict = "FAKE" if visual_score > VISUAL_FAKE_THRESH else "REAL"

    # --- CHANNEL 2: rPPG VERDICT ---
    rppg_bpm       = None
    rppg_snr       = None
    rppg_verdict   = "INCONCLUSIVE"
    rppg_reason    = "Insufficient signal"
    filtered_signal = np.array([])
    fft_freqs_plot  = np.array([])
    fft_mag_plot    = np.array([])
    peak_freq_hz    = None

    MIN_RPPG_SAMPLES = int(fps * 6)
    mean_jitter = round(float(np.mean(velocities)), 2) if velocities else 0.0
    max_jitter  = round(float(np.max(velocities)), 2)  if velocities else 0.0

    if len(green_signal) >= MIN_RPPG_SAMPLES:
        sig = np.array(green_signal, dtype=float)
        sig -= np.polyval(np.polyfit(np.arange(len(sig)), sig, 1), np.arange(len(sig)))

        try:
            filtered_signal = butterworth_bandpass(sig, RPPG_BPF_LOW, RPPG_BPF_HIGH, fps)
            N = len(filtered_signal)
            fft_full = np.abs(np.fft.rfft(filtered_signal, n=N * 4))
            fft_f    = np.fft.rfftfreq(N * 4, d=1.0 / fps)

            band_mask      = (fft_f >= RPPG_BPF_LOW) & (fft_f <= RPPG_BPF_HIGH)
            fft_mag_plot   = fft_full[band_mask]
            fft_freqs_plot = fft_f[band_mask]

            peak_idx     = np.argmax(fft_mag_plot)
            peak_freq_hz = fft_freqs_plot[peak_idx]
            peak_bpm     = peak_freq_hz * 60.0
            snr          = compute_rppg_snr(fft_freqs_plot, fft_mag_plot, peak_freq_hz)

            rppg_bpm = round(float(peak_bpm), 1)
            rppg_snr = round(float(snr), 3)

            if snr >= RPPG_SNR_THRESH and RPPG_BPM_MIN <= peak_bpm <= RPPG_BPM_MAX:
                rppg_verdict = "REAL"
                rppg_reason  = f"Valid biological pulse detected: {rppg_bpm:.1f} BPM (SNR={snr:.2f})"
            else:
                rppg_verdict = "SYNTHETIC"
                rppg_reason  = f"No physiological pulse detected: BPM={peak_bpm:.1f}, SNR={snr:.2f} (need SNR>={RPPG_SNR_THRESH})"
        except Exception as exc:
            rppg_verdict = "INCONCLUSIVE"
            rppg_reason  = f"rPPG filter error: {exc}"
    else:
        rppg_verdict = "INCONCLUSIVE"
        rppg_reason  = f"Only {len(green_signal)} stable samples (need >={MIN_RPPG_SAMPLES}, discarded {discarded_rppg} jitter frames)"

    # --- ENSEMBLE VERDICT ---
    if visual_score > VISUAL_FAKE_THRESH:
        final_verdict = "FAKE"
        final_reason  = f"Visual AI score {visual_score*100:.1f}% > {VISUAL_FAKE_THRESH*100:.0f}% threshold."
    elif rppg_verdict == "REAL":
        final_verdict = "REAL"
        final_reason  = f"Visual AI score below threshold ({visual_score*100:.1f}%) AND biological pulse confirmed ({rppg_bpm} BPM)."
    elif "INCONCLUSIVE" in rppg_verdict:
        final_verdict = "REAL (Visual only - rPPG inconclusive)"
        final_reason  = f"Visual score below threshold ({visual_score*100:.1f}%); rPPG was inconclusive ({rppg_reason})."
    else:
        final_verdict = "FAKE"
        final_reason  = f"Visual score below threshold ({visual_score*100:.1f}%) but no physiological pulse detected by rPPG ({rppg_reason})."

    result = {
        "video"        : os.path.basename(video_path),
        "visual_ai"    : {
            "score_trimmed_median": round(visual_score * 100, 2),
            "score_90th_pct"      : round(visual_score * 100, 2),
            "frames_evaluated"    : len(fake_scores),
            "threshold"           : VISUAL_FAKE_THRESH,
            "verdict"             : visual_verdict,
        },
        "rppg"         : {
            "bpm"                     : rppg_bpm,
            "snr"                     : rppg_snr,
            "samples_analyzed"        : len(green_signal),
            "discarded_jitter_frames" : discarded_rppg,
            "mean_velocity_px"        : mean_jitter,
            "max_velocity_px"         : max_jitter,
            "verdict"                 : rppg_verdict,
            "reason"                  : rppg_reason,
        },
        "final_verdict": final_verdict,
        "final_reason" : final_reason,
    }

    # --- EVIDENCE PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")

    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    verdict_color = "#f85149" if "FAKE" in final_verdict else "#3fb950"
    if fake_scores:
        ax1.plot(frame_indices, [s * 100 for s in fake_scores],
                 color="#58a6ff", linewidth=1.4, alpha=0.85, label="Frame AI Score")
        ax1.axhline(y=VISUAL_FAKE_THRESH * 100, color="#f85149", linestyle="--", linewidth=1.5,
                    label=f"Decision Threshold ({VISUAL_FAKE_THRESH*100:.0f}%)")
        ax1.axhline(y=visual_score * 100, color="#ffa657", linestyle=":",
                    linewidth=1.5, label=f"Trimmed Median: {visual_score*100:.1f}%")
        ax1.fill_between(frame_indices, [s * 100 for s in fake_scores], alpha=0.15, color="#58a6ff")
    ax1.set_title(f"Visual AI Deepfake Score\nVerdict: {visual_verdict}  ({visual_score*100:.1f}%)",
                  color="#e6edf3", fontsize=12, pad=12)
    ax1.set_xlabel("Frame Number", color="#8b949e")
    ax1.set_ylabel("AI Confidence - FAKE (%)", color="#8b949e")
    ax1.set_ylim(-3, 103)
    ax1.tick_params(colors="#8b949e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")
    ax1.grid(True, color="#21262d", linewidth=0.7)
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    if len(filtered_signal) > 0 and len(fft_freqs_plot) > 0:
        t_axis = np.arange(len(filtered_signal)) / fps
        ax2_t  = ax2
        ax2_f  = ax2.twinx()
        ax2_t.plot(t_axis, filtered_signal, color="#3fb950", linewidth=1.2,
                   alpha=0.75, label="Filtered Green Pulse")
        ax2_t.set_xlabel("Time (seconds)", color="#8b949e")
        ax2_t.set_ylabel("Amplitude (a.u.)", color="#3fb950")
        ax2_t.tick_params(axis="y", colors="#3fb950")
        bpm_axis = fft_freqs_plot * 60
        ax2_f.plot(bpm_axis, fft_mag_plot, color="#ffa657", linewidth=1.4,
                   alpha=0.85, linestyle="--", label="FFT Spectrum")
        if peak_freq_hz is not None:
            ax2_f.axvline(x=peak_freq_hz * 60, color="#f85149", linewidth=1.5,
                          linestyle=":", label=f"Peak: {rppg_bpm} BPM")
        ax2_f.set_ylabel("FFT Magnitude", color="#ffa657")
        ax2_f.tick_params(axis="y", colors="#ffa657")
        lines1, labels1 = ax2_t.get_legend_handles_labels()
        lines2, labels2 = ax2_f.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9, loc="upper right")
        bpm_str = f"{rppg_bpm} BPM" if rppg_bpm else "N/A"
        snr_str = f"SNR={rppg_snr:.2f}" if rppg_snr else "SNR=N/A"
        ax2.set_title(f"rPPG Capillary Pulse - {bpm_str}, {snr_str}\nVerdict: {rppg_verdict}",
                      color="#e6edf3", fontsize=12, pad=12)
    else:
        ax2.text(0.5, 0.5, f"rPPG Signal Unavailable\n{rppg_reason}",
                 ha="center", va="center", color="#8b949e", fontsize=11, transform=ax2.transAxes, wrap=True)
        ax2.set_title(f"rPPG Capillary Pulse\nVerdict: {rppg_verdict}", color="#e6edf3", fontsize=12, pad=12)
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    ax2.grid(True, color="#21262d", linewidth=0.7)

    fig.suptitle(
        f"USMFE Pillar 2 Hybrid Forensic Report - {os.path.basename(video_path)}\nFinal Verdict: {final_verdict}",
        color=verdict_color, fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(EVIDENCE_IMG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Evidence plot saved -> {EVIDENCE_IMG}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        for candidate in ["fake_avatar.mp4", "real_face.mp4", "test_video.mp4"]:
            candidate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), candidate)
            if os.path.exists(candidate_path):
                video_path = candidate_path
                break
        else:
            print("Usage: python pillar2_hybrid.py <video_path>")
            sys.exit(1)
    else:
        video_path = sys.argv[1]

    print("\n" + "=" * 64)
    print("  USMFE Pillar 2 - Hybrid Visual + rPPG Deepfake Detector")
    print("=" * 64)

    verdict_data = analyze_video(video_path)

    print("\n" + "=" * 64)
    print("  STRUCTURED JSON VERDICT")
    print("=" * 64)
    print(json.dumps(verdict_data, indent=2))
    print("=" * 64 + "\n")
