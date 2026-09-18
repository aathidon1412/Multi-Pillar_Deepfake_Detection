import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from backend.models.model_interface import visual_model

def analyze_boundary_anomaly(face_crop: np.ndarray) -> float:
    """
    Measures edge gradient discontinuity along the perimeter of the face crop
    which is characteristic of face swap / seam blending masks.
    """
    if face_crop is None or face_crop.size == 0:
        return 0.1
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return 0.1

    # Sample perimeter strip
    border_px = max(2, min(8, int(min(h, w) * 0.05)))
    top = gray[:border_px, :]
    bottom = gray[-border_px:, :]
    left = gray[:, :border_px]
    right = gray[:, -border_px:]
    center = gray[border_px:-border_px, border_px:-border_px]

    perimeter_var = float((np.var(top) + np.var(bottom) + np.var(left) + np.var(right)) / 4.0)
    center_var = float(np.var(center)) + 1e-6

    # Disparity between border texture and internal face texture
    ratio = abs(perimeter_var - center_var) / (perimeter_var + center_var)
    return float(np.clip(ratio * 1.5, 0.0, 1.0))

def analyze_frequency_texture(image_rgb: np.ndarray) -> float:
    """
    Performs FFT analysis to check for high-frequency attenuation or unnatural grid patterns.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    # Downsample large images for consistent FFT
    if max(h, w) > 256:
        gray = cv2.resize(gray, (256, 256))

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    cy, cx = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    r = int(min(cy, cx) * 0.5)

    # Low frequency vs High frequency energy
    y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
    mask_low = (x - cx)**2 + (y - cy)**2 <= r**2
    low_freq_energy = np.mean(magnitude_spectrum[mask_low])
    high_freq_energy = np.mean(magnitude_spectrum[~mask_low])

    if low_freq_energy > 0:
        hf_ratio = high_freq_energy / low_freq_energy
        # AI generated faces often have lower high-frequency noise or periodic checkerboard
        anomaly = max(0.0, min(1.0, 1.0 - (hf_ratio * 1.8)))
    else:
        anomaly = 0.2

    return float(np.clip(anomaly, 0.0, 1.0))

def analyze_sensor_noise(image_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Measures physical camera sensor noise using median filter residual.
    Real mobile phone / camera sensors under compression typically have noise std >= 1.8.
    Pure synthetic CGI / GAN / AI flat generation produces near-zero residuals (std < 1.3).
    Returns (noise_std, anomaly_score).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    med = cv2.medianBlur(gray, 3)
    res = cv2.absdiff(gray, med)
    noise_std = float(np.std(res))
    anomaly = float(np.clip((2.0 - noise_std) / 1.0, 0.0, 1.0))
    return noise_std, anomaly

def run_visual_analysis(extracted_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executes visual feature evaluation across sampled frames.
    Calculates overall visual anomaly score, detected anomaly categories,
    and individual frame visual scores.
    """
    frame_scores = []
    sensor_noises = []
    anomaly_counts = {
        "Facial texture inconsistency": 0,
        "Unusual face boundary": 0,
        "High-frequency compression artifacts": 0,
        "Lighting & shadow inconsistency": 0,
        "Synthetic texture (Sensor noise absence)": 0,
        "Generative diffusion rendering": 0
    }

    for frame_info in extracted_frames:
        faces = frame_info.get("faces", [])
        reasons_for_frame = []
        img_rgb = frame_info["image_rgb"]

        # 1. Full-frame sensor noise evaluation
        noise_std, noise_anomaly = analyze_sensor_noise(img_rgb)
        sensor_noises.append(noise_std)

        # 2. Face vs Full-Frame branch
        if faces:
            # Analyze human face crop
            face_roi = faces[0]["face_crop"]
            pil_face = Image.fromarray(face_roi)
            
            # Model inference score
            model_score = visual_model.predict_frame(pil_face)
            # Boundary seam anomaly
            boundary_score = analyze_boundary_anomaly(face_roi)
            # Frequency domain texture
            freq_score = analyze_frequency_texture(face_roi)
            
            # Combined frame score
            frame_visual_score = 0.45 * model_score + 0.30 * boundary_score + 0.15 * freq_score + 0.10 * noise_anomaly

            if model_score > 0.60:
                reasons_for_frame.append("Facial texture inconsistency")
                anomaly_counts["Facial texture inconsistency"] += 1
            if boundary_score > 0.55:
                reasons_for_frame.append("Unusual face boundary")
                anomaly_counts["Unusual face boundary"] += 1
            if freq_score > 0.60:
                reasons_for_frame.append("High-frequency compression artifacts")
                anomaly_counts["High-frequency compression artifacts"] += 1
        else:
            # Full-frame generative analysis (Text-to-Video, scenery, AI characters)
            pil_frame = Image.fromarray(img_rgb)
            model_score = visual_model.predict_frame(pil_frame)
            freq_score = analyze_frequency_texture(img_rgb)

            # Measure gradient variance for diffusion smoothness vs sharp contours
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gy, gx = np.gradient(gray.astype(float))
            grad_mag = np.sqrt(gx**2 + gy**2)
            mean_grad = float(np.mean(grad_mag))
            # Generative AI video often combines low mean gradient with clean denoised surfaces
            diff_smoothness = float(np.clip(1.0 - (mean_grad / 28.0), 0.0, 1.0))

            frame_visual_score = 0.40 * noise_anomaly + 0.30 * diff_smoothness + 0.20 * freq_score + 0.10 * model_score

            if noise_anomaly > 0.50:
                reasons_for_frame.append("Synthetic texture (Sensor noise absence)")
                anomaly_counts["Synthetic texture (Sensor noise absence)"] += 1
            if diff_smoothness > 0.60 or freq_score > 0.55:
                reasons_for_frame.append("Generative diffusion rendering")
                anomaly_counts["Generative diffusion rendering"] += 1
            if model_score > 0.65:
                reasons_for_frame.append("Lighting & shadow inconsistency")
                anomaly_counts["Lighting & shadow inconsistency"] += 1

        frame_visual_score = round(float(np.clip(frame_visual_score, 0.05, 0.95)), 3)
        frame_info["visual_score"] = frame_visual_score
        frame_info["visual_reasons"] = reasons_for_frame
        frame_scores.append(frame_visual_score)

    overall_visual_score = float(np.mean(frame_scores)) if frame_scores else 0.20
    # Include 85th percentile to capture localized segments
    if frame_scores:
        p85 = float(np.percentile(frame_scores, 85))
        overall_visual_score = round(0.60 * overall_visual_score + 0.40 * p85, 3)

    mean_sensor_noise = float(np.mean(sensor_noises)) if sensor_noises else 3.5
    # True sensor noise absence requires very low residual (< 1.65), typical of clean synthetic CGI/diffusion
    sensor_noise_absence = bool(mean_sensor_noise < 1.65)
    is_generative_texture = bool(sensor_noise_absence or overall_visual_score >= 0.55)

    # Determine anomalies list (requires at least 25% of frames to avoid single-frame motion blur noise)
    min_count = max(2, len(extracted_frames) // 4)
    detected_anomalies = [k for k, v in anomaly_counts.items() if v >= min_count and overall_visual_score >= 0.45]

    status = "suspicious" if overall_visual_score >= 0.60 else ("slight_anomaly" if overall_visual_score >= 0.45 else "normal")

    return {
        "score": overall_visual_score,
        "status": status,
        "anomalies": detected_anomalies,
        "frame_scores": frame_scores,
        "sensor_noise_absence": sensor_noise_absence,
        "mean_sensor_noise": round(mean_sensor_noise, 3),
        "is_generative_texture": is_generative_texture
    }
