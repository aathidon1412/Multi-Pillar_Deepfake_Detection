import cv2
import numpy as np
from typing import List, Dict, Any

def run_temporal_analysis(extracted_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates temporal continuity between consecutive frames:
    - Luminance variance & flickering
    - Motion vector jump / structural difference
    - Face bounding box jitter
    """
    if len(extracted_frames) < 2:
        return {
            "score": 0.15,
            "status": "normal",
            "flickering_detected": False,
            "motion_inconsistency": False
        }

    luminances = []
    structural_diffs = []
    face_jitters = []

    for i, frame in enumerate(extracted_frames):
        gray = cv2.cvtColor(frame["image_rgb"], cv2.COLOR_RGB2GRAY)
        luminances.append(float(np.mean(gray)))

        # Compare with previous frame
        if i > 0:
            prev_gray = cv2.cvtColor(extracted_frames[i - 1]["image_rgb"], cv2.COLOR_RGB2GRAY)
            # Resize if resolution changed
            if prev_gray.shape != gray.shape:
                prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

            # Frame difference
            abs_diff = cv2.absdiff(prev_gray, gray)
            diff_score = float(np.mean(abs_diff)) / 255.0
            structural_diffs.append(diff_score)

            # Face position jitter if faces exist in both
            f_curr = frame.get("faces", [])
            f_prev = extracted_frames[i - 1].get("faces", [])
            if f_curr and f_prev:
                b1 = f_curr[0]["bbox"]
                b2 = f_prev[0]["bbox"]
                area1 = b1[2] * b1[3]
                area2 = b2[2] * b2[3]
                size_ratio = max(area1, area2) / (min(area1, area2) + 1e-4)
                
                # Center point displacement normalized by frame diagonal
                c1 = np.array([b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0])
                c2 = np.array([b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0])
                disp = np.linalg.norm(c1 - c2) / np.sqrt(frame["width"]**2 + frame["height"]**2)
                
                # Only count displacement as jitter if tracking the SAME subject
                # If displacement > 0.18 or size ratio > 2.0, the camera panned or a new person was detected
                if disp <= 0.18 and size_ratio <= 2.0:
                    face_jitters.append(float(disp))
                else:
                    face_jitters.append(0.0)
            else:
                face_jitters.append(0.0)

    # 1. Flickering evaluation (high second derivative of luminance)
    lum_diffs = np.abs(np.diff(luminances))
    lum_jitter = float(np.std(lum_diffs))
    flickering_detected = bool(lum_jitter > 12.0 or (len(lum_diffs) > 0 and np.max(lum_diffs) > 35.0))

    # 2. Structural differences and motion flow analysis
    s_diffs = np.array(structural_diffs) if structural_diffs else np.array([0.0])
    mean_diff = float(np.mean(s_diffs))
    max_diff = float(np.max(s_diffs))
    std_diff = float(np.std(s_diffs))
    peak_ratio = float(max_diff / (mean_diff + 1e-6))

    # Differentiate Generative Diffusion Flow vs Digital Splicing vs Natural Camera Panning
    # Splicing (FORGED): an isolated sharp cut spike at a single transition point
    is_splicing = bool(peak_ratio >= 2.8 and max_diff > 0.38)
    
    # Check if faces are present across sequence
    has_human_faces = any(len(f.get("faces", [])) > 0 for f in extracted_frames)
    
    # Generative AI Video (AI_GENERATED): continuous elevated diffusion morphing without camera hardware provenance
    is_generative_flow = bool(mean_diff >= 0.06 and peak_ratio < 2.5 and not is_splicing and not has_human_faces)
    
    # Face jitter check for genuine human faces tracking the same subject
    face_jitter_detected = bool(len(face_jitters) > 0 and np.max(face_jitters) > 0.15)
    motion_inconsistency = bool(is_splicing or face_jitter_detected)

    # Calculate calibrated temporal score
    base_score = 0.15
    if is_splicing:
        base_score += 0.55
    elif is_generative_flow:
        base_score += min(0.45, 0.25 + mean_diff * 1.5)
    elif motion_inconsistency:
        base_score += 0.30

    if flickering_detected:
        base_score += 0.20

    base_score += min(0.15, mean_diff * 0.4)
    temporal_score = round(float(np.clip(base_score, 0.08, 0.95)), 3)
    status = "suspicious" if temporal_score >= 0.65 else ("slight_anomaly" if temporal_score >= 0.45 else "normal")

    # Annotate frame-level temporal jump
    for i in range(1, len(extracted_frames)):
        d = structural_diffs[i - 1]
        extracted_frames[i]["temporal_diff"] = round(d, 3)
        if "visual_reasons" not in extracted_frames[i]:
            extracted_frames[i]["visual_reasons"] = []
            
        if is_splicing and d > 0.35:
            extracted_frames[i]["visual_reasons"].append("Splice transition cut")
        elif is_generative_flow and d >= 0.07:
            extracted_frames[i]["visual_reasons"].append("Generative temporal morphing")
        elif d > 0.30 or (flickering_detected and lum_diffs[i - 1] > 20):
            extracted_frames[i]["visual_reasons"].append("Temporal inconsistency")

    return {
        "score": temporal_score,
        "status": status,
        "flickering_detected": flickering_detected,
        "motion_inconsistency": motion_inconsistency,
        "is_generative_flow": is_generative_flow,
        "is_splicing": is_splicing,
        "mean_diff": round(mean_diff, 4),
        "max_diff": round(max_diff, 4),
        "peak_ratio": round(peak_ratio, 2)
    }
