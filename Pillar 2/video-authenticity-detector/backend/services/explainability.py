import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from backend.config import SUSPICIOUS_FRAMES_DIR

def extract_and_annotate_suspicious_frames(
    extracted_frames: List[Dict[str, Any]],
    video_id: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Ranks extracted frames by composite anomaly score and saves top suspicious frames
    to storage/suspicious_frames/ with visual evidence markings.
    """
    SUSPICIOUS_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Score each frame by visual score and temporal jump
    scored_candidates = []
    for f in extracted_frames:
        v_score = f.get("visual_score", 0.0)
        t_score = f.get("temporal_diff", 0.0) * 1.5
        reasons = f.get("visual_reasons", [])
        
        t_raw = f.get("temporal_diff", 0.0)
        composite = max(v_score, t_score)

        has_faces = any(len(fr.get("faces", [])) > 0 for fr in extracted_frames)

        if reasons and v_score >= 0.52:
            reason_candidate = reasons[0]
            if has_faces and "Synthetic texture" in reason_candidate:
                primary_reason = "Authentic reference frame"
            else:
                primary_reason = reason_candidate
        elif t_raw > 0.38:
            primary_reason = "Splice transition cut"
        elif v_score >= 0.60:
            primary_reason = "Visual texture anomaly"
        elif t_raw >= 0.08 and not has_faces:
            primary_reason = "Generative temporal morphing"
        else:
            primary_reason = "Authentic reference frame"

        scored_candidates.append({
            "frame": f,
            "score": round(float(np.clip(composite, 0.10, 0.98)), 2),
            "reason": primary_reason
        })

    # Sort descending by anomaly score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Pick top distinct frames (ensure at least 1.0s between suspicious frames if possible)
    selected = []
    seen_times = []

    for item in scored_candidates:
        f = item["frame"]
        t = f["timestamp_seconds"]
        
        # Avoid clustering all suspicious frames at the exact same second
        if any(abs(t - st) < 0.80 for st in seen_times) and len(selected) > 0 and len(selected) < top_k:
            continue

        seen_times.append(t)
        selected.append(item)
        if len(selected) >= top_k:
            break

    # If all frames were low anomaly, pick at least top 1-2 representative frames
    if not selected and scored_candidates:
        selected = scored_candidates[:2]

    suspicious_records = []

    for item in selected:
        f = item["frame"]
        img_annotated = f["image_bgr"].copy()
        h, w = img_annotated.shape[:2]

        # Annotate face bounding boxes ONLY for validated human faces
        faces = f.get("faces", [])
        human_faces = [face for face in faces if face.get("is_human_face", True)]
        if human_faces:
            for face in human_faces:
                x, y, fw, fh = face["bbox"]
                # High-tech neon cyan / amber bounding box
                cv2.rectangle(img_annotated, (x, y), (x + fw, y + fh), (0, 215, 255), 2)
                # Corner accents
                corner_len = min(15, fw // 4)
                cv2.line(img_annotated, (x, y), (x + corner_len, y), (0, 255, 255), 3)
                cv2.line(img_annotated, (x, y), (x, y + corner_len), (0, 255, 255), 3)
                cv2.line(img_annotated, (x + fw, y + fh), (x + fw - corner_len, y + fh), (0, 255, 255), 3)
                cv2.line(img_annotated, (x + fw, y + fh), (x + fw, y + fh - corner_len), (0, 255, 255), 3)

        # Forensic badge in top-left corner
        badge_text = f"FORENSIC MARKER: {item['reason']} ({int(item['score']*100)}%)"
        text_w = min(w - 10, 10 + len(badge_text) * 10)
        cv2.rectangle(img_annotated, (10, 10), (text_w, 38), (15, 20, 30), -1)
        cv2.rectangle(img_annotated, (10, 10), (text_w, 38), (0, 180, 255), 1)
        cv2.putText(img_annotated, badge_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)

        # Save annotated image
        frame_num = f["frame_number"]
        out_filename = f"{video_id}_frame_{frame_num}.jpg"
        out_path = SUSPICIOUS_FRAMES_DIR / out_filename
        cv2.imwrite(str(out_path), img_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        suspicious_records.append({
            "frame_number": frame_num,
            "timestamp_seconds": f["timestamp_seconds"],
            "image": f"suspicious_frames/{out_filename}",
            "reason": item["reason"],
            "score": item["score"]
        })

    # Sort final suspicious records by timestamp
    suspicious_records.sort(key=lambda x: x["timestamp_seconds"])
    return suspicious_records
