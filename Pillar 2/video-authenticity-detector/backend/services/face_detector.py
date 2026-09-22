import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from backend.config import CASCADE_XML

class FaceDetector:
    def __init__(self):
        # 1. Check local Pillar 2 cascade
        cascade_path = str(CASCADE_XML)
        if not os.path.isfile(cascade_path):
            # 2. Fall back to cv2 built-in cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print(f"[FACE DETECTOR WARNING] Could not load cascade from {cascade_path}")

    def detect_in_frame(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in an RGB frame.
        Returns list of detected face dictionaries containing:
        - bbox: [x, y, w, h]
        - face_crop: np.ndarray (RGB)
        - mouth_crop: np.ndarray (RGB)
        - confidence: float
        """
        if self.face_cascade.empty() or frame_rgb is None:
            return []

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        h_frame, w_frame = gray.shape[:2]

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        detected_faces = []
        for (x, y, w, h) in faces:
            # Add 20% padding around face for forensic context
            pad_x = int(w * 0.20)
            pad_y = int(h * 0.20)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_frame, x + w + pad_x)
            y2 = min(h_frame, y + h + pad_y)

            face_crop = frame_rgb[y1:y2, x1:x2]

            # Approximate mouth region: lower 35% of detected face, centered horizontally
            mouth_y1 = int(y + 0.65 * h)
            mouth_y2 = min(h_frame, y + h)
            mouth_x1 = int(x + 0.20 * w)
            mouth_x2 = int(x + 0.80 * w)
            mouth_crop = frame_rgb[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

            # Validate human skin tone in YCrCb color space
            # Human skin across all ethnicities typically clusters in Cr: [130, 175], Cb: [75, 130]
            ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_RGB2YCrCb)
            cr = ycrcb[..., 1]
            cb = ycrcb[..., 2]
            skin_mask = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)
            skin_ratio = float(np.mean(skin_mask))

            # Only classify as valid human face if it possesses actual skin coloration
            # This eliminates false-positive boxes on mechanical robots, bamboo, textures, and clothing
            is_human_face = bool(skin_ratio >= 0.08)

            # Confidence based on size and skin coherence
            base_conf = float((w * h) / (w_frame * h_frame * 0.10) + 0.65)
            if not is_human_face:
                base_conf *= 0.4  # Drastically down-weight non-human false positives
            confidence = min(0.99, max(0.20, base_conf))

            detected_faces.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "padded_bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "face_crop": face_crop,
                "mouth_crop": mouth_crop if mouth_crop.size > 0 else face_crop,
                "is_human_face": is_human_face,
                "skin_ratio": round(skin_ratio, 3),
                "confidence": round(confidence, 2)
            })

        return detected_faces

# Singleton instance
face_detector = FaceDetector()

def detect_faces_in_frames(extracted_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs face detection over all extracted frames.
    Attaches detected faces to each frame and returns summary stats.
    """
    total_faces_found = 0
    frames_with_faces = 0

    for frame_info in extracted_frames:
        faces = face_detector.detect_in_frame(frame_info["image_rgb"])
        # Only preserve genuine human faces for downstream face modeling
        human_faces = [f for f in faces if f.get("is_human_face", False)]
        frame_info["faces"] = human_faces
        if human_faces:
            frames_with_faces += 1
            total_faces_found += len(human_faces)

    return {
        "faces_detected": frames_with_faces > 0,
        "face_count": total_faces_found,
        "frames_with_faces": frames_with_faces,
        "total_sampled_frames": len(extracted_frames)
    }
