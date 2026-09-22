import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from backend.config import FRAMES_DIR, MAX_ANALYSIS_FRAMES, SAMPLE_FRAME_INTERVAL_SEC

def extract_sampled_frames(video_path: str, video_id: str) -> List[Dict[str, Any]]:
    """
    Extracts sampled frames from a video file into storage/frames/{video_id}/.
    Returns list of dicts with frame index, timestamp, path, and cv2 image array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame extraction: {video_path}")

    target_dir = FRAMES_DIR / video_id
    target_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine sampling interval
    step_frames = max(1, int(fps * SAMPLE_FRAME_INTERVAL_SEC))
    
    # If video is very long, dynamically adjust step so we don't exceed MAX_ANALYSIS_FRAMES
    expected_samples = total_frames // step_frames
    if expected_samples > MAX_ANALYSIS_FRAMES:
        step_frames = max(1, total_frames // MAX_ANALYSIS_FRAMES)

    extracted = []
    frame_idx = 0
    sampled_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step_frames == 0:
                timestamp = round(frame_idx / fps, 2)
                frame_filename = f"frame_{frame_idx:05d}.jpg"
                frame_path = target_dir / frame_filename

                # Save temporary frame to disk
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                extracted.append({
                    "frame_number": frame_idx,
                    "timestamp_seconds": timestamp,
                    "frame_path": str(frame_path),
                    "image_bgr": frame,
                    "image_rgb": frame_rgb,
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                })

                sampled_count += 1
                if sampled_count >= MAX_ANALYSIS_FRAMES:
                    break

            frame_idx += 1

    finally:
        cap.release()

    if not extracted:
        raise RuntimeError("No frames could be extracted from video.")

    return extracted
