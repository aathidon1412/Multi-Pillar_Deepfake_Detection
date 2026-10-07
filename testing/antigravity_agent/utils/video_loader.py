"""
Video loader utility for antigravity_agent.
Extracts equidistant frames, container metadata, and audio streams from input videos.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch


def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[torch.Tensor, List[float], Dict[str, Any]]:
    """
    Uniformly sample `num_frames` from a video file.
    
    Args:
        video_path: Absolute or relative path to the .mp4 video file.
        num_frames: Number of equidistant frames to sample (default 16).
        target_size: (H, W) to resize frames for deep learning backbones.
        
    Returns:
        frames_tensor: PyTorch FloatTensor of shape [num_frames, 3, H, W] normalized in [0, 1].
        timestamps: List of timestamp offsets (in seconds) for each sampled frame.
        metadata: Dictionary with fps, total_frames, duration_sec, width, height.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    metadata: Dict[str, Any] = {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": duration_sec,
        "width": orig_w,
        "height": orig_h,
    }

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has zero readable frames.")

    # Determine equidistant frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        # Pad with last frame if needed
        while len(indices) < num_frames:
            indices.append(indices[-1])
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    sampled_frames: List[np.ndarray] = []
    timestamps: List[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            # Fallback frame
            frame_rgb = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        else:
            # Pure numpy BGR to RGB channel reverse (100% crash-proof)
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            if (frame_rgb.shape[0], frame_rgb.shape[1]) != target_size:
                frame_rgb = cv2.resize(frame_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

        sampled_frames.append(frame_rgb)
        timestamps.append(round(idx / fps, 3))

    cap.release()

    # Fill if short
    while len(sampled_frames) < num_frames:
        if sampled_frames:
            sampled_frames.append(sampled_frames[-1].copy())
            timestamps.append(timestamps[-1])
        else:
            sampled_frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
            timestamps.append(0.0)

    # Convert to Tensor: [N, H, W, C] -> [N, C, H, W] normalized to [0, 1]
    frames_np = np.stack(sampled_frames, axis=0).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)

    return frames_tensor, timestamps, metadata


def extract_audio_stream(video_path: str, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """
    Extract audio samples and sample rate from video file using moviepy or fallback.
    Returns (audio_array, sample_rate) or (None, target_sr) if no audio track exists.
    """
    try:
        import importlib
        editor = importlib.import_module("moviepy.editor")
        video = editor.VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            return None, target_sr

        # Export audio array
        audio_array = video.audio.to_soundarray(fps=target_sr)
        video.close()

        # Convert stereo to mono if needed
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        return audio_array.astype(np.float32), target_sr
    except Exception:
        # Fallback or silent audio
        return None, target_sr
