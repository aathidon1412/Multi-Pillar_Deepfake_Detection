import os
import subprocess
import json
import cv2
from typing import Dict, Any
from backend.config import FFMPEG_PATH

def run_metadata_analysis(video_path: str, video_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts container and stream metadata using OpenCV and FFmpeg probe.
    Classifies metadata status as normal, suspicious, or inconclusive.
    """
    codec = "unknown"
    encoder = "unknown"
    creation_time = "unknown"
    metadata_status = "inconclusive"
    forensic_flags = []

    # 1. OpenCV basic stream probe
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        if fourcc_str and fourcc_str.isprintable():
            codec = fourcc_str.lower()
        cap.release()

    # 2. FFmpeg detailed probe
    cmd = [
        str(FFMPEG_PATH),
        "-i", video_path,
        "-hide_banner"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        stderr_output = proc.stderr

        for line in stderr_output.splitlines():
            line_str = line.strip()
            if "encoder" in line_str.lower():
                parts = line_str.split(":", 1)
                if len(parts) > 1:
                    encoder = parts[1].strip()
            if "Stream #0:" in line_str and "Video:" in line_str:
                parts = line_str.split("Video:", 1)
                if len(parts) > 1:
                    codec_part = parts[1].split(",")[0].strip()
                    if codec_part:
                        codec = codec_part
            if "creation_time" in line_str.lower():
                parts = line_str.split(":", 1)
                if len(parts) > 1:
                    creation_time = parts[1].strip()
    except Exception as e:
        print(f"[METADATA PROBE] FFmpeg probe warning: {e}")

    # Heuristic checks
    fps = video_details.get("fps", 30.0)
    # Check for atypical framerates common in AI video generation models (e.g. exactly 8fps, 12fps, 16fps)
    if fps in [8.0, 12.0, 14.0, 16.0]:
        forensic_flags.append(f"Atypical generation framerate detected ({fps} FPS)")
        metadata_status = "suspicious"
    elif encoder != "unknown" and any(k in encoder.lower() for k in ["ffmpeg", "lavf", "handbrake"]):
        forensic_flags.append("Re-encoding signature detected")
        metadata_status = "suspicious"
    elif codec != "unknown":
        metadata_status = "normal"
    else:
        metadata_status = "inconclusive"

    return {
        "codec": codec,
        "encoder": encoder,
        "creation_time": creation_time,
        "metadata_status": metadata_status,
        "forensic_flags": forensic_flags
    }
