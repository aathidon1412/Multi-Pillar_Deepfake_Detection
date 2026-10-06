import os
import sys
import numpy as np
import cv2
from pathlib import Path
import scipy.io.wavfile as wavfile
import subprocess

# Ensure backend can be imported
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))

from backend.config import UPLOADS_DIR, FFMPEG_PATH
from backend.services.video_processor import inspect_video
from backend.services.frame_extractor import extract_sampled_frames
from backend.services.face_detector import detect_faces_in_frames
from backend.services.visual_analyzer import run_visual_analysis
from backend.services.temporal_analyzer import run_temporal_analysis
from backend.services.audio_analyzer import extract_audio_track, run_audio_analysis
from backend.services.lip_sync_analyzer import run_lip_sync_analysis
from backend.services.metadata_analyzer import run_metadata_analysis
from backend.services.classifier import run_feature_fusion_and_classification
from backend.services.explainability import extract_and_annotate_suspicious_frames
from backend.services.cleanup import cleanup_temporary_frames
from backend.services.json_storage import save_result, load_result, list_results

def create_synthetic_test_video(output_mp4: str, duration_sec: int = 3, fps: int = 30):
    """
    Creates a valid synthetic MP4 video with a simulated face-like oval and audio track.
    """
    width, height = 640, 480
    temp_raw_avi = output_mp4.replace(".mp4", "_raw.avi")
    temp_wav = output_mp4.replace(".mp4", "_audio.wav")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_raw_avi, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create gradient background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        
        # Draw face-like oval with slight movement
        cx = int(width / 2 + 20 * np.sin(i * 0.1))
        cy = int(height / 2 + 10 * np.cos(i * 0.1))
        
        # Skin tone ellipse
        cv2.ellipse(frame, (cx, cy), (70, 90), 0, 0, 360, (180, 210, 240), -1)
        # Eyes
        cv2.circle(frame, (cx - 25, cy - 20), 8, (60, 40, 20), -1)
        cv2.circle(frame, (cx + 25, cy - 20), 8, (60, 40, 20), -1)
        # Mouth moving
        mouth_h = int(6 + 6 * abs(np.sin(i * 0.2)))
        cv2.ellipse(frame, (cx, cy + 40), (20, mouth_h), 0, 0, 360, (50, 50, 180), -1)
        
        out.write(frame)
    out.release()

    # Generate synthetic 440Hz sine audio track
    sample_rate = 16000
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio_signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(temp_wav, sample_rate, audio_signal)

    # Merge into standard MP4 via ffmpeg
    cmd = [
        str(FFMPEG_PATH),
        "-y",
        "-i", temp_raw_avi,
        "-i", temp_wav,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean temps
    if os.path.exists(temp_raw_avi):
        os.remove(temp_raw_avi)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

def test_full_pipeline():
    test_id = "VID_TEST_001"
    test_video_path = str(UPLOADS_DIR / f"{test_id}.mp4")

    print("[TEST] 1. Generating synthetic test video...")
    create_synthetic_test_video(test_video_path, duration_sec=3, fps=30)
    assert os.path.exists(test_video_path), "Test video was not created!"

    print("[TEST] 2. Inspecting video metadata...")
    details = inspect_video(test_video_path)
    print("   Details:", details)
    assert details["duration_seconds"] > 0
    assert details["fps"] == 30.0

    print("[TEST] 3. Extracting sampled frames...")
    frames = extract_sampled_frames(test_video_path, test_id)
    print(f"   Extracted {len(frames)} frames.")
    assert len(frames) > 0

    print("[TEST] 4. Detecting faces...")
    face_summary = detect_faces_in_frames(frames)
    print("   Face Summary:", face_summary)

    print("[TEST] 5. Running visual analysis...")
    v_res = run_visual_analysis(frames)
    print("   Visual Analysis:", v_res["score"], v_res["status"], v_res["anomalies"])
    assert 0.0 <= v_res["score"] <= 1.0

    print("[TEST] 6. Running temporal analysis...")
    t_res = run_temporal_analysis(frames)
    print("   Temporal Analysis:", t_res["score"], t_res["status"])
    assert 0.0 <= t_res["score"] <= 1.0

    print("[TEST] 7. Audio extraction & analysis...")
    wav_path = extract_audio_track(test_video_path, test_id)
    a_res = run_audio_analysis(wav_path)
    print("   Audio Analysis:", a_res["available"], a_res["score"], a_res["status"])

    print("[TEST] 8. Lip-sync analysis...")
    ls_res = run_lip_sync_analysis(frames, wav_path)
    print("   Lip-Sync Analysis:", ls_res["available"], ls_res.get("score"))

    print("[TEST] 9. Metadata analysis...")
    m_res = run_metadata_analysis(test_video_path, details)
    print("   Metadata Analysis:", m_res["codec"], m_res["metadata_status"])

    print("[TEST] 10. Classification & Feature Fusion...")
    classification = run_feature_fusion_and_classification(v_res, t_res, a_res, ls_res, m_res)
    print("   Classification:", classification)
    assert classification["prediction"] in ["REAL", "AI_GENERATED", "FORGED"]

    print("[TEST] 11. Extracting suspicious frames...")
    s_frames = extract_and_annotate_suspicious_frames(frames, test_id)
    print(f"   Suspicious frames extracted: {len(s_frames)}")
    assert len(s_frames) > 0

    print("[TEST] 12. Assembling JSON report and saving...")
    report = {
        "video_id": test_id,
        "file": {
            "original_filename": "test_synthetic.mp4",
            "stored_filename": f"{test_id}.mp4",
            "format": "mp4",
            "size_mb": round(os.path.getsize(test_video_path) / (1024 * 1024), 2)
        },
        "video_details": details,
        "metadata": m_res,
        "analysis": {
            "face_detection": face_summary,
            "visual": v_res,
            "temporal": t_res,
            "audio": a_res,
            "lip_sync": ls_res
        },
        "classification": classification,
        "suspicious_frames": s_frames,
        "processing": {
            "status": "completed",
            "processed_at": "2026-09-18T13:30:00",
            "processing_time_seconds": 2.1
        }
    }
    saved_path = save_result(test_id, report)
    print(f"   Saved JSON to: {saved_path}")

    print("[TEST] 13. Verifying JSON retrieval and history...")
    loaded = load_result(test_id)
    assert loaded is not None
    assert loaded["video_id"] == test_id
    history = list_results()
    assert any(item["video_id"] == test_id for item in history)
    print(f"   History contains {len(history)} items.")

    print("[TEST] 14. Cleaning temporary frames...")
    cleanup_temporary_frames(test_id)

    print("\n[SUCCESS] Entire Multi-Pillar Pipeline verified successfully!")

if __name__ == "__main__":
    test_full_pipeline()
