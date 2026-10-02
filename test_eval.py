"""
================================================================================
Universal Multi-Pillar Test Suite Evaluator
USMFE Deepfake & Document Forensics Framework
================================================================================
Evaluates all samples in the `testing/` folder using the consensus and domain
routing logic implemented in `app_streamlit.py`.
"""

import glob
import os
import cv2
import numpy as np
from PIL import Image
from core import (
    load_pillar1_vit,
    load_pillar5_ml_bundle,
    run_pillar1_inference,
    run_pillar5_inference,
    run_pillar4_inference,
    fuse_multi_pillar_verdict
)

def evaluate_testing_suite():
    print("=" * 80)
    print("EVALUATING MULTI-PILLAR FORENSICS ON TESTING/ BENCHMARK SUITE")
    print("=" * 80)
    
    p1_proc, p1_model, device, p1_name = load_pillar1_vit()
    p5_bundle, p5_name = load_pillar5_ml_bundle()

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    files = sorted([f for f in glob.glob('testing/*.*') if os.path.splitext(f)[1].lower() in valid_exts])
    img_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    files = [f for f in sorted(glob.glob('testing/*.*')) if os.path.splitext(f)[1].lower() in img_exts]
    print(f"Found {len(files)} benchmark test images.\n")

    results = []
    p1_correct_count = 0
    p5_correct_count = 0
    for f in files:
        fname = os.path.basename(f)
        is_true_fake = fname.lower().startswith('fake') or fname.lower().startswith('ai_')
        
        pil_im = Image.open(f).convert('RGB')
        np_im = np.array(pil_im)
        
        # Pillar 5
        p5_res = run_pillar5_inference(np_im, pil_img=pil_im, p5_bundle=p5_bundle)
        
        # Pillar 1
        p1_res = run_pillar1_inference(pil_im, p1_proc, p1_model, device, model_name=str(p1_name or "ViT"))
        
        # Pillar 4
        p4_res = run_pillar4_inference(np_im, is_pdf=False)
        
        # Centralized Domain-Aware Consensus Fusion
        fusion = fuse_multi_pillar_verdict(pil_im, np_im, p1_res, p4_res, p5_res)
        is_real = fusion["is_real"]
        override_reason = fusion["override_reason"] or "Multi-Pillar Consensus"
            
        actual = 'FAKE' if is_true_fake else 'REAL'
        predicted = 'REAL' if is_real else 'FAKE'
        correct = (actual == predicted)
        
        p1_pred = 'REAL' if p1_res.get('is_real') else 'FAKE'
        p5_pred = 'REAL' if p5_res.get('is_real') else 'FAKE'
        if p1_pred == actual:
            p1_correct_count += 1
        if p5_pred == actual:
            p5_correct_count += 1
        
        results.append({
            'file': fname,
            'actual': actual,
            'predicted': predicted,
            'p1_pred': p1_pred,
            'p1_conf': p1_res.get('confidence', 0),
            'p5_pred': p5_pred,
            'p5_conf': p5_res.get('confidence', 0),
            'correct': correct,
            'override': override_reason
        })

    print(f"{'FILE':18s} | {'ACTUAL':6s} | {'P1':6s} | {'P5':6s} | {'FUSED':6s} | {'MATCH':5s} | {'REASON / OVERRIDE'}")
    print('-' * 95)
    fails = 0
    for r in results:
        match_str = 'PASS' if r['correct'] else 'FAIL'
        if not r['correct']:
            fails += 1
        print(f"{r['file']:18s} | {r['actual']:6s} | {r['p1_pred']:6s} | {r['p5_pred']:6s} | {r['predicted']:6s} | {match_str:5s} | {r['override']}")

    acc = (len(results) - fails) / len(results) * 100
    p1_acc = (p1_correct_count / len(results)) * 100
    p5_acc = (p5_correct_count / len(results)) * 100
    print('-' * 95)
    print(f"Pillar 1 Standalone Accuracy : {p1_acc:.2f}% ({p1_correct_count}/{len(results)})")
    print(f"Pillar 5 Standalone Accuracy : {p5_acc:.2f}% ({p5_correct_count}/{len(results)})")
    print(f"Fused Multi-Pillar Accuracy  : {acc:.2f}% ({len(results) - fails}/{len(results)})\n")
    print('-' * 80)
    print(f"Total: {len(results)} | Passed: {len(results) - fails} | Failed: {fails} | Accuracy: {acc:.2f}%\n")


def evaluate_pillar2_video_suite():
    print("=" * 125)
    print("EVALUATING PILLAR 2 (VIDEO AUTHENTICITY & BIOLOGICAL rPPG) BENCHMARK SUITE")
    print("=" * 125)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    p2_dir = os.path.join(base_dir, "Pillar 2", "video-authenticity-detector")

    test_videos = [
        (os.path.join(base_dir, "Pillar 2", "real_face.mp4"), "REAL"),
        (os.path.join(base_dir, "Pillar 2", "fake_avatar.mp4"), "AI_GENERATED"),
        (os.path.join(p2_dir, "storage", "uploads", "VID_TEST_001.mp4"), "AI_GENERATED"),
        (os.path.join(p2_dir, "storage", "uploads", "VID_44B3AA.mp4"), "AI_GENERATED"),
        (os.path.join(p2_dir, "storage", "uploads", "VID_A304CF.mp4"), "REAL"),
        (os.path.join(p2_dir, "storage", "uploads", "VID_E5CC8A.mp4"), "REAL"),
        (os.path.join(p2_dir, "storage", "uploads", "VID_F78EF3.mp4"), "REAL"),
    ]

    try:
        from backend.services.video_processor import inspect_video
        from backend.services.frame_extractor import extract_sampled_frames
        from backend.services.face_detector import detect_faces_in_frames
        from backend.services.visual_analyzer import run_visual_analysis
        from backend.services.temporal_analyzer import run_temporal_analysis
        from backend.services.audio_analyzer import extract_audio_track, run_audio_analysis
        from backend.services.lip_sync_analyzer import run_lip_sync_analysis
        from backend.services.metadata_analyzer import run_metadata_analysis
        from backend.services.classifier import run_feature_fusion_and_classification
        from backend.services.rppg_analyzer import run_rppg_analysis
        from backend.services.cleanup import cleanup_temporary_frames
    except ImportError:
        from core import (  # type: ignore
            inspect_video,
            extract_sampled_frames,
            detect_faces_in_frames,
            run_visual_analysis,
            run_temporal_analysis,
            extract_audio_track,
            run_audio_analysis,
            run_lip_sync_analysis,
            run_metadata_analysis,
            run_rppg_analysis,
            run_feature_fusion_and_classification,
            cleanup_temporary_frames
        )

    if inspect_video is None or run_feature_fusion_and_classification is None:
        print("[!] Video authenticity detector backend is unavailable. Skipping video benchmark.")
        return

    correct = 0
    total = 0
    print(f"{'VIDEO FILE':22s} | {'ACTUAL':12s} | {'PREDICTED':12s} | {'CONF':6s} | {'STATUS':6s} | {'VIS':5s} | {'rPPG':12s} | {'REASON / OVERRIDE'}")
    print("-" * 125)

    for vpath, expected in test_videos:
        if not os.path.exists(vpath):
            continue
        vid_id = "EVAL_P2_" + os.path.basename(vpath).split(".")[0]
        details = inspect_video(vpath)
        meta = run_metadata_analysis(vpath, details)
        frames = extract_sampled_frames(vpath, vid_id)
        detect_faces_in_frames(frames)
        vis = run_visual_analysis(frames)
        temp = run_temporal_analysis(frames)
        wav = extract_audio_track(vpath, vid_id)
        aud = run_audio_analysis(wav)
        ls = run_lip_sync_analysis(frames, wav)
        rppg = run_rppg_analysis(vpath) if run_rppg_analysis else {}
        clf = run_feature_fusion_and_classification(vis, temp, aud, ls, meta, rppg)
        cleanup_temporary_frames(vid_id)

        pred = clf["prediction"]
        conf = clf["confidence"]
        match = (pred == expected)
        if match:
            correct += 1
        total += 1

        match_str = "PASS" if match else "FAIL"
        r_str = f"{rppg.get('verdict')} ({rppg.get('bpm','N/A')}bpm)" if rppg.get("verdict") and rppg.get("verdict") != "INCONCLUSIVE" else "INCONCLUSIVE"
        reason = clf.get("override_reason") or "Consensus"
        print(f"{os.path.basename(vpath):22s} | {expected:12s} | {pred:12s} | {conf*100:5.1f}% | {match_str:6s} | {vis['score']:5.2f} | {r_str:12s} | {reason}")

    acc = (correct / total * 100) if total > 0 else 0
    print("-" * 125)
    print(f"Pillar 2 Standalone Accuracy : {acc:.2f}% ({correct}/{total} Passed)\n")


if __name__ == '__main__':
    import sys
    if "--images" in sys.argv:
        evaluate_testing_suite()
    elif "--all" in sys.argv:
        evaluate_pillar2_video_suite()
        evaluate_testing_suite()
    else:
        evaluate_pillar2_video_suite()

