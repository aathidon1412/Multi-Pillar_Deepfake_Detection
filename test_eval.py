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
from app_streamlit import (
    load_pillar1_vit,
    load_pillar5_ml_bundle,
    run_pillar1_inference,
    run_pillar5_inference,
    run_pillar4_inference
)

def evaluate_testing_suite():
    print("=" * 80)
    print("EVALUATING MULTI-PILLAR FORENSICS ON TESTING/ BENCHMARK SUITE")
    print("=" * 80)
    
    p1_proc, p1_model, device, p1_name = load_pillar1_vit()
    p5_bundle, p5_name = load_pillar5_ml_bundle()

    files = sorted(glob.glob('testing/*.*'))
    print(f"Found {len(files)} benchmark test images.\n")

    results = []
    for f in files:
        fname = os.path.basename(f)
        is_true_fake = fname.lower().startswith('fake')
        
        pil_im = Image.open(f).convert('RGB')
        np_im = np.array(pil_im)
        
        # Pillar 5
        p5_res = run_pillar5_inference(np_im, pil_img=pil_im, p5_bundle=p5_bundle)
        p5_real = p5_res['real_probability']
        p5_fake = p5_res['fake_probability']
        
        # Pillar 1
        p1_res = run_pillar1_inference(pil_im, p1_proc, p1_model, device, model_name=p1_name)
        p1_real = p1_res['real_probability'] if p1_res.get('available') else p5_real
        p1_fake = 1.0 - p1_real
        
        # Pillar 4
        p4_res = run_pillar4_inference(np_im, is_pdf=False)
        
        # Domain Telemetry & Sensor EXIF
        exif = pil_im.getexif()
        has_cam_exif = bool(exif.get(0x010f) or exif.get(0x0110) or exif.get(0x0131))
        gray = cv2.cvtColor(np_im, cv2.COLOR_RGB2GRAY)
        white_ratio = float(np.mean(gray > 220))
        is_paper_doc = (white_ratio > 0.40) or (p4_res.get("applicable", False) and p4_res.get("digits_count", 0) >= 5)
        
        override_reason = None
        if has_cam_exif:
            is_real = True
            override_reason = "Camera Hardware Sensor EXIF Signature Confirmed"
        elif is_paper_doc and p5_fake < 0.90:
            is_real = True
            override_reason = "Physical Document / Scanned Paper Domain Gating (Pillar 4)"
        elif p5_res.get("is_prnu_anomaly") and p5_fake >= 0.60:
            is_real = False
            override_reason = "Pillar 5 PRNU Steganalysis Override (Synthetic Noise Anomaly)"
        elif p1_res.get("available", False) and p1_fake >= 0.70:
            is_real = False
            override_reason = "Pillar 1 ViT Neural Override (Facial Spectral Anomaly)"
        else:
            fused_fake = 0.55 * p5_fake + 0.45 * p1_fake
            is_real = (fused_fake < 0.45)
            override_reason = "Multi-Pillar Calibrated Weighted Consensus"
            
        actual = 'FAKE' if is_true_fake else 'REAL'
        predicted = 'REAL' if is_real else 'FAKE'
        correct = (actual == predicted)
        
        results.append({
            'file': fname,
            'actual': actual,
            'predicted': predicted,
            'correct': correct,
            'override': override_reason
        })

    print(f"{'FILE':15s} | {'ACTUAL':6s} | {'PRED':6s} | {'MATCH':5s} | {'REASON / OVERRIDE'}")
    print('-' * 80)
    fails = 0
    for r in results:
        match_str = 'PASS' if r['correct'] else 'FAIL'
        if not r['correct']:
            fails += 1
        print(f"{r['file']:15s} | {r['actual']:6s} | {r['predicted']:6s} | {match_str:5s} | {r['override']}")

    acc = (len(results) - fails) / len(results) * 100
    print('-' * 80)
    print(f"Total: {len(results)} | Passed: {len(results) - fails} | Failed: {fails} | Accuracy: {acc:.2f}%\n")

if __name__ == '__main__':
    evaluate_testing_suite()
