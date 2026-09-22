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
        
        # Pillar 1
        p1_res = run_pillar1_inference(pil_im, p1_proc, p1_model, device, model_name=p1_name)
        
        # Pillar 4
        p4_res = run_pillar4_inference(np_im, is_pdf=False)
        
        # Centralized Domain-Aware Consensus Fusion
        fusion = fuse_multi_pillar_verdict(pil_im, np_im, p1_res, p4_res, p5_res)
        is_real = fusion["is_real"]
        override_reason = fusion["override_reason"] or "Multi-Pillar Consensus"
            
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
