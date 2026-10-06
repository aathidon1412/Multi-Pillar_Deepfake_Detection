"""
Batch Evaluation & Benchmarking Script for Antigravity Agent.
Evaluates videos across multiple manipulation types (faces, objects, scenes, animals, inpainting)
and computes Accuracy, Precision, Recall, F1-Score, and per-video forensic breakdown.
"""

import os
import glob
import json
import time
from typing import Dict, List, Any
import numpy as np

from antigravity_agent.core.orchestrator import OrchestratorAgent


def evaluate_dataset(
    dataset_dir: str = "test_videos",
    results_output_dir: str = "benchmark_results",
) -> Dict[str, Any]:
    os.makedirs(results_output_dir, exist_ok=True)
    agent = OrchestratorAgent()

    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "deepfake")

    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    real_files = []
    fake_files = []
    for ext in video_extensions:
        real_files.extend(glob.glob(os.path.join(real_dir, ext)))
        fake_files.extend(glob.glob(os.path.join(fake_dir, ext)))

    print(f"\n=======================================================")
    print(f"📊 Starting Antigravity Forensic Benchmark Evaluation")
    print(f"   Real Videos Found:     {len(real_files)}")
    print(f"   Deepfake Videos Found: {len(fake_files)}")
    print(f"=======================================================\n")

    if not real_files and not fake_files:
        print("[!] No videos found in test_videos/real or test_videos/deepfake.")
        return {}

    eval_records = []
    y_true = []
    y_pred = []
    y_scores = []

    def process_list(file_list: List[str], true_label: int, label_str: str):
        for idx, vpath in enumerate(file_list, 1):
            fname = os.path.basename(vpath)
            print(f"[{label_str.upper()} {idx}/{len(file_list)}] Analyzing: {fname}...")
            start_t = time.time()
            try:
                per_vid_out = os.path.join(results_output_dir, f"{os.path.splitext(fname)[0]}_artifacts")
                res = agent.run(vpath, output_dir=per_vid_out)
                consensus = res["consensus"]
                prob = consensus["deepfake_probability"]
                verdict = consensus["verdict"]
                dur = time.time() - start_t

                # Binary decision: threshold 0.50 for binary evaluation metrics
                pred_label = 1 if prob >= 0.50 else 0

                y_true.append(true_label)
                y_pred.append(pred_label)
                y_scores.append(prob)

                eval_records.append({
                    "filename": fname,
                    "filepath": vpath,
                    "ground_truth": label_str,
                    "verdict": verdict,
                    "deepfake_probability": prob,
                    "confidence_level": consensus["confidence_level"],
                    "latency_sec": round(dur, 2),
                    "spatial_anomaly": res["spatial_results"]["mean_spatial_score"],
                    "temporal_anomaly": res["temporal_results"]["temporal_anomaly_score"],
                    "has_audio": res["audio_results"]["has_audio"],
                    "findings_summary": res["report"]["itemized_findings"],
                })
                print(f"    -> Result: {verdict} (Prob: {prob*100:.1f}%) in {dur:.2f}s")
            except Exception as e:
                print(f"    [!] Error evaluating {fname}: {e}")

    # Process both categories
    process_list(real_files, true_label=0, label_str="real")
    process_list(fake_files, true_label=1, label_str="deepfake")

    # Compute metrics
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    total = len(y_t)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    benchmark_summary = {
        "metrics": {
            "total_evaluated": total,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        },
        "records": eval_records,
    }

    report_path = os.path.join(results_output_dir, "benchmark_summary.json")
    with open(report_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 55)
    print("📈 BENCHMARK RESULTS SUMMARY")
    print(f"   Total Tested:   {total}")
    print(f"   Accuracy:       {accuracy * 100:.2f}%")
    print(f"   Precision:      {precision * 100:.2f}%")
    print(f"   Recall:         {recall * 100:.2f}%")
    print(f"   F1-Score:       {f1 * 100:.2f}%")
    print(f"   Detailed JSON:  {report_path}")
    print("=" * 55 + "\n")

    return benchmark_summary


if __name__ == "__main__":
    evaluate_dataset()
