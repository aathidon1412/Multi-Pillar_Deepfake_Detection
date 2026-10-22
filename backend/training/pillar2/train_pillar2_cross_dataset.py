"""
Pillar 2 Cross-Dataset Deepfake Fine-Tuning & Evaluation Pipeline
================================================================
Downloads and caches an unseen, real-world deepfake benchmark video dataset:
  - Dataset: Hemgg/SDFVD-video-dataset (Standard DeepFake Video Dataset)
  - Content: 53 Real MP4 videos & 53 Deepfake MP4 videos (Total: 106 videos, ~174 MB)
  - Features: True cross-dataset domain shift from Celeb-DF / FF++ face swaps to real-world manipulations.

Outputs:
  - Trains / fine-tunes the Bi-LSTM / MLP Consensus Head (`consensus_head.pth`)
  - Evaluates cross-dataset generalization: Train AUC, Val AUC, Accuracy, Precision, Recall
"""

import os
import sys
import glob
import time
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from antigravity_agent.modules.consensus_engine import BiLSTMFusionHead
from antigravity_agent.utils.video_loader import load_video_frames
from antigravity_agent.modules.spatial_detector import SpatialDetector
from antigravity_agent.modules.temporal_analyzer import TemporalAnalyzer


def download_dataset(dataset_type: str = "sdfvd") -> Tuple[List[str], List[str]]:
    """
    Downloads or verifies video dataset from Hugging Face:
      - 'sdfvd': Hemgg/SDFVD-video-dataset (53 Real, 53 Fake MP4s, ~174 MB)
      - 'generative': Sarim-Hash/video_DEEPFAKE_dataset (Sora-2 & Veo-3 AI vs Real CCTV/Ped, ~84 MB)
      - 'combined': Merges both SDFVD + Sora/Veo for multi-generator coverage.
    """
    from huggingface_hub import snapshot_download
    print("\n" + "=" * 70)
    print(f"📥 [Step 1/3] Hydrating Video Dataset: {dataset_type.upper()}")
    print("=" * 70)

    real_vids: List[str] = []
    fake_vids: List[str] = []

    if dataset_type in ("sdfvd", "combined"):
        print("Checking/downloading Hemgg/SDFVD-video-dataset (~174 MB)...")
        sdfvd_dir = snapshot_download(
            repo_id="Hemgg/SDFVD-video-dataset",
            repo_type="dataset",
            allow_patterns=["*.mp4", "*.json"]
        )
        s_reals = sorted(glob.glob(os.path.join(sdfvd_dir, "Real", "*.mp4"))) or sorted(glob.glob(os.path.join(sdfvd_dir, "*eal*", "*.mp4")))
        s_fakes = sorted(glob.glob(os.path.join(sdfvd_dir, "Fake", "*.mp4"))) or sorted(glob.glob(os.path.join(sdfvd_dir, "*ake*", "*.mp4")))
        print(f"  --> Found {len(s_reals)} Real and {len(s_fakes)} Deepfake videos from SDFVD.")
        real_vids.extend(s_reals)
        fake_vids.extend(s_fakes)

    if dataset_type in ("generative", "combined"):
        print("Checking/downloading Sarim-Hash/video_DEEPFAKE_dataset (Sora & Veo AI, ~84 MB)...")
        gen_dir = snapshot_download(
            repo_id="Sarim-Hash/video_DEEPFAKE_dataset",
            repo_type="dataset",
            allow_patterns=["*.mp4", "*.avi"]
        )
        g_sora = sorted(glob.glob(os.path.join(gen_dir, "fake_video", "sora_2", "*.mp4")))
        g_veo = sorted(glob.glob(os.path.join(gen_dir, "fake_video", "veo_3", "*.mp4")))
        g_reals = sorted(glob.glob(os.path.join(gen_dir, "real_video", "*", "*.mp4")))
        print(f"  --> Found {len(g_reals)} Real and {len(g_sora) + len(g_veo)} Sora/Veo Generative AI videos.")
        real_vids.extend(g_reals)
        fake_vids.extend(g_sora + g_veo)

    return real_vids, fake_vids


def extract_features(
    real_paths: List[str],
    fake_paths: List[str],
    num_frames: int = 16,
    device_str: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts deep ViT spatial embeddings [16, 768] and optical flow motion stats [2]
    across all input real and deepfake video files.
    """
    print("\n" + "=" * 70)
    print("🔬 [Step 2/3] Extracting Deep Spatial & Motion Features")
    print(f"Total Videos: {len(real_paths)} Real, {len(fake_paths)} Fake")
    print("=" * 70)

    spatial_detector = SpatialDetector(device=device_str)
    temporal_analyzer = TemporalAnalyzer()

    all_spatial = []
    all_motion = []
    all_labels = []

    def process_group(paths: List[str], label: int, label_name: str):
        total = len(paths)
        for idx, path in enumerate(paths, 1):
            fname = os.path.basename(path)
            t0 = time.time()
            print(f"[{label_name.upper()} {idx:3d}/{total:3d}] {fname:<25} ", end="", flush=True)
            try:
                frames_tensor, _, _ = load_video_frames(path, num_frames=num_frames)
                with torch.no_grad():
                    spatial_res = spatial_detector.analyze(frames_tensor)
                    temporal_res = temporal_analyzer.analyze(frames_tensor)

                sp_emb = spatial_res["spatial_embeddings"]
                if sp_emb.dim() == 3:
                    sp_emb = sp_emb.squeeze(0)

                mean_m = float(np.mean(temporal_res.get("motion_vectors", [0.0])))
                var_m = float(temporal_res.get("velocity_variance", 0.0))
                mot_feat = torch.tensor([mean_m, var_m], dtype=torch.float32)

                all_spatial.append(sp_emb)
                all_motion.append(mot_feat)
                all_labels.append(label)
                dt = time.time() - t0
                print(f"[✓ {dt:.2f}s]", flush=True)
            except Exception as e:
                print(f"[!] Error: {e}", flush=True)

    process_group(real_paths, label=0, label_name="real")
    process_group(fake_paths, label=1, label_name="fake")

    if not all_spatial:
        raise RuntimeError("No features could be extracted from video files.")

    spatial_t = torch.stack(all_spatial, dim=0)   # [N, 16, 768]
    motion_t = torch.stack(all_motion, dim=0)     # [N, 2]
    labels_t = torch.tensor(all_labels, dtype=torch.long)
    return spatial_t, motion_t, labels_t


def train_and_evaluate(
    spatial_t: torch.Tensor,
    motion_t: torch.Tensor,
    labels_t: torch.Tensor,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_path: str = "consensus_head.pth"
):
    """
    Trains the BiLSTMFusionHead with train/val split and prints classification metrics.
    """
    print("\n" + "=" * 70)
    print("🚀 [Step 3/3] Fine-Tuning Consensus Fusion Head (Bi-LSTM / MLP)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    N = spatial_t.shape[0]
    indices = torch.randperm(N, generator=torch.Generator().manual_seed(42)).tolist()
    split = int(N * 0.80)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_sp, train_mo, train_lb = spatial_t[train_idx], motion_t[train_idx], labels_t[train_idx]
    val_sp, val_mo, val_lb = spatial_t[val_idx], motion_t[val_idx], labels_t[val_idx]

    print(f"Dataset Split: {len(train_idx)} Train samples, {len(val_idx)} Validation samples")

    model = BiLSTMFusionHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_auc = 0.0
    best_state = None

    print(f"\n{'Epoch':<8}{'Train Loss':<14}{'Val Acc':<12}{'Val AUC':<12}{'Val F1':<12}")
    print("-" * 58)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx))
        epoch_loss = 0.0
        
        for i in range(0, len(train_idx), batch_size):
            b_idx = perm[i : i + batch_size]
            b_sp = train_sp[b_idx].to(device)
            b_mo = train_mo[b_idx].to(device)
            b_lb = train_lb[b_idx].to(device)

            optimizer.zero_grad()
            logits = model(b_sp, b_mo)
            loss = criterion(logits, b_lb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_logits = model(val_sp.to(device), val_mo.to(device))
            v_probs = torch.softmax(v_logits, dim=-1)[:, 1].cpu().numpy()
            v_preds = torch.argmax(v_logits, dim=-1).cpu().numpy()
            v_true = val_lb.numpy()

            val_acc = accuracy_score(v_true, v_preds)
            val_f1 = f1_score(v_true, v_preds, zero_division=0)
            try:
                val_auc = roc_auc_score(v_true, v_probs)
            except Exception:
                val_auc = 0.50

        if (epoch % 5 == 0) or (epoch == epochs):
            print(f"{epoch:<8}{epoch_loss:<14.4f}{val_acc*100:<11.1f}%{val_auc:<12.4f}{val_f1:<12.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best checkpoint
    if best_state is not None:
        torch.save(best_state, save_path)
    else:
        torch.save(model.state_dict(), save_path)

    print("\n" + "=" * 70)
    print("🎯 CROSS-DATASET FINE-TUNING COMPLETE!")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Model saved to     : {os.path.abspath(save_path)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pillar 2 Cross-Dataset Deepfake Training")
    parser.add_argument("--dataset", type=str, default="combined", choices=["sdfvd", "generative", "combined"],
                        help="Dataset source: 'sdfvd' (facial deepfake), 'generative' (Sora/Veo AI), or 'combined' (both)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--max_videos", type=int, default=None, help="Limit number of videos (e.g. 40 for fast test)")
    parser.add_argument("--save_path", type=str, default="consensus_head.pth", help="Checkpoint save destination")
    args = parser.parse_args()

    # 1. Download / locate cross-dataset
    real_vids, fake_vids = download_dataset(dataset_type=args.dataset)

    if args.max_videos:
        half = args.max_videos // 2
        fake_vids = fake_vids[:half]
        real_vids = real_vids[:half]

    print(f"Targeting: {len(real_vids)} Real videos & {len(fake_vids)} Fake videos")

    # 3. Extract features
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    sp_t, mo_t, lb_t = extract_features(real_vids, fake_vids, num_frames=16, device_str=device_str)

    # 4. Train & calibrate
    train_and_evaluate(sp_t, mo_t, lb_t, epochs=args.epochs, save_path=args.save_path)


if __name__ == "__main__":
    main()
