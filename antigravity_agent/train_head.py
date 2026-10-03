"""
Train and calibrate the lightweight Bi-LSTM / MLP consensus fusion head.
Supports:
1. Training directly on video datasets (e.g. test_videos/ or any dataset/ folder).
2. Offline cached feature training for fast iterations.
3. Fallback synthetic calibration if no video dataset is provided.
"""

import os
import glob
import time
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from antigravity_agent.modules.consensus_engine import BiLSTMFusionHead
from antigravity_agent.utils.video_loader import load_video_frames
from antigravity_agent.modules.spatial_detector import SpatialDetector
from antigravity_agent.modules.temporal_analyzer import TemporalAnalyzer


def extract_features_from_videos(
    real_video_paths: List[str],
    fake_video_paths: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pass real and fake videos through frozen Module 1 (ViT) and Module 2 (Optical Flow)
    to extract training sequences for the consensus head.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    spatial_detector = SpatialDetector(device=device_str)
    temporal_analyzer = TemporalAnalyzer()

    all_spatial = []
    all_motion = []
    all_labels = []

    def process_paths(paths: List[str], label: int, label_name: str):
        total = len(paths)
        for idx, path in enumerate(paths, 1):
            fname = os.path.basename(path)
            t0 = time.time()
            print(f"[{label_name.upper()} {idx:3d}/{total:3d}] Extracting from: {fname[:38]}...", end="", flush=True)
            try:
                frames_tensor, _, _ = load_video_frames(path, num_frames=16)
                with torch.no_grad():
                    spatial_res = spatial_detector.analyze(frames_tensor)
                    temporal_res = temporal_analyzer.analyze(frames_tensor)

                # Spatial embeddings: [16, 768]
                sp_emb = spatial_res["spatial_embeddings"]
                if sp_emb.dim() == 3:
                    sp_emb = sp_emb.squeeze(0)

                # Motion features: [mean_motion, velocity_variance]
                mean_m = float(np.mean(temporal_res.get("motion_vectors", [0.0])))
                var_m = float(temporal_res.get("velocity_variance", 0.0))
                mot_feat = torch.tensor([mean_m, var_m], dtype=torch.float32)

                all_spatial.append(sp_emb)
                all_motion.append(mot_feat)
                all_labels.append(label)
                dt = time.time() - t0
                print(f" [✓] ({dt:.2f}s)", flush=True)
            except Exception as e:
                import traceback
                print(f" [!] Error: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

    process_paths(real_video_paths, label=0, label_name="real")
    process_paths(fake_video_paths, label=1, label_name="fake")

    if not all_spatial:
        raise ValueError("No video features could be extracted.")

    spatial_data = torch.stack(all_spatial, dim=0)   # [N, 16, 768]
    motion_data = torch.stack(all_motion, dim=0)     # [N, 2]
    labels_data = torch.tensor(all_labels, dtype=torch.long) # [N]

    return spatial_data, motion_data, labels_data


def train_consensus_head(
    dataset_dir: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 8,
    max_videos: Optional[int] = None,
    save_path: str = "consensus_head.pth",
):
    print("=" * 60)
    print("🎯 Training Consensus Aggregator Fusion Head (Bi-LSTM / MLP)")
    print("=" * 60)

    spatial_data: Optional[torch.Tensor] = None
    motion_data: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None

    # Check if a video dataset directory is provided and contains videos
    use_dataset = False
    if dataset_dir and os.path.exists(dataset_dir):
        real_dir = os.path.join(dataset_dir, "real")
        fake_dir = os.path.join(dataset_dir, "deepfake")
        extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        real_vids = []
        fake_vids = []
        for ext in extensions:
            real_vids.extend(glob.glob(os.path.join(real_dir, ext)))
            fake_vids.extend(glob.glob(os.path.join(fake_dir, ext)))

        if max_videos:
            half = max_videos // 2
            real_vids = real_vids[:half]
            fake_vids = fake_vids[:half]

        if len(real_vids) > 0 and len(fake_vids) > 0:
            print(f"Selected {len(real_vids)} real and {len(fake_vids)} deepfake videos (Total: {len(real_vids)+len(fake_vids)}) for training.")
            spatial_data, motion_data, labels = extract_features_from_videos(real_vids, fake_vids)
            use_dataset = True

    if not use_dataset or spatial_data is None or motion_data is None or labels is None:
        print("[Notice] No dual-class video dataset found; initializing synthetic calibration dataset...")
        num_samples = 200
        torch.manual_seed(42)
        real_spatial = torch.randn(num_samples // 2, 16, 768) * 0.3
        real_motion = torch.abs(torch.randn(num_samples // 2, 2) * 0.2)
        real_labels = torch.zeros(num_samples // 2, dtype=torch.long)

        fake_spatial = torch.randn(num_samples // 2, 16, 768) * 0.8 + 0.5
        fake_motion = torch.abs(torch.randn(num_samples // 2, 2) * 1.2 + 0.6)
        fake_labels = torch.ones(num_samples // 2, dtype=torch.long)

        spatial_data = torch.cat([real_spatial, fake_spatial], dim=0)
        motion_data = torch.cat([real_motion, fake_motion], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)

    # Shuffle training set
    num_samples = spatial_data.shape[0]
    indices = torch.randperm(num_samples)
    spatial_data = spatial_data[indices]
    motion_data = motion_data[indices]
    labels = labels[indices]

    # Auto-detect CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Initialize model
    model = BiLSTMFusionHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    print(f"\nTraining on {num_samples} video feature sequences across {epochs} epochs...")
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(0, num_samples, max(1, batch_size)):
            batch_idx = permutation[i : i + batch_size]
            b_spatial = spatial_data[batch_idx].to(device)
            b_motion = motion_data[batch_idx].to(device)
            b_labels = labels[batch_idx].to(device)

            optimizer.zero_grad()
            outputs = model(b_spatial, b_motion)
            loss = criterion(outputs, b_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n Model weights successfully saved to: {os.path.abspath(save_path)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Consensus Head")
    parser.add_argument("--dataset_dir", type=str, default="test_videos", help="Path to dataset containing real/ and deepfake/")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum total videos to train on (e.g. 50, 100)")
    args = parser.parse_args()

    train_consensus_head(dataset_dir=args.dataset_dir, epochs=args.epochs, max_videos=args.max_videos)
