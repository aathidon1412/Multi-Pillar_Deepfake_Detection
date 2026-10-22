"""
================================================================================
Pillar 5: Complete Training & Evaluation Runner
USMFE Multi-Pillar Deepfake Detection Framework
================================================================================

Description:
    This script executes the complete Pillar 5 training and evaluation workflow:
    1. Samples a balanced benchmark of 2,000 images (1,000 authentic + 1,000 fake across generators).
    2. Extracts the 24 physical & lighting geometry features in the EXACT order of `PHYSICS_FEATURE_NAMES`.
    3. Extracts the 1,280-dim multi-scale EfficientNet-B0 embeddings.
    4. Splits into train (80%) and test (20%) sets.
    5. Performs regularized LightGBM feature reduction (1,304 -> 160 features).
    6. Fits the strongly regularized 4-model soft voting ensemble (LightGBM, XGBoost, RF, ExtraTrees).
    7. Evaluates test Accuracy, ROC-AUC, Confusion Matrix, and Classification Report.
    8. Reports feature explainability (number of retained physics vs deep features).
    9. Saves the production model bundle to `Pillar 5/pillar5_ml_model_v2.pkl`.
"""

import os
import sys
import glob
import time
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_schema import PHYSICS_FEATURE_NAMES, FEATURE_INDEX_MAP
from extract_features import extract_physics_vector, get_default_efficientnet
from pillar5_pipeline import train_and_evaluate_pillar5

# Deterministic seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def extract_worker(item: Tuple[str, int]) -> Optional[Tuple[str, int, np.ndarray]]:
    """Worker function for parallel tabular feature extraction."""
    img_path, label = item
    try:
        vec = extract_physics_vector(img_path)
        if vec is not None and vec.shape == (24,):
            return (img_path, label, vec)
    except Exception as err:
        pass
    return None


def collect_balanced_dataset(
    base_dir: str,
    target_auth: int = 1000,
    target_fake: int = 1000
) -> List[Tuple[str, int]]:
    """Collects balanced paths: 1,000 authentic and 1,000 fake across available generator subfolders."""
    dataset_dir = os.path.join(base_dir, "dataset")
    auth_dir = os.path.join(dataset_dir, "authentic")
    fake_dir = os.path.join(dataset_dir, "fake")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # 1. Authentic images
    auth_files = [
        os.path.join(r, f)
        for r, _, files in os.walk(auth_dir)
        for f in files if os.path.splitext(f)[1].lower() in valid_exts
    ]
    random.shuffle(auth_files)
    picked_auth = auth_files[:min(target_auth, len(auth_files))]
    print(f"Collected {len(picked_auth)} Authentic images (out of {len(auth_files)} available).")

    # 2. Fake images balanced across generators
    fake_subdirs = [d for d in glob.glob(os.path.join(fake_dir, "*")) if os.path.isdir(d)]
    picked_fake = []
    if fake_subdirs:
        per_gen = max(1, target_fake // len(fake_subdirs))
        for sd in fake_subdirs:
            gen_files = [
                os.path.join(r, f)
                for r, _, files in os.walk(sd)
                for f in files if os.path.splitext(f)[1].lower() in valid_exts
            ]
            random.shuffle(gen_files)
            selected = gen_files[:per_gen]
            picked_fake.extend(selected)
            print(f" -> Generator '{os.path.basename(sd)}': Sampled {len(selected)} images")

    # Ensure total fake matches target
    random.shuffle(picked_fake)
    picked_fake = picked_fake[:target_fake]
    print(f"Total Fake images sampled: {len(picked_fake)}")

    records = [(p, 0) for p in picked_auth] + [(p, 1) for p in picked_fake]
    random.shuffle(records)
    print(f"\nFinal Combined Balanced Dataset: {len(records)} images ({len(picked_auth)} Authentic [0], {len(picked_fake)} Fake [1]).")
    return records


def main():
    print("=" * 80)
    print("LAUNCHING PILLAR 5 TRAINING & EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Collect dataset
    records = collect_balanced_dataset(current_dir, target_auth=1000, target_fake=1000)

    # 2. Extract 24-dim Physics Features in parallel
    workers = min(8, os.cpu_count() or 4)
    print(f"\n[Step 1/3] Extracting 24 Physics Features in parallel using {workers} CPU workers...")
    print(f"Schema: Features strictly locked to PHYSICS_FEATURE_NAMES ({len(PHYSICS_FEATURE_NAMES)} features).")
    
    t0 = time.time()
    valid_paths = []
    labels = []
    physics_rows = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_worker, item) for item in records]
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res is not None:
                p, lbl, vec = res
                valid_paths.append(p)
                labels.append(lbl)
                physics_rows.append(vec)
            if done % 250 == 0 or done == len(records):
                elapsed = time.time() - t0
                speed = done / max(1.0, elapsed)
                print(f" -> Physics features: {done}/{len(records)} images processed ({speed:.1f} img/s)...", flush=True)

    X_physics = np.vstack(physics_rows).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Physics feature extraction completed in {time.time() - t0:.1f}s. Shape: {X_physics.shape}")

    # 3. Extract 1,280-dim EfficientNet-B0 Embeddings
    print(f"\n[Step 2/3] Extracting 1,280-dim EfficientNet-B0 Deep Embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.to(device).eval()

    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = 32
    embeddings = []
    t1 = time.time()

    with torch.no_grad():
        for i in range(0, len(valid_paths), batch_size):
            batch_paths = valid_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    with Image.open(p) as img:
                        tensors.append(prep(img.convert("RGB")))
                except Exception:
                    tensors.append(torch.zeros(3, 224, 224))
            batch_t = torch.stack(tensors).to(device)
            out = model(batch_t).cpu().numpy()
            embeddings.append(out)

            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(valid_paths):
                processed = min(i + batch_size, len(valid_paths))
                print(f" -> Deep features: {processed}/{len(valid_paths)} images processed...", flush=True)

    X_effnet = np.vstack(embeddings).astype(np.float32)
    print(f"Deep embedding extraction completed in {time.time() - t1:.1f}s. Shape: {X_effnet.shape}")

    # 4. Train/Test Split (80% Train, 20% Test)
    print(f"\n[Step 3/3] Partitioning into Train (80%) and Test (20%) splits with stratification...")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

    X_train_p, X_test_p = X_physics[train_idx], X_physics[test_idx]
    X_train_e, X_test_e = X_effnet[train_idx], X_effnet[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
    print(f"X_train_physics: {X_train_p.shape} | X_test_physics: {X_test_p.shape}")
    print(f"X_train_effnet : {X_train_e.shape} | X_test_effnet : {X_test_e.shape}")

    # 5. Execute train_and_evaluate_pillar5 with LightGBM reduction (160 features)
    results = train_and_evaluate_pillar5(
        X_train_physics=X_train_p,
        X_train_efficientnet=X_train_e,
        y_train=y_train,
        X_test_physics=X_test_p,
        X_test_efficientnet=X_test_e,
        y_test=y_test,
        reduction_method="lgbm",
        n_features=160
    )

    # 6. Save Model Bundle
    save_bundle_path = os.path.join(current_dir, "pillar5_ml_model_v2.pkl")
    bundle = {
        "type": "regularized_hybrid_ensemble",
        "backbone": "efficientnet_b0",
        "feature_cols": PHYSICS_FEATURE_NAMES,
        "scaler": results["scaler"],
        "reducer": results["reducer"],
        "selected_indices": results["selected_indices"],
        "classifier": results["ensemble"],
        "metrics": results["metrics"],
        "total_trained_samples": len(valid_paths),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(save_bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[Saved] Updated Pillar 5 model bundle saved to: {save_bundle_path}")


if __name__ == "__main__":
    main()
