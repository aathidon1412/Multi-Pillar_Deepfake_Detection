"""
Prepare small temporary dataset for Pillar 1 testing:
- train/real: 300 images (from Pillar 5/dataset/authentic)
- val/real:   100 images (from Pillar 5/dataset/authentic)
- train/fake: 300 images (from Pillar 5/dataset/fake)
- val/fake:   100 images (from Pillar 5/dataset/fake)
"""

import os
import shutil
import random
from pathlib import Path

# Fix seed for reproducibility
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
P5_DATASET_DIR = BASE_DIR / "Pillar 5" / "dataset"
TEMP_DATASET_DIR = BASE_DIR / "Pillar 1" / "temp_dataset"

TRAIN_REAL = TEMP_DATASET_DIR / "train" / "real"
TRAIN_FAKE = TEMP_DATASET_DIR / "train" / "fake"
VAL_REAL   = TEMP_DATASET_DIR / "val" / "real"
VAL_FAKE   = TEMP_DATASET_DIR / "val" / "fake"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prepare_dataset():
    print("=" * 70)
    print("PREPARING TEMPORARY DATASET FOR PILLAR 1")
    print("=" * 70)

    # 1. Create directory structure
    for d in [TRAIN_REAL, TRAIN_FAKE, VAL_REAL, VAL_FAKE]:
        d.mkdir(parents=True, exist_ok=True)

    auth_dir = P5_DATASET_DIR / "authentic"
    fake_dir = P5_DATASET_DIR / "fake"

    if not auth_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Source datasets missing: auth_dir={auth_dir.exists()}, fake_dir={fake_dir.exists()}"
        )

    # 2. Collect authentic images
    print("Collecting authentic images...")
    all_auth = [p for p in auth_dir.rglob("*.*") if p.suffix.lower() in VALID_EXTENSIONS]
    random.shuffle(all_auth)
    print(f"Total authentic images found: {len(all_auth)}")

    auth_train_sample = all_auth[:300]
    auth_val_sample = all_auth[300:400]

    # 3. Collect fake images across generator subfolders
    print("Collecting fake images across generative subfolders...")
    all_fake = [p for p in fake_dir.rglob("*.*") if p.suffix.lower() in VALID_EXTENSIONS]
    random.shuffle(all_fake)
    print(f"Total fake images found: {len(all_fake)}")

    fake_train_sample = all_fake[:300]
    fake_val_sample = all_fake[300:400]

    # 4. Copy files
    print("\nCopying images to temporary train/val directories...")
    def copy_subset(files, dst_dir, label_tag):
        copied = 0
        for src in files:
            dst = dst_dir / f"{label_tag}_{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
        return copied

    c_tr_r = copy_subset(auth_train_sample, TRAIN_REAL, "real")
    c_vl_r = copy_subset(auth_val_sample, VAL_REAL, "real")
    c_tr_f = copy_subset(fake_train_sample, TRAIN_FAKE, "fake")
    c_vl_f = copy_subset(fake_val_sample, VAL_FAKE, "fake")

    print("\nDataset Preparation Complete:")
    print(f"  -> Train Real: {len(list(TRAIN_REAL.iterdir()))} images at {TRAIN_REAL}")
    print(f"  -> Train Fake: {len(list(TRAIN_FAKE.iterdir()))} images at {TRAIN_FAKE}")
    print(f"  -> Val Real  : {len(list(VAL_REAL.iterdir()))} images at {VAL_REAL}")
    print(f"  -> Val Fake  : {len(list(VAL_FAKE.iterdir()))} images at {VAL_FAKE}")
    print("=" * 70)


if __name__ == "__main__":
    prepare_dataset()
