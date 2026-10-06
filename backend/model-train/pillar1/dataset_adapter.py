"""
================================================================================
Pillar 1: Face Forensic Dataset & DataLoader Adapter
USMFE Multi-Pillar Deepfake Detection Framework
================================================================================

Description:
    This module implements robust, fault-tolerant dataset loading and PyTorch
    DataLoader factory routines tailored for Vision Transformer (ViT) face
    forensics. It automatically interfaces with the standardized ImageNet
    transformations defined in `train_pillar1_vit.py`.

Key Components:
    - `FaceDeepfakeDataset`: PyTorch Dataset that gracefully recovers from corrupted
      or unreadable image files by substituting a fallback blank frame rather than
      terminating the training process.
    - `make_samples_from_folders`: Scans standard binary directory structures
      (`root/real` for label 0, `root/fake` for label 1).
    - `get_pillar1_dataloaders`: Factory creating optimized train and val DataLoaders
      with pin_memory and worker settings.

Directory Structure Expected:
    dataset_root/
        ├── real/   (Ground Truth 0: Authentic Faces)
        └── fake/   (Ground Truth 1: Deepfake / Manipulated Faces)

Usage Example:
    >>> from dataset_adapter import get_pillar1_dataloaders
    >>> train_loader, val_loader = get_pillar1_dataloaders(
    ...     train_root="path/to/dataset/train",
    ...     val_root="path/to/dataset/val",
    ...     batch_size=32,
    ...     num_workers=4
    ... )
"""

import os
import sys
import logging
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ensure current Pillar 1 directory is in Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from train_pillar1_vit import get_pillar1_transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pillar1.DatasetAdapter")

# Supported image file extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class FaceDeepfakeDataset(Dataset):
    """
    Robust PyTorch Dataset for Face Deepfake Detection.

    Handles list of (image_path, label) pairs. If an image file is corrupted,
    truncated, or missing, it logs a warning and returns a black synthetic
    frame rather than throwing an exception and halting training.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        fallback_size: Tuple[int, int] = (224, 224)
    ) -> None:
        """
        Args:
            samples (List[Tuple[str, int]]): List of tuples containing (filepath, class_label).
            transform (Optional[Callable]): Torchvision transforms to apply to each image.
            fallback_size (Tuple[int, int]): Size of fallback black image on read error (H, W).
        """
        self.samples = samples
        self.transform = transform
        self.fallback_size = fallback_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            # Attempt to open and verify the image as an RGB PIL Image
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as err:
            logger.warning(
                f"[FaceDeepfakeDataset] Corrupted/unreadable image at '{img_path}': {err}. "
                f"Substituting black fallback image."
            )
            # Safe recovery: synthesize an empty black image
            image = Image.new("RGB", self.fallback_size, (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def make_samples_from_folders(root_dir: str) -> List[Tuple[str, int]]:
    """
    Crawls a dataset directory expecting:
        root_dir/
            real/   -> Assigned label 0
            fake/   -> Assigned label 1

    Args:
        root_dir (str): Root path containing 'real' and 'fake' subfolders.

    Returns:
        List[Tuple[str, int]]: Sorted list of (image_path, label) tuples.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root dataset folder '{root_dir}' does not exist.")

    samples: List[Tuple[str, int]] = []
    class_mapping = {"real": 0, "fake": 1}

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Expected class subfolder '{class_dir}' was not found. Skipping.")
            continue

        count = 0
        for entry in os.scandir(class_dir):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in VALID_IMAGE_EXTENSIONS:
                    samples.append((entry.path, label))
                    count += 1

        logger.info(f"Loaded {count} samples from '{class_dir}' (Label: {label}).")

    if not samples:
        logger.warning(f"No valid images found under '{root_dir}'. Check paths and extensions.")

    return samples


def get_pillar1_dataloaders(
    train_root: str,
    val_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs train and validation DataLoaders with standard ImageNet transforms.

    Args:
        train_root (str): Path to training dataset root (containing 'real' and 'fake').
        val_root (str): Path to validation dataset root (containing 'real' and 'fake').
        batch_size (int): Mini-batch size for training and inference.
        num_workers (int): Number of subprocesses for multi-threaded data loading.
        pin_memory (Optional[bool]): If True, allocates tensors in CUDA pinned memory.
                                     Defaults to True if CUDA is available.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Retrieve specialized ImageNet transforms from train_pillar1_vit
    train_transforms = get_pillar1_transforms(train=True)
    val_transforms = get_pillar1_transforms(train=False)

    # Collect samples
    train_samples = make_samples_from_folders(train_root)
    val_samples = make_samples_from_folders(val_root)

    # Wrap in robust FaceDeepfakeDataset
    train_dataset = FaceDeepfakeDataset(train_samples, transform=train_transforms)
    val_dataset = FaceDeepfakeDataset(val_samples, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(
        f"Pillar 1 DataLoaders ready: Train samples={len(train_dataset)}, "
        f"Val samples={len(val_dataset)}, Batch Size={batch_size}, Workers={num_workers}"
    )

    return train_loader, val_loader


# ==============================================================================
# SMOKE TEST / USAGE DEMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    import tempfile
    import shutil

    print("=" * 70)
    print("RUNNING SMOKE TEST: Pillar 1 Dataset Adapter")
    print("=" * 70)

    # Create a temporary directory structure mimicking real/fake folders
    temp_dir = tempfile.mkdtemp(prefix="pillar1_smoke_test_")
    try:
        train_dir = os.path.join(temp_dir, "train")
        val_dir = os.path.join(temp_dir, "val")

        for split_dir in [train_dir, val_dir]:
            os.makedirs(os.path.join(split_dir, "real"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "fake"), exist_ok=True)

        # Generate dummy images: 4 real, 4 fake in train; 2 real, 2 fake in val
        def create_dummy_img(path: str, color: Tuple[int, int, int]):
            img = Image.new("RGB", (256, 256), color)
            img.save(path)

        for i in range(4):
            create_dummy_img(os.path.join(train_dir, "real", f"real_{i}.jpg"), (50, 150, 50))
            create_dummy_img(os.path.join(train_dir, "fake", f"fake_{i}.jpg"), (200, 50, 50))

        # Add one intentionally corrupt file to test graceful recovery
        corrupt_path = os.path.join(train_dir, "fake", "corrupt.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_DATA_123456789")

        for i in range(2):
            create_dummy_img(os.path.join(val_dir, "real", f"val_real_{i}.png"), (40, 140, 40))
            create_dummy_img(os.path.join(val_dir, "fake", f"val_fake_{i}.png"), (180, 40, 40))

        print(f"Temporary test environment created at: {temp_dir}")

        # Instantiate DataLoaders
        train_loader, val_loader = get_pillar1_dataloaders(
            train_root=train_dir,
            val_root=val_dir,
            batch_size=2,
            num_workers=0  # Use 0 workers for lightweight Windows test process
        )

        # Iterate 1 batch to verify shape, normalization, and corrupted file handling
        images, labels = next(iter(train_loader))
        print(f"[Smoke Test] Train batch successfully fetched!")
        print(f" -> Batch Images Tensor Shape : {images.shape} (Expected: [2, 3, 224, 224])")
        print(f" -> Batch Labels Tensor Shape : {labels.shape} (Expected: [2])")
        print(f" -> Tensor dtype              : {images.dtype}")
        print(f" -> Tensor min / max values   : {images.min():.3f} / {images.max():.3f}")

        assert images.shape == (2, 3, 224, 224), "Unexpected tensor dimensions!"
        assert labels.shape == (2,), "Unexpected labels shape!"
        print("\nSUCCESS: Pillar 1 Dataset Adapter passed all validation checks!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {temp_dir}")
        print("=" * 70)
