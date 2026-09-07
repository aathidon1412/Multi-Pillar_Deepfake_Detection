"""
Pillar 1: Vision Transformer (ViT) Face Forensics Training Pipeline
USMFE Multi-Pillar Deepfake Detection System

This module provides:
1. get_pillar1_transforms(train: bool): Standardized ImageNet transform pipelines with data augmentations.
2. build_pillar1_vit(): Pretrained ViT loader with early-layer freezing (fine-tuning top-k blocks + classifier).
3. get_vit_optimizer_groups(): Differential learning rate optimizer setup (AdamW).
4. train_pillar1_pipeline(): Complete training loop with mixed precision (AMP), Cosine Annealing Warm Restarts,
   label smoothing, and Early Stopping based on validation AUC.
"""

import os
import copy
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


# ==============================================================================
# 1. TRANSFORMS PIPELINE
# ==============================================================================

# Official ImageNet statistics - matching the pretraining distribution of google/vit-base-patch16-224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_pillar1_transforms(train: bool = True) -> transforms.Compose:
    """
    Constructs the image transform pipeline for Pillar 1 (ViT Face Forensics).

    Args:
        train (bool): If True, returns augmentations designed to prevent overfitting on facial artifacts.
                      If False, returns deterministic validation/test preprocessing.

    Returns:
        transforms.Compose: Composed PyTorch image transformations.
    """
    if train:
        return transforms.Compose([
            # 1. Resize slightly larger to allow scale-invariant random cropping
            transforms.Resize((256, 256)),
            # 2. Random crop to 224x224 (native ViT patch grid 14x14 of 16x16 patches)
            transforms.RandomCrop((224, 224)),
            # 3. Horizontal flip - facial symmetries preserve deepfake forensic traces
            transforms.RandomHorizontalFlip(p=0.5),
            # 4. Color jitter introduces robustness against variable lighting / color grading
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.05
            ),
            # 5. Gaussian blur simulates compression/optical softness to prevent reliance on single high-frequency cues
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
            ], p=0.2),
            # 6. Convert PIL Image to FloatTensor [0.0, 1.0]
            transforms.ToTensor(),
            # 7. Normalize using standard ImageNet mean and std
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            # Deterministic resize directly to 224x224 for validation/testing
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ==============================================================================
# 2. MODEL ARCHITECTURE & DIFFERENTIAL LR SETUP
# ==============================================================================

def build_pillar1_vit(
    num_classes: int = 2,
    freeze_early_layers: bool = True,
    trainable_blocks: int = 4,
    pretrained_model_name_or_path: str = "google/vit-base-patch16-224"
) -> ViTForImageClassification:
    """
    Loads `google/vit-base-patch16-224` (or local checkpoint) and configures layer freezing strategy.

    ViT-Base consists of:
    - Patch + Position Embeddings
    - 12 Transformer Encoder Blocks (0 through 11)
    - Classification Head (MLP / Linear)

    Freezing early layers preserves general visual features (edges, textures)
    while adapting the final 4 blocks and classifier to forensic boundary artifacts.

    Args:
        num_classes (int): Number of target classes (default 2: Real=0, Fake=1).
        freeze_early_layers (bool): Whether to freeze embeddings and early encoder blocks.
        trainable_blocks (int): Number of top encoder blocks to keep trainable (default: 4).
        pretrained_model_name_or_path (str): Hugging Face repo ID or local checkpoint path.

    Returns:
        ViTForImageClassification: Loaded and configured Hugging Face ViT model.
    """
    print(f"[Pillar 1] Loading pretrained model from '{pretrained_model_name_or_path}' with {num_classes} classes...")
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # Replaces the default head with a 2-class head
    )

    if freeze_early_layers:
        # Freeze patch embeddings and position embeddings
        for param in model.vit.embeddings.parameters():
            param.requires_grad = False

        # Support both transformers v5 (model.vit.layers) and v4 (model.vit.encoder.layer)
        if hasattr(model.vit, "layers"):
            encoder_layers = model.vit.layers
        elif hasattr(model.vit, "encoder") and hasattr(model.vit.encoder, "layer"):
            encoder_layers = model.vit.encoder.layer
        else:
            encoder_layers = []

        total_layers = len(encoder_layers)  # 12 blocks for ViT-Base
        freeze_until = max(0, total_layers - trainable_blocks)

        for idx in range(freeze_until):
            for param in encoder_layers[idx].parameters():
                param.requires_grad = False

        print(f"[Pillar 1] Frozen embeddings + encoder blocks 0 to {freeze_until - 1}.")
        print(f"[Pillar 1] Active training on blocks {freeze_until} to {total_layers - 1} + classifier head.")

    return model


def get_vit_optimizer_groups(
    model: ViTForImageClassification,
    backbone_lr: float = 1e-5,
    classifier_lr: float = 1e-4,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Creates an AdamW optimizer with differential learning rates:
    - Lower learning rate for fine-tuning the transformer backbone (prevents catastrophic forgetting).
    - Higher learning rate for the fresh classification head.

    Args:
        model (ViTForImageClassification): The ViT model instance.
        backbone_lr (float): Learning rate for unfrozen transformer encoder layers.
        classifier_lr (float): Learning rate for classification head.
        weight_decay (float): L2 regularization weight decay.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    classifier_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay},
    ]

    return torch.optim.AdamW(optimizer_grouped_parameters)


# ==============================================================================
# 3. TRAINING & VALIDATION ROUTINES
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> Tuple[float, float]:
    """
    Runs a single training epoch with PyTorch Automatic Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass (active on CUDA)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            outputs = model(pixel_values=images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluates the model on validation/test set and computes Loss, Accuracy, and ROC-AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            outputs = model(pixel_values=images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)

    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        # Fallback if a single-class batch occurs
        val_auc = 0.5

    return val_loss, val_acc, val_auc


def train_pillar1_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    patience: int = 6,
    save_path: str = "best_pillar1_vit.pth",
    pretrained_model_name_or_path: str = "google/vit-base-patch16-224",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Complete end-to-end training loop for Pillar 1 ViT.
    
    Features:
    - Trains up to 30 epochs with Early Stopping based on Validation AUC (patience=6).
    - AdamW with differential learning rates (1e-5 backbone, 1e-4 classifier).
    - CosineAnnealingWarmRestarts scheduler (cycle = 10 epochs).
    - CrossEntropyLoss with label_smoothing=0.08.
    - Mixed precision (torch.amp) for acceleration and memory efficiency.
    - Saves the best checkpoint according to highest Validation AUC.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pillar 1] Initializing training on target device: {device}")

    # Build model
    model = build_pillar1_vit(
        num_classes=2,
        freeze_early_layers=True,
        trainable_blocks=4,
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    model.to(device)

    # Optimizer with differential learning rates
    optimizer = get_vit_optimizer_groups(
        model, backbone_lr=1e-5, classifier_lr=1e-4, weight_decay=0.01
    )

    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    # CrossEntropyLoss with label smoothing to prevent overconfidence on noisy/subtle deepfake faces
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    # Mixed precision gradient scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_auc = 0.0
    best_epoch = -1
    best_weights = None
    epochs_no_improve = 0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": []
    }

    print("\n" + "=" * 80)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Val Loss':<14}{'Val Acc':<14}{'Val AUC':<12}")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_auc = evaluate_model(
            model, val_loader, criterion, device
        )

        # Step the learning rate scheduler
        scheduler.step()

        # Log history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<14.4f}{val_loss:<14.4f}{val_acc:<14.4f}{val_auc:<12.4f}")

        # Checkpoint criterion: Validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            
            # Ensure target directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"  --> [Saved] New best model at epoch {epoch} with Val AUC = {best_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Pillar 1] Early stopping triggered! Validation AUC did not improve for {patience} consecutive epochs.")
                break

    print("=" * 80)
    print(f"[Pillar 1] Training Complete. Best Validation AUC: {best_auc:.4f} achieved at Epoch {best_epoch}.")

    # Reload best model weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "model": model,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "history": history
    }


if __name__ == "__main__":
    print("Pillar 1 ViT module loaded successfully.")
    print("Example usage:")
    print("  train_transform = get_pillar1_transforms(train=True)")
    print("  val_transform = get_pillar1_transforms(train=False)")
    print("  results = train_pillar1_pipeline(train_loader, val_loader, num_epochs=30, patience=6)")
