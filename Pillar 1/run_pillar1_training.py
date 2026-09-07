"""
Runner script for Pillar 1 ViT test training:
- Loads DataLoaders with get_pillar1_dataloaders (batch_size=16, num_workers=0)
- Executes train_pillar1_pipeline for 8 epochs, patience=3
- Checkpoint saved to: Pillar 1/checkpoints/best_pillar1_vit_test.pth
- Prints final metrics & saves history
"""

import os
import sys
import json
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset_adapter import get_pillar1_dataloaders
from train_pillar1_vit import train_pillar1_pipeline

def main():
    print("=" * 80)
    print("LAUNCHING PILLAR 1 ViT TEST TRAINING")
    print("=" * 80)

    train_root = os.path.join(current_dir, "temp_dataset", "train")
    val_root   = os.path.join(current_dir, "temp_dataset", "val")
    save_path  = os.path.join(current_dir, "checkpoints", "best_pillar1_vit_test.pth")
    history_path = os.path.join(current_dir, "checkpoints", "pillar1_test_history.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Train root : {train_root}")
    print(f"Val root   : {val_root}")
    print(f"Save path  : {save_path}")

    # 1. Instantiate DataLoaders
    print("\n[Step 1/2] Creating DataLoaders with ImageNet normalization & augmentations...")
    train_loader, val_loader = get_pillar1_dataloaders(
        train_root=train_root,
        val_root=val_root,
        batch_size=16,
        num_workers=0
    )

    print(f" -> Train batches: {len(train_loader)} (total {len(train_loader.dataset)} samples)")
    print(f" -> Val batches  : {len(val_loader)} (total {len(val_loader.dataset)} samples)")

    # 2. Train ViT
    print("\n[Step 2/2] Training Vision Transformer (google/vit-base-patch16-224)...")
    t0 = time.time()
    try:
        results = train_pillar1_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=8,
            patience=3,
            save_path=save_path,
            pretrained_model_name_or_path="google/vit-base-patch16-224"
        )
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    history = results["history"]

    # 3. Print Final Summary Metrics
    print("\n" + "=" * 80)
    print("                      PILLAR 1 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Elapsed Time         : {elapsed / 60:.2f} minutes")
    print(f"Best Validation AUC  : {results['best_auc']:.4f}")
    print(f"Best Epoch           : {results['best_epoch']}")
    print(f"Final Train Loss     : {history['train_loss'][-1]:.4f}")
    print(f"Final Train Accuracy : {history['train_acc'][-1] * 100:.2f}%")
    print(f"Final Val Loss       : {history['val_loss'][-1]:.4f}")
    print(f"Final Val Accuracy   : {history['val_acc'][-1] * 100:.2f}%")
    print(f"Checkpoint Saved To  : {save_path}")
    print("=" * 80)

    # 4. Save history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    main()
