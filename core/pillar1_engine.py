"""
================================================================================
Pillar 1 Core Engine: Vision Transformer (ViT) & Neural Spectral Forensics
================================================================================
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILLAR1_CHECKPOINT_V2 = os.path.join(BASE_DIR, "Pillar 1", "checkpoints", "best_pillar1_vit_v2.pth")
PILLAR1_CHECKPOINT_TEST = os.path.join(BASE_DIR, "Pillar 1", "checkpoints", "best_pillar1_vit_test.pth")
PILLAR1_LEGACY_DIR = os.path.join(BASE_DIR, "Pillar 1", "usmfe_vit_ultimate_90_model")

PILLAR1_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_pillar1_vit():
    """
    Loads Pillar 1 Vision Transformer Model (Hugging Face ViT).
    Prioritizes the newly trained checkpoints (best_pillar1_vit_v2.pth / best_pillar1_vit_test.pth).
    Falls back to legacy directory if checkpoints are absent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "None"
    
    target_ckpt = None
    if os.path.exists(PILLAR1_CHECKPOINT_V2):
        target_ckpt = PILLAR1_CHECKPOINT_V2
        model_name = "best_pillar1_vit_v2.pth"
    elif os.path.exists(PILLAR1_CHECKPOINT_TEST):
        target_ckpt = PILLAR1_CHECKPOINT_TEST
        model_name = "best_pillar1_vit_test.pth"

    if target_ckpt is not None:
        try:
            print(f"[Pillar 1] Loading ViT checkpoint: {target_ckpt}")
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            ckpt = torch.load(target_ckpt, map_location=device)
            model.load_state_dict(ckpt)
            model.to(device)
            model.eval()
            print(f"[Pillar 1] Successfully loaded {model_name} onto {device}!")
            return PILLAR1_TRANSFORM, model, device, model_name
        except Exception as e:
            print(f"[Pillar 1] Error loading checkpoint {target_ckpt}: {e}")

    if os.path.exists(PILLAR1_LEGACY_DIR):
        try:
            print(f"[Pillar 1] Loading legacy ViT directory: {PILLAR1_LEGACY_DIR}")
            processor = ViTImageProcessor.from_pretrained(PILLAR1_LEGACY_DIR)
            model = ViTForImageClassification.from_pretrained(PILLAR1_LEGACY_DIR)
            model.to(device)
            model.eval()
            return processor, model, device, "ViT-Ultimate-90 (Legacy)"
        except Exception as e:
            print(f"[Pillar 1] Error loading legacy model: {e}")
            return None, None, device, f"Error: {e}"

    return None, None, device, "Model file not found"

def run_pillar1_inference(image, transform_or_proc, model, device, model_name="ViT"):
    """
    Executes Pillar 1: Vision Transformer & Frequency Forensics.
    Uses ImageNet normalization and executes forward pass with pixel_values=images.
    Labels: 0 = Authentic (Real), 1 = Fake (AI Generated).
    """
    try:
        if model is not None and transform_or_proc is not None:
            if callable(transform_or_proc):
                pixel_values = transform_or_proc(image.convert("RGB")).unsqueeze(0).to(device)
            else:
                inputs = transform_or_proc(images=image.convert("RGB"), return_tensors="pt").to(device)
                pixel_values = inputs["pixel_values"]
                
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                
            probabilities = F.softmax(logits, dim=-1)[0]
            real_prob = float(probabilities[0].item())
            fake_prob = float(probabilities[1].item())
            
            is_real = (real_prob >= 0.50)
            confidence = (real_prob * 100.0) if is_real else (fake_prob * 100.0)
            confidence = round(confidence, 2)
            pred_idx = 0 if is_real else 1
            verdict = "AUTHENTIC" if is_real else "FAKE (AI)"
            
            return {
                "available": True,
                "verdict": verdict,
                "is_real": is_real,
                "confidence": confidence,
                "real_probability": real_prob,
                "fake_probability": fake_prob,
                "pred_idx": pred_idx,
                "status": f"Active ({model_name})"
            }
        else:
            return {
                "available": False,
                "verdict": "MODEL NOT LOADED",
                "is_real": False,
                "confidence": 50.0,
                "real_probability": 0.50,
                "pred_idx": -1,
                "status": "Model Missing"
            }
    except Exception as e:
        return {"available": False, "verdict": "ERROR", "is_real": False, "confidence": 50.0, "details": str(e)}
