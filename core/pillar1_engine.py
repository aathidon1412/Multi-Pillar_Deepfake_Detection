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
PILLAR1_ULTIMATE_DIR = os.path.join(BASE_DIR, "Pillar 1", "usmfe_vit_ultimate_90_model")
PILLAR1_CHECKPOINT_V2 = os.path.join(BASE_DIR, "Pillar 1", "checkpoints", "best_pillar1_vit_v2.pth")
PILLAR1_CHECKPOINT_TEST = os.path.join(BASE_DIR, "Pillar 1", "checkpoints", "best_pillar1_vit_test.pth")
PILLAR1_DIFFUSION_HEAD = os.path.join(BASE_DIR, "Pillar 1", "checkpoints", "diffusion_vit_head.pt")

PILLAR1_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Global module caches to avoid redundant file I/O
_CACHED_P1_PROC = None
_CACHED_P1_MODEL = None
_CACHED_P1_DIFF_HEAD = None
_CACHED_P1_DEVICE = None
_CACHED_P1_NAME = None

def compute_ela_image(image_input, quality=90):
    """
    Computes JPEG Error Level Analysis (ELA) at specified quality (default 90).
    Amplifies compression error residuals to highlight digital splicing and AI synthesis.
    """
    import numpy as np
    import cv2
    from PIL import Image

    if isinstance(image_input, Image.Image):
        cv_img = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            cv_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            cv_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        elif image_input.shape[2] == 3:
            cv_img = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            cv_img = image_input
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', cv_img, encode_param)
    dec = cv2.imdecode(enc, 1)
    ela = np.abs(cv_img.astype(np.float32) - dec.astype(np.float32))
    scale = 255.0 / max(1.0, float(np.max(ela)))
    ela_scaled = np.clip(ela * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(ela_scaled, cv2.COLOR_BGR2RGB))

def load_pillar1_vit():
    """
    Loads Pillar 1 Vision Transformer Model (Hugging Face ViT) with Dual-Expert Multi-Generator Head.
    Prioritizes the authoritative USMFE ViT Ultimate 90 ELA model combined with the Diffusion Forensic Head.
    Caches model, processor, and expert heads in memory for high-throughput inference.
    """
    global _CACHED_P1_PROC, _CACHED_P1_MODEL, _CACHED_P1_DIFF_HEAD, _CACHED_P1_DEVICE, _CACHED_P1_NAME

    if _CACHED_P1_MODEL is not None and _CACHED_P1_PROC is not None:
        return _CACHED_P1_PROC, _CACHED_P1_MODEL, _CACHED_P1_DEVICE, _CACHED_P1_NAME

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load optional fine-tuned diffusion head
    diff_head = None
    if os.path.exists(PILLAR1_DIFFUSION_HEAD):
        try:
            import torch.nn as nn
            dh = nn.Linear(768, 2)
            ckpt = torch.load(PILLAR1_DIFFUSION_HEAD, map_location=device)
            dh.load_state_dict(ckpt['state_dict'])
            dh.to(device)
            dh.eval()
            diff_head = dh
            _CACHED_P1_DIFF_HEAD = dh
            print(f"[Pillar 1] Successfully loaded Dual-Expert Diffusion Head from: {PILLAR1_DIFFUSION_HEAD}")
        except Exception as e:
            print(f"[Pillar 1] Warning: Could not load diffusion head: {e}")

    # 1. Authoritative USMFE ViT Ultimate 90 ELA directory
    if os.path.exists(PILLAR1_ULTIMATE_DIR):
        try:
            print(f"[Pillar 1] Loading authoritative ViT-Ultimate-90 model: {PILLAR1_ULTIMATE_DIR}")
            processor = ViTImageProcessor.from_pretrained(PILLAR1_ULTIMATE_DIR)
            model = ViTForImageClassification.from_pretrained(PILLAR1_ULTIMATE_DIR)
            model.to(device)
            model.eval()
            _CACHED_P1_PROC = processor
            _CACHED_P1_MODEL = model
            _CACHED_P1_DEVICE = device
            expert_suffix = " + Dual-Expert Diffusion Head" if diff_head is not None else ""
            _CACHED_P1_NAME = f"ViT-Ultimate-90 (ELA Forensics{expert_suffix})"
            print(f"[Pillar 1] Successfully loaded {_CACHED_P1_NAME} onto {device}!")
            return _CACHED_P1_PROC, _CACHED_P1_MODEL, _CACHED_P1_DEVICE, _CACHED_P1_NAME
        except Exception as e:
            print(f"[Pillar 1] Error loading authoritative model: {e}")

    # 2. Checkpoints fallback
    target_ckpt = None
    model_name = "None"
    if os.path.exists(PILLAR1_CHECKPOINT_V2):
        target_ckpt = PILLAR1_CHECKPOINT_V2
        model_name = "best_pillar1_vit_v2.pth"
    elif os.path.exists(PILLAR1_CHECKPOINT_TEST):
        target_ckpt = PILLAR1_CHECKPOINT_TEST
        model_name = "best_pillar1_vit_test.pth"

    if target_ckpt is not None:
        try:
            print(f"[Pillar 1] Loading ViT checkpoint fallback: {target_ckpt}")
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            ckpt = torch.load(target_ckpt, map_location=device)
            model.load_state_dict(ckpt)
            model.to(device)
            model.eval()
            _CACHED_P1_PROC = PILLAR1_TRANSFORM
            _CACHED_P1_MODEL = model
            _CACHED_P1_DEVICE = device
            _CACHED_P1_NAME = model_name
            print(f"[Pillar 1] Successfully loaded {model_name} onto {device}!")
            return _CACHED_P1_PROC, _CACHED_P1_MODEL, _CACHED_P1_DEVICE, _CACHED_P1_NAME
        except Exception as e:
            print(f"[Pillar 1] Error loading checkpoint {target_ckpt}: {e}")

    return None, None, device, "Model file not found"

def run_pillar1_inference(image, transform_or_proc, model, device, model_name="ViT"):
    """
    Executes Pillar 1: Vision Transformer & Spectral ELA Forensics.
    Applies Error Level Analysis (ELA at Q=90) to expose compression and AI synthesis artifacts.
    Evaluates both USMFE ViT feature representations and Diffusion Forensics Head.
    Labels: 0 = Fake (AI Generated / Spliced), 1 = Authentic (Real).
    """
    try:
        if model is not None and transform_or_proc is not None:
            # Preprocess with JPEG Error Level Analysis (Q=90)
            ela_image = compute_ela_image(image, quality=90)

            if hasattr(transform_or_proc, "image_processor_type") or hasattr(transform_or_proc, "feature_extractor_type") or hasattr(transform_or_proc, "preprocess"):
                inputs = transform_or_proc(images=ela_image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
            elif callable(transform_or_proc):
                try:
                    inputs = transform_or_proc(images=ela_image, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                except Exception:
                    pixel_values = transform_or_proc(ela_image).unsqueeze(0).to(device)
            else:
                inputs = transform_or_proc(images=ela_image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                
            with torch.no_grad():
                if hasattr(model, "vit"):
                    vit_out = model.vit(pixel_values=pixel_values)
                    cls_token = vit_out.last_hidden_state[:, 0, :]
                    orig_logits = model.classifier(cls_token)[0]
                else:
                    outputs = model(pixel_values=pixel_values)
                    cls_token = None
                    orig_logits = outputs.logits[0]

            orig_probabilities = F.softmax(orig_logits, dim=-1)
            orig_fake = float(orig_probabilities[0].item())
            orig_real = float(orig_probabilities[1].item())

            # Evaluate diffusion head if available
            diff_fake, diff_real = orig_fake, orig_real
            global _CACHED_P1_DIFF_HEAD
            if _CACHED_P1_DIFF_HEAD is not None and cls_token is not None:
                with torch.no_grad():
                    diff_logits = _CACHED_P1_DIFF_HEAD(cls_token)[0]
                    diff_probabilities = F.softmax(diff_logits, dim=-1)
                    diff_fake = float(diff_probabilities[0].item())
                    diff_real = float(diff_probabilities[1].item())

            # Physical Sensor & Texture Telemetry Gating
            import cv2
            import numpy as np
            from PIL import Image

            if isinstance(image, Image.Image):
                pil_im = image
                cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                cv_img = image
                pil_im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_im = None
                cv_img = None

            exif = pil_im.getexif() if (pil_im is not None and hasattr(pil_im, 'getexif')) else {}
            has_cam = bool(exif.get(0x010f) or exif.get(0x0110) or exif.get(0x0131))

            lap_var = 0.0
            if cv_img is not None:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Calibrated Dual-Expert Fusion
            if has_cam:
                real_prob = max(orig_real, diff_real, 0.88)
            elif orig_fake >= 0.50:
                # Digital splicing seam / boundary manipulation detected by USMFE ViT
                real_prob = orig_real
            elif diff_fake >= 0.80 and not (orig_real > 0.85 and lap_var > 600):
                # Generative diffusion anomaly detected by diffusion head
                real_prob = diff_real
            elif orig_real >= 0.85:
                real_prob = orig_real
            else:
                real_prob = 0.60 * orig_real + 0.40 * diff_real

            fake_prob = 1.0 - real_prob
            is_real = (real_prob >= 0.50)
            confidence = (real_prob * 100.0) if is_real else (fake_prob * 100.0)
            confidence = round(confidence, 2)
            pred_idx = 1 if is_real else 0
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
                "fake_probability": 0.50,
                "pred_idx": -1,
                "status": "Model Missing"
            }
    except Exception as e:
        return {
            "available": False,
            "verdict": "ERROR",
            "is_real": False,
            "confidence": 50.0,
            "real_probability": 0.50,
            "fake_probability": 0.50,
            "details": str(e)
        }
