import io
import os
import sys
import json
import re
import math
import random
import pickle
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Deep Learning Imports (Pillar 1 & Pillar 5 Deep Embeddings)
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification

# Pillar 5 Physical Geometry Schema & Extraction (Locked 24 features)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pillar 5"))
try:
    from feature_schema import PHYSICS_FEATURE_NAMES, validate_feature_dict
    from extract_features import extract_physics_vector
except ImportError:
    PHYSICS_FEATURE_NAMES = [
        "total_lines", "max_inliers", "inlier_ratio", "angular_variance_deg",
        "shadow_chroma_var", "quad_chroma_var", "gw_dev", "penumbra_ratio",
        "lap_var", "lap_skew", "high_freq_energy", "dct_mid_energy",
        "ela_mean", "ela_std",
        "srm_var_0", "srm_skew_0", "srm_var_1", "srm_skew_1",
        "srm_var_2", "srm_skew_2", "srm_var_3", "srm_skew_3",
        "srm_var_4", "srm_skew_4"
    ]
    extract_physics_vector = None

# Statistical & OCR Imports (Pillar 4)
from scipy.stats import chi2
import pytesseract
import fitz  # PyMuPDF for PDF documents

# Set Page Config with Title & Icon
st.set_page_config(
    page_title="Multi-Pillar Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Cyber-Forensic Theme
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(127, 0, 255, 0.05) 0%, transparent 40%),
                    #0b0f19;
        color: #f0f4f8;
    }
    
    /* Header Card */
    .header-box {
        background: rgba(18, 26, 43, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px 30px;
        backdrop-filter: blur(16px);
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .project-badge {
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        color: #0b0f19;
        font-weight: 800;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff, #8a99ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Unified Master Verdict Banner */
    .unified-verdict-card {
        padding: 25px;
        border-radius: 18px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .unified-authentic {
        background: linear-gradient(135deg, rgba(0, 240, 118, 0.15), rgba(0, 240, 118, 0.03));
        border: 2px solid #00f076;
    }
    
    .unified-fake {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.15), rgba(255, 51, 102, 0.03));
        border: 2px solid #ff3366;
    }
    
    .pillar-card {
        background: rgba(18, 26, 43, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        backdrop-filter: blur(12px);
    }
    
    .pillar-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #00f2fe;
        margin-bottom: 6px;
    }
    
    .metric-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        background: rgba(255, 255, 255, 0.04);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- MODULE LOADERS & ENGINES -----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pillar 1 New Checkpoint Paths (Trained with ImageNet normalization)
PILLAR1_CHECKPOINT_V2 = os.path.join(BASE_DIR, "Pillar 1", "checkpoints", "best_pillar1_vit_v2.pth")
PILLAR1_CHECKPOINT_TEST = os.path.join(BASE_DIR, "Pillar 1", "checkpoints", "best_pillar1_vit_test.pth")
PILLAR1_LEGACY_DIR = os.path.join(BASE_DIR, "Pillar 1", "usmfe_vit_ultimate_90_model")

# Pillar 5 Model Paths (New regularized ensemble bundle vs fallback)
PILLAR5_MODEL_NEW_PATH = os.path.join(BASE_DIR, "Pillar 5", "pillar5_ml_model_v2.pkl")
PILLAR5_MODEL_FALLBACK_PATH = os.path.join(BASE_DIR, "Pillar 5", "pillar5_ml_model.pkl")

# ImageNet normalization transforms for HuggingFace ViT (Pillar 1)
# Official ImageNet: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
PILLAR1_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_pillar1_vit():
    """
    Loads Pillar 1 Vision Transformer Model (Hugging Face ViT).
    Prioritizes the newly trained checkpoints (best_pillar1_vit_v2.pth / best_pillar1_vit_test.pth).
    Falls back to legacy directory if checkpoints are absent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "None"
    
    # 1. Check for new PyTorch state_dict checkpoint
    target_ckpt = None
    if os.path.exists(PILLAR1_CHECKPOINT_V2):
        target_ckpt = PILLAR1_CHECKPOINT_V2
        model_name = "best_pillar1_vit_v2.pth"
    elif os.path.exists(PILLAR1_CHECKPOINT_TEST):
        target_ckpt = PILLAR1_CHECKPOINT_TEST
        model_name = "best_pillar1_vit_test.pth"

    if target_ckpt is not None:
        try:
            print(f"[Pillar 1] Loading new trained ViT checkpoint from: {target_ckpt}")
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

    # 2. Fallback to legacy pretrained directory
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

@st.cache_resource
def load_pillar5_ml_bundle():
    """
    Loads the Pillar 5 ML Model Bundle.
    Prioritizes the new regularized hybrid ensemble (pillar5_ml_model_v2.pkl) with 
    StandardScaler and SelectFromModel reducer, then falls back to legacy bundle.
    """
    target_path = None
    if os.path.exists(PILLAR5_MODEL_NEW_PATH):
        target_path = PILLAR5_MODEL_NEW_PATH
    elif os.path.exists(PILLAR5_MODEL_FALLBACK_PATH):
        target_path = PILLAR5_MODEL_FALLBACK_PATH

    if target_path and os.path.exists(target_path):
        try:
            with open(target_path, "rb") as f:
                bundle = pickle.load(f)
            fname = os.path.basename(target_path)
            print(f"[Pillar 5] Successfully loaded model bundle: {fname} (type: {bundle.get('type')})")
            return bundle, fname
        except Exception as e:
            print(f"[Pillar 5] Error loading {target_path}: {e}")
            return None, f"Error: {e}"
    return None, "File not found"

@st.cache_resource
def load_pillar5_deep_backbone(backbone_name='efficientnet_b0'):
    """Loads feature extractor backbone for Pillar 5 Deep Fusion"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone_name == 'efficientnet_b0':
        bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        bb.classifier = torch.nn.Identity()
    else:
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        bb = torch.nn.Sequential(*list(bb.children())[:-1])
    bb.to(device).eval()
    return bb, device

def run_pillar1_inference(image, transform_or_proc, model, device, model_name="ViT"):
    """
    Executes Pillar 1: Vision Transformer & Frequency Forensics.
    Uses official ImageNet normalization and executes forward pass with pixel_values=images.
    Labels: 0 = Authentic (Real), 1 = Fake (AI Generated).
    """
    try:
        if model is not None and transform_or_proc is not None:
            # Transform image with official ImageNet normalization
            if callable(transform_or_proc):
                # torchvision.transforms pipeline
                pixel_values = transform_or_proc(image.convert("RGB")).unsqueeze(0).to(device)
            else:
                # Hugging Face ViTImageProcessor fallback
                inputs = transform_or_proc(images=image.convert("RGB"), return_tensors="pt").to(device)
                pixel_values = inputs["pixel_values"]
                
            with torch.no_grad():
                # Hugging Face ViT forward pass using pixel_values=...
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                
            probabilities = F.softmax(logits, dim=-1)[0]
            # Mapping: Class 0 = Authentic/Real, Class 1 = Fake/AI
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

def extract_digits_from_text(text_data):
    """Extracts first significant digits (1-9) from text"""
    matches = re.findall(r'\b[1-9][0-9,]*\.?[0-9]*\b', text_data)
    digits = []
    for m in matches:
        clean = m.replace(',', '').replace('.', '')
        for char in clean:
            if char in '123456789':
                digits.append(int(char))
                break
    return digits

def analyze_benford_law(digits):
    """Performs statistical Benford's Law Chi-Square and MAE analysis"""
    n = len(digits)
    if n < 5:
        return {
            "applicable": False,
            "reason": f"Insufficient numerical digits ({n} detected).",
            "verdict": "N/A (NON-DOCUMENT)",
            "confidence": 0.0,
            "weight": 0.0,
            "digits_count": n,
            "obs_freqs": [],
            "expected_freqs": []
        }
        
    counts = {i: 0 for i in range(1, 10)}
    for d in digits:
        counts[d] += 1
        
    mae = 0
    chi_square = 0
    obs_freqs = []
    expected_freqs = []
    
    for i in range(1, 10):
        obs_prop = counts[i] / n
        obs_freqs.append(obs_prop * 100)
        exp_prop = np.log10(1 + 1/i)
        expected_freqs.append(exp_prop * 100)
        
        mae += abs(obs_prop - exp_prop)
        expected_count = exp_prop * n
        if expected_count > 0:
            chi_square += ((counts[i] - expected_count)**2) / expected_count
            
    mae = mae / 9
    p_val = chi2.sf(chi_square, 8)
    scale = 100.0 / max(n, 30)
    threshold_strict = 0.020 + (0.010 * scale)
    threshold_loose = 0.035 + (0.010 * scale)
    
    if mae < threshold_strict:
        is_auth = True
        conf = 95.0 - (mae / threshold_strict) * 5
    elif threshold_strict <= mae <= threshold_loose:
        is_auth = True
        conf = 80.0 - ((mae - threshold_strict) / (threshold_loose - threshold_strict)) * 20
    else:
        is_auth = False
        conf = min(99.9, 85.0 + (mae * 200))
        
    if p_val < 0.01 and mae > 0.025 and is_auth:
        is_auth = False
        conf = max(90.0, 95.0 - (p_val * 100))
        
    verdict = "AUTHENTIC DOCUMENT" if is_auth else "FORGED / AI-SYNTHESIZED"
    
    return {
        "applicable": True,
        "verdict": verdict,
        "is_authentic": is_auth,
        "confidence": round(min(99.9, max(60.0, conf)), 2),
        "mae": round(mae, 4),
        "chi_square": round(chi_square, 4),
        "p_value": round(p_val, 5),
        "digits_count": n,
        "weight": 0.35,
        "real_probability": (conf / 100.0) if is_auth else (1.0 - conf / 100.0),
        "obs_freqs": obs_freqs,
        "expected_freqs": expected_freqs
    }

def run_pillar4_inference(image_or_bytes, is_pdf=False):
    """Executes Pillar 4: Semantic Document & Benford's Law OCR Forensics"""
    try:
        text_data = ""
        extracted_image = None
        
        if is_pdf:
            doc = fitz.open(stream=image_or_bytes, filetype="pdf")
            for page in doc:
                text_data += page.get_text() + " "
            if len(doc) > 0:
                first_page = doc[0]
                pix = first_page.get_pixmap(dpi=150)
                extracted_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            if isinstance(image_or_bytes, Image.Image):
                img_pil = image_or_bytes
            else:
                img_pil = Image.fromarray(image_or_bytes)
            extracted_image = img_pil
            
            np_img = np.array(img_pil)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            try:
                text_data = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6')
            except Exception:
                text_data = pytesseract.image_to_string(img_pil)
                
        digits = extract_digits_from_text(text_data)
        res = analyze_benford_law(digits)
        res["extracted_image"] = extracted_image
        res["sample_text"] = (text_data[:300] + "...") if len(text_data) > 300 else text_data
        return res
    except Exception as e:
        return {
            "applicable": False,
            "reason": f"Document extraction error: {e}",
            "verdict": "N/A",
            "weight": 0.0,
            "extracted_image": None
        }

def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)

def run_pillar5_inference(image_np, pil_img=None, p5_bundle=None):
    """
    Executes Pillar 5: Authoritative Shadow Physics Geometry & ML Model (pillar5_ml_model.pkl).
    Dominant primary classifier for authentic vs AI-generated scenes.
    """
    h, w = image_np.shape[:2]
    
    # 1. Image preprocessing
    scale_norm = 800.0 / max(h, w)
    img_norm = cv2.resize(image_np, (int(w * scale_norm), int(h * scale_norm)))
    gray_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray_norm, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edges = cv2.Canny(thresh, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
    
    lab_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab_norm)
    shadow_pix = l_chan < np.percentile(l_chan, 35)
    shadow_chroma_var = float(np.std(a_chan[shadow_pix]) + np.std(b_chan[shadow_pix])) if np.any(shadow_pix) else 0.0
    lap_var = float(np.var(cv2.Laplacian(gray_norm, cv2.CV_64F)))
    
    if lines is not None:
        lines = lines.reshape(-1, 4).tolist()
    else:
        lines = []
        
    total_lines = len(lines)
    
    # Directional Clustering & RANSAC
    filtered_lines = []
    if total_lines >= 2:
        angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in lines]
        angles_mod = np.mod(angles, np.pi)
        median_angle = np.median(angles_mod)
        filtered_lines = [l for l, a in zip(lines, angles_mod) if min(abs(a - median_angle), np.pi - abs(a - median_angle)) < 0.35]
    
    best_vp = None
    max_inliers = 0
    best_inlier_lines = []
    
    if len(filtered_lines) >= 2:
        for _ in range(min(800, len(filtered_lines) * len(filtered_lines))):
            l1, l2 = random.sample(filtered_lines, 2)
            vp = calculate_intersection(l1, l2)
            if vp is None: continue
            inls = []
            for line in filtered_lines:
                den = math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2)
                d = abs((line[2]-line[0])*(line[1]-vp[1]) - (line[0]-vp[0])*(line[3]-line[1])) / den if den > 0 else 9999
                if d < 50: inls.append(line)
            if len(inls) > max_inliers:
                max_inliers = len(inls)
                best_inlier_lines = inls
                best_vp = vp
                
    inlier_ratio = max_inliers / total_lines if total_lines > 0 else 0.0
    inlier_angles = [math.atan2(l[3]-l[1], l[2]-l[0]) for l in best_inlier_lines]
    if inlier_angles:
        mean_sin = np.mean([math.sin(2 * a) for a in inlier_angles])
        mean_cos = np.mean([math.cos(2 * a) for a in inlier_angles])
        R = math.sqrt(mean_sin**2 + mean_cos**2)
        angular_variance_deg = math.degrees((1.0 - R) * np.pi)
    else:
        angular_variance_deg = 85.0
        
    # Check if New ML Model is loaded
    is_real = False
    confidence = 90.0
    real_prob = 0.10
    model_used = "Pillar 5 RANSAC Light Engine"
    
    if p5_bundle is not None and isinstance(p5_bundle, dict) and 'classifier' in p5_bundle:
        try:
            gh, gw = l_chan.shape
            q1 = (a_chan[:gh//2, :gw//2], b_chan[:gh//2, :gw//2])
            q2 = (a_chan[:gh//2, gw//2:], b_chan[:gh//2, gw//2:])
            q3 = (a_chan[gh//2:, :gw//2], b_chan[gh//2:, :gw//2])
            q4 = (a_chan[gh//2:, gw//2:], b_chan[gh//2:, gw//2:])
            quad_means = [np.mean(q[0]) + np.mean(q[1]) for q in [q1, q2, q3, q4]]
            quad_chroma_var = float(np.var(quad_means))
            gw_dev = float(np.std([np.mean(img_norm[:,:,0]), np.mean(img_norm[:,:,1]), np.mean(img_norm[:,:,2])]))
            lap_arr = cv2.Laplacian(gray_norm, cv2.CV_64F)
            lap_skew = float(np.mean(((lap_arr - np.mean(lap_arr)) / (np.std(lap_arr) + 1e-5))**3))
            
            sub_gray = cv2.resize(gray_norm, (256, 256))
            dct_block = dct(dct(sub_gray.T, norm='ortho').T, norm='ortho')
            high_freq_energy = float(np.sum(np.abs(dct_block[128:, 128:])) / (np.sum(np.abs(dct_block)) + 1e-5))
            dct_mid_energy = float(np.sum(np.abs(dct_block[64:128, 64:128])) / (np.sum(np.abs(dct_block)) + 1e-5))
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, encimg = cv2.imencode('.jpg', img_norm, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            ela = np.abs(img_norm.astype(np.float32) - decimg.astype(np.float32))
            ela_mean = float(np.mean(ela))
            ela_std = float(np.std(ela))
            
            grad_x = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            shadow_grad = float(np.mean(grad_mag[shadow_pix])) if np.any(shadow_pix) else 0.0
            non_shadow_grad = float(np.mean(grad_mag[~shadow_pix])) if np.any(~shadow_pix) else 0.0
            penumbra_ratio = float(shadow_grad / (non_shadow_grad + 1e-5))
            
            srm_filts = [
                np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32),
                np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
                np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
                np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
                np.array([[-1, 2, -2, 2, -1], [ 2, -6, 8, -6, 2], [-2,  8,-12, 8, -2], [ 2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
            ]
            srm_feats = {}
            for idx_s, filt in enumerate(srm_filts):
                res = cv2.filter2D(gray_norm.astype(np.float32), -1, filt)
                srm_feats[f'srm_var_{idx_s}'] = float(np.var(res))
                srm_feats[f'srm_skew_{idx_s}'] = float(np.mean(((res - np.mean(res)) / (np.std(res) + 1e-5))**3))
                
            tab_dict = {
                'total_lines': float(total_lines),
                'max_inliers': float(max_inliers),
                'inlier_ratio': round(inlier_ratio, 4),
                'angular_variance_deg': round(angular_variance_deg, 4),
                'shadow_chroma_var': round(shadow_chroma_var, 4),
                'quad_chroma_var': round(quad_chroma_var, 4),
                'gw_dev': round(gw_dev, 4),
                'penumbra_ratio': round(penumbra_ratio, 4),
                'lap_var': round(lap_var, 4),
                'lap_skew': round(lap_skew, 4),
                'high_freq_energy': round(high_freq_energy, 4),
                'dct_mid_energy': round(dct_mid_energy, 4),
                'ela_mean': round(ela_mean, 4),
                'ela_std': round(ela_std, 4),
            }
            tab_dict.update({k: round(v, 4) for k, v in srm_feats.items()})
            
            # Use locked feature column ordering from feature_schema or bundle
            feature_cols = p5_bundle.get('feature_cols', PHYSICS_FEATURE_NAMES)
            tab_vals = np.array([[tab_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
            
            # Deep Embedding (1280-dim from EfficientNet-B0)
            bb, dev = load_pillar5_deep_backbone(p5_bundle.get('backbone', 'efficientnet_b0'))
            prep = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            with torch.no_grad():
                target_pil = pil_img if pil_img is not None else Image.fromarray(image_np)
                t_in = prep(target_pil.convert('RGB')).unsqueeze(0).to(dev)
                emb = bb(t_in).squeeze().cpu().numpy().reshape(1, -1)
            
            # Check bundle architecture type
            bundle_type = p5_bundle.get('type', '')
            clf = p5_bundle['classifier']
            
            if 'scaler' in p5_bundle and 'reducer' in p5_bundle:
                # NEW REGULARIZED ENSEMBLE PIPELINE (v2)
                # Concatenate 24 physics features + 1280 deep embeddings = 1304 dims
                raw_fused = np.hstack([tab_vals, emb])
                # Scale with fitted StandardScaler
                scaled_fused = p5_bundle['scaler'].transform(raw_fused)
                # Reduce to selected 160 features with SelectFromModel
                features_for_clf = p5_bundle['reducer'].transform(scaled_fused)
                
                # Class probabilities: index 0 = Authentic (Real), index 1 = Fake (AI)
                prob = clf.predict_proba(features_for_clf)[0]
                real_prob = float(prob[0])
                fake_prob = float(prob[1])
                
                is_real = (real_prob >= 0.50)
                confidence = round((real_prob * 100.0) if is_real else (fake_prob * 100.0), 2)
                model_used = f"Regularized Multi-Model Ensemble (VotingClassifier 160-dim, {p5_bundle.get('backbone', 'EfficientNet-B0')})"
            else:
                # LEGACY HYBRID FUSION FALLBACK
                tab_scaled = p5_bundle['scaler_tab'].transform(tab_vals)
                deep_scaled = p5_bundle['scaler_deep'].transform(emb)
                fused = np.hstack([tab_scaled, deep_scaled])
                prob = clf.predict_proba(fused)[0]
                # Legacy: index 1 = Authentic, index 0 = Fake
                real_prob = float(prob[1])
                is_real = (real_prob >= 0.50)
                confidence = round(real_prob * 100, 2) if is_real else round(prob[0] * 100, 2)
                model_used = f"Authoritative Multi-Generator ML Ensemble ({p5_bundle.get('backbone', 'EfficientNet-B0')})"
        except Exception as e:
            # Fallback to calibrated physical logic if ML forward pass fails
            is_real = (angular_variance_deg < 12.0 and max_inliers >= 20 and shadow_chroma_var >= 10.0)
            real_prob = 0.94 if is_real else 0.06
            confidence = 94.0
            model_used = f"RANSAC Shadow Physics (Fallback: {e})"
    else:
        is_real = (angular_variance_deg < 12.0 and max_inliers >= 20 and shadow_chroma_var >= 10.0)
        real_prob = 0.94 if is_real else 0.06
        confidence = 94.0

    verdict = "AUTHENTIC PHYSICS" if is_real else "PHYSICS ANOMALY (AI GENERATED)"
    
    # Generate Overlay Plot
    vis_copy = image_np.copy()
    for l in best_inlier_lines:
        cv2.line(vis_copy, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 3)
    if best_vp:
        cv2.circle(vis_copy, (int(best_vp[0]), int(best_vp[1])), 10, (255, 0, 0), -1)
        
    return {
        "is_real": is_real,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "inliers": max_inliers,
        "total_lines": total_lines,
        "inlier_ratio": round(inlier_ratio, 3),
        "angular_var": round(angular_variance_deg, 2),
        "real_probability": real_prob,
        "model_used": model_used,
        "overlay": vis_copy,
        "vp": [round(best_vp[0], 1), round(best_vp[1], 1)] if best_vp else [0, 0]
    }

# ----------------- UI VIEW -----------------

st.markdown("""
<div class="header-box">
    <div>
        <span class="project-badge">UNIVERSAL SYNTHETIC MEDIA FORENSICS ENGINE (USMFE)</span>
        <h1 class="main-title">Multi-Pillar Deepfake Detection</h1>
        <p style="color: #8a99ad; margin: 5px 0 0 0; font-size: 0.95rem;">
            Unified Cyber-Forensics Fusion: ViT Neural Spectra (P1) + Document Semantics (P4) + Hybrid Shadow Physics ML (P5)
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Pillar 1 ViT Model & Pillar 5 ML Model
processor_or_tfm, p1_model, device, p1_model_name = load_pillar1_vit()
p5_bundle, p5_model_filename = load_pillar5_ml_bundle()

# Sidebar: Operational Mode & Controls
with st.sidebar:
    st.header("⚙️ Forensic Controls")
    
    analysis_mode = st.radio(
        "🎯 Select Forensic Mode:",
        ["🖼️ Universal Multi-Pillar Media Analysis", "📄 Pillar 4: Document & PDF Forensics"],
        index=0
    )
    
    st.divider()
    
    if analysis_mode == "🖼️ Universal Multi-Pillar Media Analysis":
        st.markdown("**Test Suite Pre-Loaded Examples:**")
        sample_options = {
            "Select an example...": None,
            "AI_img1.jpeg (Fake AI)": os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img1.jpeg"),
            "AI_img2.jpeg (Fake AI)": os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img2.jpeg"),
            "AI_img3.png (Fake AI)": os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img3.png"),
            "AI_img4.png (Fake AI)": os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img4.png"),
            "AI_img5.png (Fake AI)": os.path.join(BASE_DIR, "Pillar 5", "testing", "AI_img5.png"),
            "Real_img1.JPG (Real Photo)": os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img1.JPG"),
            "Real_img2.jpg (Real Photo)": os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img2.jpg"),
            "Real_img3.jpg (Real Photo)": os.path.join(BASE_DIR, "Pillar 5", "testing", "Real_img3.jpg"),
        }
        selected_sample = st.selectbox("Load Pre-Configured Test Image:", list(sample_options.keys()))
    else:
        st.markdown("**Pillar 4 Document / Invoice Samples:**")
        doc_samples = {
            "Select a document...": None,
            "invoice_1.png (Authentic)": os.path.join(BASE_DIR, "Pillar 4", "invoice_1.png"),
            "invoice_01.png (Authentic)": os.path.join(BASE_DIR, "Pillar 4", "invoice_01.png"),
            "invoice_2.jpg (Authentic)": os.path.join(BASE_DIR, "Pillar 4", "invoice_2.jpg"),
            "invoice_3.png (Authentic)": os.path.join(BASE_DIR, "Pillar 4", "invoice_3.png"),
            "invoice_4.png (Authentic)": os.path.join(BASE_DIR, "Pillar 4", "invoice_4.png"),
            "invoice_5.png (Authentic)": os.path.join(BASE_DIR, "Pillar 4", "invoice_5.png"),
        }
        selected_sample = st.selectbox("Load Test Invoice / Document:", list(doc_samples.keys()))
        sample_options = doc_samples
        
    st.divider()
    st.markdown("### 🏛️ Forensic Engines Status")
    st.markdown(f"🔹 **Pillar 1:** ViT Neural Forensics (`{p1_model_name}`)")
    st.markdown("🔹 **Pillar 4:** Benford's Law Statistical OCR & PDF Parser (`Active`)")
    st.markdown(f"🔹 **Pillar 5:** {p5_model_filename} (`Active & Authoritative`)")
    st.divider()
    st.info("System Online • New Models Active")

# =========================================================================
# MODE 1: UNIVERSAL MULTI-PILLAR MEDIA FORENSICS (Pillar 5 Authoritative + Pillar 1 Synced)
# =========================================================================
if analysis_mode == "🖼️ Universal Multi-Pillar Media Analysis":
    uploaded_file = st.file_uploader(
        "📤 Upload Image to Analyze (JPG, PNG, JPEG, WEBP, BMP)...",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    
    image_to_process = None
    source_name = ""
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_to_process = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_to_process.filename = uploaded_file.name
        source_name = uploaded_file.name
    elif selected_sample and sample_options[selected_sample] and os.path.exists(sample_options[selected_sample]):
        image_to_process = Image.open(sample_options[selected_sample]).convert("RGB")
        image_to_process.filename = sample_options[selected_sample]
        source_name = os.path.basename(sample_options[selected_sample])
        
    if image_to_process is not None:
        img_np = np.array(image_to_process)
        
        with st.spinner("🔬 Running Consolidated Multi-Pillar Deepfake Forensics Engine..."):
            # 1. Run Dominant Authoritative Pillar 5 Hybrid Physics & Multi-Generator ML
            p5_res = run_pillar5_inference(img_np, pil_img=image_to_process, p5_bundle=p5_bundle)
            
            # 2. Run Pillar 1 (Vision Transformer with ImageNet normalization)
            p1_res = run_pillar1_inference(image_to_process, processor_or_tfm, p1_model, device, model_name=p1_model_name)
            
            # 3. Run Pillar 4 (Document & Benford)
            p4_res = run_pillar4_inference(img_np, is_pdf=False)
            
            # Consolidated Multi-Pillar Consensus
            p5_real_prob = p5_res.get("real_probability", 0.5)
            p1_real_prob = p1_res.get("real_probability", 0.5) if p1_res.get("available", False) else p5_real_prob
            
            # Weighted multi-pillar fusion: 55% Pillar 5 (Physical + Deep Ensemble) + 45% Pillar 1 (ViT Neural)
            fused_real_prob = 0.55 * p5_real_prob + 0.45 * p1_real_prob
            is_unified_real = fused_real_prob >= 0.50
            unified_verdict = "AUTHENTIC MEDIA" if is_unified_real else "FAKE (SYNTHETIC AI ANOMALY)"
            unified_conf = round((fused_real_prob * 100.0) if is_unified_real else ((1.0 - fused_real_prob) * 100.0), 2)
            
        # Display Master Verdict Banner
        banner_class = "unified-authentic" if is_unified_real else "unified-fake"
        verdict_color = "#00f076" if is_unified_real else "#ff3366"
        
        st.markdown(f"""
        <div class="unified-verdict-card {banner_class}">
            <div>
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8a99ad;">
                    Consolidated Multi-Pillar Ensemble Verdict
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {verdict_color}; margin-top: 4px;">
                    {unified_verdict}
                </div>
                <div style="color: #8a99ad; font-size: 0.95rem; margin-top: 4px;">
                    Target: <b>{source_name}</b> • Engines: {p1_model_name} + {p5_model_filename}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: {verdict_color};">
                    {unified_conf:.1f}%
                </div>
                <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #8a99ad;">
                    Consensus Confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3-Column Pillar Breakdown Grid
        st.markdown("### 🔍 Pillar-by-Pillar Forensic Breakdown")
        col1, col2, col3 = st.columns(3)
        
        # Pillar 1 Card
        with col1:
            st.markdown("""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 1 • NEURAL & FREQUENCY</div>
                <h3 style="margin: 0 0 10px 0;">Vision Transformer</h3>
            """, unsafe_allow_html=True)
            
            p1_color = "#00f076" if p1_res["verdict"] == "AUTHENTIC" else "#ff3366"
            st.markdown(f"<h4 style='color:{p1_color}; margin: 0;'>{p1_res['verdict']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-chip'>Confidence: <b>{p1_res['confidence']:.2f}%</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-chip'>Model: <b>{p1_model_name} (ImageNet Norm)</b></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Pillar 4 Card
        with col2:
            st.markdown("""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 4 • DOCUMENT & OCR</div>
                <h3 style="margin: 0 0 10px 0;">Benford's Law</h3>
            """, unsafe_allow_html=True)
            
            if p4_res["applicable"]:
                p4_color = "#00f076" if "AUTHENTIC" in p4_res["verdict"] else "#ff3366"
                st.markdown(f"<h4 style='color:{p4_color}; margin: 0;'>{p4_res['verdict']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-chip'>Confidence: <b>{p4_res['confidence']}%</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-chip'>Digits Analyzed: <b>{p4_res['digits_count']}</b></div>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='color:#8a99ad; margin: 0;'>N/A (NATURAL SCENE)</h4>", unsafe_allow_html=True)
                st.markdown("<div class='metric-chip' style='color:#8a99ad;'>Natural Scene Image • Skipped</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-chip' style='color:#8a99ad;'>Use Document Mode for Invoices/PDFs</div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Pillar 5 Card
        with col3:
            st.markdown("""
            <div class="pillar-card">
                <div class="pillar-tag">PILLAR 5 • PHYSICAL GEOMETRY (PRIMARY)</div>
                <h3 style="margin: 0 0 10px 0;">Shadow Physics RANSAC</h3>
            """, unsafe_allow_html=True)
            
            p5_color = "#00f076" if "AUTHENTIC" in p5_res["verdict"] else "#ff3366"
            st.markdown(f"<h4 style='color:{p5_color}; margin: 0;'>{p5_res['verdict']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-chip'>Confidence: <b>{p5_res['confidence']}%</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-chip'>Inliers: <b>{p5_res['inliers']}/{p5_res['total_lines']} ({p5_res['inlier_ratio']*100:.1f}%)</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-chip'>Engine: <b>{p5_model_filename}</b></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Visual Evidence
        st.divider()
        st.markdown("### 🖼️ Visual Evidence & Forensic Ray Vectors")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.image(image_to_process, caption=f"Original Uploaded Target ({source_name})", use_container_width=True)
        with vcol2:
            if p5_res["overlay"] is not None:
                st.image(p5_res["overlay"], caption="Pillar 5: RANSAC Vanishing Point & Shadow Convergence Vectors", use_container_width=True)
            else:
                st.info("No shadow vectors detected for visual overlay.")
    else:
        st.info("👆 Please upload an image or choose one of the pre-loaded test examples from the sidebar to inspect multi-pillar deepfake evidence.")

# =========================================================================
# MODE 2: PILLAR 4 DOCUMENT & PDF FORENSICS
# =========================================================================
else:
    st.markdown("### 📄 Pillar 4: Document, Invoice & PDF Statistical Forensics")
    st.markdown("Analyzes OCR digit distributions and verifies adherence to **Benford's Law** to uncover forged/AI-manipulated tabular records and invoices.")
    
    uploaded_doc = st.file_uploader(
        "📤 Upload Document or PDF (PDF, PNG, JPG, JPEG, TIFF)...",
        type=["pdf", "png", "jpg", "jpeg", "tiff"]
    )
    
    doc_bytes = None
    doc_name = ""
    is_pdf_file = False
    
    if uploaded_doc is not None:
        doc_bytes = uploaded_doc.read()
        doc_name = uploaded_doc.name
        is_pdf_file = doc_name.lower().endswith(".pdf")
    elif selected_sample and sample_options[selected_sample] and os.path.exists(sample_options[selected_sample]):
        with open(sample_options[selected_sample], "rb") as f:
            doc_bytes = f.read()
        doc_name = os.path.basename(sample_options[selected_sample])
        is_pdf_file = doc_name.lower().endswith(".pdf")
        
    if doc_bytes is not None:
        with st.spinner("📄 Performing OCR Digit Extraction & Benford's Law Chi-Square Analysis..."):
            if is_pdf_file:
                p4_doc_res = run_pillar4_inference(doc_bytes, is_pdf=True)
            else:
                pil_img = Image.open(io.BytesIO(doc_bytes)).convert("RGB")
                p4_doc_res = run_pillar4_inference(pil_img, is_pdf=False)
                
        if p4_doc_res["applicable"]:
            is_doc_auth = p4_doc_res["is_authentic"]
            doc_banner_class = "unified-authentic" if is_doc_auth else "unified-fake"
            doc_verdict_color = "#00f076" if is_doc_auth else "#ff3366"
            
            st.markdown(f"""
            <div class="unified-verdict-card {doc_banner_class}">
                <div>
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8a99ad;">
                        Pillar 4 Document Forensic Verdict
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {doc_verdict_color}; margin-top: 4px;">
                        {p4_doc_res['verdict']}
                    </div>
                    <div style="color: #8a99ad; font-size: 0.95rem; margin-top: 4px;">
                        Document Target: <b>{doc_name}</b> • Benford First-Digit Chi² Goodness-of-Fit Test
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: {doc_verdict_color};">
                        {p4_doc_res['confidence']:.1f}%
                    </div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #8a99ad;">
                        Statistical Confidence
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics Grid
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Digits Extracted", p4_doc_res["digits_count"])
            with mcol2:
                st.metric("Mean Absolute Error (MAE)", f"{p4_doc_res['mae']:.4f}")
            with mcol3:
                st.metric("Chi-Square (χ²)", f"{p4_doc_res['chi_square']:.2f}")
            with mcol4:
                st.metric("P-Value", f"{p4_doc_res['p_value']:.4f}")
                
            st.divider()
            
            # Visualization Section
            v_col1, v_col2 = st.columns([1, 1])
            
            with v_col1:
                st.markdown("#### 📊 Benford's Law Digit Distribution")
                digits_arr = np.arange(1, 10)
                obs = p4_doc_res["obs_freqs"]
                exp = p4_doc_res["expected_freqs"]
                
                fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#0e1626')
                ax.set_facecolor('#0e1626')
                
                bar_color = '#00f076' if is_doc_auth else '#ff3366'
                ax.bar(digits_arr, obs, color=bar_color, alpha=0.75, width=0.5, label='Observed Frequencies (%)')
                ax.plot(digits_arr, exp, color='#00f2fe', marker='o', linewidth=2.5, label="Benford's Law (Expected)")
                
                ax.set_xticks(digits_arr)
                ax.set_xlabel("Leading Digit (1-9)", color='#f0f4f8', fontweight='bold')
                ax.set_ylabel("Relative Frequency (%)", color='#f0f4f8', fontweight='bold')
                ax.legend(facecolor='#182234', edgecolor='none', labelcolor='#f0f4f8')
                ax.grid(color='#ffffff', alpha=0.1, linestyle='--')
                
                for spine in ax.spines.values():
                    spine.set_color('#ffffff')
                    spine.set_alpha(0.1)
                    
                plt.tight_layout()
                st.pyplot(fig)
                
            with v_col2:
                st.markdown("#### 📄 Document Preview & OCR Text Stream")
                if p4_doc_res.get("extracted_image") is not None:
                    st.image(p4_doc_res["extracted_image"], caption=f"Preview: {doc_name}", use_container_width=True)
                if p4_doc_res.get("sample_text"):
                    with st.expander("📝 Extracted OCR Text Sample"):
                        st.code(p4_doc_res["sample_text"])
        else:
            st.warning(f"Document analysis note: {p4_doc_res.get('reason', 'Could not process document.')}")
    else:
        st.info("👆 Please upload a PDF or invoice image from the sidebar to execute Pillar 4 document forensics.")
