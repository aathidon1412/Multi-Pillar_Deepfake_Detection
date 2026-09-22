"""
================================================================================
Pillar 5 Core Engine: Hybrid Perspective Geometry & Multi-Generator ML Ensemble
================================================================================
Implements:
- RANSAC Vanishing Point & Shadow Ray Convergence Vectors
- 24 Tabular Physics & Steganalysis Features (SRM residuals 0-4, ELA, DCT, Penumbra)
- 1,280-dim EfficientNet-B0 Deep Visual Representation
- Regularized Multi-Model Soft Voting Ensemble (LightGBM + XGBoost + RF + ExtraTrees)
"""

import os
import sys
import math
import random
import pickle
import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import dct
import torch
import torchvision.models as models
import torchvision.transforms as transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P5_DIR = os.path.join(BASE_DIR, "Pillar 5")
if P5_DIR not in sys.path:
    sys.path.insert(0, P5_DIR)

try:
    from feature_schema import PHYSICS_FEATURE_NAMES
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

PILLAR5_MODEL_NEW_PATH = os.path.join(P5_DIR, "pillar5_ml_model_v2.pkl")
PILLAR5_MODEL_FALLBACK_PATH = os.path.join(P5_DIR, "pillar5_ml_model.pkl")

def load_pillar5_ml_bundle():
    """Loads the regularized multi-model ensemble bundle (pillar5_ml_model_v2.pkl)."""
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

def load_pillar5_deep_backbone(backbone_name='efficientnet_b0'):
    """Loads feature extractor backbone for Pillar 5 Deep Fusion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone_name == 'efficientnet_b0':
        bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        bb.classifier = torch.nn.Identity()
    else:
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        bb = torch.nn.Sequential(*list(bb.children())[:-1])
    bb.to(device).eval()
    return bb, device

def calculate_intersection(line1, line2):
    """Computes Cartesian intersection point between two line segments."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)

def run_pillar5_inference(image_np, pil_img=None, p5_bundle=None):
    """
    Executes Pillar 5: Authoritative Shadow Physics Geometry & ML Model (pillar5_ml_model_v2.pkl).
    Extracts 24 physics features + 1,280 deep embeddings, scales with StandardScaler,
    reduces with SelectFromModel, and predicts class probabilities with soft-voting ensemble.
    """
    h, w = image_np.shape[:2]
    
    # 1. Image preprocessing
    scale_norm = 800.0 / max(h, w)
    img_norm = cv2.resize(image_np, (int(w * scale_norm), int(h * scale_norm)))
    gray_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray_norm, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    edges = cv2.Canny(thresh, 50, 150)
    
    # 2. Vector Extraction (Hough Lines)
    hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=60, maxLineGap=10)
    lines = [l[0].tolist() for l in hough_lines] if hough_lines is not None else []
    total_lines = len(lines)
    
    # Color conversions for illumination
    lab = cv2.cvtColor(img_norm, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    shadow_pix = (l_chan < np.percentile(l_chan, 25))
    shadow_chroma_var = float(np.var(a_chan[shadow_pix]) + np.var(b_chan[shadow_pix])) if np.any(shadow_pix) else 15.0
    lap_var = float(cv2.Laplacian(gray_norm, cv2.CV_64F).var())
    
    # 3. RANSAC Vanishing Point Estimation
    filtered_lines = []
    if lines:
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
            if vp is None:
                continue
            inls = []
            for line in filtered_lines:
                den = math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2)
                d = abs((line[2]-line[0])*(line[1]-vp[1]) - (line[0]-vp[0])*(line[3]-line[1])) / den if den > 0 else 9999
                if d < 50:
                    inls.append(line)
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
        
    # ML Classification
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
            
            feature_cols = p5_bundle.get('feature_cols', PHYSICS_FEATURE_NAMES)
            tab_vals = np.array([[tab_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
            
            # Deep Embedding (1,280-dim from EfficientNet-B0)
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
            
            clf = p5_bundle['classifier']
            
            if 'scaler' in p5_bundle and 'reducer' in p5_bundle:
                raw_fused = np.hstack([tab_vals, emb])
                scaled_fused = p5_bundle['scaler'].transform(raw_fused)
                features_for_clf = p5_bundle['reducer'].transform(scaled_fused)
                
                prob = clf.predict_proba(features_for_clf)[0]
                real_prob = float(prob[0])
                fake_prob = float(prob[1])
                
                is_real = (real_prob >= 0.50)
                confidence = round((real_prob * 100.0) if is_real else (fake_prob * 100.0), 2)
                model_used = f"Regularized Ensemble ({p5_bundle.get('backbone', 'EfficientNet-B0')})"
            else:
                tab_scaled = p5_bundle['scaler_tab'].transform(tab_vals)
                deep_scaled = p5_bundle['scaler_deep'].transform(emb)
                fused = np.hstack([tab_scaled, deep_scaled])
                prob = clf.predict_proba(fused)[0]
                real_prob = float(prob[1])
                is_real = (real_prob >= 0.50)
                confidence = round(real_prob * 100, 2) if is_real else round(prob[0] * 100, 2)
                model_used = f"Legacy Ensemble ({p5_bundle.get('backbone', 'EfficientNet-B0')})"
        except Exception as e:
            is_real = (angular_variance_deg < 12.0 and max_inliers >= 20 and shadow_chroma_var >= 10.0)
            real_prob = 0.94 if is_real else 0.06
            confidence = 94.0
            model_used = f"RANSAC Shadow Physics (Fallback: {e})"
    else:
        is_real = (angular_variance_deg < 12.0 and max_inliers >= 20 and shadow_chroma_var >= 10.0)
        real_prob = 0.94 if is_real else 0.06
        confidence = 94.0

    fake_prob = 1.0 - real_prob
    srm4_val = float(tab_dict.get('srm_var_4', 0.0))
    srm2_val = float(tab_dict.get('srm_var_2', 0.0))
    ela_val = float(tab_dict.get('ela_std', 0.0))

    # Steganographic Anomaly Detection
    is_prnu_anomaly = (srm4_val < 100.0)
    if is_prnu_anomaly and fake_prob < 0.60:
        fake_prob = max(fake_prob, 0.65 + 0.25 * (1.0 - min(1.0, srm4_val / 100.0)))
        real_prob = 1.0 - fake_prob
        is_real = False
        confidence = round(fake_prob * 100.0, 2)
        model_used += " + PRNU Noise Steganalysis Override"

    verdict = "AUTHENTIC PHYSICS" if is_real else "PHYSICS ANOMALY (AI GENERATED)"
    
    # Overlay Plot
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
        "fake_probability": fake_prob,
        "srm_var_4": srm4_val,
        "srm_var_2": srm2_val,
        "ela_std": ela_val,
        "is_prnu_anomaly": is_prnu_anomaly,
        "model_used": model_used,
        "overlay": vis_copy,
        "vp": [round(best_vp[0], 1), round(best_vp[1], 1)] if best_vp else [0, 0]
    }
