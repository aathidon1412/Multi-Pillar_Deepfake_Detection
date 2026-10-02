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

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

PILLAR5_MODEL_NEW_PATH = os.path.join(P5_DIR, "pillar5_ml_model_v2.pkl")
PILLAR5_MODEL_FALLBACK_PATH = os.path.join(P5_DIR, "pillar5_ml_model.pkl")

# Global caches for instant inference
_CACHED_P5_BUNDLE = None
_CACHED_P5_BACKBONE = None
_CACHED_P5_DEV = None

def load_pillar5_ml_bundle():
    """Loads the dual-model ensemble bundle (pillar5_ml_model.pkl + pillar5_ml_model_v2.pkl)."""
    global _CACHED_P5_BUNDLE
    if _CACHED_P5_BUNDLE is not None:
        return _CACHED_P5_BUNDLE, "Dual-Model Ensemble (v1 + v2) (cached)"

    bundle1 = None
    bundle2 = None

    if os.path.exists(PILLAR5_MODEL_FALLBACK_PATH):
        try:
            with open(PILLAR5_MODEL_FALLBACK_PATH, "rb") as f:
                bundle1 = pickle.load(f)
        except Exception as e:
            print(f"[Pillar 5] Warning loading {PILLAR5_MODEL_FALLBACK_PATH}: {e}")

    if os.path.exists(PILLAR5_MODEL_NEW_PATH):
        try:
            with open(PILLAR5_MODEL_NEW_PATH, "rb") as f:
                bundle2 = pickle.load(f)
        except Exception as e:
            print(f"[Pillar 5] Warning loading {PILLAR5_MODEL_NEW_PATH}: {e}")

    if bundle1 is not None and bundle2 is not None:
        dual_bundle = {
            'type': 'dual_model_ensemble',
            'b1': bundle1,
            'b2': bundle2,
            'classifier': bundle1.get('classifier'),
            'scaler': bundle2.get('scaler'),
            'reducer': bundle2.get('reducer'),
            'scaler_tab': bundle1.get('scaler_tab'),
            'scaler_deep': bundle1.get('scaler_deep'),
            'feature_cols': bundle2.get('feature_cols', PHYSICS_FEATURE_NAMES),
            'backbone': 'efficientnet_b0'
        }
        _CACHED_P5_BUNDLE = dual_bundle
        print("[Pillar 5] Successfully loaded Dual-Model Ensemble (v1 + v2)")
        return dual_bundle, "Dual-Model Ensemble (v1 + v2)"
    elif bundle2 is not None:
        _CACHED_P5_BUNDLE = bundle2
        return bundle2, os.path.basename(PILLAR5_MODEL_NEW_PATH)
    elif bundle1 is not None:
        _CACHED_P5_BUNDLE = bundle1
        return bundle1, os.path.basename(PILLAR5_MODEL_FALLBACK_PATH)
    return None, "File not found"

_cached_p5_backbone = {}

def load_pillar5_deep_backbone(backbone_name='efficientnet_b0'):
    """Loads feature extractor backbone for Pillar 5 Deep Fusion (cached in memory)."""
    global _CACHED_P5_BACKBONE, _CACHED_P5_DEV
    if _CACHED_P5_BACKBONE is not None and _CACHED_P5_DEV is not None:
        return _CACHED_P5_BACKBONE, _CACHED_P5_DEV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = f"{backbone_name}_{device}"
    if key in _cached_p5_backbone:
        return _cached_p5_backbone[key], device

    if backbone_name == 'efficientnet_b0':
        bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        bb.classifier = torch.nn.Identity()
    else:
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        bb = torch.nn.Sequential(*list(bb.children())[:-1])
    bb.to(device).eval()
    _CACHED_P5_BACKBONE = bb
    _CACHED_P5_DEV = device
    _cached_p5_backbone[key] = bb
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
    tab_dict = {}
    features_for_clf = None
    
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
            bb, dev = load_pillar5_deep_backbone(p5_bundle.get('backbone', 'efficientnet_b0') if isinstance(p5_bundle, dict) else 'efficientnet_b0')
            prep = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            with torch.no_grad():
                target_pil = pil_img if pil_img is not None else Image.fromarray(image_np)
                t_in = prep(target_pil.convert('RGB')).unsqueeze(0).to(dev)
                emb = bb(t_in).squeeze().cpu().numpy().reshape(1, -1)
            
            bundle1 = None
            bundle2 = None
            if isinstance(p5_bundle, dict):
                if p5_bundle.get('type') == 'dual_model_ensemble':
                    bundle1 = p5_bundle.get('b1')
                    bundle2 = p5_bundle.get('b2')
                elif 'scaler_tab' in p5_bundle:
                    bundle1 = p5_bundle
                elif 'scaler' in p5_bundle and 'reducer' in p5_bundle:
                    bundle2 = p5_bundle

            if bundle1 is None and os.path.exists(PILLAR5_MODEL_FALLBACK_PATH):
                try:
                    with open(PILLAR5_MODEL_FALLBACK_PATH, "rb") as f:
                        bundle1 = pickle.load(f)
                except Exception:
                    pass
            if bundle2 is None and os.path.exists(PILLAR5_MODEL_NEW_PATH):
                try:
                    with open(PILLAR5_MODEL_NEW_PATH, "rb") as f:
                        bundle2 = pickle.load(f)
                except Exception:
                    pass

            rp1 = 0.50
            rp2 = 0.50

            # Predict Model 1 (Physics & Document Specialized Model)
            if bundle1 is not None and 'classifier' in bundle1 and 'scaler_tab' in bundle1:
                tab1_scaled = bundle1['scaler_tab'].transform(tab_vals)
                deep1_scaled = bundle1['scaler_deep'].transform(emb)
                fused1 = np.hstack([tab1_scaled, deep1_scaled])
                prob1 = bundle1['classifier'].predict_proba(fused1)[0]
                rp1 = float(prob1[1])  # b1: class 1 = Real

            # Predict Model 2 (2,000 Multi-Generator Diffusion Specialized Model)
            if bundle2 is not None and 'classifier' in bundle2 and 'scaler' in bundle2 and 'reducer' in bundle2:
                raw2 = np.hstack([tab_vals, emb])
                scaled2 = bundle2['scaler'].transform(raw2)
                reduced2 = bundle2['reducer'].transform(scaled2)
                prob2 = bundle2['classifier'].predict_proba(reduced2)[0]
                rp2 = float(prob2[1])  # b2: class 1 = Real
            elif bundle1 is not None:
                rp2 = rp1

        except Exception as e:
            is_real = (angular_variance_deg < 12.0 and max_inliers >= 20 and shadow_chroma_var >= 10.0)
            rp1 = 0.94 if is_real else 0.06
            rp2 = rp1
            model_used = f"RANSAC Shadow Physics (Fallback: {e})"
    else:
        is_real = (angular_variance_deg < 12.0 and max_inliers >= 20 and shadow_chroma_var >= 10.0)
        rp1 = 0.94 if is_real else 0.06
        rp2 = rp1
        model_used = "RANSAC Shadow Physics Fallback"

    srm4_val = float(tab_dict.get('srm_var_4', 0.0))
    srm2_val = float(tab_dict.get('srm_var_2', 0.0))
    ela_val = float(tab_dict.get('ela_std', 0.0))

    # Steganographic PRNU Residual Anomaly & Surrounding Environment Forensics
    # Synthetic diffusion models lack physical sensor shot noise.
    # Evaluated on un-interpolated raw pixel data to avoid bilinear/bicubic resizing artifacts.
    gray_raw = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    res4_raw = cv2.filter2D(gray_raw.astype(np.float32), -1, srm_filts[4])
    raw_srm4_var = float(np.var(res4_raw))

    # Surrounding Environment Margin Analysis (outer 18% margin vs central 64% focal region)
    margin_mask = np.ones((h, w), dtype=bool)
    margin_mask[int(h*0.18):int(h*0.82), int(w*0.18):int(w*0.82)] = False
    env_srm = float(np.var(res4_raw[margin_mask]))
    center_srm = float(np.var(res4_raw[~margin_mask]))
    noise_ratio = center_srm / (env_srm + 1e-4)

    # Domain Telemetry: Camera Hardware EXIF & Scanned Document Gating
    exif = pil_img.getexif() if (pil_img is not None and hasattr(pil_img, 'getexif')) else {}
    has_cam_exif = bool(exif.get(0x010f) or exif.get(0x0110) or exif.get(0x0131))
    
    white_ratio = float(np.mean(gray_norm > 220))
    text_edges = float(np.mean(cv2.Canny(gray_norm, 100, 200) > 0))
    is_true_doc = (white_ratio >= 0.70) or (white_ratio >= 0.40 and text_edges >= 0.065)

    # Environmental Forensics Evaluation
    env_notes = []
    is_env_forged = False

    if not has_cam_exif and not is_true_doc:
        if raw_srm4_var <= 0.75 or srm4_val <= 2.0:
            is_env_forged = True
            env_notes.append(f"Global PRNU diffusion noise floor deficit (var={raw_srm4_var:.2f} <= 0.75)")
        if env_srm <= 0.85 and noise_ratio >= 3.0:
            is_env_forged = True
            env_notes.append(f"Peripheral environment noise suppression with high focal disparity (disparity ratio={noise_ratio:.2f})")
        if quad_chroma_var >= 85.0 and raw_srm4_var <= 10.0:
            is_env_forged = True
            env_notes.append(f"Contradictory environmental quadrant illumination/chrominance (var={quad_chroma_var:.1f})")

    if not env_notes:
        env_notes.append("Environmental noise floor and ambient illumination consistent with natural physical optics.")

    env_verdict = "AI GENERATED / FORGED ENVIRONMENT ANOMALY" if is_env_forged else "AUTHENTIC PHYSICAL ENVIRONMENT"
    env_confidence = 94.0 if is_env_forged else 90.0

    # Decision Logic: Calibrated Dual-Model Ensemble
    is_authentic_sensor = (raw_srm4_var >= 150.0) or (rp1 >= 0.60 and raw_srm4_var >= 15.0 and rp2 >= 0.16)

    if has_cam_exif:
        fused_rp = max(rp1, rp2, 0.88)
        model_used = "Dual Ensemble + Camera Hardware Confirmed"
    elif is_true_doc:
        fused_rp = max(rp1, rp2, 0.85)
        model_used = "Dual Ensemble + Scanned Paper Document Domain"
    elif is_env_forged:
        fused_rp = min(rp1, rp2, 0.12)
        model_used = "Dual Ensemble + Environmental Anomaly Override"
    elif is_authentic_sensor:
        fused_rp = max(rp1, 0.60)
        model_used = "Dual Ensemble + Authentic Sensor Noise Protection"
    elif rp2 <= 0.25:
        fused_rp = rp2
        model_used = "Dual Ensemble (Model 2 High-Risk Fake Flag)"
    else:
        fused_rp = 0.50 * rp1 + 0.50 * rp2
        model_used = "Dual Ensemble Soft Voting"

    is_real = (fused_rp >= 0.33)
    fake_prob = 1.0 - fused_rp
    confidence = round((fused_rp * 100.0) if is_real else (fake_prob * 100.0), 2)
    verdict = "AUTHENTIC PHYSICS" if is_real else "PHYSICS ANOMALY (AI GENERATED)"

    # Overlay Plot
    vis_copy = image_np.copy()
    for l in best_inlier_lines:
        cv2.line(vis_copy, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 3)
    if best_vp:
        cv2.circle(vis_copy, (int(best_vp[0]), int(best_vp[1])), 10, (255, 0, 0), -1)
        
    res = {
        "is_real": is_real,
        "verdict": verdict,
        "confidence": confidence,
        "inliers": max_inliers,
        "total_lines": total_lines,
        "inlier_ratio": round(inlier_ratio, 3),
        "angular_var": round(angular_variance_deg, 2),
        "real_probability": fused_rp,
        "fake_probability": fake_prob,
        "srm_var_4": srm4_val,
        "raw_srm4_var": raw_srm4_var,
        "srm_var_2": srm2_val,
        "ela_std": ela_val,
        "is_prnu_anomaly": is_env_forged,
        "has_cam_exif": has_cam_exif,
        "is_doc": is_true_doc,
        "model_used": model_used,
        "overlay": vis_copy,
        "environment_analysis": {
            "is_environment_forged_or_synthetic": is_env_forged,
            "environment_verdict": env_verdict,
            "environment_confidence": env_confidence,
            "environment_noise_floor": round(env_srm, 4),
            "central_noise_floor": round(center_srm, 4),
            "subject_environment_noise_disparity": round(noise_ratio, 2),
            "quadrant_chroma_variance": round(quad_chroma_var, 2),
            "forensic_notes": "; ".join(env_notes)
        },
        "is_environment_synthetic": is_env_forged,
        "environment_verdict": env_verdict,
        "vp": [round(best_vp[0], 1), round(best_vp[1], 1)] if best_vp else [0, 0]
    }
    try:
        from .xai import generate_pillar5_xai
        res["xai"] = generate_pillar5_xai(res, features_for_clf=features_for_clf, tab_dict=tab_dict, p5_bundle=p5_bundle)
    except Exception as xai_err:
        print(f"[Pillar 5] XAI extraction notice: {xai_err}")
    return res

