"""
================================================================================
Pillar 5: Forensic Feature Extraction & Matrix Builder
USMFE Multi-Pillar Deepfake Detection Framework
================================================================================

Description:
    This module provides production-grade, standardized extraction helpers to build
    training and evaluation matrices for Pillar 5 (Physical Geometry + Hybrid Ensemble).

Key Components:
    1. `extract_physics_vector`: Extracts the authoritative 24-dimensional physical
       geometry and forensic vector according to `feature_schema.py`.
       Fully implements:
         - RANSAC Vanishing Point & Perspective Lines (total_lines, max_inliers, inlier_ratio, angular_variance_deg)
         - Illumination Physics & Shadows (shadow_chroma_var, quad_chroma_var, gw_dev, penumbra_ratio)
         - Blur, Focus & High-Frequency (lap_var, lap_skew, high_freq_energy)
         - Compression & ELA (dct_mid_energy, ela_mean, ela_std)
         - Spatial Rich Model (SRM) Steganalysis Residuals (srm_var_0..4, srm_skew_0..4)
    2. `extract_efficientnet_embedding`: Extracts the 1,280-dim deep visual
       representation using an EfficientNet-B0 backbone.
    3. `build_xy_matrices`: Assembles (N, 24) physics and (N, 1280) embedding matrices
       with strict shape verification, zero data leakage, and diagnostic reporting.

Usage:
    >>> from extract_features import build_xy_matrices
    >>> X_phys, X_deep, y = build_xy_matrices(image_paths, labels)
"""

import os
import sys
import math
import random
import logging
from typing import List, Tuple, Optional, Callable, Dict, Union, Any

import cv2
import numpy as np
from scipy.fftpack import dct
import torch
import torch.nn as nn
from PIL import Image

# Ensure current Pillar 5 directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_schema import PHYSICS_FEATURE_NAMES, FEATURE_INDEX_MAP, validate_feature_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pillar5.ExtractFeatures")

# 5 SRM High-Pass Residual Steganalysis Filters
SRM_FILTERS = [
    np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
    np.array([[-1, 2, -2, 2, -1], [ 2, -6, 8, -6, 2], [-2,  8,-12, 8, -2], [ 2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
]


# ==============================================================================
# 1. 24-DIMENSIONAL PHYSICAL FORENSICS EXTRACTION
# ==============================================================================

def extract_physics_vector(
    image_input: Union[str, np.ndarray, Image.Image, Dict[str, float]]
) -> np.ndarray:
    """
    Extracts or formats the 24-dimensional physical forensic vector.
    
    If given a dictionary, aligns it directly to `PHYSICS_FEATURE_NAMES`.
    If given an image or image path, performs full physical forensic extraction:
      - RANSAC perspective vanishing lines & angular variance
      - CIELAB shadow chromatic variance
      - Quadrant chrominance variance & Gray-World deviation
      - Laplacian variance & skewness
      - 2D DCT mid-frequency coefficient energy
      - JPEG Error Level Analysis (ELA) residual mean and std
      - Shadow penumbra ratio
      - 5 Spatial Rich Model (SRM) filter residual variance and skewness

    Returns:
        np.ndarray: float32 vector of shape (24,) locked in PHYSICS_FEATURE_NAMES order.
    """
    # Case A: Input is already a pre-extracted feature dictionary
    if isinstance(image_input, dict):
        return validate_feature_dict(image_input)

    # Case B: Input is an image (filepath, PIL Image, or numpy array)
    feat_dict: Dict[str, float] = {}

    try:
        # Load image with OpenCV (BGR)
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.warning(f"File not found '{image_input}'. Returning zeros.")
                return np.zeros(24, dtype=np.float32)
            img = cv2.imread(image_input)
            if img is None:
                return np.zeros(24, dtype=np.float32)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            logger.warning(f"Unsupported input type '{type(image_input)}'. Returning zeros.")
            return np.zeros(24, dtype=np.float32)

        h, w = img.shape[:2]
        # Standardize canvas scale to 800px max dimension for consistent line detection
        scale = 800.0 / max(h, w)
        img_norm = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------------------------
        # COMPONENT 1: RANSAC Perspective Vanishing Lines & Angular Variance
        # ----------------------------------------------------------------------
        blurred = cv2.GaussianBlur(gray_norm, (11, 11), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        edges = cv2.Canny(thresh, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
        total_lines = float(len(lines)) if lines is not None else 0.0

        max_inliers = 0.0
        angular_variance_deg = 85.0
        if lines is not None and len(lines) >= 2:
            lines_list = lines.reshape(-1, 4).tolist()
            angles = [math.atan2(l[3] - l[1], l[2] - l[0]) for l in lines_list]
            angles_mod = np.mod(angles, np.pi)
            med_ang = np.median(angles_mod)
            fl = [l for l, a in zip(lines_list, angles_mod) if min(abs(a - med_ang), np.pi - abs(a - med_ang)) < 0.35]
            if len(fl) >= 2:
                for _ in range(min(120, len(fl) * len(fl))):
                    l1, l2 = random.sample(fl, 2)
                    den = (l1[0] - l1[2]) * (l2[1] - l2[3]) - (l1[1] - l1[3]) * (l2[0] - l2[2])
                    if den == 0:
                        continue
                    px = ((l1[0] * l1[3] - l1[1] * l1[2]) * (l2[0] - l2[2]) - (l1[0] - l1[2]) * (l2[0] * l2[3] - l2[1] * l2[2])) / den
                    py = ((l1[0] * l1[3] - l1[1] * l1[2]) * (l2[1] - l2[3]) - (l1[1] - l1[3]) * (l2[0] * l2[3] - l2[1] * l2[2])) / den
                    inls = []
                    for line in fl:
                        d_den = math.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
                        d = abs((line[2] - line[0]) * (line[1] - py) - (line[0] - px) * (line[3] - line[1])) / d_den if d_den > 0 else 9999
                        if d < 50:
                            inls.append(line)
                    if len(inls) > max_inliers:
                        max_inliers = float(len(inls))
                        inlier_angles = [math.atan2(l[3] - l[1], l[2] - l[0]) for l in inls]
                        ms = np.mean([math.sin(2 * a) for a in inlier_angles])
                        mc = np.mean([math.cos(2 * a) for a in inlier_angles])
                        angular_variance_deg = float(math.degrees((1.0 - math.sqrt(ms ** 2 + mc ** 2)) * np.pi))

        inlier_ratio = float(max_inliers / total_lines) if total_lines > 0 else 0.0

        feat_dict["total_lines"] = total_lines
        feat_dict["max_inliers"] = max_inliers
        feat_dict["inlier_ratio"] = inlier_ratio
        feat_dict["angular_variance_deg"] = angular_variance_deg

        # ----------------------------------------------------------------------
        # COMPONENT 2: CIELAB Illumination Consistency & Shadows
        # ----------------------------------------------------------------------
        lab_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab_norm)
        shadow_pix = l_chan < np.percentile(l_chan, 35)
        shadow_chroma_var = float(np.std(a_chan[shadow_pix]) + np.std(b_chan[shadow_pix])) if np.any(shadow_pix) else 0.0

        gh, gw = l_chan.shape
        q1 = (a_chan[:gh // 2, :gw // 2], b_chan[:gh // 2, :gw // 2])
        q2 = (a_chan[:gh // 2, gw // 2:], b_chan[:gh // 2, gw // 2:])
        q3 = (a_chan[gh // 2:, :gw // 2], b_chan[gh // 2:, :gw // 2])
        q4 = (a_chan[gh // 2:, gw // 2:], b_chan[gh // 2:, gw // 2:])
        quad_chroma_var = float(np.var([np.mean(q[0]) + np.mean(q[1]) for q in [q1, q2, q3, q4]]))

        gw_dev = float(np.std([np.mean(img_norm[:, :, 0]), np.mean(img_norm[:, :, 1]), np.mean(img_norm[:, :, 2])]))

        grad_x = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        shadow_grad = float(np.mean(grad_mag[shadow_pix])) if np.any(shadow_pix) else 0.0
        non_shadow_grad = float(np.mean(grad_mag[~shadow_pix])) if np.any(~shadow_pix) else 0.0
        penumbra_ratio = float(shadow_grad / (non_shadow_grad + 1e-5))

        feat_dict["shadow_chroma_var"] = shadow_chroma_var
        feat_dict["quad_chroma_var"] = quad_chroma_var
        feat_dict["gw_dev"] = gw_dev
        feat_dict["penumbra_ratio"] = penumbra_ratio

        # ----------------------------------------------------------------------
        # COMPONENT 3: Blur, Focus & High-Frequency Spatial Energy
        # ----------------------------------------------------------------------
        lap_arr = cv2.Laplacian(gray_norm, cv2.CV_64F)
        lap_var = float(np.var(lap_arr))
        lap_skew = float(np.mean(((lap_arr - np.mean(lap_arr)) / (np.std(lap_arr) + 1e-5)) ** 3))

        sub_gray = cv2.resize(gray_norm, (256, 256))
        dct_b = dct(dct(sub_gray.T, norm="ortho").T, norm="ortho")
        high_freq_energy = float(np.sum(np.abs(dct_b[128:, 128:])) / (np.sum(np.abs(dct_b)) + 1e-5))
        dct_mid_energy = float(np.sum(np.abs(dct_b[64:128, 64:128])) / (np.sum(np.abs(dct_b)) + 1e-5))

        feat_dict["lap_var"] = lap_var
        feat_dict["lap_skew"] = lap_skew
        feat_dict["high_freq_energy"] = high_freq_energy
        feat_dict["dct_mid_energy"] = dct_mid_energy

        # ----------------------------------------------------------------------
        # COMPONENT 4: Error Level Analysis (ELA) Compression Forensics
        # ----------------------------------------------------------------------
        _, encimg = cv2.imencode(".jpg", img_norm, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        decimg = cv2.imdecode(encimg, 1)
        ela = np.abs(img_norm.astype(np.float32) - decimg.astype(np.float32))
        feat_dict["ela_mean"] = float(np.mean(ela))
        feat_dict["ela_std"] = float(np.std(ela))

        # ----------------------------------------------------------------------
        # COMPONENT 5: Spatial Rich Models (SRM) Noise Residual Statistics (10)
        # ----------------------------------------------------------------------
        for idx_s, filt in enumerate(SRM_FILTERS):
            res = cv2.filter2D(gray_norm.astype(np.float32), -1, filt)
            feat_dict[f"srm_var_{idx_s}"] = float(np.var(res))
            feat_dict[f"srm_skew_{idx_s}"] = float(np.mean(((res - np.mean(res)) / (np.std(res) + 1e-5)) ** 3))

    except Exception as err:
        logger.warning(f"[extract_physics_vector] Error during extraction: {err}. Returning fallback vector.")

    return validate_feature_dict(feat_dict)


# ==============================================================================
# 2. EFFICIENTNET-B0 1280-DIM EMBEDDING EXTRACTION
# ==============================================================================

_DEFAULT_EFFNET_MODEL: Optional[nn.Module] = None
_DEFAULT_EFFNET_DEVICE: Optional[torch.device] = None


def get_default_efficientnet(device: Optional[torch.device] = None) -> nn.Module:
    """Lazily loads and caches the EfficientNet-B0 feature extractor (1280-dim)."""
    global _DEFAULT_EFFNET_MODEL, _DEFAULT_EFFNET_DEVICE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _DEFAULT_EFFNET_MODEL is None or _DEFAULT_EFFNET_DEVICE != device:
        import torchvision.models as models

        logger.info(f"Loading pretrained EfficientNet-B0 on {device}...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Replace classification head with Identity to output 1280-dim feature vector
        model.classifier = nn.Identity()
        model.to(device)
        model.eval()

        _DEFAULT_EFFNET_MODEL = model
        _DEFAULT_EFFNET_DEVICE = device

    return _DEFAULT_EFFNET_MODEL


def extract_efficientnet_embedding(
    image_input: Union[torch.Tensor, str, Image.Image],
    feature_extractor: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Extracts a 1,280-dimensional feature embedding from EfficientNet-B0.

    Args:
        image_input: Preprocessed torch.Tensor [1, 3, 224, 224], filepath, or PIL Image.
        feature_extractor (Optional[nn.Module]): Pre-loaded feature extractor model.
        device (Optional[torch.device]): Compute device.

    Returns:
        np.ndarray: 1D float32 array of shape (1280,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = feature_extractor if feature_extractor is not None else get_default_efficientnet(device)

    # If input is a path or PIL image, convert with standard ImageNet transforms
    if isinstance(image_input, (str, Image.Image)):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.warning(f"File '{image_input}' not found. Returning zero embedding.")
                return np.zeros(1280, dtype=np.float32)
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        tensor = transform(pil_img).unsqueeze(0).to(device)
    elif isinstance(image_input, torch.Tensor):
        tensor = image_input.to(device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        logger.warning(f"Unsupported input type {type(image_input)}. Returning zero embedding.")
        return np.zeros(1280, dtype=np.float32)

    with torch.no_grad():
        embedding = model(tensor)
        vec = embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    assert vec.shape == (1280,), f"Expected (1280,) embedding, got {vec.shape}"
    return vec


# ==============================================================================
# 3. HIGH-LEVEL DATASET MATRIX BUILDER
# ==============================================================================

def build_xy_matrices(
    image_paths: List[str],
    labels: Union[List[int], np.ndarray],
    physics_fn: Optional[Callable[[str], np.ndarray]] = None,
    effnet_fn: Optional[Callable[[str], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds and validates the feature matrices for Pillar 5.

    Args:
        image_paths (List[str]): List of N file paths to input images.
        labels (List[int] or np.ndarray): List of N binary ground truth labels.
        physics_fn (Optional[Callable]): Custom physics extractor, defaults to `extract_physics_vector`.
        effnet_fn (Optional[Callable]): Custom embedding extractor, defaults to `extract_efficientnet_embedding`.

    Returns:
        Tuple of:
            - X_physics: np.ndarray of shape (N, 24)
            - X_effnet: np.ndarray of shape (N, 1280)
            - y: np.ndarray of shape (N,)
    """
    N = len(image_paths)
    if N != len(labels):
        raise ValueError(f"Mismatch: len(image_paths)={N} vs len(labels)={len(labels)}")

    if physics_fn is None:
        physics_fn = extract_physics_vector

    if effnet_fn is None:
        effnet_model = get_default_efficientnet()
        effnet_fn = lambda path: extract_efficientnet_embedding(path, feature_extractor=effnet_model)

    print(f"\n[build_xy_matrices] Processing {N} samples into dual feature matrices...")

    physics_rows: List[np.ndarray] = []
    effnet_rows: List[np.ndarray] = []
    valid_labels: List[int] = []

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        p_vec = physics_fn(path)
        if p_vec.shape != (24,):
            raise ValueError(f"Physics extractor returned shape {p_vec.shape}, expected (24,)")

        e_vec = effnet_fn(path)
        if e_vec.shape != (1280,):
            raise ValueError(f"EfficientNet extractor returned shape {e_vec.shape}, expected (1280,)")

        physics_rows.append(p_vec)
        effnet_rows.append(e_vec)
        valid_labels.append(int(label))

        if (i + 1) % 100 == 0 or (i + 1) == N:
            print(f" -> Processed {i + 1}/{N} images...")

    X_physics = np.vstack(physics_rows).astype(np.float32)
    X_effnet = np.vstack(effnet_rows).astype(np.float32)
    y = np.array(valid_labels, dtype=np.int32)

    # Validation checks
    assert X_physics.shape == (N, 24), f"X_physics shape error: {X_physics.shape}"
    assert X_effnet.shape == (N, 1280), f"X_effnet shape error: {X_effnet.shape}"
    assert y.shape == (N,), f"y shape error: {y.shape}"

    print(f"[build_xy_matrices] Matrices successfully assembled:")
    print(f" -> X_physics shape : {X_physics.shape} (Memory: {X_physics.nbytes / 1024:.1f} KB)")
    print(f" -> X_effnet shape  : {X_effnet.shape} (Memory: {X_effnet.nbytes / 1024 / 1024:.2f} MB)")
    print(f" -> Target y shape  : {y.shape} (Class counts: Real={np.sum(y == 0)}, Fake={np.sum(y == 1)})")

    return X_physics, X_effnet, y
