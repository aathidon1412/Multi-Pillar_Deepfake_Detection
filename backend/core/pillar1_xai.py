"""
================================================================================
Pillar 1 Explainable AI (XAI): Vision Transformer Attention Rollout & Heatmaps
================================================================================
Implements:
1. Attention Rollout (Abnar & Zuidema, 2020) across all ViT transformer layers
2. 2D Spatial Token Projection (196 tokens -> 14x14 grid -> Image Dimensions)
3. OpenCV Colormap Generation & Alpha-Blended Visual Overlays
4. Peak Spatial Attribution Region Localization (Quadrant & Focal Mapping)
5. Zero-Database Storage & Base64 Data URI Serialization
6. Fact-Grounded Human-Readable and Researcher Explanations
"""

import os
import uuid
import math
import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F

from .xai import create_xai_evidence, create_xai_response, create_fallback_xai_response

# Storage directory resolution for API static serving
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "Pillar 2", "video-authenticity-detector", "storage", "suspicious_frames")


def compute_attention_rollout(attentions: tuple, discard_ratio: float = 0.05, head_fusion: str = "mean") -> np.ndarray:
    """
    Computes Attention Rollout across ViT multi-head attention layers.
    
    Args:
        attentions: Tuple of tensors from ViT output, each [batch_size, num_heads, seq_len, seq_len]
        discard_ratio: Bottom percentage of attention values to prune for noise reduction (0.05 = 5%)
        head_fusion: How to fuse multi-head attention ('mean', 'max', 'min')
    
    Returns:
        1D numpy array of length 196 representing CLS-to-patch attention tokens.
    """
    with torch.no_grad():
        device = attentions[0].device
        seq_len = attentions[0].size(-1) # 197 for ViT-Base (1 CLS + 196 patches)
        result = torch.eye(seq_len, device=device)

        for attention in attentions:
            # 1. Fuse attention heads
            if head_fusion == "mean":
                attn_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attn_fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attn_fused = attention.min(dim=1)[0]
            else:
                attn_fused = attention.mean(dim=1)

            # 2. Prune low-attention noise if discard_ratio > 0
            if discard_ratio > 0:
                flat = attn_fused.view(attn_fused.size(0), -1)
                k = int(flat.size(-1) * discard_ratio)
                if k > 0:
                    _, indices = flat.topk(k, dim=-1, largest=False)
                    flat[0, indices] = 0
                    attn_fused = flat.view(attn_fused.shape)

            # 3. Add Identity matrix for residual skip-connections
            I = torch.eye(seq_len, device=device).unsqueeze(0)
            a = (attn_fused + I) / 2.0
            # Re-normalize rows to sum to 1
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)

        # 4. Extract attention from [CLS] token (index 0) to all spatial patch tokens (indices 1..196)
        cls_attn = result[0, 0, 1:].cpu().numpy()
        return cls_attn


def identify_highest_attribution_region(heatmap_14x14: np.ndarray) -> dict:
    """
    Analyzes the 14x14 patch attention grid to locate the peak focal region
    without hardcoding facial assumptions.
    """
    grid_size = heatmap_14x14.shape[0] # 14
    # Focus on top 15% most activated patches
    threshold = np.percentile(heatmap_14x14, 85)
    y_idx, x_idx = np.where(heatmap_14x14 >= threshold)
    
    if len(y_idx) == 0:
        mean_y, mean_x = grid_size / 2.0, grid_size / 2.0
    else:
        mean_y = float(np.mean(y_idx))
        mean_x = float(np.mean(x_idx))

    # Normalized coordinates [0.0, 1.0]
    norm_y = mean_y / (grid_size - 1)
    norm_x = mean_x / (grid_size - 1)

    # Vertical descriptor
    if norm_y < 0.35:
        vert = "upper"
    elif norm_y > 0.65:
        vert = "lower"
    else:
        vert = "central"

    # Horizontal descriptor
    if norm_x < 0.35:
        horiz = "left"
    elif norm_x > 0.65:
        horiz = "right"
    else:
        horiz = "center"

    if vert == "central" and horiz == "center":
        region_desc = "central focal region"
    elif vert == "central":
        region_desc = f"mid-{horiz} region"
    elif horiz == "center":
        region_desc = f"{vert}-center region"
    else:
        region_desc = f"{vert}-{horiz} quadrant"

    # Peak attribution score
    peak_score = float(np.max(heatmap_14x14))
    mean_score = float(np.mean(heatmap_14x14))
    concentration_ratio = float(peak_score / (mean_score + 1e-8))

    return {
        "region_description": region_desc,
        "peak_norm_coords": [round(norm_x, 3), round(norm_y, 3)],
        "peak_score": round(peak_score, 4),
        "concentration_ratio": round(concentration_ratio, 2)
    }


def generate_pillar1_attention_xai(
    model,
    pixel_values: torch.Tensor,
    orig_pil: Image.Image,
    p1_prediction: str,
    p1_confidence: float,
    is_real: bool,
    real_prob: float,
    fake_prob: float
) -> Dict[str, Any]:
    """
    Executes full Attention Rollout XAI on Vision Transformer inference.
    
    Returns:
        Structured canonical XAI payload including overlay image, raw heatmap,
        base64 data URIs, peak attribution localization, and plain-English explanation.
    """
    try:
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        # 1. Forward pass with attention extraction (weights remain untouched)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, output_attentions=True)
            attentions = outputs.attentions

        if not attentions or len(attentions) == 0:
            return create_fallback_xai_response(
                "Pillar 1: Vision Transformer",
                p1_prediction,
                p1_confidence,
                "Model did not return attention weights"
            )

        # 2. Attention Rollout
        cls_attn_tokens = compute_attention_rollout(attentions, discard_ratio=0.05, head_fusion="mean")
        grid_size = int(math.sqrt(len(cls_attn_tokens))) # 14 for 196 patches
        heatmap_14x14 = cls_attn_tokens.reshape(grid_size, grid_size)

        # 3. Min-Max Normalization
        h_min, h_max = heatmap_14x14.min(), heatmap_14x14.max()
        heatmap_norm = (heatmap_14x14 - h_min) / (h_max - h_min + 1e-8)

        # 4. Regional localization & peak metrics
        region_info = identify_highest_attribution_region(heatmap_norm)
        highest_region = region_info["region_description"]

        # 5. Spatial Resizing to original image dimensions
        orig_w, orig_h = orig_pil.size
        heatmap_resized = cv2.resize(heatmap_norm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # 6. Colormap Application (JET / Turbo for high contrast)
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
        orig_np_rgb = np.array(orig_pil.convert("RGB"))

        # 7. Alpha-Blended Overlay
        alpha = 0.52
        overlay_np_rgb = np.uint8(alpha * heatmap_colored_rgb + (1.0 - alpha) * orig_np_rgb)

        # 8. Save Images to Storage
        xai_id = f"p1_vit_rollout_{uuid.uuid4().hex[:8]}"
        overlay_filename = f"{xai_id}_overlay.jpg"
        heatmap_filename = f"{xai_id}_heatmap.jpg"
        orig_filename    = f"{xai_id}_orig.jpg"

        overlay_path = os.path.join(STORAGE_DIR, overlay_filename)
        heatmap_path = os.path.join(STORAGE_DIR, heatmap_filename)
        orig_path    = os.path.join(STORAGE_DIR, orig_filename)

        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_np_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(heatmap_path, heatmap_colored_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(orig_path, cv2.cvtColor(orig_np_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 9. Base64 Data URI for instant browser rendering without HTTP lag
        def np_to_base64_data_uri(img_rgb):
            pil_i = Image.fromarray(img_rgb)
            buf = io.BytesIO()
            pil_i.save(buf, format="JPEG", quality=88)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"

        overlay_data_uri = np_to_base64_data_uri(overlay_np_rgb)
        heatmap_data_uri = np_to_base64_data_uri(heatmap_colored_rgb)
        orig_data_uri    = np_to_base64_data_uri(orig_np_rgb)

        # 10. Construct Evidence Objects
        evidence = []
        direction = "supports_fake" if not is_real else "supports_real"
        
        evidence.append(create_xai_evidence(
            feature="highest_attribution_region",
            value=highest_region,
            direction=direction,
            importance=0.92,
            description=f"Highest transformer attention concentration observed in the {highest_region} (Attribution Peak: {region_info['peak_score']:.2f}, concentration ratio: {region_info['concentration_ratio']:.1f}x)."
        ))

        evidence.append(create_xai_evidence(
            feature="attention_rollout_dispersion",
            value=region_info["concentration_ratio"],
            direction=direction,
            importance=0.85,
            description=f"Multi-layer attention flow mapped across 12 transformer encoder blocks (14x14 spatial patches)."
        ))

        evidence.append(create_xai_evidence(
            feature="target_class_posterior",
            value=round(fake_prob if not is_real else real_prob, 4),
            direction=direction,
            importance=0.88,
            description=f"Final ViT classification posterior towards target class '{p1_prediction}' with {p1_confidence:.1f}% confidence."
        ))

        # 11. Human & Technical Explanations (Strictly grounded in evidence)
        target_label = "FAKE (AI Generated)" if not is_real else "AUTHENTIC (Real)"
        human_explanation = (
            f"Visual explanation: The model's {p1_prediction} prediction ({p1_confidence:.1f}% confidence) "
            f"was mainly influenced by high-attribution regions around the {highest_region}. "
            f"The highlighted areas represent image patches that received higher model attention for the selected prediction."
        )

        tech_explanation = (
            f"Attention Rollout across 12 transformer layers (16x16 patch resolution, {grid_size}x{grid_size} grid) "
            f"identified peak token saliency at normalized coordinates (X={region_info['peak_norm_coords'][0]}, Y={region_info['peak_norm_coords'][1]}). "
            f"Classification posterior P({target_label}) = {max(real_prob, fake_prob):.4f} with token attention concentration ratio of {region_info['concentration_ratio']:.2f}x above uniform baseline."
        )

        visualizations = [
            {
                "type": "attention_rollout_overlay",
                "title": "ViT Attention Rollout Heatmap Overlay",
                "overlay_url": f"suspicious_frames/{overlay_filename}",
                "heatmap_url": f"suspicious_frames/{heatmap_filename}",
                "orig_url": f"suspicious_frames/{orig_filename}",
                "overlay_data_uri": overlay_data_uri,
                "heatmap_data_uri": heatmap_data_uri,
                "orig_data_uri": orig_data_uri,
                "colormap": "JET",
                "grid_size": [14, 14],
                "peak_region": highest_region,
                "peak_coords": region_info["peak_norm_coords"],
                "disclaimer": "Highlighted regions indicate model attribution and should be interpreted as supporting evidence, not proof of manipulation."
            }
        ]

        limitations = [
            "Attention rollout reflects patch token flow and may highlight background contours if high contrast is present.",
            "Visual heatmap highlights supporting model attribution and should not be used as standalone proof of forgery."
        ]

        resp = create_xai_response(
            pillar="Pillar 1: Vision Transformer",
            prediction=p1_prediction,
            confidence=p1_confidence,
            evidence=evidence,
            visualizations=visualizations,
            human_explanation=human_explanation,
            technical_explanation=tech_explanation,
            limitations=limitations,
            xai_available=True
        )
        # Add convenient top-level accessors
        resp.update({
            "method": "Attention Rollout",
            "target_class": p1_prediction,
            "heatmap_url": f"suspicious_frames/{heatmap_filename}",
            "overlay_url": f"suspicious_frames/{overlay_filename}",
            "orig_url": f"suspicious_frames/{orig_filename}",
            "heatmap_data_uri": heatmap_data_uri,
            "overlay_data_uri": overlay_data_uri,
            "orig_data_uri": orig_data_uri,
            "highest_attribution_region": highest_region,
            "attribution_summary": evidence,
            "disclaimer": "Highlighted regions indicate model attribution and should be interpreted as supporting evidence, not proof of manipulation."
        })
        return resp

    except Exception as e:
        import traceback
        print(f"[Pillar 1 XAI Error] Failed to generate attention rollout: {e}")
        traceback.print_exc()
        return create_fallback_xai_response(
            "Pillar 1: Vision Transformer",
            p1_prediction,
            p1_confidence,
            str(e)
        )
