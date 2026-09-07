"""
================================================================================
Pillar 5: Physical Geometry Feature Schema & Alignment Specification
USMFE Multi-Pillar Deepfake Detection Framework
================================================================================

Description:
    This module defines the authoritative, locked schema for the 24 handcrafted
    physical and forensic geometry features utilized in Pillar 5.

    Maintaining this exact ordering is strictly mandatory across:
    1. Feature extraction routines
    2. Tabular caching / CSV persistence
    3. Scaler transformation (`StandardScaler` / `RobustScaler`)
    4. Feature selection (`SelectFromModel` / `PCA`)
    5. Multi-model ensemble inference (LightGBM, XGBoost, Random Forest, Extra Trees)

Feature Taxonomy (24 Features Total):
    - Geometric Perspective (4) : Vanishing point consistency and angular variance.
    - Illumination Physics  (4) : Chrominance consistency, Gray-World, and shadow penumbra.
    - Blur & Edge Sharpness (3) : Laplacian variance, gradient skewness, high-frequency energy.
    - Compression & Freq    (3) : DCT mid-band frequency energy and Error Level Analysis (ELA).
    - Steganographic SRM   (10) : Variance and skewness across 5 SRM high-pass noise filters.

Usage:
    >>> from feature_schema import PHYSICS_FEATURE_NAMES, validate_feature_dict
    >>> vector = validate_feature_dict(extracted_dict)
    >>> print(vector.shape)  # (24,)
"""

from typing import List, Dict
import numpy as np


# ==============================================================================
# AUTHORITATIVE 24-DIMENSIONAL FEATURE SCHEMA
# ==============================================================================

PHYSICS_FEATURE_NAMES: List[str] = [
    # --- Group 1: Geometric Perspective & Vanishing Point Geometry (4) ---
    "total_lines",           # Total line segments extracted via probabilistic Hough transform
    "max_inliers",           # Max line support converging to dominant perspective vanishing point
    "inlier_ratio",          # Ratio of converging lines to total detected lines (max_inliers / total_lines)
    "angular_variance_deg",  # Angular dispersion of 2D line vectors in degrees

    # --- Group 2: Illumination & Color Consistency (4) ---
    "shadow_chroma_var",     # Color variance across identified shadow / penumbra boundaries
    "quad_chroma_var",       # Quadrant-level chroma variance (detects composite face/body lighting)
    "gw_dev",                # Gray-World color constancy deviation (divergence between RGB channels)
    "penumbra_ratio",        # Shadow edge gradient width (ratio between soft penumbra and umbra)

    # --- Group 3: Blur, Focus & High-Frequency Spatial Energy (3) ---
    "lap_var",               # Laplacian operator variance (edge sharpness / focus metric)
    "lap_skew",              # Skewness of Laplacian gradient distribution
    "high_freq_energy",      # Ratio of high-frequency energy in gradient domain

    # --- Group 4: Frequency & Compression Forensics (3) ---
    "dct_mid_energy",        # Discrete Cosine Transform (DCT) energy in mid-frequency spectral band
    "ela_mean",              # Error Level Analysis (ELA) mean residual from recompressed JPEG
    "ela_std",               # Error Level Analysis (ELA) standard deviation of residuals

    # --- Group 5: Spatial Rich Models (SRM) Noise Residuals (10) ---
    "srm_var_0",             # Filter 0 (1st-order horizontal residual): Variance
    "srm_skew_0",            # Filter 0 (1st-order horizontal residual): Skewness
    "srm_var_1",             # Filter 1 (1st-order vertical residual): Variance
    "srm_skew_1",            # Filter 1 (1st-order vertical residual): Skewness
    "srm_var_2",             # Filter 2 (2nd-order cross residual): Variance
    "srm_skew_2",            # Filter 2 (2nd-order cross residual): Skewness
    "srm_var_3",             # Filter 3 (3x3 Laplacian edge residual): Variance
    "srm_skew_3",            # Filter 3 (3x3 Laplacian edge residual): Skewness
    "srm_var_4",             # Filter 4 (5x5 square edge residual): Variance
    "srm_skew_4"             # Filter 4 (5x5 square edge residual): Skewness
]

# Strict validation: Exactly 24 features
assert len(PHYSICS_FEATURE_NAMES) == 24, (
    f"FATAL SCHEMA ERROR: PHYSICS_FEATURE_NAMES must contain exactly 24 features, "
    f"found {len(PHYSICS_FEATURE_NAMES)}."
)

# Logical taxonomy map for explainability and inspection
FEATURE_GROUPS: Dict[str, List[str]] = {
    "geometric_perspective": [
        "total_lines", "max_inliers", "inlier_ratio", "angular_variance_deg"
    ],
    "illumination_consistency": [
        "shadow_chroma_var", "quad_chroma_var", "gw_dev", "penumbra_ratio"
    ],
    "blur_and_focus": [
        "lap_var", "lap_skew", "high_freq_energy"
    ],
    "compression_and_dct": [
        "dct_mid_energy", "ela_mean", "ela_std"
    ],
    "srm_noise_residuals": [
        "srm_var_0", "srm_skew_0", "srm_var_1", "srm_skew_1",
        "srm_var_2", "srm_skew_2", "srm_var_3", "srm_skew_3",
        "srm_var_4", "srm_skew_4"
    ]
}

# Lookup dictionary from feature name to zero-based vector index
FEATURE_INDEX_MAP: Dict[str, int] = {
    name: idx for idx, name in enumerate(PHYSICS_FEATURE_NAMES)
}


def validate_feature_dict(feat_dict: Dict[str, float]) -> np.ndarray:
    """
    Converts a feature dictionary into a 24-dimensional numpy vector
    guaranteed to match the exact schema order.

    Missing values are filled with 0.0, and infinite/NaN values are clipped.

    Args:
        feat_dict (Dict[str, float]): Key-value map of extracted forensic metrics.

    Returns:
        np.ndarray: 1D float32 array of shape (24,).
    """
    vector = np.zeros(24, dtype=np.float32)

    for idx, name in enumerate(PHYSICS_FEATURE_NAMES):
        val = feat_dict.get(name, 0.0)
        # Sanitize non-finite values
        if val is None or np.isnan(val) or np.isinf(val):
            val = 0.0
        vector[idx] = float(val)

    return vector


# ==============================================================================
# USAGE DEMONSTRATION / VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PILLAR 5: FEATURE SCHEMA VERIFICATION")
    print("=" * 70)
    print(f"Total defined features : {len(PHYSICS_FEATURE_NAMES)}")
    print(f"Taxonomy groups        : {list(FEATURE_GROUPS.keys())}")
    print("\nOrdered Feature Index Mapping:")
    for idx, name in enumerate(PHYSICS_FEATURE_NAMES):
        print(f"  [{idx:02d}] {name}")

    # Test dictionary alignment
    dummy_input = {
        "lap_var": 142.5,
        "gw_dev": 0.034,
        "total_lines": 88,
        "srm_var_0": 1.25,
        "unknown_extra_feature": 999.0  # Should be safely ignored
    }
    aligned_vec = validate_feature_dict(dummy_input)
    assert aligned_vec.shape == (24,), "Vector shape mismatch!"
    assert aligned_vec[FEATURE_INDEX_MAP["lap_var"]] == 142.5, "Index alignment failed!"
    assert aligned_vec[FEATURE_INDEX_MAP["gw_dev"]] == 0.034, "Index alignment failed!"
    print("\nSUCCESS: Feature schema alignment and validation verified!")
    print("=" * 70)
