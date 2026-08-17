# UNIVERSAL SYNTHETIC MEDIA FORENSICS ENGINE (USMFE)
## PILLAR 5: PHYSICAL GEOMETRY & SHADOW PHYSICS FORENSICS
### Multi-Iteration Accuracy Optimization & Experimental Benchmark Report

---

## 1. Executive Summary & Benchmark Overview

This report documents the systematic multi-iteration research, feature engineering, and ensemble optimization conducted on **Pillar 5 (Physical Geometry & Shadow Physics Forensics)** within the Multi-Pillar Deepfake Detection Framework.

Evaluations were performed across a balanced multi-generator dataset comprising **1,994 images** (**1,000 Authentic Photographs** and **994 AI-Generated Images** from 7 generative pipelines: `BigGAN`, `VQDM`, `Stable Diffusion v5`, `Wukong`, `ADM`, `GLIDE`, and `Midjourney`).

### Global Iteration Progression Matrix

| Iteration | Architectural Description | Feature Count | Test Accuracy | Precision | Recall (Auth) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iteration 0** | Baseline Handcrafted Geometry (RANSAC Lines) | 6 | **72.18%** | 67.45% | 86.00% | 75.60% | 78.30% |
| **Iteration 1** | Multi-Domain Light Transport & Frequency Physics | 14 | **75.44%** | 71.61% | 84.50% | 77.52% | 85.45% |
| **Iteration 2** | SRM High-Order Noise Steganalysis + Stacking | 24 | **78.95%** | 74.79% | 87.50% | 80.65% | 87.28% |
| **Iteration 3** | Hybrid Physics + ResNet-18 Deep Feature Fusion | 536 | **88.72%** | 88.56% | 89.00% | 88.78% | 95.83% |
| **Iteration 4 (Final)** | **Multi-Scale EfficientNet-B0 + 5-Model Soft Ensemble** | **1,304** | **92.23%** | **90.05%** | **95.00%** | **92.46%** | **97.41%** |

* **Overall Accuracy Increase:** $+20.05\%$ (from $72.18\%$ to $92.23\%$)
* **ROC-AUC Discriminative Power:** $+19.11\%$ (from $78.30\%$ to $97.41\%$)
* **5-Fold Cross-Validation Score:** $92.03\% \pm 1.38\%$

---

## 2. Iteration-by-Iteration Breakdown

```
[Baseline Geometry (6 Feats)]  -->  Accuracy: 72.18%  | AUC: 78.30%
            │
            ▼ (+8 Light/Freq Feats)
[Multi-Domain Physics (14 Feats)] -->  Accuracy: 75.44%  | AUC: 85.45%
            │
            ▼ (+10 SRM Residual Feats + Stacking)
[Physics + SRM Steganalysis (24 Feats)] -->  Accuracy: 78.95%  | AUC: 87.28%
            │
            ▼ (+512-dim ResNet-18 Embeddings)
[Hybrid Deep Fusion (536 Feats)] -->  Accuracy: 88.72%  | AUC: 95.83%
            │
            ▼ (+1280-dim Multi-Scale EfficientNet-B0 + 5-Model Soft Ensemble)
[OPTIMIZED ENSEMBLE (1304 Feats)] -->  Accuracy: 92.23%  | AUC: 97.41%
```

---

### 🔹 Iteration 0: Baseline Handcrafted Geometry
* **Concept:** Relies purely on geometric line detection, Hough Transforms, and RANSAC Vanishing Point convergence to detect physical consistency in shadows.
* **Extracted Features (6):**
  1. `total_lines`: Total number of extracted linear segments.
  2. `max_inliers`: Number of lines converging to the primary vanishing point.
  3. `inlier_ratio`: Proportion of inlier lines relative to total lines.
  4. `angular_variance_deg`: Angular dispersion of cast shadow trajectories.
  5. `shadow_chroma_var`: Standard deviation of chrominance inside shadow regions.
  6. `lap_var`: Global Laplacian edge variance.
* **Classifier:** Soft Voting Ensemble (`RandomForestClassifier` + `GradientBoostingClassifier`).
* **Performance:**
  * **Accuracy:** 72.18%
  * **Precision:** 67.45%
  * **Recall:** 86.00%
  * **F1-Score:** 75.60%
  * **ROC-AUC:** 78.30%
* **Confusion Matrix:**
  $$\begin{pmatrix} 116 & 83 \\ 28 & 172 \end{pmatrix}$$
* **Analysis & Failure Modes:** Handcrafted line heuristics alone struggle on organic AI-generated scenes (portraits, landscapes, curvilinear textures) where straight linear shadow boundaries are sparse, producing high false positive rates ($83/199 = 41.7\%$).

---

### 🔹 Iteration 1: Multi-Domain Light Transport & Frequency Forensics
* **Concept:** Added physical illuminant constancy across image sub-regions, compression boundary consistency, and 2D discrete cosine frequency harmonics.
* **New Features Added (+8):**
  7. `quad_chroma_var`: Inter-quadrant CIELAB chromaticity variance testing uniform scene illumination.
  8. `gw_dev`: Gray-World illuminant deviation testing light source spectrum consistency.
  9. `lap_skew`: Skewness of Laplacian second derivatives (high-order gradient asymmetry).
  10. `high_freq_energy`: High-frequency spectral energy ratio via 2D DCT ($128 \times 128$ corner).
  11. `dct_mid_energy`: Mid-frequency spectral energy band ($64 \times 64$ to $128 \times 128$).
  12. `ela_mean`: Error Level Analysis compression error mean.
  13. `ela_std`: Error Level Analysis compression standard deviation.
  14. `penumbra_ratio`: Shadow penumbra edge gradient decay ratio.
* **Classifier:** Hyperparameter-tuned `XGBoost` + `LightGBM` + `RandomForest`.
* **Performance:**
  * **Accuracy:** 75.44% (+3.26%)
  * **Precision:** 71.61%
  * **Recall:** 84.50%
  * **F1-Score:** 77.52%
  * **ROC-AUC:** 85.45% (+7.15%)
* **Confusion Matrix:**
  $$\begin{pmatrix} 132 & 67 \\ 31 & 169 \end{pmatrix}$$

---

### 🔹 Iteration 2: Spatial Rich Model (SRM) Steganalysis & Stacking Ensemble
* **Concept:** Incorporated 5 directional high-pass convolution kernels derived from modern steganalysis (Spatial Rich Model) to detect micro-level generative lattice fingerprints and PRNU disruption.
* **New Features Added (+10):**
  15–19. `srm_var_0` through `srm_var_4`: Variance of residuals across 1st-order gradients, 2nd-order Laplacians, and 5x5 high-pass kernels.
  20–24. `srm_skew_0` through `srm_skew_4`: Skewness of residual distributions across each SRM kernel.
* **Classifier:** 5-Fold Cross-Validated **Stacking Classifier** (`XGBoost` + `LightGBM` + `RandomForest` feeding into a `LogisticRegression` meta-learner with `RobustScaler`).
* **Performance:**
  * **Accuracy:** 78.95% (+6.77% over baseline)
  * **Precision:** 74.79%
  * **Recall:** 87.50%
  * **F1-Score:** 80.65%
  * **ROC-AUC:** 87.28% (+8.98%)
* **Confusion Matrix:**
  $$\begin{pmatrix} 140 & 59 \\ 25 & 175 \end{pmatrix}$$

---

### 🔹 Iteration 3: Hybrid Physics & Deep ResNet-18 Feature Fusion
* **Concept:** Fused domain-specific physical/SRM features with high-level visual semantic embeddings extracted from a deep convolutional forensic backbone (`ResNet-18`, 512 dimensions).
* **Representation:** 536-dimensional concatenated feature vector ($24\text{ Tabular} + 512\text{ Deep Embeddings}$).
* **Preprocessing:** Dual pipeline scaling using `RobustScaler` (tabular) and `StandardScaler` (deep embeddings).
* **Classifier:** Stacking Meta-Learner with Stratified K-Fold.
* **Performance:**
  * **Accuracy:** 88.72% (+16.54% over baseline)
  * **Precision:** 88.56%
  * **Recall:** 89.00%
  * **F1-Score:** 88.78%
  * **ROC-AUC:** 95.83% (+17.53%)
* **Confusion Matrix:**
  $$\begin{pmatrix} 176 & 23 \\ 22 & 178 \end{pmatrix}$$

---

### 🔹 Iteration 4: Final Optimized Multi-Scale Deep Fusion & 5-Model Ensemble
* **Concept:** Replaced single-scale embeddings with **Multi-Scale EfficientNet-B0 Receptive Fields (1,280 dimensions)** and built a heterogeneous **5-Model Soft Voting Ensemble** combining Gradient Boosting, Random Forests, ExtraTrees, and Multi-Layer Perceptrons. Handled zero-line curvilinear scenes gracefully.
* **Representation:** 1,304-dimensional multi-domain representation ($24\text{ Physics/SRM} + 1280\text{ Multi-Scale Deep Embeddings}$).
* **Classifiers in Soft Ensemble:**
  1. `XGBClassifier` ($350\text{ trees, max depth } 5, \text{colsample } 0.85$)
  2. `LGBMClassifier` ($350\text{ trees, } 31\text{ leaves, lr } 0.03$)
  3. `RandomForestClassifier` ($300\text{ trees, max depth } 12$)
  4. `ExtraTreesClassifier` ($300\text{ trees, max depth } 14$)
  5. `MLPClassifier` ($256 \times 64\text{ hidden layers, } \alpha=0.01$)
* **Performance on Holdout Test Set (399 Samples):**
  * **Accuracy:** **92.23%** (+20.05% gain over baseline)
  * **Precision:** **90.05%** (Fake detection precision: **95%**)
  * **Recall:** **95.00%** (Authentic photo recall: **95%**)
  * **F1-Score:** **92.46%**
  * **ROC-AUC:** **97.41%** (+19.11% gain)
  * **5-Fold Cross-Validation Accuracy:** **92.03% ($\pm 1.38\%$)**

#### Final Classification Report (Iteration 4)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Fake (AI-Generated)** | **0.95** | **0.89** | **0.92** | 199 |
| **Authentic Photography** | **0.90** | **0.95** | **0.92** | 200 |
| **Accuracy** | | | **92.23%** | **399** |
| **Macro Average** | **0.92** | **0.92** | **0.92** | 399 |
| **Weighted Average** | **0.92** | **0.92** | **0.92** | 399 |

#### Final Confusion Matrix (Iteration 4)
$$\begin{pmatrix} 178 & 21 \\ 10 & 190 \end{pmatrix}$$

---

## 3. Comparative Summary Table

| Metric | Iteration 0 (Baseline) | Iteration 1 (Multi-Domain) | Iteration 2 (SRM Steg) | Iteration 3 (Deep ResNet) | **Iteration 4 (Final Optimized)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Features** | 6 | 14 | 24 | 536 | **1,304** |
| **Test Accuracy** | 72.18% | 75.44% | 78.95% | 88.72% | **92.23%** |
| **ROC-AUC** | 78.30% | 85.45% | 87.28% | 95.83% | **97.41%** |
| **Precision** | 67.45% | 71.61% | 74.79% | 88.56% | **90.05%** |
| **Recall (Auth)** | 86.00% | 84.50% | 87.50% | 89.00% | **95.00%** |
| **F1-Score** | 75.60% | 77.52% | 80.65% | 88.78% | **92.46%** |
| **Cross-Val Score** | 71.85% | 75.12% | 78.40% | 88.10% | **92.03% (±1.38%)** |

---

## 4. Codebase Artifacts & Integration

1. **Serialized Model Bundle:** [pillar5_ml_model.pkl](file:///d:/Projects/Mini%20Project/Multi-Pillar_Deepfake_Detection/Pillar%205/pillar5_ml_model.pkl)
   * Contains the complete multi-scale deep fusion pipeline, feature column schemas, robust scaling transformers, and the 5-model soft voting ensemble.
2. **End-to-End Optimization Script:** [optimize_pillar5_above90.py](file:///d:/Projects/Mini%20Project/Multi-Pillar_Deepfake_Detection/Pillar%205/optimize_pillar5_above90.py)
3. **Core Inference & Visual Evidence Engine:** [pillar_5_forensics.py](file:///d:/Projects/Mini%20Project/Multi-Pillar_Deepfake_Detection/Pillar%205/pillar_5_forensics.py)
   * Produces multi-panel forensic visualizations, RANSAC vanishing point overlays, JSON telemetry logs, and quantitative CSV outputs.

---
*Report Generated by Universal Synthetic Media Forensics Engine (USMFE) Research Pipeline.*
