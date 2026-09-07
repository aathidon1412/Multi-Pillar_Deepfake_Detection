## Pillar 2: Hybrid Visual Forensics & Biological rPPG Analysis

### Algorithms & Mathematics Used

#### Channel 1 — Spatial / Visual AI (Vision Transformer)
Facial regions-of-interest (ROI) are extracted per-frame using the Viola-Jones Haar Cascade detector (`haarcascade_frontalface_default.xml`), sorting detected candidates by area descending to ensure primary facial tracking. To prevent aspect-ratio distortion and edge cutoffs that trigger false-positive deepfake classifications, face ROIs are transformed into proportional, centered 1:1 square crops using high-quality Lanczos interpolation (`Image.Resampling.LANCZOS`) and border reflection (`cv2.BORDER_REFLECT_101`), normalized to exactly 224×224 pixels before being passed to `dima806/deepfake_vs_real_image_detection` (ViT fine-tuned on FaceForensics++ and DFDC).

Frame scores are aggregated using a **Trimmed Median** (filtering out the top and bottom 10% outliers to eliminate transient head-turn artifacts) against a recalibrated visual decision threshold of **0.70** (accounting for standard H.264/H.265 compression noise):

$$\hat{p}_{\text{visual}} = \text{Median}\Bigl(\{p_i \mid p_{(0.10)} \le p_i \le p_{(0.90)}\}\Bigr)$$

#### Channel 2 — Biological rPPG (Remote Photoplethysmography with Velocity Filtering & EMA)
To eliminate rPPG failure from natural head motion, the face bounding box is stabilized across frames using an Exponential Moving Average ($\alpha = 0.70$):

$$\mathbf{b}_t = \alpha \mathbf{b}_t^{\text{raw}} + (1 - \alpha)\mathbf{b}_{t-1}$$

Forehead skin patches (upper 5%–32% of the EMA face box) are sampled for green-channel spatial intensity. Frame-to-frame centroid jitter velocity is tracked:

$$v_t = \sqrt{(c_x^t - c_x^{t-1})^2 + (c_y^t - c_y^{t-1})^2}$$

Rather than aborting the entire rPPG pipeline when head motion occurs, individual frames with $v_t > 12$ px (sudden jerks) are dropped dynamically. The accepted signal is linearly detrended and passed through a zero-phase 4th-order Butterworth bandpass filter ($0.75 \text{ Hz} \le f \le 2.50 \text{ Hz}$, corresponding to 45–150 BPM). A 4× zero-padded FFT extracts the dominant pulse frequency $f^*$ and biological Signal-to-Noise Ratio:

$$\text{BPM} = 60 \cdot f^*, \quad \text{SNR} = \frac{\overline{|X(f^*)|^2}}{\overline{|X(f \neq f^*)|^2}}$$

A physiological pulse is declared verified when $\text{SNR} \ge 2.2$ and $50 \le \text{BPM} \le 130$.

#### Ensemble Verdict Logic

| Visual AI Score | rPPG Result | Final Verdict |
| :--- | :--- | :--- |
| > 70% | Any | **FAKE** |
| ≤ 70% | Valid pulse (REAL) | **REAL (Biologically Confirmed)** |
| ≤ 70% | Inconclusive | **REAL (Visual only — rPPG inconclusive)** |
| ≤ 70% | Synthetic (no pulse) | **FAKE** |

---

### Challenges & Resolutions

1. **False-Positive Suppression via 1:1 Square Normalization**: Rectangular bounding-box crops previously caused aspect-ratio distortion and black border voids during ViT preprocessing, inflating real human faces into false-positive deepfakes. Centered square extraction with `cv2.BORDER_REFLECT_101` and Lanczos 224×224 scaling restores natural facial geometry.
2. **Outlier-Resistant Aggregation**: The aggressive 90th percentile aggregated transient compression artifacts into false alerts. The trimmed median effectively isolates true baseline characteristics while rejecting short-lived boundary artifacts.
3. **Velocity-Gated rPPG Tracking**: Cumulative centroid drift previously aborted rPPG on any extended video with natural head movements. Replacing cumulative drift with instantaneous velocity jitter ($v_t \le 12$ px) and EMA bounding-box stabilization enables uninterrupted cardiac pulse extraction (SNR 3.161, 51.9 BPM) on genuine human subjects.

---

### Experimental Findings

Quantitative benchmark results on both test clips following engine recalibration:

| Parameter | fake_avatar.mp4 | real_face.mp4 |
| :--- | :--- | :--- |
| Evaluated Frames | 32 frames (every 3rd) | 90 frames (every 3rd) |
| Visual AI Score (Trimmed Median) | **83.31%** | **64.70%** |
| Decision Threshold | 70.0% | 70.0% |
| Visual AI Verdict | **FAKE** | **REAL** |
| rPPG BPM | N/A | **51.9 BPM** |
| rPPG SNR | N/A | **3.161** (threshold: 2.2) |
| Jitter Velocity (Mean / Max) | 1.78 px / 5.10 px | 1.64 px / 6.96 px |
| Discarded Jitter Frames | 0 | 0 |
| rPPG Verdict | Inconclusive (98 samples < 144) | **REAL (Valid biological pulse)** |
| **Final Ensemble Verdict** | **FAKE** | **REAL** |
| Final Reason | Visual AI score 83.3% > 70% threshold. | Visual AI score < 70% and biological pulse confirmed (51.9 BPM). |

---

### Metadata

| Parameter | Value |
| :--- | :--- |
| Engine | `pillar2_hybrid.py` |
| Visual Model | `dima806/deepfake_vs_real_image_detection` (ViT) |
| Preprocessing | 1:1 centered square crop, `cv2.BORDER_REFLECT_101`, Lanczos 224×224 |
| Visual Aggregation | Trimmed Median (10th–90th percentile) |
| Visual Threshold | 0.70 (70.0%) |
| Face Box Smoothing | Exponential Moving Average ($\alpha = 0.70$) |
| Instant Jitter Limit | $v_t \le 12.0$ px/frame |
| rPPG BPF | 4th-Order Butterworth, 0.75 – 2.50 Hz |
| FFT Zero-Padding | 4× |
| SNR Threshold | $\ge 2.20$ |
| Valid BPM Range | 50 – 130 BPM |
| Evidence Artifacts | `fake_avatar_hybrid_evidence.png`, `real_face_hybrid_evidence.png` |
