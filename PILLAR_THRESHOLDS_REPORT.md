# Multi-Pillar Deepfake Detection System
## Forensic Thresholds & Decision Boundaries Reference Guide

This document summarizes the mathematical criteria, operational thresholds, and decision boundaries used across all forensic pillars and the Master Consensus Engine.

---

### Executive Summary: Decision Boundaries at a Glance

| Pillar | Domain | Core Technology | Primary Decision Threshold |
| :--- | :--- | :--- | :--- |
| **Pillar 1** | Image / Facial Artifacts | Vision Transformer (`ViT-Base-224`) | $\text{Real Prob} \ge 0.50 \implies \textbf{AUTHENTIC}$ |
| **Pillar 2** | Video Forensics | Temporal Flickering + Lip-Sync + Seams | $\text{Temporal Fake Score} \ge 0.50 \implies \textbf{FAKE}$ |
| **Pillar 3** | Audio / Voice Cloning | Wav2Vec2 + HPSS Acoustic Filter | $\text{Fake Confidence} \ge 50.0\% \implies \textbf{FAKE}$ |
| **Pillar 4** | Document / Invoice OCR | Benford's Law 1st-Digit Distribution | Dynamic MAE: Strict $\le 0.02$, Loose $\le 0.035$ |
| **Pillar 5** | Light Physics & Sensor Noise | RANSAC Vanishing Point + PRNU SRM | $\text{Real Prob} \ge 0.50$ (Noise Floor: $\text{SRM}_4 \ge 100$) |
| **Consensus** | Master Multi-Pillar Fusion | Hierarchical Gating + Soft Voting | $\text{Fused Fake Score} < 0.45 \implies \textbf{AUTHENTIC}$ |

---

### 1. Pillar 1: Vision Transformer (ViT) & Spectral Forensics
* **File:** `core/pillar1_engine.py`
* **Target:** Images, human faces, and AI diffusion synthesis patterns.
* **Architecture:** `google/vit-base-patch16-224` (Fine-tuned).

#### Preprocessing & Normalization
* **Resolution:** $224 \times 224$ pixels.
* **ImageNet Standard:** $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

#### Decision Thresholds
* **Authentic Boundary:**
  $$\mathbf{\text{real\_prob} \ge 0.50 \implies \text{AUTHENTIC}}$$
  * Confidence: $\text{real\_prob} \times 100\%$
* **Fake Boundary:**
  $$\mathbf{\text{real\_prob} < 0.50 \implies \text{FAKE (AI)}}$$
  * Confidence: $\text{fake\_prob} \times 100\%$
* **Consensus Hard Override:**
  $$\mathbf{\text{fake\_prob} \ge 0.70 \implies \text{Instant Fake Flag}}$$
  *(If ViT is $\ge 70\%$ confident, it triggers an authoritative neural override).*

---

### 2. Pillar 2: Temporal & Multi-Modal Video Forensics
* **File:** `core/pillar2_engine.py` (`Pillar 2/video-authenticity-detector`)
* **Target:** Deepfake videos, face-swaps, and lip-sync manipulation.

#### Multi-Modal Analysis Components
1. **Visual Seams:** Frame-level boundary blending artifacts around detected faces.
2. **Temporal Flickering:** Frame-to-frame feature variance across consecutive frames.
3. **Audio-Visual Lip Sync:** Correlation between vocal phonemes and mouth visemes.

#### Decision Thresholds
* **Authentic Video:** Fused Video Fake Score $< 0.50$ ($< 50\%$).
* **Manipulated Deepfake:** Fused Video Fake Score $\ge 0.50$ ($\ge 50\%$).

---

### 3. Pillar 3: Acoustic & Voice Synthetic Speech Forensics
* **File:** `Pillar 3/detect.py`
* **Target:** Audio files, synthetic speech cloning (ElevenLabs, Bark, Tortoise-TTS).
* **Architecture:** `Hemgg/Deepfake-audio-detection` (Wav2Vec2) + Librosa HPSS.

#### Windowing & Preprocessing
* **Sample Rate:** $16,000\text{ Hz}$ mono.
* **Sliding Window:** $4.0\text{ s}$ duration with a $2.0\text{ s}$ hop/stride.

#### Primary Decision Boundary
$$\mathbf{\text{fake\_conf} \ge \text{real\_conf} \implies \text{FAKE (AI CLONED)}}$$
$$\mathbf{\text{real\_conf} > \text{fake\_conf} \implies \text{REAL (HUMAN)}}$$

#### Domain Safeguards & False Alarm Filters
1. **WhatsApp / VoIP Compression Filter:**
   * **Condition:** $\text{ZCR} < 0.08$, $\text{Centroid} < 500\text{ Hz}$, and $\text{Percussive Power} < 0.004$.
   * **Action:** Recalibrates $\text{real\_raw} \ge 94.5\%$ to prevent mobile codec distortion from being flagged as AI.
2. **Studio Music / Autotune Guard:**
   * **Condition:** Harmonic-to-Percussive Ratio ($\text{HPR}$) $> 1.5$ in Music Mode.
   * **Action:** Restricts confidence to $[40\%, 60\%]$ with an *"Inconclusive Studio Track"* warning.

---

### 4. Pillar 4: Statistical OCR & Benford’s Law (Documents)
* **File:** `core/pillar4_engine.py`
* **Target:** Scanned invoices, contracts, receipts, financial records.
* **Technique:** PyTesseract OCR $\rightarrow$ First significant digit extraction ($d \in [1, 9]$) $\rightarrow$ Mean Absolute Error (MAE) from Benford's distribution.

#### Minimum Data Requirement
* **Condition:** $N < 5$ first digits $\implies$ **Inapplicable / Non-Document** (Weight = 0.0).

#### Dynamic Sample-Scaled Thresholds
To prevent false alarms on short documents, thresholds scale dynamically with digit count $N$:
$$\text{scale} = \frac{100.0}{\max(N, 30)}$$
$$\text{threshold\_strict} = 0.020 + (0.010 \times \text{scale})$$
$$\text{threshold\_loose} = 0.035 + (0.010 \times \text{scale})$$

#### Decision Boundaries
* **Strict Authentic:** $\text{MAE} < \text{threshold\_strict} \implies \textbf{AUTHENTIC DOCUMENT}$ ($90\% - 95\%$ conf).
* **Acceptable Authentic:** $\text{threshold\_strict} \le \text{MAE} \le \text{threshold\_loose} \implies \textbf{AUTHENTIC DOCUMENT}$ ($60\% - 80\%$ conf).
* **Synthesized / Tampered:** $\text{MAE} > \text{threshold\_loose} \implies \textbf{FORGED / AI-SYNTHESIZED}$ ($\ge 85\%$ conf).
* **Chi-Square Override:** If $p\text{-value} < 0.01$ and $\text{MAE} > 0.025 \implies \textbf{FORGED}$ ($99\%$ statistical confidence).

---

### 5. Pillar 5: Shadow Physics, Vanishing Points & PRNU Steganalysis
* **File:** `core/pillar5_engine.py`
* **Target:** Illumination physics, shadow consistency, and hardware camera sensor noise.
* **Components:** Canny Edges + RANSAC Light Tracing + SRM High-Pass Noise + EfficientNet-B0 ML Ensemble.

#### Decision Thresholds
1. **Machine Learning Ensemble (Primary):**
   * $\mathbf{\text{real\_prob} \ge 0.50 \implies \text{AUTHENTIC PHYSICS}}$
   * $\mathbf{\text{real\_prob} < 0.50 \implies \text{PHYSICS ANOMALY (AI GENERATED)}}$
2. **PRNU Sensor Noise Steganalysis Override:**
   * Generative AI models lack physical camera silicon noise.
   * **Rule:** If $\mathbf{\text{srm\_var\_4} < 100.0}$ (artificially smooth noise floor) and $\text{fake\_prob} < 0.60$:
     * Forces `is_real = False` and sets $\text{fake\_prob} \ge 0.65 \text{ to } 0.90$.
     * Verdict: **`PHYSICS ANOMALY (AI GENERATED) + PRNU Steganalysis Override`**.
3. **Deterministic Fallback (If ML Model is Offline):**
   * Checks pure geometry:
     $$\mathbf{\text{angular\_variance} < 12.0^\circ \quad \text{AND} \quad \text{max\_inliers} \ge 20 \quad \text{AND} \quad \text{shadow\_chroma\_var} \ge 10.0}$$
   * If satisfied $\implies \textbf{AUTHENTIC PHYSICS}$ ($94\%$ confidence).
   * If failed $\implies \textbf{PHYSICS ANOMALY}$ ($94\%$ fake confidence).

---

### 6. The Master Consensus Engine: Final Fusion
* **File:** `core/consensus.py`
* **Target:** Unified decision across all pillars using hierarchical priority gating.

#### Priority Hierarchy
```text
1. Hardware EXIF Present? ──► YES ──► AUTHENTIC (>= 92%)
          │ NO
2. Scanned Paper Document? ─► YES ──► AUTHENTIC (88% - Pillar 4 Gating)
          │ NO
3. PRNU Noise Missing? ─────► YES ──► FAKE (>= 82% - Sensor Noise Override)
          │ NO
4. ViT Neural Fake >= 0.70? ─► YES ──► FAKE (Pillar 1 Neural Override)
          │ NO
5. Weighted Multi-Pillar Fusion:
   Fused Fake Prob = (0.55 * P5_fake) + (0.45 * P1_fake)
```

#### Final Consensus Decision Boundary:
$$\mathbf{\text{fused\_fake\_prob} < 0.45 \implies \text{AUTHENTIC MEDIA}}$$
$$\mathbf{\text{fused\_fake\_prob} \ge 0.45 \implies \text{FAKE (SYNTHETIC AI ANOMALY)}}$$
