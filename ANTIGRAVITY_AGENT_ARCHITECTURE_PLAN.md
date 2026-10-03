# Antigravity Agent: Modular Deepfake Detection Architecture Plan

## Executive Summary & Mapping to Current Codebase

This document reviews the proposed 5-module autonomous deepfake detection architecture and outlines the concrete technical mapping, data flow, and phased execution plan.

The workspace currently houses a 5-pillar multimodal detection system in [core/](file:///d:/College%20Studies/Mini%20Project/core/):
- **Pillar 1 (`pillar1_engine.py`)**: Vision Transformer (ViT) & Spatial ELA artifact analysis.
- **Pillar 2 (`pillar2_engine.py`)**: Temporal & Optical Flow dynamics (motion consistency).
- **Pillar 3 (`pillar3_engine.py`)**: Audio spectrogram & synthetic speech detection.
- **Pillar 4 (`pillar4_engine.py`)**: Inconsistency & cross-modal synchronization / features.
- **Pillar 5 (`pillar5_engine.py`) / Consensus (`consensus.py`)**: Aggregation and multi-pillar decision fusion.
- **XAI modules (`pillar*_xai.py`, `xai.py`)**: Heatmaps, Grad-CAM, Shapley values, and telemetry.

The proposed `antigravity_agent/` refactors and packages these pillars into a modular, standalone agent framework with a unified orchestrator, confidence-calibrated classification head, and timeline reporting.

---

## 1. System Architecture & Component Mapping

```
                               ┌────────────────────────────┐
                               │     Input Video (.mp4)     │
                               └─────────────┬──────────────┘
                                             │
                                             ▼
                               ┌────────────────────────────┐
                               │   utils/video_loader.py    │
                               │  - 16 Equidistant Frames   │
                               │  - Audio Track Extraction  │
                               │  - Codec & Container Meta  │
                               └─────────────┬──────────────┘
                                             │
                                             ▼
                               ┌────────────────────────────┐
                               │  core/orchestrator.py      │
                               │  - Pipeline Coordinator    │
                               │  - Parallel / Async dispatch│
                               └─────────────┬──────────────┘
                                             │
        ┌────────────────────────────────────┼────────────────────────────────────┐
        ▼                                    ▼                                    ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
│  modules/spatial_detector.py │ │ modules/temporal_analyzer.py │ │  modules/audio_detector.py   │
│  [Module 1]                  │ │ [Module 2]                   │ │ [Module 3]                   │
│  - Frozen ViT (Base / SigLIP)│ │ - Farnebäck Optical Flow     │ │ - Container & Codec checks   │
│  - Spatial artifact scores   │ │ - Inter-frame pixel diffs    │ │ - Mel-spectrogram analysis   │
│  - Per-frame suspicion tensor│ │ - Motion velocity variance   │ │ - Synthetic voice profile    │
└──────────────┬───────────────┘ └──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │                                │
               └────────────────────────────────┼────────────────────────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │  modules/consensus_engine.py   │
                               │  [Module 4]                    │
                               │  - Bi-LSTM / MLP Fusion Head   │
                               │  - Softmax / Calibrated Probs  │
                               │  - Threshold Confidence Zones  │
                               │    * < 0.35: Real              │
                               │    * 0.35 - 0.65: Inconclusive │
                               │    * > 0.65: Deepfake          │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │      modules/explainer.py      │
                               │  [Module 5]                    │
                               │  - Anomaly timeline plot       │
                               │  - Natural language verdicts   │
                               │  - High-suspicion frame dump   │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │ app.py / Streamlit Dashboard   │
                               │ Interactive UI & Report Export │
                               └────────────────────────────────┘
```

---

## 2. Detailed Technical Specifications

### Module 1: Spatial Forensics (`antigravity_agent/modules/spatial_detector.py`)
- **Input:** $N=16$ sampled RGB frames $[N, 3, 224, 224]$.
- **Backbone:** Hugging Face `transformers` ViT (`google/vit-base-patch16-224-in21k` or timm ViT/SigLIP).
- **Features Extracted:** Pooled CLS token embeddings (dimension 768) + normalized patch variance / artifact scores per frame.
- **Output:**
  ```python
  {
      "frame_scores": List[float],      # Length 16, range [0, 1]
      "spatial_embeddings": Tensor,     # Shape: [16, 768]
      "top_suspicious_frames": List[int] # Indices of top 3 peak artifact frames
  }
  ```

### Module 2: Temporal Inconsistency (`antigravity_agent/modules/temporal_analyzer.py`)
- **Input:** Consecutive frame pairs $(F_t, F_{t+1})$ for $t \in [0, N-2]$.
- **Algorithm:**
  - Grayscale conversion and Farnebäck Dense Optical Flow: `cv2.calcOpticalFlowFarneback`.
  - Velocity magnitude matrix: $M_t = \sqrt{u^2 + v^2}$.
  - Mean absolute pixel difference: $\Delta_t = \frac{1}{HW}\sum |F_{t+1} - F_t|$.
  - Motion Acceleration & Variance: Velocity divergence $\sigma_{\text{velocity}}$ to capture morphing and flicker glitches.
- **Output:**
  ```python
  {
      "motion_vectors": List[float],    # Pairwise mean motion magnitudes
      "pixel_diffs": List[float],       # Pairwise frame delta values
      "velocity_variance": float,       # Overall temporal smoothness metric
      "temporal_spike_indices": List[int] # Pair indices where sudden jumps occur
  }
  ```

### Module 3: Audio & Track Integrity (`antigravity_agent/modules/audio_detector.py`)
- **Input:** Extracted audio stream (via `moviepy` or `ffmpeg-python`).
- **Processing:**
  - Container header inspection (checking for missing/truncated audio streams or mismatched sample rates).
  - Mel-spectrogram computation via `librosa` / `torchaudio`.
  - Spectral flatness and pitch continuity checks to identify TTS/voice cloned robotic artifacts.
- **Output:**
  ```python
  {
      "has_audio": bool,
      "audio_anomaly_score": float,     # Range [0, 1]
      "codec_metadata": dict,
      "spectrogram_irregularities": List[str]
  }
  ```

### Module 4: Consensus Aggregator (`antigravity_agent/modules/consensus_engine.py`)
- **Input:** Fused feature vectors:
  - Sequence of spatial features ($16 \times 768$ or reduced $16 \times 64$ via projection linear layer).
  - Sequence of motion metrics ($15 \times 2$).
- **Architecture:**
  - Lightweight Bi-LSTM (hidden size = 64, 1 layer) or multi-head attention fusion + MLP classifier.
- **Decision Logic:**
  - $P(\text{Fake}) < 0.35 \implies \textbf{Authentic / Real}$
  - $0.35 \le P(\text{Fake}) \le 0.65 \implies \textbf{Inconclusive / Review Required}$
  - $P(\text{Fake}) > 0.65 \implies \textbf{AI-Generated / Deepfake}$

### Module 5: Explainability & Reporting (`antigravity_agent/modules/explainer.py`)
- **Outputs Generated:**
  1. **Timeline Anomaly Graph:** Plot with twin y-axes showing Spatial Artifact Score vs. Motion Delta across timestamps $t_0 \dots t_{15}$.
  2. **Synthesized Verdict Report:** Markdown/JSON report pinpointing the exact timestamp triggers (e.g. *"Frame 6 (00:04.2): ViT detected unnatural boundary blur (score: 0.82)"*).
  3. **Visual Evidence:** Top artifact frame images with bounding boxes or high-gradient zones highlighted.

---

## 3. Directory Layout Specification

```text
antigravity_agent/
│
├── core/
│   ├── __init__.py
│   └── orchestrator.py             # Pipeline coordinator & execution graph
│
├── modules/
│   ├── __init__.py
│   ├── spatial_detector.py         # ViT spatial artifact detector
│   ├── temporal_analyzer.py        # Farnebäck optical flow & delta analyzer
│   ├── audio_detector.py           # Audio container & spectrogram integrity
│   ├── consensus_engine.py         # Bi-LSTM/MLP aggregator & calibration
│   └── explainer.py                # Visual timelines & natural language verdicts
│
├── utils/
│   ├── __init__.py
│   ├── video_loader.py             # 16-frame uniform sampler & audio extractor
│   └── visualization.py            # Matplotlib timeline & anomaly curves
│
├── app.py                          # Streamlit / Gradio front-end interface
├── train_head.py                   # Script to train & calibrate the consensus head
└── requirements.txt                # Specific pinned dependencies
```

---

## 4. 48-Hour Phased Implementation Roadmap

| Phase | Time Window | Deliverables | Key Verification Milestone |
|---|---|---|---|
| **Phase 1** | Hours 01–06 | `utils/video_loader.py`, `core/orchestrator.py` | Load `.mp4`, sample 16 frames uniformly as tensors, extract audio track. |
| **Phase 2** | Hours 07–16 | `modules/spatial_detector.py`, `modules/temporal_analyzer.py` | Run ViT inference and Farnebäck flow; return structured anomaly vectors. |
| **Phase 3** | Hours 17–24 | `modules/audio_detector.py`, `modules/consensus_engine.py`, `train_head.py` | Bi-LSTM fusion head trained/calibrated with 3-tier confidence classification. |
| **Phase 4** | Hours 25–34 | `modules/explainer.py`, `utils/visualization.py`, `app.py` | Interactive timeline charts, natural language evidence generation, and web UI. |
| **Phase 5** | Hours 35–48 | Benchmarking, threshold tuning, stress testing | Validate on real & deepfake test videos; generate final report. |

---

## 5. Next Steps

1. Review this architectural plan.
2. When ready to build, start **Phase 1**: initialize `antigravity_agent/` directory, set up `utils/video_loader.py`, and create `core/orchestrator.py`.
