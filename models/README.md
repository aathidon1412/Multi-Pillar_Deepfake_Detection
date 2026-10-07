# Pretrained Model Checkpoints & Forensic Weights
## Universal Synthetic Media Forensics Engine (USMFE)

This directory stores offline pretrained model weights, classification heads, and heuristic XML classifiers for all analytical pillars.

---

## 📂 Directory Layout & Checkpoint Summary

```text
models/
├── pillar1/
│   ├── ViT_Model/                    # Hugging Face Vision Transformer backbone (google/vit-base-patch16-224)
│   └── diffusion_vit_head.pt         # Fine-tuned classification head for facial spectral artifacts
├── pillar2/
│   ├── Spatial_ViT_Model/            # Spatial image classifier (dima806/deepfake_vs_real_image_detection)
│   ├── consensus_head.pth            # Cross-dataset calibrated video classification head
│   └── haarcascade_frontalface_default.xml # OpenCV Haar cascade for real-time face tracking
├── pillar3/
│   └── Acoustic_Model/               # Hugging Face Audio Transformer (Hemgg/Deepfake-audio-detection)
├── pillar4/
│   └── LayoutLM_OCR_Model/           # Document OCR LayoutLM weights (Rule-based Benford's Law analysis)
└── pillar5/
    ├── Authoritative_Hybrid_Model.pkl # 160-dim Voting Classifier Ensemble (Shadow RANSAC + SRM + PRNU)
    └── Fallback_Model.pkl             # Fallback geometry ML classifier
```

---

## 📥 Model Acquisition & Download Instructions

If pulling this repository on a new environment where large binaries are excluded via `.gitignore`:

### Pillar 1: Vision Transformer (ViT)
- **Source:** Automatically fetched from Hugging Face or loaded from `models/pillar1/ViT_Model/`.
- **Hugging Face ID:** `google/vit-base-patch16-224` (or custom fine-tuned checkpoint).
- **Head:** `diffusion_vit_head.pt` (3-class output: Real, AI-Generated, Adversarial).

### Pillar 2: Video Authenticity & Biological rPPG
- **Spatial Model:** Pretrained on FaceForensics++ / DFDC. Cached in `models/pillar2/Spatial_ViT_Model/` or auto-downloaded from `dima806/deepfake_vs_real_image_detection`.
- **Face Cascade:** `haarcascade_frontalface_default.xml` (bundled in OpenCV).
- **Consensus Head:** `consensus_head.pth` (calibrated soft-voting weights).

### Pillar 3: Acoustic & Voice Forensics
- **Acoustic Model:** Hugging Face Wav2Vec2 / Audio Transformer.
- **Hugging Face ID:** `Hemgg/Deepfake-audio-detection` (cached in `models/pillar3/Acoustic_Model/`).
- **Demixing:** Librosa Harmonic-Percussive Source Separation (HPSS) executes on-the-fly (no weights required).

### Pillar 4: Statistical Document Forensics
- **Engine:** PyMuPDF + Tesseract OCR + Benford's Law $\chi^2$ statistical goodness-of-fit test.
- **Weights:** No binary weights required (deterministic statistical divergence $P(d) = \log_{10}(1 + 1/d)$).

### Pillar 5: Physical Geometry & Sensor Steganalysis
- **Primary Bundle:** `Authoritative_Hybrid_Model.pkl` (Voting Classifier combining ExtraTrees, Random Forest, HistGradientBoosting, and LogisticRegression).
- **Fallback:** `Fallback_Model.pkl`.

---

## ⚙️ Hardware & Compute Support

Models automatically utilize:
- **NVIDIA GPU (CUDA):** If PyTorch with CUDA is installed (detected automatically via `backend/api/ml_device.py`).
- **CPU (OpenMP / MKL):** Fallback mode with thread-safety optimizations enabled.
