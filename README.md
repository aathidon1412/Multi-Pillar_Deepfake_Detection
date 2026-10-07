# Universal Synthetic Media Forensics Engine (USMFE)
## Multi-Pillar Deepfake & Synthetic Media Detection Framework

A state-of-the-art multimodal cyber-forensic engine designed to detect synthetic, AI-generated, and manipulated media across **Visual, Acoustic, Biological, Document, and Physical Geometry domains**.

---

## 🏛️ System Architecture: The Multi-Pillar Framework

Unlike traditional monolithic deepfake detectors that overfit to specific generative models, USMFE uses **orthogonal forensic pillars** fused via a **deterministic override mechanism** to withstand domain shift:

| Pillar | Domain / Modality | Architecture / Algorithm | Key Forensic Indicators |
| :--- | :--- | :--- | :--- |
| **Pillar 1** | **Face & Frequency Forensics** | Hugging Face Vision Transformer (`ViT-Base`) | High-frequency facial blending borders, color-space transitions, ImageNet-normalized spectral anomalies. |
| **Pillar 2** | **Biological Forensics** | Remote Photoplethysmography (`rPPG`) + FFT + Temporal ViT | Subdermal blood volume pulse (BVP) extraction, cardiac periodicity disruption, and frame-to-frame inconsistency. |
| **Pillar 3** | **Acoustic Speech Forensics** | Hugging Face Audio Transformer (`Hemgg/Deepfake-audio-detection`) + Librosa HPSS | Synthetic TTS voice cloning, phase discontinuity, harmonic vocal isolation from percussion. |
| **Pillar 4** | **Financial Document Forensics** | Tesseract OCR + PyMuPDF + Benford's Law ($\chi^2$ test) | Statistical first-digit frequency divergence ($P(d) = \log_{10}(1 + 1/d)$) in invoices, receipts, and PDFs. |
| **Pillar 5** | **Physical Geometry & Steganalysis** | RANSAC Vanishing Point Rays + Spatial Rich Model (SRM) + Hybrid Ensemble | Shadow convergence consistency, JPEG Error Level Analysis (ELA), camera sensor PRNU noise fingerprint (`srm_var_4`). |

---

## ⚡ False Negative Mitigation & Deterministic Override

To prevent deep semantic backbones from mistaking clean modern diffusion generations (Midjourney v6, SDXL, DALL-E) as authentic photos:
- **Sensor PRNU Fingerprint Verification:** Physical camera sensors imprint photo-response non-uniformity (PRNU) noise ($\text{SRM}_4 \ge 100$). Synthetic AI images lack camera sensor noise ($\text{SRM}_4 \le 5 - 46$), automatically triggering anomaly flags.
- **Deterministic Override Rule:** If any specialized forensic pillar detects an anomaly with high confidence ($> 65\%$), the system flags the media as **FAKE** rather than letting a soft weighted average smooth out the forensic flaw.

---

## 📁 Repository Structure

The backend follows a standard, clean layered architecture designed for production maintainability and clear academic explanation:

```text
Multi-Pillar_Deepfake_Detection/
├── backend/
│   ├── api/                          # REST API layer (FastAPI routers, config, device detection)
│   │   ├── config.py                 # Centralized configuration, storage paths, CORS
│   │   ├── main.py                   # FastAPI application entry point
│   │   ├── ml_device.py              # PyTorch/CUDA compute device selection
│   │   ├── runtime_env.py            # Thread safety & OpenMP initialization
│   │   └── routes/                   # API route handlers
│   │       ├── upload.py             # File upload endpoint
│   │       ├── analysis.py           # Video analysis lifecycle endpoints
│   │       ├── history.py            # JSON-based forensic history endpoints
│   │       └── pillars.py            # Multi-pillar universal endpoints (Image, Audio, Doc)
│   ├── core/                         # Forensic engines & analytical consensus
│   │   ├── consensus.py              # Calibrated domain-aware decision fusion
│   │   ├── feature_schema.py         # Authoritative 24-dim physical feature schema
│   │   ├── pillar1_engine.py         # Vision Transformer spectral analyzer
│   │   ├── pillar2_engine.py         # Video pipeline wrapper & exporter
│   │   ├── pillar3_engine.py         # Acoustic transformer & HPSS demixing
│   │   ├── pillar4_engine.py         # Benford's Law OCR invoice analyzer
│   │   ├── pillar5_engine.py         # Geometry RANSAC, SRM & hybrid ML classifier
│   │   └── xai/                      # Explainable AI (XAI) evidence generators
│   ├── models/                       # Data schemas and model interfaces
│   │   ├── model_interface.py        # Abstract interfaces for visual, temporal & audio models
│   │   └── schemas.py                # Pydantic models for status, scores, reports
│   ├── services/                     # Business logic and processing pipelines
│   │   ├── classifier.py             # Feature fusion & classification
│   │   ├── explainability.py         # Heatmaps & visual bounding boxes
│   │   ├── face_detector.py          # Haar & DNN face detection
│   │   ├── frame_extractor.py        # Video frame decoders
│   │   ├── lip_sync_analyzer.py      # Audio-visual synchronization
│   │   ├── metadata_analyzer.py      # EXIF, container, & compression analysis
│   │   ├── rppg_analyzer.py          # Biological blood volume pulse (rPPG)
│   │   ├── storage.py                # JSON file-based database service
│   │   ├── temporal_analyzer.py      # Frame sequence temporal coherence
│   │   ├── video_pipeline.py         # End-to-end video analysis orchestrator
│   │   └── visual_analyzer.py        # Spatial artifact inspection
│   ├── storage/                      # File storage (uploads, frames, results, xai)
│   ├── workers/                      # Background execution & CLI tasks
│   │   ├── run_analysis_cli.py       # Standalone video processing CLI
│   │   └── video_worker.py           # Asynchronous background job worker
│   ├── requirements.txt              # Backend dependencies
│   ├── detect.py                     # CLI Voice Deepfake Detector
│   └── test_eval.py                  # Evaluation suite runner
├── frontend/                         # Modern React + Vite + Tailwind CSS User Interface
│   ├── src/                          # UI components, pages, charts & XAI visualizers
│   └── package.json                  # Frontend npm dependencies
├── models/                           # Pretrained model checkpoints (ViT, Wav2Vec, ML bundles)
├── testing/                          # Benchmark media suite (Videos, Images, Audios, Documents)
├── training/                         # Model training pipelines & datasets (Pillars 1 to 5)
├── run_backend.bat                   # 1-click FastAPI backend launcher
├── run_fullstack.bat                 # 1-click Full-Stack launcher (Backend + Frontend)
└── start.bat                         # Unified system starter
```

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Recommended: 3.10 or 3.11 with virtual environment `venv`)
- Node.js 18+ & npm (for the React frontend)
- Tesseract OCR (Required for Pillar 4 Document Mode)
  - **Windows:** Install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Add `C:\Program Files\Tesseract-OCR` to `PATH`.

### 2. Python Environment Setup
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## 🖥️ Running the Application

### Option A: One-Click Full-Stack (Recommended)
Double-click `start.bat` or run:
```cmd
start.bat
```
This automatically starts:
- **FastAPI Backend:** `http://127.0.0.1:8000` (Swagger docs at `/docs`)
- **React Frontend:** `http://localhost:5173`

### Option B: Individual Launchers
```cmd
# Terminal 1 - Backend:
run_backend.bat

# Terminal 2 - Frontend:
cd frontend
npm run dev
```

---

## 🛠️ CLI Standalone Tools

### Audio Deepfake Detection (Pillar 3):
```bash
python backend/detect.py testing/Audios/Real_voice.mp3 --mode spoken
python backend/detect.py testing/Audios/AI_Voice.mp3 --mode spoken
```

### Video Authenticity Analysis:
```bash
python backend/workers/run_analysis_cli.py testing/Videos/real_face.mp4
```

### Multi-Pillar Test Suite Evaluation:
```bash
python backend/test_eval.py
```
