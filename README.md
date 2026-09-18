# Universal Synthetic Media Forensics Engine (USMFE)
## Multi-Pillar Deepfake Detection Framework

A state-of-the-art multimodal cyber-forensic engine designed to detect synthetic, AI-generated, and manipulated media across **Visual, Acoustic, Document, and Physical Geometry domains**.

---

## 🏛️ System Architecture: The Multi-Pillar Framework

Unlike traditional monolithic deepfake detectors that overfit to specific generative models, USMFE uses **orthogonal forensic pillars** fused via a **deterministic override mechanism** to withstand domain shift:

| Pillar | Domain / Modality | Architecture / Algorithm | Key Forensic Indicators |
| :--- | :--- | :--- | :--- |
| **Pillar 1** | **Face & Frequency Forensics** | Hugging Face Vision Transformer (`ViT-Base`) | High-frequency facial blending borders, color-space transitions, ImageNet-normalized spectral anomalies. |
| **Pillar 2** | **Biological Forensics** | Remote Photoplethysmography (`rPPG`) + FFT | Subdermal blood volume pulse (BVP) extraction and cardiac periodicity disruption. |
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

```text
Multi-Pillar_Deepfake_Detection/
├── app_streamlit.py                  # Main Unified Web Application (Pillars 1, 3, 4, 5)
├── detect.py                         # Standalone Pillar 3 Audio & Voice Forensics CLI
├── requirements.txt                  # Python dependencies
├── Pillar 1/                         # Vision Transformer face forensics
│   ├── checkpoints/                  # Trained ViT PyTorch checkpoints (.pth)
│   ├── train_pillar1_vit.py          # Fine-tuning pipeline for ViT
│   └── dataset_adapter.py            # DataLoader & ImageNet augmentation pipeline
├── Pillar 3/                         # Voice and acoustic classification
├── Pillar 4/                         # Benford's Law OCR invoice forensics
│   └── invoice_*.png                 # Sample documents & invoices
├── Pillar 5/                         # Physical Geometry & Steganography
│   ├── extract_features.py           # 24-dimensional physical & SRM feature extractor
│   ├── feature_schema.py             # Schema definition for 24 physics features
│   ├── pillar5_pipeline.py           # Regularized ensemble & feature reducer
│   ├── pillar5_ml_model_v2.pkl       # Authoritative regularized hybrid model bundle
│   └── testing/                      # Test suite images (AI vs Real)
└── test_audio/                       # Test audio clips (.wav)
```

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Multi-Pillar_Deepfake_Detection.git
cd Multi-Pillar_Deepfake_Detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (Required for Pillar 4 Document Mode)
- **Windows:** Download and run the installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Ensure it is added to your System `PATH` (`C:\Program Files\Tesseract-OCR`).
- **Linux:**
  ```bash
  sudo apt update && sudo apt install tesseract-ocr
  ```
- **macOS:**
  ```bash
  brew install tesseract
  ```

---

## 📦 Large Checkpoints & Test Assets (Non-Git Binaries)

Due to file size limitations, binary checkpoints are excluded from Git via `.gitignore`. If pulling this repository fresh, ensure the following files are placed in their respective folders:

| File Name | Target Destination | Purpose |
| :--- | :--- | :--- |
| `best_pillar1_vit_v2.pth` | `Pillar 1/checkpoints/` | Fine-tuned Vision Transformer weights |
| `pillar5_ml_model_v2.pkl` | `Pillar 5/` | Fused 160-dim Voting Classifier Ensemble |
| `testing/` | `Pillar 5/testing/` | Pre-configured test suite (AI vs Real images) |
| `sample_test.wav` | `test_audio/` | Pre-configured voice forensic test audio |

---

## 🖥️ Running the Application

Launch the unified dashboard:
```bash
streamlit run app_streamlit.py
```
Open your browser at `http://localhost:8501`.

### Available Analysis Modes:
1. **Universal Multi-Pillar Media Analysis:** Evaluates images simultaneously through Pillar 1 (ViT), Pillar 4 (Document OCR), and Pillar 5 (Shadow RANSAC & PRNU Steganalysis) with the consensus banner and deterministic override.
2. **Pillar 3: Voice & Audio Synthetic Speech Forensics:** Upload voice recordings or music tracks to detect cloned speech and TTS synthesis.
3. **Pillar 4: Document, Invoice & PDF Statistical Forensics:** Upload invoices, receipts, or PDF contracts to detect forged financial figures via Benford's Law.

---

## 🛠️ CLI Standalone Tools

### Audio Deepfake Detection (Pillar 3):
```bash
# Spoken speech or phone calls
python detect.py test_audio/sample_test.wav --mode spoken

# Music or songs (Harmonic separation applied)
python detect.py path/to/song.mp3 --mode music
```

---

## 📊 Evaluation & Verification

To run feature extraction or test model outputs:
```bash
# Verify Pillar 5 feature extraction and sensor PRNU metrics
python -c "from Pillar_5.extract_features import extract_physics_vector; print(extract_physics_vector('Pillar 5/testing/AI_img1.jpeg'))"
```
