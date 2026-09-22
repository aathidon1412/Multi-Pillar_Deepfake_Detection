# Video Authenticity Detector (AuthentiGuard AI)
**USMFE Pillar 2: Multi-Pillar Deepfake & Video Forgery Detection Engine**

An AI-powered, full-stack forensic web application that analyzes uploaded videos to determine whether they are:
1. **REAL** — genuine authentic video with natural physical and biological consistency
2. **AI_GENERATED** — synthesized or substantially altered using deepfake, face-swap, or neural generative techniques
3. **FORGED** — manipulated through digital editing, splicing, frame cutting, or motion alteration

The system uses **JSON files as its primary persistence layer** (zero external SQL or NoSQL database required), explains **why** it reached each verdict, extracts and annotates **suspicious keyframes**, and features an interactive **timeline scrubber**.

---

## 1. Project Overview

Digital video manipulation has advanced rapidly with generative AI and sophisticated editing suites. AuthentiGuard AI provides a multi-pillar defense that simultaneously assesses:
* **Spatial / Visual Artifacts**: Facial boundary discontinuities, skin texture anomalies, high-frequency energy attenuation.
* **Temporal Stability**: Inter-frame flickering, sudden luminance jumps, optical flow displacement, and identity jitter.
* **Audio & Voice Forensics**: Synthetic voice signatures, acoustic frequency rolloff, and speech consistency.
* **Lip-Sync Coherence**: Cross-correlation between mouth aspect ratio (MAR) movements and spoken speech audio energy envelopes.
* **Forensic Metadata**: Codec, container, and encoder compression signatures.

---

## 2. Features

* **Multi-Format Support**: Upload MP4, MOV, AVI, MKV, and WEBM videos up to 500MB.
* **Live Video Preview**: Immediate playback preview before starting forensic inspection.
* **Live Multi-Stage Pipeline Stepper**: Real-time progress bar (0%–100%) tracking 10 distinct processing stages.
* **3-Way Probabilistic Classification**: Clear percentage breakdown for `REAL`, `AI_GENERATED`, and `FORGED`.
* **Explainable Evidence**: Technical decomposition across all 5 forensic channels with severity badges.
* **Interactive Suspicious Timeline**: Video player synchronized with flagged timestamp pins. Clicking a pin jumps directly to the anomaly.
* **Suspicious Keyframe Gallery**: High-resolution gallery of extracted anomaly frames with bounding boxes and zoom modal.
* **Downloadable JSON Reports**: Export complete machine-readable forensic verdicts.
* **Zero-Database History**: Dynamic history dashboard scanning `storage/results/*.json`.

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite, Tailwind CSS, Lucide Icons, Axios |
| **Backend** | Python 3.11, FastAPI, Starlette, Uvicorn |
| **Video Engine** | OpenCV (`cv2`), `imageio-ffmpeg` (bundled FFmpeg 7.1) |
| **AI / Machine Learning** | PyTorch, Hugging Face Transformers (`dima806/deepfake_vs_real_image_detection`), Torchvision |
| **Audio Processing** | FFmpeg PCM extraction, SciPy (`scipy.io.wavfile`, `scipy.signal`), NumPy |
| **Storage** | Local Filesystem + Atomic JSON document store |

---

## 4. Folder Structure

```text
video-authenticity-detector/
│
├── backend/
│   ├── main.py                     # FastAPI server, static mounts & CORS
│   ├── config.py                   # Storage paths, limits & thresholds
│   ├── requirements.txt            # Python dependencies
│   │
│   ├── routes/
│   │   ├── upload.py               # POST /api/upload
│   │   ├── analysis.py             # POST /api/analyze, GET /api/status, GET /api/result
│   │   └── history.py              # GET /api/history, DELETE /api/history/{id}
│   │
│   ├── services/
│   │   ├── json_storage.py         # Safe atomic JSON persistence
│   │   ├── video_processor.py      # Video validation, duration/fps inspection
│   │   ├── frame_extractor.py      # Configurable time-based frame sampling
│   │   ├── face_detector.py        # OpenCV Haar/DNN face detection & ROI crop
│   │   ├── visual_analyzer.py      # Texture, boundary seam, and frequency analysis
│   │   ├── temporal_analyzer.py    # Flickering, optical motion, and jitter
│   │   ├── audio_analyzer.py       # FFmpeg audio extraction & synthetic voice probe
│   │   ├── lip_sync_analyzer.py    # Mouth motion vs speech audio correlation
│   │   ├── metadata_analyzer.py    # Codec and container forensic signatures
│   │   ├── classifier.py           # Adaptive multi-pillar feature fusion
│   │   ├── explainability.py       # Keyframe annotation & evidence synthesis
│   │   └── cleanup.py              # Temporary frame cleanup
│   │
│   ├── models/
│   │   └── model_interface.py      # Abstract model interface with PyTorch/HF & heuristic fallback
│   │
│   └── tests/
│       └── test_pipeline.py        # Automated end-to-end multi-pillar pipeline test
│
├── frontend/
│   ├── src/
│   │   ├── components/             # Navbar, VideoPlayer, SuspiciousTimeline, SuspiciousGallery, etc.
│   │   ├── pages/                  # UploadPage, ProcessingPage, ResultPage, HistoryPage, ArchitecturePage
│   │   ├── services/               # Axios API client
│   │   ├── App.jsx                 # App root & navigation router
│   │   └── index.css               # Cyber-forensic dark styling
│   ├── package.json
│   └── vite.config.js
│
├── storage/
│   ├── uploads/                    # Persistent uploaded original videos
│   ├── frames/                     # Temporary frame extraction cache (auto-cleaned)
│   ├── suspicious_frames/          # Saved keyframes exhibiting high anomalies
│   └── results/                    # Permanent JSON report documents (VID_xxx.json)
│
├── README.md
└── .gitignore
```

---

## 5. Installation & Setup

### Prerequisites
* Python 3.10+ (tested with 3.11)
* Node.js 18+ and npm

### 1. Backend Setup
From the root workspace:
```bash
# Activate virtual environment or install dependencies
pip install -r "Pillar 2/video-authenticity-detector/backend/requirements.txt"
```

The system automatically resolves FFmpeg via `imageio-ffmpeg` without requiring a manual system install of FFmpeg.

### 2. Frontend Setup
```bash
cd "Pillar 2/video-authenticity-detector/frontend"
npm install
```

---

## 6. How to Run

### Start the Backend Server (Port 8000)
```powershell
cd "d:\Projects\Mini Project\Multi-Pillar_Deepfake_Detection"
& .venv\Scripts\python.exe -m uvicorn backend.main:app --app-dir "Pillar 2/video-authenticity-detector" --host 127.0.0.1 --port 8000 --reload
```

### Start the Frontend Dev Server (Port 5173)
```powershell
cd "d:\Projects\Mini Project\Multi-Pillar_Deepfake_Detection\Pillar 2\video-authenticity-detector\frontend"
npm run dev
```

Open your browser at:
`http://localhost:5173`

---

## 7. API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Uploads and validates video file; saves to `storage/uploads/` |
| `POST` | `/api/analyze/{video_id}` | Dispatches background multi-pillar analysis pipeline |
| `GET` | `/api/status/{video_id}` | Polls progress percentage (0–100%) and current pipeline stage |
| `GET` | `/api/result/{video_id}` | Retrieves full JSON forensic analysis report |
| `GET` | `/api/history` | Scans `storage/results/*.json` and returns all records |
| `DELETE` | `/api/history/{video_id}` | Deletes JSON report, uploaded video, and suspicious frames |
| `GET` | `/storage/...` | Direct static access to uploaded videos and suspicious keyframe images |

---

## 8. JSON Storage Format

Each analyzed video produces a standalone document in `storage/results/{video_id}.json`:

```json
{
  "video_id": "VID_001",
  "file": {
    "original_filename": "sample.mp4",
    "stored_filename": "VID_001.mp4",
    "format": "mp4",
    "size_mb": 15.7
  },
  "video_details": {
    "duration_seconds": 32.5,
    "resolution": "1920x1080",
    "fps": 30.0,
    "frame_count": 975
  },
  "metadata": {
    "codec": "h264",
    "encoder": "Lavc",
    "metadata_status": "normal"
  },
  "analysis": {
    "face_detection": {
      "faces_detected": true,
      "face_count": 30
    },
    "visual": {
      "score": 0.82,
      "status": "suspicious",
      "anomalies": ["Facial texture inconsistency", "Unusual face boundary"]
    },
    "temporal": {
      "score": 0.76,
      "status": "suspicious",
      "flickering_detected": true,
      "motion_inconsistency": true
    },
    "audio": {
      "available": true,
      "score": 0.32,
      "status": "normal"
    },
    "lip_sync": {
      "available": true,
      "score": 0.71,
      "status": "slight_anomaly"
    }
  },
  "classification": {
    "prediction": "AI_GENERATED",
    "scores": {
      "real": 0.08,
      "ai_generated": 0.87,
      "forged": 0.05
    },
    "confidence": 0.87
  },
  "suspicious_frames": [
    {
      "frame_number": 124,
      "timestamp_seconds": 4.13,
      "image": "suspicious_frames/VID_001_frame_124.jpg",
      "reason": "Facial texture inconsistency",
      "score": 0.91
    }
  ],
  "processing": {
    "status": "completed",
    "processed_at": "2026-09-18T13:30:00",
    "processing_time_seconds": 18.4
  }
}
```

---

## 9. Automated Testing

Run the automated integration test to verify all 14 pipeline stages:
```powershell
cd "d:\Projects\Mini Project\Multi-Pillar_Deepfake_Detection"
& .venv\Scripts\python.exe "Pillar 2/video-authenticity-detector/backend/tests/test_pipeline.py"
```

---

## 10. Limitations & Future Work

1. **Model Swap**: The interface in `backend/models/model_interface.py` is ready to plug in customized PyTorch weights trained on FaceForensics++ or DFDC.
2. **Audio Model Depth**: Advanced neural vocoder detection (e.g. RawNet2 / AASIST) can be added into `predict_audio()`.
3. **Biological rPPG**: Integration with green-channel remote photoplethysmography heart-rate estimation (as prototyped in `Pillar 2/pillar2_hybrid.py`).
