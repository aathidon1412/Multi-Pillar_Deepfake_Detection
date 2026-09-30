# Antigravity Codebase & UI Audit Specification: Universal Synthetic Media Forensics Engine (USMFE / AuthentiGuard AI)

**Document Target:** `codebase-spec.md`  
**Workspace:** Multi-Pillar Deepfake Detection System  
**Analysis Engine:** Extracted via Source Inspection & Graphify AST Knowledge Graph  

---

## 1. System Core & Primary Workflows

### 1.1 Core Purpose & Target Persona
- **Core Purpose:** The application provides a unified, multi-modal forensics intelligence platform designed to detect, authenticate, and explain synthetic media manipulations across video, static images, recorded voice/music audio, and financial/official documents. Rather than relying on a single black-box classification, it executes an orthogonal multi-pillar decomposition (Visual ViT, Biological rPPG, Optical Flow Temporal Consistency, Mel-scale/HPSS Audio Acoustics, Document Benford's Law Chi-Square goodness-of-fit, and 3D Illumination/Shadow Physics) with verifiable forensic telemetry and keyframe-level explainability (XAI).
- **Primary Persona:**
  - **Forensic Analysts & OSINT Investigators:** Requiring frame-by-frame timestamps, anomaly confidence distributions, and downloadable forensic telemetry (`.json` audit trails).
  - **Financial & Fraud Auditors:** Validating the authenticity of invoices, scanned receipts, and regulatory documents against digit manipulation via Benford's law.
  - **Media & Journalism Verification Teams:** Rapidly triage deepfakes, synthetic speech cloning, voice swaps, face swaps, diffusion morphing, and spliced video cuts before broadcast.

---

### 1.2 Critical User Journeys

#### Journey 1: Video Forensics Pipeline (Pillar 2 Core Flow)
1. **Upload & Ingestion:** User drags and drops or browses an MP4, MOV, AVI, MKV, or WEBM video file on `UploadPage.jsx`. Client-side validation checks extension compatibility.
2. **REST Upload & Job Initiation:** `POST /api/upload` uploads video to `storage/uploads/{video_id}.ext`. Upon receiving `video_id`, client invokes `POST /api/analyze/{video_id}` which spawns `execute_video_analysis_pipeline` in a background worker thread.
3. **Reactive Polling & Stage Visualization:** User is navigated to `ProcessingPage.jsx`. A `setInterval` (900ms) polls `GET /api/status/{video_id}`. Real-time feedback transitions through 10 stages:
   - `Preprocessing (10%)` → `Metadata Analysis (20%)` → `Extracting Frames (30%)` → `Detecting Faces (45%)` → `Visual Analysis (60%)` → `Temporal Analysis (72%)` → `Audio Analysis (80%)` → `Lip-Sync Analysis (86%)` → `Classification (92%)` → `Generating Report & Keyframes (96%)` → `Completed (100%)`.
4. **Interactive Evidence Review:** Upon status `completed`, routing updates to hash `#result/{video_id}` (`ResultPage.jsx`):
   - **Verdict Hero:** Displays 3-way probabilistic classification (`REAL`, `AI_GENERATED`, or `FORGED`) with confidence gauge and stacked probability bars.
   - **Synced Evidence Player & Timeline:** Synchronizes HTML5 video playback with `SuspiciousTimeline.jsx` pins. Clicking pins seeks playback to exact artifact timestamps.
   - **Pillar Evidence Decomposition:** 5-card breakdown covering Visual, Temporal, Audio, Lip-Sync, and File Metadata channels.
   - **Keyframe Gallery:** Grid of annotated bounding box crops with zoom modal and anomaly metrics.
5. **Telemetry Export:** User clicks **"Download JSON"** to export `storage/results/{video_id}.json` for legal or chain-of-custody documentation.

#### Journey 2: Universal Image & Shadow Physics Forensics (Pillars 1 & 5)
1. **Target Selection:** User navigates to **"Pillars 1 & 5: Image"** (`Pillars1And5Page.jsx`) and submits an image (JPG, PNG, WEBP, BMP).
2. **Parallel Neural & Physical Inference:** Client sends multipart file to `POST /api/pillars/image`.
   - **Pillar 1:** Vision Transformer (`dima806/deepfake_vs_real_image_detection`) evaluates patch-level frequency artifacts and synthetic texture signatures.
   - **Pillar 5:** Evaluates 2D gradient magnitude Laplacian variance, line consistency, and shadow/lighting ray convergence (RANSAC inlier ratio).
3. **Consensus Computation:** Dynamic fusion merges neural fake probability with shadow inlier ratio `(0.5 * (1 - p1_score)) + (0.5 * p5_inlier_ratio)`.
4. **Display:** Returns dual-pillar breakdown cards, inlier counts, lighting convergence stats, and consolidated verdict.

#### Journey 3: Acoustic & Speech Cloning Forensics (Pillar 3)
1. **File Selection & Mode Configuration:** User navigates to **"Pillar 3: Audio"** (`Pillar3Page.jsx`), uploads audio (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`), and selects mode:
   - `Spoken Voice (Conversational)`
   - `Song / Music Track (HPSS Demixing)`
2. **Audio Analysis Request:** Client calls `POST /api/pillars/audio` with `mode` and file.
3. **Backend Execution:** Invokes `classify_audio()` (using librosa/PyTorch): runs Mel-frequency spectral analysis, neural vocoder artifact detection, and harmonic-to-percussive source separation if music mode is chosen.
4. **Result Card Rendering:** Displays Real vs. Fake confidence, duration, sample rate, demixing status, and voice synthesis badges.

#### Journey 4: Document & Invoice Statistical Forensics (Pillar 4)
1. **Document Upload:** User navigates to **"Pillar 4: Document"** (`Pillar4Page.jsx`) and uploads PDF, PNG, JPG, or TIFF.
2. **OCR & Leading Digit Extraction:** Client submits to `POST /api/pillars/document`. Backend utilizes PyMuPDF (`fitz`) for PDFs or Tesseract OCR for images to isolate all leading digits `[1-9]` matching `\b[1-9][0-9,]*\.?[0-9]*\b`.
3. **Benford Conformity & Chi-Square Testing:** Calculates Mean Absolute Error (MAE) and Chi-Square goodness-of-fit against theoretical $\log_{10}(1 + 1/d)$ logarithmic distribution with degree of freedom = 8.
4. **Visual Statistical Comparison:** Renders comparative bar distribution (`Observed Frequencies %` vs `Theoretical Benford %`), MAE, Chi-Square statistic, and p-value verdict.

#### Journey 5: Historical Audits & Registry Management
1. **History Retrieval:** User navigates to **"History"** (`HistoryPage.jsx`). Client fetches `GET /api/history`.
2. **Filtering & Searching:** In-memory client filters by query (filename or video ID) and verdict (`ALL`, `REAL`, `AI_GENERATED`, `FORGED`).
3. **Action Triggers:**
   - **View:** Opens `#result/{video_id}` directly.
   - **Delete:** Invokes `DELETE /api/history/{video_id}`, which purges the stored `.json`, original uploaded video, extracted frames, and thumbnail cache atomically.

---

### 1.3 Domain Entities & Data Models

#### 1. Forensic Report (`FinalReport` JSON Model)
Persisted on filesystem at `storage/results/{video_id}.json`.

```json
{
  "video_id": "VID_20260930_123456_abc",
  "file": {
    "original_filename": "interview_source.mp4",
    "stored_filename": "VID_20260930_123456_abc.mp4",
    "format": "mp4",
    "size_mb": 24.85
  },
  "video_details": {
    "duration_seconds": 18.5,
    "resolution": "1920x1080",
    "fps": 30.0,
    "frame_count": 555
  },
  "metadata": {
    "codec": "h264",
    "encoder": "Lavf58.76.100",
    "metadata_status": "normal | suspicious | inconclusive"
  },
  "analysis": {
    "face_detection": {
      "faces_detected": true,
      "face_count": 1
    },
    "visual": {
      "score": 0.88,
      "status": "suspicious | slight_anomaly | normal | inconclusive",
      "anomalies": ["Facial boundary warping", "High-frequency blur", "GAN blend seam"]
    },
    "temporal": {
      "score": 0.76,
      "status": "suspicious | slight_anomaly | normal | inconclusive",
      "flickering_detected": true,
      "motion_inconsistency": true
    },
    "audio": {
      "available": true,
      "score": 0.12,
      "status": "normal"
    },
    "lip_sync": {
      "available": true,
      "score": 0.82,
      "status": "suspicious"
    }
  },
  "classification": {
    "prediction": "REAL | AI_GENERATED | FORGED",
    "confidence": 0.94,
    "scores": {
      "real": 0.04,
      "ai_generated": 0.94,
      "forged": 0.02
    }
  },
  "suspicious_frames": [
    {
      "frame_number": 142,
      "timestamp_seconds": 4.73,
      "score": 0.89,
      "reason": "Lip-sync desync & facial boundary blend artifact",
      "image": "suspicious_frames/VID_xxx_frame_142.jpg"
    }
  ],
  "processing": {
    "status": "completed | processing | failed",
    "processed_at": "2026-09-30T20:25:00",
    "processing_time_seconds": 6.42
  }
}
```

#### 2. Progress Tracker Model (`_PROGRESS_REGISTRY`)
In-memory dictionary in `json_storage.py`:
- `video_id` (str, primary key)
- `status` (`"processing" | "completed" | "failed"`)
- `progress` (int: 0 to 100)
- `current_stage` (str: Human-readable stage)
- `error` (Optional[str])
- `updated_at` (float: unix timestamp)

#### 3. Image Forensics Consensus Entity (Pillars 1 & 5)
- `consensus_verdict`: `"AUTHENTIC MEDIA" | "FAKE (SYNTHETIC AI ANOMALY)"`
- `is_real`: boolean
- `consensus_confidence`: float percentage (0 - 100)
- `pillar1`: `{ verdict, confidence, fake_score, model }`
- `pillar5`: `{ verdict, confidence, inliers, total_lines, inlier_ratio, engine }`
- `filename`: string

#### 4. Document Benford Entity (Pillar 4)
- `applicable`: boolean (requires $\ge 10$ extracted digits)
- `verdict`: `"AUTHENTIC" | "FORGED / AI-GENERATED"`
- `confidence`: float percentage
- `digits_count`: integer
- `mae`: float Mean Absolute Error
- `chi_square`: float Chi-Square statistic
- `p_value`: float p-value
- `obs_freqs`: list of 9 floats (observed digit percentages 1..9)
- `expected_freqs`: list of 9 floats (theoretical Benford percentages 1..9)

---

## 2. Existing Frontend & UI Audit

### 2.1 Current Tech Stack
- **Framework:** React 19 (`react` ^19.2.8, `react-dom` ^19.2.8) built on **Vite 8** (`vite` ^8.3.0).
- **Styling Engine:** Tailwind CSS v3 (`tailwindcss` ^3.4.17, `postcss` ^8.5.28, `autoprefixer` ^10.6.1) configured with dark-mode cyber aesthetic (`#07090e` base, `.glass-panel` backdrop-blur-md, and cyber grid background).
- **Typography:** Google Fonts: `Inter`, `Outfit`, and `JetBrains Mono`.
- **Icon Library:** Lucide React (`lucide-react` ^1.47.0).
- **HTTP Client:** Axios (`axios` ^1.20.0) configured with `/api` proxy prefix and 60-second timeouts.
- **State Management:** Local React state (`useState`, `useEffect`, `useRef`), window hash-based pseudo-routing (`#result/{id}`), and cross-component callback lifecycles. No Redux, Zustand, or TanStack Query.
- **Linter:** Oxlint (`oxlint` ^1.81.0).

---

### 2.2 Route & View Inventory

The application uses hash navigation and conditional view-switching within [App.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/App.jsx):

| Navigation Tab / Hash | Component File | View Description |
| :--- | :--- | :--- |
| `activeTab === 'video'` / `'upload'` | [UploadPage.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/UploadPage.jsx) | Video dropzone, supported format tags, format badge, and upload action. |
| `activeTab === 'processing'` | [ProcessingPage.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/ProcessingPage.jsx) | 10-stage animated pipeline progress gauge with real-time status polling. |
| `activeTab === 'result'` or `#result/:videoId` | [ResultPage.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/ResultPage.jsx) | Complete forensic dossier: Verdict Hero, Synced Video Player, Timeline, Pillar Breakdown, Suspicious Frame Gallery. |
| `activeTab === 'image'` | [Pillars1And5Page.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/Pillars1And5Page.jsx) | Dual-model image analysis (Vision Transformer + Shadow Geometry RANSAC). |
| `activeTab === 'audio'` | [Pillar3Page.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/Pillar3Page.jsx) | Audio deepfake inspector with vocal vs. music HPSS toggle. |
| `activeTab === 'document'` | [Pillar4Page.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/Pillar4Page.jsx) | PDF/Invoice OCR digit parser & Benford's Law Chi-Square distribution grapher. |
| `activeTab === 'history'` | [HistoryPage.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/HistoryPage.jsx) | Historical reports table with search filter, metric badges, and deletion controls. |
| `activeTab === 'architecture'` | [ArchitecturePage.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/pages/ArchitecturePage.jsx) | ASCII workflow diagram, pipeline architecture docs, and technical specifications. |

---

### 2.3 Component Breakdown

#### Form Inputs & Buttons
- **Dropzones:** Custom dashed borders (`border-dashed rounded-xl p-8`) with drag-and-drop state toggles in `UploadPage`, `Pillars1And5Page`, `Pillar3Page`, and `Pillar4Page`.
- **Search & Filter Inputs:** Filter pills and search bar in `HistoryPage.jsx` with real-time substring filtering.
- **Button Variants:**
  - *Primary Action:* `bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 shadow-glow-cyan text-white font-bold`.
  - *Secondary / Download:* `bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700`.
  - *Destructive:* `p-2 hover:bg-rose-950/80 text-rose-400 border border-transparent hover:border-rose-800`.
  - *Mode Switcher Pills:* Dual button selector (`bg-purple-950/60` active vs `bg-slate-900` inactive).

#### Data Displays & Metric Cards
- **Verdict Hero Banner** ([ProbabilityChart.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/components/ProbabilityChart.jsx)): Dynamic color theming:
  - `REAL`: Emerald gradient (`from-emerald-950/40 to-slate-900/80`, `border-emerald-500/50`).
  - `AI_GENERATED`: Rose gradient (`from-rose-950/40 to-slate-900/80`, `border-rose-500/50`).
  - `FORGED`: Amber gradient (`from-amber-950/40 to-slate-900/80`, `border-amber-500/50`).
- **Probability Bars:** CSS width percentage bars comparing `AI_GENERATED` (rose), `REAL` (emerald), and `FORGED` (amber).
- **Pillar Evidence Decomposition Grid** ([PillarBreakdown.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/components/PillarBreakdown.jsx)): 5-card responsive grid displaying channel score, anomaly tags, and status badges (`Suspicious`, `Slight Anomaly`, `Normal`, `N/A`, `Inconclusive`).
- **Interactive Scrubber & Anomaly Pins** ([SuspiciousTimeline.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/components/SuspiciousTimeline.jsx)): Dual-layer timeline tracking current playback time against anomalous keyframe pins with pulsating indicator rings and hover tooltips.
- **Keyframe Gallery with Modal Zoom** ([SuspiciousGallery.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/components/SuspiciousGallery.jsx)): Thumbnail grid with overlay pills showing timestamp and score, plus lightbox modal.
- **Synchronized Video Player** ([VideoPlayer.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/components/VideoPlayer.jsx)): Custom HTML5 video wrapper providing Play/Pause, Reset, Mute, Scrubber time tracking, HUD watermark, and external seek synchronization.

#### Navigation Elements
- **Header Bar** ([Navbar.jsx](file:///d:/College%20Studies/Mini%20Project/Pillar%202/video-authenticity-detector/frontend/src/components/Navbar.jsx)): Sticky top glassmorphism bar with responsive tabs, brand badge (`AuthentiGuard AI - 5 Pillars`), and active tab indicator highlights.
- **System Footer:** System state indicator and modality breakdown.

---

### 2.4 UX & Visual Debt Identified

1. **Hash Routing & Direct Link Fragility:**
   - Routing uses raw hash parsing (`#result/{videoId}`). If a user opens a link without an existing ID or uses the browser back/forward buttons repeatedly, tab state can get out of sync with `activeTab` state. No formal client-side router (like React Router) is installed.
2. **Missing Empty & Skeleton States:**
   - When loading `ResultPage`, a generic centered spinning loader is displayed. If telemetry fields (e.g. metadata or lip sync) are missing, components fallback abruptly to static placeholders or unstyled N/A chips without skeleton placeholders.
3. **Responsive Breakpoints & Timeline Pin Overflow:**
   - In `SuspiciousTimeline.jsx`, timeline marker pins calculate left percentage via `left: calc(1rem + ratio * (100% - 2rem) - 8px)`. On small mobile viewports (<640px), close consecutive pins overlap, causing tooltips to clip offscreen.
4. **Lack of Tab Persistence on Refresh:**
   - Reloading the page while on `activeTab === 'image'`, `'audio'`, `'document'`, or `'history'` forces the state back to `'video'` unless a `#result/` hash is active.
5. **Chart Visual Redundancy in Pillar 4:**
   - In `Pillar4Page.jsx`, Benford's Law comparison is rendered using custom stacked HTML divs rather than SVG/Canvas charts, leading to vertical bar misalignment when text wraps on narrow screens.
6. **Error Recovery & Re-try Handling:**
   - If an upload or analysis fails mid-pipeline on `ProcessingPage.jsx`, the user only sees an error alert and a "Cancel" button; there is no single-click "Retry Failed Analysis" action.

---

## 3. Stitch Extraction Brief (Output Summary)

### 3.1 The Single Most Critical Screen That Needs Redesign
**The Forensic Dossier & Evidence Screen (`ResultPage.jsx` / `#result/:videoId`)**

*Rationale:* This is the terminal destination and core value proposition of the entire USMFE platform. Forensic investigators, fraud analysts, and court experts spend 90% of their operational time on this screen evaluating multi-pillar signals, inspecting flagged frames, cross-referencing acoustic/visual anomalies, and seeking verification proofs.

### 3.2 Specification: Data Points, Actions, Filters, & Indicators Required

#### A. Header Bar & Metadata Dossier
- [x] **Video Identifier:** `video_id` badge with single-click "Copy ID" clipboard button.
- [x] **File Specifications:** Original filename, stored format extension, file size (MB), resolution ($W \times H$), frame count, and FPS.
- [x] **Container Metadata:** Video codec, audio codec, encoder footprint (`Lavf` / ffmpeg signatures), and container sanity status.
- [x] **Execution Telemetry:** Timestamp processed (ISO formatted) and pipeline execution duration in seconds.
- [x] **Actions:**
  - Primary: "Analyze Another Video" (reset state).
  - Secondary: "Export Forensic JSON" (full raw payload).
  - New High-Value Action: "Generate PDF Forensic Report" (printable audit summary).

#### B. Primary Verdict & Confidence Hero
- [x] **Tri-State Verdict Indicator:** High-prominence badge (`REAL`, `AI_GENERATED`, or `FORGED`) with semantic color scheme (Emerald, Rose, Amber).
- [x] **Confidence Gauge:** Percentage metric (e.g., `94%`) with calibration subtext.
- [x] **Probabilistic Distribution Bar:** Stacked 3-tier probability distribution showing individual scores for `real`, `ai_generated`, and `forged`.
- [x] **Executive Forensic Summary:** One-paragraph explainable verdict summarizing which modalities triggered the classification.

#### C. Bi-Directional Synchronized Evidence Player & Scrubber
- [x] **Active Video Surface:** Aspect-video player with HUD playback overlay (Current Time / Total Duration).
- [x] **Transport Controls:** Play/Pause, Rewind 5s, Forward 5s, Playback Rate selector ($0.25\times, 0.5\times, 1\times, 2\times$ for frame-by-frame scrutiny), Mute/Volume toggle.
- [x] **Interactive Marker Timeline:** Horizontal scrubber bar with color-coded pins for each suspicious keyframe.
- [x] **Active Playhead Tracking:** Real-time scrubbing that automatically highlights the closest suspicious keyframe card below as playback progresses.

#### D. Multi-Pillar 5-Channel Evidence Decomposition Grid
Every card must display Channel Status Badge (`Suspicious`, `Slight Anomaly`, `Normal`, `N/A`), Anomaly Percentage Gauge, and Concrete Findings:
1. **Channel 1 — Spatial / Visual Artifacts:** Boundary warping, texture blurring, GAN grid seams, sensor noise absence.
2. **Channel 2 — Temporal & Motion Coherence:** Optical flow continuity, frame flickering, unnatural motion displacement, splicing discontinuities.
3. **Channel 3 — Acoustic & Voice Forensics:** Spectral roll-off, neural vocoder robotic artifacts, unnatural pitch variance, speech synthesis markers.
4. **Channel 4 — Lip-Sync & Audio-Visual Coherence:** Phoneme-to-viseme temporal offset, lip boundary alignment error.
5. **Channel 5 — Physical Geometry & Sensor Consistency:** Facial symmetry, illumination vector convergence, PRNU sensor noise footprint.

#### E. Suspicious Keyframe Evidence Gallery & Lightbox
- [x] **Keyframe Thumbnail Cards:** Annotated bounding-box frame crops with:
  - Timestamp indicator (`MM:SS.ms`).
  - Anomaly severity score (`%`).
  - Specific detection reason tag.
  - Quick action: "Jump Player to Frame".
- [x] **Full-Screen Forensic Lightbox:** Zoom-pan inspection tool with toggleable anomaly overlays (heatmaps, face landmarks, bounding boxes).

---
*Audit completed and extracted directly from repository source code.*
