# Graph Report - Mini Project  (2026-09-22)

## Corpus Check
- 129 files · ~4,216,763 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 19 file(s) not represented in the graph (top: (none) 6, .bat 4, .pth 2)

## Summary
- 698 nodes · 976 edges · 63 communities (49 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ac07d971`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pillar1_pipeline
- run_pillar5_training.py
- main
- Universal Synthetic Media Forensics Engine (USMFE)
- Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics
- classify_audio
- Module: Pillar 3 — Acoustic & Voice Deepfake Forensics
- 5. Visual Proofs & Explainability (XAI)
- 2. Iteration-by-Iteration Breakdown
- 2. Iteration-by-Iteration Breakdown
- analyze_shadows
- generate_artifacts
- __init__.py
- Pillar 2: Hybrid Visual Forensics & Biological rPPG Analysis
- Pillar V: Physical Geometry and Shadow Physics Forensics
- pillar2_hybrid.py
- Pillar 1: Vision Transformer Based Synthetic Image Detection
- Pillar 3: Acoustic Forensics
- Pillar 4/app.py
- Pillar 4: Financial Forensics and Statistical Anomalies
- Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law
- generate_pillar5_dataset.py
- Pillar 1: Vision Transformer Based Synthetic Image Detection
- Pillar 1: Vision Transformer Based Synthetic Image Detection
- Pillar 2: Biological Forensics via rPPG & Fast Fourier Transform
- Pillar 2: Biological Forensics via rPPG & Fast Fourier Transform
- Pillar 3: Acoustic Forensics
- Pillar 3: Acoustic Forensics
- Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law
- Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law
- prepare_temp_dataset.py
- pillar_5_interactive.py
- load_vit_model
- rules/graphify.md
- workflows/graphify.md
- App.jsx
- package.json
- model_interface.py
- test_pipeline.py
- load_result
- Video Authenticity Detector (AuthentiGuard AI)
- config.py
- inspect_video
- execute_video_analysis_pipeline
- pillars_integrated.py
- run_lip_sync_analysis
- save_uploaded_video
- visual_analyzer.py
- .detect_in_frame
- .oxlintrc.json
- analysis.py
- React + Vite
- get_history

## God Nodes (most connected - your core abstractions)
1. `execute_video_analysis_pipeline()` - 17 edges
2. `test_full_pipeline()` - 17 edges
3. `react` - 17 edges
4. `run_video_pipeline()` - 17 edges
5. `lucide-react` - 15 edges
6. `train_pillar1_pipeline()` - 11 edges
7. `load_result()` - 11 edges
8. `run_visual_analysis()` - 11 edges
9. `Video Authenticity Detector (AuthentiGuard AI)` - 11 edges
10. `save_result()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `run_video_pipeline()` --calls--> `run_feature_fusion_and_classification()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/classifier.py
- `run_video_pipeline()` --calls--> `extract_and_annotate_suspicious_frames()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/explainability.py
- `run_video_pipeline()` --calls--> `extract_sampled_frames()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/frame_extractor.py
- `run_video_pipeline()` --calls--> `save_result()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/json_storage.py
- `run_video_pipeline()` --calls--> `load_result()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/json_storage.py

## Import Cycles
- 3-file cycle: `core/__init__.py -> core/pillar3_engine.py -> detect.py -> core/__init__.py`

## Communities (63 total, 4 thin omitted)

### Community 0 - "train_pillar1_pipeline"
Cohesion: 0.07
Nodes (35): Compose, Dataset, GradScaler, no_grad, Optimizer, FaceDeepfakeDataset, get_pillar1_dataloaders(), make_samples_from_folders() (+27 more)

### Community 1 - "run_pillar5_training.py"
Cohesion: 0.08
Nodes (35): BaseEstimator, build_xy_matrices(), extract_efficientnet_embedding(), extract_physics_vector(), get_default_efficientnet(), device, Image, Module (+27 more)

### Community 2 - "main"
Cohesion: 0.18
Nodes (12): extract_efficientnet_embeddings(), main(), extract_single_image_features(), main(), Worker function for parallel tabular feature extraction. Takes a tuple…, calculate_intersection(), extract_deep_embeddings(), extract_multi_domain_features() (+4 more)

### Community 3 - "Universal Synthetic Media Forensics Engine (USMFE)"
Cohesion: 0.12
Nodes (16): 1. Clone the Repository, 2. Create and Activate Virtual Environment, 3. Install Dependencies, 4. Install Tesseract OCR (Required for Pillar 4 Document Mode), Audio Deepfake Detection (Pillar 3):, Available Analysis Modes:, 🛠️ CLI Standalone Tools, 📊 Evaluation & Verification (+8 more)

### Community 4 - "Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics"
Cohesion: 0.10
Nodes (19): 1.1 The Challenge of High-Resolution Generative Media, 1.2 Legacy Vulnerabilities & Root Causes of Detection Failure, 1. Introduction & Problem Diagnosis, 2. Multi-Domain Dataset Architecture, 3.1 Tabular Physics, Illumination & Steganalysis Features (24 Features), 3.2 Deep Visual Feature Embeddings (1,280 Features), 3. Comprehensive Feature Engineering (1,304 Dimensions), 4. Retraining Pipeline & Parallel Execution (+11 more)

### Community 5 - "classify_audio"
Cohesion: 0.43
Nodes (6): analyze_audio_composition(), classify_audio(), get_model(), main(), Classifies an audio file as REAL vs FAKE. Parameters: audio_path: path to audio…, Uses librosa to compute acoustic metrics and detect whether the input audio…

### Community 6 - "Module: Pillar 3 — Acoustic & Voice Deepfake Forensics"
Cohesion: 0.12
Nodes (16): 1. Executive Summary, 2.1 Mel-Scale Spectral Representation, 2.2 Neural Vocoder Artifact Fingerprinting, 2.3 Harmonic-to-Percussive Source Separation (HPSS), 2. Theoretical Formulation & Internal Mechanics, 3. Engineering Challenges & Solutions, 4. Benchmark Performance & Evaluation Metrics, 5. Visual Proofs & Explainability (XAI) (+8 more)

### Community 7 - "5. Visual Proofs & Explainability (XAI)"
Cohesion: 0.12
Nodes (16): 1. Executive Summary, 2.1 Benford's Law Logarithmic First-Digit Distribution, 2.2 Mathematical Conformity Metrics, 2. Algorithms & Internal Mathematical Framework, 3. Engineering Challenges & Resolutions, 4. Experimental Benchmark Results (412 Document Dataset), 5. Visual Proofs & Explainability (XAI), 6. Directory Artifacts & Reproducibility (+8 more)

### Community 8 - "2. Iteration-by-Iteration Breakdown"
Cohesion: 0.12
Nodes (15): 1. Executive Summary & Benchmark Overview, 2. Iteration-by-Iteration Breakdown, 3. Comparative Summary Table, 4. Codebase Artifacts & Integration, Final Classification Report (Iteration 4), Final Confusion Matrix (Iteration 4), Global Iteration Progression Matrix, 🔹 Iteration 0: Baseline Handcrafted Geometry (+7 more)

### Community 9 - "2. Iteration-by-Iteration Breakdown"
Cohesion: 0.12
Nodes (15): 1. Executive Summary & Benchmark Overview, 2. Iteration-by-Iteration Breakdown, 3. Comparative Summary Table, 4. Codebase Artifacts & Integration, Final Classification Report (Iteration 4), Final Confusion Matrix (Iteration 4), Global Iteration Progression Matrix, 🔹 Iteration 0: Baseline Handcrafted Geometry (+7 more)

### Community 10 - "analyze_shadows"
Cohesion: 0.23
Nodes (8): ForensicsHTTPHandler, process_all_test_images(), analyze_shadows(), calculate_intersection(), line_point_distance(), Calculate the intersection of two lines defined by two points each., Distance from a point to a line segment extended infinitely., Executes Pillar 5 of the Universal Synthetic Media Forensics Engine (USMFE).…

### Community 11 - "generate_artifacts"
Cohesion: 0.22
Nodes (12): calculate_benford_metrics(), extract_digits_from_csv(), extract_leading_digits(), extract_text_from_image(), generate_artifacts(), generate_decision(), Calculates observed, theoretical frequencies, MAE, and Chi-Square stats., Determines the verdict based on classification thresholds, dynamically scaled… (+4 more)

### Community 12 - "__init__.py"
Cohesion: 0.06
Nodes (51): predict_audio(), Takes an uploaded or recorded audio file path and mode ("spoken" or "music"): -…, detect_modality(), get_cached_pillar1(), get_cached_pillar5(), cache_resource, ===============================================================================…, Detects media modality from file extension. (+43 more)

### Community 13 - "Pillar 2: Hybrid Visual Forensics & Biological rPPG Analysis"
Cohesion: 0.22
Nodes (8): Algorithms & Mathematics Used, Challenges & Resolutions, Channel 1 — Spatial / Visual AI (Vision Transformer), Channel 2 — Biological rPPG (Remote Photoplethysmography with Velocity Filtering & EMA), Ensemble Verdict Logic, Experimental Findings, Metadata, Pillar 2: Hybrid Visual Forensics & Biological rPPG Analysis

### Community 14 - "Pillar V: Physical Geometry and Shadow Physics Forensics"
Cohesion: 0.25
Nodes (7): 1. Introduction, 2. Methodology, 3. Experimental Results, 4. Conclusion, Abstract, Pillar V: Physical Geometry and Shadow Physics Forensics, Universal Synthetic Media Forensics Engine (USMFE)

### Community 15 - "pillar2_hybrid.py"
Cohesion: 0.48
Nodes (6): analyze_video(), butterworth_bandpass(), compute_rppg_snr(), get_fake_score(), load_detector(), pillar2_hybrid.py - USMFE Pillar 2 Hybrid Deepfake Detector…

### Community 16 - "Pillar 1: Vision Transformer Based Synthetic Image Detection"
Cohesion: 0.33
Nodes (5): 1. Mathematical & Algorithmic Foundation, 2. Engineering Challenges & Solutions, 3. Experimental Findings & Analysis, 4. Experimental Metadata Table, Pillar 1: Vision Transformer Based Synthetic Image Detection

### Community 17 - "Pillar 3: Acoustic Forensics"
Cohesion: 0.33
Nodes (5): Engineering Challenges & Solutions, Implementation & Preprocessing, Mathematical Formulation, Pillar 3: Acoustic Forensics, Results Summary

### Community 18 - "Pillar 4/app.py"
Cohesion: 0.53
Nodes (5): analyze(), analyze_benford(), extract_leading_digits(), index(), route

### Community 19 - "Pillar 4: Financial Forensics and Statistical Anomalies"
Cohesion: 0.33
Nodes (5): 1. Mathematical & Algorithmic Foundation, 2. Engineering Challenges & Solutions, 3. Experimental Findings & Analysis, 4. Experimental Metadata Table, Pillar 4: Financial Forensics and Statistical Anomalies

### Community 20 - "Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law"
Cohesion: 0.33
Nodes (5): 4.1 Algorithms and Mathematical Framework, 4.2 Engineering Challenges and Resolutions, 4.3 Experimental Findings, 4.4 Experimental Metadata, Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law

### Community 21 - "generate_pillar5_dataset.py"
Cohesion: 0.48
Nodes (6): Path, calculate_intersection(), extract_features(), generate_dataset(), process_image_wrapper(), Extracts multi-domain physics & illumination forensic features with scale-…

### Community 22 - "Pillar 1: Vision Transformer Based Synthetic Image Detection"
Cohesion: 0.33
Nodes (5): 1. Mathematical & Algorithmic Foundation, 2. Engineering Challenges & Solutions, 3. Experimental Findings & Analysis, 4. Experimental Metadata Table, Pillar 1: Vision Transformer Based Synthetic Image Detection

### Community 23 - "Pillar 1: Vision Transformer Based Synthetic Image Detection"
Cohesion: 0.33
Nodes (5): 1. Mathematical & Algorithmic Foundation, 2. Engineering Challenges & Solutions, 3. Experimental Findings & Analysis, 4. Experimental Metadata Table, Pillar 1: Vision Transformer Based Synthetic Image Detection

### Community 24 - "Pillar 2: Biological Forensics via rPPG & Fast Fourier Transform"
Cohesion: 0.33
Nodes (5): Algorithms & Mathematics Used, Challenges & Resolutions, Experimental Findings, Metadata, Pillar 2: Biological Forensics via rPPG & Fast Fourier Transform

### Community 25 - "Pillar 2: Biological Forensics via rPPG & Fast Fourier Transform"
Cohesion: 0.33
Nodes (5): Algorithms & Mathematics Used, Challenges & Resolutions, Experimental Findings, Metadata, Pillar 2: Biological Forensics via rPPG & Fast Fourier Transform

### Community 26 - "Pillar 3: Acoustic Forensics"
Cohesion: 0.33
Nodes (5): Engineering Challenges & Solutions, Implementation & Preprocessing, Mathematical Formulation, Pillar 3: Acoustic Forensics, Results Summary

### Community 27 - "Pillar 3: Acoustic Forensics"
Cohesion: 0.33
Nodes (5): Engineering Challenges & Solutions, Implementation & Preprocessing, Mathematical Formulation, Pillar 3: Acoustic Forensics, Results Summary

### Community 28 - "Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law"
Cohesion: 0.33
Nodes (5): 4.1 Algorithms and Mathematical Framework, 4.2 Engineering Challenges and Resolutions, 4.3 Experimental Findings, 4.4 Experimental Metadata, Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law

### Community 29 - "Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law"
Cohesion: 0.33
Nodes (5): 4.1 Algorithms and Mathematical Framework, 4.2 Engineering Challenges and Resolutions, 4.3 Experimental Findings, 4.4 Experimental Metadata, Pillar 4: Financial Forensics via Optical Character Recognition and Benford's Law

### Community 31 - "pillar_5_interactive.py"
Cohesion: 0.83
Nodes (3): analyze_interactive(), calculate_intersection(), draw_line()

### Community 42 - "App.jsx"
Cohesion: 0.12
Nodes (27): App(), Navbar(), PillarBreakdown(), ProbabilityChart(), SuspiciousGallery(), SuspiciousTimeline(), VideoPlayer(), ArchitecturePage() (+19 more)

### Community 43 - "package.json"
Cohesion: 0.06
Nodes (33): dependencies, autoprefixer, axios, lucide-react, postcss, react, react-dom, tailwindcss (+25 more)

### Community 44 - "model_interface.py"
Cohesion: 0.09
Nodes (20): BaseAudioModel, BaseTemporalModel, BaseVisualModel, DefaultAudioModel, DefaultTemporalModel, HybridVisualModel, Any, Image (+12 more)

### Community 45 - "test_pipeline.py"
Cohesion: 0.14
Nodes (19): Executes the full 10-step Pillar 2 Video Authenticity & Deepfake Forensics…, run_video_pipeline(), extract_audio_track(), Any, Extracts the audio track from a video as a 16kHz mono WAV file using FFmpeg.…, Runs audio authenticity detection on extracted WAV file. Does not penalize…, run_audio_analysis(), cleanup_temporary_frames() (+11 more)

### Community 46 - "load_result"
Cohesion: 0.21
Nodes (14): delete, Deletes an analysis report and cleans up uploaded video and suspicious frames., remove_history_item(), create_result(), delete_result(), get_result_file_path(), get_status(), load_result() (+6 more)

### Community 47 - "Video Authenticity Detector (AuthentiGuard AI)"
Cohesion: 0.12
Nodes (16): 10. Limitations & Future Work, 1. Backend Setup, 1. Project Overview, 2. Features, 2. Frontend Setup, 3. Technology Stack, 4. Folder Structure, 5. Installation & Setup (+8 more)

### Community 48 - "config.py"
Cohesion: 0.15
Nodes (9): extract_and_annotate_suspicious_frames(), Any, Ranks extracted frames by composite anomaly score and saves top suspicious…, extract_sampled_frames(), Any, Extracts sampled frames from a video file into storage/frames/{video_id}/.…, Any, Extracts container and stream metadata using OpenCV and FFmpeg probe.… (+1 more)

### Community 49 - "inspect_video"
Cohesion: 0.18
Nodes (11): get, root(), post, UploadFile, Accepts video upload, validates format & size, saves to storage/uploads/, and…, upload_video(), Updates real-time status of analysis., update_status() (+3 more)

### Community 50 - "execute_video_analysis_pipeline"
Cohesion: 0.20
Nodes (9): BackgroundTasks, execute_video_analysis_pipeline(), post, Triggers the video authenticity detection pipeline for a previously uploaded…, Executes the comprehensive multi-pillar video authenticity detection pipeline…, start_analysis(), Any, Fuses multi-pillar forensic signals and computes probabilistic 3-way… (+1 more)

### Community 51 - "pillars_integrated.py"
Cohesion: 0.33
Nodes (8): analyze_pillar1_and_5_image(), analyze_pillar3_audio(), analyze_pillar4_document(), post, UploadFile, Pillars 1 & 5: Vision Transformer Neural Forensics and Shadow Physics Analysis, Pillar 3: Audio & Speech Deepfake Detection using detect.py, Pillar 4: Document, Invoice & PDF Statistical Forensics using Benford's Law

### Community 52 - "run_lip_sync_analysis"
Cohesion: 0.36
Nodes (7): compute_audio_energy_at_timestamps(), compute_mouth_motion_series(), Any, Extracts RMS speech energy from WAV file corresponding to video frame…, Cross-correlates mouth movement series with audio speech activity series.…, Computes frame-by-frame mouth region optical motion / variance., run_lip_sync_analysis()

### Community 53 - "save_uploaded_video"
Cohesion: 0.36
Nodes (7): generate_video_id(), UploadFile, Generates a clean, unique ID in the format VID_XXXXXX., Validates file extension and basic properties. Returns the lowercase file…, Saves an uploaded file to storage/uploads/ with a unique video_id. Returns…, save_uploaded_video(), validate_video_file()

### Community 54 - "visual_analyzer.py"
Cohesion: 0.32
Nodes (7): analyze_boundary_anomaly(), analyze_frequency_texture(), analyze_sensor_noise(), ndarray, Performs FFT analysis to check for high-frequency attenuation or unnatural grid…, Measures physical camera sensor noise using median filter residual. Real mobile…, Measures edge gradient discontinuity along the perimeter of the face crop which…

### Community 55 - ".detect_in_frame"
Cohesion: 0.33
Nodes (4): FaceDetector, Any, ndarray, Detects faces in an RGB frame. Returns list of detected face dictionaries…

### Community 56 - ".oxlintrc.json"
Cohesion: 0.33
Nodes (5): plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 57 - "analysis.py"
Cohesion: 0.22
Nodes (8): check_status(), get_result(), get, Returns current analysis stage, progress percentage, and status., Fetches full JSON analysis report for a completed video., Any, Evaluates temporal continuity between consecutive frames: - Luminance variance…, run_temporal_analysis()

### Community 58 - "React + Vite"
Cohesion: 0.50
Nodes (3): Expanding the Oxlint configuration, React Compiler, React + Vite

### Community 62 - "get_history"
Cohesion: 0.67
Nodes (3): get_history(), get, Returns all past video authenticity detection records from JSON files. No SQL…

## Knowledge Gaps
- **176 isolated node(s):** `$schema`, `plugins`, `react/rules-of-hooks`, `react/only-export-components`, `name` (+171 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 372 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_video_pipeline()` connect `test_pipeline.py` to `__init__.py`, `load_result`, `config.py`, `inspect_video`, `execute_video_analysis_pipeline`, `run_lip_sync_analysis`, `analysis.py`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `run_video_pipeline()` (e.g. with `extract_audio_track()` and `run_audio_analysis()`) actually correct?**
  _`run_video_pipeline()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `plugins`, `react/rules-of-hooks` to the rest of the system?**
  _176 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pillar1_pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07084785133565621 - nodes in this community are weakly interconnected._
- **Should `run_pillar5_training.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08097165991902834 - nodes in this community are weakly interconnected._
- **Should `Universal Synthetic Media Forensics Engine (USMFE)` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Should `Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._