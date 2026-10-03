# Graph Report - Mini Project  (2026-10-03)

## Corpus Check
- 150 files · ~5,377,336 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 19 file(s) not represented in the graph (top: (none) 5, .bat 4, .pth 2)

## Summary
- 930 nodes · 1451 edges · 75 communities (59 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 44 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4e788806`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pillar1_pipeline
- run_pillar5_training.py
- 1.2 Critical User Journeys
- Universal Synthetic Media Forensics Engine (USMFE)
- Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics
- classify_audio
- Module: Pillar 3 — Acoustic & Voice Deepfake Forensics
- 5. Visual Proofs & Explainability (XAI)
- 2. Iteration-by-Iteration Breakdown
- 2. Iteration-by-Iteration Breakdown
- analyze_shadows
- generate_artifacts
- app_streamlit.py
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
- api.js
- package.json
- model_interface.py
- analysis.py
- json_storage.py
- Video Authenticity Detector (AuthentiGuard AI)
- run_video_pipeline
- inspect_video
- execute_video_analysis_pipeline
- pillars_integrated.py
- run_lip_sync_analysis
- create_fallback_xai_response
- run_visual_analysis
- .detect_in_frame
- .oxlintrc.json
- ui_components.py
- React + Vite
- Forensic Thresholds & Decision Boundaries Reference Guide
- chat1.md
- core/__init__.py
- generate_pillar3_audio_xai
- DESIGN.md
- generate_pillar1_attention_xai
- generate_pillar4_statistical_xai
- test_full_pipeline
- chat2.md
- generate_pillar5_shap_xai
- pillar5_engine.py

## God Nodes (most connected - your core abstractions)
1. `react` - 24 edges
2. `create_fallback_xai_response()` - 24 edges
3. `lucide-react` - 21 edges
4. `getMediaUrl()` - 21 edges
5. `execute_video_analysis_pipeline()` - 20 edges
6. `run_video_pipeline()` - 19 edges
7. `test_full_pipeline()` - 17 edges
8. `create_xai_response()` - 15 edges
9. `run_pillar4_inference()` - 14 edges
10. `analyze_universal_media()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `run_video_pipeline()` --calls--> `extract_audio_track()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/audio_analyzer.py
- `evaluate_pillar2_video_suite()` --calls--> `extract_audio_track()`  [INFERRED]
  test_eval.py → Pillar 2/video-authenticity-detector/backend/services/audio_analyzer.py
- `run_video_pipeline()` --calls--> `run_audio_analysis()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/audio_analyzer.py
- `evaluate_pillar2_video_suite()` --calls--> `run_audio_analysis()`  [INFERRED]
  test_eval.py → Pillar 2/video-authenticity-detector/backend/services/audio_analyzer.py
- `run_video_pipeline()` --calls--> `cleanup_temporary_frames()`  [INFERRED]
  app_streamlit.py → Pillar 2/video-authenticity-detector/backend/services/cleanup.py

## Import Cycles
- 3-file cycle: `core/__init__.py -> core/pillar3_engine.py -> detect.py -> core/__init__.py`

## Communities (75 total, 4 thin omitted)

### Community 0 - "train_pillar1_pipeline"
Cohesion: 0.07
Nodes (35): Compose, Dataset, GradScaler, no_grad, Optimizer, FaceDeepfakeDataset, get_pillar1_dataloaders(), make_samples_from_folders() (+27 more)

### Community 1 - "run_pillar5_training.py"
Cohesion: 0.06
Nodes (47): BaseEstimator, build_xy_matrices(), extract_efficientnet_embedding(), extract_physics_vector(), get_default_efficientnet(), device, Image, Module (+39 more)

### Community 2 - "1.2 Critical User Journeys"
Cohesion: 0.06
Nodes (30): 1.1 Core Purpose & Target Persona, 1.2 Critical User Journeys, 1.3 Domain Entities & Data Models, 1. Forensic Report (`FinalReport` JSON Model), 1. System Core & Primary Workflows, 2.1 Current Tech Stack, 2.2 Route & View Inventory, 2.3 Component Breakdown (+22 more)

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

### Community 12 - "app_streamlit.py"
Cohesion: 0.12
Nodes (21): detect_modality(), get_cached_pillar1(), get_cached_pillar5(), cache_resource, ===============================================================================…, Detects media modality from file extension., Renders a clean inline badge showing detected file modality., render_modality_badge() (+13 more)

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

### Community 42 - "api.js"
Cohesion: 0.09
Nodes (37): App(), Navbar(), Pillar1XaiExplanation(), Pillar2XaiTemporalExplanation(), Pillar3XaiAudioExplanation(), Pillar4XaiDocumentExplanation(), Pillar5XaiPhysicsExplanation(), SuspiciousGallery() (+29 more)

### Community 43 - "package.json"
Cohesion: 0.06
Nodes (33): dependencies, autoprefixer, axios, lucide-react, postcss, react, react-dom, tailwindcss (+25 more)

### Community 44 - "model_interface.py"
Cohesion: 0.09
Nodes (20): BaseAudioModel, BaseTemporalModel, BaseVisualModel, DefaultAudioModel, DefaultTemporalModel, HybridVisualModel, Any, Image (+12 more)

### Community 45 - "analysis.py"
Cohesion: 0.25
Nodes (7): ===============================================================================…, extract_audio_track(), Extracts the audio track from a video as a 16kHz mono WAV file using FFmpeg.…, cleanup_temporary_frames(), Safely removes temporary raw frames extracted in storage/frames/{video_id}/…, detect_faces_in_frames(), Runs face detection over all extracted frames. Attaches detected faces to each…

### Community 46 - "json_storage.py"
Cohesion: 0.12
Nodes (24): delete, check_status(), get_result(), get, Returns current analysis stage, progress percentage, and status., Fetches full JSON analysis report for a completed video., get_history(), get (+16 more)

### Community 47 - "Video Authenticity Detector (AuthentiGuard AI)"
Cohesion: 0.12
Nodes (16): 10. Limitations & Future Work, 1. Backend Setup, 1. Project Overview, 2. Features, 2. Frontend Setup, 3. Technology Stack, 4. Folder Structure, 5. Installation & Setup (+8 more)

### Community 48 - "run_video_pipeline"
Cohesion: 0.11
Nodes (22): Executes the full 10-step Pillar 2 Video Authenticity & Deepfake Forensics…, run_video_pipeline(), Any, Fuses multi-pillar forensic signals and computes probabilistic 3-way…, run_feature_fusion_and_classification(), extract_sampled_frames(), Any, Extracts sampled frames from a video file into storage/frames/{video_id}/.… (+14 more)

### Community 49 - "inspect_video"
Cohesion: 0.15
Nodes (16): get, root(), post, UploadFile, Accepts video upload, validates format & size, saves to storage/uploads/, and…, upload_video(), generate_video_id(), inspect_video() (+8 more)

### Community 50 - "execute_video_analysis_pipeline"
Cohesion: 0.29
Nodes (8): BackgroundTasks, execute_video_analysis_pipeline(), post, Triggers the video authenticity detection pipeline for a previously uploaded…, Executes the comprehensive multi-pillar video authenticity detection pipeline…, start_analysis(), Updates real-time status of analysis., update_status()

### Community 51 - "pillars_integrated.py"
Cohesion: 0.21
Nodes (18): Executes Pillar 4: Semantic Document & Benford's Law OCR Forensics., run_pillar4_inference(), load_pillar5_ml_bundle(), Loads the dual-model ensemble bundle (pillar5_ml_model.pkl +…, Synthesizes a unified, standardized XAI dossier dictionary across all 5…, synthesize_multi_pillar_xai(), analyze_pillar1_and_5_image(), analyze_pillar3_audio() (+10 more)

### Community 52 - "run_lip_sync_analysis"
Cohesion: 0.36
Nodes (7): compute_audio_energy_at_timestamps(), compute_mouth_motion_series(), Any, Extracts RMS speech energy from WAV file corresponding to video frame…, Cross-correlates mouth movement series with audio speech activity series.…, Computes frame-by-frame mouth region optical motion / variance., run_lip_sync_analysis()

### Community 53 - "create_fallback_xai_response"
Cohesion: 0.14
Nodes (28): ===============================================================================…, calculate_segment_signal_contributions(), generate_pillar2_temporal_xai(), Any, ===============================================================================…, Computes normalized signal contributions for an existing flagged video…, Executes full Temporal Evidence Attribution for Pillar 2 Video Forensics., create_fallback_xai_response() (+20 more)

### Community 54 - "run_visual_analysis"
Cohesion: 0.24
Nodes (10): analyze_boundary_anomaly(), analyze_frequency_texture(), analyze_sensor_noise(), Any, ndarray, Performs FFT analysis to check for high-frequency attenuation or unnatural grid…, Measures physical camera sensor noise using median filter residual. Real mobile…, Measures edge gradient discontinuity along the perimeter of the face crop which… (+2 more)

### Community 55 - ".detect_in_frame"
Cohesion: 0.33
Nodes (4): FaceDetector, Any, ndarray, Detects faces in an RGB frame. Returns list of detected face dictionaries…

### Community 56 - ".oxlintrc.json"
Cohesion: 0.33
Nodes (5): plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 57 - "ui_components.py"
Cohesion: 0.10
Nodes (25): apply_custom_theme(), plot_benford_distribution(), ===============================================================================…, Injects custom cyber-forensics dark CSS theme., Renders the top title banner., Renders a responsive, glowing master verdict card., Generates a dark-themed Benford's Law comparison chart., Renders a standardized XAI evidence panel in Streamlit. (+17 more)

### Community 58 - "React + Vite"
Cohesion: 0.50
Nodes (3): Expanding the Oxlint configuration, React Compiler, React + Vite

### Community 62 - "Forensic Thresholds & Decision Boundaries Reference Guide"
Cohesion: 0.09
Nodes (22): 1. Pillar 1: Vision Transformer (ViT) & Spectral Forensics, 2. Pillar 2: Temporal & Multi-Modal Video Forensics, 3. Pillar 3: Acoustic & Voice Synthetic Speech Forensics, 4. Pillar 4: Statistical OCR & Benford’s Law (Documents), 5. Pillar 5: Shadow Physics, Vanishing Points & PRNU Steganalysis, 6. The Master Consensus Engine: Final Fusion, Decision Boundaries, Decision Thresholds (+14 more)

### Community 63 - "chat1.md"
Cohesion: 0.09
Nodes (21): 1. PILLAR 1: Vision Transformer (ViT) & Spectral Forensics, 2. PILLAR 3: Acoustic & Voice Synthetic Speech Forensics, 3. PILLAR 4: Document, Invoice & Benford's Law Statistical OCR, 4. PILLAR 5: Physical Geometry & Shadow Physics Forensics, 5. THE MASTER CONSENSUS ENGINE: Final Unified Verdict, Applicability Gate, Audio Composition & Domain Thresholds, Classification Thresholds (+13 more)

### Community 64 - "core/__init__.py"
Cohesion: 0.15
Nodes (13): predict_audio(), Takes an uploaded or recorded audio file path and mode ("spoken" or "music"): -…, ===============================================================================…, ===============================================================================…, classify_audio(), ===============================================================================…, analyze_benford_law(), extract_digits_from_text() (+5 more)

### Community 65 - "generate_pillar3_audio_xai"
Cohesion: 0.16
Nodes (16): compute_integrated_gradients_audio(), extract_important_audio_segments(), generate_pillar3_audio_xai(), generate_saliency_spectrogram_plot(), Any, device, Module, ndarray (+8 more)

### Community 66 - "DESIGN.md"
Cohesion: 0.14
Nodes (13): Brand & Style, Buttons, Colors, Components, Data Tables & Evidence Lists, Elevation & Depth, Form Fields & Scrub Filters, Layout & Spacing (+5 more)

### Community 67 - "generate_pillar1_attention_xai"
Cohesion: 0.20
Nodes (10): compute_attention_rollout(), generate_pillar1_attention_xai(), identify_highest_attribution_region(), Any, Image, ndarray, Tensor, Executes full Attention Rollout XAI on Vision Transformer inference. Returns:… (+2 more)

### Community 68 - "generate_pillar4_statistical_xai"
Cohesion: 0.33
Nodes (8): compute_benford_comparison_table(), generate_benford_chart_plot(), generate_pillar4_statistical_xai(), Any, ===============================================================================…, Renders a publication-grade dark-themed comparison plot for Benford…, Generates explainable statistical evidence for Pillar 4 Benford's Law OCR…, Computes per-digit comparisons between observed and expected Benford…

### Community 69 - "test_full_pipeline"
Cohesion: 0.22
Nodes (9): Any, Runs audio authenticity detection on extracted WAV file. Does not penalize…, run_audio_analysis(), extract_and_annotate_suspicious_frames(), Any, Ranks extracted frames by composite anomaly score and saves top suspicious…, create_synthetic_test_video(), Creates a valid synthetic MP4 video with a simulated face-like oval and audio… (+1 more)

### Community 70 - "chat2.md"
Cohesion: 0.25
Nodes (7): 1. What is Canny Edge Detection?, 2. What is it Doing in Deepfake Detection? (The Core Physics Problem), 3. Why Canny Edge is Used in Your Pipeline, 4. Why it is Valid and Why You Should Keep It, 5. What Happens if You Remove It?, Conclusion, Executive Summary

### Community 71 - "generate_pillar5_shap_xai"
Cohesion: 0.32
Nodes (7): generate_pillar5_shap_xai(), generate_shap_waterfall_plot(), Any, ndarray, ===============================================================================…, Renders a publication-grade dark-themed SHAP Waterfall plot. Features pushing…, Computes TreeSHAP feature attributions for Pillar 5 Physical Geometry &…

### Community 72 - "pillar5_engine.py"
Cohesion: 0.33
Nodes (5): calculate_intersection(), load_pillar5_deep_backbone(), ===============================================================================…, Loads feature extractor backbone for Pillar 5 Deep Fusion (cached in memory)., Computes Cartesian intersection point between two line segments.

## Knowledge Gaps
- **252 isolated node(s):** `$schema`, `plugins`, `react/rules-of-hooks`, `react/only-export-components`, `name` (+247 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 510 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `create_fallback_xai_response()` connect `create_fallback_xai_response` to `core/__init__.py`, `generate_pillar3_audio_xai`, `generate_pillar1_attention_xai`, `app_streamlit.py`, `analysis.py`, `execute_video_analysis_pipeline`, `pillars_integrated.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `run_video_pipeline()` connect `run_video_pipeline` to `test_full_pipeline`, `app_streamlit.py`, `analysis.py`, `json_storage.py`, `inspect_video`, `run_lip_sync_analysis`, `run_visual_analysis`, `ui_components.py`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Why does `execute_video_analysis_pipeline()` connect `execute_video_analysis_pipeline` to `test_full_pipeline`, `analysis.py`, `json_storage.py`, `run_video_pipeline`, `inspect_video`, `pillars_integrated.py`, `run_lip_sync_analysis`, `create_fallback_xai_response`, `run_visual_analysis`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **What connects `$schema`, `plugins`, `react/rules-of-hooks` to the rest of the system?**
  _252 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pillar1_pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07084785133565621 - nodes in this community are weakly interconnected._
- **Should `run_pillar5_training.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05656565656565657 - nodes in this community are weakly interconnected._
- **Should `1.2 Critical User Journeys` be split into smaller, more focused modules?**
  _Cohesion score 0.06451612903225806 - nodes in this community are weakly interconnected._