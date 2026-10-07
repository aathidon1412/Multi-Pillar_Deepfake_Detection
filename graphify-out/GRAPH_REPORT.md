# Graph Report - Mini Project  (2026-10-07)

## Corpus Check
- 131 files · ~3,514,924 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 19 file(s) not represented in the graph (top: (none) 3, .safetensors 3, .bat 3)

## Summary
- 746 nodes · 1408 edges · 55 communities (38 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `60e1f1be`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pillar1_pipeline
- video_pipeline.py
- run_pillar5_training.py
- Universal Synthetic Media Forensics Engine (USMFE)
- backend/core/__init__.py
- pillars.py
- orchestrator.py
- generate_pillar3_audio_xai
- generate_pillar1_attention_xai
- generate_artifacts
- load_video_frames
- train_pillar2_cross_dataset.py
- classify_audio
- schemas.py
- SpatialDetector
- pillar5_engine.py
- generate_pillar4_statistical_xai
- BiLSTMFusionHead
- .oxlintrc.json
- explainer.py
- train_head.py
- React + Vite
- analyze_shadows
- generate_pillar5_shap_xai
- Path
- pillar2_hybrid.py
- classify_audio
- pillar4_engine.py
- .detect_in_frame
- core/feature_schema.py
- .analyze
- .analyze
- prepare_temp_dataset.py
- rules/graphify.md
- workflows/graphify.md
- download_dataset.py
- pillar_5_interactive.py
- .fuse
- api/__init__.py
- models/__init__.py
- api.js
- package.json
- model_interface.py
- workers/__init__.py
- storage.py

## God Nodes (most connected - your core abstractions)
1. `execute_video_analysis_pipeline()` - 25 edges
2. `create_fallback_xai_response()` - 24 edges
3. `react` - 24 edges
4. `lucide-react` - 21 edges
5. `getMediaUrl()` - 21 edges
6. `test_full_pipeline()` - 17 edges
7. `run_pillar4_inference()` - 15 edges
8. `run_visual_analysis()` - 15 edges
9. `create_xai_response()` - 14 edges
10. `run_lip_sync_analysis()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `classify_audio()` --calls--> `generate_pillar3_xai()`  [INFERRED]
  training/pillar3/detect_legacy.py → backend/core/xai/__init__.py
- `get_trained_consensus_engine()` --uses--> `ConsensusEngine`  [INFERRED]
  backend/services/classifier.py → testing/antigravity_agent/modules/consensus_engine.py
- `train_consensus_head()` --uses--> `BiLSTMFusionHead`  [INFERRED]
  training/pillar2/train_head.py → testing/antigravity_agent/modules/consensus_engine.py
- `train_and_evaluate()` --uses--> `BiLSTMFusionHead`  [INFERRED]
  training/pillar2/train_pillar2_cross_dataset.py → testing/antigravity_agent/modules/consensus_engine.py
- `extract_features_from_videos()` --uses--> `SpatialDetector`  [INFERRED]
  training/pillar2/train_head.py → testing/antigravity_agent/modules/spatial_detector.py

## Import Cycles
- None detected.

## Communities (55 total, 7 thin omitted)

### Community 0 - "train_pillar1_pipeline"
Cohesion: 0.07
Nodes (35): Compose, Dataset, GradScaler, no_grad, Optimizer, FaceDeepfakeDataset, get_pillar1_dataloaders(), make_samples_from_folders() (+27 more)

### Community 1 - "video_pipeline.py"
Cohesion: 0.06
Nodes (69): Configure native ML/OpenCV threading before heavy libraries load. Reduces…, ===============================================================================…, extract_audio_track(), Any, Extracts the audio track from a video as a 16kHz mono WAV file using FFmpeg.…, Runs audio authenticity detection on extracted WAV file. Does not penalize…, run_audio_analysis(), get_trained_consensus_engine() (+61 more)

### Community 2 - "run_pillar5_training.py"
Cohesion: 0.06
Nodes (47): BaseEstimator, StandardScaler, build_xy_matrices(), extract_efficientnet_embedding(), extract_physics_vector(), get_default_efficientnet(), device, Image (+39 more)

### Community 3 - "Universal Synthetic Media Forensics Engine (USMFE)"
Cohesion: 0.12
Nodes (16): 1. Prerequisites, 2. Python Environment Setup, 3. Frontend Setup, Audio Deepfake Detection (Pillar 3):, 🛠️ CLI Standalone Tools, ⚡ False Negative Mitigation & Deterministic Override, Multi-Pillar Deepfake & Synthetic Media Detection Framework, Multi-Pillar Test Suite Evaluation: (+8 more)

### Community 4 - "backend/core/__init__.py"
Cohesion: 0.15
Nodes (27): ===============================================================================…, ===============================================================================…, create_fallback_xai_response(), create_xai_evidence(), create_xai_response(), generate_consensus_xai(), generate_pillar1_xai(), generate_pillar2_xai() (+19 more)

### Community 5 - "pillars.py"
Cohesion: 0.16
Nodes (28): analyze_pillar1_and_5_image(), analyze_pillar3_audio(), analyze_pillar4_document(), analyze_universal_media(), get_p1_model(), get_p5_model(), post, UploadFile (+20 more)

### Community 6 - "orchestrator.py"
Cohesion: 0.16
Nodes (15): OrchestratorAgent, Core Orchestrator Agent (pipeline_manager.py) Coordinates end-to-end ingestion,…, Main pipeline coordinator for the Antigravity Deepfake Detection system., evaluate_dataset(), Any, Batch Evaluation & Benchmarking Script for Antigravity Agent. Evaluates videos…, AudioDetector, Module 3: Audio & Track Integrity Detector Inspects container metadata, audio… (+7 more)

### Community 7 - "generate_pillar3_audio_xai"
Cohesion: 0.16
Nodes (16): compute_integrated_gradients_audio(), extract_important_audio_segments(), generate_pillar3_audio_xai(), generate_saliency_spectrogram_plot(), Any, device, Module, ndarray (+8 more)

### Community 8 - "generate_pillar1_attention_xai"
Cohesion: 0.14
Nodes (13): compute_ela_image(), ===============================================================================…, Computes JPEG Error Level Analysis (ELA) at specified quality (default 90).…, compute_attention_rollout(), generate_pillar1_attention_xai(), identify_highest_attribution_region(), Any, Image (+5 more)

### Community 9 - "generate_artifacts"
Cohesion: 0.22
Nodes (12): calculate_benford_metrics(), extract_digits_from_csv(), extract_leading_digits(), extract_text_from_image(), generate_artifacts(), generate_decision(), Calculates observed, theoretical frequencies, MAE, and Chi-Square stats., Determines the verdict based on classification thresholds, dynamically scaled… (+4 more)

### Community 10 - "load_video_frames"
Cohesion: 0.18
Nodes (10): Any, Execute full forensic analysis pipeline on input video. Args: video_path: Path…, extract_audio_stream(), load_video_frames(), Any, ndarray, Tensor, Video loader utility for antigravity_agent. Extracts equidistant frames,… (+2 more)

### Community 11 - "train_pillar2_cross_dataset.py"
Cohesion: 0.25
Nodes (10): download_dataset(), extract_features(), process_group(), main(), Tensor, Pillar 2 Cross-Dataset Deepfake Fine-Tuning & Evaluation Pipeline…, Trains the BiLSTMFusionHead with train/val split and prints classification…, Downloads or verifies video dataset from Hugging Face: - 'sdfvd': Hemgg/SDFVD-… (+2 more)

### Community 12 - "classify_audio"
Cohesion: 0.27
Nodes (8): analyze_audio_composition(), classify_audio(), get_model(), ===============================================================================…, Classifies an audio file as REAL vs FAKE., Uses librosa to compute acoustic metrics and detect whether the input audio…, main(), ===============================================================================…

### Community 13 - "schemas.py"
Cohesion: 0.33
Nodes (9): AnalysisReport, ClassificationResult, ClassificationScores, ConsensusResult, FileMetadata, backend/models/schemas.py ========================= Pydantic schemas for API…, StatusResponse, VideoDetails (+1 more)

### Community 14 - "SpatialDetector"
Cohesion: 0.22
Nodes (6): Any, Tensor, Module 1: Spatial Forensics Detector Analyzes static, high-frequency frame…, Extracts spatial embeddings and frame suspicion scores. Uses pre-trained ViT…, Analyze N sampled frames. Args: frames_tensor: Tensor of shape [N, 3, 224, 224]…, SpatialDetector

### Community 15 - "pillar5_engine.py"
Cohesion: 0.22
Nodes (8): calculate_intersection(), load_pillar5_deep_backbone(), ===============================================================================…, Loads feature extractor backbone for Pillar 5 Deep Fusion (cached in memory)., Computes Cartesian intersection point between two line segments., generate_pillar5_xai(), ndarray, Delegates to TreeSHAP explainability engine in pillar5_xai.

### Community 16 - "generate_pillar4_statistical_xai"
Cohesion: 0.33
Nodes (8): compute_benford_comparison_table(), generate_benford_chart_plot(), generate_pillar4_statistical_xai(), Any, ===============================================================================…, Renders a publication-grade dark-themed comparison plot for Benford…, Generates explainable statistical evidence for Pillar 4 Benford's Law OCR…, Computes per-digit comparisons between observed and expected Benford…

### Community 17 - "BiLSTMFusionHead"
Cohesion: 0.25
Nodes (5): BiLSTMFusionHead, Tensor, Module 4: Consensus Aggregator & Confidence Calibrator Combines spatial…, Lightweight Bi-LSTM sequence fusion head for temporal-spatial artifact…, Args: spatial_emb: [Batch, SequenceLen=16, 768] motion_feats: [Batch, 2] (e.g.…

### Community 18 - ".oxlintrc.json"
Cohesion: 0.33
Nodes (5): plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 19 - "explainer.py"
Cohesion: 0.25
Nodes (6): Any, Module 5: Explainability & Verdict Reporting Synthesizes natural language…, Builds the itemized report and renders the timeline chart., generate_anomaly_timeline(), Visualization utilities for antigravity_agent. Generates timeline anomaly…, Generate dual-axis timeline graph showing spatial artifact score and motion…

### Community 20 - "train_head.py"
Cohesion: 0.25
Nodes (7): Module 2: Temporal Inconsistency Analyzer Detects motion anomalies, generative…, extract_features_from_videos(), process_paths(), Tensor, Train and calibrate the lightweight Bi-LSTM / MLP consensus fusion head.…, Pass real and fake videos through frozen Module 1 (ViT) and Module 2 (Optical…, train_consensus_head()

### Community 21 - "React + Vite"
Cohesion: 0.50
Nodes (3): Expanding the Oxlint configuration, React Compiler, React + Vite

### Community 22 - "analyze_shadows"
Cohesion: 0.33
Nodes (7): process_all_test_images(), analyze_shadows(), calculate_intersection(), line_point_distance(), Calculate the intersection of two lines defined by two points each., Distance from a point to a line segment extended infinitely., Executes Pillar 5 of the Universal Synthetic Media Forensics Engine (USMFE).…

### Community 23 - "generate_pillar5_shap_xai"
Cohesion: 0.32
Nodes (7): generate_pillar5_shap_xai(), generate_shap_waterfall_plot(), Any, ndarray, ===============================================================================…, Renders a publication-grade dark-themed SHAP Waterfall plot. Features pushing…, Computes TreeSHAP feature attributions for Pillar 5 Physical Geometry &…

### Community 24 - "Path"
Cohesion: 0.39
Nodes (6): Path, calculate_intersection(), extract_features(), generate_dataset(), process_image_wrapper(), Extracts multi-domain physics & illumination forensic features with scale-…

### Community 25 - "pillar2_hybrid.py"
Cohesion: 0.48
Nodes (6): analyze_video(), butterworth_bandpass(), compute_rppg_snr(), get_fake_score(), load_detector(), pillar2_hybrid.py - USMFE Pillar 2 Hybrid Deepfake Detector…

### Community 26 - "classify_audio"
Cohesion: 0.43
Nodes (6): analyze_audio_composition(), classify_audio(), get_model(), main(), Classifies an audio file as REAL vs FAKE. Parameters: audio_path: path to audio…, Uses librosa to compute acoustic metrics and detect whether the input audio…

### Community 27 - "pillar4_engine.py"
Cohesion: 0.33
Nodes (5): analyze_benford_law(), extract_digits_from_text(), ===============================================================================…, Extracts first significant digits (1-9) from numerical text strings., Performs statistical Benford's Law Chi-Square and MAE distribution analysis.

### Community 28 - ".detect_in_frame"
Cohesion: 0.33
Nodes (4): FaceDetector, Any, ndarray, Detects faces in an RGB frame. Returns list of detected face dictionaries…

### Community 29 - "core/feature_schema.py"
Cohesion: 0.40
Nodes (4): ndarray, ===============================================================================…, Converts a feature dictionary into a 24-dimensional numpy vector guaranteed to…, validate_feature_dict()

### Community 30 - ".analyze"
Cohesion: 0.50
Nodes (3): Any, ndarray, Args: audio_data: 1D numpy array of audio samples (or None if video has no…

### Community 31 - ".analyze"
Cohesion: 0.50
Nodes (3): Any, Tensor, Analyze motion consistency across sequential frames. Args: frames_tensor:…

### Community 36 - "download_dataset.py"
Cohesion: 0.50
Nodes (3): download_dataset(), Automated Large-Scale Dataset Downloader for Antigravity Agent. Downloads up to…, Download video files from repo. max_per_class: number of videos per class (e.g.…

### Community 37 - "pillar_5_interactive.py"
Cohesion: 0.83
Nodes (3): analyze_interactive(), calculate_intersection(), draw_line()

### Community 42 - "api.js"
Cohesion: 0.08
Nodes (40): App(), Navbar(), Pillar1XaiExplanation(), Pillar2XaiTemporalExplanation(), Pillar3XaiAudioExplanation(), Pillar4XaiDocumentExplanation(), Pillar5XaiPhysicsExplanation(), SuspiciousGallery() (+32 more)

### Community 43 - "package.json"
Cohesion: 0.06
Nodes (33): dependencies, autoprefixer, axios, lucide-react, postcss, react, react-dom, tailwindcss (+25 more)

### Community 44 - "model_interface.py"
Cohesion: 0.07
Nodes (31): log_compute_backend(), get, root(), describe_compute_device(), device_summary(), get_hf_pipeline_device(), get_torch_device(), device (+23 more)

### Community 46 - "storage.py"
Cohesion: 0.07
Nodes (48): check_status(), get_result(), get, post, Run heavy ML pipeline in a separate Python process so a native crash cannot…, Triggers the video authenticity detection pipeline for a previously uploaded…, Returns current analysis stage, progress percentage, and status., Fetches full JSON analysis report for a completed video. (+40 more)

## Knowledge Gaps
- **53 isolated node(s):** `$schema`, `plugins`, `react/rules-of-hooks`, `react/only-export-components`, `name` (+48 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 338 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_trained_consensus_engine()` connect `video_pipeline.py` to `Path`, `orchestrator.py`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `ConsensusEngine` connect `orchestrator.py` to `.fuse`, `video_pipeline.py`, `BiLSTMFusionHead`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **What connects `$schema`, `plugins`, `react/rules-of-hooks` to the rest of the system?**
  _53 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pillar1_pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07084785133565621 - nodes in this community are weakly interconnected._
- **Should `video_pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.060424469413233456 - nodes in this community are weakly interconnected._
- **Should `run_pillar5_training.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05656565656565657 - nodes in this community are weakly interconnected._
- **Should `Universal Synthetic Media Forensics Engine (USMFE)` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._