# Graph Report - Mini Project  (2026-10-07)

## Corpus Check
- 128 files · ~3,514,086 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 20 file(s) not represented in the graph (top: (none) 4, .safetensors 3, .bat 3)

## Summary
- 736 nodes · 1322 edges · 63 communities (45 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7244ab4a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pillar1_pipeline
- extract_efficientnet_embedding
- train_and_evaluate_pillar5
- Universal Synthetic Media Forensics Engine (USMFE)
- orchestrator.py
- classify_audio
- load_video_frames
- execute_video_analysis_pipeline
- train_head.py
- train_pillar2_cross_dataset.py
- analyze_shadows
- generate_artifacts
- SpatialDetector
- TemporalAnalyzer
- .generate_report
- pillar2_hybrid.py
- generate_pillar2_temporal_xai
- pillar4_engine.py
- .oxlintrc.json
- run_feature_fusion_and_classification
- .analyze
- React + Vite
- prepare_temp_dataset.py
- download_dataset.py
- .fuse
- inspect_video
- pillar_5_interactive.py
- rules/graphify.md
- workflows/graphify.md
- api.js
- package.json
- main.py
- video_analysis_pipeline.py
- json_storage.py
- Video Authenticity Detector (AuthentiGuard AI)
- evaluate_pillar2_video_suite
- video_processor.py
- pillars_integrated.py
- run_lip_sync_analysis
- core/__init__.py
- run_visual_analysis
- detect_faces_in_frames
- classify_audio
- generate_pillar3_audio_xai
- generate_pillar1_attention_xai
- generate_pillar4_statistical_xai
- test_full_pipeline
- generate_pillar5_shap_xai
- pillar5_engine.py

## God Nodes (most connected - your core abstractions)
1. `execute_video_analysis_pipeline()` - 25 edges
2. `create_fallback_xai_response()` - 24 edges
3. `react` - 24 edges
4. `lucide-react` - 21 edges
5. `getMediaUrl()` - 21 edges
6. `test_full_pipeline()` - 17 edges
7. `create_xai_response()` - 15 edges
8. `analyze_universal_media()` - 14 edges
9. `generate_pillar1_attention_xai()` - 13 edges
10. `run_pillar4_inference()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `evaluate_pillar2_video_suite()` --calls--> `extract_audio_track()`  [INFERRED]
  test_eval.py → Pillar 2/video-authenticity-detector/backend/services/audio_analyzer.py
- `evaluate_pillar2_video_suite()` --calls--> `run_audio_analysis()`  [INFERRED]
  test_eval.py → Pillar 2/video-authenticity-detector/backend/services/audio_analyzer.py
- `evaluate_pillar2_video_suite()` --calls--> `run_feature_fusion_and_classification()`  [INFERRED]
  test_eval.py → Pillar 2/video-authenticity-detector/backend/services/classifier.py
- `evaluate_pillar2_video_suite()` --calls--> `cleanup_temporary_frames()`  [INFERRED]
  test_eval.py → Pillar 2/video-authenticity-detector/backend/services/cleanup.py
- `evaluate_pillar2_video_suite()` --calls--> `detect_faces_in_frames()`  [INFERRED]
  test_eval.py → Pillar 2/video-authenticity-detector/backend/services/face_detector.py

## Import Cycles
- 3-file cycle: `core/__init__.py -> core/pillar3_engine.py -> detect.py -> core/__init__.py`

## Communities (63 total, 4 thin omitted)

### Community 0 - "train_pillar1_pipeline"
Cohesion: 0.07
Nodes (35): Compose, Dataset, GradScaler, FaceDeepfakeDataset, get_pillar1_dataloaders(), make_samples_from_folders(), DataLoader, Tensor (+27 more)

### Community 1 - "extract_efficientnet_embedding"
Cohesion: 0.11
Nodes (25): collect_balanced_dataset(), extract_worker(), main(), ndarray, ===============================================================================…, Worker function for parallel tabular feature extraction., Collects balanced paths: 1,000 authentic and 1,000 fake across available…, build_xy_matrices() (+17 more)

### Community 2 - "train_and_evaluate_pillar5"
Cohesion: 0.11
Nodes (22): BaseEstimator, extract_efficientnet_embeddings(), main(), extract_single_image_features(), main(), Worker function for parallel tabular feature extraction. Takes a tuple…, calculate_intersection(), extract_deep_embeddings() (+14 more)

### Community 3 - "Universal Synthetic Media Forensics Engine (USMFE)"
Cohesion: 0.12
Nodes (16): 1. Clone the Repository, 2. Create and Activate Virtual Environment, 3. Install Dependencies, 4. Install Tesseract OCR (Required for Pillar 4 Document Mode), Audio Deepfake Detection (Pillar 3):, Available Analysis Modes:, 🛠️ CLI Standalone Tools, 📊 Evaluation & Verification (+8 more)

### Community 4 - "orchestrator.py"
Cohesion: 0.15
Nodes (14): OrchestratorAgent, Core Orchestrator Agent (pipeline_manager.py) Coordinates end-to-end ingestion,…, Main pipeline coordinator for the Antigravity Deepfake Detection system., evaluate_dataset(), Any, Batch Evaluation & Benchmarking Script for Antigravity Agent. Evaluates videos…, AudioDetector, Module 3: Audio & Track Integrity Detector Inspects container metadata, audio… (+6 more)

### Community 5 - "classify_audio"
Cohesion: 0.43
Nodes (6): analyze_audio_composition(), classify_audio(), get_model(), main(), Classifies an audio file as REAL vs FAKE. Parameters: audio_path: path to audio…, Uses librosa to compute acoustic metrics and detect whether the input audio…

### Community 6 - "load_video_frames"
Cohesion: 0.14
Nodes (14): Any, Execute full forensic analysis pipeline on input video. Args: video_path: Path…, extract_audio_stream(), load_video_frames(), Any, ndarray, Tensor, Video loader utility for antigravity_agent. Extracts equidistant frames,… (+6 more)

### Community 7 - "execute_video_analysis_pipeline"
Cohesion: 0.19
Nodes (10): Configure native ML/OpenCV threading before heavy libraries load. Reduces…, execute_video_analysis_pipeline(), Any, Drop large in-memory pixel buffers after disk artifacts are written., Executes the comprehensive multi-pillar video authenticity detection pipeline.…, _release_extracted_frame_memory(), _release_ml_runtime_memory(), main() (+2 more)

### Community 8 - "train_head.py"
Cohesion: 0.21
Nodes (7): BiLSTMFusionHead, Tensor, Module 4: Consensus Aggregator & Confidence Calibrator Combines spatial…, Lightweight Bi-LSTM sequence fusion head for temporal-spatial artifact…, Args: spatial_emb: [Batch, SequenceLen=16, 768] motion_feats: [Batch, 2] (e.g.…, Train and calibrate the lightweight Bi-LSTM / MLP consensus fusion head.…, train_consensus_head()

### Community 9 - "train_pillar2_cross_dataset.py"
Cohesion: 0.25
Nodes (10): download_dataset(), extract_features(), process_group(), main(), Tensor, Pillar 2 Cross-Dataset Deepfake Fine-Tuning & Evaluation Pipeline…, Trains the BiLSTMFusionHead with train/val split and prints classification…, Downloads or verifies video dataset from Hugging Face: - 'sdfvd': Hemgg/SDFVD-… (+2 more)

### Community 10 - "analyze_shadows"
Cohesion: 0.33
Nodes (7): process_all_test_images(), analyze_shadows(), calculate_intersection(), line_point_distance(), Calculate the intersection of two lines defined by two points each., Distance from a point to a line segment extended infinitely., Executes Pillar 5 of the Universal Synthetic Media Forensics Engine (USMFE).…

### Community 11 - "generate_artifacts"
Cohesion: 0.22
Nodes (12): calculate_benford_metrics(), extract_digits_from_csv(), extract_leading_digits(), extract_text_from_image(), generate_artifacts(), generate_decision(), Calculates observed, theoretical frequencies, MAE, and Chi-Square stats., Determines the verdict based on classification thresholds, dynamically scaled… (+4 more)

### Community 12 - "SpatialDetector"
Cohesion: 0.22
Nodes (6): Any, Tensor, Module 1: Spatial Forensics Detector Analyzes static, high-frequency frame…, Extracts spatial embeddings and frame suspicion scores. Uses pre-trained ViT…, Analyze N sampled frames. Args: frames_tensor: Tensor of shape [N, 3, 224, 224]…, SpatialDetector

### Community 13 - "TemporalAnalyzer"
Cohesion: 0.25
Nodes (6): Any, Tensor, Module 2: Temporal Inconsistency Analyzer Detects motion anomalies, generative…, Computes dense optical flow and pixel difference across sequential frames., Analyze motion consistency across sequential frames. Args: frames_tensor:…, TemporalAnalyzer

### Community 14 - ".generate_report"
Cohesion: 0.29
Nodes (5): Any, Builds the itemized report and renders the timeline chart., generate_anomaly_timeline(), Visualization utilities for antigravity_agent. Generates timeline anomaly…, Generate dual-axis timeline graph showing spatial artifact score and motion…

### Community 15 - "pillar2_hybrid.py"
Cohesion: 0.48
Nodes (6): analyze_video(), butterworth_bandpass(), compute_rppg_snr(), get_fake_score(), load_detector(), pillar2_hybrid.py - USMFE Pillar 2 Hybrid Deepfake Detector…

### Community 16 - "generate_pillar2_temporal_xai"
Cohesion: 0.38
Nodes (6): calculate_segment_signal_contributions(), generate_pillar2_temporal_xai(), Any, ===============================================================================…, Computes normalized signal contributions for an existing flagged video…, Executes full Temporal Evidence Attribution for Pillar 2 Video Forensics.

### Community 17 - "pillar4_engine.py"
Cohesion: 0.33
Nodes (5): analyze_benford_law(), extract_digits_from_text(), ===============================================================================…, Extracts first significant digits (1-9) from numerical text strings., Performs statistical Benford's Law Chi-Square and MAE distribution analysis.

### Community 18 - ".oxlintrc.json"
Cohesion: 0.33
Nodes (5): plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 19 - "run_feature_fusion_and_classification"
Cohesion: 0.50
Nodes (4): get_trained_consensus_engine(), Any, Fuses multi-pillar forensic signals and computes probabilistic 3-way…, run_feature_fusion_and_classification()

### Community 20 - ".analyze"
Cohesion: 0.50
Nodes (3): Any, ndarray, Args: audio_data: 1D numpy array of audio samples (or None if video has no…

### Community 21 - "React + Vite"
Cohesion: 0.50
Nodes (3): Expanding the Oxlint configuration, React Compiler, React + Vite

### Community 23 - "download_dataset.py"
Cohesion: 0.50
Nodes (3): download_dataset(), Automated Large-Scale Dataset Downloader for Antigravity Agent. Downloads up to…, Download video files from repo. max_per_class: number of videos per class (e.g.…

### Community 26 - "inspect_video"
Cohesion: 0.67
Nodes (3): inspect_video(), Any, Opens video with OpenCV and extracts fundamental video parameters: fps, frame…

### Community 31 - "pillar_5_interactive.py"
Cohesion: 0.83
Nodes (3): analyze_interactive(), calculate_intersection(), draw_line()

### Community 42 - "api.js"
Cohesion: 0.08
Nodes (40): App(), Navbar(), Pillar1XaiExplanation(), Pillar2XaiTemporalExplanation(), Pillar3XaiAudioExplanation(), Pillar4XaiDocumentExplanation(), Pillar5XaiPhysicsExplanation(), SuspiciousGallery() (+32 more)

### Community 43 - "package.json"
Cohesion: 0.06
Nodes (33): dependencies, autoprefixer, axios, lucide-react, postcss, react, react-dom, tailwindcss (+25 more)

### Community 44 - "main.py"
Cohesion: 0.07
Nodes (31): on_event, log_compute_backend(), get, root(), describe_compute_device(), device_summary(), get_hf_pipeline_device(), get_torch_device() (+23 more)

### Community 45 - "video_analysis_pipeline.py"
Cohesion: 0.31
Nodes (5): ===============================================================================…, extract_audio_track(), Extracts the audio track from a video as a 16kHz mono WAV file using FFmpeg.…, cleanup_temporary_frames(), Safely removes temporary raw frames extracted in storage/frames/{video_id}/…

### Community 46 - "json_storage.py"
Cohesion: 0.08
Nodes (40): delete, calculate_intersection(), extract_features(), generate_dataset(), process_image_wrapper(), Extracts multi-domain physics & illumination forensic features with scale-…, Path, check_status() (+32 more)

### Community 47 - "Video Authenticity Detector (AuthentiGuard AI)"
Cohesion: 0.12
Nodes (16): 10. Limitations & Future Work, 1. Backend Setup, 1. Project Overview, 2. Features, 2. Frontend Setup, 3. Technology Stack, 4. Folder Structure, 5. Installation & Setup (+8 more)

### Community 48 - "evaluate_pillar2_video_suite"
Cohesion: 0.14
Nodes (14): Any, Extracts container and stream metadata using OpenCV and FFmpeg probe.…, run_metadata_analysis(), butterworth_bandpass(), Any, ndarray, backend/services/rppg_analyzer.py ================================== Pillar 2…, Applies zero-phase 4th-order Butterworth bandpass filtering. (+6 more)

### Community 49 - "video_processor.py"
Cohesion: 0.22
Nodes (11): post, UploadFile, Accepts video upload, validates format & size, saves to storage/uploads/, and…, upload_video(), generate_video_id(), UploadFile, Generates a clean, unique ID in the format VID_XXXXXX., Validates file extension and basic properties. Returns the lowercase file… (+3 more)

### Community 51 - "pillars_integrated.py"
Cohesion: 0.15
Nodes (28): fuse_multi_pillar_verdict(), Fuses inferences from Pillar 1, Pillar 4, and Pillar 5 into a unified verdict.…, load_pillar1_vit(), Executes Pillar 1: Vision Transformer & Spectral ELA Forensics. Applies Error…, Loads Pillar 1 Vision Transformer Model (Hugging Face ViT) with Dual-Expert…, run_pillar1_inference(), Executes Pillar 4: Semantic Document & Benford's Law OCR Forensics., run_pillar4_inference() (+20 more)

### Community 52 - "run_lip_sync_analysis"
Cohesion: 0.36
Nodes (7): compute_audio_energy_at_timestamps(), compute_mouth_motion_series(), Any, Extracts RMS speech energy from WAV file corresponding to video frame…, Cross-correlates mouth movement series with audio speech activity series.…, Computes frame-by-frame mouth region optical motion / variance., run_lip_sync_analysis()

### Community 53 - "core/__init__.py"
Cohesion: 0.16
Nodes (24): ===============================================================================…, ===============================================================================…, ===============================================================================…, create_fallback_xai_response(), create_xai_evidence(), create_xai_response(), generate_consensus_xai(), generate_pillar1_xai() (+16 more)

### Community 54 - "run_visual_analysis"
Cohesion: 0.27
Nodes (10): analyze_boundary_anomaly(), analyze_frequency_texture(), analyze_sensor_noise(), Any, ndarray, Performs FFT analysis to check for high-frequency attenuation or unnatural grid…, Measures physical camera sensor noise using median filter residual. Real mobile…, Measures edge gradient discontinuity along the perimeter of the face crop which… (+2 more)

### Community 55 - "detect_faces_in_frames"
Cohesion: 0.25
Nodes (6): detect_faces_in_frames(), FaceDetector, Any, ndarray, Detects faces in an RGB frame. Returns list of detected face dictionaries…, Runs face detection over all extracted frames. Attaches detected faces to each…

### Community 64 - "classify_audio"
Cohesion: 0.47
Nodes (4): classify_audio(), ===============================================================================…, main(), ===============================================================================…

### Community 65 - "generate_pillar3_audio_xai"
Cohesion: 0.15
Nodes (17): get_model(), compute_integrated_gradients_audio(), extract_important_audio_segments(), generate_pillar3_audio_xai(), generate_saliency_spectrogram_plot(), Any, device, Module (+9 more)

### Community 67 - "generate_pillar1_attention_xai"
Cohesion: 0.14
Nodes (13): compute_ela_image(), ===============================================================================…, Computes JPEG Error Level Analysis (ELA) at specified quality (default 90).…, compute_attention_rollout(), generate_pillar1_attention_xai(), identify_highest_attribution_region(), Any, Image (+5 more)

### Community 68 - "generate_pillar4_statistical_xai"
Cohesion: 0.33
Nodes (8): compute_benford_comparison_table(), generate_benford_chart_plot(), generate_pillar4_statistical_xai(), Any, ===============================================================================…, Renders a publication-grade dark-themed comparison plot for Benford…, Generates explainable statistical evidence for Pillar 4 Benford's Law OCR…, Computes per-digit comparisons between observed and expected Benford…

### Community 69 - "test_full_pipeline"
Cohesion: 0.17
Nodes (12): Any, Runs audio authenticity detection on extracted WAV file. Does not penalize…, run_audio_analysis(), extract_and_annotate_suspicious_frames(), Any, Ranks extracted frames by composite anomaly score and saves top suspicious…, extract_sampled_frames(), Any (+4 more)

### Community 71 - "generate_pillar5_shap_xai"
Cohesion: 0.32
Nodes (7): generate_pillar5_shap_xai(), generate_shap_waterfall_plot(), Any, ndarray, ===============================================================================…, Renders a publication-grade dark-themed SHAP Waterfall plot. Features pushing…, Computes TreeSHAP feature attributions for Pillar 5 Physical Geometry &…

### Community 72 - "pillar5_engine.py"
Cohesion: 0.33
Nodes (5): calculate_intersection(), load_pillar5_deep_backbone(), ===============================================================================…, Loads feature extractor backbone for Pillar 5 Deep Fusion (cached in memory)., Computes Cartesian intersection point between two line segments.

## Knowledge Gaps
- **66 isolated node(s):** `$schema`, `plugins`, `react/rules-of-hooks`, `react/only-export-components`, `name` (+61 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 343 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConsensusEngine` connect `orchestrator.py` to `train_head.py`, `.fuse`, `run_feature_fusion_and_classification`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `create_fallback_xai_response()` connect `core/__init__.py` to `generate_pillar3_audio_xai`, `generate_pillar1_attention_xai`, `execute_video_analysis_pipeline`, `video_analysis_pipeline.py`, `generate_pillar2_temporal_xai`, `pillar4_engine.py`, `pillars_integrated.py`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `OrchestratorAgent` connect `orchestrator.py` to `SpatialDetector`, `TemporalAnalyzer`, `load_video_frames`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **What connects `$schema`, `plugins`, `react/rules-of-hooks` to the rest of the system?**
  _66 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pillar1_pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07084785133565621 - nodes in this community are weakly interconnected._
- **Should `extract_efficientnet_embedding` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `train_and_evaluate_pillar5` be split into smaller, more focused modules?**
  _Cohesion score 0.10541310541310542 - nodes in this community are weakly interconnected._