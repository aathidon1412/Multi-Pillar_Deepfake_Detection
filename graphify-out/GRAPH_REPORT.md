# Graph Report - Mini Project  (2026-09-13)

## Corpus Check
- 71 files · ~539,340 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 1, .xml 1)

## Summary
- 390 nodes · 443 edges · 42 communities (31 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bec567b9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pillar1_pipeline
- extract_features.py
- main
- Universal Synthetic Media Forensics Engine (USMFE)
- Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics
- app_streamlit.py
- Module: Pillar 3 — Acoustic & Voice Deepfake Forensics
- 5. Visual Proofs & Explainability (XAI)
- 2. Iteration-by-Iteration Breakdown
- 2. Iteration-by-Iteration Breakdown
- analyze_shadows
- generate_artifacts
- classify_audio
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

## God Nodes (most connected - your core abstractions)
1. `train_pillar1_pipeline()` - 11 edges
2. `Universal Synthetic Media Forensics Engine (USMFE)` - 10 edges
3. `extract_efficientnet_embedding()` - 9 edges
4. `analyze_shadows()` - 9 edges
5. `Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics` - 9 edges
6. `get_pillar1_dataloaders()` - 8 edges
7. `train_one_epoch()` - 8 edges
8. `extract_physics_vector()` - 8 edges
9. `train_and_evaluate_pillar5()` - 8 edges
10. `classify_audio()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `predict_audio()` --calls--> `classify_audio()`  [EXTRACTED]
  app.py → detect.py
- `process_all_test_images()` --calls--> `analyze_shadows()`  [EXTRACTED]
  Pillar 5/batch_run_testing.py → Pillar 5/pillar_5_forensics.py
- `get_pillar1_dataloaders()` --calls--> `get_pillar1_transforms()`  [EXTRACTED]
  Pillar 1/dataset_adapter.py → Pillar 1/train_pillar1_vit.py
- `main()` --calls--> `get_pillar1_dataloaders()`  [EXTRACTED]
  Pillar 1/run_pillar1_training.py → Pillar 1/dataset_adapter.py
- `main()` --calls--> `train_pillar1_pipeline()`  [EXTRACTED]
  Pillar 1/run_pillar1_training.py → Pillar 1/train_pillar1_vit.py

## Import Cycles
- None detected.

## Communities (42 total, 4 thin omitted)

### Community 0 - "train_pillar1_pipeline"
Cohesion: 0.07
Nodes (35): Compose, Dataset, GradScaler, no_grad, Optimizer, FaceDeepfakeDataset, get_pillar1_dataloaders(), make_samples_from_folders() (+27 more)

### Community 1 - "extract_features.py"
Cohesion: 0.08
Nodes (35): BaseEstimator, Image, build_xy_matrices(), extract_efficientnet_embedding(), extract_physics_vector(), get_default_efficientnet(), device, Module (+27 more)

### Community 2 - "main"
Cohesion: 0.18
Nodes (12): extract_efficientnet_embeddings(), main(), extract_single_image_features(), main(), Worker function for parallel tabular feature extraction. Takes a tuple…, calculate_intersection(), extract_deep_embeddings(), extract_multi_domain_features() (+4 more)

### Community 3 - "Universal Synthetic Media Forensics Engine (USMFE)"
Cohesion: 0.12
Nodes (16): 1. Clone the Repository, 2. Create and Activate Virtual Environment, 3. Install Dependencies, 4. Install Tesseract OCR (Required for Pillar 4 Document Mode), Audio Deepfake Detection (Pillar 3):, Available Analysis Modes:, 🛠️ CLI Standalone Tools, 📊 Evaluation & Verification (+8 more)

### Community 4 - "Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics"
Cohesion: 0.10
Nodes (19): 1.1 The Challenge of High-Resolution Generative Media, 1.2 Legacy Vulnerabilities & Root Causes of Detection Failure, 1. Introduction & Problem Diagnosis, 2. Multi-Domain Dataset Architecture, 3.1 Tabular Physics, Illumination & Steganalysis Features (24 Features), 3.2 Deep Visual Feature Embeddings (1,280 Features), 3. Comprehensive Feature Engineering (1,304 Dimensions), 4. Retraining Pipeline & Parallel Execution (+11 more)

### Community 5 - "app_streamlit.py"
Cohesion: 0.15
Nodes (18): analyze_benford_law(), calculate_intersection(), extract_digits_from_text(), load_pillar1_vit(), load_pillar5_deep_backbone(), load_pillar5_ml_bundle(), cache_resource, Loads Pillar 1 Vision Transformer Model (Hugging Face ViT). Prioritizes the… (+10 more)

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
Nodes (12): calculate_benford_metrics(), extract_digits_from_csv(), extract_leading_digits(), extract_text_from_image(), generate_artifacts(), generate_decision(), Determines the verdict based on classification thresholds, dynamically scaled…, Runs the forensic analysis and generates required artifacts. (+4 more)

### Community 12 - "classify_audio"
Cohesion: 0.31
Nodes (8): predict_audio(), Takes an uploaded or recorded audio file path and mode ("spoken" or "music"): -…, analyze_audio_composition(), classify_audio(), get_model(), main(), Classifies an audio file as REAL vs FAKE. Parameters: audio_path: path to audio…, Uses librosa to compute acoustic metrics and detect whether the input audio…

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
Cohesion: 0.53
Nodes (5): calculate_intersection(), extract_features(), generate_dataset(), process_image_wrapper(), Extracts multi-domain physics & illumination forensic features with scale-…

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

## Knowledge Gaps
- **129 isolated node(s):** `graphify`, `Workflow: graphify`, `1. Mathematical & Algorithmic Foundation`, `2. Engineering Challenges & Solutions`, `3. Experimental Findings & Analysis` (+124 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 230 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_regularized_ensemble()` connect `extract_features.py` to `main`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Why does `reduce_features()` connect `extract_features.py` to `main`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **What connects `graphify`, `Workflow: graphify`, `1. Mathematical & Algorithmic Foundation` to the rest of the system?**
  _129 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pillar1_pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07084785133565621 - nodes in this community are weakly interconnected._
- **Should `extract_features.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0782051282051282 - nodes in this community are weakly interconnected._
- **Should `Universal Synthetic Media Forensics Engine (USMFE)` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Should `Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._