# Graph Report - Mini Project  (2026-09-14)

## Corpus Check
- 78 files · ~539,997 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 1, .xml 1)

## Summary
- 423 nodes · 507 edges · 42 communities (31 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `24eea6df`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train_pillar1_pipeline
- run_pillar5_training.py
- train_and_evaluate_pillar5
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
10. `run_pillar4_inference()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `classify_audio()`  [EXTRACTED]
  detect.py → core/pillar3_engine.py
- `predict_audio()` --calls--> `classify_audio()`  [EXTRACTED]
  app.py → core/pillar3_engine.py
- `evaluate_testing_suite()` --calls--> `fuse_multi_pillar_verdict()`  [EXTRACTED]
  test_eval.py → core/consensus.py
- `evaluate_testing_suite()` --calls--> `load_pillar1_vit()`  [EXTRACTED]
  test_eval.py → core/pillar1_engine.py
- `evaluate_testing_suite()` --calls--> `run_pillar1_inference()`  [EXTRACTED]
  test_eval.py → core/pillar1_engine.py

## Import Cycles
- 3-file cycle: `core/__init__.py -> core/pillar3_engine.py -> detect.py -> core/__init__.py`

## Communities (42 total, 4 thin omitted)

### Community 0 - "train_pillar1_pipeline"
Cohesion: 0.07
Nodes (35): Compose, Dataset, GradScaler, no_grad, Optimizer, FaceDeepfakeDataset, get_pillar1_dataloaders(), make_samples_from_folders() (+27 more)

### Community 1 - "run_pillar5_training.py"
Cohesion: 0.12
Nodes (25): Image, build_xy_matrices(), extract_efficientnet_embedding(), extract_physics_vector(), get_default_efficientnet(), device, Module, ndarray (+17 more)

### Community 2 - "train_and_evaluate_pillar5"
Cohesion: 0.11
Nodes (22): BaseEstimator, extract_efficientnet_embeddings(), main(), build_regularized_ensemble(), Any, ndarray, ===============================================================================…, Builds a strongly regularized soft-voting ensemble across 4 distinct… (+14 more)

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
Nodes (12): calculate_benford_metrics(), extract_digits_from_csv(), extract_leading_digits(), extract_text_from_image(), generate_artifacts(), generate_decision(), Determines the verdict based on classification thresholds, dynamically scaled…, Runs the forensic analysis and generates required artifacts. (+4 more)

### Community 12 - "__init__.py"
Cohesion: 0.06
Nodes (45): predict_audio(), Takes an uploaded or recorded audio file path and mode ("spoken" or "music"): -…, ===============================================================================…, Renders a clean inline badge showing detected file modality., render_modality_badge(), fuse_multi_pillar_verdict(), ===============================================================================…, Fuses inferences from Pillar 1, Pillar 4, and Pillar 5 into a unified verdict.… (+37 more)

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
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 247 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_and_evaluate_pillar5()` connect `train_and_evaluate_pillar5` to `run_pillar5_training.py`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **What connects `graphify`, `Workflow: graphify`, `1. Mathematical & Algorithmic Foundation` to the rest of the system?**
  _129 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_pillar1_pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07084785133565621 - nodes in this community are weakly interconnected._
- **Should `run_pillar5_training.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1164021164021164 - nodes in this community are weakly interconnected._
- **Should `train_and_evaluate_pillar5` be split into smaller, more focused modules?**
  _Cohesion score 0.10541310541310542 - nodes in this community are weakly interconnected._
- **Should `Universal Synthetic Media Forensics Engine (USMFE)` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Should `Pillar V: Physical Geometry, Multi-Domain Steganalysis & Deep Fusion Forensics` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._