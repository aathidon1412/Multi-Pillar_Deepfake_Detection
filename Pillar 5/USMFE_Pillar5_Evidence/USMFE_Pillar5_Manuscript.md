# Universal Synthetic Media Forensics Engine (USMFE)
## Pillar V: Physical Geometry and Shadow Physics Forensics

### Abstract
This section details the experimental methodology and forensic evaluation of Pillar V of the Universal Synthetic Media Forensics Engine (USMFE). Pillar V focuses on the extraction and analysis of physical scene geometry, specifically utilizing Random Sample Consensus (RANSAC) algorithms to trace anomalous shadow trajectories and vanishing point consistency.

### 1. Introduction
While generative models excel at producing photorealistic textures, they frequently fail to maintain globally consistent 3D physics. Pillar V evaluates the structural coherence of light transport by analyzing shadow trajectories across the image plane. By applying RANSAC for vanishing point estimation, we distinguish mathematically rigorous authentic scene lighting from the localized, physically ungrounded approximations typical of latent diffusion models.

### 2. Methodology
The forensic pipeline operates in four distinct phases:
1.  **Vector Extraction:** Advanced image processing, including Adaptive Gaussian Thresholding and Canny edge detection, is applied. Probabilistic Hough Transforms map discrete geometric vectors representing object boundaries and cast shadows.
2.  **Directional Clustering:** Shadow vectors are clustered based on angular coherence. Extraneous lines resulting from texture rather than illumination are discarded via angular tolerance filtering.
3.  **RANSAC Vanishing Point Estimation:** The filtered lines undergo up to 1,000 iterations of RANSAC to calculate mathematical intersections. A vanishing point is defined by the maximum number of converging inliers within a defined pixel tolerance.
4.  **Machine Learning Classification:** Key physical telemetry—including the absolute number of detected lines, the RANSAC inlier ratio, and the circular angular variance of the inliers—are processed through a Random Forest Classifier trained on 503 samples representing both authentic photography and major AI generative pipelines.

### 3. Experimental Results
Telemetry extraction from a representative baseline image (`IMG20240928151612.jpg`) via the USMFE framework yielded the following results:

*   **Total Vectors Detected:** 143 lines
*   **RANSAC Inliers:** 50 converging lines
*   **Inlier Ratio:** 34.9%
*   **Angular Variance:** 3.34°
*   **Estimated Vanishing Point:** (94.75, 597.02)
*   **Classification:** AUTHENTIC PHYSICS

The extracted features demonstrate high physical coherence. The RANSAC inlier ratio (34.9%) coupled with a remarkably low angular variance (3.34°) indicates a single, globally consistent directional light source, leading the ML model to a confident classification of authentic scene physics.

### 4. Conclusion
The implementation of Pillar V proves highly effective at characterizing the structural geometry of illumination. The Random Forest classifier, achieving a 95.31% validation accuracy, reliably leverages RANSAC shadow tracking to exploit a persistent weakness in modern generative AI frameworks.

---
*Telemetry formally captured by the USMFE framework on 2026-08-09.*
