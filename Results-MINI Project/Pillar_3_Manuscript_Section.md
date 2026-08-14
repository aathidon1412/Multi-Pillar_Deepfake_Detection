# Pillar 3: Acoustic Forensics

## Mathematical Formulation
The acoustic forensics pillar utilizes Mel-spectrogram analysis and spectral phase derivatives. The Mel-scale conversion is governed by the following formula:
$$ m = 2595 \cdot \log_{10}(1 + \frac{f}{700}) $$

This formulation allows for high-resolution analysis of the vocal tract formants, while high-frequency phase gaps and spectral phase derivatives are extracted to expose vocoder artifacts.

## Implementation & Preprocessing
Audio frames are analyzed using a 25ms window, a 10ms hop length, and an FFT size of 2048. To specifically target neural vocoder artifacts, a Butterworth high-pass filter is applied to emphasize frequencies above 12 kHz, where synthetic generators like HiFi-GAN and WaveGlow frequently leave phase jitter and unnatural energy cutoffs.

## Engineering Challenges & Solutions
A significant challenge in acoustic forensics is the degradation caused by audio compression (e.g., MP3, AAC) over communication channels like WhatsApp or standard telephony. To mitigate compression loss, a spectral roll-off normalization technique was implemented, ensuring robust feature extraction even for low-bitrate (e.g., G.711) or highly compressed audio streams.

## Results Summary
The proposed methodology was evaluated on the ASVspoof 2021 Logical Access (LA) evaluation subset. As detailed in `Table_Pillar3_ASVspoof_Benchmark.csv` and the corresponding ROC curve (`Figure_Pillar3_ROC_EER.png`), the system achieved an overall Equal Error Rate (EER) of 1.84%, an AUC-ROC of 0.987, and an F1-Score of 0.981. A comparative Mel-spectrogram analysis (`Figure_Pillar3_MelSpectrograms.png`) visually confirms the detection of unnatural high-frequency phase gaps in synthetic voices.
