import cv2
import pytesseract
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare
import json
import os
import sys
from typing import Union

# Auto-configure Tesseract OCR executable path if not in system PATH
if not shutil.which("tesseract"):
    tesseract_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        r"D:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in tesseract_candidates:
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

# Set Seaborn style for publication-quality plots
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

def extract_text_from_image(image_path: str) -> str:
    """Preprocesses the image and extracts text using Tesseract OCR."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # Preprocessing
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Extract text
    # Assuming standard pytesseract installation
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # Save extracted text to extracted.txt in Pillar 4 directory
    pillar_4_dir = os.path.dirname(os.path.abspath(__file__))
    extracted_txt_path = os.path.join(pillar_4_dir, "extracted.txt")
    with open(extracted_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return text

def extract_leading_digits(text: str) -> list:
    """Extracts the first non-zero digit from numerical values in text."""
    pattern = r'\b[1-9][0-9,]*\.?[0-9]*\b'
    matches = re.findall(pattern, text)
    
    leading_digits = []
    for match in matches:
        clean_num = match.replace(',', '')
        for char in clean_num:
            if char.isdigit() and char != '0':
                leading_digits.append(int(char))
                break
                
    return leading_digits

def extract_digits_from_csv(csv_path: str) -> list:
    """Extracts leading digits from a CSV containing numerical records."""
    df = pd.read_csv(csv_path)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    values = df.values.flatten()
    values = values[~np.isnan(values)]
    
    leading_digits = []
    for val in values:
        if val > 0:
            val_str = str(val).replace('.', '')
            for char in val_str:
                if char.isdigit() and char != '0':
                    leading_digits.append(int(char))
                    break
                    
    return leading_digits

def calculate_benford_metrics(leading_digits: list):
    """Calculates observed, theoretical frequencies, MAE, and Chi-Square stats."""
    n = len(leading_digits)
    if n == 0:
        raise ValueError("No valid numerical data extracted to perform Benford's Law analysis.")

    counts = {d: 0 for d in range(1, 10)}
    for d in leading_digits:
        counts[d] += 1
        
    observed_counts = np.array([counts[d] for d in range(1, 10)])
    observed_freq = observed_counts / n
    
    digits = np.arange(1, 10)
    expected_freq = np.log10(1 + 1/digits)
    expected_counts = expected_freq * n
    
    mae = np.mean(np.abs(observed_freq - expected_freq))
    
    expected_counts = np.maximum(expected_counts, 1e-8)
    chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    
    return {
        'n': n,
        'observed_counts': observed_counts,
        'observed_freq': observed_freq,
        'expected_freq': expected_freq,
        'mae': mae,
        'chi2_stat': chi2_stat,
        'p_value': p_value
    }

def generate_decision(mae: float, p_value: float, n: int) -> tuple:
    """Determines the verdict based on classification thresholds, dynamically scaled by sample size."""
    # Dynamic MAE thresholds based on sample size (n)
    # The smaller the n, the more leeway we give for the natural volatility of small datasets
    scale = 100.0 / max(n, 30)
    threshold_strict = 0.012 + (0.01 * scale)
    threshold_loose = 0.025 + (0.010 * scale)

    if mae < threshold_strict:
        verdict = "AUTHENTIC"
        confidence = 95.0 - (mae / threshold_strict) * 5
    elif threshold_strict <= mae <= threshold_loose:
        verdict = "AUTHENTIC"
        confidence = 80.0 - ((mae - threshold_strict) / (threshold_loose - threshold_strict)) * 20
    else:
        # If MAE fails the loose threshold but the Chi-Square test still passes significantly, save it
        if p_value > 0.15:
            verdict = "AUTHENTIC"
            confidence = 60.0
        else:
            verdict = "FORGED / AI-GENERATED"
            confidence = min(99.9, 85.0 + (mae * 200))
            if p_value < 0.01:
                confidence = max(confidence, 96.5)
        
    # Override authentic if statistical test completely fails, unless MAE is exceptionally good
    if p_value < 0.01 and verdict == "AUTHENTIC" and mae > 0.015:
        verdict = "FORGED / AI-GENERATED"
        confidence = max(90.0, 95.0 - (p_value * 100))
        
    return verdict, round(confidence, 2)

def generate_artifacts(input_file: str, output_prefix: str = "usmfe_pillar4"):
    """Runs the forensic analysis and generates required artifacts."""
    print(f"Starting USMFE Pillar 4 Analysis on: {input_file}")
    
    # Create a directory named after the output prefix to store the results
    os.makedirs(output_prefix, exist_ok=True)
    
    ext = os.path.splitext(input_file)[1].lower()
    try:
        if ext in ['.csv']:
            leading_digits = extract_digits_from_csv(input_file)
        elif ext in ['.png', '.jpg', '.jpeg']:
            text = extract_text_from_image(input_file)
            leading_digits = extract_leading_digits(text)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        print(f"Error during extraction: {e}")
        return

    try:
        metrics = calculate_benford_metrics(leading_digits)
    except ValueError as e:
        print(e)
        return

    verdict, confidence = generate_decision(metrics['mae'], metrics['p_value'], metrics['n'])
    digits = np.arange(1, 10)
    
    # --- ARTIFACT A: Publication-Quality Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(digits, metrics['observed_freq'], color='gray', alpha=0.7, label='Observed Frequency')
    ax.plot(digits, metrics['expected_freq'], color='darkblue', marker='o', linewidth=2, markersize=8, label='Benford (Theoretical)')
    
    ax.set_xticks(digits)
    ax.set_xlabel("Leading Digit (1-9)", fontweight='bold')
    ax.set_ylabel("Frequency", fontweight='bold')
    ax.set_title("Benford's Law Analysis: Document Authenticity", fontweight='bold')
    ax.set_ylim(0, max(max(metrics['observed_freq']), max(metrics['expected_freq'])) + 0.05)
    
    annotation_text = f"MAE: {metrics['mae']:.4f}\np-value: {metrics['p_value']:.4e}\nVerdict: {verdict}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.65, 0.95, annotation_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
            
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    plot_file = os.path.join(output_prefix, f"{os.path.basename(output_prefix)}_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[+] Artifact A generated: {plot_file}")

    # --- ARTIFACT B: Quantitative Forensic Table ---
    abs_diff = np.abs(metrics['observed_freq'] - metrics['expected_freq'])
    
    df = pd.DataFrame({
        'Leading Digit (1-9)': digits,
        'Observed Count': metrics['observed_counts'],
        'Observed %': (metrics['observed_freq'] * 100).round(2),
        'Expected Benford %': (metrics['expected_freq'] * 100).round(2),
        'Absolute Difference': abs_diff.round(4)
    })
    
    table_file = os.path.join(output_prefix, f"{os.path.basename(output_prefix)}_table.csv")
    df.to_csv(table_file, index=False)
    md_table = df.to_markdown(index=False)
    print(f"[+] Artifact B generated: {table_file}")
    print("\n" + md_table + "\n")

    # --- ARTIFACT C: Telemetry Verdict JSON ---
    telemetry = {
        "pillar": "Pillar 4 - Financial Forensics",
        "total_numbers_analyzed": metrics['n'],
        "mean_absolute_error_mae": float(metrics['mae']),
        "chi_square_statistic": float(metrics['chi2_stat']),
        "p_value": float(metrics['p_value']),
        "verdict": verdict,
        "confidence_score": float(confidence)
    }
    
    json_file = os.path.join(output_prefix, f"{os.path.basename(output_prefix)}_telemetry.json")
    with open(json_file, 'w') as f:
        json.dump(telemetry, f, indent=4)
    print(f"[+] Artifact C generated: {json_file}")
    
    return telemetry

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="USMFE Pillar 4 - Document & Financial Forensics")
    parser.add_argument("input", help="Path to image, csv file, or a directory containing images for batch processing")
    parser.add_argument("--out", default="usmfe_pillar4", help="Prefix for output artifacts (only used if input is a single file)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        print(f"[*] Batch processing directory: {args.input}")
        valid_exts = ('.png', '.jpg', '.jpeg', '.csv')
        for filename in os.listdir(args.input):
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(args.input, filename)
                base_name = os.path.splitext(filename)[0]
                print(f"\n--- Processing {filename} ---")
                generate_artifacts(full_path, base_name)
    else:
        generate_artifacts(args.input, args.out)
