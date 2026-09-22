"""
================================================================================
Pillar 4 Core Engine: Document, Invoice & Benford's Law Statistical OCR Forensics
================================================================================
"""

import io
import os
import re
import shutil
import cv2
import numpy as np
from PIL import Image
from scipy.stats import chi2
import pytesseract
import fitz  # PyMuPDF for PDF documents

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

def extract_digits_from_text(text_data: str) -> list:
    """Extracts first significant digits (1-9) from numerical text strings."""
    matches = re.findall(r'\b[1-9][0-9,]*\.?[0-9]*\b', text_data)
    digits = []
    for m in matches:
        clean = m.replace(',', '').replace('.', '')
        for char in clean:
            if char in '123456789':
                digits.append(int(char))
                break
    return digits

def analyze_benford_law(digits: list) -> dict:
    """Performs statistical Benford's Law Chi-Square and MAE distribution analysis."""
    n = len(digits)
    if n < 5:
        return {
            "applicable": False,
            "reason": f"Insufficient numerical digits ({n} detected).",
            "verdict": "N/A (NON-DOCUMENT)",
            "confidence": 0.0,
            "weight": 0.0,
            "digits_count": n,
            "obs_freqs": [],
            "expected_freqs": []
        }
        
    counts = {i: 0 for i in range(1, 10)}
    for d in digits:
        counts[d] += 1
        
    mae = 0
    chi_square = 0
    obs_freqs = []
    expected_freqs = []
    
    for i in range(1, 10):
        obs_prop = counts[i] / n
        obs_freqs.append(obs_prop * 100)
        exp_prop = np.log10(1 + 1/i)
        expected_freqs.append(exp_prop * 100)
        
        mae += abs(obs_prop - exp_prop)
        expected_count = exp_prop * n
        if expected_count > 0:
            chi_square += ((counts[i] - expected_count)**2) / expected_count
            
    mae = mae / 9
    p_val = chi2.sf(chi_square, 8)
    scale = 100.0 / max(n, 30)
    threshold_strict = 0.020 + (0.010 * scale)
    threshold_loose = 0.035 + (0.010 * scale)
    
    if mae < threshold_strict:
        is_auth = True
        conf = 95.0 - (mae / threshold_strict) * 5
    elif threshold_strict <= mae <= threshold_loose:
        is_auth = True
        conf = 80.0 - ((mae - threshold_strict) / (threshold_loose - threshold_strict)) * 20
    else:
        is_auth = False
        conf = min(99.9, 85.0 + (mae * 200))
        
    if p_val < 0.01 and mae > 0.025 and is_auth:
        is_auth = False
        conf = max(90.0, 95.0 - (p_val * 100))
        
    verdict = "AUTHENTIC DOCUMENT" if is_auth else "FORGED / AI-SYNTHESIZED"
    
    return {
        "applicable": True,
        "verdict": verdict,
        "is_authentic": is_auth,
        "confidence": round(min(99.9, max(60.0, conf)), 2),
        "mae": round(mae, 4),
        "chi_square": round(chi_square, 4),
        "p_value": round(p_val, 5),
        "digits_count": n,
        "weight": 0.35,
        "real_probability": (conf / 100.0) if is_auth else (1.0 - conf / 100.0),
        "obs_freqs": obs_freqs,
        "expected_freqs": expected_freqs
    }

def run_pillar4_inference(image_or_bytes, is_pdf=False) -> dict:
    """Executes Pillar 4: Semantic Document & Benford's Law OCR Forensics."""
    try:
        text_data = ""
        extracted_image = None
        
        if is_pdf:
            if isinstance(image_or_bytes, (bytes, bytearray)):
                doc = fitz.open(stream=image_or_bytes, filetype="pdf")
            elif isinstance(image_or_bytes, str) and os.path.exists(image_or_bytes):
                doc = fitz.open(image_or_bytes)
            else:
                doc = fitz.open(stream=image_or_bytes, filetype="pdf")
            for page in doc:
                text_data += page.get_text() + " "
            if len(doc) > 0:
                first_page = doc[0]
                pix = first_page.get_pixmap(dpi=150)
                extracted_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            if isinstance(image_or_bytes, Image.Image):
                img_pil = image_or_bytes.convert("RGB")
            elif isinstance(image_or_bytes, (bytes, bytearray)):
                img_pil = Image.open(io.BytesIO(image_or_bytes)).convert("RGB")
            elif isinstance(image_or_bytes, np.ndarray):
                img_pil = Image.fromarray(image_or_bytes).convert("RGB")
            elif isinstance(image_or_bytes, str) and os.path.exists(image_or_bytes):
                img_pil = Image.open(image_or_bytes).convert("RGB")
            elif hasattr(image_or_bytes, "read"):
                img_pil = Image.open(image_or_bytes).convert("RGB")
            else:
                img_pil = Image.fromarray(image_or_bytes)
            extracted_image = img_pil
            
            np_img = np.array(img_pil)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            try:
                text_data = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6')
            except Exception:
                try:
                    text_data = pytesseract.image_to_string(img_pil)
                except Exception:
                    text_data = ""
                
        digits = extract_digits_from_text(text_data)
        res = analyze_benford_law(digits)
        res["extracted_image"] = extracted_image
        res["sample_text"] = (text_data[:300] + "...") if len(text_data) > 300 else text_data
        return res
    except Exception as e:
        return {
            "applicable": False,
            "reason": f"Document extraction error: {e}",
            "verdict": "N/A",
            "weight": 0.0,
            "extracted_image": None
        }
