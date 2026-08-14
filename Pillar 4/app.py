import os
import re
import numpy as np
import pytesseract
from PIL import Image
import fitz
import io
from flask import Flask, request, jsonify
from scipy.stats import chi2

app = Flask(__name__)

# You may need to set the tesseract cmd path if it's not in your system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_leading_digits(text_data):
    digits = []
    matches = re.findall(r'\b[1-9][0-9,]*\.?[0-9]*\b', text_data)
    for m in matches:
        clean = m.replace(',', '').replace('.', '')
        for char in clean:
            if char in '123456789':
                digits.append(int(char))
                break
    return digits

def analyze_benford(digits):
    n = len(digits)
    if n == 0: return "INSUFFICIENT DATA", 0, 0, 1.0, 0.0, [], []
    
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
    p_value = chi2.sf(chi_square, 8)
    
    scale = 100.0 / max(n, 30)
    threshold_strict = 0.020 + (0.010 * scale)
    threshold_loose = 0.035 + (0.010 * scale)
    
    if mae < threshold_strict:
        verdict = "AUTHENTIC"
        confidence = 95.0 - (mae / threshold_strict) * 5
    elif threshold_strict <= mae <= threshold_loose:
        verdict = "AUTHENTIC"
        confidence = 80.0 - ((mae - threshold_strict) / (threshold_loose - threshold_strict)) * 20
    else:
        verdict = "FORGED / AI-GENERATED"
        confidence = min(99.9, 85.0 + (mae * 200))
        if p_value < 0.01:
            confidence = max(confidence, 96.5)
            
    if p_value < 0.01 and mae > 0.025 and verdict == "AUTHENTIC":
        verdict = "FORGED / AI-GENERATED"
        confidence = max(90.0, 95.0 - (p_value * 100))
        
    return verdict, mae, chi_square, p_value, min(99.9, max(0.0, confidence)), obs_freqs, expected_freqs

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>USMFE - Pillar 4 Forensics</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: rgba(255, 255, 255, 0.03);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #00ffcc;
            --text-main: #ffffff;
            --text-muted: #8b949e;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }
        
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
            background: radial-gradient(circle at top, #162032 0%, var(--bg-color) 70%);
        }
        
        .container {
            width: 100%;
            max-width: 900px;
        }
        
        h1 {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00ffcc, #0088ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 40px;
            font-size: 1.1rem;
        }
        
        .glass-panel {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            margin-bottom: 30px;
        }
        
        textarea {
            width: 100%;
            height: 150px;
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-main);
            padding: 15px;
            font-size: 1rem;
            resize: vertical;
            outline: none;
            transition: border 0.3s ease;
            margin-bottom: 15px;
        }
        
        textarea:focus {
            border-color: var(--accent);
        }
        
        .file-upload-wrapper {
            position: relative;
            width: 100%;
            height: 60px;
            border: 2px dashed var(--border);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: border 0.3s ease;
            cursor: pointer;
            overflow: hidden;
            background: rgba(0,0,0,0.2);
            margin-bottom: 15px;
        }
        
        .file-upload-wrapper:hover {
            border-color: var(--accent);
        }
        
        .file-upload-wrapper input[type="file"] {
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .file-upload-text {
            color: var(--text-muted);
            font-weight: 600;
            pointer-events: none;
        }
        
        .btn {
            display: inline-block;
            background: linear-gradient(90deg, #00ffcc, #0088ff);
            color: #000;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            margin-top: 10px;
            width: 100%;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 255, 204, 0.3);
        }
        
        #results {
            display: none;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .metric-title {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
        }
        
        .verdict-authentic { color: #00ffcc; }
        .verdict-forged { color: #ff3366; }
        
        .chart-container {
            position: relative;
            height: 400px;
            width: 100%;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
            color: var(--accent);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>USMFE Forensics</h1>
        <p class="subtitle">Financial Document Forgery Detection via Benford's Law</p>
        
        <div class="glass-panel">
            <h3 style="margin-bottom: 15px; font-weight: 600;">Data Input (Paste Text OR Upload Image)</h3>
            
            <div class="file-upload-wrapper">
                <input type="file" id="imageInput" accept="image/*,application/pdf" onchange="updateFileName()">
                <span class="file-upload-text" id="fileNameText">Click to upload an Invoice Image or PDF</span>
            </div>
            
            <textarea id="inputText" placeholder="Or paste raw text containing numerical values here... (e.g. OCR output, CSV rows)"></textarea>
            
            <button class="btn" onclick="analyzeData()">Analyze Evidence</button>
            <div id="loading" class="loading">Processing AI Mathematical Forensics (OCR may take a moment)...</div>
        </div>
        
        <div id="results" class="glass-panel">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Verdict</div>
                    <div id="resVerdict" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Confidence</div>
                    <div id="resConfidence" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Mean Absolute Error</div>
                    <div id="resMAE" class="metric-value" style="color: #fff;">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Numbers Found (n)</div>
                    <div id="resN" class="metric-value" style="color: #fff;">-</div>
                </div>
            </div>
            
            <h3 style="margin-bottom: 20px; text-align: center;">Benford Distribution Analysis</h3>
            <div class="chart-container">
                <canvas id="benfordChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let chartInstance = null;
        
        function updateFileName() {
            const fileInput = document.getElementById('imageInput');
            const fileNameText = document.getElementById('fileNameText');
            if(fileInput.files.length > 0) {
                fileNameText.innerHTML = `<span style="color: var(--accent)">Selected File: ${fileInput.files[0].name}</span>`;
            } else {
                fileNameText.innerText = "Click to upload an Invoice Image or PDF";
            }
        }

        async function analyzeData() {
            const text = document.getElementById('inputText').value;
            const fileInput = document.getElementById('imageInput');
            
            if(!text.trim() && fileInput.files.length === 0) {
                return alert("Please enter some text OR upload an image to analyze.");
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            const formData = new FormData();
            formData.append('text', text);
            if(fileInput.files.length > 0) {
                formData.append('image', fileInput.files[0]);
            }
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';
                
                if(data.n === 0) {
                    alert("No numbers found in the provided data!");
                    return;
                }
                
                // Update UI
                const verdictEl = document.getElementById('resVerdict');
                verdictEl.innerText = data.verdict;
                verdictEl.className = 'metric-value ' + (data.verdict === 'AUTHENTIC' ? 'verdict-authentic' : 'verdict-forged');
                
                document.getElementById('resConfidence').innerText = data.confidence + '%';
                document.getElementById('resConfidence').className = 'metric-value ' + (data.verdict === 'AUTHENTIC' ? 'verdict-authentic' : 'verdict-forged');
                
                document.getElementById('resMAE').innerText = data.mae;
                document.getElementById('resN').innerText = data.n;
                
                // Render Chart
                renderChart(data.obs_freqs, data.expected_freqs, data.verdict);
                
                document.getElementById('results').style.display = 'block';
                
            } catch (err) {
                alert("Error connecting to forensic engine.");
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function renderChart(obs, exp, verdict) {
            const ctx = document.getElementById('benfordChart').getContext('2d');
            
            if (chartInstance) {
                chartInstance.destroy();
            }
            
            const barColor = verdict === 'AUTHENTIC' ? 'rgba(0, 255, 204, 0.8)' : 'rgba(255, 51, 102, 0.8)';
            const barBorder = verdict === 'AUTHENTIC' ? '#00ffcc' : '#ff3366';
            
            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                    datasets: [
                        {
                            label: 'Observed Frequency (%)',
                            data: obs,
                            backgroundColor: barColor,
                            borderColor: barBorder,
                            borderWidth: 1,
                            borderRadius: 4
                        },
                        {
                            label: "Benford's Expected (%)",
                            data: exp,
                            type: 'line',
                            borderColor: '#ffffff',
                            backgroundColor: '#ffffff',
                            borderWidth: 3,
                            pointRadius: 5,
                            pointBackgroundColor: '#ffffff',
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#8b949e' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#8b949e' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff', font: { family: 'Outfit', size: 14 } } }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/analyze', methods=['POST'])
def analyze():
    text_data = request.form.get('text', '')
    
    # Check if a file was uploaded
    if 'image' in request.files:
        upload_file = request.files['image']
        if upload_file.filename != '':
            try:
                file_bytes = upload_file.read()
                if upload_file.filename.lower().endswith('.pdf'):
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    pdf_text = ""
                    for page in doc:
                        pdf_text += page.get_text() + " "
                    text_data += " " + pdf_text
                else:
                    img = Image.open(io.BytesIO(file_bytes))
                    # Run OCR on the image
                    ocr_text = pytesseract.image_to_string(img)
                    text_data += " " + ocr_text
            except Exception as e:
                print(f"File Extraction Error: {e}")
                pass
    
    digits = extract_leading_digits(text_data)
    verdict, mae, chi_square, p_value, conf, obs_freqs, expected_freqs = analyze_benford(digits)
    
    return jsonify({
        "n": len(digits),
        "verdict": verdict,
        "mae": round(mae, 5) if len(digits)>0 else 0,
        "chi_square": round(chi_square, 4),
        "p_value": round(p_value, 5),
        "confidence": round(conf, 2),
        "obs_freqs": obs_freqs,
        "expected_freqs": expected_freqs
    })

if __name__ == '__main__':
    print("Starting USMFE Pillar 4 UI Server with OCR capabilities on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
