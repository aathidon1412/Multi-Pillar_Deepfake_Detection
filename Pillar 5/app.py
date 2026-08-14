import http.server
import socketserver
import os
import json
import urllib.parse
import cgi
import webbrowser
import threading
import sys
import shutil
import base64

# Import forensics engine
from pillar_5_forensics import analyze_shadows

PORT = 8501
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>USMFE - Pillar 5: Physical Geometry & Shadow Forensics</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-card: rgba(18, 26, 43, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-purple: #7f00ff;
            --accent-green: #00f076;
            --accent-red: #ff3366;
            --text-main: #f0f4f8;
            --text-muted: #8a99ad;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(127, 0, 255, 0.08) 0%, transparent 40%);
            padding-bottom: 60px;
        }

        header {
            padding: 30px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .badge {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: #000;
            font-weight: 800;
            font-size: 0.75rem;
            padding: 5px 12px;
            border-radius: 20px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .logo-title {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .logo-title span {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .container {
            max-width: 1200px;
            margin: 40px auto 0;
            padding: 0 24px;
        }

        .hero {
            text-align: center;
            margin-bottom: 40px;
        }

        .hero h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 12px;
            letter-spacing: -1px;
        }

        .hero p {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 650px;
            margin: 0 auto;
        }

        .main-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 35px;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        }

        .upload-zone {
            border: 2px dashed rgba(0, 242, 254, 0.35);
            border-radius: 18px;
            padding: 50px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(0, 242, 254, 0.02);
            position: relative;
        }

        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.06);
            transform: translateY(-2px);
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            color: var(--accent-cyan);
        }

        .upload-text {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .upload-subtext {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        #fileInput {
            display: none;
        }

        .btn-analyze {
            width: 100%;
            margin-top: 25px;
            padding: 16px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            border: none;
            border-radius: 14px;
            color: #0b0f19;
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn-analyze:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 242, 254, 0.35);
        }

        .btn-analyze:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Results Section */
        #resultsSection {
            display: none;
            margin-top: 40px;
        }

        .verdict-banner {
            padding: 24px 30px;
            border-radius: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .verdict-banner.authentic {
            background: linear-gradient(135deg, rgba(0, 240, 118, 0.15), rgba(0, 240, 118, 0.05));
            border-color: var(--accent-green);
        }

        .verdict-banner.fake {
            background: linear-gradient(135deg, rgba(255, 51, 102, 0.15), rgba(255, 51, 102, 0.05));
            border-color: var(--accent-red);
        }

        .verdict-left h3 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .verdict-title {
            font-size: 1.75rem;
            font-weight: 800;
        }

        .verdict-banner.authentic .verdict-title {
            color: var(--accent-green);
        }

        .verdict-banner.fake .verdict-title {
            color: var(--accent-red);
        }

        .verdict-score {
            text-align: right;
        }

        .score-val {
            font-size: 2.2rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        .score-lbl {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 25px;
        }

        @media(max-width: 900px) {
            .grid-layout {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: rgba(14, 21, 37, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 24px;
        }

        .card h4 {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--accent-cyan);
        }

        .plot-container img {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: block;
        }

        .metrics-list {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }

        .metric-name {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1rem;
            color: var(--text-main);
        }

        /* Spinner */
        .spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(0, 0, 0, 0.2);
            border-radius: 50%;
            border-top-color: #000;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Quick Test Examples */
        .examples-bar {
            margin-top: 25px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .examples-title {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
        }

        .example-chip {
            padding: 6px 14px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .example-chip:hover {
            background: rgba(0, 242, 254, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }
    </style>
</head>
<body>

<header>
    <div class="logo-area">
        <span class="badge">USMFE</span>
        <div class="logo-title">Pillar 5 <span>Forensics Engine</span></div>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);">
        Physical Geometry & Shadow Physics
    </div>
</header>

<div class="container">
    <div class="hero">
        <h1>Synthetic Media Forensics UI</h1>
        <p>Upload any image to verify physical light convergence, 3D shadow vanishing points, and ambient chromaticity.</p>
    </div>

    <div class="main-card">
        <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">⚡</div>
            <div class="upload-text" id="uploadPrompt">Drag & Drop an image here or click to browse</div>
            <div class="upload-subtext" id="fileChosenText">Supports JPG, PNG, JPEG</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <div class="examples-bar" id="examplesContainer">
            <span class="examples-title">Quick Test:</span>
            <button class="example-chip" onclick="testExample('image1.png')">image1.png (AI)</button>
            <button class="example-chip" onclick="testExample('image2.png')">image2.png (AI)</button>
            <button class="example-chip" onclick="testExample('imag7.jpeg')">imag7.jpeg (AI)</button>
            <button class="example-chip" onclick="testExample('image3.JPG')">image3.JPG (Real)</button>
            <button class="example-chip" onclick="testExample('image4.jpg')">image4.jpg (Real)</button>
            <button class="example-chip" onclick="testExample('image5.jpg')">image5.jpg (Real)</button>
            <button class="example-chip" onclick="testExample('image6.jpg')">image6.jpg (Real)</button>
        </div>

        <button class="btn-analyze" id="btnAnalyze" onclick="runAnalysis()">
            <div class="spinner" id="spinner"></div>
            <span id="btnText">Analyze Shadow Physics</span>
        </button>
    </div>

    <!-- Results Section -->
    <div id="resultsSection">
        <div class="verdict-banner" id="verdictBanner">
            <div class="verdict-left">
                <h3>Forensic Evaluation</h3>
                <div class="verdict-title" id="verdictText">AUTHENTIC PHYSICS</div>
            </div>
            <div class="verdict-score">
                <div class="score-val" id="confidenceScore">95.4%</div>
                <div class="score-lbl">Confidence Score</div>
            </div>
        </div>

        <div class="grid-layout">
            <div class="card">
                <h4>📐 Vanishing Point & Vector Overlay</h4>
                <div class="plot-container">
                    <img id="overlayPlot" src="" alt="Forensic Plot">
                </div>
            </div>

            <div class="card">
                <h4>📊 Quantitative Physics Telemetry</h4>
                <div class="metrics-list">
                    <div class="metric-row">
                        <span class="metric-name">Pillar Engine</span>
                        <span class="metric-val" style="color: var(--accent-cyan);">Pillar 5 (RANSAC & ML)</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Shadow Vectors Extracted</span>
                        <span class="metric-val" id="metaTotalLines">0</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">RANSAC Light Inliers</span>
                        <span class="metric-val" id="metaInliers">0</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Inlier Coherence Ratio</span>
                        <span class="metric-val" id="metaInlierRatio">0.0%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Angular Dispersion Variance</span>
                        <span class="metric-val" id="metaAngularVar">0.0°</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Light Source Vanishing Point</span>
                        <span class="metric-val" id="metaVP">[0, 0]</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let selectedFile = null;

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            document.getElementById('uploadPrompt').innerText = selectedFile.name;
            document.getElementById('fileChosenText').innerText = (selectedFile.size / 1024).toFixed(1) + ' KB selected';
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            selectedFile = e.dataTransfer.files[0];
            document.getElementById('uploadPrompt').innerText = selectedFile.name;
            document.getElementById('fileChosenText').innerText = (selectedFile.size / 1024).toFixed(1) + ' KB selected';
        }
    });

    async function testExample(filename) {
        document.getElementById('btnAnalyze').disabled = true;
        document.getElementById('spinner').style.display = 'inline-block';
        document.getElementById('btnText').innerText = 'Analyzing ' + filename + '...';
        
        try {
            const resp = await fetch('/api/test-example?file=' + encodeURIComponent(filename));
            const data = await resp.json();
            displayResults(data);
        } catch(err) {
            alert('Error running analysis: ' + err);
        } finally {
            document.getElementById('btnAnalyze').disabled = false;
            document.getElementById('spinner').style.display = 'none';
            document.getElementById('btnText').innerText = 'Analyze Shadow Physics';
        }
    }

    async function runAnalysis() {
        if (!selectedFile) {
            alert('Please select or drop an image first!');
            return;
        }

        const formData = new FormData();
        formData.append('image', selectedFile);

        document.getElementById('btnAnalyze').disabled = true;
        document.getElementById('spinner').style.display = 'inline-block';
        document.getElementById('btnText').innerText = 'Computing Physical Geometry...';

        try {
            const resp = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            displayResults(data);
        } catch (err) {
            alert('Analysis error: ' + err);
        } finally {
            document.getElementById('btnAnalyze').disabled = false;
            document.getElementById('spinner').style.display = 'none';
            document.getElementById('btnText').innerText = 'Analyze Shadow Physics';
        }
    }

    function displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const verdictBanner = document.getElementById('verdictBanner');
        const verdictText = document.getElementById('verdictText');
        const confidenceScore = document.getElementById('confidenceScore');

        verdictBanner.className = 'verdict-banner ' + (data.verdict.includes('AUTHENTIC') ? 'authentic' : 'fake');
        verdictText.innerText = data.verdict;
        confidenceScore.innerText = data.confidence_score + '%';

        document.getElementById('overlayPlot').src = data.plot_url + '?t=' + new Date().getTime();
        document.getElementById('metaTotalLines').innerText = data.total_lines_detected;
        document.getElementById('metaInliers').innerText = data.ransac_inlier_count;
        document.getElementById('metaInlierRatio').innerText = (data.ransac_inlier_ratio * 100).toFixed(1) + '%';
        document.getElementById('metaAngularVar').innerText = data.angular_variance_deg + '°';
        document.getElementById('metaVP').innerText = '(' + data.estimated_vanishing_point[0] + ', ' + data.estimated_vanishing_point[1] + ')';

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
</script>

</body>
</html>
"""

class ForensicsHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
            return
            
        elif parsed.path == "/api/test-example":
            query = urllib.parse.parse_qs(parsed.query)
            filename = query.get('file', [''])[0]
            
            src_path = os.path.join(os.path.dirname(__file__), "testing", filename)
            if not os.path.exists(src_path):
                self.send_error(404, "Test file not found")
                return
                
            base_name = os.path.splitext(filename)[0]
            out_sub = os.path.join(RESULTS_DIR, base_name)
            
            analyze_shadows(src_path, output_dir=out_sub, use_ml=True)
            
            telemetry_path = os.path.join(out_sub, f"{base_name}_telemetry.json")
            with open(telemetry_path) as f:
                telemetry = json.load(f)
                
            telemetry['plot_url'] = f"/results/{base_name}/{base_name}_overlay_plot.png"
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(telemetry).encode('utf-8'))
            return
            
        elif parsed.path.startswith("/results/"):
            rel_path = parsed.path.replace("/results/", "")
            full_path = os.path.join(RESULTS_DIR, rel_path)
            if os.path.exists(full_path):
                self.send_response(200)
                if full_path.endswith(".png"):
                    self.send_header("Content-type", "image/png")
                elif full_path.endswith(".json"):
                    self.send_header("Content-type", "application/json")
                self.end_headers()
                with open(full_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
                
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/analyze":
            content_type = self.headers['Content-Type']
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Bad request")
                return
                
            # Parse multipart upload
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
            )
            
            if 'image' not in form:
                self.send_error(400, "No image uploaded")
                return
                
            file_item = form['image']
            filename = file_item.filename
            if not filename:
                filename = "uploaded_image.png"
                
            save_path = os.path.join(UPLOAD_DIR, filename)
            with open(save_path, 'wb') as f:
                f.write(file_item.file.read())
                
            base_name = os.path.splitext(filename)[0]
            out_sub = os.path.join(RESULTS_DIR, base_name)
            
            analyze_shadows(save_path, output_dir=out_sub, use_ml=True)
            
            telemetry_path = os.path.join(out_sub, f"{base_name}_telemetry.json")
            with open(telemetry_path) as f:
                telemetry = json.load(f)
                
            telemetry['plot_url'] = f"/results/{base_name}/{base_name}_overlay_plot.png"
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(telemetry).encode('utf-8'))
            return

def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ForensicsHTTPHandler) as httpd:
        print(f"================================================================")
        print(f"🚀 Pillar 5 Forensics Web UI running at: http://localhost:{PORT}")
        print(f"   Open your browser to upload and test any image!")
        print(f"================================================================")
        webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
