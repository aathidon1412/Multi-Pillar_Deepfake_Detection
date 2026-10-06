import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("  Multi-Pillar Deepfake & Video Authenticity Detector")
    print("  Starting Full-Stack Web App (React + FastAPI on http://localhost:5173)")
    print("=" * 60)
    
    fullstack_bat = ROOT_DIR / "run_fullstack.bat"
    subprocess.run([str(fullstack_bat)], shell=True, cwd=str(ROOT_DIR))

if __name__ == "__main__":
    main()
