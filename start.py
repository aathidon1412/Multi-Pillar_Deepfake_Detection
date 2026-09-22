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
    print("=" * 60)
    print("1. Full-Stack Web App (React + FastAPI on http://localhost:5173) [DEFAULT]")
    print("2. Streamlit Unified Dashboard (All 5 Pillars on http://localhost:8501)")
    print("=" * 60)
    
    try:
        choice = input("Enter choice (1 or 2, default is 1): ").strip()
    except Exception:
        choice = "1"
    
    if choice == "2":
        print("\n[+] Starting Streamlit Dashboard and launching browser...")
        streamlit_exe = ROOT_DIR / ".venv" / "Scripts" / "streamlit.exe"
        if not streamlit_exe.exists():
            streamlit_exe = "streamlit"
        
        # Open in new browser
        webbrowser.open_new("http://localhost:8501")
        subprocess.run([str(streamlit_exe), "run", "app_streamlit.py", "--server.headless", "false"], cwd=str(ROOT_DIR))
    else:
        print("\n[+] Starting Full-Stack Web App and launching browser...")
        fullstack_bat = ROOT_DIR / "run_fullstack.bat"
        subprocess.run([str(fullstack_bat)], shell=True, cwd=str(ROOT_DIR))

if __name__ == "__main__":
    main()
