@echo off
title Multi-Pillar Deepfake Detection - Streamlit Runner
echo ========================================================
echo   Starting Multi-Pillar Deepfake Detection (Streamlit)
echo ========================================================
echo.

cd /d "%~dp0"

echo Starting Streamlit on http://localhost:8501 ...
start "Multi-Pillar Streamlit" cmd /k ""%~dp0.venv\Scripts\streamlit.exe" run app_streamlit.py --server.headless false"

timeout /t 3 /nobreak >nul
start "" "http://localhost:8501"

echo.
echo Streamlit app running at http://localhost:8501
echo.
pause
