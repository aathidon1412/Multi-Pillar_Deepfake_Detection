@echo off
title Multi-Pillar Deepfake Detection - Launcher
cd /d "%~dp0"

echo ========================================================
echo   Multi-Pillar Deepfake ^& Video Authenticity Detection
echo ========================================================
echo   [1] Full-Stack Web App (React + FastAPI)  [DEFAULT]
echo   [2] Streamlit Unified Dashboard (All 5 Pillars)
echo ========================================================
echo.
set /p choice="Select mode (1 or 2, press Enter for 1): "

if "%choice%"=="2" (
    echo Starting Streamlit Dashboard and launching browser...
    call run_streamlit.bat
) else (
    echo Starting Full-Stack Web App and launching browser...
    call run_fullstack.bat
)
