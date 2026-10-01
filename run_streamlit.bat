@echo off
title Multi-Pillar Deepfake Detection - Streamlit Runner
echo ========================================================
echo   Starting Multi-Pillar Deepfake Detection (Streamlit)
echo ========================================================
echo.

cd /d "%~dp0"

:: Check if virtual environment exists, if not, create and install automatically
if not exist "%~dp0venv\Scripts\streamlit.exe" if not exist "%~dp0.venv\Scripts\streamlit.exe" (
    echo [SETUP] No virtual environment detected!
    echo [SETUP] Creating Python virtual environment (venv)...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Python was not found on your system PATH.
        echo Please install Python 3.10+ and re-run this script.
        pause
        exit /b 1
    )
    echo [SETUP] Installing all dependencies from requirements.txt...
    "%~dp0venv\Scripts\pip.exe" install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation encountered an issue.
        pause
        exit /b 1
    )
    echo [SETUP] Environment setup completed successfully!
    echo.
)

:: Launch Streamlit dashboard
if exist "%~dp0venv\Scripts\streamlit.exe" (
    start "Multi-Pillar Streamlit" cmd /k ""%~dp0venv\Scripts\streamlit.exe" run app_streamlit.py --server.headless false"
) else if exist "%~dp0.venv\Scripts\streamlit.exe" (
    start "Multi-Pillar Streamlit" cmd /k ""%~dp0.venv\Scripts\streamlit.exe" run app_streamlit.py --server.headless false"
) else (
    start "Multi-Pillar Streamlit" cmd /k "streamlit run app_streamlit.py --server.headless false"
)

timeout /t 3 /nobreak >nul
start "" "http://localhost:8501"

echo.
echo Streamlit app running at http://localhost:8501
echo.
pause
