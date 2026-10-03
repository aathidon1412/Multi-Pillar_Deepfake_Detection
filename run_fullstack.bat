@echo off
title Video Authenticity Detection System - FullStack Runner
echo ========================================================
echo   Starting Video Authenticity Detector (All 5 Pillars)
echo ========================================================
echo.

cd /d "%~dp0"

:: 1. Detect Python virtual environment
set "PY_EXE="
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Backend Server (FastAPI)" cmd /k "call "%~dp0run_backend.bat""

timeout /t 3 /nobreak >nul

echo [2/2] Starting React + Vite Frontend on http://localhost:5173 ...
start "Frontend UI (Vite)" cmd /k "cd /d "%~dp0Pillar 2\video-authenticity-detector\frontend" && npm run dev"

echo.
echo Both servers are launching!
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo Vite will automatically open your browser.
echo.
pause
