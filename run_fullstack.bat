@echo off
title Video Authenticity Detection System - FullStack Runner
echo ========================================================
echo   Starting Video Authenticity Detector (All 5 Pillars)
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Backend Server (FastAPI)" cmd /k ""%~dp0.venv\Scripts\python.exe" -m uvicorn backend.main:app --app-dir "Pillar 2\video-authenticity-detector" --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Starting React + Vite Frontend on http://localhost:5173 ...
cd /d "%~dp0Pillar 2\video-authenticity-detector\frontend"
start "Frontend UI (Vite)" cmd /k "npm run dev"

timeout /t 2 /nobreak >nul

echo Opening browser at http://localhost:5173 ...
start "" "http://localhost:5173"

echo.
echo Both servers are running!
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo.
pause
