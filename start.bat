@echo off
title Multi-Pillar Deepfake Detection System
cd /d "%~dp0"

:: Detect Python executable in venv or system
set "PY_EXE="
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

:: Set backend runtime environment variables
set "PYTHONPATH=%~dp0"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TOKENIZERS_PARALLELISM=false"

if /i "%~1"=="backend" goto :run_backend
if /i "%~1"=="frontend" goto :run_frontend
goto :run_all

:run_backend
echo ========================================================
echo   Starting Backend Server (FastAPI)
echo   URL: http://127.0.0.1:8000
echo ========================================================
:start_uvicorn
"%PY_EXE%" -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
echo.
echo [Backend Process Exited with ExitCode: %ERRORLEVEL%] Restarting in 2s...
ping -n 3 127.0.0.1 >nul 2>&1
goto :start_uvicorn

:run_frontend
echo ========================================================
echo   Starting Frontend UI (React + Vite)
echo   URL: http://localhost:5173
echo ========================================================
cd /d "%~dp0frontend"
npm run dev
goto :eof

:run_all
echo ========================================================
echo   Starting Multi-Pillar Deepfake Detection System
echo ========================================================
echo [1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "Backend Server (FastAPI)" cmd /k ""%~f0" backend"

ping -n 4 127.0.0.1 >nul 2>&1

echo [2/2] Launching React + Vite Frontend on http://localhost:5173 ...
start "Frontend UI (Vite)" cmd /k ""%~f0" frontend"

echo.
echo Both servers are launching!
echo   - Backend API: http://127.0.0.1:8000 (API Docs: http://127.0.0.1:8000/docs)
echo   - Frontend UI: http://localhost:5173
echo.
pause
