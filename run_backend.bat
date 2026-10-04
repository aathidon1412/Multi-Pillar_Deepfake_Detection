@echo off
title Backend Server (FastAPI)
cd /d "%~dp0"

set "PY_EXE="
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

set "PYTHONPATH=%~dp0"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TOKENIZERS_PARALLELISM=false"
rem Leave FORCE_CPU unset to use NVIDIA GPU when CUDA PyTorch is installed.

echo ========================================================
echo   Starting Video Authenticity Detector Backend
echo   URL: http://127.0.0.1:8000
echo ========================================================

goto :start_server

:after_crash
echo.
echo ========================================================
echo [Backend Process Exited with ExitCode: %ERRORLEVEL%] Restarting in 2s...
echo ========================================================
timeout /t 2 /nobreak >nul

:start_server
"%PY_EXE%" -m uvicorn backend.main:app --app-dir "Pillar 2\video-authenticity-detector" --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
goto :after_crash
