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

echo ========================================================
echo   Starting Video Authenticity Detector Backend
echo   URL: http://127.0.0.1:8000
echo ========================================================

:run_loop
"%PY_EXE%" -m uvicorn backend.main:app --app-dir "Pillar 2\video-authenticity-detector" --host 0.0.0.0 --port 8000 --timeout-keep-alive 120 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%~dp0server_crash.log' -Append"
echo.
echo ========================================================
echo [Backend Process Exited with ExitCode: %ERRORLEVEL%] Restarting in 2s...
echo Details saved to server_crash.log
echo ========================================================
timeout /t 2 >nul
goto run_loop
