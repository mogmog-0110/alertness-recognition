@echo off
REM Set up the dev environment: venv, dependencies, and the face landmark model.
setlocal
cd /d "%~dp0.."

where python >nul 2>nul
if errorlevel 1 (
  echo [setup] Python not found. Install Python 3.10-3.12 first, then re-run:
  echo         https://www.python.org/downloads/    ^(check "Add python.exe to PATH"^)
  echo         or run:  winget install Python.Python.3.12
  exit /b 1
)

if not exist ".venv" (
  echo [setup] Creating virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo [setup] Installing dependencies ...
pip install -e ".[dev]"
if errorlevel 1 exit /b 1

if not exist "models" mkdir models
if not exist "models\face_landmarker.task" (
  echo [setup] Downloading FaceLandmarker model ...
  curl -L -o models\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  if errorlevel 1 (
    echo [setup] Model download failed. Check your network.
    exit /b 1
  )
)

echo [setup] Done. Run scripts\run.bat to start.
endlocal
