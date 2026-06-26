@echo off
REM Launch the demo while screen-recording to an mp4 (needs ffmpeg on PATH).
REM Extra args pass through to the app (e.g. record.bat --video clip.mp4).
setlocal
cd /d "%~dp0.."
if not exist ".venv" (
  echo [record] .venv not found. Run scripts\setup.bat first.
  exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [record] ffmpeg not found on PATH. Install it first ^(winget install Gyan.FFmpeg^).
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m alertness.record_demo %*
endlocal
