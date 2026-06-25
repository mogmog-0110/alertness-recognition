@echo off
REM Show per-label feature distributions from recorded CSV sessions.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m alertness.analyze %*
endlocal
