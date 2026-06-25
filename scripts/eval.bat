@echo off
REM Score recorded CSV sessions (rule-based prediction vs label).
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m alertness.evaluate %*
endlocal
