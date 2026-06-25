@echo off
REM Score a predictions CSV (true vs predicted) with the same metrics as the rules.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m alertness.score %*
endlocal
