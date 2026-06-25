@echo off
REM Feature distributions + scorecard together. No args = process all CSVs in runs/.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m alertness.report %*
endlocal
