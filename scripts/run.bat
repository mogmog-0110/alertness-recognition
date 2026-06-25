@echo off
REM Start the app. Extra args are passed through (e.g. run.bat --record).
setlocal
cd /d "%~dp0.."
if not exist ".venv" (
  echo [run] .venv not found. Run scripts\setup.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m alertness %*
endlocal
