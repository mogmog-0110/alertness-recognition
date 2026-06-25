@echo off
REM Guided data collection: follow the on-screen prompts (auto-labeled).
setlocal
cd /d "%~dp0.."
if not exist ".venv" (
  echo [collect] .venv not found. Run scripts\setup.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m alertness --guided %*
endlocal
