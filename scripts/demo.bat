@echo off
REM Start the browser demo. No arguments needed.
REM It detects this PC's LAN address, prepares the certificate, and prints
REM the URL to open on the phone.
setlocal
cd /d "%~dp0.."
if not exist ".venv" (
  echo [demo] .venv not found. Run scripts\setup.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m alertness.demo %*
endlocal
