@echo off
REM Run the type checker (pyright). Catches Protocol mismatches early.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
pyright
endlocal
