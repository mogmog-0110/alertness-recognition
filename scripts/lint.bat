@echo off
REM Run the linter.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
ruff check src tests
endlocal
