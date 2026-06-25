@echo off
REM Format the code and apply auto-fixable lint rules.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
ruff format src tests
ruff check --fix src tests
endlocal
