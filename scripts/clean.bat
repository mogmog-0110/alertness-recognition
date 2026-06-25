@echo off
REM Remove the virtual environment, downloaded model, and caches (keeps code and data).
setlocal
cd /d "%~dp0.."
echo [clean] Removing .venv, models, and caches...
if exist ".venv" rmdir /s /q ".venv"
if exist "models" rmdir /s /q "models"
if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
if exist ".ruff_cache" rmdir /s /q ".ruff_cache"
for /d /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"
echo [clean] Done. Re-run scripts\setup.bat to reinstall.
endlocal
