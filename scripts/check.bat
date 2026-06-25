@echo off
REM Local gate before committing: lint -> type check -> tests.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat

echo [check] ruff...
ruff check src tests || exit /b 1
echo [check] pyright...
pyright || exit /b 1
echo [check] pytest...
pytest || exit /b 1
echo [check] All passed.
endlocal
