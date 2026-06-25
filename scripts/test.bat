@echo off
REM Run the tests. Extra args are passed through to pytest.
setlocal
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
pytest %*
endlocal
