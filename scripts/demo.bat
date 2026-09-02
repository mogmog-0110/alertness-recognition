@echo off
REM ブラウザ版のデモを起動する。引数は要らない。
REM IP の検出も証明書の作成も自動で行い、端末で開く URL を表示する。
setlocal
cd /d "%~dp0.."
if not exist ".venv" (
  echo [demo] .venv がありません。先に scripts\setup.bat を実行してください。
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m alertness.demo %*
endlocal
