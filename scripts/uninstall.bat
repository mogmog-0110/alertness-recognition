@echo off
REM Uninstall menu. Choose how much to remove.
setlocal
cd /d "%~dp0.."
set "PROJ=%cd%"

echo What do you want to remove?
echo   [1] Libraries + model only  (.venv, models, caches) - keeps code and data
echo   [2] EVERYTHING incl. code   (delete the whole project folder)
echo   [C] Cancel
choice /c 12C /n /m "Select 1/2/C: "
if errorlevel 3 goto :cancel
if errorlevel 2 goto :everything
goto :libs

:libs
echo [uninstall] Removing .venv, models, and caches...
if exist ".venv" rmdir /s /q ".venv"
if exist "models" rmdir /s /q "models"
if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
if exist ".ruff_cache" rmdir /s /q ".ruff_cache"
for /d /r %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"
echo [uninstall] Done.
goto :end

:everything
echo.
echo WARNING: this deletes the ENTIRE folder including code and git history:
echo   %PROJ%
set /p CONFIRM="Type YES to confirm: "
if /i not "%CONFIRM%"=="YES" goto :cancel
REM 自分自身が中にいるので、別プロセスに削除させてから抜ける。
> "%TEMP%\al_uninstall.bat" echo @echo off
>> "%TEMP%\al_uninstall.bat" echo timeout /t 1 ^>nul
>> "%TEMP%\al_uninstall.bat" echo rmdir /s /q "%PROJ%"
>> "%TEMP%\al_uninstall.bat" echo del "%%~f0"
start "" "%TEMP%\al_uninstall.bat"
echo [uninstall] Deleting the project folder... you can close this window.
goto :end

:cancel
echo [uninstall] Cancelled.

:end
endlocal
