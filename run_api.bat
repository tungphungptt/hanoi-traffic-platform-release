@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Hanoi Traffic Platform + AICC - Unified Local Bootstrap
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_demo.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Demo bootstrap failed. Read the error above and fix it before retrying.
  pause
  exit /b 1
)
endlocal
