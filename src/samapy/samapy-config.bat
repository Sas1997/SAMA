@echo off
REM SAMAPy Configuration Wizard - Windows
REM Tries multiple Python commands

REM Try 'py' launcher first (recommended for Windows)
where py >nul 2>nul
if %errorlevel% == 0 (
    py "%~dp0samapy-config.py"
    exit /b
)

REM Try 'python3'
where python3 >nul 2>nul
if %errorlevel% == 0 (
    python3 "%~dp0samapy-config.py"
    exit /b
)

REM Try 'python'
where python >nul 2>nul
if %errorlevel% == 0 (
    python "%~dp0samapy-config.py"
    exit /b
)

REM If nothing works, show error
echo.
echo ======================================================================
echo ERROR: Python not found!
echo ======================================================================
echo.
echo Please install Python from: https://www.python.org/downloads/
echo.
echo Or try running directly:
echo   py samapy-config.py
echo   python3 samapy-config.py
echo   python samapy-config.py
echo.
echo ======================================================================
pause
