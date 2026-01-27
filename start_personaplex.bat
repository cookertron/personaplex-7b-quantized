@echo off
REM PersonaPlex Launcher

cd /d "%~dp0"

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Launch GUI
python -m gui.app %*
