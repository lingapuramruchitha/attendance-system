@echo off
cd /d "%~dp0"

echo Starting Face Recognition...

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    pause
    exit /b
)

if not exist "recognize_face.py" (
    echo ERROR: recognize_face.py not found!
    pause
    exit /b
)

venv\Scripts\python.exe recognize_face.py

echo.
echo Face Recognition program stopped.
pause