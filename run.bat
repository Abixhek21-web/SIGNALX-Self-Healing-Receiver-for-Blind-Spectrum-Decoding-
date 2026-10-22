@echo off
echo ===================================================================
echo Automated IQ/WAV Signal Analysis & Signal Parameter Extraction
echo React.js + Tailwind CSS Frontend & FastAPI Backend Launcher
echo ===================================================================

echo Starting FastAPI Backend on port 8001...
start "RF Signal Analyzer - Backend API" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload"

echo Starting React + Tailwind Frontend on port 5174...
cd frontend
start "RF Signal Analyzer - React Frontend" cmd /k "npm.cmd run dev"

echo.
echo ===================================================================
echo System is launching!
echo Backend API Docs: http://127.0.0.1:8001/docs
echo React Web UI:     http://127.0.0.1:5174
echo ===================================================================
pause
