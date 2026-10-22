Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Automated IQ/WAV Signal Analysis & Signal Parameter Extraction" -ForegroundColor Green
Write-Host "React.js + Tailwind CSS Frontend & FastAPI Backend Launcher" -ForegroundColor Yellow
Write-Host "===================================================================" -ForegroundColor Cyan

Write-Host "Starting FastAPI Backend (http://127.0.0.1:8001)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload"

Write-Host "Starting React + Tailwind Frontend (http://127.0.0.1:5174)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location 'frontend'; npm.cmd run dev"

Write-Host ""
Write-Host "Backend API docs: http://127.0.0.1:8001/docs" -ForegroundColor Green
Write-Host "React UI URL:    http://127.0.0.1:5174" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Cyan
