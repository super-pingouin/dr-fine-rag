@echo off
title Dr Fine RAG Launcher
cls

echo ==================================================
echo 🩺 Starting Dr Fine RAG ecosystem...
echo ==================================================

:: 1) Launch FastAPI backend in a new terminal window
echo [1/2] Launching FastAPI Backend server (Port 8000)...
start "Dr Fine RAG - Backend" cmd /k "uvicorn main:app --port 8000"

:: Small 2-second pause to let the server initialize
timeout /t 2 /nobreak > nul

:: 2) Launch Streamlit frontend in a new terminal window
echo [2/2] Launching Streamlit Frontend interface (Port 8501)...
start "Dr Fine RAG - Frontend" cmd /k "streamlit run app.py"

echo ==================================================
echo  Everything is up and running!
echo    - API Docs : http://127.0.0.1:8000/docs
echo    - Web App  : http://localhost:8501
echo ==================================================
echo You can close the terminal windows to stop the services.
pause