@echo off
cd /d "%~dp0"

echo ============================================
echo  DocuMind - SRS Requirement Intelligence
echo ============================================
echo.
echo  Starting server on http://127.0.0.1:8001
echo.
echo  Wait for "Application startup complete"
echo  then open http://127.0.0.1:8001 in browser.
echo.
echo  Press Ctrl+C to stop the server.
echo ============================================
echo.

python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
pause
