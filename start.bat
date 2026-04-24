@echo off
cd /d "d:\SEM 4 PROJECT\real-vs-ai-detector"
echo Starting Unified Streamlit Router...
python -m streamlit run app.py --server.port=8500
echo.
echo If the server crashed, please read the error above!
pause
