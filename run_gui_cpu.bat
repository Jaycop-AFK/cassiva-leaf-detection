@echo off
cd /d "%~dp0"
echo Starting Cassava Leaf AI GUI with CPU PyTorch...
venv_cpu\Scripts\python.exe -m streamlit run app.py --server.port 8503
pause
