@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating the Python environment...
  py -m venv .venv || python -m venv .venv
)
call .venv\Scripts\activate.bat

rem Install PyTorch once, in the build that matches this machine
python -c "import torch" 2>nul || (
  where nvidia-smi >nul 2>nul && (
    echo NVIDIA GPU found: installing PyTorch with CUDA... this can take several minutes.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ) || (
    echo No NVIDIA GPU found: installing the CPU version of PyTorch. Translation will be slower.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  )
)
pip install -q -r requirements.txt
python app.py
pause
