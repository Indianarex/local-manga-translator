#!/usr/bin/env sh
# Linux / macOS launcher: creates the environment on first run, then starts the app.
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "Creating the Python environment..."
  "$PYTHON" -m venv .venv
fi
. .venv/bin/activate

# Install PyTorch once, in the build that matches this machine
if ! python -c "import torch" 2>/dev/null; then
  if [ "$(uname)" = "Darwin" ]; then
    echo "macOS: installing PyTorch (uses the Apple Silicon GPU when available)..."
    pip install torch torchvision
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found: installing PyTorch with CUDA... this can take several minutes."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  else
    echo "No NVIDIA GPU found: installing the CPU version of PyTorch. Translation will be slower."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
fi
pip install -q -r requirements.txt
exec python app.py
