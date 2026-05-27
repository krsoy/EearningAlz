#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Setup uv CUDA + vLLM environment"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "Started at: $(date)"
echo "============================================================"

# ============================================================
# Cache paths
# ============================================================

export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"

export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR"

# ============================================================
# Remove old broken environment
# ============================================================

echo ""
echo "Removing old .venv..."
rm -rf .venv

# ============================================================
# Create fresh uv environment
# ============================================================

echo ""
echo "Creating uv venv..."
uv venv --python 3.12 .venv

source .venv/bin/activate

echo ""
echo "Python:"
which python
python --version

# ============================================================
# Upgrade basic tools
# ============================================================

echo ""
echo "Installing basic tools..."
uv pip install --upgrade pip setuptools wheel packaging ninja

# ============================================================
# IMPORTANT:
# Install CUDA PyTorch FIRST.
#
# For L4 + recent cluster CUDA, cu126 is usually safe if your driver supports it.
# This avoids CPU torch being installed by other packages.
# ============================================================

echo ""
echo "Installing CUDA PyTorch FIRST..."

uv pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

# ============================================================
# Verify CUDA torch before installing anything else
# ============================================================

echo ""
echo "Checking torch CUDA immediately after install..."

python - <<'PY'
import torch
import inspect

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("torch file:", inspect.getfile(torch))

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA torch install failed. "
        "Do not continue. You probably installed CPU torch or have no GPU allocated."
    )

print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

# ============================================================
# Install common data / RAG packages
# ============================================================

echo ""
echo "Installing data / RAG packages..."

uv pip install \
  pandas \
  numpy \
  pyarrow \
  tqdm \
  scikit-learn \
  chromadb \
  sentence-transformers \
  requests \
  pydantic \
  networkx \
  matplotlib

# ============================================================
# Install vLLM AFTER CUDA torch is already installed
# ============================================================

echo ""
echo "Installing vLLM..."

uv pip install vllm

# ============================================================
# Final verification
# ============================================================

echo ""
echo "Final environment check..."

python - <<'PY'
import sys
import inspect
import torch

print("PYTHON:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("torch file:", inspect.getfile(torch))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA disappeared after package installation.")

print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

import pandas as pd
import numpy as np
import chromadb
import sentence_transformers
import vllm

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("chromadb:", chromadb.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
print("vllm:", vllm.__version__)
PY

echo ""
echo "============================================================"
echo "DONE. Environment ready."
echo "Finished at: $(date)"
echo "============================================================"