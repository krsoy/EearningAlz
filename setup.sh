#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "PROJECT_ROOT=$PROJECT_ROOT"

# ============================================================
# Paths
# ============================================================

DATA_CSV="$PROJECT_ROOT/data/combined_transcript_data/combined_transcripts_deduplicated.csv"

CACHE_DIR="$PROJECT_ROOT/.cache"
HF_HOME_DIR="$CACHE_DIR/huggingface"
TORCH_HOME_DIR="$CACHE_DIR/torch"
UV_CACHE_DIR="$CACHE_DIR/uv"

RAG_OUT_DIR="$PROJECT_ROOT/rag_chroma_output"
CHROMA_DIR="$RAG_OUT_DIR/chroma_db"

mkdir -p "$HF_HOME_DIR" "$TORCH_HOME_DIR" "$UV_CACHE_DIR" "$RAG_OUT_DIR" "$CHROMA_DIR"

# ============================================================
# Install uv if missing
# ============================================================

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv found: $(uv --version)"
fi

# ============================================================
# Environment variables
# ============================================================

export PROJECT_ROOT="$PROJECT_ROOT"
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export TORCH_HOME="$TORCH_HOME_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"

export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ============================================================
# Remove old lock and old environment
# ============================================================

echo "Removing old uv.lock and .venv..."
rm -f "$PROJECT_ROOT/uv.lock"
rm -rf "$PROJECT_ROOT/.venv"

# ============================================================
# Create minimal pyproject.toml
# ============================================================

echo "Writing pyproject.toml..."

cat > pyproject.toml <<'EOF'
[project]
name = "earning-call-rag"
version = "0.1.0"
description = "RAG pipeline for earnings call transcript analysis"
requires-python = ">=3.10,<3.13"
dependencies = []

[tool.uv]
package = false
EOF

# ============================================================
# Create virtual environment
# ============================================================

echo "Creating .venv with uv..."
uv venv --python 3.11 .venv

source "$PROJECT_ROOT/.venv/bin/activate"

echo "Python:"
python --version

# ============================================================
# Install CUDA PyTorch FIRST
# ============================================================

echo "Installing CUDA PyTorch first..."

uv pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

echo "Checking PyTorch after CUDA install..."

python - <<'PY'
import torch, sys, inspect

print("PY:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("torch file:", inspect.getfile(torch))

if torch.version.cuda is None:
    raise RuntimeError("CPU-only torch is installed. CUDA PyTorch installation failed.")
PY

# ============================================================
# Install normal dependencies AFTER CUDA torch
# ============================================================

echo "Installing normal dependencies after CUDA torch..."

uv pip install \
  pandas \
  numpy \
  pyarrow \
  tqdm \
  scikit-learn \
  sentence-transformers \
  chromadb \
  datasets \
  pydantic \
  networkx \
  matplotlib

# ============================================================
# Final verification
# ============================================================

echo ""
echo "Verifying final environment..."

python - <<'PY'
import sys
import inspect
import pandas as pd
import chromadb
import sentence_transformers
import datasets
import sklearn
import pyarrow
import torch

print("PY:", sys.executable)
print("pandas:", pd.__version__)
print("chromadb:", chromadb.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
print("datasets:", datasets.__version__)
print("sklearn:", sklearn.__version__)
print("pyarrow:", pyarrow.__version__)

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("torch file:", inspect.getfile(torch))

if torch.version.cuda is None:
    raise RuntimeError("CPU-only torch is installed after dependency installation.")

print("Environment OK.")
PY

# ============================================================
# Save env.sh
# ============================================================

cat > env.sh <<EOF
#!/usr/bin/env bash

export PROJECT_ROOT="$PROJECT_ROOT"
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export TORCH_HOME="$TORCH_HOME_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"

export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

source "$PROJECT_ROOT/.venv/bin/activate"
EOF

chmod +x env.sh

# ============================================================
# Check data file
# ============================================================

echo ""
echo "Checking data file..."

if [ -f "$DATA_CSV" ]; then
    echo "FOUND: $DATA_CSV"
    ls -lh "$DATA_CSV"
else
    echo "ERROR: data file not found:"
    echo "$DATA_CSV"
    echo ""
    echo "Expected structure:"
    echo "$PROJECT_ROOT/data/combined_transcript_data/combined_transcripts_deduplicated.csv"
    exit 1
fi

echo ""
echo "DONE."
echo "Next time run:"
echo "source env.sh"