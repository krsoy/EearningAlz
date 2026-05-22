#!/bin/bash
#SBATCH --nodelist=ailab-l4-02
#SBATCH --job-name=rag_chroma
#SBATCH --output=logs/rag_chroma_%A.out
#SBATCH --error=logs/rag_chroma_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "============================================================"

# ============================================================
# Environment
# ============================================================

export PROJECT_ROOT="$(pwd)"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"

export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR" rag_chroma_output logs

echo ""
echo "Environment paths:"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"

echo ""
echo "Disk usage:"
df -h "$PROJECT_ROOT" || true

echo ""
echo "GPU status:"
nvidia-smi || true

echo ""
echo "Python environment check:"
uv run --no-sync python - <<'PY'
import sys
print("PYTHON:", sys.executable)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu count:", torch.cuda.device_count())
        print("gpu name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check failed:", repr(e))

import pandas as pd
import chromadb
import sentence_transformers

print("pandas:", pd.__version__)
print("chromadb:", chromadb.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
PY

echo ""
echo "Input data check:"
INPUT_CSV="$PROJECT_ROOT/data/combined_transcript_data/combined_transcripts_deduplicated.csv"

if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: input CSV not found:"
    echo "$INPUT_CSV"
    exit 1
fi

ls -lh "$INPUT_CSV"

echo ""
echo "============================================================"
echo "Step 1: Build Chroma RAG index"
echo "Started at: $(date)"
echo "============================================================"
cd "$PROJECT_ROOT/RAG"
uv run --no-sync python build_chroma_rag_index.py 2>&1 | tee logs/build_chroma_${SLURM_JOB_ID:-local}.log

echo ""
echo "Build index finished at: $(date)"

echo ""
echo "Output after build:"
ls -lh rag_chroma_output || true
ls -lh rag_chroma_output/chroma_db || true

echo ""
echo "Build summary:"
if [ -f rag_chroma_output/build_summary.json ]; then
    cat rag_chroma_output/build_summary.json
else
    echo "WARNING: build_summary.json not found"
fi

echo ""
echo "============================================================"
echo "Step 2: Retrieve RAG evidence"
echo "Started at: $(date)"
echo "============================================================"

uv run --no-sync python retrieve_chroma_rag_evidence.py 2>&1 | tee logs/retrieve_chroma_${SLURM_JOB_ID:-local}.log

echo ""
echo "Retrieval finished at: $(date)"

echo ""
echo "Final output files:"
ls -lh rag_chroma_output || true

echo ""
echo "Retrieval summary:"
if [ -f rag_chroma_output/retrieval_summary.json ]; then
    cat rag_chroma_output/retrieval_summary.json
else
    echo "WARNING: retrieval_summary.json not found"
fi

echo ""
echo "Sample evidence preview:"
if [ -f rag_chroma_output/rag_evidence_chunks_flat.csv ]; then
    uv run --no-sync python - <<'PY'
import pandas as pd

path = "rag_chroma_output/rag_evidence_chunks_flat.csv"
df = pd.read_csv(path)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

cols = [
    "ticker",
    "company",
    "quarter",
    "evidence_type",
    "chunk_id",
    "hybrid_score",
    "keyword_count"
]
cols = [c for c in cols if c in df.columns]
print(df[cols].head(20).to_string(index=False))
PY
else
    echo "WARNING: rag_evidence_chunks_flat.csv not found"
fi

echo ""
echo "Disk usage after job:"
du -sh rag_chroma_output || true
df -h "$PROJECT_ROOT" || true

echo ""
echo "GPU status after job:"
nvidia-smi || true

echo ""
echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"