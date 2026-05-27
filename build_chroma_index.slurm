#!/bin/bash
#SBATCH --nodelist=ailab-l4-02
#SBATCH --job-name=retrieve_full_gpu
#SBATCH --output=logs/retrieve_full_gpu_%A.out
#SBATCH --error=logs/retrieve_full_gpu_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# ============================================================
# Project root
# Submit this script from:
# ~/sem2
# ============================================================

PROJECT_ROOT="$(pwd)"
RAG_DIR="$PROJECT_ROOT/RAG"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RAG_DIR/logs"

echo "============================================================"
echo "Full GPU direct retrieval job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RAG_DIR=$RAG_DIR"
echo "============================================================"

# ============================================================
# Environment
# ============================================================

export PROJECT_ROOT="$PROJECT_ROOT"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"

export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR"

echo ""
echo "Environment paths:"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"

echo ""
echo "Disk usage before job:"
df -h "$PROJECT_ROOT" || true

echo ""
echo "GPU status before job:"
nvidia-smi || true

# ============================================================
# Environment check
# ============================================================

echo ""
echo "Python environment check:"
cd "$PROJECT_ROOT"

uv run --no-sync python - <<'PY'
import sys
import inspect

print("PYTHON:", sys.executable)

import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("torch file:", inspect.getfile(torch))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside this Slurm GPU job.")

print("gpu name:", torch.cuda.get_device_name(0))

import pandas as pd
import numpy as np
import chromadb
import sentence_transformers

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("chromadb:", chromadb.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
PY

# ============================================================
# Input checks
# ============================================================

echo ""
echo "Input checks:"

if [ ! -d "$RAG_DIR" ]; then
    echo "ERROR: RAG directory not found:"
    echo "$RAG_DIR"
    exit 1
fi

cd "$RAG_DIR"

if [ ! -f "retrieve_chroma_rag_evidence_full_gpu_direct.py" ]; then
    echo "ERROR: retrieval script not found:"
    echo "$RAG_DIR/retrieve_chroma_rag_evidence_full_gpu_direct.py"
    exit 1
fi

if [ ! -d "rag_chroma_output/chroma_db" ]; then
    echo "ERROR: Chroma DB not found:"
    echo "$RAG_DIR/rag_chroma_output/chroma_db"
    exit 1
fi

if [ ! -f "rag_chroma_output/chunk_index.csv" ]; then
    echo "ERROR: chunk_index.csv not found:"
    echo "$RAG_DIR/rag_chroma_output/chunk_index.csv"
    exit 1
fi

echo "FOUND retrieval script:"
ls -lh retrieve_chroma_rag_evidence_full_gpu_direct.py

echo ""
echo "FOUND Chroma DB:"
du -sh rag_chroma_output/chroma_db || true

echo ""
echo "FOUND chunk index:"
ls -lh rag_chroma_output/chunk_index.csv




echo ""
echo "Chroma / chunk_index quick check:"
uv run --no-sync python - <<'PY'
import pandas as pd
import chromadb

chunk_df = pd.read_csv("rag_chroma_output/chunk_index.csv")
print("chunk_index rows:", len(chunk_df))
print("chunk_index columns:", chunk_df.columns.tolist())

if "chunk_uid" not in chunk_df.columns:
    raise RuntimeError("chunk_index.csv does not contain chunk_uid.")

print("chunk_uid examples:")
print(chunk_df["chunk_uid"].head(5).tolist())

client = chromadb.PersistentClient(path="rag_chroma_output/chroma_db")
col = client.get_collection("earnings_call_chunks")
print("collection count:", col.count())

sample_ids = chunk_df["chunk_uid"].head(5).astype(str).tolist()
x = col.get(ids=sample_ids, include=["metadatas"])

print("requested sample ids:", sample_ids)
print("returned sample ids:", x["ids"])

if len(x["ids"]) == 0:
    raise RuntimeError("chunk_uid does not match ChromaDB ids.")
else:
    print("OK: chunk_uid matches ChromaDB ids.")
PY

# ============================================================
# Run retrieval
# ============================================================

echo ""
echo "============================================================"
echo "Step: Full GPU direct similarity retrieval"
echo "Started at: $(date)"
echo "============================================================"

uv run --no-sync python retrieve_chroma_rag_evidence_full_gpu_direct.py \
    2>&1 | tee "logs/retrieve_full_gpu_direct_${SLURM_JOB_ID:-local}.log"

echo ""
echo "Retrieval finished at: $(date)"

# ============================================================
# Output checks
# ============================================================

echo ""
echo "Output files:"
ls -lh rag_chroma_output | grep "full_gpu_direct" || true

echo ""
echo "Retrieval summary:"
if [ -f "rag_chroma_output/retrieval_summary_full_gpu_direct.json" ]; then
    cat rag_chroma_output/retrieval_summary_full_gpu_direct.json
else
    echo "WARNING: retrieval_summary_full_gpu_direct.json not found"
fi

echo ""
echo "Sample evidence preview:"
if [ -f "rag_chroma_output/rag_evidence_chunks_flat_full_gpu_direct.csv" ]; then
    uv run --no-sync python - <<'PY'
import pandas as pd

path = "rag_chroma_output/rag_evidence_chunks_flat_full_gpu_direct.csv"
df = pd.read_csv(path)

print("File:", path)
print("Rows:", len(df))
print("Documents:", df["doc_id"].nunique() if "doc_id" in df.columns else "NA")
print("Columns:", df.columns.tolist())

if "quarter" in df.columns:
    print("\nQuarter distribution:")
    print(df["quarter"].value_counts(dropna=False).sort_index())

if "evidence_type" in df.columns:
    print("\nEvidence type distribution:")
    print(df["evidence_type"].value_counts(dropna=False))

cols = [
    "ticker",
    "company",
    "quarter",
    "evidence_type",
    "chunk_id",
    "hybrid_score",
    "semantic_similarity",
    "keyword_count"
]
cols = [c for c in cols if c in df.columns]

print("\nPreview:")
print(df[cols].head(30).to_string(index=False))
PY
else
    echo "WARNING: rag_evidence_chunks_flat_full_gpu_direct.csv not found"
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
echo "Full GPU direct retrieval job finished at: $(date)"
echo "============================================================"