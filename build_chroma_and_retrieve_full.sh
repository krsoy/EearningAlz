#!/bin/bash
#SBATCH --job-name=build_rag_full
#SBATCH --output=logs/build_rag_full_%A.out
#SBATCH --error=logs/build_rag_full_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# ============================================================
# Submit from project root:
#
# cd ~/sem2
# sbatch build_chroma_and_retrieve_full.slurm
#
# Purpose:
# From zero:
# 1. Build ChromaDB index
# 2. Generate full evidence JSONL
#
# Output:
# RAG/rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl
# ============================================================

PROJECT_ROOT="$(pwd)"
RAG_DIR="$PROJECT_ROOT/RAG"

INPUT_CSV="$PROJECT_ROOT/data/combined_transcript_data/combined_transcripts_deduplicated.csv"

BUILD_SCRIPT="$RAG_DIR/build_chroma_rag_index.py"
RETRIEVE_SCRIPT="$RAG_DIR/retrieve_chroma_rag_evidence_full_gpu_direct.py"

RAG_OUTPUT_DIR="$RAG_DIR/rag_chroma_output"
CHROMA_DIR="$RAG_OUTPUT_DIR/chroma_db"
CHUNK_INDEX_CSV="$RAG_OUTPUT_DIR/chunk_index.csv"

EXPECTED_JSONL="$RAG_OUTPUT_DIR/rag_evidence_packages_full_gpu_direct.jsonl"
EXPECTED_FLAT_CSV="$RAG_OUTPUT_DIR/rag_evidence_chunks_flat_full_gpu_direct.csv"
EXPECTED_SUMMARY="$RAG_OUTPUT_DIR/retrieval_summary_full_gpu_direct.json"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RAG_DIR/logs"

echo "============================================================"
echo "Build ChromaDB + Full Retrieval job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RAG_DIR=$RAG_DIR"
echo "INPUT_CSV=$INPUT_CSV"
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
# Input / script checks
# ============================================================

echo ""
echo "Input and script checks:"

if [ ! -d "$RAG_DIR" ]; then
    echo "ERROR: RAG directory not found:"
    echo "$RAG_DIR"
    exit 1
fi

if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: input CSV not found:"
    echo "$INPUT_CSV"
    exit 1
fi

if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "ERROR: build script not found:"
    echo "$BUILD_SCRIPT"
    exit 1
fi

if [ ! -f "$RETRIEVE_SCRIPT" ]; then
    echo "ERROR: full retrieval script not found:"
    echo "$RETRIEVE_SCRIPT"
    exit 1
fi

echo "FOUND input CSV:"
ls -lh "$INPUT_CSV"

echo ""
echo "FOUND build script:"
ls -lh "$BUILD_SCRIPT"

echo ""
echo "FOUND retrieval script:"
ls -lh "$RETRIEVE_SCRIPT"

# ============================================================
# Python / CUDA environment check
# ============================================================

echo ""
echo "Python / CUDA environment check:"
cd "$PROJECT_ROOT"

uv run --no-sync python - <<'PY'
import sys
import inspect
import torch

print("PYTHON:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("torch file:", inspect.getfile(torch))

if torch.version.cuda is None:
    raise RuntimeError("BAD: CPU torch detected. Reinstall CUDA torch first.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside this Slurm GPU job.")

print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

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
# Clean old output
# ============================================================

echo ""
echo "============================================================"
echo "Step 0: Clean old partial Chroma output"
echo "Started at: $(date)"
echo "============================================================"

cd "$RAG_DIR"

if [ -d "$RAG_OUTPUT_DIR" ]; then
    echo "Old rag_chroma_output exists:"
    du -sh "$RAG_OUTPUT_DIR" || true

    echo ""
    echo "Removing old rag_chroma_output to rebuild from zero:"
    rm -rf "$RAG_OUTPUT_DIR"
else
    echo "No old rag_chroma_output found."
fi

mkdir -p "$RAG_OUTPUT_DIR"
mkdir -p "$RAG_DIR/logs"

echo "Clean done."

# ============================================================
# Step 1: Build ChromaDB
# ============================================================

echo ""
echo "============================================================"
echo "Step 1: Build ChromaDB RAG index from zero"
echo "Started at: $(date)"
echo "============================================================"

cd "$RAG_DIR"

uv run --no-sync python build_chroma_rag_index.py \
    2>&1 | tee "logs/build_chroma_${SLURM_JOB_ID:-local}.log"

echo ""
echo "Build finished at: $(date)"

echo ""
echo "Output after build:"
ls -lh "$RAG_OUTPUT_DIR" || true

if [ -d "$CHROMA_DIR" ]; then
    echo ""
    echo "Chroma DB size:"
    du -sh "$CHROMA_DIR" || true
else
    echo "ERROR: Chroma DB was not created:"
    echo "$CHROMA_DIR"
    exit 1
fi

if [ -f "$CHUNK_INDEX_CSV" ]; then
    echo ""
    echo "chunk_index.csv:"
    ls -lh "$CHUNK_INDEX_CSV"
else
    echo "ERROR: chunk_index.csv was not created:"
    echo "$CHUNK_INDEX_CSV"
    exit 1
fi

if [ -f "$RAG_OUTPUT_DIR/build_summary.json" ]; then
    echo ""
    echo "Build summary:"
    cat "$RAG_OUTPUT_DIR/build_summary.json"
else
    echo "WARNING: build_summary.json not found"
fi

# ============================================================
# Step 1.5: Validate ChromaDB and chunk_uid
# ============================================================

echo ""
echo "============================================================"
echo "Step 1.5: Validate ChromaDB / chunk_uid consistency"
echo "Started at: $(date)"
echo "============================================================"

cd "$RAG_DIR"

uv run --no-sync python - <<'PY'
import pandas as pd
import chromadb

chunk_path = "rag_chroma_output/chunk_index.csv"
chroma_path = "rag_chroma_output/chroma_db"

chunk_df = pd.read_csv(chunk_path)

print("chunk_index rows:", len(chunk_df))
print("chunk_index columns:", chunk_df.columns.tolist())

if "chunk_uid" not in chunk_df.columns:
    raise RuntimeError("chunk_index.csv does not contain chunk_uid.")

if "quarter" in chunk_df.columns:
    print("\nQuarter distribution in chunk_index:")
    print(chunk_df["quarter"].value_counts(dropna=False).sort_index())

print("\nchunk_uid examples:")
print(chunk_df["chunk_uid"].head(10).astype(str).tolist())

client = chromadb.PersistentClient(path=chroma_path)
col = client.get_collection("earnings_call_chunks")

count = col.count()
print("\nChroma collection count:", count)

if count <= 0:
    raise RuntimeError("Chroma collection is empty.")

sample_ids = chunk_df["chunk_uid"].head(10).astype(str).tolist()

x = col.get(
    ids=sample_ids,
    include=["metadatas"]
)

print("\nRequested sample ids:")
print(sample_ids)

print("\nReturned sample ids:")
print(x["ids"])

print("\nReturned count:", len(x["ids"]))

if len(x["ids"]) == 0:
    raise RuntimeError(
        "chunk_uid does not match ChromaDB ids. "
        "Build script may not have used chunk_uid as ids."
    )

print("\nOK: chunk_uid matches ChromaDB ids.")
PY

# ============================================================
# Step 2: Full GPU direct retrieval
# ============================================================

echo ""
echo "============================================================"
echo "Step 2: Generate full evidence with GPU direct retrieval"
echo "Started at: $(date)"
echo "============================================================"

cd "$RAG_DIR"

uv run --no-sync python retrieve_chroma_rag_evidence_full_gpu_direct.py \
    2>&1 | tee "logs/retrieve_full_gpu_direct_${SLURM_JOB_ID:-local}.log"

echo ""
echo "Full retrieval finished at: $(date)"

# ============================================================
# Step 3: Output checks
# ============================================================

echo ""
echo "============================================================"
echo "Step 3: Output checks"
echo "============================================================"

if [ -f "$EXPECTED_JSONL" ]; then
    echo "FOUND full evidence JSONL:"
    ls -lh "$EXPECTED_JSONL"
else
    echo "ERROR: full evidence JSONL not found:"
    echo "$EXPECTED_JSONL"
    exit 1
fi

if [ -f "$EXPECTED_FLAT_CSV" ]; then
    echo ""
    echo "FOUND flat evidence CSV:"
    ls -lh "$EXPECTED_FLAT_CSV"
else
    echo "WARNING: flat evidence CSV not found:"
    echo "$EXPECTED_FLAT_CSV"
fi

if [ -f "$EXPECTED_SUMMARY" ]; then
    echo ""
    echo "Retrieval summary:"
    cat "$EXPECTED_SUMMARY"
else
    echo "WARNING: retrieval summary not found:"
    echo "$EXPECTED_SUMMARY"
fi

echo ""
echo "Full evidence line count:"
wc -l "$EXPECTED_JSONL"

echo ""
echo "Flat evidence preview:"
if [ -f "$EXPECTED_FLAT_CSV" ]; then
    uv run --no-sync python - <<'PY'
import pandas as pd

path = "rag_chroma_output/rag_evidence_chunks_flat_full_gpu_direct.csv"
df = pd.read_csv(path)

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
fi

echo ""
echo "Disk usage after job:"
du -sh "$RAG_OUTPUT_DIR" || true
df -h "$PROJECT_ROOT" || true

echo ""
echo "GPU status after job:"
nvidia-smi || true

echo ""
echo "============================================================"
echo "Build ChromaDB + Full Retrieval job finished at: $(date)"
echo "============================================================"