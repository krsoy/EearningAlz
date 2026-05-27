#!/bin/bash
#SBATCH --job-name=llm_q2q3
#SBATCH --output=logs/llm_q2q3_%A_%a.out
#SBATCH --error=logs/llm_q2q3_%A_%a.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# ============================================================
# Submit from project root:
#
# cd ~/sem2
# sbatch --array=0-1 run_llm_agents_4l4_q2q3_shard.slurm
#
# NUM_SHARDS is set here.
# SHARD_ID comes from SLURM_ARRAY_TASK_ID.
# ============================================================

PROJECT_ROOT="$(pwd)"
RAG_DIR="$PROJECT_ROOT/RAG"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RAG_DIR/logs"

# ============================================================
# Shard settings
# ============================================================

export NUM_SHARDS=6
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Only process these quarters. Python script will filter from full evidence.
export TARGET_QUARTERS="2025Q2,2025Q3"

# Agent tasks:
# concepts
# relationships
# outlook
# concepts,relationships,outlook
export AGENT_TASKS="concepts,relationships,outlook"

# Testing controls. Empty means no limit.
export MAX_DOCS=""
export START_OFFSET=0
export LIMIT_DOCS=""

# ============================================================
# Model settings
# ============================================================

export LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"

# Do NOT export VLLM_URL before starting the vLLM server.
# vLLM 0.21 prints warnings for unknown VLLM_* env vars.
# We export VLLM_URL only immediately before running extraction.
VLLM_API_URL="http://127.0.0.1:8000/v1/chat/completions"
VLLM_MODEL_URL="http://127.0.0.1:8000/v1/models"

export LLM_MAX_WORKERS=6
export LLM_REQUEST_TIMEOUT=240
export LLM_MAX_RETRIES=3
export LLM_MAX_TOKENS=1600
export LLM_TEMPERATURE=0.0
export LLM_TOP_P=1.0

# ============================================================
# Input and output
# ============================================================

# Use full evidence. Python filters TARGET_QUARTERS internally.
export INPUT_JSONL="rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl"
export OUTPUT_DIR="rag_chroma_output/llm_csv_outputs_2025Q2_Q3"

export RUN_NAME="q2q3_shard$(printf "%03d" "$SHARD_ID")_of$(printf "%03d" "$NUM_SHARDS")"

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

# ============================================================
# Activate project virtual environment
# This is critical: vLLM/FlashInfer workers call external executable `ninja`.
# Activating venv ensures .venv/bin is inherited by all child processes.
# ============================================================

cd "$PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "ERROR: venv activate script not found:"
    echo "$PROJECT_ROOT/.venv/bin/activate"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
hash -r

# If ninja is missing, install only ninja. This should not touch torch.
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja not found in PATH after activating venv. Installing ninja into .venv..."
    uv pip install ninja
    hash -r
fi

if ! command -v ninja >/dev/null 2>&1; then
    echo "ERROR: ninja executable still not found after installation."
    echo "PATH=$PATH"
    exit 1
fi

echo "============================================================"
echo "LLM Agent extraction Q2/Q3 job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RAG_DIR=$RAG_DIR"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "SHARD_ID=$SHARD_ID"
echo "RUN_NAME=$RUN_NAME"
echo "TARGET_QUARTERS=$TARGET_QUARTERS"
echo "AGENT_TASKS=$AGENT_TASKS"
echo "LLM_MODEL=$LLM_MODEL"
echo "LLM_MAX_WORKERS=$LLM_MAX_WORKERS"
echo "INPUT_JSONL=$INPUT_JSONL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "============================================================"

echo ""
echo "Activated venv:"
echo "python=$(which python)"
python --version
echo "ninja=$(which ninja)"
ninja --version

echo ""
echo "PATH=$PATH"

# ============================================================
# Cleanup
# ============================================================

VLLM_PID=""
cleanup() {
    if [ -n "${VLLM_PID:-}" ]; then
        if kill -0 "$VLLM_PID" >/dev/null 2>&1; then
            echo "Stopping vLLM server PID=$VLLM_PID ..."
            kill "$VLLM_PID" || true
            sleep 5
        fi
    fi
}
trap cleanup EXIT

# ============================================================
# Basic checks
# ============================================================

echo ""
echo "GPU status:"
nvidia-smi || true

echo ""
echo "Runtime environment check:"
cd "$PROJECT_ROOT"

python - <<'PY'
import sys
import inspect
import shutil
import torch

print("PYTHON:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("torch file:", inspect.getfile(torch))
print("gpu count:", torch.cuda.device_count())
print("ninja:", shutil.which("ninja"))

if shutil.which("ninja") is None:
    raise RuntimeError("ninja executable is not visible to Python subprocesses.")

if torch.version.cuda is None:
    raise RuntimeError("BAD: CPU torch detected. Recreate environment with CUDA torch first.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside this Slurm job.")

if torch.cuda.device_count() < 4:
    raise RuntimeError("This job expects 4 GPUs.")

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

import requests
import vllm
import chromadb
import sentence_transformers

print("requests:", requests.__version__)
print("vllm:", vllm.__version__)
print("chromadb:", chromadb.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
PY

# ============================================================
# Input check
# ============================================================

echo ""
echo "Input check:"
cd "$RAG_DIR"

if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: input JSONL not found:"
    echo "$RAG_DIR/$INPUT_JSONL"
    exit 1
fi

if [ ! -f "extract_llm_agents_csv_vllm.py" ]; then
    echo "ERROR: extraction script not found:"
    echo "$RAG_DIR/extract_llm_agents_csv_vllm.py"
    exit 1
fi

ls -lh "$INPUT_JSONL"
ls -lh extract_llm_agents_csv_vllm.py

echo ""
echo "Input line count:"
wc -l "$INPUT_JSONL"

echo ""
echo "Input quarter quick check:"
python - <<PY
import json
from collections import Counter
from pathlib import Path

path = Path("$INPUT_JSONL")

print("Checking input:", path)

if not path.exists():
    raise FileNotFoundError(f"Input JSONL not found: {path}")

counter = Counter()
n = 0

with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        counter[str(obj.get("quarter", "")).strip()] += 1
        n += 1

print("rows:", n)
print("quarter distribution:")
for k, v in sorted(counter.items()):
    print(k, v)

target = {"2025Q2", "2025Q3"}
target_rows = sum(v for k, v in counter.items() if k in target)
print("target 2025Q2/Q3 rows:", target_rows)

if target_rows == 0:
    raise RuntimeError("No 2025Q2/Q3 rows found in INPUT_JSONL.")
PY

# ============================================================
# Start vLLM server
# ============================================================

echo ""
echo "============================================================"
echo "Starting vLLM server"
echo "Started at: $(date)"
echo "============================================================"

cd "$PROJECT_ROOT"

python -m vllm.entrypoints.openai.api_server \
  --model "$LLM_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --disable-custom-all-reduce \
  --trust-remote-code \
  > "logs/vllm_server_${RUN_NAME}_${SLURM_JOB_ID:-local}.log" 2>&1 &

VLLM_PID=$!

echo "vLLM PID: $VLLM_PID"

echo ""
echo "Waiting for vLLM server..."

for i in $(seq 1 180); do
    if curl -s "$VLLM_MODEL_URL" >/dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi

    if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
        echo "ERROR: vLLM server exited early."
        echo "Last server log:"
        tail -n 180 "logs/vllm_server_${RUN_NAME}_${SLURM_JOB_ID:-local}.log" || true
        exit 1
    fi

    echo "Waiting... $i"

    if (( i % 10 == 0 )); then
        echo ""
        echo "---- vLLM server log tail at wait $i ----"
        tail -n 80 "logs/vllm_server_${RUN_NAME}_${SLURM_JOB_ID:-local}.log" || true
        echo "-----------------------------------------"
        echo ""
    fi

    sleep 10
done

if ! curl -s "$VLLM_MODEL_URL" >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become ready."
    tail -n 180 "logs/vllm_server_${RUN_NAME}_${SLURM_JOB_ID:-local}.log" || true
    exit 1
fi

echo ""
echo "Available models:"
curl -s "$VLLM_MODEL_URL" || true

# ============================================================
# Run extraction
# ============================================================

echo ""
echo "============================================================"
echo "Running LLM agent extraction for 2025Q2/Q3"
echo "Started at: $(date)"
echo "============================================================"

cd "$RAG_DIR"

# Export this only after vLLM server has started, so vLLM itself does not see unknown VLLM_* env vars.
export VLLM_URL="$VLLM_API_URL"

python extract_llm_agents_csv_vllm.py \
  2>&1 | tee "logs/llm_agents_${RUN_NAME}_${SLURM_JOB_ID:-local}.log"

echo ""
echo "Extraction finished at: $(date)"

# ============================================================
# Stop server
# ============================================================

echo ""
echo "Stopping vLLM server..."
cleanup
VLLM_PID=""

# ============================================================
# Output checks
# ============================================================

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR" || true

echo ""
echo "Progress:"
if [ -f "$OUTPUT_DIR/progress_${RUN_NAME}.json" ]; then
    cat "$OUTPUT_DIR/progress_${RUN_NAME}.json"
else
    echo "WARNING: progress file not found"
fi

echo ""
echo "CSV row counts:"
for f in "$OUTPUT_DIR"/*"${RUN_NAME}"*.csv; do
    if [ -f "$f" ]; then
        echo "$f"
        wc -l "$f"
    fi
done

echo ""
echo "Failed preview:"
if [ -f "$OUTPUT_DIR/failed_${RUN_NAME}.csv" ]; then
    head -n 20 "$OUTPUT_DIR/failed_${RUN_NAME}.csv" || true
fi

echo ""
echo "GPU status after job:"
nvidia-smi || true

echo ""
echo "============================================================"
echo "LLM Agent extraction Q2/Q3 job finished at: $(date)"
echo "============================================================"
