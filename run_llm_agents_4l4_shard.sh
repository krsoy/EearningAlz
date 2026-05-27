#!/bin/bash
#SBATCH --job-name=llm_q2q3
#SBATCH --output=logs/llm_q2q3_%A_%a.out
#SBATCH --error=logs/llm_q2q3_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

# ============================================================
# Submit from project root:
#
# cd ~/sem2
# sbatch --array=0-1 run_llm_agents_4l4_q2q3_shard.slurm
#
# NUM_SHARDS is set in this bash script.
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

export TARGET_QUARTERS="2025Q2,2025Q3"

# Options:
# concepts
# relationships
# outlook
# concepts,relationships,outlook
export AGENT_TASKS="concepts,relationships,outlook"

# Testing controls.
# Empty means no limit.
export MAX_DOCS=""
export START_OFFSET=0
export LIMIT_DOCS=""

# ============================================================
# Model settings
# ============================================================

export LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
export VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"

export LLM_MAX_WORKERS=6
export LLM_REQUEST_TIMEOUT=240
export LLM_MAX_RETRIES=3
export LLM_MAX_TOKENS=1600
export LLM_TEMPERATURE=0.0
export LLM_TOP_P=1.0

# ============================================================
# Input and output
# ============================================================

export INPUT_JSONL="rag_chroma_output/rag_evidence_packages_2025Q2_Q3_gpu_direct.jsonl"
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
echo "GPU status:"
nvidia-smi || true

echo ""
echo "Python environment check:"
cd "$PROJECT_ROOT"

uv run --no-sync python - <<'PY'
import sys
import torch

print("PYTHON:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

if torch.cuda.device_count() < 4:
    raise RuntimeError("This job expects 4 GPUs.")
PY

echo ""
echo "Install/check runtime packages:"
uv pip install requests vllm || true

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
uv run --no-sync python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("rag_chroma_output/rag_evidence_packages_2025Q2_Q3_gpu_direct.jsonl")

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
  --trust-remote-code \
  > "logs/vllm_server_${RUN_NAME}_${SLURM_JOB_ID:-local}.log" 2>&1 &

VLLM_PID=$!

echo "vLLM PID: $VLLM_PID"

echo ""
echo "Waiting for vLLM server..."

for i in $(seq 1 180); do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi

    if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
        echo "ERROR: vLLM server exited early."
        echo "Last server log:"
        tail -n 120 "logs/vllm_server_${RUN_NAME}_${SLURM_JOB_ID:-local}.log" || true
        exit 1
    fi

    echo "Waiting... $i"
    sleep 10
done

if ! curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become ready."
    tail -n 120 "logs/vllm_server_${RUN_NAME}_${SLURM_JOB_ID:-local}.log" || true
    exit 1
fi

echo ""
echo "Available models:"
curl -s http://127.0.0.1:8000/v1/models || true

# ============================================================
# Run extraction
# ============================================================

echo ""
echo "============================================================"
echo "Running LLM agent extraction for 2025Q2/Q3"
echo "Started at: $(date)"
echo "============================================================"

cd "$RAG_DIR"

uv run --no-sync python extract_llm_agents_csv_vllm.py \
  2>&1 | tee "logs/llm_agents_${RUN_NAME}_${SLURM_JOB_ID:-local}.log"

echo ""
echo "Extraction finished at: $(date)"

# ============================================================
# Stop server
# ============================================================

echo ""
echo "Stopping vLLM server..."
kill "$VLLM_PID" || true
sleep 5

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