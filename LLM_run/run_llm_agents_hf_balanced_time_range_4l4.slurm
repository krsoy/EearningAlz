#!/bin/bash
#SBATCH --job-name=llm_hf_balanced
#SBATCH --output=logs/llm_hf_balanced_%A_%a.out
#SBATCH --error=logs/llm_hf_balanced_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# ============================================================
# Balanced LLM extraction from Hugging Face evidence Parquet.
#
# Example:
# cd ~/sem2
# N=$(tail -n +2 llm_tasks_pre2024_2000docs_hf.tsv | wc -l)
# TASK_FILE=~/sem2/llm_tasks_pre2024_2000docs_hf.tsv \
# sbatch --array=0-$((N-1))%2 run_llm_agents_hf_balanced_time_range_4l4.slurm
# ============================================================

PROJECT_ROOT="$(pwd)"
RAG_DIR="$PROJECT_ROOT/RAG"

TASK_FILE="${TASK_FILE:-$PROJECT_ROOT/llm_tasks_hf.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RAG_DIR/logs"

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: TASK_FILE not found: $TASK_FILE"
    exit 1
fi

TASK_ROW="$(awk -F'\t' -v id="$TASK_ID" 'NR > 1 && $1 == id {print $0}' "$TASK_FILE")"
if [ -z "$TASK_ROW" ]; then
    echo "ERROR: no task row found for TASK_ID=$TASK_ID"
    echo "TASK_FILE=$TASK_FILE"
    exit 1
fi

START_QUARTER="$(echo "$TASK_ROW" | awk -F'\t' '{print $2}')"
END_QUARTER="$(echo "$TASK_ROW" | awk -F'\t' '{print $3}')"
TARGET_QUARTERS_VALUE="$(echo "$TASK_ROW" | awk -F'\t' '{print $4}')"
SELECTED_DOC_COUNT="$(echo "$TASK_ROW" | awk -F'\t' '{print $5}')"
MAX_DOCS_PER_TASK="$(echo "$TASK_ROW" | awk -F'\t' '{print $6}')"
NUM_SHARDS_VALUE="$(echo "$TASK_ROW" | awk -F'\t' '{print $7}')"
SHARD_ID_VALUE="$(echo "$TASK_ROW" | awk -F'\t' '{print $8}')"
OUTPUT_TAG="$(echo "$TASK_ROW" | awk -F'\t' '{print $9}')"
AGENT_TASKS_VALUE="$(echo "$TASK_ROW" | awk -F'\t' '{print $10}')"
OUTPUT_BASE_DIR="$(echo "$TASK_ROW" | awk -F'\t' '{print $11}')"

export TARGET_QUARTERS="$TARGET_QUARTERS_VALUE"
export NUM_SHARDS="$NUM_SHARDS_VALUE"
export SHARD_ID="$SHARD_ID_VALUE"
export RUN_NAME="$OUTPUT_TAG"
export AGENT_TASKS="$AGENT_TASKS_VALUE"

export MAX_DOCS=""
export START_OFFSET=0
export LIMIT_DOCS=""

# Hugging Face evidence source
export HF_DATASET="${HF_DATASET:-soysouce/earningALZ_SBERT_evidence}"
export HF_EVIDENCE_FILE="${HF_EVIDENCE_FILE:-rag_evidence_packages_full_gpu_direct.parquet}"
export HF_REVISION="${HF_REVISION:-main}"

# Model settings
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
export LLM_MAX_WORKERS="${LLM_MAX_WORKERS:-6}"
export LLM_REQUEST_TIMEOUT="${LLM_REQUEST_TIMEOUT:-240}"
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-3}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1600}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
export LLM_TOP_P="${LLM_TOP_P:-1.0}"

# Wide port gap for vLLM internal ports.
export PORT_BASE=$((20000 + SLURM_ARRAY_TASK_ID * 100))
export VLLM_PORT=${PORT_BASE}
export MASTER_PORT=$((PORT_BASE + 10))
export VLLM_HOST=127.0.0.1
export VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"

export OUTPUT_DIR="${OUTPUT_BASE_DIR}/${START_QUARTER}_${END_QUARTER}"

# Environment/cache
export PROJECT_ROOT="$PROJECT_ROOT"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

echo "============================================================"
echo "HF Balanced LLM extraction job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: $(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RAG_DIR=$RAG_DIR"
echo "TASK_FILE=$TASK_FILE"
echo "TASK_ID=$TASK_ID"
echo "START_QUARTER=$START_QUARTER"
echo "END_QUARTER=$END_QUARTER"
echo "TARGET_QUARTERS=$TARGET_QUARTERS"
echo "SELECTED_DOC_COUNT=$SELECTED_DOC_COUNT"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "SHARD_ID=$SHARD_ID"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HF_DATASET=$HF_DATASET"
echo "HF_EVIDENCE_FILE=$HF_EVIDENCE_FILE"
echo "VLLM_PORT=$VLLM_PORT"
echo "============================================================"

cd "$PROJECT_ROOT"

# Activate Python.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

PYTHON_BIN="$(command -v python)"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
print("python:", sys.executable)
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "n_gpu:", torch.cuda.device_count())
import vllm
print("vllm:", vllm.__version__)
import pandas, pyarrow, huggingface_hub
print("pandas/pyarrow/hf ok")
PY

VLLM_LOG="$PROJECT_ROOT/logs/vllm_hf_${SLURM_JOB_ID:-nojob}_${SLURM_ARRAY_TASK_ID:-0}.log"

echo "Starting vLLM server..."
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --model "$LLM_MODEL" \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID=$VLLM_PID"
echo "vLLM log=$VLLM_LOG"

cleanup() {
    echo "Cleaning up..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 5
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for vLLM server..."
for i in $(seq 1 80); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server exited early."
        echo "Last server log:"
        tail -n 120 "$VLLM_LOG" || true
        exit 1
    fi

    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi

    echo "Waiting... $i"
    sleep 10
done

if ! curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become ready."
    tail -n 120 "$VLLM_LOG" || true
    exit 1
fi

echo "Running HF evidence LLM extraction..."
"$PYTHON_BIN" "$RAG_DIR/extract_llm_agents_hf_evidence_vllm.py"

echo "Job completed at: $(date)"
