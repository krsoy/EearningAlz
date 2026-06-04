#!/bin/bash
#SBATCH --job-name=llm_judge
#SBATCH --output=logs/llm_judge_%A_%a.out
#SBATCH --error=logs/llm_judge_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-1

# ============================================================
# LLM-as-Judge SLURM array job  (2 × 4 GPU = 8 GPU total)
# Each array task processes JUDGE_CASES_PER_SHARD cases
# from a different offset.
#
# Submit:
#   cd ~/sem2
#   sbatch LLM_result_alz/llm_judge/run_judge.slurm
#
# Or with specific case filter:
#   JUDGE_FILTER_SOURCE=AAPL JUDGE_FILTER_TARGET=CARR \
#   sbatch LLM_result_alz/llm_judge/run_judge.slurm
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(pwd)"
JUDGE_DIR="$PROJECT_ROOT/LLM_result_alz/llm_judge"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$JUDGE_DIR"

# ============================================================
# Model settings
# ============================================================

export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

# ============================================================
# Judge settings
# ============================================================

export JUDGE_CASES="$JUDGE_DIR/judge_cases.jsonl"
export JUDGE_N_PER_GROUP="${JUDGE_N_PER_GROUP:-10}"
export JUDGE_CASES_PER_SHARD="${JUDGE_CASES_PER_SHARD:-500}"

# ============================================================
# Array task / shard settings
# ============================================================

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
JOB_ID="${SLURM_JOB_ID:-local}"

SHARD_OFFSET=$(( TASK_ID * JUDGE_CASES_PER_SHARD ))
export JUDGE_OUT="$JUDGE_DIR/judge_results_shard${TASK_ID}.jsonl"

# ============================================================
# vLLM ports
# ============================================================
# IMPORTANT:
# - VLLM_API_PORT is the HTTP API port used by CrewAI/OpenAI client.
# - VLLM_PORT is reserved for vLLM internal distributed communication.
# Do not use the same variable for both purposes.
# ============================================================

export VLLM_API_PORT=$(( 21000 + TASK_ID * 100 ))
export VLLM_PORT=$(( 31000 + TASK_ID * 100 ))
export VLLM_URL="http://127.0.0.1:${VLLM_API_PORT}/v1"

# ============================================================
# Optional filters
# ============================================================

export JUDGE_FILTER_SOURCE="${JUDGE_FILTER_SOURCE:-}"
export JUDGE_FILTER_TARGET="${JUDGE_FILTER_TARGET:-}"
export JUDGE_FILTER_SIGNAL="${JUDGE_FILTER_SIGNAL:-}"
export JUDGE_FILTER_RELATION="${JUDGE_FILTER_RELATION:-}"

# ============================================================
# Environment
# ============================================================

# Shared model/package caches. These are usually safe to share after the model
# and packages are already downloaded.
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"

# Per-array-task runtime directories. These prevent CrewAI/vLLM/Torch runtime
# files from being written concurrently by shard 0 and shard 1.
export TASK_RUN_DIR="$PROJECT_ROOT/.runs/llm_judge/${ARRAY_JOB_ID}/task_${TASK_ID}"
export CREWAI_STORAGE_DIR="$TASK_RUN_DIR/crewai_storage"
export XDG_CACHE_HOME="$TASK_RUN_DIR/xdg_cache"
export TORCHINDUCTOR_CACHE_DIR="$TASK_RUN_DIR/torchinductor"
export VLLM_CACHE_ROOT="$TASK_RUN_DIR/vllm_cache"
export TMPDIR="/tmp/${USER:-user}_vj_${ARRAY_JOB_ID}_${TASK_ID}"

export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$UV_CACHE_DIR" \
  "$TASK_RUN_DIR" \
  "$CREWAI_STORAGE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" \
  "$TMPDIR"

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export VLLM_LOG="$PROJECT_ROOT/logs/vllm_judge_${ARRAY_JOB_ID}_${TASK_ID}.log"
export WORKFLOW_LOG="$PROJECT_ROOT/logs/judge_workflow_${ARRAY_JOB_ID}_${TASK_ID}.log"

# ============================================================
# Diagnostics
# ============================================================

echo "============================================================"
echo "LLM Judge Job started: $(date)"
echo "Array Job ID       : $ARRAY_JOB_ID"
echo "Job ID             : $JOB_ID"
echo "Task ID            : $TASK_ID"
echo "Node               : $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "MODEL              : $LLM_MODEL"
echo "API_PORT           : $VLLM_API_PORT"
echo "VLLM_INTERNAL_PORT : $VLLM_PORT"
echo "VLLM_URL           : $VLLM_URL"
echo "FILTER             : source=$JUDGE_FILTER_SOURCE target=$JUDGE_FILTER_TARGET signal=$JUDGE_FILTER_SIGNAL relation=$JUDGE_FILTER_RELATION"
echo "OFFSET             : $SHARD_OFFSET"
echo "MAX CASES          : $JUDGE_CASES_PER_SHARD"
echo "JUDGE_OUT          : $JUDGE_OUT"
echo "TASK_RUN_DIR       : $TASK_RUN_DIR"
echo "CREWAI_STORAGE_DIR : $CREWAI_STORAGE_DIR"
echo "VLLM_LOG           : $VLLM_LOG"
echo "WORKFLOW_LOG       : $WORKFLOW_LOG"
echo "============================================================"

nvidia-smi || true

# ============================================================
# Step 1: Verify judge cases
# ============================================================

echo ""
echo "Step 1: Verifying judge_cases.jsonl exists..."
if [ ! -f "$JUDGE_CASES" ]; then
    echo "ERROR: $JUDGE_CASES not found."
    echo "Run sample_cases.py first:"
    echo "  python $JUDGE_DIR/sample_cases.py --n-per-group 10 --out $JUDGE_CASES"
    exit 1
fi

echo "Cases file: $(wc -l < "$JUDGE_CASES") lines  (shard $TASK_ID will process offset=$SHARD_OFFSET max=$JUDGE_CASES_PER_SHARD)"

# ============================================================
# Step 2: Start vLLM server
# ============================================================

echo ""
echo "Step 2: Starting vLLM server on API port $VLLM_API_PORT..."

env -u VLLM_URL python -m vllm.entrypoints.openai.api_server \
  --model "$LLM_MODEL" \
  --host 127.0.0.1 \
  --port "$VLLM_API_PORT" \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --disable-custom-all-reduce \
  --trust-remote-code \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!

cleanup() {
    echo "Cleanup: stopping vLLM PID=${VLLM_PID:-NA}"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    sleep 3
    kill -9 "$VLLM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "vLLM PID: $VLLM_PID"

for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${VLLM_API_PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM server ready."
        break
    fi

    if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
        echo "ERROR: vLLM process exited before becoming ready."
        echo "Last 120 lines of vLLM log:"
        tail -n 120 "$VLLM_LOG" || true
        exit 1
    fi

    [ "$i" -eq 120 ] && {
        echo "ERROR: vLLM did not start."
        echo "Last 120 lines of vLLM log:"
        tail -n 120 "$VLLM_LOG" || true
        exit 1
    }

    echo "Waiting for vLLM... $i"
    sleep 10
done

# ============================================================
# Step 3: Run CrewAI judge workflow
# ============================================================

echo ""
echo "Step 3: Running CrewAI LLM judge workflow..."

python "$JUDGE_DIR/crewai_judge_workflow.py" \
  --cases "$JUDGE_CASES" \
  --out   "$JUDGE_OUT" \
  --vllm-url "$VLLM_URL" \
  --model "$LLM_MODEL" \
  --offset "$SHARD_OFFSET" \
  --max-cases "$JUDGE_CASES_PER_SHARD" \
  --resume \
  --retry-errors \
  --max-retries 2 \
  2>&1 | tee "$WORKFLOW_LOG"

if [ -f "$JUDGE_OUT" ]; then
    echo ""
    echo "Judge results: $(wc -l < "$JUDGE_OUT") lines in $JUDGE_OUT"
else
    echo ""
    echo "WARNING: Judge output file was not created: $JUDGE_OUT"
fi

# ============================================================
# Step 4: Stats summary (non-interactive)
# ============================================================

echo ""
echo "Step 4: Quick stats on judge results (no HITL in cloud)..."

python - <<'PY'
import json
import sys
from pathlib import Path
from collections import Counter

judge_dir = Path("LLM_result_alz/llm_judge")
files = sorted(judge_dir.glob("judge_results_shard*.jsonl"))
if not files:
    files = [judge_dir / "judge_results.jsonl"]

records = []
bad_lines = 0
for p in files:
    if not p.exists():
        continue
    for line_no, line in enumerate(p.read_text(errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception as e:
            bad_lines += 1
            print(f"WARNING: failed to parse {p}:{line_no}: {e}")

if not records:
    print("No results found.")
    sys.exit(0)

verdicts = Counter(r.get("verdict", "UNKNOWN") for r in records)
print(f"Total cases: {len(records)}")
if bad_lines:
    print(f"Bad JSON lines skipped: {bad_lines}")

for v, n in sorted(verdicts.items()):
    print(f"  {v}: {n}")

by_signal = {}
for r in records:
    sig = r.get("signal", "?")
    by_signal.setdefault(sig, []).append(r.get("verdict", "UNKNOWN"))

print("\nBy signal:")
for sig, vs in sorted(by_signal.items()):
    vc = Counter(vs)
    print(f"  {sig}: {dict(vc)}")
PY

echo ""
echo "============================================================"
echo "LLM Judge Job finished: $(date)"
echo "============================================================"
echo ""
echo "Next step (run locally for HITL, merges both shards):"
echo "  python LLM_result_alz/llm_judge/hitl_validator.py \\"
echo "    --results LLM_result_alz/llm_judge/judge_results_shard0.jsonl \\"
echo "              LLM_result_alz/llm_judge/judge_results_shard1.jsonl \\"
echo "    --out LLM_result_alz/llm_judge/hitl_validated.jsonl"