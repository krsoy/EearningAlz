#!/usr/bin/env bash

# ============================================================
# EarningALZ exam-ready workflow
# ============================================================
# Run from the project root or from RAG/.
#
# Recommended on AAU AI-LAB:
#   cd ~/sem2
#   bash RAG/exam_ready_sh.sh
#
# The script has two blocking points:
#   1. Step 5 submits Slurm LLM extraction jobs. Wait until all array jobs finish.
#   2. Step 5b submits optional LLM judge jobs. Wait until judge jobs finish if you need validation results.
#
# After the LLM extraction jobs finish, rerun the post-LLM commands from Step 6 onward,
# or run them manually as documented in README.md.
# ============================================================

set -euo pipefail


# ------------------------------------------------------------
# 0. Common paths / time range
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAG_DIR="$PROJECT_ROOT/RAG"
DATA_DIR="$PROJECT_ROOT/data"
JUDGE_DIR="$PROJECT_ROOT/LLM_result_alz/llm_judge"

COMBINED_CSV="$DATA_DIR/combined_transcript_data/combined_transcripts_deduplicated.csv"
RAG_OUT_DIR="$RAG_DIR/rag_chroma_output"
TASK_TSV="$PROJECT_ROOT/llm_tasks_2024Q1_2026Q2.tsv"

START_QUARTER="${START_QUARTER:-2024Q1}"
END_QUARTER="${END_QUARTER:-2026Q2}"
MAX_DOCS_PER_TASK="${MAX_DOCS_PER_TASK:-160}"
MAX_CONCURRENT_LLM_JOBS="${MAX_CONCURRENT_LLM_JOBS:-2}"

EVIDENCE_SUFFIX="${EVIDENCE_SUFFIX:-full_gpu_direct}"
EVIDENCE_JSONL="$RAG_OUT_DIR/rag_evidence_packages_${EVIDENCE_SUFFIX}.jsonl"

mkdir -p "$RAG_OUT_DIR" "$PROJECT_ROOT/logs"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RAG_DIR=$RAG_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "RAG_OUT_DIR=$RAG_OUT_DIR"
echo "START_QUARTER=$START_QUARTER"
echo "END_QUARTER=$END_QUARTER"


# ------------------------------------------------------------
# 1. Merge two data sources and clean transcripts
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 1: merge and clean transcripts"
echo "============================================================"

cd "$DATA_DIR"
python exam_ready_combined_transcripts_kaggle_motley.py

if [ ! -f "$COMBINED_CSV" ]; then
  echo "ERROR: combined transcript CSV not found: $COMBINED_CSV"
  exit 1
fi


# ------------------------------------------------------------
# 2. Build embedding index and store it inside ChromaDB
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 2: build local ChromaDB index"
echo "============================================================"

cd "$RAG_DIR"
python exam_ready_build_chroma_rag_index.py \
  --input-csv "$COMBINED_CSV"


# ------------------------------------------------------------
# 3. Label/retrieve chunks with SBERT + Chroma
# ------------------------------------------------------------
# This writes:
#   RAG/rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl
#   RAG/rag_chroma_output/rag_evidence_chunks_flat_full_gpu_direct.csv
#   RAG/rag_chroma_output/retrieval_summary_full_gpu_direct.json
#
# The command below uses the hosted HF ChromaDB source. If you want to use
# the local ChromaDB from Step 2, remove --download-hf and set:
#   --chroma-dir "$RAG_OUT_DIR/chroma_db"
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 3: retrieve evidence packages"
echo "============================================================"

python exam_ready_sbert_chunk_label.py \
  --download-hf \
  --repo-id soysouce/earning_chroma \
  --hf-local-dir hf_earning_chroma \
  --chroma-dir hf_earning_chroma/chroma_db \
  --collection earnings_call_chunks \
  --device cuda \
  --out-dir "$RAG_OUT_DIR" \
  --suffix "$EVIDENCE_SUFFIX"

if [ ! -f "$EVIDENCE_JSONL" ]; then
  echo "ERROR: evidence JSONL not found: $EVIDENCE_JSONL"
  exit 1
fi


# ------------------------------------------------------------
# 4. Generate balanced time-range task list for LLM extraction
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 4: create balanced LLM task TSV"
echo "============================================================"

python make_balanced_time_range_tasks.py \
  --input-jsonl "$EVIDENCE_JSONL" \
  --out-tsv "$TASK_TSV" \
  --start-quarter "$START_QUARTER" \
  --end-quarter "$END_QUARTER" \
  --max-docs-per-task "$MAX_DOCS_PER_TASK" \
  --run-prefix "y${START_QUARTER,,}_${END_QUARTER,,}"


# ------------------------------------------------------------
# 5. Submit LLM extraction jobs on AAU AI-LAB
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Step 5: submit LLM extraction Slurm array"
echo "============================================================"

cd "$PROJECT_ROOT"
N=$(tail -n +2 "$TASK_TSV" | wc -l)

if [ "$N" -le 0 ]; then
  echo "ERROR: no tasks found in $TASK_TSV"
  exit 1
fi

TASK_FILE="$TASK_TSV" \
sbatch --array=0-$((N-1))%"$MAX_CONCURRENT_LLM_JOBS" run_llm_agents_balanced_time_range_4l4.slurm

echo ""
echo "Submitted LLM extraction jobs."
echo "Wait for the Slurm array to finish before running Step 6 onward."


# ------------------------------------------------------------
# 5b. Optional: submit LLM-as-judge validation jobs
# ------------------------------------------------------------
# Before this step, create judge cases if needed:
#   cd LLM_result_alz/llm_judge
#   python sample_cases.py --n-per-group 10 --out judge_cases.jsonl
# ------------------------------------------------------------

if [ "${RUN_LLM_JUDGE:-0}" = "1" ]; then
  echo ""
  echo "============================================================"
  echo "Step 5b: submit optional LLM judge Slurm jobs"
  echo "============================================================"
  cd "$PROJECT_ROOT"
  sbatch "$JUDGE_DIR/run_judge.slurm"
fi


# ============================================================
# POST-LLM ANALYSIS
# ============================================================
# Run these commands only after all LLM array jobs have finished.
# They are kept below as copy-ready commands and are also documented in README.md.
# ============================================================

cat <<EOF

Next steps after the LLM extraction jobs finish:

cd "$RAG_DIR"

ls "$RAG_OUT_DIR"/llm_csv_outputs_balanced_time_range/*/outlook_*.csv
ls "$RAG_OUT_DIR"/llm_csv_outputs_balanced_time_range/*/relationships_*.csv

python run_two_part_network_prediction_analysis.py \\
  --rag-output-dir "$RAG_OUT_DIR" \\
  --out-dir "$RAG_OUT_DIR/two_part_network_prediction_analysis" \\
  --start-quarter "$START_QUARTER" \\
  --end-quarter "$END_QUARTER" \\
  --min-exposed-for-plot 5

python cluster_method_comparison.py \\
  --rag-output-dir "$RAG_OUT_DIR" \\
  --two-part-dir "$RAG_OUT_DIR/two_part_network_prediction_analysis" \\
  --combined-transcripts "$COMBINED_CSV" \\
  --evidence-jsonl "$EVIDENCE_JSONL" \\
  --out-dir "$RAG_OUT_DIR/cluster_method_comparison_v4" \\
  --start-quarter "$START_QUARTER" \\
  --end-quarter "$END_QUARTER" \\
  --min-k 5 \\
  --max-k 100 \\
  --min-cluster-size 10 \\
  --min-exposed 10 \\
  --cooccur-min-weight 2

EOF
