# Earnings Call RAG–LLM–Network Analysis: Full Python and Command Workflow

This document summarizes the full workflow **including both the Python scripts and the exact command-line execution flow**. It covers data preparation, RAG index construction, evidence retrieval, LLM multi-agent extraction on Slurm/vLLM, task sharding, network analysis, rolling contagion analysis, two-part prediction analysis, and Overleaf outputs.

---

## 0. Project Structure

Recommended working directory on the cluster:

```bash
cd ~/sem2
```

Main project layout:

```text
~/sem2/
├── data/
│   └── combined_transcript_data/
│       └── combined_transcripts_deduplicated.csv
├── RAG/
│   ├── build_chroma_rag_index.py
│   ├── retrieve_chroma_rag_evidence_full_gpu_direct.py
│   ├── extract_llm_agents_csv_vllm.py
│   ├── make_balanced_time_range_tasks.py
│   ├── run_network_contagion_master_analysis.py
│   ├── run_rolling_full_range_contagion.py
│   ├── run_two_part_network_prediction_analysis.py
│   ├── analyze_signal_contagion.py
│   ├── analyze_signal_falsification.py
│   ├── visualize_information_flow_network_v2.py
│   └── rag_chroma_output/
├── run_llm_agents_balanced_time_range_4l4.slurm
└── logs/
```

On Windows local machine, the equivalent working directory is usually:

```bash
cd E:\Projects\EearningAlz\RAG
```

---

# 1. Environment and Basic Checks

## 1.1 Activate the Python Environment

On the AAU cluster:

```bash
cd ~/sem2
source .venv/bin/activate
```

Check Python:

```bash
which python
python --version
```

Check CUDA PyTorch:

```bash
python - <<'PY'
import torch
import inspect

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("torch file:", inspect.getfile(torch))

if torch.version.cuda is None:
    raise RuntimeError("CPU torch detected.")
PY
```

Check important packages:

```bash
python - <<'PY'
import pandas as pd
import chromadb
import sentence_transformers
import vllm
import requests

print("pandas:", pd.__version__)
print("chromadb:", chromadb.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
print("vllm:", vllm.__version__)
print("requests:", requests.__version__)
PY
```

If `ninja` is missing:

```bash
uv pip install ninja
```

---

# 2. Build the RAG ChromaDB Index

Main script:

```text
RAG/build_chroma_rag_index.py
```

Input file:

```text
data/combined_transcript_data/combined_transcripts_deduplicated.csv
```

Output directory:

```text
RAG/rag_chroma_output/
```

## 2.1 Run Locally / Interactively

```bash
cd ~/sem2/RAG

python build_chroma_rag_index.py
```

## 2.2 Expected Outputs

```text
rag_chroma_output/
├── chroma_db/
├── chunk_index.csv
├── chunk_index.parquet
└── build_summary.json
```

## 2.3 Check Outputs

```bash
cd ~/sem2/RAG

ls -lh rag_chroma_output
ls -lh rag_chroma_output/chroma_db
cat rag_chroma_output/build_summary.json
```

Expected completed scale:

```text
Transcript-level documents: 25,795
Total chunks: 1,151,208
Embedding dimension: 384
```

---

# 3. Retrieve RAG Evidence with Full GPU Direct Similarity

Main script:

```text
RAG/retrieve_chroma_rag_evidence_full_gpu_direct.py
```

This script reads ChromaDB embeddings and creates transcript-level evidence packages for LLM extraction.

## 3.1 Run Retrieval

```bash
cd ~/sem2/RAG

python retrieve_chroma_rag_evidence_full_gpu_direct.py
```

## 3.2 Expected Outputs

```text
rag_chroma_output/
├── rag_evidence_packages_full_gpu_direct.jsonl
├── rag_evidence_chunks_flat_full_gpu_direct.csv
├── retrieval_summary_full_gpu_direct.json
└── sample_agent_input_full_gpu_direct.json
```

## 3.3 Check Retrieval Outputs

```bash
cd ~/sem2/RAG

ls -lh rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl
ls -lh rag_chroma_output/rag_evidence_chunks_flat_full_gpu_direct.csv
cat rag_chroma_output/retrieval_summary_full_gpu_direct.json
```

Check quarter distribution:

```bash
cd ~/sem2/RAG

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl")
counter = Counter()
n = 0

with path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        counter[str(obj.get("quarter", "")).strip()] += 1
        n += 1

print("rows:", n)
for q, c in sorted(counter.items()):
    print(q, c)
PY
```

---

# 4. Generate Balanced LLM Task Files

Main script:

```text
RAG/make_balanced_time_range_tasks.py
```

This script automatically creates a Slurm task table based on:

```text
start quarter
end quarter
max docs per task
```

It avoids manually writing quarters or shards.

---

## 4.1 Generate Tasks for 2024Q1–2026Q2

```bash
cd ~/sem2/RAG

python make_balanced_time_range_tasks.py   --input-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl   --out-tsv ../llm_tasks_2024Q1_2026Q2_2000docs.tsv   --start-quarter 2024Q1   --end-quarter 2026Q2   --max-docs-per-task 2000   --run-prefix y2024q1_2026q2_2000docs
```

Check task count:

```bash
cd ~/sem2

N=$(tail -n +2 llm_tasks_2024Q1_2026Q2_2000docs.tsv | wc -l)
echo $N

cat llm_tasks_2024Q1_2026Q2_2000docs.summary.txt
```

---

## 4.2 Generate Tasks for Pre-2024 Data

```bash
cd ~/sem2/RAG

python make_balanced_time_range_tasks.py   --input-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl   --out-tsv ../llm_tasks_pre2024_2000docs.tsv   --start-quarter 2019Q1   --end-quarter 2023Q4   --max-docs-per-task 2000   --run-prefix pre2024_2000docs
```

Check task file:

```bash
cd ~/sem2

N=$(tail -n +2 llm_tasks_pre2024_2000docs.tsv | wc -l)
echo $N

cat llm_tasks_pre2024_2000docs.summary.txt
```

Expected planning result:

```text
selected_doc_count = 17,172
num_shards = 9
max_docs_per_task = 2000
```

Check TSV columns:

```bash
cd ~/sem2

awk -F'\t' 'NR==1 {print "header columns:", NF; print $0} NR>1 {print NR-2, "columns="NF, "agent_tasks="$10, "output_tag="$9}' llm_tasks_pre2024_2000docs.tsv
```

Expected:

```text
columns=11
agent_tasks=concepts,relationships,outlook
```

---

# 5. Submit LLM Multi-Agent Extraction Jobs on Slurm

Main Slurm script:

```text
run_llm_agents_balanced_time_range_4l4.slurm
```

Each Slurm array task starts one vLLM server on:

```text
4 × NVIDIA L4
```

and runs the three agents:

```text
concepts
relationships
outlook
```

---

## 5.1 Submit 2024Q1–2026Q2 Tasks

```bash
cd ~/sem2

N=$(tail -n +2 llm_tasks_2024Q1_2026Q2_2000docs.tsv | wc -l)

TASK_FILE=~/sem2/llm_tasks_2024Q1_2026Q2_2000docs.tsv sbatch --array=0-$((N-1))%2 run_llm_agents_balanced_time_range_4l4.slurm
```

---

## 5.2 Submit Pre-2024 Tasks

```bash
cd ~/sem2

N=$(tail -n +2 llm_tasks_pre2024_2000docs.tsv | wc -l)

TASK_FILE=~/sem2/llm_tasks_pre2024_2000docs.tsv sbatch --array=0-$((N-1))%2 run_llm_agents_balanced_time_range_4l4.slurm
```

If GPU quota is tight, submit with `%1`:

```bash
TASK_FILE=~/sem2/llm_tasks_pre2024_2000docs.tsv sbatch --array=0-$((N-1))%1 run_llm_agents_balanced_time_range_4l4.slurm
```

---

## 5.3 Monitor Slurm Jobs

```bash
squeue -u $USER
```

Detailed view:

```bash
squeue -u $USER -o "%.18i %.8T %.10M %.20R %.10b %.40j"
```

If the pending reason is:

```text
QOSMaxGRESPerUser
```

it means the user GPU/GRES quota has been reached. If the job remains in the queue, it does **not** need to be resubmitted. It will start automatically after previous GPU jobs finish.

Cancel a job if needed:

```bash
scancel <jobid>
```

---

## 5.4 Monitor Logs

Main Slurm logs:

```bash
cd ~/sem2

ls -lh logs/
tail -f logs/llm_balanced_<jobid>_<arrayid>.out
```

vLLM server logs:

```bash
cd ~/sem2

ls -lh logs/vllm_server_*.log
tail -n 120 logs/vllm_server_<run_name>_<jobid>.log
```

Extraction logs inside RAG:

```bash
cd ~/sem2/RAG

ls -lh logs/
tail -f logs/llm_agents_<run_name>_<jobid>.log
```

---

## 5.5 Check LLM Extraction Outputs

Balanced time-range output directory:

```bash
cd ~/sem2/RAG

find rag_chroma_output/llm_csv_outputs_balanced_time_range -name "*.csv" -type f -exec wc -l {} \;
```

Check only outlook files:

```bash
find rag_chroma_output/llm_csv_outputs_balanced_time_range -name "outlook_*.csv" -type f -exec wc -l {} \;
```

Check failed files:

```bash
find rag_chroma_output/llm_csv_outputs_balanced_time_range -name "failed_*.csv" -type f -exec sh -c 'echo "===== $1 ====="; head -n 20 "$1"' _ {} \;
```

---

# 6. Run Master Network + Contagion Analysis

Main script:

```text
RAG/run_network_contagion_master_analysis.py
```

This script merges existing LLM outputs and creates:

```text
cleaned concepts
cleaned relationships
cleaned outlook
network nodes
network edges
pilot contagion events
falsification summaries
figures
```

---

## 6.1 Run Master Analysis

```bash
cd ~/sem2/RAG

python run_network_contagion_master_analysis.py   --rag-output-dir rag_chroma_output   --out-dir rag_chroma_output/network_contagion_master_analysis   --start-quarter 2019Q2   --end-quarter 2026Q2   --source-quarter 2025Q2   --target-quarter 2025Q3   --focus-signal margin_outlook   --focus-label improving
```

Windows:

```bash
cd E:\Projects\EearningAlz\RAG

python run_network_contagion_master_analysis.py ^
  --rag-output-dir rag_chroma_output ^
  --out-dir rag_chroma_output\network_contagion_master_analysis ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2 ^
  --source-quarter 2025Q2 ^
  --target-quarter 2025Q3 ^
  --focus-signal margin_outlook ^
  --focus-label improving
```

---

## 6.2 Output Directory

```text
rag_chroma_output/network_contagion_master_analysis/
```

Important outputs:

```text
input_file_manifest.csv
cleaned_concepts_all.csv
cleaned_relationships_all.csv
cleaned_outlook_all.csv
company_quarter_signal_matrix.csv
network_nodes.csv
network_edges.csv
network_centrality_degree.csv
contagion_events.csv
contagion_summary_by_signal_label_relation.csv
falsification_cases_margin_outlook_improving.csv
falsification_summary_margin_outlook_improving.csv
relationship_network_interactive.html
analysis_summary.md
```

Figures:

```text
figures/outlook_label_distribution_by_signal.png
figures/concept_distribution.png
figures/relationship_network_static.png
figures/top_direction_transmission_rates.png
figures/falsification_rate_margin_outlook_improving.png
```

---

## 6.3 Check Summary

```bash
cd ~/sem2/RAG

cat rag_chroma_output/network_contagion_master_analysis/analysis_summary.md
```

Current master results:

```text
Cleaned concepts rows: 11,087
Cleaned relationship rows: 40,070
Cleaned outlook rows: 58,584
Network nodes: 16,914
Network edges: 23,189
Contagion event rows for 2025Q2 → 2025Q3: 498
```

---

# 7. Run Rolling Full-Range Contagion Analysis

Main script:

```text
RAG/run_rolling_full_range_contagion.py
```

This script computes contagion for **all adjacent-quarter windows**, not just 2025Q2 → 2025Q3.

---

## 7.1 Run Rolling Analysis

```bash
cd ~/sem2/RAG

python run_rolling_full_range_contagion.py   --rag-output-dir rag_chroma_output   --out-dir rag_chroma_output/rolling_full_range_contagion_analysis   --start-quarter 2019Q2   --end-quarter 2026Q2   --focus-signal margin_outlook   --focus-label improving
```

Windows:

```bash
cd E:\Projects\EearningAlz\RAG

python run_rolling_full_range_contagion.py ^
  --rag-output-dir rag_chroma_output ^
  --out-dir rag_chroma_output\rolling_full_range_contagion_analysis ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2 ^
  --focus-signal margin_outlook ^
  --focus-label improving
```

---

## 7.2 Outputs

```text
rolling_contagion_events_all.csv
rolling_unmatched_relationship_entities_all.csv
rolling_window_summary.csv
rolling_contagion_summary_by_window_signal_relation.csv
rolling_contagion_summary_aggregated.csv
rolling_falsification_cases_margin_outlook_improving.csv
rolling_falsification_summary_margin_outlook_improving.csv
rolling_signal_quarter_label_counts.csv
rolling_analysis_summary.md
```

Figures:

```text
figures/rolling_event_rows_by_window.png
figures/rolling_transmission_rate_by_relation.png
figures/rolling_transmission_rate_by_signal.png
figures/rolling_falsification_rate_margin_outlook_improving.png
```

Check report:

```bash
cat rag_chroma_output/rolling_full_range_contagion_analysis/rolling_analysis_summary.md
```

---

# 8. Run Two-Part Network Prediction Analysis

Main script:

```text
RAG/run_two_part_network_prediction_analysis.py
```

This is the latest main analysis. It separates the analysis into:

```text
Part A: Cross-quarter lead-lag prediction
Part B: Same-quarter network correlation
```

---

## 8.1 Run Two-Part Analysis

```bash
cd ~/sem2/RAG

python run_two_part_network_prediction_analysis.py   --rag-output-dir rag_chroma_output   --out-dir rag_chroma_output/two_part_network_prediction_analysis   --start-quarter 2019Q2   --end-quarter 2026Q2
```

Windows:

```bash
cd E:\Projects\EearningAlz\RAG

python run_two_part_network_prediction_analysis.py ^
  --rag-output-dir rag_chroma_output ^
  --out-dir rag_chroma_output\two_part_network_prediction_analysis ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2
```

---

## 8.2 Optional: Use Only Quarter-Specific Relationships

Default uses all matched relationships as structural links. If you want stricter quarter-specific relationship edges:

```bash
python run_two_part_network_prediction_analysis.py   --rag-output-dir rag_chroma_output   --out-dir rag_chroma_output/two_part_network_prediction_analysis_quarter_specific   --start-quarter 2019Q2   --end-quarter 2026Q2   --use-quarter-specific-relationships
```

---

## 8.3 Outputs

```text
two_part_network_prediction_analysis/
├── input_file_manifest.csv
├── cleaned_outlook_all.csv
├── cleaned_relationships_all.csv
├── matched_company_relationships.csv
├── unmatched_relationship_entities.csv
├── cross_quarter_events.csv
├── cross_quarter_summary_by_window_signal_relation.csv
├── cross_quarter_prediction_accuracy.csv
├── same_quarter_events.csv
├── same_quarter_summary_by_quarter_signal_relation.csv
├── same_quarter_correlation_by_signal_relation.csv
├── combined_events_cross_and_same_quarter.csv
├── combined_summary_cross_and_same_quarter.csv
├── combined_accuracy_correlation_summary.csv
└── two_part_analysis_summary.md
```

Figures:

```text
figures/cross_quarter_event_rows_by_window.png
figures/cross_quarter_accuracy_by_signal.png
figures/cross_quarter_accuracy_by_relation.png
figures/same_quarter_event_rows_by_quarter.png
figures/same_quarter_similarity_by_signal.png
figures/same_quarter_similarity_by_relation.png
```

Check report:

```bash
cat rag_chroma_output/two_part_network_prediction_analysis/two_part_analysis_summary.md
```

Current results:

```text
Cleaned outlook rows: 58,584
Cleaned relationship rows: 40,070
Matched company relationships: 9,833
Unmatched relationship entities: 28,556
Available quarters: 21
Adjacent-quarter windows: 19
Cross-quarter event rows: 75,282
Same-quarter event rows: 99,090
```

---

# 9. Run Focused Contagion Analysis

Main script:

```text
RAG/analyze_signal_contagion.py
```

This is useful if you want to focus on one signal, for example:

```text
margin_outlook = improving
```

---

## 9.1 Run Focused Contagion

```bash
cd ~/sem2/RAG

python analyze_signal_contagion.py   --input-dir rag_chroma_output/llm_csv_outputs_2025Q2_Q3   --out-dir rag_chroma_output/contagion_analysis_2025Q2_Q3   --source-quarter 2025Q2   --target-quarter 2025Q3   --focus-signal margin_outlook   --focus-label improving
```

Windows:

```bash
cd E:\Projects\EearningAlz\RAG

python analyze_signal_contagion.py ^
  --input-dir rag_chroma_output\llm_csv_outputs_2025Q2_Q3 ^
  --out-dir rag_chroma_output\contagion_analysis_2025Q2_Q3 ^
  --source-quarter 2025Q2 ^
  --target-quarter 2025Q3 ^
  --focus-signal margin_outlook ^
  --focus-label improving
```

---

## 9.2 Outputs

```text
contagion_events.csv
contagion_summary_by_signal_label_relation.csv
contagion_summary_margin_improving.csv
unmatched_relationship_entities.csv
contagion_summary.md
figures/
```

---

# 10. Run Falsification / Non-Transmission Analysis

Main script:

```text
RAG/analyze_signal_falsification.py
```

This script reads `contagion_events.csv` and identifies cases where exposure did **not** lead to transmission.

---

## 10.1 Run Falsification Analysis

```bash
cd ~/sem2/RAG

python analyze_signal_falsification.py   --contagion-events rag_chroma_output/contagion_analysis_2025Q2_Q3/contagion_events.csv   --out-dir rag_chroma_output/contagion_analysis_2025Q2_Q3/falsification_margin_improving   --focus-signal margin_outlook   --focus-label improving
```

Only downstream:

```bash
python analyze_signal_falsification.py   --contagion-events rag_chroma_output/contagion_analysis_2025Q2_Q3/contagion_events.csv   --out-dir rag_chroma_output/contagion_analysis_2025Q2_Q3/falsification_margin_improving_downstream   --focus-signal margin_outlook   --focus-label improving   --relation-group downstream
```

---

## 10.2 Outputs

```text
falsification_cases_margin_outlook_improving.csv
falsification_summary_margin_outlook_improving.csv
falsification_reason_counts_margin_outlook_improving.csv
falsification_summary.md
```

---

# 11. Run Network Visualization

Main script:

```text
RAG/visualize_information_flow_network_v2.py
```

---

## 11.1 Run V2 Network Visualization

```bash
cd ~/sem2/RAG

python visualize_information_flow_network_v2.py   --input-dir rag_chroma_output/llm_csv_outputs_2025Q2_Q3   --out-dir rag_chroma_output/information_flow_network_demo_v2   --source-quarter 2025Q2   --target-quarter 2025Q3   --top-relationship-edges 200   --top-temporal-edges 200
```

Windows:

```bash
cd E:\Projects\EearningAlz\RAG

python visualize_information_flow_network_v2.py ^
  --input-dir rag_chroma_output\llm_csv_outputs_2025Q2_Q3 ^
  --out-dir rag_chroma_output\information_flow_network_demo_v2 ^
  --source-quarter 2025Q2 ^
  --target-quarter 2025Q3 ^
  --top-relationship-edges 200 ^
  --top-temporal-edges 200
```

---

## 11.2 Outputs

```text
flow_nodes_v2.csv
flow_edges_v2.csv
information_flow_network_v2.html
information_flow_network_v2_static.png
edge_type_counts.csv
edge_type_counts.png
signal_flow_counts.csv
signal_flow_counts.png
README_information_flow_v2.md
```

Open the interactive graph:

```bash
open rag_chroma_output/information_flow_network_demo_v2/information_flow_network_v2.html
```

On Windows, open the HTML file directly from File Explorer.

---

# 12. Overleaf / Paper Output Workflow

The latest Overleaf file is:

```text
updated_two_part_network_prediction_overleaf.tex
```

The latest workflow summary is based on:

```text
rag_chroma_output/two_part_network_prediction_analysis/two_part_analysis_summary.md
```

---

## 12.1 Figures to Insert into Overleaf

```text
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_event_rows_by_window.png
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_accuracy_by_signal.png
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_accuracy_by_relation.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_event_rows_by_quarter.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_similarity_by_signal.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_similarity_by_relation.png
```

---

## 12.2 Important Table Correction

In the cross-quarter results table, do **not** show both:

```text
Direction match rate
Prediction accuracy
```

under the current definition, because they are mathematically identical:

```text
prediction_accuracy = direction_match_edges / exposed_edges
direction_match_rate = direction_match_edges / exposed_edges
```

Use this table structure instead:

```text
Signal and source label
Relation group
Exposed edges
Target active rate
Exact match rate
Direction match rate
```

Interpretation:

```text
Direction match rate is a preliminary transcript-signal prediction measure.
It is not a complete machine-learning forecasting accuracy metric.
```

---

# 13. Final End-to-End Command Flow

This is the complete execution sequence from scratch.

---

## Step 1: Build RAG Index

```bash
cd ~/sem2/RAG
python build_chroma_rag_index.py
```

---

## Step 2: Retrieve Full RAG Evidence

```bash
cd ~/sem2/RAG
python retrieve_chroma_rag_evidence_full_gpu_direct.py
```

---

## Step 3: Generate LLM Task Table

Example for 2024Q1–2026Q2:

```bash
cd ~/sem2/RAG

python make_balanced_time_range_tasks.py   --input-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl   --out-tsv ../llm_tasks_2024Q1_2026Q2_2000docs.tsv   --start-quarter 2024Q1   --end-quarter 2026Q2   --max-docs-per-task 2000   --run-prefix y2024q1_2026q2_2000docs
```

Example for pre-2024:

```bash
cd ~/sem2/RAG

python make_balanced_time_range_tasks.py   --input-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl   --out-tsv ../llm_tasks_pre2024_2000docs.tsv   --start-quarter 2019Q1   --end-quarter 2023Q4   --max-docs-per-task 2000   --run-prefix pre2024_2000docs
```

---

## Step 4: Submit LLM Extraction Jobs

```bash
cd ~/sem2

N=$(tail -n +2 llm_tasks_2024Q1_2026Q2_2000docs.tsv | wc -l)

TASK_FILE=~/sem2/llm_tasks_2024Q1_2026Q2_2000docs.tsv sbatch --array=0-$((N-1))%2 run_llm_agents_balanced_time_range_4l4.slurm
```

Pre-2024:

```bash
cd ~/sem2

N=$(tail -n +2 llm_tasks_pre2024_2000docs.tsv | wc -l)

TASK_FILE=~/sem2/llm_tasks_pre2024_2000docs.tsv sbatch --array=0-$((N-1))%2 run_llm_agents_balanced_time_range_4l4.slurm
```

---

## Step 5: Monitor Jobs

```bash
squeue -u $USER
tail -f logs/llm_balanced_<jobid>_<arrayid>.out
```

---

## Step 6: Check Extraction Outputs

```bash
cd ~/sem2/RAG

find rag_chroma_output/llm_csv_outputs_balanced_time_range -name "*.csv" -type f -exec wc -l {} \;
```

---

## Step 7: Run Two-Part Network Prediction Analysis

```bash
cd ~/sem2/RAG

python run_two_part_network_prediction_analysis.py   --rag-output-dir rag_chroma_output   --out-dir rag_chroma_output/two_part_network_prediction_analysis   --start-quarter 2019Q2   --end-quarter 2026Q2
```

---

## Step 8: Check Final Analysis Report

```bash
cat rag_chroma_output/two_part_network_prediction_analysis/two_part_analysis_summary.md
```

---

## Step 9: Use Results in Overleaf

Use the following output files:

```text
rag_chroma_output/two_part_network_prediction_analysis/cross_quarter_prediction_accuracy.csv
rag_chroma_output/two_part_network_prediction_analysis/same_quarter_correlation_by_signal_relation.csv
rag_chroma_output/two_part_network_prediction_analysis/two_part_analysis_summary.md
```

Use the following figures:

```text
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_event_rows_by_window.png
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_accuracy_by_signal.png
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_accuracy_by_relation.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_event_rows_by_quarter.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_similarity_by_signal.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_similarity_by_relation.png
```

---

# 14. One-Sentence Summary

The current Python workflow is:

```text
Raw earnings call transcripts
        ↓
RAG ChromaDB chunk embedding storage
        ↓
GPU direct RAG evidence retrieval
        ↓
vLLM multi-agent extraction
        ↓
structured concepts / relationships / outlook signals
        ↓
corporate intelligence network
        ↓
cross-quarter lead-lag prediction
        ↓
same-quarter network correlation
        ↓
Overleaf research results
```

The core research conclusion is:

```text
Earnings call transcripts can be transformed into a structured corporate intelligence network.
Cross-quarter network signals provide preliminary lead-lag predictive value.
Same-quarter connected firms show meaningful signal co-movement.
Information propagation is selective and relationship-dependent.
```
