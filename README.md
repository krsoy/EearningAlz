# EarningALZ

Exam-ready workflow for predicting delayed earnings-call signals in corporate networks.

The cleaned project is organized around one reproducible pipeline:

```text
data merge
  -> ChromaDB transcript chunk index
  -> SBERT/Chroma evidence retrieval
  -> Qwen2.5-14B-Instruct LLM extraction
  -> two-part network prediction
  -> V4 cluster diffusion analysis
  -> optional LLM-as-judge validation
```

All old experiments, dashboard prototypes, scraping side paths, and legacy analysis branches have been moved to `Archive/`.


## Project Layout

```text
.
├── README.md
├── requirements.txt
├── setup_uv_cuda_vllm_env.sh
├── run_llm_agents_balanced_time_range_4l4.slurm
├── data/
│   ├── exam_ready_combined_transcripts_kaggle_motley.py
│   ├── merge_motley_hf_transcripts.py
│   ├── motley-fool-data.pkl
│   └── combined_transcript_data/
├── RAG/
│   ├── exam_ready_sh.sh
│   ├── exam_ready_build_chroma_rag_index.py
│   ├── exam_ready_sbert_chunk_label.py
│   ├── make_balanced_time_range_tasks.py
│   ├── extract_llm_agents_csv_vllm.py
│   ├── run_two_part_network_prediction_analysis.py
│   ├── cluster_method_comparison.py
│   └── rag_chroma_output/
├── LLM_result_alz/
│   └── llm_judge/
└── Archive/
```


## Core Entry Points

| Stage | Entry file | Purpose |
|---|---|---|
| Full command workflow | `RAG/exam_ready_sh.sh` | Main orchestrator based on the exam workflow. |
| Environment setup | `setup_uv_cuda_vllm_env.sh` | CUDA/vLLM-oriented environment setup. |
| Data merge | `data/exam_ready_combined_transcripts_kaggle_motley.py` | Merge Kaggle and Hugging Face Motley Fool transcripts, clean, deduplicate. |
| Chroma build | `RAG/exam_ready_build_chroma_rag_index.py` | Chunk transcripts and build a ChromaDB embedding index. |
| Evidence retrieval | `RAG/exam_ready_sbert_chunk_label.py` | Retrieve relationship, supply-chain, and outlook evidence chunks. |
| Task generation | `RAG/make_balanced_time_range_tasks.py` | Create balanced Slurm task TSV by quarter range. |
| LLM extraction | `run_llm_agents_balanced_time_range_4l4.slurm` + `RAG/extract_llm_agents_csv_vllm.py` | Run Qwen/vLLM extraction for concepts, relationships, and outlook. |
| Two-part analysis | `RAG/run_two_part_network_prediction_analysis.py` | Cross-quarter lead-lag and same-quarter co-movement analysis. |
| Cluster V4 | `RAG/cluster_method_comparison.py` | Relationship/signal graph cluster comparison and diffusion analysis. |
| LLM judge | `LLM_result_alz/llm_judge/` | Optional CrewAI/HITL validation workflow. |


## Environment

On the AI-LAB server, create or refresh the environment:

```bash
cd ~/sem2
bash setup_uv_cuda_vllm_env.sh
```

Minimal Python dependencies are listed in:

```bash
requirements.txt
```

The LLM extraction and judge steps expect a CUDA-capable environment and local vLLM serving of:

```text
Qwen/Qwen2.5-14B-Instruct
```


## Full Workflow

Run the main workflow from the project root:

```bash
cd ~/sem2
bash RAG/exam_ready_sh.sh
```

The default quarter range is:

```text
2024Q1 to 2026Q2
```

Override it if needed:

```bash
START_QUARTER=2019Q2 END_QUARTER=2026Q2 bash RAG/exam_ready_sh.sh
```

The script submits Slurm jobs in Step 5 and then prints the post-LLM commands. Wait until all LLM array jobs finish before running the analysis steps.


## Manual Commands

### 1. Merge And Clean Data

```bash
cd ~/sem2/data
python exam_ready_combined_transcripts_kaggle_motley.py
```

Main output:

```text
data/combined_transcript_data/combined_transcripts_deduplicated.csv
```


### 2. Build Chroma Index

```bash
cd ~/sem2/RAG
python exam_ready_build_chroma_rag_index.py \
  --input-csv ../data/combined_transcript_data/combined_transcripts_deduplicated.csv
```

Main outputs:

```text
RAG/rag_chroma_output/chroma_db/
RAG/rag_chroma_output/chunk_index.parquet
RAG/rag_chroma_output/build_summary.json
```


### 3. Retrieve Evidence

```bash
cd ~/sem2/RAG
python exam_ready_sbert_chunk_label.py \
  --download-hf \
  --repo-id soysouce/earning_chroma \
  --hf-local-dir hf_earning_chroma \
  --chroma-dir hf_earning_chroma/chroma_db \
  --collection earnings_call_chunks \
  --device cuda \
  --out-dir rag_chroma_output \
  --suffix full_gpu_direct
```

Main outputs:

```text
RAG/rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl
RAG/rag_chroma_output/rag_evidence_chunks_flat_full_gpu_direct.csv
RAG/rag_chroma_output/retrieval_summary_full_gpu_direct.json
```


### 4. Create LLM Task TSV

```bash
cd ~/sem2/RAG
python make_balanced_time_range_tasks.py \
  --input-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl \
  --out-tsv ../llm_tasks_2024Q1_2026Q2.tsv \
  --start-quarter 2024Q1 \
  --end-quarter 2026Q2 \
  --max-docs-per-task 160 \
  --run-prefix y2024q1_2026q2
```


### 5. Submit LLM Extraction

```bash
cd ~/sem2
N=$(tail -n +2 llm_tasks_2024Q1_2026Q2.tsv | wc -l)
TASK_FILE=~/sem2/llm_tasks_2024Q1_2026Q2.tsv \
sbatch --array=0-$((N-1))%2 run_llm_agents_balanced_time_range_4l4.slurm
```

Expected extraction outputs:

```text
RAG/rag_chroma_output/llm_csv_outputs_balanced_time_range/
  <range>/
    concepts_*.csv
    relationships_*.csv
    outlook_*.csv
    failed_*.csv
    progress_*.json
```


### 6. Run Two-Part Network Prediction

Run this only after all LLM extraction jobs finish.

```bash
cd ~/sem2/RAG
python run_two_part_network_prediction_analysis.py \
  --rag-output-dir rag_chroma_output \
  --out-dir rag_chroma_output/two_part_network_prediction_analysis \
  --start-quarter 2024Q1 \
  --end-quarter 2026Q2 \
  --min-exposed-for-plot 5
```

Main outputs:

```text
RAG/rag_chroma_output/two_part_network_prediction_analysis/cross_quarter_events.csv
RAG/rag_chroma_output/two_part_network_prediction_analysis/cross_quarter_prediction_accuracy.csv
RAG/rag_chroma_output/two_part_network_prediction_analysis/same_quarter_events.csv
RAG/rag_chroma_output/two_part_network_prediction_analysis/same_quarter_correlation_by_signal_relation.csv
RAG/rag_chroma_output/two_part_network_prediction_analysis/two_part_analysis_summary.md
RAG/rag_chroma_output/two_part_network_prediction_analysis/figures/
```


### 7. Run V4 Cluster Diffusion

```bash
cd ~/sem2/RAG
python cluster_method_comparison.py \
  --rag-output-dir rag_chroma_output \
  --two-part-dir rag_chroma_output/two_part_network_prediction_analysis \
  --combined-transcripts ../data/combined_transcript_data/combined_transcripts_deduplicated.csv \
  --evidence-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl \
  --out-dir rag_chroma_output/cluster_method_comparison_v4 \
  --start-quarter 2024Q1 \
  --end-quarter 2026Q2 \
  --min-k 5 \
  --max-k 100 \
  --min-cluster-size 10 \
  --min-exposed 10 \
  --cooccur-min-weight 2
```

Main outputs:

```text
RAG/rag_chroma_output/cluster_method_comparison_v4/v4_clustering_method_comparison_summary.md
RAG/rag_chroma_output/cluster_method_comparison_v4/figures/
```


## Optional LLM Judge Workflow

The judge workflow validates sampled relationship/signal cases using a CrewAI-based judge plus optional human review.

Create cases:

```bash
cd ~/sem2/LLM_result_alz/llm_judge
python sample_cases.py --n-per-group 10 --out judge_cases.jsonl
```

Submit judge jobs:

```bash
cd ~/sem2
sbatch LLM_result_alz/llm_judge/run_judge.slurm
```

Merge and review outputs:

```bash
cd ~/sem2/LLM_result_alz/llm_judge
python merge_result.py
python hitl_validator.py --stats-only --validated hitl_validated.jsonl
```


## Archive Policy

`Archive/` contains material that is intentionally kept out of the active workflow:

```text
Archive/legacy_dashboard/       old AAPL dashboard/demo files
Archive/legacy_scraping/        scraping and SEC side paths
Archive/legacy_hf_pipeline/     older Hugging Face/parquet branch
Archive/legacy_rag_pipeline/    older RAG shell/Slurm entry points
Archive/legacy_rag_scripts/     scripts not called by exam_ready_sh.sh
Archive/legacy_outputs/         old demo/progress/output folders
Archive/legacy_analysis_*       older downstream analysis branches/results
Archive/test/                   exploratory EDA and validation experiments
```

Nothing in `Archive/` is required to run the exam-ready workflow.


## Clean Working Rule

For future work, keep new files in one of these active locations:

```text
data/                  data merge inputs and cleaned transcript outputs
RAG/                   workflow scripts and generated RAG/network outputs
LLM_result_alz/llm_judge/  judge workflow only
Archive/               old experiments or deprecated branches
```

Avoid adding new root-level scripts unless they are true project entry points.
