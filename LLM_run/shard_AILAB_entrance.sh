cd ~/sem2/RAG

# first we need to split the data into tasks, bcs the data is too large
python make_balanced_time_range_tasks_hf.py \
  --hf-dataset soysouce/earningALZ_SBERT_evidence \
  --hf-metadata-file rag_evidence_package_metadata_full_gpu_direct.parquet \
  --start-quarter 2019Q1 \
  --end-quarter 2023Q4 \
  --max-docs-per-task 2000 \
  --output-tag-prefix pre2024_2000docs_hf \
  --output-base-dir rag_chroma_output/llm_parquet_outputs_hf \
  --out-tsv ../llm_tasks_pre2024_2000docs_hf.tsv


# now we have the tasks, we can run the agents on the tasks
cd ~/sem2

N=$(tail -n +2 llm_tasks_pre2024_2000docs_hf.tsv | wc -l)

TASK_FILE=~/sem2/llm_tasks_pre2024_2000docs_hf.tsv \
sbatch --array=0-$((N-1))%2 run_llm_agents_hf_balanced_time_range_4l4.slurm