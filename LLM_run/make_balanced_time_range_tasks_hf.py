#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create balanced LLM task TSV from Hugging Face evidence metadata.

Default data source:
    soysouce/earningALZ_SBERT_evidence

Default file:
    rag_evidence_package_metadata_full_gpu_direct.parquet

The TSV schema is intentionally compatible with the existing SLURM script:

1 task_id
2 start_quarter
3 end_quarter
4 target_quarters
5 selected_doc_count
6 max_docs_per_task
7 num_shards
8 shard_id
9 output_tag
10 agent_tasks
11 output_base_dir

Example
-------
python make_balanced_time_range_tasks_hf.py ^
  --hf-dataset soysouce/earningALZ_SBERT_evidence ^
  --start-quarter 2019Q1 ^
  --end-quarter 2023Q4 ^
  --max-docs-per-task 2000 ^
  --output-tag-prefix pre2024_2000docs ^
  --output-base-dir rag_chroma_output/llm_parquet_outputs_hf ^
  --out-tsv llm_tasks_pre2024_2000docs_hf.tsv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def quarter_to_index(q: str) -> float:
    q = str(q).strip()
    m = re.match(r"^(\d{4})Q([1-4])$", q)
    if not m:
        return float("nan")
    return int(m.group(1)) * 4 + int(m.group(2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", default="soysouce/earningALZ_SBERT_evidence")
    p.add_argument("--hf-metadata-file", default="rag_evidence_package_metadata_full_gpu_direct.parquet")
    p.add_argument("--hf-revision", default="main")
    p.add_argument("--start-quarter", required=True)
    p.add_argument("--end-quarter", required=True)
    p.add_argument("--max-docs-per-task", type=int, default=2000)
    p.add_argument("--agent-tasks", default="concepts,relationships,outlook")
    p.add_argument("--output-tag-prefix", default="")
    p.add_argument("--output-base-dir", default="rag_chroma_output/llm_parquet_outputs_hf")
    p.add_argument("--out-tsv", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    local_path = hf_hub_download(
        repo_id=args.hf_dataset,
        filename=args.hf_metadata_file,
        repo_type="dataset",
        revision=args.hf_revision,
    )

    df = pd.read_parquet(local_path)
    if "quarter" not in df.columns:
        raise ValueError(f"HF metadata file does not contain quarter column: {args.hf_metadata_file}")

    df["quarter"] = df["quarter"].astype(str).str.strip()
    df["_qidx"] = df["quarter"].map(quarter_to_index)

    start_idx = quarter_to_index(args.start_quarter)
    end_idx = quarter_to_index(args.end_quarter)
    selected = df[df["_qidx"].notna() & (df["_qidx"] >= start_idx) & (df["_qidx"] <= end_idx)].copy()

    selected_doc_count = len(selected)
    if selected_doc_count == 0:
        raise RuntimeError(f"No documents selected for {args.start_quarter} to {args.end_quarter}")

    num_shards = max(1, math.ceil(selected_doc_count / args.max_docs_per_task))
    quarters = sorted(selected["quarter"].dropna().unique(), key=quarter_to_index)
    target_quarters = ",".join(quarters)

    prefix = args.output_tag_prefix.strip()
    if not prefix:
        prefix = f"{args.start_quarter.lower()}_{args.end_quarter.lower()}_{args.max_docs_per_task}docs"

    rows = []
    for shard_id in range(num_shards):
        rows.append({
            "task_id": shard_id,
            "start_quarter": args.start_quarter,
            "end_quarter": args.end_quarter,
            "target_quarters": target_quarters,
            "selected_doc_count": selected_doc_count,
            "max_docs_per_task": args.max_docs_per_task,
            "num_shards": num_shards,
            "shard_id": shard_id,
            "output_tag": f"{prefix}_s{shard_id:03d}_of{num_shards:03d}",
            "agent_tasks": args.agent_tasks,
            "output_base_dir": args.output_base_dir,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

    print("SAVED", out_path)
    print("HF dataset:", args.hf_dataset)
    print("HF metadata file:", args.hf_metadata_file)
    print("Selected docs:", selected_doc_count)
    print("Quarters:", target_quarters)
    print("Max docs per task:", args.max_docs_per_task)
    print("Num shards:", num_shards)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
