#!/usr/bin/env python3
"""
Create balanced LLM extraction tasks by time range and total workload.

Key idea:
- User only provides a time range, e.g. 2024Q1 to 2026Q2.
- Script scans rag_evidence_packages_full_gpu_direct.jsonl.
- It counts all documents in that range.
- It creates N balanced shards based on --max-docs-per-task.
- Every Slurm task uses the same TARGET_QUARTERS list, but different SHARD_ID.
- extract_llm_agents_csv_vllm.py will filter by TARGET_QUARTERS and then apply NUM_SHARDS/SHARD_ID.

This avoids hand-writing quarters and works later for 2000Q1-2026Q2.

Example:
  cd ~/sem2/RAG

  python make_balanced_time_range_tasks.py \
    --input-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl \
    --out-tsv ../llm_tasks_2024Q1_2026Q2.tsv \
    --start-quarter 2024Q1 \
    --end-quarter 2026Q2 \
    --max-docs-per-task 160 \
    --run-prefix y2024q1_2026q2

Then submit:
  cd ~/sem2
  N=$(tail -n +2 llm_tasks_2024Q1_2026Q2.tsv | wc -l)
  TASK_FILE=~/sem2/llm_tasks_2024Q1_2026Q2.tsv \
  sbatch --array=0-$((N-1))%2 run_llm_agents_balanced_time_range_4l4.slurm
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter

import pandas as pd


def quarter_to_index(q: str) -> int:
    q = str(q).strip()
    m = re.match(r"^(\d{4})Q([1-4])$", q)
    if not m:
        raise ValueError(f"Invalid quarter format: {q}. Expected YYYYQn, e.g. 2024Q1")
    return int(m.group(1)) * 4 + int(m.group(2))


def index_to_quarter(idx: int) -> str:
    year = (idx - 1) // 4
    q = idx - year * 4
    return f"{year}Q{q}"


def quarter_range(start_q: str, end_q: str) -> list[str]:
    s = quarter_to_index(start_q)
    e = quarter_to_index(end_q)
    if s > e:
        raise ValueError(f"start-quarter {start_q} is after end-quarter {end_q}")
    return [index_to_quarter(i) for i in range(s, e + 1)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-jsonl",
        default="rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl",
        help="Full RAG evidence JSONL.",
    )
    p.add_argument(
        "--out-tsv",
        required=True,
        help="Output task TSV path, usually ../llm_tasks_YYYYQn_YYYYQn.tsv",
    )
    p.add_argument("--start-quarter", required=True, help="Example: 2024Q1")
    p.add_argument("--end-quarter", required=True, help="Example: 2026Q2")
    p.add_argument(
        "--max-docs-per-task",
        type=int,
        default=160,
        help=(
            "Target maximum documents per Slurm task. "
            "Based on observed speed: 144 docs took about 34 min extraction plus vLLM startup."
        ),
    )
    p.add_argument(
        "--run-prefix",
        default="",
        help="Optional output tag prefix. Default is auto-generated from start/end quarters.",
    )
    p.add_argument(
        "--agent-tasks",
        default="concepts,relationships,outlook",
        help="Tasks to run. Stored in TSV for Slurm.",
    )
    p.add_argument(
        "--output-base-dir",
        default="rag_chroma_output/llm_csv_outputs_balanced_time_range",
        help="Base output directory used by Slurm.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_jsonl)
    out_path = Path(args.out_tsv)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    selected_quarters = quarter_range(args.start_quarter, args.end_quarter)
    selected_set = set(selected_quarters)

    quarter_counts = Counter()
    total_rows = 0
    selected_docs = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_rows += 1
            obj = json.loads(line)
            q = str(obj.get("quarter", "")).strip()
            if q in selected_set:
                quarter_counts[q] += 1
                selected_docs += 1

    if selected_docs == 0:
        raise RuntimeError(
            f"No documents found for selected range {args.start_quarter} to {args.end_quarter}"
        )

    num_shards = max(1, math.ceil(selected_docs / args.max_docs_per_task))
    target_quarters_csv = ",".join([q for q in selected_quarters if quarter_counts[q] > 0])

    if not target_quarters_csv:
        raise RuntimeError("No non-empty quarters in selected range.")

    run_prefix = args.run_prefix.strip()
    if not run_prefix:
        run_prefix = f"{args.start_quarter}_{args.end_quarter}".lower()

    rows = []
    for shard_id in range(num_shards):
        output_tag = f"{run_prefix}_s{shard_id:03d}_of{num_shards:03d}"
        rows.append({
            "task_id": shard_id,
            "start_quarter": args.start_quarter,
            "end_quarter": args.end_quarter,
            "target_quarters": target_quarters_csv,
            "selected_doc_count": selected_docs,
            "max_docs_per_task": args.max_docs_per_task,
            "num_shards": num_shards,
            "shard_id": shard_id,
            "output_tag": output_tag,
            "agent_tasks": args.agent_tasks,
            "output_base_dir": args.output_base_dir,
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)

    summary_path = out_path.with_suffix(".summary.txt")
    lines = []
    lines.append("Balanced LLM task generation summary")
    lines.append("=" * 60)
    lines.append(f"input_jsonl: {input_path}")
    lines.append(f"out_tsv: {out_path}")
    lines.append(f"start_quarter: {args.start_quarter}")
    lines.append(f"end_quarter: {args.end_quarter}")
    lines.append(f"target_quarters: {target_quarters_csv}")
    lines.append(f"total_jsonl_rows: {total_rows}")
    lines.append(f"selected_docs: {selected_docs}")
    lines.append(f"max_docs_per_task: {args.max_docs_per_task}")
    lines.append(f"num_shards: {num_shards}")
    lines.append("")
    lines.append("Quarter distribution:")
    for q in selected_quarters:
        if quarter_counts[q] > 0:
            lines.append(f"{q}\t{quarter_counts[q]}")
    lines.append("")
    lines.append("Task preview:")
    lines.append(df.head(20).to_string(index=False))

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print("SAVED TSV:", out_path)
    print("SAVED SUMMARY:", summary_path)


if __name__ == "__main__":
    main()
