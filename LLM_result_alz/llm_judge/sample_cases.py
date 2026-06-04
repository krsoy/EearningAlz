#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample 10 cases per (signal × relation_group) from events parquets on HuggingFace,
then enrich each case with its original evidence chunks from the evidence packages.

Output: judge_cases.jsonl  — one case per line, ready for CrewAI judge workflow.

Usage:
    python sample_cases.py
    python sample_cases.py --n-per-group 10 --out judge_cases.jsonl
    python sample_cases.py --filter-source AAPL --filter-target CARR
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


TWO_PART_DATASET   = "soysouce/earningALZ_twopart"
EVIDENCE_DATASET   = "soysouce/earningALZ_SBERT_evidence"
EVIDENCE_FILE      = "rag_evidence_packages_full_gpu_direct.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--two-part-dataset", default=TWO_PART_DATASET)
    p.add_argument("--evidence-dataset",  default=EVIDENCE_DATASET)
    p.add_argument("--evidence-file",     default=EVIDENCE_FILE)
    p.add_argument("--n-per-group",  type=int, default=10,
                   help="Number of cases to sample per (signal × relation_group)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--mode",         default="both",
                   choices=["cross_quarter", "same_quarter", "both"])
    p.add_argument("--filter-source", default="",  help="Filter by source ticker, e.g. AAPL")
    p.add_argument("--filter-target", default="",  help="Filter by target ticker, e.g. CARR")
    p.add_argument("--filter-signal", default="",  help="Filter by signal, e.g. supply_outlook")
    p.add_argument("--filter-direction", default="", help="Filter by direction, e.g. negative")
    p.add_argument("--filter-relation", default="", help="Filter by relation_group, e.g. upstream")
    p.add_argument("--out", default="judge_cases.jsonl")
    return p.parse_args()


def hf_parquet(repo: str, filename: str) -> pd.DataFrame:
    path = hf_hub_download(repo_id=repo, filename=filename,
                           repo_type="dataset", revision="main")
    df = pd.read_parquet(path)
    print(f"  loaded {repo}/{filename}: {len(df):,} rows")
    return df


def ticker_from_node(node: str) -> str:
    return str(node).replace("COMPANY::", "").strip()


def load_events(args) -> pd.DataFrame:
    dfs = []
    if args.mode in ("cross_quarter", "both"):
        df = hf_parquet(args.two_part_dataset, "cross_quarter_events.parquet")
        df["analysis_mode"] = "cross_quarter"
        dfs.append(df)
    if args.mode in ("same_quarter", "both"):
        df = hf_parquet(args.two_part_dataset, "same_quarter_events.parquet")
        df["analysis_mode"] = "same_quarter"
        dfs.append(df)
    events = pd.concat(dfs, ignore_index=True)

    # Normalise string columns
    for c in ["source_node", "target_node", "signal", "relation_group",
              "source_direction", "target_direction", "analysis_mode"]:
        if c in events.columns:
            events[c] = events[c].astype(str).str.strip()

    # Derive ticker columns
    events["source_ticker"] = events["source_node"].map(ticker_from_node)
    events["target_ticker"] = events["target_node"].map(ticker_from_node)

    return events


def apply_filters(events: pd.DataFrame, args) -> pd.DataFrame:
    mask = pd.Series(True, index=events.index)
    if args.filter_source:
        mask &= events["source_ticker"].str.upper() == args.filter_source.upper()
    if args.filter_target:
        mask &= events["target_ticker"].str.upper() == args.filter_target.upper()
    if args.filter_signal:
        mask &= events["signal"] == args.filter_signal
    if args.filter_direction:
        mask &= events["source_direction"] == args.filter_direction
    if args.filter_relation:
        mask &= events["relation_group"] == args.filter_relation
    filtered = events[mask].copy()
    print(f"  after filters: {len(filtered):,} rows")
    return filtered


def sample_cases(events: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows per (signal, relation_group, analysis_mode) combination."""
    rng = random.Random(seed)
    groups = events.groupby(["signal", "relation_group", "analysis_mode"], dropna=False)
    sampled = []
    for (signal, relation, mode), grp in groups:
        take = min(n, len(grp))
        idx = rng.sample(list(grp.index), take)
        sampled.append(grp.loc[idx])
        print(f"    ({signal}, {relation}, {mode}): {take} samples from {len(grp):,}")
    return pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()


def load_evidence_packages(args) -> dict[str, dict]:
    """Load evidence packages, keyed by doc_id."""
    path = hf_hub_download(
        repo_id=args.evidence_dataset,
        filename=args.evidence_file,
        repo_type="dataset", revision="main",
    )
    df = pd.read_parquet(path)
    print(f"  loaded evidence packages: {len(df):,} rows")

    pkg_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        obj = row.to_dict()
        doc_id = str(obj.get("doc_id", "")).strip()
        if doc_id:
            pkg_map[doc_id] = obj
    return pkg_map


def maybe_json(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                pass
    return x


def extract_chunks(pkg: dict) -> list[dict]:
    """Extract all retrieved chunks from an evidence package."""
    evidence = maybe_json(pkg.get("retrieved_evidence", {}))
    if not isinstance(evidence, dict):
        return []
    chunks = []
    for group_key in ("relationship_chunks", "supply_chain_chunks", "expectation_chunks"):
        group = maybe_json(evidence.get(group_key, []))
        if isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    item = dict(item)
                    item["evidence_group"] = group_key
                    chunks.append(item)
    return chunks


def find_source_doc_id(pkg_map: dict, ticker: str, quarter: str) -> str | None:
    """Find the doc_id for a given ticker + quarter."""
    ticker = ticker.upper().strip()
    quarter = quarter.strip()
    for doc_id, pkg in pkg_map.items():
        if (str(pkg.get("ticker", "")).upper().strip() == ticker and
                str(pkg.get("quarter", "")).strip() == quarter):
            return doc_id
    return None


def build_judge_case(row: pd.Series, pkg_map: dict, case_id: int) -> dict:
    src_ticker  = ticker_from_node(row.get("source_node", ""))
    tgt_ticker  = ticker_from_node(row.get("target_node", ""))
    src_quarter = str(row.get("source_quarter", "")).strip()
    tgt_quarter = str(row.get("target_quarter", "")).strip()

    # Find source doc
    src_doc_id = find_source_doc_id(pkg_map, src_ticker, src_quarter)
    tgt_doc_id = find_source_doc_id(pkg_map, tgt_ticker, tgt_quarter)

    src_pkg = pkg_map.get(src_doc_id, {}) if src_doc_id else {}
    tgt_pkg = pkg_map.get(tgt_doc_id, {}) if tgt_doc_id else {}

    src_chunks = extract_chunks(src_pkg)
    tgt_chunks = extract_chunks(tgt_pkg)

    return {
        "case_id": case_id,
        "analysis_mode":    str(row.get("analysis_mode", "")),
        "signal":           str(row.get("signal", "")),
        "relation_group":   str(row.get("relation_group", "")),
        "source_direction": str(row.get("source_direction", "")),
        "target_direction": str(row.get("target_direction", "")),
        "direction_match":  bool(row.get("direction_match", False)),
        "source_active":    bool(row.get("source_active", False)),
        "target_active":    bool(row.get("target_active", False)),
        # Source transcript info
        "source_ticker":    src_ticker,
        "source_node":      str(row.get("source_node", "")),
        "source_quarter":   src_quarter,
        "source_doc_id":    src_doc_id or "",
        "source_company":   str(src_pkg.get("current_company", src_ticker)),
        "source_publish_date": str(src_pkg.get("publish_date", "")),
        "source_chunks":    src_chunks,
        # Target transcript info
        "target_ticker":    tgt_ticker,
        "target_node":      str(row.get("target_node", "")),
        "target_quarter":   tgt_quarter,
        "target_doc_id":    tgt_doc_id or "",
        "target_company":   str(tgt_pkg.get("current_company", tgt_ticker)),
        "target_publish_date": str(tgt_pkg.get("publish_date", "")),
        "target_chunks":    tgt_chunks,
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("LLM Judge Case Sampler")
    print("=" * 60)

    print("\n[1/4] Loading events...")
    events = load_events(args)
    events = apply_filters(events, args)

    if events.empty:
        print("No events after filtering. Exiting.")
        return

    print(f"\n[2/4] Sampling {args.n_per_group} per group...")
    sampled = sample_cases(events, args.n_per_group, args.seed)
    print(f"  total sampled cases: {len(sampled):,}")

    print("\n[3/4] Loading evidence packages...")
    pkg_map = load_evidence_packages(args)

    print("\n[4/4] Building judge cases...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_src = 0
    missing_tgt = 0

    with out_path.open("w", encoding="utf-8") as f:
        for case_id, (_, row) in enumerate(sampled.iterrows()):
            case = build_judge_case(row, pkg_map, case_id)
            if not case["source_doc_id"]:
                missing_src += 1
            if not case["target_doc_id"]:
                missing_tgt += 1
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. Written: {written} cases → {out_path}")
    print(f"  source doc not found: {missing_src}")
    print(f"  target doc not found: {missing_tgt}")
    print("\nSignal × Relation distribution:")
    print(sampled.groupby(["signal", "relation_group", "analysis_mode"]).size().to_string())


if __name__ == "__main__":
    main()

