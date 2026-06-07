#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_pre_call_target_dataset.py

Build a pre-call target-firm-quarter-signal prediction dataset.

Old event-level target:
    y = 1 if target_active and direction_match, else 0

New business target:
    For each target firm, target quarter, and signal type,
    predict the content of the target firm's upcoming earnings call:
        1. whether this signal will be active;
        2. what direction it will have.

Unit of observation:
    target_company_node + target_quarter + signal

Allowed input information:
    - connected firms' already available source signals;
    - previous-quarter connected source signals;
    - same-quarter ordered source signals where source_publish_date < target_publish_date;
    - target firm's own historical signal behaviour;
    - relationship and community context.

Inputs expected in --input-dir:
    prediction_dataset_cross_quarter.parquet
    prediction_dataset_same_quarter_ordered.parquet

Output:
    pre_call_target_signal_dataset.parquet
    pre_call_target_signal_dataset.csv
    pre_call_target_signal_dataset_summary.json

Run:
    python 03_build_pre_call_target_dataset.py ^
      --input-dir prediction_model_outputs ^
      --out-dir prediction_model_outputs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="prediction_model_outputs")
    p.add_argument("--out-dir", default="prediction_model_outputs")
    p.add_argument("--history-window", type=int, default=2)
    p.add_argument("--min-target-rows", type=int, default=1)
    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find any of columns: {candidates}")
    return None


def quarter_to_index(q: Any) -> float:
    m = re.match(r"^(\d{4})Q([1-4])$", str(q).strip())
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def index_to_quarter(idx: int | float) -> str:
    if pd.isna(idx):
        return ""
    idx = int(idx)
    year = (idx - 1) // 4
    q = idx - year * 4
    return f"{year}Q{q}"


def to_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def normalize_direction(x: Any) -> str:
    """
    Normalize LLM direction / label into:
        positive, negative, neutral, mixed, not_active
    """
    if x is None or pd.isna(x):
        return "not_active"

    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    if s in {"", "nan", "none", "null", "not_mentioned", "not mentioned", "notmentioned"}:
        return "not_active"

    # Mixed labels sometimes contain multiple tokens.
    if "mixed" in s:
        return "mixed"

    positive_keys = [
        "positive",
        "increase",
        "increasing",
        "improve",
        "improving",
        "improved",
        "growth",
        "strong",
        "favorable",
        "favourable",
        "up",
    ]
    negative_keys = [
        "negative",
        "decrease",
        "decreasing",
        "decline",
        "declining",
        "deteriorating",
        "worsening",
        "weak",
        "unfavorable",
        "unfavourable",
        "down",
        "pressure",
    ]
    neutral_keys = [
        "neutral",
        "stable",
        "flat",
        "unchanged",
        "steady",
    ]

    if any(k in s for k in positive_keys):
        return "positive"
    if any(k in s for k in negative_keys):
        return "negative"
    if any(k in s for k in neutral_keys):
        return "neutral"

    return "mixed"


def safe_divide(a: float, b: float) -> float:
    if b is None or b == 0 or pd.isna(b):
        return 0.0
    if pd.isna(a):
        return 0.0
    return float(a) / float(b)


def mode_or_first(values: pd.Series, default: str = "") -> str:
    values = values.dropna().astype(str)
    values = values[values.str.len() > 0]
    if len(values) == 0:
        return default
    vc = values.value_counts()
    return str(vc.index[0])


def first_existing_value(g: pd.DataFrame, col: str | None, default: Any = None) -> Any:
    if col is None or col not in g.columns:
        return default
    s = g[col].dropna()
    if len(s) == 0:
        return default
    return s.iloc[0]


def add_prefix_to_dict(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


# ============================================================
# Load event data
# ============================================================

def load_event_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross_path = input_dir / "prediction_dataset_cross_quarter.parquet"
    same_path = input_dir / "prediction_dataset_same_quarter_ordered.parquet"

    if not cross_path.exists():
        raise FileNotFoundError(f"Missing: {cross_path}")
    if not same_path.exists():
        raise FileNotFoundError(f"Missing: {same_path}")

    cross = pd.read_parquet(cross_path)
    same = pd.read_parquet(same_path)

    cross["input_source_sample"] = "cross_quarter_previous_quarter"
    same["input_source_sample"] = "same_quarter_ordered_pre_call"

    print(f"Loaded cross-quarter events       : rows={len(cross):,}, cols={len(cross.columns):,}")
    print(f"Loaded same-quarter ordered events: rows={len(same):,}, cols={len(same.columns):,}")

    return cross, same


# ============================================================
# Label construction
# ============================================================

def prepare_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    col_map = {
        "target_company": find_col(
            df,
            ["target_company_node", "target_node", "target_company", "target_ticker"],
        ),
        "source_company": find_col(
            df,
            ["source_company_node", "source_node", "source_company", "source_ticker"],
        ),
        "target_quarter": find_col(
            df,
            ["target_quarter", "q_prime", "target_q", "quarter_target"],
        ),
        "source_quarter": find_col(
            df,
            ["source_quarter", "source_q", "quarter_source"],
            required=False,
        ),
        "signal": find_col(df, ["signal", "signal_type", "outlook_signal"]),
        "relation_group": find_col(
            df,
            ["relation_group", "relationship_group", "relation_type", "relationship_type"],
            required=False,
        ),
        "source_label": find_col(
            df,
            ["source_label", "source_outlook_label", "source_signal_label"],
            required=False,
        ),
        "source_direction": find_col(
            df,
            ["source_direction", "source_polarity", "source_signal_direction"],
            required=False,
        ),
        "target_label": find_col(
            df,
            ["target_label", "target_outlook_label", "target_signal_label"],
            required=False,
        ),
        "target_direction": find_col(
            df,
            ["target_direction", "target_polarity", "target_signal_direction"],
            required=False,
        ),
        "source_active": find_col(
            df,
            ["source_active", "source_signal_active"],
            required=False,
        ),
        "target_active": find_col(
            df,
            ["target_active", "target_signal_active"],
            required=False,
        ),
        "source_community": find_col(
            df,
            ["source_community", "source_cluster", "source_community_id", "source_cluster_id"],
            required=False,
        ),
        "target_community": find_col(
            df,
            ["target_community", "target_cluster", "target_community_id", "target_cluster_id"],
            required=False,
        ),
    }

    out = pd.DataFrame(index=df.index)

    for new_col, old_col in col_map.items():
        if old_col is not None and old_col in df.columns:
            out[new_col] = df[old_col]
        else:
            out[new_col] = np.nan

    out["input_source_sample"] = df.get("input_source_sample", "")

    if col_map["source_active"] is not None:
        out["source_active_bool"] = to_bool(out["source_active"])
    else:
        # If not present, treat non-empty source direction/label as active.
        out["source_active_bool"] = (
            out["source_direction"].apply(normalize_direction).ne("not_active")
            | out["source_label"].apply(normalize_direction).ne("not_active")
        )

    if col_map["target_active"] is not None:
        out["target_active_bool"] = to_bool(out["target_active"])
    else:
        out["target_active_bool"] = (
            out["target_direction"].apply(normalize_direction).ne("not_active")
            | out["target_label"].apply(normalize_direction).ne("not_active")
        )

    # Direction normalization.
    out["source_direction_norm"] = out["source_direction"].apply(normalize_direction)
    missing_source_dir = out["source_direction_norm"].eq("not_active")
    out.loc[missing_source_dir, "source_direction_norm"] = (
        out.loc[missing_source_dir, "source_label"].apply(normalize_direction)
    )

    out["target_direction_norm"] = out["target_direction"].apply(normalize_direction)
    missing_target_dir = out["target_direction_norm"].eq("not_active")
    out.loc[missing_target_dir, "target_direction_norm"] = (
        out.loc[missing_target_dir, "target_label"].apply(normalize_direction)
    )

    out.loc[~out["source_active_bool"], "source_direction_norm"] = "not_active"
    out.loc[~out["target_active_bool"], "target_direction_norm"] = "not_active"

    out["target_quarter_index"] = out["target_quarter"].map(quarter_to_index)
    out["source_quarter_index"] = out["source_quarter"].map(quarter_to_index)

    out = out[
        out["target_company"].notna()
        & out["target_quarter"].notna()
        & out["signal"].notna()
        & out["target_quarter_index"].notna()
    ].copy()

    out["target_company"] = out["target_company"].astype(str)
    out["source_company"] = out["source_company"].astype(str)
    out["signal"] = out["signal"].astype(str)
    out["target_quarter"] = out["target_quarter"].astype(str)
    out["relation_group"] = out["relation_group"].fillna("unknown").astype(str)
    out["source_label"] = out["source_label"].fillna("unknown").astype(str)
    out["target_label"] = out["target_label"].fillna("unknown").astype(str)
    out["source_community"] = out["source_community"].fillna("unknown").astype(str)
    out["target_community"] = out["target_community"].fillna("unknown").astype(str)

    out["same_community"] = (
        out["source_community"].astype(str).eq(out["target_community"].astype(str))
        & out["source_community"].astype(str).ne("unknown")
    )

    return out


def build_target_labels(all_events: pd.DataFrame) -> pd.DataFrame:
    """
    Build one label row per target firm + target quarter + signal.

    Label:
        target_active_label
        target_direction_label
    """
    keys = ["target_company", "target_quarter", "signal"]

    rows = []

    for key, g in all_events.groupby(keys, dropna=False):
        target_company, target_quarter, signal = key

        active = bool(g["target_active_bool"].any())

        if active:
            active_dirs = g.loc[g["target_active_bool"], "target_direction_norm"]
            # Prefer the most frequent active direction.
            direction = mode_or_first(active_dirs, default="mixed")
        else:
            direction = "not_active"

        row = {
            "target_company": target_company,
            "target_quarter": target_quarter,
            "target_quarter_index": float(g["target_quarter_index"].iloc[0]),
            "signal": signal,
            "target_active_label": int(active),
            "target_direction_label": direction,
            "target_label_raw_mode": mode_or_first(g["target_label"], default="unknown"),
            "target_community": mode_or_first(g["target_community"], default="unknown"),
            "label_event_rows": int(len(g)),
        }

        rows.append(row)

    labels = pd.DataFrame(rows)

    direction_order = {
        "not_active": 0,
        "negative": 1,
        "neutral": 2,
        "mixed": 3,
        "positive": 4,
    }
    labels["target_direction_code"] = labels["target_direction_label"].map(direction_order).fillna(3).astype(int)

    return labels


# ============================================================
# Feature aggregation
# ============================================================

def aggregate_source_features(events: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = ["target_company", "target_quarter", "signal"]
    rows = []

    for key, g in events.groupby(keys, dropna=False):
        target_company, target_quarter, signal = key

        total = len(g)
        source_active = int(g["source_active_bool"].sum())
        unique_sources = g["source_company"].nunique(dropna=True)

        direction_counts = g.loc[g["source_active_bool"], "source_direction_norm"].value_counts()
        relation_counts = g["relation_group"].value_counts()
        relation_active = g.loc[g["source_active_bool"], "relation_group"].value_counts()

        row = {
            "target_company": target_company,
            "target_quarter": target_quarter,
            "signal": signal,

            "event_count": int(total),
            "unique_source_count": int(unique_sources),
            "source_active_count": int(source_active),
            "source_active_rate": safe_divide(source_active, total),

            "same_community_event_count": int(g["same_community"].sum()),
            "same_community_event_rate": safe_divide(int(g["same_community"].sum()), total),
            "same_community_source_active_count": int(
                (g["same_community"] & g["source_active_bool"]).sum()
            ),
            "same_community_source_active_rate": safe_divide(
                int((g["same_community"] & g["source_active_bool"]).sum()),
                total,
            ),
        }

        for d in ["positive", "negative", "neutral", "mixed", "not_active"]:
            cnt = int(direction_counts.get(d, 0))
            row[f"source_dir_{d}_count"] = cnt
            row[f"source_dir_{d}_share_all"] = safe_divide(cnt, total)
            row[f"source_dir_{d}_share_active"] = safe_divide(cnt, source_active)

        # Major relationship groups seen in this project.
        major_relations = [
            "upstream",
            "downstream",
            "partner",
            "competitor",
            "customer",
            "customer_group",
            "supplier_group",
            "parent",
            "subsidiary",
            "acquirer",
            "acquired_company",
            "related",
            "investment",
            "international",
            "internal",
        ]

        for rel in major_relations:
            total_rel = int(relation_counts.get(rel, 0))
            active_rel = int(relation_active.get(rel, 0))
            row[f"rel_{rel}_count"] = total_rel
            row[f"rel_{rel}_share"] = safe_divide(total_rel, total)
            row[f"rel_{rel}_active_count"] = active_rel
            row[f"rel_{rel}_active_share"] = safe_divide(active_rel, source_active)

        # Concentration indicators.
        row["top_relation_group"] = mode_or_first(g["relation_group"], default="unknown")
        row["top_source_direction"] = mode_or_first(
            g.loc[g["source_active_bool"], "source_direction_norm"],
            default="not_active",
        )
        row["top_source_label"] = mode_or_first(
            g.loc[g["source_active_bool"], "source_label"],
            default="unknown",
        )

        rows.append(row)

    feat = pd.DataFrame(rows)
    key_cols = ["target_company", "target_quarter", "signal"]
    rename_cols = {
        c: f"{prefix}{c}"
        for c in feat.columns
        if c not in key_cols
    }
    feat = feat.rename(columns=rename_cols)

    return feat


def add_target_history_features(df: pd.DataFrame, history_window: int) -> pd.DataFrame:
    """
    Add target's own previous-quarter behaviour for each signal.
    """
    df = df.copy()
    df = df.sort_values(["target_company", "signal", "target_quarter_index"]).reset_index(drop=True)

    group_cols = ["target_company", "signal"]

    for lag in range(1, history_window + 1):
        df[f"hist_lag{lag}_active"] = (
            df.groupby(group_cols)["target_active_label"].shift(lag).fillna(0).astype(float)
        )
        df[f"hist_lag{lag}_direction_code"] = (
            df.groupby(group_cols)["target_direction_code"].shift(lag).fillna(0).astype(float)
        )

    hist_active_cols = [f"hist_lag{lag}_active" for lag in range(1, history_window + 1)]
    hist_dir_cols = [f"hist_lag{lag}_direction_code" for lag in range(1, history_window + 1)]

    df[f"hist_past{history_window}_active_rate"] = df[hist_active_cols].mean(axis=1)
    df[f"hist_past{history_window}_mean_direction_code"] = df[hist_dir_cols].mean(axis=1)

    # Previous quarter same direction flags.
    for direction in ["positive", "negative", "neutral", "mixed"]:
        code = {
            "negative": 1,
            "neutral": 2,
            "mixed": 3,
            "positive": 4,
        }[direction]
        for lag in range(1, history_window + 1):
            df[f"hist_lag{lag}_was_{direction}"] = (
                df[f"hist_lag{lag}_direction_code"].eq(code).astype(float)
            )

    return df


# ============================================================
# Main build
# ============================================================

def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cross_raw, same_raw = load_event_data(input_dir)

    cross = prepare_common_columns(cross_raw)
    same = prepare_common_columns(same_raw)

    print(f"Prepared cross-quarter events       : rows={len(cross):,}")
    print(f"Prepared same-quarter ordered events: rows={len(same):,}")

    all_events = pd.concat([cross, same], ignore_index=True)

    labels = build_target_labels(all_events)
    labels = labels[labels["label_event_rows"] >= args.min_target_rows].copy()

    cross_features = aggregate_source_features(cross, prefix="prevq_")
    same_features = aggregate_source_features(same, prefix="sameq_pre_")

    dataset = labels.merge(
        cross_features,
        on=["target_company", "target_quarter", "signal"],
        how="left",
    )
    dataset = dataset.merge(
        same_features,
        on=["target_company", "target_quarter", "signal"],
        how="left",
    )

    dataset = add_target_history_features(dataset, history_window=args.history_window)

    # Fill numeric and categorical feature missing values.
    for col in dataset.columns:
        if col in {
            "target_company",
            "target_quarter",
            "signal",
            "target_direction_label",
            "target_label_raw_mode",
            "target_community",
        }:
            dataset[col] = dataset[col].fillna("unknown").astype(str)
        elif pd.api.types.is_numeric_dtype(dataset[col]):
            dataset[col] = dataset[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            dataset[col] = dataset[col].fillna("unknown").astype(str)

    dataset = dataset.sort_values(
        ["target_quarter_index", "target_company", "signal"]
    ).reset_index(drop=True)

    parquet_path = out_dir / "pre_call_target_signal_dataset.parquet"
    csv_path = out_dir / "pre_call_target_signal_dataset.csv"
    summary_path = out_dir / "pre_call_target_signal_dataset_summary.json"

    dataset.to_parquet(parquet_path, index=False)
    dataset.to_csv(csv_path, index=False)

    summary = {
        "objective": (
            "Predict target firm-quarter-signal content before the target earnings call, "
            "using previous-quarter connected source features, same-quarter ordered "
            "pre-call source features, community features, and target history."
        ),
        "unit": "target_company + target_quarter + signal",
        "rows": int(len(dataset)),
        "target_companies": int(dataset["target_company"].nunique()),
        "quarters": sorted(dataset["target_quarter"].unique().tolist()),
        "signals": sorted(dataset["signal"].unique().tolist()),
        "active_positive_rate": float(dataset["target_active_label"].mean()),
        "direction_distribution": dataset["target_direction_label"].value_counts().to_dict(),
        "history_window": args.history_window,
        "input_cross_rows": int(len(cross)),
        "input_same_ordered_rows": int(len(same)),
        "output_parquet": str(parquet_path),
        "output_csv": str(csv_path),
    }
    save_json(summary, summary_path)

    print("\nSAVED")
    print(f"  {parquet_path} rows={len(dataset):,}, cols={len(dataset.columns):,}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()