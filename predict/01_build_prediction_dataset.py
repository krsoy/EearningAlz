#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build event-level prediction datasets for the EarningALZ project.

Target:
    y = 1 if target_active is True and direction_match is True, else 0.

Cross-quarter objective:
    source signal in quarter t -> target same-direction signal in quarter t+1.

Same-quarter ordered objective:
    source publishes earlier in the same quarter -> later target same-direction signal.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--two-part-dataset", default="soysouce/earningALZ_twopart")
    p.add_argument("--evidence-dataset", default="soysouce/earningALZ_SBERT_evidence")
    p.add_argument("--revision", default="main")
    p.add_argument("--cross-file", default="cross_quarter_events.parquet")
    p.add_argument("--same-file", default="same_quarter_events.parquet")
    p.add_argument("--outlook-file", default="cleaned_outlook_all.parquet")
    p.add_argument("--metadata-file", default="rag_evidence_package_metadata_full_gpu_direct.parquet")
    p.add_argument("--community-file", default="", help="Optional community assignment file in HF dataset, or a local path if --community-local-file is not used.")
    p.add_argument("--community-local-file", default="", help="Local CSV/Parquet cluster assignment file, e.g. cluster_method_comparison_v4/best_company_cluster_assignment.csv.")
    p.add_argument("--community-dataset", default="", help="Defaults to --two-part-dataset if empty.")
    p.add_argument("--out-dir", default="prediction_model_outputs")
    p.add_argument("--history-window", type=int, default=2)
    p.add_argument("--same-quarter-include-unordered", action="store_true")
    p.add_argument("--include-inactive-source", action="store_true")
    p.add_argument("--write-csv-copy", action="store_true")
    return p.parse_args()


def list_parquets(repo_id: str, revision: str) -> list[str]:
    return sorted([f for f in list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision) if f.endswith(".parquet")])


def find_hf_file(repo_id: str, requested: str, revision: str) -> str:
    files = list_parquets(repo_id, revision)
    if requested in files:
        return requested
    base = Path(requested).name
    for f in files:
        if Path(f).name == base:
            return f
    stem = Path(requested).stem
    for f in files:
        if stem in Path(f).stem:
            return f
    raise FileNotFoundError(f"Cannot find {requested} in {repo_id}. Available parquet files: {files[:100]}")


def read_hf_parquet(repo_id: str, filename: str, revision: str) -> pd.DataFrame:
    remote = find_hf_file(repo_id, filename, revision)
    path = hf_hub_download(repo_id=repo_id, filename=remote, repo_type="dataset", revision=revision)
    df = pd.read_parquet(path)
    print(f"Loaded {repo_id}/{remote}: rows={len(df):,}, cols={len(df.columns):,}")
    return df


def quarter_to_index(q: str) -> float:
    m = re.match(r"^(\d{4})Q([1-4])$", str(q).strip())
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def index_to_quarter(idx: int) -> str:
    year = (idx - 1) // 4
    q = idx - year * 4
    return f"{year}Q{q}"


def previous_quarter(q: str, n: int = 1) -> str:
    idx = quarter_to_index(q)
    if pd.isna(idx):
        return ""
    return index_to_quarter(int(idx) - n)


def normalize_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def clean_node(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "0"}:
        return ""
    return s


def node_to_ticker(node: str) -> str:
    node = clean_node(node)
    if node.startswith("COMPANY::"):
        return node.replace("COMPANY::", "").strip().upper()
    return node.strip().upper()


def direction_score(value: str) -> int:
    v = str(value).strip().lower()
    if v in {"positive", "increase", "increasing", "improving", "growth", "up", "strong", "higher"}:
        return 1
    if v in {"negative", "decrease", "decreasing", "deteriorating", "worsening", "down", "weak", "lower"}:
        return -1
    return 0


def safe_divide(a, b):
    """
    Safe division for pandas Series / numpy arrays / scalars.
    Returns 0 when denominator is 0, NaN, or infinite.
    Avoids RuntimeWarning because np.divide uses where=.
    """
    index = None

    if isinstance(a, pd.Series):
        index = a.index
        a_arr = pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")
    else:
        a_arr = np.asarray(a, dtype="float64")

    if isinstance(b, pd.Series):
        if index is None:
            index = b.index
        b_arr = pd.to_numeric(b, errors="coerce").to_numpy(dtype="float64")
    else:
        b_arr = np.asarray(b, dtype="float64")

    out = np.zeros(np.broadcast_shapes(a_arr.shape, b_arr.shape), dtype="float64")

    valid = (
        np.isfinite(a_arr)
        & np.isfinite(b_arr)
        & (b_arr != 0)
    )

    np.divide(a_arr, b_arr, out=out, where=valid)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    if index is not None and out.shape[0] == len(index):
        return pd.Series(out, index=index)

    return out


def prepare_events(cross: pd.DataFrame, same: pd.DataFrame) -> pd.DataFrame:
    cross = cross.copy(); same = same.copy()
    if "analysis_mode" not in cross.columns:
        cross["analysis_mode"] = "cross_quarter"
    if "analysis_mode" not in same.columns:
        same["analysis_mode"] = "same_quarter"
    events = pd.concat([cross, same], ignore_index=True, sort=False)
    for c in ["analysis_mode", "source_node", "target_node", "source_quarter", "target_quarter", "signal", "relation_group", "source_direction", "target_direction", "source_label", "target_label"]:
        if c not in events.columns:
            events[c] = ""
        events[c] = events[c].astype(str).str.strip()
    for c in ["source_active", "target_active", "direction_match", "exact_match"]:
        events[c] = normalize_bool(events[c]) if c in events.columns else False
    events["source_ticker"] = events["source_node"].map(node_to_ticker)
    events["target_ticker"] = events["target_node"].map(node_to_ticker)
    events["source_direction_score"] = events["source_direction"].map(direction_score)
    events["target_direction_score"] = events["target_direction"].map(direction_score)
    events["y"] = (events["target_active"] & events["direction_match"]).astype(int)
    events["source_q_idx"] = events["source_quarter"].map(quarter_to_index)
    events["target_q_idx"] = events["target_quarter"].map(quarter_to_index)
    events["target_q_num"] = events["target_quarter"].str.extract(r"Q([1-4])")[0].astype("float")
    events["source_target_quarter_gap"] = events["target_q_idx"] - events["source_q_idx"]
    events["event_id"] = np.arange(len(events))
    return events


def build_publish_dates(metadata: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.copy()
    required = {"publish_date", "ticker", "quarter"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()
    meta["company_node"] = "COMPANY::" + meta["ticker"]
    meta["quarter"] = meta["quarter"].astype(str).str.strip()
    meta["publish_date"] = pd.to_datetime(meta["publish_date"], errors="coerce")
    out = meta.dropna(subset=["publish_date"]).groupby(["company_node", "ticker", "quarter"], as_index=False).agg(
        publish_date=("publish_date", "min"), publish_date_count=("publish_date", "count")
    )
    print(f"Built publish-date table: rows={len(out):,}, companies={out['ticker'].nunique():,}")
    return out


def attach_publish_dates(events: pd.DataFrame, publish_dates: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    src = publish_dates.rename(columns={"company_node": "source_node", "quarter": "source_quarter", "publish_date": "source_publish_date", "publish_date_count": "source_publish_date_count"})
    tgt = publish_dates.rename(columns={"company_node": "target_node", "quarter": "target_quarter", "publish_date": "target_publish_date", "publish_date_count": "target_publish_date_count"})
    out = out.merge(src[["source_node", "source_quarter", "source_publish_date", "source_publish_date_count"]], on=["source_node", "source_quarter"], how="left")
    out = out.merge(tgt[["target_node", "target_quarter", "target_publish_date", "target_publish_date_count"]], on=["target_node", "target_quarter"], how="left")
    out["source_publish_date"] = pd.to_datetime(out["source_publish_date"], errors="coerce")
    out["target_publish_date"] = pd.to_datetime(out["target_publish_date"], errors="coerce")
    out["publish_gap_days"] = (out["target_publish_date"] - out["source_publish_date"]).dt.days
    out["source_before_target"] = out["publish_gap_days"] > 0
    print("Publish-date join coverage:", f"source={out['source_publish_date'].notna().mean():.2%}", f"target={out['target_publish_date'].notna().mean():.2%}")
    return out


def normalize_outlook(outlook: pd.DataFrame) -> pd.DataFrame:
    df = outlook.copy()
    if "company_node" not in df.columns:
        if "ticker" in df.columns:
            df["company_node"] = "COMPANY::" + df["ticker"].astype(str).str.strip().str.upper()
        else:
            raise ValueError("cleaned_outlook_all must contain company_node or ticker")
    if "ticker" not in df.columns:
        df["ticker"] = df["company_node"].map(node_to_ticker)
    if "quarter" not in df.columns:
        for c in ["source_quarter", "target_quarter", "fiscal_quarter"]:
            if c in df.columns:
                df["quarter"] = df[c]; break
    if "signal" not in df.columns:
        raise ValueError("cleaned_outlook_all must contain signal")
    if "direction" not in df.columns:
        if "label" in df.columns:
            df["direction"] = df["label"]
        elif "outlook_label" in df.columns:
            df["direction"] = df["outlook_label"]
        else:
            df["direction"] = ""
    if "label" not in df.columns:
        df["label"] = df["direction"]
    if "active" not in df.columns:
        df["active"] = df["direction"].astype(str).str.lower().ne("not_mentioned") & df["direction"].astype(str).str.strip().ne("")
    df["company_node"] = df["company_node"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["quarter"] = df["quarter"].astype(str).str.strip()
    df["signal"] = df["signal"].astype(str).str.strip()
    df["direction"] = df["direction"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["active"] = normalize_bool(df["active"]) if df["active"].dtype != bool else df["active"]
    df["direction_score"] = df["direction"].map(direction_score)
    return df.groupby(["company_node", "ticker", "quarter", "signal"], as_index=False).agg(
        active=("active", "max"),
        direction_score=("direction_score", lambda x: int(np.sign(np.sum(x)))),
        direction=("direction", lambda x: next((str(v) for v in x if str(v) not in ["", "not_mentioned"]), "")),
        label=("label", lambda x: next((str(v) for v in x if str(v) not in ["", "not_mentioned"]), "")),
    )


def attach_target_history(events: pd.DataFrame, outlook: pd.DataFrame, history_window: int) -> pd.DataFrame:
    out = events.copy(); ol = normalize_outlook(outlook)
    for k in range(1, history_window + 1):
        hist = ol.rename(columns={"company_node": "target_node", "quarter": f"target_prev_q{k}", "active": f"target_prev{k}_signal_active", "direction_score": f"target_prev{k}_direction_score", "direction": f"target_prev{k}_direction", "label": f"target_prev{k}_label"})
        out[f"target_prev_q{k}"] = out["target_quarter"].map(lambda q: previous_quarter(q, k))
        out = out.merge(hist[["target_node", f"target_prev_q{k}", "signal", f"target_prev{k}_signal_active", f"target_prev{k}_direction_score", f"target_prev{k}_direction", f"target_prev{k}_label"]], on=["target_node", f"target_prev_q{k}", "signal"], how="left")
        out[f"target_prev{k}_signal_active"] = out[f"target_prev{k}_signal_active"].fillna(False).astype(bool)
        out[f"target_prev{k}_direction_score"] = out[f"target_prev{k}_direction_score"].fillna(0).astype(int)
        out[f"target_prev{k}_same_direction_as_source"] = (out[f"target_prev{k}_direction_score"].eq(out["source_direction_score"]) & out[f"target_prev{k}_signal_active"]).astype(int)
    active_cols = [f"target_prev{k}_signal_active" for k in range(1, history_window + 1)]
    same_cols = [f"target_prev{k}_same_direction_as_source" for k in range(1, history_window + 1)]
    score_cols = [f"target_prev{k}_direction_score" for k in range(1, history_window + 1)]
    out["target_signal_active_count_last_kq"] = out[active_cols].sum(axis=1)
    out["target_same_direction_count_last_kq"] = out[same_cols].sum(axis=1)
    out["target_direction_score_mean_last_kq"] = out[score_cols].mean(axis=1)
    return out


def attach_edge_and_degree_features(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    stats = out.groupby(["source_node", "target_node", "relation_group"], as_index=False).agg(
        edge_event_count=("event_id", "count"), edge_source_active_rate=("source_active", "mean"), edge_success_rate_historical=("y", "mean")
    )
    pair = out.groupby(["source_node", "target_node"], as_index=False).agg(pair_event_count=("event_id", "count"), pair_relation_count=("relation_group", "nunique"))
    out = out.merge(stats.merge(pair, on=["source_node", "target_node"], how="left"), on=["source_node", "target_node", "relation_group"], how="left")
    edges = out[["source_node", "target_node"]].drop_duplicates()
    src_deg = edges.groupby("source_node")["target_node"].nunique().rename("source_out_degree").reset_index()
    tgt_deg = edges.groupby("target_node")["source_node"].nunique().rename("target_in_degree").reset_index()
    out = out.merge(src_deg, on="source_node", how="left").merge(tgt_deg, on="target_node", how="left")
    return out


def attach_neighbor_aggregates(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["source_positive"] = (df["source_direction_score"] > 0).astype(int)
    df["source_negative"] = (df["source_direction_score"] < 0).astype(int)
    df["source_mixed_or_neutral"] = (df["source_direction_score"] == 0).astype(int)
    df["source_active_int"] = df["source_active"].astype(int)
    group_cols = ["target_node", "target_quarter", "signal"]
    agg = df.groupby(group_cols, as_index=False).agg(
        neighbor_event_count=("event_id", "count"), neighbor_active_count=("source_active_int", "sum"),
        neighbor_positive_count=("source_positive", "sum"), neighbor_negative_count=("source_negative", "sum"),
        neighbor_mixed_neutral_count=("source_mixed_or_neutral", "sum"), neighbor_signal_score_mean=("source_direction_score", "mean")
    )
    agg["neighbor_positive_share"] = safe_divide(agg["neighbor_positive_count"], agg["neighbor_active_count"])
    agg["neighbor_negative_share"] = safe_divide(agg["neighbor_negative_count"], agg["neighbor_active_count"])
    agg["neighbor_signal_balance"] = agg["neighbor_positive_share"] - agg["neighbor_negative_share"]
    out = df.merge(agg, on=group_cols, how="left")
    rel_agg = df.groupby(group_cols + ["relation_group"], as_index=False).agg(
        rel_neighbor_active_count=("source_active_int", "sum"), rel_neighbor_positive_count=("source_positive", "sum"), rel_neighbor_negative_count=("source_negative", "sum")
    )
    for rel in sorted(df["relation_group"].dropna().unique()):
        sub = rel_agg[rel_agg["relation_group"].eq(rel)].drop(columns=["relation_group"]).copy()
        prefix = re.sub(r"[^a-zA-Z0-9]+", "_", str(rel)).strip("_").lower()
        sub = sub.rename(columns={"rel_neighbor_active_count": f"{prefix}_neighbor_active_count", "rel_neighbor_positive_count": f"{prefix}_neighbor_positive_count", "rel_neighbor_negative_count": f"{prefix}_neighbor_negative_count"})
        out = out.merge(sub, on=group_cols, how="left")
    return out


def auto_detect_community_file(repo_id: str, revision: str) -> str:
    files = list_parquets(repo_id, revision)
    candidates = [
        f for f in files
        if any(k in Path(f).name.lower() for k in [
            "community", "cluster", "graph_greedy", "company_cluster",
            "firm_cluster", "assignment"
        ])
    ]
    if not candidates:
        return ""
    preferred = [
        f for f in candidates
        if any(k in Path(f).name.lower() for k in [
            "best_company_cluster_assignment", "assignment", "company", "node"
        ])
    ]
    return (preferred or candidates)[0]


def read_local_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local community file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(p)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"Unsupported local community file type: {p}")


def normalize_company_node_from_any(df: pd.DataFrame) -> pd.Series:
    d = df.copy()

    # Direct company node columns from clustering output.
    for c in [
        "company_node", "node", "firm_node", "company_id",
        "source_node", "target_node"
    ]:
        if c in d.columns:
            s = d[c].astype(str).str.strip()
            return np.where(
                s.str.startswith("COMPANY::"),
                s,
                "COMPANY::" + s.str.replace(r"^COMPANY::", "", regex=True).str.upper()
            )

    # Ticker columns.
    for c in [
        "ticker", "symbol", "company_ticker", "firm_ticker"
    ]:
        if c in d.columns:
            ticker = d[c].astype(str).str.strip().str.upper()
            ticker = ticker.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "NULL": np.nan})
            return "COMPANY::" + ticker.fillna("")

    # Last fallback: company name. This only works if events also use name-based nodes.
    for c in ["company", "current_company", "company_name", "firm_name", "name"]:
        if c in d.columns:
            cleaned = (
                d[c].astype(str).str.lower()
                .str.replace(r"[^a-z0-9&.\- ]+", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            return "COMPANY::" + cleaned

    raise ValueError(
        "Community/cluster file must contain a company identifier column. "
        "Accepted examples: company_node, node, ticker, symbol, company, current_company."
    )


def normalize_community(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a clustering output into:
        company_node
        community_id
        community_label, if available

    This supports the V4 clustering output:
        best_company_cluster_assignment.csv
    whose core columns are:
        company_node, ticker, company, cluster_id, cluster_theme_label.
    """
    d = df.copy()
    d["company_node"] = normalize_company_node_from_any(d)
    d["company_node"] = d["company_node"].astype(str).str.strip()
    d = d[d["company_node"].ne("COMPANY::") & d["company_node"].ne("")].copy()

    # Cluster/community id columns. V4 uses cluster_id.
    if "community_id" not in d.columns:
        for c in [
            "cluster_id", "best_cluster_id", "selected_cluster_id",
            "graph_greedy_rel", "graph_greedy_modularity",
            "community", "cluster", "label"
        ]:
            if c in d.columns:
                d["community_id"] = d[c]
                break

    if "community_id" not in d.columns:
        raise ValueError(
            "Community/cluster file must contain community_id or cluster_id. "
            "For V4 clustering, use best_company_cluster_assignment.csv."
        )

    # Label columns. V4 uses cluster_theme_label.
    if "community_label" not in d.columns:
        for c in [
            "cluster_theme_label", "community_label", "theme_label",
            "cluster_label", "industry_label", "label_name"
        ]:
            if c in d.columns:
                d["community_label"] = d[c]
                break
    if "community_label" not in d.columns:
        d["community_label"] = ""

    out = d[["company_node", "community_id", "community_label"]].drop_duplicates("company_node").copy()
    out["community_id"] = out["community_id"].astype(str)
    out["community_label"] = out["community_label"].astype(str)

    print(
        "Normalized community assignment:",
        f"rows={len(out):,}",
        f"communities={out['community_id'].nunique():,}",
        f"matched-company-node-sample={out['company_node'].head(3).tolist()}",
    )
    return out


def load_community_assignment(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """
    Load community assignment either from a local file or from HF.

    Priority:
    1. --community-local-file
    2. --community-file as a local existing path
    3. --community-file from HF dataset
    4. auto-detected HF community/cluster parquet
    """
    if args.community_local_file:
        df = read_local_table(args.community_local_file)
        return normalize_community(df), f"local:{args.community_local_file}"

    if args.community_file and Path(args.community_file).exists():
        df = read_local_table(args.community_file)
        return normalize_community(df), f"local:{args.community_file}"

    repo = args.community_dataset or args.two_part_dataset
    community_file = args.community_file or auto_detect_community_file(repo, args.revision)
    if not community_file:
        return pd.DataFrame(), ""

    df = read_hf_parquet(repo, community_file, args.revision)
    return normalize_community(df), f"hf:{repo}/{community_file}"


def attach_community_features(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = events.copy()

    try:
        community, source_desc = load_community_assignment(args)
    except Exception as exc:
        print(f"Warning: failed to load community/cluster assignment: {exc}")
        community = pd.DataFrame()
        source_desc = ""

    if community.empty:
        print("No community file detected. Community features use 'unknown'.")
        out["source_community_id"] = "unknown"
        out["target_community_id"] = "unknown"
        out["source_community_label"] = ""
        out["target_community_label"] = ""
        out["same_community"] = 0
        return out

    print(f"Using community assignment: {source_desc}")

    src = community.rename(columns={
        "company_node": "source_node",
        "community_id": "source_community_id",
        "community_label": "source_community_label",
    })
    tgt = community.rename(columns={
        "company_node": "target_node",
        "community_id": "target_community_id",
        "community_label": "target_community_label",
    })

    out = out.merge(src, on="source_node", how="left")
    out = out.merge(tgt, on="target_node", how="left")

    out["source_community_id"] = out["source_community_id"].fillna("unknown")
    out["target_community_id"] = out["target_community_id"].fillna("unknown")
    out["source_community_label"] = out["source_community_label"].fillna("")
    out["target_community_label"] = out["target_community_label"].fillna("")

    out["same_community"] = (
        out["source_community_id"].eq(out["target_community_id"])
        & out["source_community_id"].ne("unknown")
        & out["source_community_id"].ne("-1")
    ).astype(int)

    source_cov = out["source_community_id"].ne("unknown").mean() if len(out) else 0
    target_cov = out["target_community_id"].ne("unknown").mean() if len(out) else 0
    print(f"Community join coverage: source={source_cov:.2%}, target={target_cov:.2%}")

    tmp = out.copy()
    tmp["source_active_int"] = tmp["source_active"].astype(int)
    tmp["source_positive"] = (tmp["source_direction_score"] > 0).astype(int)
    tmp["source_negative"] = (tmp["source_direction_score"] < 0).astype(int)

    comm_agg = tmp.groupby(["target_community_id", "target_quarter", "signal"], as_index=False).agg(
        target_community_event_count=("event_id", "count"),
        target_community_active_count=("source_active_int", "sum"),
        target_community_positive_count=("source_positive", "sum"),
        target_community_negative_count=("source_negative", "sum"),
        target_community_signal_score_mean=("source_direction_score", "mean"),
    )
    comm_agg["target_community_positive_share"] = safe_divide(
        comm_agg["target_community_positive_count"],
        comm_agg["target_community_active_count"],
    )
    comm_agg["target_community_negative_share"] = safe_divide(
        comm_agg["target_community_negative_count"],
        comm_agg["target_community_active_count"],
    )
    comm_agg["target_community_signal_balance"] = (
        comm_agg["target_community_positive_share"]
        - comm_agg["target_community_negative_share"]
    )

    return out.merge(comm_agg, on=["target_community_id", "target_quarter", "signal"], how="left")



def final_clean(df: pd.DataFrame, mode: str, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if not args.include_inactive_source:
        out = out[out["source_active"]].copy()
    if mode == "same_quarter" and not args.same_quarter_include_unordered:
        out = out[out["source_before_target"].fillna(False)].copy()
    dedup_cols = ["analysis_mode", "source_node", "target_node", "source_quarter", "target_quarter", "signal", "relation_group", "source_direction", "target_direction"]
    out = out.drop_duplicates([c for c in dedup_cols if c in out.columns]).reset_index(drop=True)
    out["target_q_idx"] = out["target_quarter"].map(quarter_to_index)
    out["target_q_num"] = out["target_quarter"].str.extract(r"Q([1-4])")[0].astype(float)
    out["source_target_pair"] = out["source_node"] + "→" + out["target_node"]
    out["community_pair"] = out.get("source_community_id", "unknown").astype(str) + "→" + out.get("target_community_id", "unknown").astype(str)
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = out[c].fillna(0)
        elif out[c].dtype == object:
            out[c] = out[c].fillna("")
    return out


def save_table(df: pd.DataFrame, path: Path, write_csv: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"SAVED {path} rows={len(df):,}, cols={len(df.columns):,}")
    if write_csv:
        df.to_csv(path.with_suffix(".csv"), index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cross = read_hf_parquet(args.two_part_dataset, args.cross_file, args.revision)
    same = read_hf_parquet(args.two_part_dataset, args.same_file, args.revision)
    outlook = read_hf_parquet(args.two_part_dataset, args.outlook_file, args.revision)
    metadata = read_hf_parquet(args.evidence_dataset, args.metadata_file, args.revision)
    events = prepare_events(cross, same)
    events = attach_publish_dates(events, build_publish_dates(metadata))
    events = attach_target_history(events, outlook, args.history_window)
    events = attach_edge_and_degree_features(events)
    events = attach_neighbor_aggregates(events)
    events = attach_community_features(events, args)
    cross_model = final_clean(events[events["analysis_mode"].eq("cross_quarter")], "cross_quarter", args)
    same_model = final_clean(events[events["analysis_mode"].eq("same_quarter")], "same_quarter", args)
    save_table(cross_model, out_dir / "prediction_dataset_cross_quarter.parquet", args.write_csv_copy)
    save_table(same_model, out_dir / "prediction_dataset_same_quarter_ordered.parquet", args.write_csv_copy)
    meta = {
        "target": "y = 1 if target_active and direction_match, else 0",
        "cross_quarter_objective": "source signal at quarter t predicts target same-direction signal at quarter t+1",
        "same_quarter_objective": "earlier same-quarter source signal predicts later same-quarter target signal",
        "history_window": args.history_window,
        "same_ordered_only": not args.same_quarter_include_unordered,
        "community_local_file": args.community_local_file,
        "community_file": args.community_file,
        "community_dataset": args.community_dataset or args.two_part_dataset,
        "cross_rows": int(len(cross_model)),
        "same_rows": int(len(same_model)),
        "cross_positive_rate": float(cross_model["y"].mean()) if len(cross_model) else None,
        "same_positive_rate": float(same_model["y"].mean()) if len(same_model) else None,
    }
    (out_dir / "prediction_dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
