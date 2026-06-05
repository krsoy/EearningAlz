#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a lightweight JSON file for a JavaScript dynamic information-flow network.

V6 logic:
1. Cross-quarter flows are treated as quarter-level lead-lag flows.
   The source quarter precedes the target quarter by construction.
   Publish-date gap is not used as a diffusion-speed metric.

2. Same-quarter flows are treated as date-level within-quarter flows.
   publish_gap_days = target_publish_date - source_publish_date.
   source_before_target is meaningful only for same-quarter flows.

3. Edge summaries are grouped by actual quarter pair:
   source, target, signal, relation, direction, analysis_mode, source_quarter, target_quarter.

Default data sources:
    soysouce/earningALZ_twopart
    soysouce/earningALZ_SBERT_evidence

Example:
    python build_stock_network_json.py --ticker AAPL --mode cross_quarter --out data/aapl_network.json
    python build_stock_network_json.py --ticker AAPL --mode same_quarter --ordered-same-quarter-only --out data/aapl_network.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import networkx as nx
from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_TWO_PART_DATASET = "soysouce/earningALZ_twopart"
DEFAULT_EVIDENCE_DATASET = "soysouce/earningALZ_SBERT_evidence"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL", help="Ticker to center the network on, for example AAPL.")
    p.add_argument("--two-part-dataset", default=DEFAULT_TWO_PART_DATASET)
    p.add_argument("--two-part-prefix", default="")
    p.add_argument("--evidence-dataset", default=DEFAULT_EVIDENCE_DATASET)
    p.add_argument("--evidence-metadata-file", default="rag_evidence_package_metadata_full_gpu_direct.parquet")
    p.add_argument("--revision", default="main")
    p.add_argument("--mode", default="cross_quarter", choices=["cross_quarter", "same_quarter", "both"])
    p.add_argument("--signal", default="All", help="All or one signal, e.g. demand_outlook.")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--hop-depth", type=int, default=2)
    p.add_argument("--max-nodes", type=int, default=120)
    p.add_argument("--max-links", type=int, default=260)
    p.add_argument("--only-successful", action="store_true")
    p.add_argument("--ordered-same-quarter-only", action="store_true",
                   help="For same-quarter mode, keep only rows where source_publish_date is earlier than target_publish_date.")
    p.add_argument("--no-publish-dates", action="store_true", help="Disable publish_date lookup from evidence metadata.")
    p.add_argument("--chunk-index", default="", help="Path to local chunk_index.csv (from build_chroma_rag_index.py). "
                   "If provided, publish_dates are read from this file instead of the evidence metadata parquet. "
                   "Example: ../RAG/rag_chroma_output/chunk_index.csv")
    p.add_argument("--out", default="data/aapl_network.json")
    return p.parse_args()


def prefixed(prefix: str, filename: str) -> str:
    prefix = prefix.strip().strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def list_parquet_files(repo_id: str, revision: str) -> list[str]:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    return sorted([f for f in files if f.endswith(".parquet")])


def select_hf_file(repo_id: str, filename: str, prefix: str, revision: str) -> str:
    files = list_parquet_files(repo_id, revision)
    stem = Path(filename).stem

    candidates = [
        prefixed(prefix, filename),
        filename,
        prefixed(prefix, f"data/{filename}"),
        prefixed(prefix, f"results/{filename}"),
        prefixed(prefix, f"two_part_network_prediction_analysis_hf/{filename}"),
        prefixed(prefix, f"two_part_network_prediction_analysis_parquet/{filename}"),
    ]
    candidates += [f for f in files if Path(f).name == filename]
    candidates += [f for f in files if stem in Path(f).stem]

    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c in files:
            return c

    raise FileNotFoundError(f"Cannot find {filename} in {repo_id}. Available parquet files: {files[:80]}")


def read_hf_parquet(repo_id: str, filename: str, prefix: str = "", revision: str = "main") -> pd.DataFrame:
    remote_file = select_hf_file(repo_id, filename, prefix, revision)
    local_path = hf_hub_download(repo_id=repo_id, filename=remote_file, repo_type="dataset", revision=revision)
    df = pd.read_parquet(local_path)
    print(f"Loaded {repo_id}/{remote_file}: rows={len(df):,}, cols={len(df.columns):,}")
    return df


def quarter_to_index(q: str) -> float:
    m = re.match(r"^(\d{4})Q([1-4])$", str(q).strip())
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def sort_quarters(values: Iterable[str]) -> list[str]:
    return sorted([str(v) for v in values if not pd.isna(quarter_to_index(v))], key=quarter_to_index)


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


def ticker_from_node(node: str) -> str:
    node = str(node)
    if node.startswith("COMPANY::"):
        return node.replace("COMPANY::", "")
    return node


def company_node_from(ticker: str, company: str) -> str:
    ticker = clean_node(ticker)
    company = clean_node(company)
    if ticker:
        return "COMPANY::" + ticker.upper()
    cleaned = re.sub(r"[^a-z0-9&.\- ]+", " ", company.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return "COMPANY::" + cleaned


def build_publish_dates_from_chunk_index(chunk_index_path: str) -> pd.DataFrame:
    """
    Build publish-date lookup directly from chunk_index.csv (output of build_chroma_rag_index.py).
    This covers all quarters in the Chroma index without needing to rerun the RAG retrieve pipeline.

    Expected columns: ticker, company, publish_date, quarter  (one row per chunk)
    Returns: one row per (company_node, quarter) with the earliest publish_date.
    """
    path = Path(chunk_index_path)
    if not path.exists():
        raise FileNotFoundError(f"chunk_index not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded chunk_index: rows={len(df):,}, cols={len(df.columns):,}  path={path}")

    # Normalise column names – chunk_index uses 'company' not 'current_company'
    ticker_col = "ticker" if "ticker" in df.columns else None
    company_col = "company" if "company" in df.columns else "current_company" if "current_company" in df.columns else None

    if "publish_date" not in df.columns:
        raise ValueError("chunk_index.csv does not contain 'publish_date'.")
    if "quarter" not in df.columns:
        raise ValueError("chunk_index.csv does not contain 'quarter'.")
    if ticker_col is None:
        raise ValueError("chunk_index.csv does not contain 'ticker'.")

    d = df[[ticker_col, company_col or ticker_col, "publish_date", "quarter"]].copy()
    d.columns = ["ticker", "current_company", "publish_date", "quarter"]
    d["company_node"] = d.apply(
        lambda r: company_node_from(r.get("ticker", ""), r.get("current_company", "")),
        axis=1,
    )
    d["quarter"] = d["quarter"].astype(str).str.strip()
    d["publish_date"] = pd.to_datetime(d["publish_date"], errors="coerce", format="mixed")
    d = d[d["publish_date"].notna() & d["company_node"].ne("")].copy()

    out = d.groupby(["company_node", "quarter"], as_index=False).agg(
        publish_date=("publish_date", "min"),
        publish_date_count=("publish_date", "count"),
    )
    print(f"Built publish-date lookup from chunk_index: rows={len(out):,}, "
          f"quarters={sorted(out['quarter'].unique())[-5:]}")
    return out


def build_publish_dates(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(columns=["company_node", "quarter", "publish_date", "publish_date_count"])

    if "publish_date" not in metadata.columns:
        raise ValueError("Metadata does not contain publish_date.")

    if "ticker" not in metadata.columns:
        raise ValueError("Metadata does not contain ticker.")

    if "quarter" not in metadata.columns:
        raise ValueError("Metadata does not contain quarter.")

    d = metadata.copy()
    d["company_node"] = d.apply(
        lambda r: company_node_from(r.get("ticker", ""), r.get("current_company", "")),
        axis=1,
    )
    d["quarter"] = d["quarter"].astype(str).str.strip()
    d["publish_date"] = pd.to_datetime(d["publish_date"], errors="coerce")
    d = d[d["publish_date"].notna() & d["company_node"].ne("")].copy()

    out = d.groupby(["company_node", "quarter"], as_index=False).agg(
        publish_date=("publish_date", "min"),
        publish_date_count=("publish_date", "count"),
    )
    print(f"Built publish-date lookup: rows={len(out):,}")
    return out


def attach_publish_dates(events: pd.DataFrame, publish_dates: pd.DataFrame) -> pd.DataFrame:
    if events.empty or publish_dates.empty:
        events = events.copy()
        events["source_publish_date"] = pd.NaT
        events["target_publish_date"] = pd.NaT
        events["publish_gap_days"] = np.nan
        events["source_before_target"] = False
        return events

    out = events.copy()
    d = publish_dates.copy()
    d["publish_date"] = pd.to_datetime(d["publish_date"], errors="coerce", format="mixed")
    # count missing value in publish_dates

    src = d.rename(columns={
        "company_node": "source_node",
        "quarter": "source_quarter",
        "publish_date": "source_publish_date",
        "publish_date_count": "source_publish_date_count",
    })
    tgt = d.rename(columns={
        "company_node": "target_node",
        "quarter": "target_quarter",
        "publish_date": "target_publish_date",
        "publish_date_count": "target_publish_date_count",
    })

    print("=== publish_dates sample ===")
    print(publish_dates[["company_node", "quarter"]].head(10).to_string())

    print("=== events source_node / source_quarter sample ===")
    print(out[["source_node", "source_quarter"]].drop_duplicates().head(10).to_string())

    print("=== events target_node / target_quarter sample ===")
    print(out[["target_node", "target_quarter"]].drop_duplicates().head(10).to_string())

    # 检查有多少能匹配上
    src_keys = set(zip(publish_dates["company_node"], publish_dates["quarter"]))
    out_src_keys = set(zip(out["source_node"], out["source_quarter"]))
    print(f"Source keys overlap: {len(src_keys & out_src_keys)} / {len(out_src_keys)}")

    print('src source publish date null count', src["source_publish_date"].isna().sum())
    print('src target_publish date null count', tgt['target_publish_date'].isna().sum())
    out = out.merge(
        src[["source_node", "source_quarter", "source_publish_date", "source_publish_date_count"]],
        on=["source_node", "source_quarter"],
        how="left",
    )
    out = out.merge(
        tgt[["target_node", "target_quarter", "target_publish_date", "target_publish_date_count"]],
        on=["target_node", "target_quarter"],
        how="left",
    )
    print('source publish date null count', out["source_publish_date"].isna().sum())
    print('target_publish date null count', out['target_publish_date'].isna().sum())
    out["source_publish_date"] = pd.to_datetime(out["source_publish_date"], errors="coerce", format="mixed")
    out["target_publish_date"] = pd.to_datetime(out["target_publish_date"], errors="coerce", format="mixed")
    print('source publish date null count', out["source_publish_date"].isna().sum())
    print('target_publish date null count', out['target_publish_date'].isna().sum())
    null_mask = out["source_publish_date"].isna()
    print("=== Null source_publish_date by quarter ===")
    print(out[null_mask]["source_quarter"].value_counts().sort_index().head(20))

    print("=== Null source_publish_date top companies ===")
    print(out[null_mask]["source_node"].value_counts().head(20))

    out["publish_gap_days"] = (out["target_publish_date"] - out["source_publish_date"]).dt.days
    out["source_before_target"] = out["publish_gap_days"] > 0

    source_coverage = out["source_publish_date"].notna().mean() if len(out) else 0
    target_coverage = out["target_publish_date"].notna().mean() if len(out) else 0
    print(f"Publish-date join coverage: source={source_coverage:.2%}, target={target_coverage:.2%}")

    return out


def prepare_events(cross: pd.DataFrame, same: pd.DataFrame) -> pd.DataFrame:
    cross = cross.copy()
    same = same.copy()

    if "analysis_mode" not in cross.columns:
        cross["analysis_mode"] = "cross_quarter"
    if "analysis_mode" not in same.columns:
        same["analysis_mode"] = "same_quarter"

    events = pd.concat([cross, same], ignore_index=True, sort=False)

    for c in ["source_active", "target_active", "direction_match", "exact_match"]:
        if c in events.columns:
            events[c] = normalize_bool(events[c])
        else:
            events[c] = False

    for c in [
        "analysis_mode", "source_node", "target_node", "source_quarter", "target_quarter",
        "signal", "relation_group", "source_direction", "target_direction",
        "source_label", "target_label",
    ]:
        if c not in events.columns:
            events[c] = ""
        events[c] = events[c].astype(str).str.strip()

    events["success"] = events["source_active"] & events["target_active"] & events["direction_match"]
    events["window"] = events["source_quarter"] + "→" + events["target_quarter"]

    return events


def filter_quarter_range(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    sidx = quarter_to_index(start_q) if start_q else None
    eidx = quarter_to_index(end_q) if end_q else None

    mask = pd.Series(True, index=out.index)
    for c in ["source_quarter", "target_quarter"]:
        idx = out[c].map(quarter_to_index)
        if sidx is not None:
            mask &= idx >= sidx
        if eidx is not None:
            mask &= idx <= eidx
    return out[mask].copy()


def build_company_table(outlook: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if {"company_node", "ticker", "current_company"}.issubset(outlook.columns):
        x = outlook[["company_node", "ticker", "current_company"]].drop_duplicates()
        x = x.rename(columns={"current_company": "company"})
        rows.append(x)

    for side in ["source", "target"]:
        node_col = f"{side}_node"
        ticker_col = f"{side}_ticker"
        company_col = f"{side}_company"
        if node_col in events.columns:
            cols = [node_col]
            if ticker_col in events.columns:
                cols.append(ticker_col)
            if company_col in events.columns:
                cols.append(company_col)
            x = events[cols].drop_duplicates()
            x = x.rename(columns={node_col: "company_node", ticker_col: "ticker", company_col: "company"})
            rows.append(x)

    if not rows:
        nodes = sorted(set(events["source_node"]) | set(events["target_node"]))
        return pd.DataFrame({"company_node": nodes, "ticker": [ticker_from_node(n) for n in nodes], "company": nodes})

    company = pd.concat(rows, ignore_index=True, sort=False).fillna("")
    if "ticker" not in company.columns:
        company["ticker"] = ""
    if "company" not in company.columns:
        company["company"] = ""

    company["company_node"] = company["company_node"].map(clean_node)
    company = company[company["company_node"].ne("")].copy()
    out = company.groupby("company_node", as_index=False).agg(
        ticker=("ticker", lambda x: next((str(v) for v in x if clean_node(v)), "")),
        company=("company", lambda x: next((str(v) for v in x if clean_node(v)), "")),
    )
    out["ticker"] = np.where(out["ticker"].astype(str).str.len() > 0, out["ticker"], out["company_node"].map(ticker_from_node))
    out["display_name"] = out["ticker"]
    return out


def get_ego_nodes(events: pd.DataFrame, center_node: str, depth: int, max_nodes: int) -> set[str]:
    g = nx.Graph()
    for row in events.itertuples(index=False):
        s = clean_node(getattr(row, "source_node"))
        t = clean_node(getattr(row, "target_node"))
        if s and t and s != t:
            g.add_edge(s, t)

    if center_node not in g:
        return {center_node}

    dist = nx.single_source_shortest_path_length(g, center_node, cutoff=depth)
    ordered = sorted(dist.keys(), key=lambda n: (dist[n], n))
    return set(ordered[:max_nodes])


def build_edge_summary(events: pd.DataFrame, max_links: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    group_cols = [
        "analysis_mode",
        "source_node",
        "target_node",
        "signal",
        "relation_group",
        "source_direction",
        "source_quarter",
        "target_quarter",
        "window",
    ]

    out = events.groupby(group_cols, dropna=False).agg(
        event_count=("signal", "count"),
        success_count=("success", "sum"),
        target_active_rate=("target_active", "mean"),
        direction_match_rate=("direction_match", "mean"),
        same_quarter_publish_gap_days=("publish_gap_days", "mean"),
        source_before_target_rate=("source_before_target", "mean"),
        source_publish_date=("source_publish_date", "min"),
        target_publish_date=("target_publish_date", "min"),
    ).reset_index()

    out["success_rate"] = out["success_count"] / out["event_count"].replace(0, np.nan)

    # Interpretation fields.
    out["time_interpretation"] = np.where(
        out["analysis_mode"].eq("same_quarter"),
        "date-level within-quarter flow",
        "quarter-level lead-lag flow",
    )
    out["display_gap_days"] = np.where(
        out["analysis_mode"].eq("same_quarter"),
        out["same_quarter_publish_gap_days"],
        np.nan,
    )
    out["display_gap_label"] = np.where(
        out["analysis_mode"].eq("same_quarter"),
        out["same_quarter_publish_gap_days"].round(1).astype(str),
        "not used for cross-quarter flow",
    )

    out["source_publish_date"] = pd.to_datetime(out["source_publish_date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    out["target_publish_date"] = pd.to_datetime(out["target_publish_date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    out["source_publish_date"] = out["source_publish_date"].replace("NaT", "")
    out["target_publish_date"] = out["target_publish_date"].replace("NaT", "")

    return out.sort_values(["success_count", "event_count"], ascending=[False, False]).head(max_links).copy()


def build_graph_layout(edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    g = nx.Graph()
    for row in edges.itertuples(index=False):
        s = row.source_node
        t = row.target_node
        w = max(1, int(row.event_count))
        g.add_edge(s, t, weight=w)

    if g.number_of_nodes() == 0:
        return {}

    pos = nx.spring_layout(g, seed=42, weight="weight", iterations=140)
    return {node: (float(x), float(y)) for node, (x, y) in pos.items()}


def compute_leader_metrics(events: pd.DataFrame) -> pd.DataFrame:
    active = events[events["source_active"]].copy()
    if active.empty:
        return pd.DataFrame()

    src = active.groupby("source_node", as_index=False).agg(
        outgoing_exposures=("signal", "count"),
        outgoing_successes=("success", "sum"),
        outgoing_match_rate=("direction_match", "mean"),
        distinct_targets=("target_node", "nunique"),
    ).rename(columns={"source_node": "company_node"})

    tgt = active.groupby("target_node", as_index=False).agg(
        incoming_exposures=("signal", "count"),
        incoming_successes=("success", "sum"),
        incoming_match_rate=("direction_match", "mean"),
        distinct_sources=("source_node", "nunique"),
    ).rename(columns={"target_node": "company_node"})

    out = src.merge(tgt, on="company_node", how="outer").fillna(0)
    out["leader_score"] = np.log1p(out["outgoing_exposures"]) * out["outgoing_match_rate"]
    out["follower_score"] = np.log1p(out["incoming_exposures"]) * out["incoming_match_rate"]
    return out


def safe_json_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_json(args: argparse.Namespace) -> dict:
    cross = read_hf_parquet(args.two_part_dataset, "cross_quarter_events.parquet", args.two_part_prefix, args.revision)
    same = read_hf_parquet(args.two_part_dataset, "same_quarter_events.parquet", args.two_part_prefix, args.revision)
    outlook = read_hf_parquet(args.two_part_dataset, "cleaned_outlook_all.parquet", args.two_part_prefix, args.revision)

    events = prepare_events(cross, same)

    if not args.no_publish_dates:
        if args.chunk_index:
            # Fast path: read publish_dates directly from local chunk_index.csv
            publish_dates = build_publish_dates_from_chunk_index(args.chunk_index)
        else:
            metadata = read_hf_parquet(args.evidence_dataset, args.evidence_metadata_file, "", args.revision)
            publish_dates = build_publish_dates(metadata)
        events = attach_publish_dates(events, publish_dates)
    else:
        events["source_publish_date"] = pd.NaT
        events["target_publish_date"] = pd.NaT
        events["publish_gap_days"] = np.nan
        events["source_before_target"] = False

    if args.mode != "both":
        events = events[events["analysis_mode"].eq(args.mode)].copy()
    if args.signal != "All":
        events = events[events["signal"].eq(args.signal)].copy()
    if args.only_successful:
        events = events[events["success"]].copy()
    events = filter_quarter_range(events, args.start_quarter, args.end_quarter)

    if args.ordered_same_quarter_only:
        events = events[
            events["analysis_mode"].eq("same_quarter")
            & events["source_before_target"].fillna(False)
        ].copy()

    center_node = f"COMPANY::{args.ticker.upper()}"
    ego_nodes = get_ego_nodes(events, center_node, args.hop_depth, args.max_nodes)
    events = events[events["source_node"].isin(ego_nodes) & events["target_node"].isin(ego_nodes)].copy()

    edges = build_edge_summary(events, args.max_links)
    layout = build_graph_layout(edges)

    company = build_company_table(outlook, events)
    company_map = company.set_index("company_node").to_dict(orient="index") if not company.empty else {}
    leader = compute_leader_metrics(events)
    leader_map = leader.set_index("company_node").to_dict(orient="index") if not leader.empty else {}

    nodes = sorted(set(edges["source_node"]) | set(edges["target_node"]))

    node_rows = []
    for node in nodes:
        x, y = layout.get(node, (0.0, 0.0))
        c = company_map.get(node, {})
        m = leader_map.get(node, {})
        node_rows.append({
            "id": node,
            "ticker": c.get("ticker", ticker_from_node(node)),
            "name": c.get("company", node),
            "label": c.get("display_name", ticker_from_node(node)),
            "x": x,
            "y": y,
            "leader_score": safe_json_float(m.get("leader_score", 0)),
            "follower_score": safe_json_float(m.get("follower_score", 0)),
            "outgoing_exposures": int(m.get("outgoing_exposures", 0)) if not pd.isna(m.get("outgoing_exposures", 0)) else 0,
            "incoming_exposures": int(m.get("incoming_exposures", 0)) if not pd.isna(m.get("incoming_exposures", 0)) else 0,
            "is_center": node == center_node,
        })

    link_rows = []
    for i, row in enumerate(edges.itertuples(index=False)):
        link_rows.append({
            "id": f"e{i}",
            "analysis_mode": row.analysis_mode,
            "source": row.source_node,
            "target": row.target_node,
            "signal": row.signal,
            "relation_group": row.relation_group,
            "source_direction": row.source_direction,
            "event_count": int(row.event_count),
            "success_count": int(row.success_count),
            "success_rate": safe_json_float(row.success_rate),
            "direction_match_rate": safe_json_float(row.direction_match_rate),
            "target_active_rate": safe_json_float(row.target_active_rate),
            "same_quarter_publish_gap_days": safe_json_float(row.same_quarter_publish_gap_days),
            "source_before_target_rate": safe_json_float(row.source_before_target_rate),
            "display_gap_days": safe_json_float(row.display_gap_days),
            "display_gap_label": str(row.display_gap_label),
            "time_interpretation": row.time_interpretation,
            "source_quarter": row.source_quarter,
            "target_quarter": row.target_quarter,
            "window": row.window,
            "source_publish_date": "" if pd.isna(row.source_publish_date) else str(row.source_publish_date),
            "target_publish_date": "" if pd.isna(row.target_publish_date) else str(row.target_publish_date),
        })

    timeline = events[events["source_active"]].groupby(["source_quarter", "signal"], as_index=False).agg(
        event_count=("signal", "count"),
        success_rate=("success", "mean"),
    )
    timeline["quarter_index"] = timeline["source_quarter"].map(quarter_to_index)
    timeline = timeline.sort_values(["quarter_index", "signal"])
    timeline_rows = [
        {
            "quarter": r.source_quarter,
            "signal": r.signal,
            "event_count": int(r.event_count),
            "success_rate": safe_json_float(r.success_rate),
        }
        for r in timeline.itertuples(index=False)
    ]

    quarters = sort_quarters(set(events["source_quarter"]) | set(events["target_quarter"]))

    metadata = {
        "ticker": args.ticker.upper(),
        "center_node": center_node,
        "mode": args.mode,
        "signal": args.signal,
        "start_quarter": args.start_quarter,
        "end_quarter": args.end_quarter,
        "hop_depth": args.hop_depth,
        "ordered_same_quarter_only": bool(args.ordered_same_quarter_only),
        "node_count": len(node_rows),
        "link_count": len(link_rows),
        "event_count": int(len(events)),
        "quarters": quarters,
        "v6_note": "cross-quarter uses quarter-level lead-lag; same-quarter uses publish_gap_days and source_before_target",
    }

    return {
        "metadata": metadata,
        "nodes": node_rows,
        "links": link_rows,
        "timeline": timeline_rows,
    }


def main() -> None:
    args = parse_args()
    data = build_json(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {out_path}")
    print(json.dumps(data["metadata"], indent=2))


if __name__ == "__main__":
    main()
