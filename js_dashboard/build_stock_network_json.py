#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a lightweight JSON file for a JavaScript dynamic information-flow network.

Example:
    python build_stock_network_json.py --ticker AAPL --out data/aapl_network.json

The script downloads Parquet files directly from Hugging Face, filters the network around
one selected stock, and exports a browser-friendly JSON file.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL", help="Ticker to center the network on, for example AAPL.")
    p.add_argument("--two-part-dataset", default=DEFAULT_TWO_PART_DATASET)
    p.add_argument("--two-part-prefix", default="")
    p.add_argument("--revision", default="main")
    p.add_argument("--mode", default="cross_quarter", choices=["cross_quarter", "same_quarter", "both"])
    p.add_argument("--signal", default="All", help="All or one signal, e.g. demand_outlook.")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--hop-depth", type=int, default=2)
    p.add_argument("--max-nodes", type=int, default=120)
    p.add_argument("--max-links", type=int, default=260)
    p.add_argument("--only-successful", action="store_true")
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
        "source_node", "target_node", "source_quarter", "target_quarter",
        "signal", "relation_group", "source_direction", "target_direction",
        "source_label", "target_label",
    ]:
        if c not in events.columns:
            events[c] = ""
        events[c] = events[c].astype(str).str.strip()

    events["success"] = events["source_active"] & events["target_active"] & events["direction_match"]

    if "release_date_gap_days" in events.columns:
        events["release_date_gap_days"] = pd.to_numeric(events["release_date_gap_days"], errors="coerce")
    else:
        events["release_date_gap_days"] = np.nan

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

    group_cols = ["source_node", "target_node", "signal", "relation_group", "source_direction"]
    out = events.groupby(group_cols, dropna=False).agg(
        event_count=("signal", "count"),
        success_count=("success", "sum"),
        target_active_rate=("target_active", "mean"),
        direction_match_rate=("direction_match", "mean"),
        avg_gap_days=("release_date_gap_days", "mean"),
        first_source_quarter=("source_quarter", "min"),
        last_target_quarter=("target_quarter", "max"),
    ).reset_index()
    out["success_rate"] = out["success_count"] / out["event_count"].replace(0, np.nan)
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

    if args.mode != "both":
        events = events[events["analysis_mode"].eq(args.mode)].copy()
    if args.signal != "All":
        events = events[events["signal"].eq(args.signal)].copy()
    if args.only_successful:
        events = events[events["success"]].copy()
    events = filter_quarter_range(events, args.start_quarter, args.end_quarter)

    center_node = f"COMPANY::{args.ticker.upper()}"
    ego_nodes = get_ego_nodes(events, center_node, args.hop_depth, args.max_nodes)
    events = events[events["source_node"].isin(ego_nodes) & events["target_node"].isin(ego_nodes)].copy()

    edges = build_edge_summary(events, args.max_links)
    layout = build_graph_layout(edges)

    company = build_company_table(outlook, events)
    company_map = company.set_index("company_node").to_dict(orient="index") if not company.empty else {}
    leader = compute_leader_metrics(events)
    leader_map = leader.set_index("company_node").to_dict(orient="index") if not leader.empty else {}

    nodes = sorted(set(edges["source_node"]) | set(edges["target_node"])) if not edges.empty else []

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
    if not edges.empty:
        for i, row in enumerate(edges.itertuples(index=False)):
            link_rows.append({
                "id": f"e{i}",
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
                "avg_gap_days": safe_json_float(row.avg_gap_days),
                "first_source_quarter": row.first_source_quarter,
                "last_target_quarter": row.last_target_quarter,
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
        "node_count": len(node_rows),
        "link_count": len(link_rows),
        "event_count": int(len(events)),
        "quarters": quarters,
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
