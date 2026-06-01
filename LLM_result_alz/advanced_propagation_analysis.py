#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced propagation analysis from Hugging Face Parquet data.

Tasks covered:
1. Propagation delay distribution.
2. Cascade depth and multi-hop paths.
3. Propagation strength by relation type.
4. Signal-specific propagation.
5. Asymmetric propagation.
6. Leader-follower ranking.
7. Community-level propagation.
8. Shock event detection.

Default input datasets:
- soysouce/earningALZ_twopart
- soysouce/earningALZ_SBERT_evidence
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-two-part-dataset", default="soysouce/earningALZ_twopart")
    parser.add_argument("--hf-two-part-revision", default="main")
    parser.add_argument("--hf-two-part-prefix", default="")
    parser.add_argument("--hf-evidence-dataset", default="soysouce/earningALZ_SBERT_evidence")
    parser.add_argument("--hf-evidence-revision", default="main")
    parser.add_argument("--hf-evidence-metadata-file", default="rag_evidence_package_metadata_full_gpu_direct.parquet")
    parser.add_argument("--out-dir", default="results/advanced_propagation_hf")
    parser.add_argument("--start-quarter", default="")
    parser.add_argument("--end-quarter", default="")
    parser.add_argument("--min-exposed", type=int, default=10)
    parser.add_argument("--max-cascade-depth", type=int, default=4)
    parser.add_argument("--max-cascade-seed-edges", type=int, default=2500)
    parser.add_argument("--max-cascade-paths", type=int, default=20000)
    parser.add_argument("--write-csv-copy", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: Path, write_csv_copy: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"SAVED {path} rows={len(df):,} cols={len(df.columns):,}")
    if write_csv_copy:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"SAVED {csv_path} rows={len(df):,}")


def quarter_to_index(quarter: str) -> float:
    match = re.match(r"^(\d{4})Q([1-4])$", str(quarter).strip())
    if not match:
        return np.nan
    return int(match.group(1)) * 4 + int(match.group(2))


def filter_quarter_range(df: pd.DataFrame, quarter_columns: list[str], start_quarter: str, end_quarter: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in quarter_columns:
        if column not in out.columns:
            continue
        qidx = out[column].map(quarter_to_index)
        mask = qidx.notna()
        if start_quarter:
            mask &= qidx >= quarter_to_index(start_quarter)
        if end_quarter:
            mask &= qidx <= quarter_to_index(end_quarter)
        out = out[mask].copy()
    return out


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def clean_node(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "0"}:
        return ""
    return text


def normalize_relation_group(value) -> str:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return "unknown"
    return text


def prefixed(prefix: str, filename: str) -> str:
    prefix = prefix.strip().strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def select_hf_file(repo_id: str, revision: str, filename: str, prefix: str, required_columns: set[str] | None = None) -> str:
    files = sorted([f for f in list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision) if f.endswith(".parquet")])
    stem = Path(filename).stem
    candidates = [
        prefixed(prefix, filename),
        filename,
        prefixed(prefix, f"two_part_network_prediction_analysis_hf/{filename}"),
        prefixed(prefix, f"two_part_network_prediction_analysis_parquet/{filename}"),
        prefixed(prefix, f"two_part_network_prediction_analysis/{filename}"),
        prefixed(prefix, f"data/{filename}"),
        prefixed(prefix, f"results/{filename}"),
    ]
    candidates += [f for f in files if Path(f).name == filename]
    candidates += [f for f in files if stem in Path(f).stem]
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    errors = []
    for candidate in candidates:
        if candidate not in files:
            continue
        local = hf_hub_download(repo_id=repo_id, filename=candidate, repo_type="dataset", revision=revision)
        if required_columns:
            cols = set(pd.read_parquet(local).columns)
            missing = required_columns - cols
            if missing:
                errors.append(f"{candidate}: missing {sorted(missing)}")
                continue
        print(f"HF SELECT: {repo_id}/{candidate}")
        return candidate
    available = "\n".join(f"  - {f}" for f in files[:120])
    raise FileNotFoundError(f"Cannot find {filename} in {repo_id}. Available parquet files:\n{available}\nErrors:\n" + "\n".join(errors[:20]))


def read_hf_parquet(repo_id: str, revision: str, filename: str, prefix: str = "", required_columns: set[str] | None = None) -> pd.DataFrame:
    remote = select_hf_file(repo_id, revision, filename, prefix, required_columns)
    local = hf_hub_download(repo_id=repo_id, filename=remote, repo_type="dataset", revision=revision)
    df = pd.read_parquet(local)
    df["_hf_dataset"] = repo_id
    df["_hf_file"] = remote
    print(f"HF LOAD: {repo_id}/{remote} rows={len(df):,} cols={len(df.columns):,}")
    return df


def load_inputs(args: argparse.Namespace):
    outlook = read_hf_parquet(args.hf_two_part_dataset, args.hf_two_part_revision, "cleaned_outlook_all.parquet", args.hf_two_part_prefix, {"company_node", "quarter", "signal", "score"})
    relationships = read_hf_parquet(args.hf_two_part_dataset, args.hf_two_part_revision, "matched_company_relationships.parquet", args.hf_two_part_prefix, {"source_company_node", "target_company_node"})
    cross = read_hf_parquet(args.hf_two_part_dataset, args.hf_two_part_revision, "cross_quarter_events.parquet", args.hf_two_part_prefix, {"source_node", "target_node", "source_quarter", "target_quarter", "signal"})
    same = read_hf_parquet(args.hf_two_part_dataset, args.hf_two_part_revision, "same_quarter_events.parquet", args.hf_two_part_prefix, {"source_node", "target_node", "source_quarter", "target_quarter", "signal"})
    evidence = read_hf_parquet(args.hf_evidence_dataset, args.hf_evidence_revision, args.hf_evidence_metadata_file, "", {"quarter"})
    return outlook, relationships, cross, same, evidence


def find_date_column(df: pd.DataFrame) -> str:
    candidates = ["release_date", "earnings_call_date", "call_date", "date", "published_date", "publish_date", "publication_date", "transcript_date", "datetime"]
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    for column in df.columns:
        if "date" in column.lower() or "time" in column.lower():
            return column
    return ""


def company_node_from(ticker: str, company: str) -> str:
    ticker = clean_node(ticker)
    company = clean_node(company)
    if ticker:
        return "COMPANY::" + ticker
    text = re.sub(r"[^a-z0-9&.\- ]+", " ", company.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return "COMPANY::" + text


def build_release_dates(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])
    date_col = find_date_column(evidence)
    if not date_col:
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])
    ticker_col = "ticker" if "ticker" in evidence.columns else ""
    company_col = next((c for c in ["current_company", "company", "company_name", "name"] if c in evidence.columns), "")
    if not ticker_col and not company_col:
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])
    d = evidence.copy()
    d["release_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d[d["release_date"].notna()].copy()
    d["quarter"] = d["quarter"].astype(str).str.strip()
    d["company_node"] = d.apply(lambda r: company_node_from(r[ticker_col] if ticker_col else "", r[company_col] if company_col else ""), axis=1)
    return d.groupby(["company_node", "quarter"], as_index=False).agg(
        release_date=("release_date", "min"),
        release_date_count=("release_date", "count"),
    )


def attach_release_dates(events: pd.DataFrame, release_dates: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if release_dates.empty:
        out["source_release_date"] = pd.NaT
        out["target_release_date"] = pd.NaT
        out["release_date_gap_days"] = np.nan
        out["abs_release_date_gap_days"] = np.nan
        out["source_before_target"] = np.nan
        return out
    d = release_dates.copy()
    src = d.rename(columns={"company_node": "source_node", "quarter": "source_quarter", "release_date": "source_release_date"})
    tgt = d.rename(columns={"company_node": "target_node", "quarter": "target_quarter", "release_date": "target_release_date"})
    out = out.merge(src[["source_node", "source_quarter", "source_release_date"]], on=["source_node", "source_quarter"], how="left")
    out = out.merge(tgt[["target_node", "target_quarter", "target_release_date"]], on=["target_node", "target_quarter"], how="left")
    out["source_release_date"] = pd.to_datetime(out["source_release_date"], errors="coerce")
    out["target_release_date"] = pd.to_datetime(out["target_release_date"], errors="coerce")
    out["release_date_gap_days"] = (out["target_release_date"] - out["source_release_date"]).dt.days
    out["abs_release_date_gap_days"] = out["release_date_gap_days"].abs()
    out["source_before_target"] = out["release_date_gap_days"] > 0
    return out


def prepare_events(cross: pd.DataFrame, same: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    cross = cross.copy()
    same = same.copy()
    if "analysis_mode" not in cross.columns:
        cross["analysis_mode"] = "cross_quarter"
    if "analysis_mode" not in same.columns:
        same["analysis_mode"] = "same_quarter"
    events = pd.concat([cross, same], ignore_index=True, sort=False)
    for col in ["source_active", "target_active", "direction_match", "exact_match"]:
        events[col] = to_bool(events[col]) if col in events.columns else False
    for col in ["source_direction", "target_direction", "relation_group", "signal"]:
        events[col] = events[col].astype(str).str.strip() if col in events.columns else ""
    events["relation_group"] = events["relation_group"].map(normalize_relation_group)
    events = attach_release_dates(events, dates)
    events["window"] = events["source_quarter"].astype(str) + "→" + events["target_quarter"].astype(str)
    events["successful_direction_event"] = events["source_active"] & events["target_active"] & events["direction_match"]
    return events


def build_relationship_graph(relationships: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in relationships.iterrows():
        s = clean_node(row.get("source_company_node", ""))
        t = clean_node(row.get("target_company_node", ""))
        if not s or not t or s == t:
            continue
        if graph.has_edge(s, t):
            graph[s][t]["weight"] += 1.0
        else:
            graph.add_edge(s, t, weight=1.0)
    return graph


def build_communities(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["company_node", "community_id", "community_size"])
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    rows = []
    for cid, nodes in enumerate(communities):
        for node in nodes:
            rows.append({"company_node": node, "community_id": cid, "community_size": len(nodes)})
    return pd.DataFrame(rows)


def summarize_group(events: pd.DataFrame, group_cols: list[str], min_exposed: int) -> pd.DataFrame:
    active = events[events["source_active"]].copy()
    if active.empty:
        return pd.DataFrame()
    s = active.groupby(group_cols, dropna=False).agg(
        exposed_events=("signal", "count"),
        target_active_events=("target_active", "sum"),
        direction_match_events=("direction_match", "sum"),
        exact_match_events=("exact_match", "sum"),
        successful_events=("successful_direction_event", "sum"),
        mean_gap_days=("release_date_gap_days", "mean"),
        median_gap_days=("release_date_gap_days", "median"),
        mean_abs_gap_days=("abs_release_date_gap_days", "mean"),
        source_before_target_rate=("source_before_target", "mean"),
    ).reset_index()
    s["target_active_rate"] = s["target_active_events"] / s["exposed_events"]
    s["direction_match_rate"] = s["direction_match_events"] / s["exposed_events"]
    s["exact_match_rate"] = s["exact_match_events"] / s["exposed_events"]
    s["success_rate"] = s["successful_events"] / s["exposed_events"]
    return s[s["exposed_events"] >= min_exposed].sort_values(["success_rate", "exposed_events"], ascending=[False, False])


def analyze_delay(events: pd.DataFrame, min_exposed: int):
    d = events[events["source_active"] & events["target_active"] & events["release_date_gap_days"].notna()].copy()
    s = summarize_group(d, ["analysis_mode", "signal", "relation_group", "source_direction", "target_direction"], min_exposed)
    return d, s


def success_edge_table(events: pd.DataFrame) -> pd.DataFrame:
    e = events[events["successful_direction_event"]].copy()
    if e.empty:
        return pd.DataFrame()
    return e.groupby(["source_node", "target_node", "signal", "source_direction", "relation_group"], dropna=False).agg(
        event_count=("signal", "count"),
        first_source_quarter=("source_quarter", "min"),
        last_target_quarter=("target_quarter", "max"),
        mean_gap_days=("release_date_gap_days", "mean"),
        source_before_target_rate=("source_before_target", "mean"),
    ).reset_index().sort_values(["event_count", "source_before_target_rate"], ascending=[False, False])


def find_cascades(edges: pd.DataFrame, max_depth: int, max_seed_edges: int, max_paths: int) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    g = nx.DiGraph()
    for _, row in edges.head(max_seed_edges).iterrows():
        s, t = row["source_node"], row["target_node"]
        if s == t:
            continue
        weight = float(row["event_count"])
        g.add_edge(s, t, weight=g[s][t]["weight"] + weight if g.has_edge(s, t) else weight)
    rows, count = [], 0
    for start in g.nodes():
        q = deque([(start, [start], 0.0)])
        while q and count < max_paths:
            node, path, score = q.popleft()
            if len(path) > 1:
                rows.append({"path": " → ".join(path), "source_node": path[0], "terminal_node": path[-1], "depth": len(path) - 1, "path_score": score})
                count += 1
            if len(path) - 1 >= max_depth:
                continue
            for nxt in g.successors(node):
                if nxt not in path:
                    q.append((nxt, path + [nxt], score + math.log1p(g[node][nxt].get("weight", 1.0))))
    return pd.DataFrame(rows).sort_values(["depth", "path_score"], ascending=[False, False]) if rows else pd.DataFrame()


def relation_orientation(r: str) -> str:
    r = normalize_relation_group(r)
    if r in {"upstream", "supplier", "vendor", "component_provider"}:
        return "source_to_upstream_entity"
    if r in {"downstream", "customer", "customer_group", "buyer", "oem"}:
        return "source_to_downstream_entity"
    if r in {"parent"}:
        return "source_to_parent"
    if r in {"subsidiary"}:
        return "source_to_subsidiary"
    if r in {"partner"}:
        return "partner_or_horizontal"
    if r in {"competitor"}:
        return "competitor_or_horizontal"
    if r in {"acquirer"}:
        return "source_to_acquirer"
    if r in {"acquired_company"}:
        return "source_to_acquired_company"
    return "other_or_related"


def leader_follower(events: pd.DataFrame) -> pd.DataFrame:
    active = events[events["source_active"]].copy()
    if active.empty:
        return pd.DataFrame()
    src = active.groupby("source_node", as_index=False).agg(
        outgoing_exposures=("signal", "count"),
        outgoing_successes=("successful_direction_event", "sum"),
        outgoing_direction_match_rate=("direction_match", "mean"),
        outgoing_source_before_target_rate=("source_before_target", "mean"),
        distinct_targets=("target_node", "nunique"),
        distinct_signals_as_source=("signal", "nunique"),
    ).rename(columns={"source_node": "company_node"})
    tgt = active.groupby("target_node", as_index=False).agg(
        incoming_exposures=("signal", "count"),
        incoming_successes=("successful_direction_event", "sum"),
        incoming_direction_match_rate=("direction_match", "mean"),
        incoming_source_before_target_rate=("source_before_target", "mean"),
        distinct_sources=("source_node", "nunique"),
        distinct_signals_as_target=("signal", "nunique"),
    ).rename(columns={"target_node": "company_node"})
    out = src.merge(tgt, on="company_node", how="outer").fillna(0)
    out["leader_score"] = np.log1p(out["outgoing_exposures"]) * out["outgoing_direction_match_rate"] * (1.0 + out["outgoing_source_before_target_rate"].fillna(0))
    out["follower_score"] = np.log1p(out["incoming_exposures"]) * out["incoming_direction_match_rate"] * (1.0 - out["incoming_source_before_target_rate"].fillna(0))
    out["leader_minus_follower_score"] = out["leader_score"] - out["follower_score"]
    return out.sort_values("leader_score", ascending=False)


def community_propagation(events: pd.DataFrame, communities: pd.DataFrame, min_exposed: int) -> pd.DataFrame:
    if communities.empty:
        return pd.DataFrame()
    cmap = communities.set_index("company_node")["community_id"].to_dict()
    e = events[events["source_active"]].copy()
    e["source_community_id"] = e["source_node"].map(cmap)
    e["target_community_id"] = e["target_node"].map(cmap)
    e["same_community"] = e["source_community_id"].notna() & e["target_community_id"].notna() & e["source_community_id"].eq(e["target_community_id"])
    return summarize_group(e, ["analysis_mode", "same_community", "signal"], min_exposed)


def shock_events(events: pd.DataFrame, min_exposed: int) -> pd.DataFrame:
    s = summarize_group(events, ["analysis_mode", "window", "signal", "source_direction"], min_exposed)
    if s.empty:
        return s
    s["event_count_zscore"] = s.groupby(["analysis_mode", "signal"])["exposed_events"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0))
    s["success_rate_zscore"] = s.groupby(["analysis_mode", "signal"])["success_rate"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0))
    s["shock_score"] = s["event_count_zscore"] + s["success_rate_zscore"] + np.log1p(s["exposed_events"])
    return s.sort_values("shock_score", ascending=False)


def plot_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, top_n: int = 25) -> None:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return
    d = df[[label_col, value_col]].dropna().sort_values(value_col, ascending=False).head(top_n)
    if d.empty:
        return
    plt.figure(figsize=(12, 7))
    plt.barh(d[label_col].astype(str)[::-1], d[value_col][::-1])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"SAVED {path}")


def write_report(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Advanced Propagation Analysis", ""]
    for title, df in tables.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append(df.head(25).to_markdown(index=False) if not df.empty else "No results.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"SAVED {path}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")

    outlook, relationships, cross, same, evidence = load_inputs(args)
    outlook = filter_quarter_range(outlook, ["quarter"], args.start_quarter, args.end_quarter)
    relationships = filter_quarter_range(relationships, ["quarter"], args.start_quarter, args.end_quarter)
    cross = filter_quarter_range(cross, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)
    same = filter_quarter_range(same, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)

    dates = filter_quarter_range(build_release_dates(evidence), ["quarter"], args.start_quarter, args.end_quarter)
    events = prepare_events(cross, same, dates)
    graph = build_relationship_graph(relationships)
    communities = build_communities(graph)

    save_table(dates, out_dir / "company_quarter_release_dates.parquet", args.write_csv_copy)
    save_table(events, out_dir / "combined_events_with_release_dates.parquet", args.write_csv_copy)
    save_table(communities, out_dir / "graph_community_assignments.parquet", args.write_csv_copy)

    delay_events, delay_summary = analyze_delay(events, args.min_exposed)
    edges = success_edge_table(events)
    cascades = find_cascades(edges, args.max_cascade_depth, args.max_cascade_seed_edges, args.max_cascade_paths)
    relation_strength = summarize_group(events, ["analysis_mode", "relation_group"], args.min_exposed)
    signal_specific = summarize_group(events, ["analysis_mode", "signal", "source_direction"], args.min_exposed)
    asym_data = events.copy()
    asym_data["relation_orientation"] = asym_data["relation_group"].map(relation_orientation)
    asymmetric = summarize_group(asym_data, ["analysis_mode", "relation_orientation", "signal"], args.min_exposed)
    leaders = leader_follower(events)
    community = community_propagation(events, communities, args.min_exposed)
    shocks = shock_events(events, args.min_exposed)

    outputs = {
        "propagation_delay_events": delay_events,
        "propagation_delay_summary": delay_summary,
        "successful_propagation_edge_table": edges,
        "cascade_paths": cascades,
        "propagation_strength_by_relation": relation_strength,
        "signal_specific_propagation": signal_specific,
        "asymmetric_propagation": asymmetric,
        "leader_follower_ranking": leaders,
        "community_level_propagation": community,
        "shock_event_detection": shocks,
    }
    for name, df in outputs.items():
        save_table(df, out_dir / f"{name}.parquet", args.write_csv_copy)

    if not relation_strength.empty:
        relation_strength["plot_label"] = relation_strength["analysis_mode"].astype(str) + " | " + relation_strength["relation_group"].astype(str)
        plot_bar(relation_strength, "plot_label", "success_rate", "Propagation strength by relationship type", fig_dir / "propagation_strength_by_relation.png")
    if not signal_specific.empty:
        signal_specific["plot_label"] = signal_specific["analysis_mode"].astype(str) + " | " + signal_specific["signal"].astype(str) + " | " + signal_specific["source_direction"].astype(str)
        plot_bar(signal_specific, "plot_label", "success_rate", "Signal-specific propagation", fig_dir / "signal_specific_propagation.png")
    if not leaders.empty:
        plot_bar(leaders, "company_node", "leader_score", "Top information leaders", fig_dir / "leader_follower_ranking.png")
    if not shocks.empty:
        shocks["plot_label"] = shocks["window"].astype(str) + " | " + shocks["signal"].astype(str) + " | " + shocks["source_direction"].astype(str)
        plot_bar(shocks, "plot_label", "shock_score", "Detected information shocks", fig_dir / "shock_event_detection.png")

    write_report(out_dir / "advanced_propagation_summary.md", {
        "Propagation Delay Summary": delay_summary,
        "Cascade Paths": cascades,
        "Propagation Strength by Relation": relation_strength,
        "Signal-Specific Propagation": signal_specific,
        "Asymmetric Propagation": asymmetric,
        "Leader-Follower Ranking": leaders,
        "Community-Level Propagation": community,
        "Shock Event Detection": shocks,
    })
    print("DONE")


if __name__ == "__main__":
    main()
