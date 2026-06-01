#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Placebo and falsification tests from Hugging Face Parquet data.

Tasks covered:
1. Random edge placebo.
2. Quarter shuffle placebo.
3. Signal shuffle placebo.
4. Community-controlled non-edge baseline.
5. Non-neighbor baseline.
6. Reverse-time test.

Default input dataset:
- soysouce/earningALZ_twopart
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
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
    parser.add_argument("--out-dir", default="results/placebo_falsification_hf")
    parser.add_argument("--start-quarter", default="")
    parser.add_argument("--end-quarter", default="")
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--min-exposed", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
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


def index_to_quarter(index: int) -> str:
    year = (index - 1) // 4
    quarter = index - year * 4
    return f"{year}Q{quarter}"


def next_quarter(quarter: str, step: int = 1) -> str:
    qidx = quarter_to_index(quarter)
    if pd.isna(qidx):
        return ""
    return index_to_quarter(int(qidx) + step)


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
    return outlook, relationships, cross, same


def prepare_events(cross: pd.DataFrame, same: pd.DataFrame):
    cross = cross.copy()
    same = same.copy()
    if "analysis_mode" not in cross.columns:
        cross["analysis_mode"] = "cross_quarter"
    if "analysis_mode" not in same.columns:
        same["analysis_mode"] = "same_quarter"
    for df in [cross, same]:
        for col in ["source_active", "target_active", "direction_match", "exact_match"]:
            df[col] = to_bool(df[col]) if col in df.columns else False
        for col in ["source_direction", "target_direction", "signal", "source_label", "target_label"]:
            df[col] = df[col].astype(str).str.strip() if col in df.columns else ""
        df["success"] = df["source_active"] & df["target_active"] & df["direction_match"]
    return cross, same


def build_outlook_lookup(outlook: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
    df = outlook.copy()
    for col in ["company_node", "quarter", "signal", "direction"]:
        df[col] = df[col].astype(str).str.strip() if col in df.columns else ""
    if "is_active" in df.columns:
        df["is_active"] = to_bool(df["is_active"])
    else:
        df["score"] = pd.to_numeric(df.get("score", np.nan), errors="coerce")
        df["is_active"] = df["score"].abs() > 0
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[(getattr(row, "company_node"), getattr(row, "quarter"), getattr(row, "signal"))] = {
            "direction": getattr(row, "direction"),
            "is_active": bool(getattr(row, "is_active")),
            "score": getattr(row, "score", np.nan),
        }
    return lookup


def edge_set_from_relationships(relationships: pd.DataFrame) -> set[tuple[str, str]]:
    edges = set()
    for _, row in relationships.iterrows():
        s = clean_node(row.get("source_company_node", ""))
        t = clean_node(row.get("target_company_node", ""))
        if not s or not t or s == t:
            continue
        edges.add((s, t))
        edges.add((t, s))
    return edges


def community_map_from_relationships(relationships: pd.DataFrame) -> dict[str, int]:
    graph = nx.Graph()
    for _, row in relationships.iterrows():
        s = clean_node(row.get("source_company_node", ""))
        t = clean_node(row.get("target_company_node", ""))
        if not s or not t or s == t:
            continue
        graph.add_edge(s, t, weight=graph[s][t]["weight"] + 1 if graph.has_edge(s, t) else 1)
    if graph.number_of_nodes() == 0:
        return {}
    cmap = {}
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    for cid, nodes in enumerate(communities):
        for node in nodes:
            cmap[node] = cid
    return cmap


def metric_row(data: pd.DataFrame, test_name: str, iteration: int | None = None) -> dict:
    if data.empty:
        return {"test_name": test_name, "iteration": iteration, "event_count": 0, "source_active_count": 0, "target_active_rate": np.nan, "direction_match_rate": np.nan, "exact_match_rate": np.nan, "success_rate": np.nan}
    active = data[data["source_active"]].copy()
    if active.empty:
        return {"test_name": test_name, "iteration": iteration, "event_count": len(data), "source_active_count": 0, "target_active_rate": np.nan, "direction_match_rate": np.nan, "exact_match_rate": np.nan, "success_rate": np.nan}
    return {
        "test_name": test_name,
        "iteration": iteration,
        "event_count": len(data),
        "source_active_count": int(active["source_active"].sum()),
        "target_active_rate": float(active["target_active"].mean()),
        "direction_match_rate": float(active["direction_match"].mean()),
        "exact_match_rate": float(active["exact_match"].mean()) if "exact_match" in active.columns else np.nan,
        "success_rate": float((active["target_active"] & active["direction_match"]).mean()),
    }


def real_metrics(cross: pd.DataFrame, min_exposed: int):
    active = cross[cross["source_active"]].copy()
    group_cols = ["analysis_mode", "signal", "source_direction"]
    s = active.groupby(group_cols, dropna=False).agg(
        exposed_events=("signal", "count"),
        target_active_events=("target_active", "sum"),
        direction_match_events=("direction_match", "sum"),
        exact_match_events=("exact_match", "sum"),
        success_events=("success", "sum"),
    ).reset_index()
    s["target_active_rate"] = s["target_active_events"] / s["exposed_events"]
    s["direction_match_rate"] = s["direction_match_events"] / s["exposed_events"]
    s["exact_match_rate"] = s["exact_match_events"] / s["exposed_events"]
    s["success_rate"] = s["success_events"] / s["exposed_events"]
    s = s[s["exposed_events"] >= min_exposed].sort_values("success_rate", ascending=False)
    return s, pd.DataFrame([metric_row(cross, "real_cross_quarter", None)])


def recompute_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["direction_match"] = out["source_active"] & out["target_active"] & out["source_direction"].astype(str).eq(out["target_direction"].astype(str))
    out["exact_match"] = False
    if "source_label" in out.columns and "target_label" in out.columns:
        out["exact_match"] = out["source_active"] & out["target_active"] & out["source_label"].astype(str).eq(out["target_label"].astype(str))
    out["success"] = out["source_active"] & out["target_active"] & out["direction_match"]
    return out


def random_edge_placebo(cross: pd.DataFrame, rng: np.random.Generator, iteration: int) -> dict:
    parts = []
    for _, group in cross.groupby(["target_quarter", "signal"], dropna=False):
        group = group.copy().reset_index(drop=True)
        cols = [c for c in ["target_node", "target_direction", "target_active", "target_label"] if c in group.columns]
        if len(group) > 1 and cols:
            shuffled = group[cols].iloc[rng.permutation(len(group))].reset_index(drop=True)
            for c in cols:
                group[c] = shuffled[c]
        parts.append(group)
    placebo = recompute_matches(pd.concat(parts, ignore_index=True))
    return metric_row(placebo, "random_edge_placebo", iteration)


def quarter_shuffle_placebo(cross: pd.DataFrame, rng: np.random.Generator, iteration: int) -> dict:
    data = cross.copy().reset_index(drop=True)
    cols = [c for c in ["target_quarter", "target_direction", "target_active", "target_label"] if c in data.columns]
    shuffled = data[cols].iloc[rng.permutation(len(data))].reset_index(drop=True)
    for c in cols:
        data[c] = shuffled[c]
    return metric_row(recompute_matches(data), "quarter_shuffle_placebo", iteration)


def signal_shuffle_placebo(cross: pd.DataFrame, rng: np.random.Generator, iteration: int) -> dict:
    data = cross.copy().reset_index(drop=True)
    cols = [c for c in ["target_direction", "target_active", "target_label"] if c in data.columns]
    shuffled = data[cols].iloc[rng.permutation(len(data))].reset_index(drop=True)
    for c in cols:
        data[c] = shuffled[c]
    return metric_row(recompute_matches(data), "signal_shuffle_placebo", iteration)


def sample_non_edge_baseline(
    cross: pd.DataFrame,
    outlook_lookup: dict,
    edge_set: set[tuple[str, str]],
    rng: np.random.Generator,
    sample_size: int,
    community_map: dict[str, int] | None,
    require_same_community: bool,
) -> pd.DataFrame:
    active = cross[cross["source_active"]].copy()
    if len(active) > sample_size:
        active = active.sample(n=sample_size, random_state=int(rng.integers(0, 1_000_000)))
    companies_by_quarter_signal = defaultdict(list)
    for company, quarter, signal in outlook_lookup.keys():
        companies_by_quarter_signal[(quarter, signal)].append(company)
    rows = []
    for row in active.itertuples(index=False):
        source = getattr(row, "source_node")
        target_quarter = getattr(row, "target_quarter")
        signal = getattr(row, "signal")
        source_direction = getattr(row, "source_direction")
        candidates = companies_by_quarter_signal.get((target_quarter, signal), [])
        if not candidates:
            continue
        selected = None
        for _ in range(50):
            candidate = candidates[int(rng.integers(0, len(candidates)))]
            if candidate == source:
                continue
            if (source, candidate) in edge_set:
                continue
            if require_same_community and community_map and community_map.get(source) != community_map.get(candidate):
                continue
            selected = candidate
            break
        if not selected:
            continue
        info = outlook_lookup.get((selected, target_quarter, signal), {})
        target_direction = str(info.get("direction", ""))
        target_active = bool(info.get("is_active", False))
        direction_match = target_active and str(source_direction) == target_direction
        rows.append({
            "source_node": source,
            "target_node": selected,
            "source_quarter": getattr(row, "source_quarter"),
            "target_quarter": target_quarter,
            "signal": signal,
            "source_direction": source_direction,
            "target_direction": target_direction,
            "source_active": True,
            "target_active": target_active,
            "direction_match": direction_match,
            "exact_match": False,
            "success": target_active and direction_match,
        })
    return pd.DataFrame(rows)


def reverse_time_tests(cross: pd.DataFrame, outlook_lookup: dict) -> pd.DataFrame:
    rows = []
    active = cross[cross["source_active"] & cross["target_active"]].copy()
    if not active.empty:
        active["reverse_direction_match"] = active["target_direction"].astype(str).eq(active["source_direction"].astype(str))
        rows.append({
            "test_name": "future_target_predicts_past_source",
            "event_count": len(active),
            "match_rate": float(active["reverse_direction_match"].mean()),
            "description": "Future target signal is compared with past source signal. Strong performance indicates common-trend risk.",
        })
    future_rows = []
    for row in active.itertuples(index=False):
        source = getattr(row, "source_node")
        target_quarter = getattr(row, "target_quarter")
        signal = getattr(row, "signal")
        target_direction = getattr(row, "target_direction")
        future_source_quarter = next_quarter(target_quarter, 1)
        info = outlook_lookup.get((source, future_source_quarter, signal))
        if not info:
            continue
        future_active = bool(info.get("is_active", False))
        future_direction = str(info.get("direction", ""))
        future_rows.append({
            "source_node": source,
            "target_quarter": target_quarter,
            "source_future_quarter": future_source_quarter,
            "signal": signal,
            "target_direction": target_direction,
            "future_source_direction": future_direction,
            "future_source_active": future_active,
            "match": future_active and future_direction == str(target_direction),
        })
    future = pd.DataFrame(future_rows)
    if not future.empty:
        rows.append({
            "test_name": "target_t_plus_1_predicts_source_t_plus_2",
            "event_count": len(future),
            "match_rate": float(future["match"].mean()),
            "description": "Target signal in t+1 is compared with original source signal in t+2.",
        })
    return pd.DataFrame(rows)


def summarize_placebos(real: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    real_success = float(real["success_rate"].iloc[0]) if not real.empty else np.nan
    rows = []
    for frame in frames:
        if frame.empty:
            continue
        name = str(frame["test_name"].iloc[0])
        values = pd.to_numeric(frame["success_rate"], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        rows.append({
            "test_name": name,
            "real_success_rate": real_success,
            "placebo_mean_success_rate": mean,
            "placebo_std_success_rate": std,
            "real_minus_placebo": real_success - mean,
            "real_percentile_vs_placebo": float((values <= real_success).mean()) if not pd.isna(real_success) else np.nan,
            "z_score": (real_success - mean) / std if std > 0 else np.nan,
            "iterations": len(values),
        })
    return pd.DataFrame(rows).sort_values("real_minus_placebo", ascending=False)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    d = summary.sort_values("real_minus_placebo")
    plt.figure(figsize=(11, 6))
    plt.barh(d["test_name"], d["real_minus_placebo"])
    plt.title("Real success rate minus placebo success rate")
    plt.xlabel("Difference in success rate")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"SAVED {path}")


def write_report(path: Path, real: pd.DataFrame, summary: pd.DataFrame, reverse: pd.DataFrame) -> None:
    lines = [
        "# Placebo and Falsification Tests",
        "",
        "## Real Cross-Quarter Metric",
        "",
        real.to_markdown(index=False) if not real.empty else "No results.",
        "",
        "## Placebo Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No results.",
        "",
        "## Reverse-Time Tests",
        "",
        reverse.to_markdown(index=False) if not reverse.empty else "No results.",
        "",
        "## Interpretation",
        "",
        "- Positive `real_minus_placebo` means the real network is stronger than the placebo baseline.",
        "- High `real_percentile_vs_placebo` means the real result is stronger than most placebo iterations.",
        "- Strong reverse-time results indicate potential common-trend or confounding risk.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"SAVED {path}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")
    rng = np.random.default_rng(args.random_state)

    outlook, relationships, cross_raw, same_raw = load_inputs(args)
    outlook = filter_quarter_range(outlook, ["quarter"], args.start_quarter, args.end_quarter)
    relationships = filter_quarter_range(relationships, ["quarter"], args.start_quarter, args.end_quarter)
    cross_raw = filter_quarter_range(cross_raw, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)
    same_raw = filter_quarter_range(same_raw, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)

    cross, same = prepare_events(cross_raw, same_raw)
    if len(cross) > args.sample_size:
        cross_sample = cross.sample(n=args.sample_size, random_state=args.random_state).reset_index(drop=True)
    else:
        cross_sample = cross.reset_index(drop=True)

    real_by_group, real_global = real_metrics(cross_sample, args.min_exposed)
    save_table(real_by_group, out_dir / "real_cross_quarter_metrics.parquet", args.write_csv_copy)
    save_table(real_global, out_dir / "real_cross_quarter_global_metric.parquet", args.write_csv_copy)

    lookup = build_outlook_lookup(outlook)
    edges = edge_set_from_relationships(relationships)
    communities = community_map_from_relationships(relationships)

    random_rows, quarter_rows, signal_rows, non_neighbor_rows, community_rows = [], [], [], [], []
    for i in range(args.n_iterations):
        random_rows.append(random_edge_placebo(cross_sample, rng, i))
        quarter_rows.append(quarter_shuffle_placebo(cross_sample, rng, i))
        signal_rows.append(signal_shuffle_placebo(cross_sample, rng, i))

        non_neighbor = sample_non_edge_baseline(cross_sample, lookup, edges, rng, args.sample_size, None, False)
        non_neighbor_rows.append(metric_row(non_neighbor, "non_neighbor_baseline", i))

        community_control = sample_non_edge_baseline(cross_sample, lookup, edges, rng, args.sample_size, communities, True)
        community_rows.append(metric_row(community_control, "community_controlled_non_edge_baseline", i))

        if (i + 1) % 10 == 0:
            print(f"Completed iteration {i + 1}/{args.n_iterations}")

    random_df = pd.DataFrame(random_rows)
    quarter_df = pd.DataFrame(quarter_rows)
    signal_df = pd.DataFrame(signal_rows)
    non_neighbor_df = pd.DataFrame(non_neighbor_rows)
    community_df = pd.DataFrame(community_rows)
    reverse = reverse_time_tests(cross_sample, lookup)

    save_table(random_df, out_dir / "random_edge_placebo_iterations.parquet", args.write_csv_copy)
    save_table(quarter_df, out_dir / "quarter_shuffle_placebo_iterations.parquet", args.write_csv_copy)
    save_table(signal_df, out_dir / "signal_shuffle_placebo_iterations.parquet", args.write_csv_copy)
    save_table(non_neighbor_df, out_dir / "non_neighbor_baseline_iterations.parquet", args.write_csv_copy)
    save_table(community_df, out_dir / "community_controlled_baseline_iterations.parquet", args.write_csv_copy)
    save_table(reverse, out_dir / "reverse_time_test.parquet", args.write_csv_copy)

    summary = summarize_placebos(real_global, [random_df, quarter_df, signal_df, non_neighbor_df, community_df])
    save_table(summary, out_dir / "placebo_test_summary.parquet", args.write_csv_copy)
    plot_summary(summary, fig_dir / "placebo_test_summary.png")
    write_report(out_dir / "placebo_falsification_summary.md", real_global, summary, reverse)
    print("DONE")


if __name__ == "__main__":
    main()
