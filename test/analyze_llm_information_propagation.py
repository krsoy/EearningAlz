#!/usr/bin/env python3
"""
Analyze LLM-extracted earnings-call signals as information propagation.

Expected input directory example:
  RAG/rag_chroma_output/llm_csv_outputs_2025Q2_Q3/

It will automatically read files like:
  concepts_q2q3_shard*_of006.csv
  relationships_q2q3_shard*_of006.csv
  outlook_q2q3_shard*_of006.csv

It can also analyze a single outlook CSV via --outlook-csv.

Outputs:
  cleaned tables, signal matrices, candidate propagation events, exposure features, and plots.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STANDARD_SIGNALS = [
    "demand_outlook",
    "supply_outlook",
    "margin_outlook",
    "capex_outlook",
    "inventory_outlook",
    "pricing_outlook",
]

SIGNAL_MAP = {
    # direct
    "demand_outlook": "demand_outlook",
    "supply_outlook": "supply_outlook",
    "margin_outlook": "margin_outlook",
    "capex_outlook": "capex_outlook",
    "inventory_outlook": "inventory_outlook",
    "pricing_outlook": "pricing_outlook",
    # common model drift / extra labels
    "supply_chain_outlook": "supply_outlook",
    "production_outlook": "supply_outlook",
    "manufacturing_outlook": "supply_outlook",
    "capacity_outlook": "supply_outlook",
    "loan_growth_outlook": "demand_outlook",
    "revenue_outlook": "demand_outlook",
    "sales_outlook": "demand_outlook",
    "credit_quality_outlook": "margin_outlook",
    "noninterest_income_outlook": "margin_outlook",
    "capital_generation_outlook": "capex_outlook",
}

LABEL_SCORE = {
    "positive": 1.0,
    "improving": 1.0,
    "increase": 1.0,
    "negative": -1.0,
    "worsening": -1.0,
    "decrease": -1.0,
    "mixed": 0.5,
    "neutral": 0.0,
    "stable": 0.0,
    "not_mentioned": np.nan,
    "": np.nan,
    "nan": np.nan,
}

CONCEPT_COLUMNS = [
    "chip_supply",
    "semiconductor_supply",
    "raw_material_supply",
    "oil_energy_supply",
    "manufacturing_capacity",
    "production_capacity",
    "inventory_pressure",
    "logistics_shipping",
    "supplier_constraint",
    "customer_demand",
    "pricing_pressure",
    "capex_expansion",
    "data_center_capacity",
    "cloud_infrastructure",
    "labor_constraint",
    "geopolitical_risk",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default="rag_chroma_output/llm_csv_outputs_2025Q2_Q3",
        help="Directory containing sharded LLM CSV outputs.",
    )
    p.add_argument(
        "--out-dir",
        default="rag_chroma_output/information_propagation_analysis_2025Q2_Q3",
        help="Output directory for analysis tables and figures.",
    )
    p.add_argument(
        "--outlook-csv",
        default="",
        help="Optional single outlook CSV. If provided, this overrides outlook shard discovery.",
    )
    p.add_argument(
        "--quarter-order",
        default="2025Q2,2025Q3",
        help="Comma-separated quarter order used for lag analysis.",
    )
    p.add_argument(
        "--top-companies",
        type=int,
        default=30,
        help="Number of companies to show in company-level plots.",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_sharded_csv(input_dir: Path, pattern: str, single_file: str = "") -> pd.DataFrame:
    if single_file:
        files = [Path(single_file)]
    else:
        files = sorted(input_dir.glob(pattern))

    frames = []
    for f in files:
        if f.exists() and f.stat().st_size > 0:
            try:
                df = pd.read_csv(f)
                df["source_file"] = f.name
                frames.append(df)
            except pd.errors.EmptyDataError:
                print(f"WARNING: empty CSV skipped: {f}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates()
    return df


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9&.\- ]+", "", s)
    s = s.replace(" corporation", " corp")
    s = s.replace(" incorporated", " inc")
    s = s.replace(" company", " co")
    s = re.sub(r"\b(the)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def quarter_to_int(q: object) -> float:
    s = str(q).strip()
    m = re.match(r"^(\d{4})Q([1-4])$", s)
    if not m:
        return np.nan
    year = int(m.group(1))
    quarter = int(m.group(2))
    return year * 4 + quarter


def clean_outlook(outlook: pd.DataFrame, quarter_order: list[str]) -> pd.DataFrame:
    if outlook.empty:
        return outlook

    required = ["doc_id", "ticker", "current_company", "quarter", "signal", "label"]
    missing = [c for c in required if c not in outlook.columns]
    if missing:
        raise ValueError(f"Outlook CSV missing required columns: {missing}")

    df = outlook.copy()
    for col in ["doc_id", "ticker", "current_company", "quarter", "signal", "label"]:
        df[col] = df[col].astype(str).str.strip()

    df["signal_raw"] = df["signal"]
    df["signal"] = df["signal"].str.lower().map(lambda x: SIGNAL_MAP.get(x, x))
    df["label_raw"] = df["label"]
    df["label"] = df["label"].str.lower().replace({"nan": "not_mentioned"})
    df["score"] = df["label"].map(LABEL_SCORE)
    df["company_norm"] = df["current_company"].map(norm_text)
    df["ticker_norm"] = df["ticker"].map(norm_text)
    df["quarter_index"] = df["quarter"].map(quarter_to_int)
    df["is_standard_signal"] = df["signal"].isin(STANDARD_SIGNALS)

    # For the core analysis, keep standard signals after mapping.
    df = df[df["quarter"].isin(quarter_order)].copy()

    # If multiple rows exist for the same doc/signal after mapping, aggregate conservatively.
    # Prefer non-null score, join evidence, join notes.
    key_cols = ["doc_id", "ticker", "current_company", "company_norm", "ticker_norm", "quarter", "quarter_index", "signal"]
    agg = {
        "label": lambda x: ";".join(sorted(set(str(v) for v in x if str(v) != "nan"))),
        "score": "mean",
        "evidence_chunk_ids": lambda x: "|".join(sorted(set(str(v) for v in x.dropna() if str(v).strip()))),
        "notes": lambda x: " || ".join(str(v) for v in x.dropna() if str(v).strip()),
        "source_file": lambda x: "|".join(sorted(set(str(v) for v in x.dropna()))),
    }
    keep_agg = {k: v for k, v in agg.items() if k in df.columns}
    df = df.groupby(key_cols, dropna=False).agg(keep_agg).reset_index()

    return df


def clean_concepts(concepts: pd.DataFrame, quarter_order: list[str]) -> pd.DataFrame:
    if concepts.empty:
        return concepts

    df = concepts.copy()
    for col in ["doc_id", "ticker", "current_company", "quarter"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df[df["quarter"].isin(quarter_order)].copy() if "quarter" in df.columns else df
    df["company_norm"] = df["current_company"].map(norm_text) if "current_company" in df.columns else ""
    df["ticker_norm"] = df["ticker"].map(norm_text) if "ticker" in df.columns else ""
    df["quarter_index"] = df["quarter"].map(quarter_to_int) if "quarter" in df.columns else np.nan

    for c in CONCEPT_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df.drop_duplicates()


def clean_relationships(rel: pd.DataFrame, quarter_order: list[str]) -> pd.DataFrame:
    if rel.empty:
        return rel

    df = rel.copy()
    for col in ["doc_id", "ticker", "current_company", "quarter", "relation_group", "entity", "entity_type", "relationship_type", "confidence"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "quarter" in df.columns:
        df = df[df["quarter"].isin(quarter_order)].copy()

    if "relation_group" in df.columns:
        df = df[df["relation_group"].str.lower().ne("none")].copy()

    if "entity" in df.columns:
        df = df[df["entity"].fillna("").astype(str).str.strip().ne("")].copy()

    if df.empty:
        return df

    df["source_company"] = df["current_company"]
    df["source_company_norm"] = df["current_company"].map(norm_text)
    df["source_ticker_norm"] = df["ticker"].map(norm_text) if "ticker" in df.columns else ""
    df["target_entity"] = df["entity"]
    df["target_entity_norm"] = df["entity"].map(norm_text)
    df["quarter_index"] = df["quarter"].map(quarter_to_int) if "quarter" in df.columns else np.nan

    return df.drop_duplicates()


def outlook_distribution(outlook: pd.DataFrame) -> pd.DataFrame:
    if outlook.empty:
        return pd.DataFrame()
    return (
        outlook.groupby(["quarter", "signal", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["quarter", "signal", "count"], ascending=[True, True, False])
    )


def make_company_quarter_signal_matrix(outlook: pd.DataFrame) -> pd.DataFrame:
    if outlook.empty:
        return pd.DataFrame()

    matrix = (
        outlook.pivot_table(
            index=["ticker", "current_company", "company_norm", "quarter", "quarter_index"],
            columns="signal",
            values="score",
            aggfunc="mean",
        )
        .reset_index()
    )

    for s in STANDARD_SIGNALS:
        if s not in matrix.columns:
            matrix[s] = np.nan

    # number of active non-zero signals in a company-quarter
    signal_cols = [s for s in STANDARD_SIGNALS if s in matrix.columns]
    matrix["active_signal_count"] = matrix[signal_cols].apply(lambda r: np.sum(r.fillna(0).abs() > 0), axis=1)
    matrix["mean_signal_score"] = matrix[signal_cols].mean(axis=1, skipna=True)

    return matrix.sort_values(["quarter_index", "ticker", "current_company"])


def build_company_lookup(signal_matrix: pd.DataFrame) -> pd.DataFrame:
    if signal_matrix.empty:
        return pd.DataFrame()
    lookup = signal_matrix[["ticker", "current_company", "company_norm"]].drop_duplicates()
    lookup["ticker_norm"] = lookup["ticker"].map(norm_text)
    return lookup


def match_relationship_targets(rel: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    if rel.empty or lookup.empty:
        return pd.DataFrame()

    # Match extracted target entity to known company name or ticker from outlook data.
    by_company = lookup.rename(columns={
        "ticker": "target_ticker",
        "current_company": "target_company",
        "company_norm": "target_entity_norm",
    })[["target_entity_norm", "target_ticker", "target_company"]]

    by_ticker = lookup.rename(columns={
        "ticker": "target_ticker",
        "current_company": "target_company",
        "ticker_norm": "target_entity_norm",
    })[["target_entity_norm", "target_ticker", "target_company"]]

    target_lookup = pd.concat([by_company, by_ticker], ignore_index=True).drop_duplicates()
    matched = rel.merge(target_lookup, on="target_entity_norm", how="left")
    matched["target_matched"] = matched["target_company"].notna()
    return matched


def build_propagation_events(outlook: pd.DataFrame, relationships: pd.DataFrame, quarter_order: list[str]) -> pd.DataFrame:
    """
    Candidate event definition:
    source company has signal s in quarter t, and a matched related target company
    has the same signal s in the next observed quarter t+1.

    This is not causal proof. It is a candidate propagation event table.
    """
    if outlook.empty or relationships.empty:
        return pd.DataFrame()

    matrix = make_company_quarter_signal_matrix(outlook)
    lookup = build_company_lookup(matrix)
    matched_edges = match_relationship_targets(relationships, lookup)
    matched_edges = matched_edges[matched_edges["target_matched"]].copy()

    if matched_edges.empty:
        return pd.DataFrame()

    q_to_next = {quarter_order[i]: quarter_order[i + 1] for i in range(len(quarter_order) - 1)}

    long_signals = outlook[outlook["signal"].isin(STANDARD_SIGNALS)].copy()
    long_signals = long_signals[[
        "ticker", "current_company", "company_norm", "quarter", "quarter_index", "signal", "label", "score", "evidence_chunk_ids", "notes"
    ]].drop_duplicates()

    source = long_signals.rename(columns={
        "ticker": "source_ticker",
        "current_company": "source_company_signal",
        "company_norm": "source_company_norm",
        "quarter": "source_quarter",
        "quarter_index": "source_quarter_index",
        "label": "source_label",
        "score": "source_score",
        "evidence_chunk_ids": "source_evidence_chunk_ids",
        "notes": "source_notes",
    })
    source["target_quarter"] = source["source_quarter"].map(q_to_next)
    source = source[source["target_quarter"].notna()].copy()

    target = long_signals.rename(columns={
        "ticker": "target_ticker_signal",
        "current_company": "target_company_signal",
        "company_norm": "target_company_norm",
        "quarter": "target_quarter",
        "quarter_index": "target_quarter_index",
        "label": "target_label",
        "score": "target_score",
        "evidence_chunk_ids": "target_evidence_chunk_ids",
        "notes": "target_notes",
    })

    # Link source signals to edges from same source company.
    edge_cols = [
        "source_company_norm", "target_company", "target_ticker", "target_entity", "relation_group",
        "entity_type", "relationship_type", "confidence", "evidence_chunk_ids"
    ]
    edge_cols = [c for c in edge_cols if c in matched_edges.columns]
    edges = matched_edges[edge_cols].drop_duplicates()

    st = source.merge(edges, on="source_company_norm", how="inner")
    events = st.merge(
        target,
        left_on=["target_company", "target_quarter", "signal"],
        right_on=["target_company_signal", "target_quarter", "signal"],
        how="inner",
    )

    if events.empty:
        return events

    events["source_active"] = events["source_score"].fillna(0).abs() > 0
    events["target_active"] = events["target_score"].fillna(0).abs() > 0
    events["same_direction"] = np.sign(events["source_score"].fillna(0)) == np.sign(events["target_score"].fillna(0))
    events["candidate_propagation"] = events["source_active"] & events["target_active"]

    # A simple propagation score for ranking, not causal inference.
    confidence_weight = events.get("confidence", pd.Series("medium", index=events.index)).map({"high": 1.0, "medium": 0.7, "low": 0.4}).fillna(0.5)
    events["propagation_score"] = (
        events["source_score"].fillna(0).abs()
        * events["target_score"].fillna(0).abs()
        * confidence_weight
        * np.where(events["same_direction"], 1.0, 0.5)
    )

    keep = [
        "source_ticker", "source_company_signal", "target_ticker", "target_company",
        "signal", "source_quarter", "target_quarter", "source_label", "target_label",
        "source_score", "target_score", "same_direction", "candidate_propagation", "propagation_score",
        "relation_group", "relationship_type", "entity_type", "confidence", "target_entity",
        "source_evidence_chunk_ids", "target_evidence_chunk_ids", "evidence_chunk_ids",
        "source_notes", "target_notes",
    ]
    keep = [c for c in keep if c in events.columns]
    events = events[keep].sort_values("propagation_score", ascending=False)
    return events


def build_exposure_features(outlook: pd.DataFrame, relationships: pd.DataFrame, quarter_order: list[str]) -> pd.DataFrame:
    """
    Exposure_j,s,t = average / sum of connected source signals at previous quarter.
    Produces target company-quarter exposure features by signal and relation group.
    """
    if outlook.empty or relationships.empty:
        return pd.DataFrame()

    matrix = make_company_quarter_signal_matrix(outlook)
    lookup = build_company_lookup(matrix)
    matched_edges = match_relationship_targets(relationships, lookup)
    matched_edges = matched_edges[matched_edges["target_matched"]].copy()

    if matched_edges.empty:
        return pd.DataFrame()

    q_to_next = {quarter_order[i]: quarter_order[i + 1] for i in range(len(quarter_order) - 1)}

    long_signals = outlook[outlook["signal"].isin(STANDARD_SIGNALS)].copy()
    long_signals = long_signals[["current_company", "company_norm", "ticker", "quarter", "signal", "score"]].drop_duplicates()
    src = long_signals.rename(columns={
        "current_company": "source_company",
        "company_norm": "source_company_norm",
        "ticker": "source_ticker",
        "quarter": "source_quarter",
        "score": "source_score",
    })
    src["target_quarter"] = src["source_quarter"].map(q_to_next)
    src = src[src["target_quarter"].notna()].copy()

    edge_cols = ["source_company_norm", "target_company", "target_ticker", "relation_group", "relationship_type", "confidence"]
    edge_cols = [c for c in edge_cols if c in matched_edges.columns]
    edges = matched_edges[edge_cols].drop_duplicates()

    exposure_long = src.merge(edges, on="source_company_norm", how="inner")
    exposure_long["source_score_abs"] = exposure_long["source_score"].fillna(0).abs()
    exposure_long["source_active"] = exposure_long["source_score_abs"] > 0

    group_cols = ["target_ticker", "target_company", "target_quarter", "signal"]
    exp = exposure_long.groupby(group_cols).agg(
        exposure_sum=("source_score", "sum"),
        exposure_mean=("source_score", "mean"),
        exposure_abs_sum=("source_score_abs", "sum"),
        active_neighbor_count=("source_active", "sum"),
        neighbor_count=("source_company", "nunique"),
    ).reset_index()

    # Add relation-specific exposure columns.
    rel_exp = exposure_long.groupby(group_cols + ["relation_group"]).agg(
        relation_exposure_sum=("source_score", "sum"),
        relation_active_neighbor_count=("source_active", "sum"),
    ).reset_index()

    rel_pivot_score = rel_exp.pivot_table(
        index=group_cols,
        columns="relation_group",
        values="relation_exposure_sum",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    rel_pivot_score = rel_pivot_score.rename(columns={c: f"exposure_{c}_sum" for c in rel_pivot_score.columns if c not in group_cols})

    exp = exp.merge(rel_pivot_score, on=group_cols, how="left")

    # Merge target's actual signal at target quarter for prediction-ready table.
    target = long_signals.rename(columns={
        "ticker": "target_ticker",
        "current_company": "target_company",
        "quarter": "target_quarter",
        "score": "target_score",
    })[["target_ticker", "target_company", "target_quarter", "signal", "target_score"]]

    exp = exp.merge(target, on=["target_ticker", "target_company", "target_quarter", "signal"], how="left")
    exp["target_active"] = exp["target_score"].fillna(0).abs() > 0
    return exp.sort_values(["target_quarter", "target_ticker", "signal"])


def concept_distribution(concepts: pd.DataFrame) -> pd.DataFrame:
    if concepts.empty:
        return pd.DataFrame()
    cols = [c for c in CONCEPT_COLUMNS if c in concepts.columns]
    rows = []
    for c in cols:
        rows.append({
            "concept": c,
            "mentions": int(concepts[c].sum()),
            "transcripts": int(concepts[c].notna().sum()),
            "mention_rate": float(concepts[c].mean()),
        })
    return pd.DataFrame(rows).sort_values("mentions", ascending=False)


def save_table(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        print(f"SKIP empty table: {path.name}")
        return
    df.to_csv(path, index=False)
    print(f"SAVED {path} rows={len(df):,}")


def plot_outlook_label_distribution(dist: pd.DataFrame, fig_dir: Path) -> None:
    if dist.empty:
        return
    df = dist.groupby(["signal", "label"], as_index=False)["count"].sum()
    pivot = df.pivot(index="signal", columns="label", values="count").fillna(0)
    pivot = pivot.reindex([s for s in STANDARD_SIGNALS if s in pivot.index])

    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title("Outlook label distribution by signal")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out = fig_dir / "outlook_label_distribution_by_signal.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"SAVED {out}")


def plot_quarter_signal_counts(outlook: pd.DataFrame, fig_dir: Path) -> None:
    if outlook.empty:
        return
    df = outlook[outlook["signal"].isin(STANDARD_SIGNALS)].copy()
    df["active"] = df["score"].fillna(0).abs() > 0
    counts = df.groupby(["quarter", "signal"], as_index=False)["active"].sum()
    pivot = counts.pivot(index="signal", columns="quarter", values="active").fillna(0)
    pivot = pivot.reindex([s for s in STANDARD_SIGNALS if s in pivot.index])

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Active outlook signals by quarter")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Active signal count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out = fig_dir / "active_outlook_signals_by_quarter.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"SAVED {out}")


def plot_concept_distribution(concept_dist: pd.DataFrame, fig_dir: Path) -> None:
    if concept_dist.empty:
        return
    df = concept_dist.sort_values("mentions", ascending=True)
    ax = df.plot(kind="barh", x="concept", y="mentions", legend=False, figsize=(10, 7))
    ax.set_title("Supply-chain concept mentions")
    ax.set_xlabel("Transcript count")
    ax.set_ylabel("Concept")
    plt.tight_layout()
    out = fig_dir / "concept_mentions.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"SAVED {out}")


def plot_relationship_counts(rel: pd.DataFrame, fig_dir: Path) -> None:
    if rel.empty or "relation_group" not in rel.columns:
        return
    counts = rel["relation_group"].value_counts().sort_values(ascending=True)
    ax = counts.plot(kind="barh", figsize=(9, 5))
    ax.set_title("Extracted relationship groups")
    ax.set_xlabel("Edge count")
    ax.set_ylabel("Relation group")
    plt.tight_layout()
    out = fig_dir / "relationship_group_counts.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"SAVED {out}")


def plot_top_company_activity(signal_matrix: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    if signal_matrix.empty:
        return
    top = (
        signal_matrix.groupby(["ticker", "current_company"], as_index=False)["active_signal_count"]
        .sum()
        .sort_values("active_signal_count", ascending=False)
        .head(top_n)
    )
    if top.empty:
        return
    labels = top["ticker"].astype(str) + " - " + top["current_company"].astype(str).str.slice(0, 30)
    ax = top.assign(label=labels).sort_values("active_signal_count", ascending=True).plot(
        kind="barh", x="label", y="active_signal_count", legend=False, figsize=(11, 8)
    )
    ax.set_title(f"Top {top_n} companies by active outlook signal count")
    ax.set_xlabel("Active signal count")
    ax.set_ylabel("Company")
    plt.tight_layout()
    out = fig_dir / "top_company_signal_activity.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"SAVED {out}")


def write_summary(out_dir: Path, **kwargs) -> None:
    lines = ["# Information Propagation Analysis Summary\n"]
    for k, v in kwargs.items():
        lines.append(f"- {k}: {v}")
    path = out_dir / "analysis_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"SAVED {path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")
    quarter_order = [q.strip() for q in args.quarter_order.split(",") if q.strip()]

    print("=" * 70)
    print("Information propagation analysis")
    print("input_dir:", input_dir)
    print("out_dir:", out_dir)
    print("quarter_order:", quarter_order)
    print("=" * 70)

    outlook_raw = read_sharded_csv(input_dir, "outlook_q2q3_shard*_of006.csv", args.outlook_csv)
    concepts_raw = read_sharded_csv(input_dir, "concepts_q2q3_shard*_of006.csv")
    relationships_raw = read_sharded_csv(input_dir, "relationships_q2q3_shard*_of006.csv")

    print(f"Loaded outlook rows: {len(outlook_raw):,}")
    print(f"Loaded concept rows: {len(concepts_raw):,}")
    print(f"Loaded relationship rows: {len(relationships_raw):,}")

    outlook = clean_outlook(outlook_raw, quarter_order)
    concepts = clean_concepts(concepts_raw, quarter_order)
    relationships = clean_relationships(relationships_raw, quarter_order)

    outlook_dist = outlook_distribution(outlook)
    signal_matrix = make_company_quarter_signal_matrix(outlook)
    concept_dist = concept_distribution(concepts)
    propagation_events = build_propagation_events(outlook, relationships, quarter_order)
    exposure_features = build_exposure_features(outlook, relationships, quarter_order)

    save_table(outlook, out_dir / "cleaned_outlook.csv")
    save_table(outlook_dist, out_dir / "outlook_signal_label_distribution.csv")
    save_table(signal_matrix, out_dir / "company_quarter_signal_matrix.csv")
    save_table(concepts, out_dir / "cleaned_concepts.csv")
    save_table(concept_dist, out_dir / "concept_distribution.csv")
    save_table(relationships, out_dir / "cleaned_relationships.csv")
    save_table(propagation_events, out_dir / "candidate_propagation_events.csv")
    save_table(exposure_features, out_dir / "network_exposure_features.csv")

    plot_outlook_label_distribution(outlook_dist, fig_dir)
    plot_quarter_signal_counts(outlook, fig_dir)
    plot_concept_distribution(concept_dist, fig_dir)
    plot_relationship_counts(relationships, fig_dir)
    plot_top_company_activity(signal_matrix, fig_dir, args.top_companies)

    write_summary(
        out_dir,
        outlook_raw_rows=len(outlook_raw),
        outlook_clean_rows=len(outlook),
        unique_transcripts=outlook["doc_id"].nunique() if not outlook.empty else 0,
        unique_companies=outlook["current_company"].nunique() if not outlook.empty else 0,
        quarters=", ".join(sorted(outlook["quarter"].dropna().unique())) if not outlook.empty else "",
        concept_rows=len(concepts),
        relationship_rows=len(relationships),
        candidate_propagation_events=len(propagation_events),
        exposure_feature_rows=len(exposure_features),
        note="Candidate propagation events are descriptive co-occurrences over network edges, not causal proof.",
    )

    print("\nDONE.")
    print("Main outputs:")
    print("-", out_dir / "cleaned_outlook.csv")
    print("-", out_dir / "company_quarter_signal_matrix.csv")
    print("-", out_dir / "candidate_propagation_events.csv")
    print("-", out_dir / "network_exposure_features.csv")
    print("-", fig_dir)


if __name__ == "__main__":
    main()
