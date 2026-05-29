#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-part network analysis for earnings-call LLM outputs.

Part A. Cross-quarter lead-lag contagion
----------------------------------------
Question:
    If a connected source firm has signal s in quarter t,
    does the target firm show the same/same-direction signal in quarter t+1?

Use case:
    Early prediction / leading indicator analysis.

Outputs:
    cross_quarter_events.csv
    cross_quarter_summary.csv
    cross_quarter_prediction_accuracy.csv
    figures/cross_quarter_accuracy_by_signal.png
    figures/cross_quarter_accuracy_by_relation.png

Part B. Same-quarter network correlation
----------------------------------------
Question:
    In the same quarter, do connected firms show similar signal states?

Use case:
    Because earnings-call dates inside one quarter differ,
    firms in the same reporting quarter can still reveal contemporaneous
    network correlation / event clustering.

Outputs:
    same_quarter_events.csv
    same_quarter_summary.csv
    same_quarter_correlation_by_signal_relation.csv
    figures/same_quarter_similarity_by_signal.png
    figures/same_quarter_similarity_by_relation.png

Important interpretation
------------------------
- Cross-quarter analysis is closer to "early prediction".
- Same-quarter analysis is not strict forecasting unless exact call dates are used.
  It is best interpreted as contemporaneous network co-movement/correlation.
- If your evidence packages or CSVs include exact earnings-call dates later,
  this script can be extended from quarter-level to date-level ordering.

Run:
----
cd ~/sem2/RAG

python run_two_part_network_prediction_analysis.py \
  --rag-output-dir rag_chroma_output \
  --out-dir rag_chroma_output/two_part_network_prediction_analysis \
  --start-quarter 2019Q2 \
  --end-quarter 2026Q2

Windows:
--------
cd E:\\Projects\\EearningAlz\\RAG

python run_two_part_network_prediction_analysis.py ^
  --rag-output-dir rag_chroma_output ^
  --out-dir rag_chroma_output\\two_part_network_prediction_analysis ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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
    "demand_outlook": "demand_outlook",
    "supply_outlook": "supply_outlook",
    "margin_outlook": "margin_outlook",
    "capex_outlook": "capex_outlook",
    "inventory_outlook": "inventory_outlook",
    "pricing_outlook": "pricing_outlook",
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

POSITIVE_LABELS = {"positive", "improving", "increase"}
NEGATIVE_LABELS = {"negative", "worsening", "decrease"}
NEUTRAL_LABELS = {"neutral", "stable"}
MIXED_LABELS = {"mixed"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag-output-dir", default="rag_chroma_output")
    p.add_argument("--out-dir", default="rag_chroma_output/two_part_network_prediction_analysis")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--include-self-edges", action="store_true")
    p.add_argument("--min-exposed-for-plot", type=int, default=5)
    p.add_argument(
        "--use-quarter-specific-relationships",
        action="store_true",
        help=(
            "If set, only relationships extracted in the source/target quarter are used. "
            "Default uses all matched relationships as structural links, which usually gives more complete network coverage."
        ),
    )
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def quarter_to_index(q: str) -> float:
    q = str(q).strip()
    m = re.match(r"^(\d{4})Q([1-4])$", q)
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def index_to_quarter(idx: int) -> str:
    year = (idx - 1) // 4
    q = idx - year * 4
    return f"{year}Q{q}"


def adjacent_pairs(quarters: list[str]) -> list[tuple[str, str]]:
    q_sorted = sorted(
        [q for q in quarters if not pd.isna(quarter_to_index(q))],
        key=quarter_to_index,
    )
    q_set = set(q_sorted)
    pairs = []
    for q in q_sorted:
        nxt = index_to_quarter(int(quarter_to_index(q)) + 1)
        if nxt in q_set:
            pairs.append((q, nxt))
    return pairs


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9&.\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    replacements = {
        " corporation": " corp",
        " incorporated": " inc",
        " company": " co",
        " limited": " ltd",
        " technologies": " tech",
        " technology": " tech",
        " international": " intl",
        " holdings": "",
        " holding": "",
        " group": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"\bthe\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def label_set(x) -> set[str]:
    if pd.isna(x):
        return set()
    return {v.strip().lower() for v in str(x).split(";") if v.strip()}


def label_direction(label: str, score: float | None = None) -> str:
    labels = label_set(label)
    if labels & POSITIVE_LABELS:
        return "positive"
    if labels & NEGATIVE_LABELS:
        return "negative"
    if labels & MIXED_LABELS:
        return "mixed"
    if labels & NEUTRAL_LABELS:
        return "neutral"
    if score is not None and not pd.isna(score):
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"
    return "not_mentioned"


def is_active_score(x) -> bool:
    return not pd.isna(x) and abs(float(x)) > 0


def discover_extraction_csvs(rag_dir: Path):
    outlook, relationships = [], []
    for f in sorted(rag_dir.rglob("*.csv")):
        name = f.name.lower()
        if name.startswith("outlook_"):
            outlook.append(f)
        elif name.startswith("relationships_"):
            relationships.append(f)
    return outlook, relationships


def read_many_csv(files: list[Path], kind: str) -> pd.DataFrame:
    frames = []
    print(f"\nLoading {kind}: {len(files)} files")
    for f in files:
        try:
            if f.stat().st_size == 0:
                continue
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            frames.append(df)
            print(f"  loaded {f} rows={len(df):,}")
        except Exception as e:
            print(f"  WARNING failed reading {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def filter_quarter_range(df: pd.DataFrame, q_col: str, start_q: str, end_q: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_qidx"] = out[q_col].map(quarter_to_index)
    out = out[out["_qidx"].notna()].copy()
    if start_q:
        out = out[out["_qidx"] >= quarter_to_index(start_q)].copy()
    if end_q:
        out = out[out["_qidx"] <= quarter_to_index(end_q)].copy()
    return out.drop(columns=["_qidx"])


def clean_outlook(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    required = ["doc_id", "ticker", "current_company", "quarter", "signal", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Outlook missing required columns: {missing}")

    out = df.copy()
    for c in required:
        out[c] = out[c].astype(str).str.strip()

    out["signal_raw"] = out["signal"]
    out["signal"] = out["signal"].str.lower().map(lambda x: SIGNAL_MAP.get(x, x))
    out = out[out["signal"].isin(STANDARD_SIGNALS)].copy()

    out["label"] = out["label"].astype(str).str.strip().str.lower().replace({"nan": "not_mentioned"})
    out["score"] = out["label"].map(LABEL_SCORE)
    out["company_norm"] = out["current_company"].map(norm_text)
    out["ticker_norm"] = out["ticker"].map(norm_text)
    out["company_node"] = np.where(
        out["ticker"].astype(str).str.strip().ne(""),
        "COMPANY::" + out["ticker"].astype(str).str.strip(),
        "COMPANY::" + out["company_norm"],
    )
    out["quarter_index"] = out["quarter"].map(quarter_to_index)
    out = filter_quarter_range(out, "quarter", start_q, end_q)

    group_cols = [
        "company_node",
        "ticker",
        "current_company",
        "company_norm",
        "ticker_norm",
        "quarter",
        "quarter_index",
        "signal",
    ]

    agg = {
        "score": "mean",
        "label": lambda x: ";".join(sorted(set(str(v) for v in x if str(v) and str(v) != "nan"))),
        "source_file": lambda x: "|".join(sorted(set(str(v) for v in x.dropna()))),
    }

    out = out.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out["direction"] = out.apply(lambda r: label_direction(r["label"], r["score"]), axis=1)
    out["is_active"] = out["score"].notna() & (out["score"].abs() > 0)

    return out


def clean_relationships(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    required = ["ticker", "current_company", "quarter", "relation_group", "entity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Relationships missing required columns: {missing}")

    rel = df.copy()
    for c in rel.columns:
        if rel[c].dtype == "object":
            rel[c] = rel[c].astype(str).str.strip()

    rel = rel[rel["entity"].fillna("").astype(str).str.strip().ne("")].copy()
    rel = rel[rel["relation_group"].fillna("").astype(str).str.lower().ne("none")].copy()

    rel["source_company_node"] = np.where(
        rel["ticker"].astype(str).str.strip().ne(""),
        "COMPANY::" + rel["ticker"].astype(str).str.strip(),
        "COMPANY::" + rel["current_company"].map(norm_text),
    )
    rel["source_company_norm"] = rel["current_company"].map(norm_text)
    rel["source_ticker_norm"] = rel["ticker"].map(norm_text)
    rel["target_entity_norm"] = rel["entity"].map(norm_text)
    rel["quarter_index"] = rel["quarter"].map(quarter_to_index)
    rel = filter_quarter_range(rel, "quarter", start_q, end_q)

    rows = []
    for _, r in rel.iterrows():
        groups = [g.strip() for g in str(r["relation_group"]).split("|") if g.strip()]
        if not groups:
            groups = [str(r["relation_group"]).strip()]
        for g in groups:
            rr = r.copy()
            rr["relation_group_clean"] = g
            rows.append(rr)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates()


def build_company_lookup(outlook: pd.DataFrame):
    company_map, ticker_map, meta = {}, {}, {}
    base = outlook[["company_node", "ticker", "current_company", "company_norm", "ticker_norm"]].drop_duplicates()

    for _, r in base.iterrows():
        node = str(r["company_node"])
        cname = str(r["company_norm"])
        ticker = str(r["ticker_norm"])
        if cname:
            company_map[cname] = node
        if ticker:
            ticker_map[ticker] = node
        meta[node] = {
            "ticker": str(r["ticker"]),
            "company": str(r["current_company"]),
        }
    return company_map, ticker_map, meta


def match_entity_to_company(entity_norm: str, company_map: dict, ticker_map: dict) -> str:
    if not entity_norm:
        return ""
    if entity_norm in ticker_map:
        return ticker_map[entity_norm]
    if entity_norm in company_map:
        return company_map[entity_norm]

    if len(entity_norm) >= 5:
        candidates = []
        for cname, node in company_map.items():
            if not cname or len(cname) < 5:
                continue
            if entity_norm in cname or cname in entity_norm:
                candidates.append((abs(len(cname) - len(entity_norm)), node))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
    return ""


def make_outlook_lookup(outlook: pd.DataFrame):
    return {
        (r.company_node, r.quarter, r.signal): r
        for r in outlook.itertuples(index=False)
    }


def prepare_matched_relationships(relationships: pd.DataFrame, outlook: pd.DataFrame, include_self_edges: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    company_map, ticker_map, meta = build_company_lookup(outlook)

    rel = relationships.copy()
    rel["target_company_node"] = rel["target_entity_norm"].map(
        lambda x: match_entity_to_company(x, company_map, ticker_map)
    )

    unmatched = rel[rel["target_company_node"].fillna("").eq("")].copy()
    matched = rel[rel["target_company_node"].fillna("").ne("")].copy()

    if not include_self_edges:
        matched = matched[matched["source_company_node"] != matched["target_company_node"]].copy()

    matched = matched.drop_duplicates(
        subset=[
            "source_company_node",
            "target_company_node",
            "relation_group_clean",
            "relationship_type",
            "quarter",
        ]
    )

    return matched, unmatched, meta


def select_relationships_for_window(
    matched_rel: pd.DataFrame,
    source_q: str,
    target_q: str,
    use_quarter_specific: bool,
) -> pd.DataFrame:
    if not use_quarter_specific:
        return matched_rel

    rel_q = matched_rel[matched_rel["quarter"].isin([source_q, target_q])].copy()
    return rel_q


def build_events_for_pair(
    outlook_lookup: dict,
    meta: dict,
    matched_rel: pd.DataFrame,
    source_q: str,
    target_q: str,
    mode: str,
) -> pd.DataFrame:
    """
    mode:
      cross_quarter: source_q -> target_q
      same_quarter: source_q == target_q
    """

    rows = []

    for _, edge in matched_rel.iterrows():
        source_node = str(edge["source_company_node"])
        target_node = str(edge["target_company_node"])

        smeta = meta.get(source_node, {})
        tmeta = meta.get(target_node, {})

        for signal in STANDARD_SIGNALS:
            srow = outlook_lookup.get((source_node, source_q, signal))
            trow = outlook_lookup.get((target_node, target_q, signal))

            if srow is None or trow is None:
                continue

            source_label = str(srow.label)
            target_label = str(trow.label)

            source_score = float(srow.score) if not pd.isna(srow.score) else np.nan
            target_score = float(trow.score) if not pd.isna(trow.score) else np.nan

            source_direction = label_direction(source_label, source_score)
            target_direction = label_direction(target_label, target_score)

            source_active = is_active_score(source_score)
            target_active = is_active_score(target_score)

            exact_match = source_active and target_active and source_label == target_label
            direction_match = source_active and target_active and source_direction == target_direction

            # Prediction interpretation:
            # If source has an active signal, predict target will have the same direction.
            # Correct if target actually has same direction.
            predicted_positive = source_active
            actual_positive = target_active and (target_direction == source_direction)
            prediction_correct = bool(predicted_positive and actual_positive) if predicted_positive else np.nan

            rows.append({
                "analysis_mode": mode,
                "source_quarter": source_q,
                "target_quarter": target_q,
                "source_node": source_node,
                "source_ticker": smeta.get("ticker", ""),
                "source_company": smeta.get("company", ""),
                "target_node": target_node,
                "target_ticker": tmeta.get("ticker", ""),
                "target_company": tmeta.get("company", ""),
                "signal": signal,
                "source_label": source_label,
                "target_label": target_label,
                "source_score": source_score,
                "target_score": target_score,
                "source_direction": source_direction,
                "target_direction": target_direction,
                "source_active": source_active,
                "target_active": target_active,
                "exact_match": exact_match,
                "direction_match": direction_match,
                "predicted_positive": predicted_positive,
                "actual_positive": actual_positive,
                "prediction_correct": prediction_correct,
                "relation_group": str(edge.get("relation_group_clean", "")),
                "relationship_type": str(edge.get("relationship_type", "")),
                "entity_type": str(edge.get("entity_type", "")),
                "confidence": str(edge.get("confidence", "")),
                "extracted_entity": str(edge.get("entity", "")),
            })

    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    exp = events[events["source_active"]].copy()
    if exp.empty:
        return pd.DataFrame()

    group_cols = [
        "analysis_mode",
        "source_quarter",
        "target_quarter",
        "signal",
        "source_label",
        "source_direction",
        "relation_group",
    ]

    rows = []

    for keys, g in exp.groupby(group_cols, dropna=False):
        (
            analysis_mode,
            source_q,
            target_q,
            signal,
            source_label,
            source_direction,
            relation_group,
        ) = keys

        exposed = len(g)
        target_active = int(g["target_active"].sum())
        exact = int(g["exact_match"].sum())
        direction = int(g["direction_match"].sum())

        # Prediction accuracy among source-active exposures
        valid_pred = g[g["prediction_correct"].notna()].copy()
        correct = int(valid_pred["prediction_correct"].sum()) if not valid_pred.empty else 0
        accuracy = correct / len(valid_pred) if len(valid_pred) else np.nan

        rows.append({
            "analysis_mode": analysis_mode,
            "source_quarter": source_q,
            "target_quarter": target_q,
            "signal": signal,
            "source_label": source_label,
            "source_direction": source_direction,
            "relation_group": relation_group,
            "exposed_edges": exposed,
            "target_active_edges": target_active,
            "exact_match_edges": exact,
            "direction_match_edges": direction,
            "exact_match_rate": exact / exposed if exposed else np.nan,
            "direction_match_rate": direction / exposed if exposed else np.nan,
            "target_active_rate": target_active / exposed if exposed else np.nan,
            "non_exact_edges": exposed - exact,
            "non_direction_edges": exposed - direction,
            "non_exact_rate": (exposed - exact) / exposed if exposed else np.nan,
            "non_direction_rate": (exposed - direction) / exposed if exposed else np.nan,
            "prediction_correct_edges": correct,
            "prediction_accuracy": accuracy,
        })

    return pd.DataFrame(rows).sort_values(
        ["analysis_mode", "source_quarter", "target_quarter", "signal", "relation_group"]
    )


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["analysis_mode", "signal", "source_label", "source_direction", "relation_group"]

    for keys, g in summary.groupby(group_cols, dropna=False):
        analysis_mode, signal, source_label, source_direction, relation_group = keys

        exposed = int(g["exposed_edges"].sum())
        target_active = int(g["target_active_edges"].sum())
        exact = int(g["exact_match_edges"].sum())
        direction = int(g["direction_match_edges"].sum())
        correct = int(g["prediction_correct_edges"].sum())

        rows.append({
            "analysis_mode": analysis_mode,
            "signal": signal,
            "source_label": source_label,
            "source_direction": source_direction,
            "relation_group": relation_group,
            "exposed_edges": exposed,
            "target_active_edges": target_active,
            "exact_match_edges": exact,
            "direction_match_edges": direction,
            "exact_match_rate": exact / exposed if exposed else np.nan,
            "direction_match_rate": direction / exposed if exposed else np.nan,
            "target_active_rate": target_active / exposed if exposed else np.nan,
            "non_exact_edges": exposed - exact,
            "non_direction_edges": exposed - direction,
            "non_exact_rate": (exposed - exact) / exposed if exposed else np.nan,
            "non_direction_rate": (exposed - direction) / exposed if exposed else np.nan,
            "prediction_correct_edges": correct,
            "prediction_accuracy": correct / exposed if exposed else np.nan,
            "num_windows": int(g[["source_quarter", "target_quarter"]].drop_duplicates().shape[0]),
        })

    return pd.DataFrame(rows).sort_values(
        ["analysis_mode", "prediction_accuracy", "exposed_edges"],
        ascending=[True, False, False],
    )


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"SAVED {path} rows={len(df):,}")


def plot_rate_by_group(agg: pd.DataFrame, mode: str, group_col: str, rate_col: str, out_png: Path, min_exposed: int):
    d = agg[(agg["analysis_mode"] == mode) & (agg["exposed_edges"] >= min_exposed)].copy()
    if d.empty:
        return

    g = (
        d.groupby(group_col, as_index=False)
        .agg(
            exposed_edges=("exposed_edges", "sum"),
            correct_edges=("prediction_correct_edges", "sum"),
            direction_edges=("direction_match_edges", "sum"),
        )
    )
    if rate_col == "prediction_accuracy":
        g[rate_col] = g["correct_edges"] / g["exposed_edges"]
    else:
        g[rate_col] = g["direction_edges"] / g["exposed_edges"]

    g = g.sort_values(rate_col, ascending=False)

    ax = g.sort_values(rate_col).plot(kind="barh", x=group_col, y=rate_col, legend=False, figsize=(10, 6))
    ax.set_title(f"{mode}: {rate_col} by {group_col}")
    ax.set_xlabel(rate_col)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def plot_window_counts(events: pd.DataFrame, mode: str, out_png: Path):
    d = events[events["analysis_mode"] == mode].copy()
    if d.empty:
        return
    counts = d.groupby(["source_quarter", "target_quarter"]).size().reset_index(name="event_rows")
    counts["window"] = counts["source_quarter"] + "→" + counts["target_quarter"]

    ax = counts.plot(kind="bar", x="window", y="event_rows", legend=False, figsize=(14, 5))
    ax.set_title(f"{mode}: event rows by quarter window")
    ax.set_xlabel("Quarter window")
    ax.set_ylabel("Event rows")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def main():
    args = parse_args()

    rag_dir = Path(args.rag_output_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")

    print("=" * 90)
    print("Two-part network prediction analysis")
    print("rag_output_dir:", rag_dir)
    print("out_dir:", out_dir)
    print("quarter range:", args.start_quarter or "ALL", "to", args.end_quarter or "ALL")
    print("use_quarter_specific_relationships:", args.use_quarter_specific_relationships)
    print("=" * 90)

    outlook_files, rel_files = discover_extraction_csvs(rag_dir)

    manifest = pd.DataFrame({
        "kind": ["outlook"] * len(outlook_files) + ["relationships"] * len(rel_files),
        "path": [str(x) for x in outlook_files + rel_files],
    })
    save_csv(manifest, out_dir / "input_file_manifest.csv")

    raw_outlook = read_many_csv(outlook_files, "outlook")
    raw_rel = read_many_csv(rel_files, "relationships")

    if raw_outlook.empty:
        raise RuntimeError("No outlook extraction files found.")
    if raw_rel.empty:
        raise RuntimeError("No relationship extraction files found.")

    outlook = clean_outlook(raw_outlook, args.start_quarter, args.end_quarter)
    relationships = clean_relationships(raw_rel, args.start_quarter, args.end_quarter)

    save_csv(outlook, out_dir / "cleaned_outlook_all.csv")
    save_csv(relationships, out_dir / "cleaned_relationships_all.csv")

    matched_rel, unmatched_rel, meta = prepare_matched_relationships(
        relationships,
        outlook,
        include_self_edges=args.include_self_edges,
    )

    save_csv(matched_rel, out_dir / "matched_company_relationships.csv")
    save_csv(unmatched_rel, out_dir / "unmatched_relationship_entities.csv")

    outlook_lookup = make_outlook_lookup(outlook)

    quarters = sorted(outlook["quarter"].dropna().unique(), key=quarter_to_index)
    pairs = adjacent_pairs(quarters)

    print("\nAvailable quarters:")
    print(quarters)
    print("\nAdjacent pairs:")
    for s, t in pairs:
        print(f"  {s} -> {t}")

    # ============================================================
    # Part A: cross-quarter lead-lag analysis
    # ============================================================

    cross_events_list = []

    for source_q, target_q in pairs:
        rel_for_pair = select_relationships_for_window(
            matched_rel,
            source_q,
            target_q,
            use_quarter_specific=args.use_quarter_specific_relationships,
        )
        if rel_for_pair.empty:
            continue

        events = build_events_for_pair(
            outlook_lookup,
            meta,
            rel_for_pair,
            source_q,
            target_q,
            mode="cross_quarter",
        )
        if not events.empty:
            cross_events_list.append(events)

        print(f"cross_quarter {source_q}->{target_q}: rel={len(rel_for_pair):,}, events={len(events):,}")

    cross_events = pd.concat(cross_events_list, ignore_index=True).drop_duplicates() if cross_events_list else pd.DataFrame()
    cross_summary = summarize_events(cross_events, "cross_quarter")
    cross_agg = aggregate_summary(cross_summary)

    save_csv(cross_events, out_dir / "cross_quarter_events.csv")
    save_csv(cross_summary, out_dir / "cross_quarter_summary_by_window_signal_relation.csv")
    save_csv(cross_agg, out_dir / "cross_quarter_prediction_accuracy.csv")

    # ============================================================
    # Part B: same-quarter correlation analysis
    # ============================================================

    same_events_list = []

    for q in quarters:
        rel_for_q = matched_rel[matched_rel["quarter"].eq(q)].copy() if args.use_quarter_specific_relationships else matched_rel
        if rel_for_q.empty:
            continue

        events = build_events_for_pair(
            outlook_lookup,
            meta,
            rel_for_q,
            q,
            q,
            mode="same_quarter",
        )
        if not events.empty:
            same_events_list.append(events)

        print(f"same_quarter {q}: rel={len(rel_for_q):,}, events={len(events):,}")

    same_events = pd.concat(same_events_list, ignore_index=True).drop_duplicates() if same_events_list else pd.DataFrame()
    same_summary = summarize_events(same_events, "same_quarter")
    same_agg = aggregate_summary(same_summary)

    save_csv(same_events, out_dir / "same_quarter_events.csv")
    save_csv(same_summary, out_dir / "same_quarter_summary_by_quarter_signal_relation.csv")
    save_csv(same_agg, out_dir / "same_quarter_correlation_by_signal_relation.csv")

    # Combined
    combined_events = pd.concat([cross_events, same_events], ignore_index=True).drop_duplicates()
    combined_summary = pd.concat([cross_summary, same_summary], ignore_index=True).drop_duplicates()
    combined_agg = pd.concat([cross_agg, same_agg], ignore_index=True).drop_duplicates()

    save_csv(combined_events, out_dir / "combined_events_cross_and_same_quarter.csv")
    save_csv(combined_summary, out_dir / "combined_summary_cross_and_same_quarter.csv")
    save_csv(combined_agg, out_dir / "combined_accuracy_correlation_summary.csv")

    # Figures
    plot_window_counts(cross_events, "cross_quarter", fig_dir / "cross_quarter_event_rows_by_window.png")
    plot_window_counts(same_events, "same_quarter", fig_dir / "same_quarter_event_rows_by_quarter.png")

    plot_rate_by_group(cross_agg, "cross_quarter", "signal", "prediction_accuracy", fig_dir / "cross_quarter_accuracy_by_signal.png", args.min_exposed_for_plot)
    plot_rate_by_group(cross_agg, "cross_quarter", "relation_group", "prediction_accuracy", fig_dir / "cross_quarter_accuracy_by_relation.png", args.min_exposed_for_plot)

    plot_rate_by_group(same_agg, "same_quarter", "signal", "direction_match_rate", fig_dir / "same_quarter_similarity_by_signal.png", args.min_exposed_for_plot)
    plot_rate_by_group(same_agg, "same_quarter", "relation_group", "direction_match_rate", fig_dir / "same_quarter_similarity_by_relation.png", args.min_exposed_for_plot)

    # Markdown report
    lines = []
    lines.append("# Two-Part Network Prediction Analysis")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This analysis separates network signal analysis into two parts:")
    lines.append("")
    lines.append("1. **Cross-quarter lead-lag analysis**: source firm signal in quarter t is used to predict whether a connected target firm shows the same-direction signal in quarter t+1.")
    lines.append("2. **Same-quarter network correlation analysis**: connected firms are compared within the same quarter to study contemporaneous signal co-movement.")
    lines.append("")
    lines.append("The cross-quarter part is closer to early prediction. The same-quarter part is better interpreted as network correlation unless exact earnings-call dates are used.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Cleaned outlook rows: {len(outlook):,}")
    lines.append(f"- Cleaned relationship rows: {len(relationships):,}")
    lines.append(f"- Matched company relationships: {len(matched_rel):,}")
    lines.append(f"- Unmatched relationship entities: {len(unmatched_rel):,}")
    lines.append(f"- Available quarters: {', '.join(quarters)}")
    lines.append(f"- Adjacent quarter windows: {len(pairs)}")
    lines.append("")
    lines.append("## Part A: Cross-quarter lead-lag prediction")
    lines.append("")
    lines.append(f"- Cross-quarter event rows: {len(cross_events):,}")
    if not cross_agg.empty:
        top = cross_agg[cross_agg["exposed_edges"] >= args.min_exposed_for_plot].head(20)
        lines.append("")
        lines.append("Top cross-quarter prediction results:")
        lines.append("")
        lines.append(top.to_markdown(index=False))
    lines.append("")
    lines.append("## Part B: Same-quarter network correlation")
    lines.append("")
    lines.append(f"- Same-quarter event rows: {len(same_events):,}")
    if not same_agg.empty:
        top = same_agg[same_agg["exposed_edges"] >= args.min_exposed_for_plot].head(20)
        lines.append("")
        lines.append("Top same-quarter correlation results:")
        lines.append("")
        lines.append(top.to_markdown(index=False))
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `figures/cross_quarter_event_rows_by_window.png`")
    lines.append("- `figures/cross_quarter_accuracy_by_signal.png`")
    lines.append("- `figures/cross_quarter_accuracy_by_relation.png`")
    lines.append("- `figures/same_quarter_event_rows_by_quarter.png`")
    lines.append("- `figures/same_quarter_similarity_by_signal.png`")
    lines.append("- `figures/same_quarter_similarity_by_relation.png`")

    report_path = out_dir / "two_part_analysis_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"SAVED {report_path}")

    print("\nDONE.")
    print("Main report:")
    print(report_path)


if __name__ == "__main__":
    main()
