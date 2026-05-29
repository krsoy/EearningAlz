#!/usr/bin/env python3
"""
Analyze signal contagion / transmissibility across extracted inter-firm relationships.

This script answers questions like:

  "If a source company has margin_outlook = improving in 2025Q2,
   do its upstream/downstream/related firms show margin_outlook = improving in 2025Q3?"

It computes directional transmission statistics:
  source relation_group -> target
  source signal/label at source quarter
  target same signal/label at target quarter

Important:
- True contagion analysis requires company-to-company edges.
- LLM relationships often contain generic entities like "cloud customers" or "suppliers".
  Those are useful for visualization, but cannot be counted as target-company infection
  unless they can be matched to another company in the dataset.
- Therefore this script reports both:
    1. matched_company_edges
    2. unmatched_generic_entities

Inputs:
  outlook_q2q3_shard*_of006.csv
  relationships_q2q3_shard*_of006.csv

Outputs:
  contagion_events.csv
  contagion_summary_by_signal_label_relation.csv
  contagion_summary_margin_improving.csv
  unmatched_relationship_entities.csv
  contagion_summary.md
  figures/*.png
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
    _SCRIPT_DIR = Path(__file__).resolve().parent   # .../test/
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default=str(_PROJECT_ROOT / "RAG" / "rag_chroma_output" / "llm_csv_outputs_2025Q2_Q3"),  # ← 修改
    )
    p.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "RAG" / "rag_chroma_output" / "information_flow_network_demo_v2"),  # ← 修改
    )
    p.add_argument("--source-quarter", default="2025Q2")
    p.add_argument("--target-quarter", default="2025Q3")
    p.add_argument("--focus-signal", default="margin_outlook")
    p.add_argument("--focus-label", default="improving")
    p.add_argument(
        "--include-self-edges",
        action="store_true",
        help="Include source==target matched edges. Default excludes self edges.",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_shards(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern))
    frames = []
    print(f"Searching {pattern}: {len(files)} files")

    for f in files:
        if f.exists() and f.stat().st_size > 0:
            try:
                df = pd.read_csv(f)
                df["source_file"] = f.name
                frames.append(df)
                print(f"  loaded {f.name}: {len(df):,} rows")
            except Exception as e:
                print(f"  WARNING skipped {f}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop_duplicates()


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


def clean_label(x) -> str:
    if pd.isna(x):
        return "not_mentioned"
    s = str(x).strip().lower()
    if not s or s == "nan":
        return "not_mentioned"
    return s


def label_direction(label: str, score: float) -> str:
    labels = set(str(label).split(";"))
    if labels & POSITIVE_LABELS:
        return "positive"
    if labels & NEGATIVE_LABELS:
        return "negative"
    if labels & MIXED_LABELS:
        return "mixed"
    if labels & NEUTRAL_LABELS:
        return "neutral"

    if pd.isna(score):
        return "not_mentioned"
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def clean_outlook(df: pd.DataFrame) -> pd.DataFrame:
    required = ["doc_id", "ticker", "current_company", "quarter", "signal", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Outlook CSV missing columns: {missing}")

    out = df.copy()

    for c in required:
        out[c] = out[c].astype(str).str.strip()

    out["signal_raw"] = out["signal"]
    out["signal"] = out["signal"].str.lower().map(lambda x: SIGNAL_MAP.get(x, x))
    out = out[out["signal"].isin(STANDARD_SIGNALS)].copy()

    out["label"] = out["label"].map(clean_label)
    out["score"] = out["label"].map(LABEL_SCORE)

    out["company_norm"] = out["current_company"].map(norm_text)
    out["ticker_norm"] = out["ticker"].map(norm_text)

    out["company_node"] = np.where(
        out["ticker"].astype(str).str.strip().ne(""),
        "COMPANY::" + out["ticker"].astype(str).str.strip(),
        "COMPANY::" + out["company_norm"],
    )

    group_cols = [
        "company_node",
        "ticker",
        "current_company",
        "company_norm",
        "ticker_norm",
        "quarter",
        "signal",
    ]

    agg = {
        "score": "mean",
        "label": lambda x: ";".join(sorted(set(str(v) for v in x if str(v) != "nan"))),
    }

    if "evidence_chunk_ids" in out.columns:
        agg["evidence_chunk_ids"] = lambda x: "|".join(
            sorted(set(str(v) for v in x.dropna() if str(v).strip()))
        )

    if "notes" in out.columns:
        agg["notes"] = lambda x: " || ".join(str(v) for v in x.dropna() if str(v).strip())

    out = out.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out["direction"] = out.apply(lambda r: label_direction(r["label"], r["score"]), axis=1)
    out["is_active"] = out["score"].notna() & (out["score"].abs() > 0)

    return out


def clean_relationships(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    required = ["ticker", "current_company", "quarter", "relation_group", "entity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Relationships CSV missing columns: {missing}")

    rel = df.copy()
    for c in rel.columns:
        if rel[c].dtype == "object":
            rel[c] = rel[c].astype(str).str.strip()

    rel = rel[rel["entity"].fillna("").astype(str).str.strip().ne("")].copy()
    rel = rel[rel["relation_group"].str.lower().ne("none")].copy()

    rel["source_company_node"] = np.where(
        rel["ticker"].astype(str).str.strip().ne(""),
        "COMPANY::" + rel["ticker"].astype(str).str.strip(),
        "COMPANY::" + rel["current_company"].map(norm_text),
    )

    rel["source_company_norm"] = rel["current_company"].map(norm_text)
    rel["source_ticker_norm"] = rel["ticker"].map(norm_text)
    rel["target_entity_norm"] = rel["entity"].map(norm_text)

    rows = []
    for _, r in rel.iterrows():
        groups = [g.strip() for g in str(r["relation_group"]).split("|") if g.strip()]
        if not groups:
            groups = [str(r["relation_group"]).strip()]

        for g in groups:
            rr = r.copy()
            rr["relation_group_clean"] = g
            rows.append(rr)

    return pd.DataFrame(rows).drop_duplicates() if rows else rel


def build_company_lookup(outlook: pd.DataFrame):
    company_map = {}
    ticker_map = {}

    base = outlook[
        ["company_node", "ticker", "current_company", "company_norm", "ticker_norm"]
    ].drop_duplicates()

    for _, r in base.iterrows():
        node = str(r["company_node"])
        cname = str(r["company_norm"])
        ticker = str(r["ticker_norm"])

        if cname:
            company_map[cname] = node
        if ticker:
            ticker_map[ticker] = node

    return company_map, ticker_map


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


def get_company_meta(outlook: pd.DataFrame):
    meta = {}
    base = outlook[["company_node", "ticker", "current_company"]].drop_duplicates()
    for _, r in base.iterrows():
        meta[str(r["company_node"])] = {
            "ticker": str(r["ticker"]),
            "company": str(r["current_company"]),
        }
    return meta


def build_contagion_events(
    outlook: pd.DataFrame,
    relationships: pd.DataFrame,
    source_q: str,
    target_q: str,
    include_self_edges: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build company-to-company directional exposure events.

    Each event:
      source company has signal at source_q
      source has extracted relationship to target entity
      target entity is matched to another observed company
      target company has same signal observed at target_q
    """

    company_map, ticker_map = build_company_lookup(outlook)
    outlook_lookup = make_outlook_lookup(outlook)
    meta = get_company_meta(outlook)

    rel = relationships.copy()

    # Prefer relationships from observed quarters.
    rel_q = rel[rel["quarter"].isin([source_q, target_q])].copy()
    if not rel_q.empty:
        rel = rel_q

    rel["target_company_node"] = rel["target_entity_norm"].map(
        lambda x: match_entity_to_company(x, company_map, ticker_map)
    )

    unmatched = rel[rel["target_company_node"].fillna("").eq("")].copy()

    matched = rel[rel["target_company_node"].fillna("").ne("")].copy()

    if not include_self_edges:
        matched = matched[matched["source_company_node"] != matched["target_company_node"]].copy()

    events = []

    for _, edge in matched.iterrows():
        source_node = str(edge["source_company_node"])
        target_node = str(edge["target_company_node"])

        source_meta = meta.get(source_node, {})
        target_meta = meta.get(target_node, {})

        for signal in STANDARD_SIGNALS:
            srow = outlook_lookup.get((source_node, source_q, signal))
            trow = outlook_lookup.get((target_node, target_q, signal))

            if srow is None:
                continue
            if trow is None:
                # Exposed but no target signal observation.
                continue

            source_label = str(srow.label)
            target_label = str(trow.label)
            source_score = float(srow.score) if not pd.isna(srow.score) else np.nan
            target_score = float(trow.score) if not pd.isna(trow.score) else np.nan

            source_direction = str(srow.direction)
            target_direction = str(trow.direction)

            source_active = not pd.isna(source_score) and abs(source_score) > 0
            target_active = not pd.isna(target_score) and abs(target_score) > 0

            exact_adoption = (
                source_active
                and target_active
                and source_label == target_label
            )

            direction_adoption = (
                source_active
                and target_active
                and source_direction == target_direction
            )

            # For "infected with margin_outlook=improving", exact adoption is stricter.
            # Direction adoption allows labels like positive/increase/improving to be considered same direction.
            events.append({
                "source_node": source_node,
                "source_ticker": source_meta.get("ticker", ""),
                "source_company": source_meta.get("company", ""),
                "target_node": target_node,
                "target_ticker": target_meta.get("ticker", ""),
                "target_company": target_meta.get("company", ""),
                "source_quarter": source_q,
                "target_quarter": target_q,
                "signal": signal,
                "source_label": source_label,
                "target_label": target_label,
                "source_score": source_score,
                "target_score": target_score,
                "source_direction": source_direction,
                "target_direction": target_direction,
                "source_active": source_active,
                "target_active": target_active,
                "exact_adoption": exact_adoption,
                "direction_adoption": direction_adoption,
                "relation_group": str(edge.get("relation_group_clean", edge.get("relation_group", ""))),
                "relationship_type": str(edge.get("relationship_type", "")),
                "entity_type": str(edge.get("entity_type", "")),
                "confidence": str(edge.get("confidence", "")),
                "extracted_entity": str(edge.get("entity", "")),
            })

    return pd.DataFrame(events), unmatched


def summarize_contagion(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    # Only source-active observations are real exposures.
    exp = events[events["source_active"]].copy()

    if exp.empty:
        return pd.DataFrame()

    group_cols = ["signal", "source_label", "source_direction", "relation_group"]

    rows = []

    for keys, g in exp.groupby(group_cols, dropna=False):
        signal, source_label, source_direction, relation_group = keys

        exposed = len(g)
        target_observed = int(g["target_label"].notna().sum())
        target_active = int(g["target_active"].sum())
        exact_adopted = int(g["exact_adoption"].sum())
        direction_adopted = int(g["direction_adoption"].sum())

        rows.append({
            "signal": signal,
            "source_label": source_label,
            "source_direction": source_direction,
            "relation_group": relation_group,
            "exposed_edges": exposed,
            "target_observed_edges": target_observed,
            "target_active_edges": target_active,
            "exact_adopted_edges": exact_adopted,
            "direction_adopted_edges": direction_adopted,
            "exact_transmission_rate": exact_adopted / exposed if exposed else np.nan,
            "direction_transmission_rate": direction_adopted / exposed if exposed else np.nan,
            "target_active_rate": target_active / exposed if exposed else np.nan,
        })

    return pd.DataFrame(rows).sort_values(
        ["exact_transmission_rate", "direction_transmission_rate", "exposed_edges"],
        ascending=[False, False, False],
    )


def focus_summary(events: pd.DataFrame, focus_signal: str, focus_label: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    focus_signal = focus_signal.strip()
    focus_label = focus_label.strip().lower()

    f = events[
        (events["signal"] == focus_signal)
        & (events["source_active"])
        & (events["source_label"].astype(str).str.contains(rf"\b{re.escape(focus_label)}\b", regex=True))
    ].copy()

    if f.empty:
        return pd.DataFrame()

    rows = []

    for relation, g in f.groupby("relation_group", dropna=False):
        exposed = len(g)
        exact = int(g["target_label"].astype(str).str.contains(rf"\b{re.escape(focus_label)}\b", regex=True).sum())
        same_direction = int(g["direction_adoption"].sum())
        active = int(g["target_active"].sum())

        rows.append({
            "focus_signal": focus_signal,
            "focus_label": focus_label,
            "relation_group": relation,
            "exposed_edges": exposed,
            "target_exact_same_label_edges": exact,
            "target_same_direction_edges": same_direction,
            "target_active_edges": active,
            "exact_label_transmission_rate": exact / exposed if exposed else np.nan,
            "same_direction_transmission_rate": same_direction / exposed if exposed else np.nan,
            "target_active_rate": active / exposed if exposed else np.nan,
        })

    return pd.DataFrame(rows).sort_values(
        ["exact_label_transmission_rate", "same_direction_transmission_rate", "exposed_edges"],
        ascending=[False, False, False],
    )


def save_figures(summary: pd.DataFrame, focus: pd.DataFrame, out_dir: Path):
    fig_dir = ensure_dir(out_dir / "figures")

    if not summary.empty:
        # Top exact transmission by signal-label-relation, requiring at least 2 exposures.
        top = summary[summary["exposed_edges"] >= 2].copy()
        top = top.sort_values("direction_transmission_rate", ascending=False).head(25)

        if not top.empty:
            top["name"] = (
                top["signal"].astype(str)
                + " | "
                + top["source_label"].astype(str)
                + " | "
                + top["relation_group"].astype(str)
            )

            ax = top.sort_values("direction_transmission_rate").plot(
                kind="barh",
                x="name",
                y="direction_transmission_rate",
                legend=False,
                figsize=(12, 8),
            )
            ax.set_title("Top candidate same-direction transmission rates")
            ax.set_xlabel("Same-direction transmission rate")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(fig_dir / "top_direction_transmission_rates.png", dpi=220)
            plt.close()

    if not focus.empty:
        ax = focus.sort_values("same_direction_transmission_rate").plot(
            kind="barh",
            x="relation_group",
            y="same_direction_transmission_rate",
            legend=False,
            figsize=(10, 5),
        )
        ax.set_title("Focus signal contagion by relationship group")
        ax.set_xlabel("Same-direction transmission rate")
        ax.set_ylabel("Relationship group")
        plt.tight_layout()
        plt.savefig(fig_dir / "focus_signal_transmission_by_relation.png", dpi=220)
        plt.close()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    print("=" * 90)
    print("Signal contagion / transmissibility analysis")
    print("input_dir:", input_dir)
    print("out_dir:", out_dir)
    print("source_quarter:", args.source_quarter)
    print("target_quarter:", args.target_quarter)
    print("focus:", args.focus_signal, args.focus_label)
    print("=" * 90)

    outlook_raw = read_shards(input_dir, "outlook_q2q3_shard*_of006.csv")
    relationships_raw = read_shards(input_dir, "relationships_q2q3_shard*_of006.csv")

    if outlook_raw.empty:
        raise FileNotFoundError("No outlook CSV files found.")
    if relationships_raw.empty:
        raise FileNotFoundError("No relationships CSV files found.")

    outlook = clean_outlook(outlook_raw)
    relationships = clean_relationships(relationships_raw)

    print("\nCleaned rows:")
    print("outlook:", len(outlook))
    print("relationships:", len(relationships))

    print("\nQuarter distribution:")
    print(outlook["quarter"].value_counts(dropna=False).sort_index())

    print("\nSignal distribution:")
    print(outlook["signal"].value_counts(dropna=False))

    print("\nRelationship group distribution:")
    print(relationships["relation_group_clean"].value_counts(dropna=False).head(30))

    events, unmatched = build_contagion_events(
        outlook=outlook,
        relationships=relationships,
        source_q=args.source_quarter,
        target_q=args.target_quarter,
        include_self_edges=args.include_self_edges,
    )

    summary = summarize_contagion(events)
    focus = focus_summary(events, args.focus_signal, args.focus_label)

    events.to_csv(out_dir / "contagion_events.csv", index=False)
    summary.to_csv(out_dir / "contagion_summary_by_signal_label_relation.csv", index=False)
    focus.to_csv(out_dir / "contagion_summary_margin_improving.csv", index=False)
    unmatched.to_csv(out_dir / "unmatched_relationship_entities.csv", index=False)

    print(f"\nSAVED {out_dir / 'contagion_events.csv'} rows={len(events):,}")
    print(f"SAVED {out_dir / 'contagion_summary_by_signal_label_relation.csv'} rows={len(summary):,}")
    print(f"SAVED {out_dir / 'contagion_summary_margin_improving.csv'} rows={len(focus):,}")
    print(f"SAVED {out_dir / 'unmatched_relationship_entities.csv'} rows={len(unmatched):,}")

    print("\nMatched company-to-company contagion events:", len(events))
    print("Unmatched generic relationship entities:", len(unmatched))

    if not focus.empty:
        print("\nFocus result:")
        print(focus.to_string(index=False))
    else:
        print(
            "\nNo focus contagion result found. This usually means either:\n"
            "1. no source company had the focus signal/label in source quarter, or\n"
            "2. relationship targets could not be matched to observed target-quarter companies, or\n"
            "3. target companies did not have the same signal observed in target quarter."
        )

    save_figures(summary, focus, out_dir)

    readme = [
        "# Signal Contagion Analysis",
        "",
        f"- Source quarter: `{args.source_quarter}`",
        f"- Target quarter: `{args.target_quarter}`",
        f"- Focus signal: `{args.focus_signal}`",
        f"- Focus label: `{args.focus_label}`",
        f"- Raw outlook rows: {len(outlook_raw):,}",
        f"- Raw relationship rows: {len(relationships_raw):,}",
        f"- Cleaned outlook rows: {len(outlook):,}",
        f"- Cleaned relationship rows: {len(relationships):,}",
        f"- Matched company-to-company contagion event rows: {len(events):,}",
        f"- Unmatched relationship entity rows: {len(unmatched):,}",
        "",
        "## Definitions",
        "",
        "`exposed_edges`: source company had a source-quarter active signal and has an extracted relationship to a matched target company.",
        "",
        "`exact_transmission_rate`: among exposed edges, share where the target company has the same signal and exact same label in the target quarter.",
        "",
        "`direction_transmission_rate`: among exposed edges, share where target company has the same signal and same sign/direction in the target quarter.",
        "",
        "`target_active_rate`: among exposed edges, share where target company has any active non-neutral state for the same signal in the target quarter.",
        "",
        "## Interpretation warning",
        "",
        "This is a candidate contagion analysis, not causal proof. "
        "It measures whether an outlook state appears in related companies later, conditional on matched company-to-company links.",
    ]

    (out_dir / "contagion_summary.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"SAVED {out_dir / 'contagion_summary.md'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
