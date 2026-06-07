#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date-aware two-part network analysis for earnings-call LLM outputs.

This is an updated version of run_two_part_network_prediction_analysis.py.

Main changes from the old version
---------------------------------
1. Loads publish_date from evidence metadata.
2. Builds a company-quarter publish-date lookup:
       company_node + quarter -> publish_date
3. Joins dates back to both cross-quarter and same-quarter event rows:
       source_node + source_quarter -> source_publish_date
       target_node + target_quarter -> target_publish_date
4. Computes:
       publish_gap_days = target_publish_date - source_publish_date
       abs_publish_gap_days
       source_before_target
5. Creates an ordered same-quarter subset:
       same_quarter_events_ordered.csv
       same_quarter_ordered_summary_by_quarter_signal_relation.csv
       same_quarter_ordered_correlation_by_signal_relation.csv

Interpretation
--------------
Cross-quarter:
    Quarter-level lead-lag. Publish gap is a temporal diagnostic, not diffusion speed.

Same-quarter:
    Without date order: same-quarter network correlation.
    With source_before_target=True: within-quarter prediction candidate.

Recommended command
-------------------
python run_two_part_network_prediction_analysis_date_aware.py ^
  --rag-output-dir rag_chroma_output ^
  --out-dir rag_chroma_output/two_part_network_prediction_analysis_date_aware ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2 ^
  --date-source hf ^
  --evidence-dataset soysouce/earningALZ_SBERT_evidence ^
  --evidence-metadata-file rag_evidence_package_metadata_full_gpu_direct.parquet

If Hugging Face is unavailable but you have local metadata:
---------------------------------------------------------
python run_two_part_network_prediction_analysis_date_aware.py ^
  --rag-output-dir rag_chroma_output ^
  --out-dir rag_chroma_output/two_part_network_prediction_analysis_date_aware ^
  --metadata-parquet rag_evidence_package_metadata_full_gpu_direct.parquet
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


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag-output-dir", default="rag_chroma_output")
    p.add_argument("--out-dir", default="rag_chroma_output/two_part_network_prediction_analysis_date_aware")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--include-self-edges", action="store_true")
    p.add_argument("--min-exposed-for-plot", type=int, default=5)
    p.add_argument(
        "--use-quarter-specific-relationships",
        action="store_true",
        help=(
            "If set, only relationships extracted in the source/target quarter are used. "
            "Default uses all matched relationships as structural links."
        ),
    )

    # Date source controls.
    p.add_argument(
        "--date-source",
        default="auto",
        choices=["auto", "hf", "local", "none"],
        help=(
            "Where to load publish_date from. "
            "auto: use local metadata if provided, otherwise HF, otherwise extraction CSV fallback."
        ),
    )
    p.add_argument("--evidence-dataset", default="soysouce/earningALZ_SBERT_evidence")
    p.add_argument("--evidence-metadata-file", default="rag_evidence_package_metadata_full_gpu_direct.parquet")
    p.add_argument("--hf-revision", default="main")
    p.add_argument("--metadata-parquet", default="", help="Local evidence metadata parquet with ticker, quarter, publish_date.")
    p.add_argument("--metadata-csv", default="", help="Local evidence metadata csv with ticker, quarter, publish_date.")
    p.add_argument("--write-parquet", action="store_true", help="Also write parquet copies of key outputs.")

    return p.parse_args()


# ============================================================
# General helpers
# ============================================================

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


def clean_node(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "0"}:
        return ""
    return s


def company_node_from(ticker: str, company: str = "") -> str:
    ticker = clean_node(ticker)
    company = clean_node(company)
    if ticker:
        return "COMPANY::" + ticker.upper()
    cleaned = norm_text(company)
    if cleaned:
        return "COMPANY::" + cleaned
    return ""


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


def save_table(df: pd.DataFrame, path: Path, write_parquet: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"SAVED {path} rows={len(df):,}")
    if write_parquet:
        pq = path.with_suffix(".parquet")
        df.to_parquet(pq, index=False)
        print(f"SAVED {pq} rows={len(df):,}")


# ============================================================
# Input CSV discovery and reading
# ============================================================

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


# ============================================================
# Date loading and attachment
# ============================================================

def read_hf_metadata(dataset: str, filename: str, revision: str) -> pd.DataFrame:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for --date-source hf. Install with: pip install huggingface_hub") from exc

    local = hf_hub_download(
        repo_id=dataset,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    df = pd.read_parquet(local)
    print(f"Loaded HF metadata {dataset}/{filename}: rows={len(df):,}, cols={len(df.columns):,}")
    return df


def build_publish_dates_from_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Build company-quarter publish-date table from evidence metadata.

    Required columns:
        ticker
        quarter
        publish_date

    Optional:
        current_company
        doc_id
    """
    if metadata is None or metadata.empty:
        return pd.DataFrame(columns=["company_node", "quarter", "publish_date", "publish_date_count"])

    required = ["ticker", "quarter", "publish_date"]
    missing = [c for c in required if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing required date columns: {missing}")

    d = metadata.copy()
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()
    d["current_company"] = d["current_company"].astype(str).str.strip() if "current_company" in d.columns else ""
    d["company_node"] = d.apply(lambda r: company_node_from(r.get("ticker", ""), r.get("current_company", "")), axis=1)
    d["quarter"] = d["quarter"].astype(str).str.strip()
    d["publish_date"] = pd.to_datetime(d["publish_date"], errors="coerce")

    d = d[d["company_node"].ne("") & d["quarter"].ne("") & d["publish_date"].notna()].copy()

    agg_cols = {
        "publish_date": "min",
    }
    out = (
        d.groupby(["company_node", "quarter"], as_index=False)
        .agg(
            publish_date=("publish_date", "min"),
            publish_date_count=("publish_date", "count"),
            metadata_doc_ids=("doc_id", lambda x: "|".join(map(str, list(x)[:5])) if "doc_id" in d.columns else ""),
        )
    )
    print(f"Built publish-date lookup: rows={len(out):,}, companies={out['company_node'].nunique():,}")
    return out


def build_publish_dates_from_extraction_csvs(raw_outlook: pd.DataFrame, raw_rel: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: if LLM extraction CSVs already preserved publish_date, build dates from them.
    This is less preferred than evidence metadata because older CSVs may not include publish_date.
    """
    frames = []
    for df, name in [(raw_outlook, "outlook"), (raw_rel, "relationships")]:
        if df is None or df.empty:
            continue
        if not {"ticker", "quarter", "publish_date"}.issubset(df.columns):
            continue
        tmp = df[["ticker", "current_company", "quarter", "publish_date"]].copy()
        tmp["source_kind"] = name
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["company_node", "quarter", "publish_date", "publish_date_count"])

    meta = pd.concat(frames, ignore_index=True).drop_duplicates()
    return build_publish_dates_from_metadata(meta)


def load_publish_dates(args, raw_outlook: pd.DataFrame, raw_rel: pd.DataFrame) -> pd.DataFrame:
    if args.date_source == "none":
        print("Date source disabled.")
        return pd.DataFrame(columns=["company_node", "quarter", "publish_date", "publish_date_count"])

    # local explicit source
    if args.metadata_parquet:
        meta = pd.read_parquet(args.metadata_parquet)
        print(f"Loaded local metadata parquet: {args.metadata_parquet}, rows={len(meta):,}")
        return build_publish_dates_from_metadata(meta)

    if args.metadata_csv:
        meta = pd.read_csv(args.metadata_csv)
        print(f"Loaded local metadata csv: {args.metadata_csv}, rows={len(meta):,}")
        return build_publish_dates_from_metadata(meta)

    # hf source
    if args.date_source in {"hf", "auto"}:
        try:
            meta = read_hf_metadata(args.evidence_dataset, args.evidence_metadata_file, args.hf_revision)
            return build_publish_dates_from_metadata(meta)
        except Exception as exc:
            if args.date_source == "hf":
                raise
            print(f"WARNING: failed to load HF metadata, trying extraction CSV fallback. Error: {exc}")

    # extraction CSV fallback
    publish_dates = build_publish_dates_from_extraction_csvs(raw_outlook, raw_rel)
    if publish_dates.empty:
        print("WARNING: No publish dates available from metadata or extraction CSVs.")
    return publish_dates


def attach_publish_dates(events: pd.DataFrame, publish_dates: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    out = events.copy()

    if publish_dates.empty:
        out["source_publish_date"] = pd.NaT
        out["target_publish_date"] = pd.NaT
        out["source_publish_date_count"] = np.nan
        out["target_publish_date_count"] = np.nan
        out["publish_gap_days"] = np.nan
        out["abs_publish_gap_days"] = np.nan
        out["source_before_target"] = np.nan
        return out

    d = publish_dates.copy()
    d["publish_date"] = pd.to_datetime(d["publish_date"], errors="coerce")

    src = d.rename(columns={
        "company_node": "source_node",
        "quarter": "source_quarter",
        "publish_date": "source_publish_date",
        "publish_date_count": "source_publish_date_count",
        "metadata_doc_ids": "source_metadata_doc_ids",
    })
    tgt = d.rename(columns={
        "company_node": "target_node",
        "quarter": "target_quarter",
        "publish_date": "target_publish_date",
        "publish_date_count": "target_publish_date_count",
        "metadata_doc_ids": "target_metadata_doc_ids",
    })

    src_cols = ["source_node", "source_quarter", "source_publish_date", "source_publish_date_count"]
    tgt_cols = ["target_node", "target_quarter", "target_publish_date", "target_publish_date_count"]
    if "source_metadata_doc_ids" in src.columns:
        src_cols.append("source_metadata_doc_ids")
    if "target_metadata_doc_ids" in tgt.columns:
        tgt_cols.append("target_metadata_doc_ids")

    out = out.merge(src[src_cols], on=["source_node", "source_quarter"], how="left")
    out = out.merge(tgt[tgt_cols], on=["target_node", "target_quarter"], how="left")

    out["source_publish_date"] = pd.to_datetime(out["source_publish_date"], errors="coerce")
    out["target_publish_date"] = pd.to_datetime(out["target_publish_date"], errors="coerce")
    out["publish_gap_days"] = (out["target_publish_date"] - out["source_publish_date"]).dt.days
    out["abs_publish_gap_days"] = out["publish_gap_days"].abs()
    out["source_before_target"] = out["publish_gap_days"] > 0

    source_cov = out["source_publish_date"].notna().mean() if len(out) else 0
    target_cov = out["target_publish_date"].notna().mean() if len(out) else 0
    both_cov = (out["source_publish_date"].notna() & out["target_publish_date"].notna()).mean() if len(out) else 0
    print(f"Date join coverage: source={source_cov:.2%}, target={target_cov:.2%}, both={both_cov:.2%}")

    return out


def make_date_coverage_report(events: pd.DataFrame, name: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame([{
            "dataset": name,
            "rows": 0,
            "source_date_coverage": np.nan,
            "target_date_coverage": np.nan,
            "both_date_coverage": np.nan,
            "source_before_target_rate": np.nan,
            "mean_publish_gap_days": np.nan,
            "median_publish_gap_days": np.nan,
        }])

    both = events["source_publish_date"].notna() & events["target_publish_date"].notna()
    return pd.DataFrame([{
        "dataset": name,
        "rows": len(events),
        "source_date_coverage": float(events["source_publish_date"].notna().mean()),
        "target_date_coverage": float(events["target_publish_date"].notna().mean()),
        "both_date_coverage": float(both.mean()),
        "source_before_target_rate": float(events.loc[both, "source_before_target"].mean()) if both.any() else np.nan,
        "mean_publish_gap_days": float(events.loc[both, "publish_gap_days"].mean()) if both.any() else np.nan,
        "median_publish_gap_days": float(events.loc[both, "publish_gap_days"].median()) if both.any() else np.nan,
    }])


# ============================================================
# Cleaning outlook and relationships
# ============================================================

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
        "COMPANY::" + out["ticker"].astype(str).str.strip().str.upper(),
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

    # Keep publish_date if available in extraction CSVs, mainly for audit/fallback.
    if "publish_date" in out.columns:
        out["publish_date"] = pd.to_datetime(out["publish_date"], errors="coerce")
        agg["publish_date"] = "min"

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
        "COMPANY::" + rel["ticker"].astype(str).str.strip().str.upper(),
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

    dedup_cols = [
        "source_company_node",
        "target_company_node",
        "relation_group_clean",
        "relationship_type",
        "quarter",
    ]
    dedup_cols = [c for c in dedup_cols if c in matched.columns]
    matched = matched.drop_duplicates(subset=dedup_cols)

    return matched, unmatched, meta


def select_relationships_for_window(
    matched_rel: pd.DataFrame,
    source_q: str,
    target_q: str,
    use_quarter_specific: bool,
) -> pd.DataFrame:
    if not use_quarter_specific:
        return matched_rel
    return matched_rel[matched_rel["quarter"].isin([source_q, target_q])].copy()


# ============================================================
# Event construction and summaries
# ============================================================

def build_events_for_pair(
    outlook_lookup: dict,
    meta: dict,
    matched_rel: pd.DataFrame,
    source_q: str,
    target_q: str,
    mode: str,
) -> pd.DataFrame:
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
                "relationship_source_quarter": str(edge.get("quarter", "")),
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

        valid_pred = g[g["prediction_correct"].notna()].copy()
        correct = int(valid_pred["prediction_correct"].sum()) if not valid_pred.empty else 0
        accuracy = correct / len(valid_pred) if len(valid_pred) else np.nan

        both_dates = g["source_publish_date"].notna() & g["target_publish_date"].notna() if "source_publish_date" in g.columns else pd.Series(False, index=g.index)
        date_obs = int(both_dates.sum())

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
            "date_observation_count": date_obs,
            "source_before_target_edges": int(g.loc[both_dates, "source_before_target"].sum()) if date_obs else 0,
            "source_before_target_rate": float(g.loc[both_dates, "source_before_target"].mean()) if date_obs else np.nan,
            "mean_publish_gap_days": float(g.loc[both_dates, "publish_gap_days"].mean()) if date_obs else np.nan,
            "median_publish_gap_days": float(g.loc[both_dates, "publish_gap_days"].median()) if date_obs else np.nan,
            "mean_abs_publish_gap_days": float(g.loc[both_dates, "abs_publish_gap_days"].mean()) if date_obs else np.nan,
            "median_abs_publish_gap_days": float(g.loc[both_dates, "abs_publish_gap_days"].median()) if date_obs else np.nan,
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
        date_obs = int(g["date_observation_count"].sum()) if "date_observation_count" in g.columns else 0
        source_before = int(g["source_before_target_edges"].sum()) if "source_before_target_edges" in g.columns else 0

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
            "date_observation_count": date_obs,
            "source_before_target_edges": source_before,
            "source_before_target_rate": source_before / date_obs if date_obs else np.nan,
            "mean_publish_gap_days": np.average(g["mean_publish_gap_days"].fillna(0), weights=g["date_observation_count"].fillna(0)) if date_obs else np.nan,
            "mean_abs_publish_gap_days": np.average(g["mean_abs_publish_gap_days"].fillna(0), weights=g["date_observation_count"].fillna(0)) if date_obs else np.nan,
            "num_windows": int(g[["source_quarter", "target_quarter"]].drop_duplicates().shape[0]),
        })

    return pd.DataFrame(rows).sort_values(
        ["analysis_mode", "prediction_accuracy", "exposed_edges"],
        ascending=[True, False, False],
    )


# ============================================================
# Figures
# ============================================================

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


def plot_source_before_target_by_signal(agg: pd.DataFrame, mode: str, out_png: Path, min_exposed: int):
    d = agg[
        (agg["analysis_mode"] == mode)
        & (agg["date_observation_count"] >= min_exposed)
        & (agg["source_before_target_rate"].notna())
    ].copy()
    if d.empty:
        return
    g = (
        d.groupby("signal", as_index=False)
        .agg(
            date_observation_count=("date_observation_count", "sum"),
            source_before_target_edges=("source_before_target_edges", "sum"),
        )
    )
    g["source_before_target_rate"] = g["source_before_target_edges"] / g["date_observation_count"]
    g = g.sort_values("source_before_target_rate")
    ax = g.plot(kind="barh", x="signal", y="source_before_target_rate", legend=False, figsize=(10, 5))
    ax.set_title(f"{mode}: source-before-target rate by signal")
    ax.set_xlabel("source_before_target_rate")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    rag_dir = Path(args.rag_output_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")

    print("=" * 90)
    print("Date-aware two-part network prediction analysis")
    print("rag_output_dir:", rag_dir)
    print("out_dir:", out_dir)
    print("quarter range:", args.start_quarter or "ALL", "to", args.end_quarter or "ALL")
    print("use_quarter_specific_relationships:", args.use_quarter_specific_relationships)
    print("date_source:", args.date_source)
    print("=" * 90)

    outlook_files, rel_files = discover_extraction_csvs(rag_dir)

    manifest = pd.DataFrame({
        "kind": ["outlook"] * len(outlook_files) + ["relationships"] * len(rel_files),
        "path": [str(x) for x in outlook_files + rel_files],
    })
    save_table(manifest, out_dir / "input_file_manifest.csv", args.write_parquet)

    raw_outlook = read_many_csv(outlook_files, "outlook")
    raw_rel = read_many_csv(rel_files, "relationships")

    if raw_outlook.empty:
        raise RuntimeError("No outlook extraction files found.")
    if raw_rel.empty:
        raise RuntimeError("No relationship extraction files found.")

    publish_dates = load_publish_dates(args, raw_outlook, raw_rel)
    save_table(publish_dates, out_dir / "company_quarter_publish_dates.csv", args.write_parquet)

    outlook = clean_outlook(raw_outlook, args.start_quarter, args.end_quarter)
    relationships = clean_relationships(raw_rel, args.start_quarter, args.end_quarter)

    save_table(outlook, out_dir / "cleaned_outlook_all.csv", args.write_parquet)
    save_table(relationships, out_dir / "cleaned_relationships_all.csv", args.write_parquet)

    matched_rel, unmatched_rel, meta = prepare_matched_relationships(
        relationships,
        outlook,
        include_self_edges=args.include_self_edges,
    )

    save_table(matched_rel, out_dir / "matched_company_relationships.csv", args.write_parquet)
    save_table(unmatched_rel, out_dir / "unmatched_relationship_entities.csv", args.write_parquet)

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
    cross_events = attach_publish_dates(cross_events, publish_dates)
    cross_summary = summarize_events(cross_events, "cross_quarter")
    cross_agg = aggregate_summary(cross_summary)

    save_table(cross_events, out_dir / "cross_quarter_events.csv", args.write_parquet)
    save_table(cross_summary, out_dir / "cross_quarter_summary_by_window_signal_relation.csv", args.write_parquet)
    save_table(cross_agg, out_dir / "cross_quarter_prediction_accuracy.csv", args.write_parquet)

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
    same_events = attach_publish_dates(same_events, publish_dates)

    # Date-ordered subset: only rows where both dates exist and source is earlier.
    same_events_ordered = same_events[
        same_events["source_publish_date"].notna()
        & same_events["target_publish_date"].notna()
        & same_events["source_before_target"].fillna(False)
    ].copy()

    same_summary = summarize_events(same_events, "same_quarter")
    same_agg = aggregate_summary(same_summary)

    same_ordered_summary = summarize_events(same_events_ordered, "same_quarter_ordered")
    same_ordered_agg = aggregate_summary(same_ordered_summary)

    save_table(same_events, out_dir / "same_quarter_events.csv", args.write_parquet)
    save_table(same_summary, out_dir / "same_quarter_summary_by_quarter_signal_relation.csv", args.write_parquet)
    save_table(same_agg, out_dir / "same_quarter_correlation_by_signal_relation.csv", args.write_parquet)

    save_table(same_events_ordered, out_dir / "same_quarter_events_ordered.csv", args.write_parquet)
    save_table(same_ordered_summary, out_dir / "same_quarter_ordered_summary_by_quarter_signal_relation.csv", args.write_parquet)
    save_table(same_ordered_agg, out_dir / "same_quarter_ordered_prediction_by_signal_relation.csv", args.write_parquet)

    # Combined
    combined_events = pd.concat([cross_events, same_events], ignore_index=True).drop_duplicates()
    combined_summary = pd.concat([cross_summary, same_summary], ignore_index=True).drop_duplicates()
    combined_agg = pd.concat([cross_agg, same_agg], ignore_index=True).drop_duplicates()

    save_table(combined_events, out_dir / "combined_events_cross_and_same_quarter.csv", args.write_parquet)
    save_table(combined_summary, out_dir / "combined_summary_cross_and_same_quarter.csv", args.write_parquet)
    save_table(combined_agg, out_dir / "combined_accuracy_correlation_summary.csv", args.write_parquet)

    # Date coverage report.
    coverage = pd.concat([
        make_date_coverage_report(cross_events, "cross_quarter_events"),
        make_date_coverage_report(same_events, "same_quarter_events"),
        make_date_coverage_report(same_events_ordered, "same_quarter_events_ordered"),
    ], ignore_index=True)
    save_table(coverage, out_dir / "date_coverage_report.csv", args.write_parquet)

    # Figures
    plot_window_counts(cross_events, "cross_quarter", fig_dir / "cross_quarter_event_rows_by_window.png")
    plot_window_counts(same_events, "same_quarter", fig_dir / "same_quarter_event_rows_by_quarter.png")
    plot_window_counts(same_events_ordered, "same_quarter", fig_dir / "same_quarter_ordered_event_rows_by_quarter.png")

    plot_rate_by_group(cross_agg, "cross_quarter", "signal", "prediction_accuracy", fig_dir / "cross_quarter_accuracy_by_signal.png", args.min_exposed_for_plot)
    plot_rate_by_group(cross_agg, "cross_quarter", "relation_group", "prediction_accuracy", fig_dir / "cross_quarter_accuracy_by_relation.png", args.min_exposed_for_plot)

    plot_rate_by_group(same_agg, "same_quarter", "signal", "direction_match_rate", fig_dir / "same_quarter_similarity_by_signal.png", args.min_exposed_for_plot)
    plot_rate_by_group(same_agg, "same_quarter", "relation_group", "direction_match_rate", fig_dir / "same_quarter_similarity_by_relation.png", args.min_exposed_for_plot)

    plot_rate_by_group(same_ordered_agg, "same_quarter", "signal", "direction_match_rate", fig_dir / "same_quarter_ordered_similarity_by_signal.png", args.min_exposed_for_plot)
    plot_rate_by_group(same_ordered_agg, "same_quarter", "relation_group", "direction_match_rate", fig_dir / "same_quarter_ordered_similarity_by_relation.png", args.min_exposed_for_plot)

    plot_source_before_target_by_signal(same_agg, "same_quarter", fig_dir / "same_quarter_source_before_target_by_signal.png", args.min_exposed_for_plot)

    # Markdown report
    lines = []
    lines.append("# Date-Aware Two-Part Network Prediction Analysis")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This analysis separates network signal analysis into three parts:")
    lines.append("")
    lines.append("1. **Cross-quarter lead-lag analysis**: source firm signal in quarter t is used to evaluate whether a connected target firm shows the same-direction signal in quarter t+1.")
    lines.append("2. **Same-quarter network correlation analysis**: connected firms are compared within the same reporting quarter.")
    lines.append("3. **Same-quarter ordered prediction candidates**: same-quarter rows are restricted to cases where `source_publish_date < target_publish_date`.")
    lines.append("")
    lines.append("Cross-quarter publish-date gaps are only a temporal diagnostic. Same-quarter publish-date order is substantively meaningful for within-quarter prediction.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Cleaned outlook rows: {len(outlook):,}")
    lines.append(f"- Cleaned relationship rows: {len(relationships):,}")
    lines.append(f"- Matched company relationships: {len(matched_rel):,}")
    lines.append(f"- Unmatched relationship entities: {len(unmatched_rel):,}")
    lines.append(f"- Publish-date lookup rows: {len(publish_dates):,}")
    lines.append(f"- Available quarters: {', '.join(quarters)}")
    lines.append(f"- Adjacent quarter windows: {len(pairs)}")
    lines.append("")
    lines.append("## Date coverage")
    lines.append("")
    lines.append(coverage.to_markdown(index=False))
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
    lines.append("## Part C: Same-quarter ordered prediction candidates")
    lines.append("")
    lines.append(f"- Same-quarter ordered event rows: {len(same_events_ordered):,}")
    if not same_ordered_agg.empty:
        top = same_ordered_agg[same_ordered_agg["exposed_edges"] >= args.min_exposed_for_plot].head(20)
        lines.append("")
        lines.append("Top same-quarter ordered results:")
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
    lines.append("- `figures/same_quarter_ordered_event_rows_by_quarter.png`")
    lines.append("- `figures/same_quarter_ordered_similarity_by_signal.png`")
    lines.append("- `figures/same_quarter_ordered_similarity_by_relation.png`")
    lines.append("- `figures/same_quarter_source_before_target_by_signal.png`")

    report_path = out_dir / "two_part_analysis_date_aware_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"SAVED {report_path}")

    print("\nDONE.")
    print("Main report:")
    print(report_path)


if __name__ == "__main__":
    main()
