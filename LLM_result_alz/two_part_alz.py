#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-part network analysis from Parquet LLM outputs, with full provenance.

Main changes from the CSV version:
1) Reads Parquet input instead of recursively reading CSV.
2) Preserves raw LLM signal names, e.g. loan_growth_outlook -> demand_outlook.
3) Preserves doc_id / row_id / source_file / parquet_source as evidence-chain indexes.
4) Writes Parquet outputs; optional CSV copies can be written with --write-csv.

Expected input:
- Either one combined parquet with record_type column: outlook / relationships / failed
- Or a directory containing llm_outlook_all.parquet and llm_relationships_all.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

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
    p.add_argument("--llm-parquet", default="", help="Combined parquet with record_type column.")
    p.add_argument("--llm-parquet-dir", default="", help="Directory with llm_outlook_all.parquet and llm_relationships_all.parquet.")
    p.add_argument("--hf-dataset", default="", help="Hugging Face dataset repo id, e.g. soysouce/earningALZ. If provided, Parquet files are downloaded from the Hub cache and read directly.")
    p.add_argument("--hf-revision", default="main", help="Hugging Face dataset revision/branch/commit.")
    p.add_argument("--hf-outlook-file", default="llm_outlook_all.parquet")
    p.add_argument("--hf-relationships-file", default="llm_relationships_all.parquet")
    p.add_argument("--hf-combined-file", default="", help="Optional combined parquet on HF, e.g. llm_csv_outputs_balanced_time_range_all.parquet. If set, this overrides typed HF files.")
    p.add_argument("--out-dir", default="rag_chroma_output/two_part_network_prediction_analysis_parquet")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--include-self-edges", action="store_true")
    p.add_argument("--use-quarter-specific-relationships", action="store_true")
    p.add_argument("--min-exposed-for-plot", type=int, default=5)
    p.add_argument("--write-csv", action="store_true")
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def quarter_to_index(q: str) -> float:
    m = re.match(r"^(\d{4})Q([1-4])$", str(q).strip())
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def index_to_quarter(idx: int) -> str:
    year = (idx - 1) // 4
    q = idx - year * 4
    return f"{year}Q{q}"


def adjacent_pairs(quarters: list[str]) -> list[tuple[str, str]]:
    q_sorted = sorted([q for q in quarters if not pd.isna(quarter_to_index(q))], key=quarter_to_index)
    q_set = set(q_sorted)
    out = []
    for q in q_sorted:
        nxt = index_to_quarter(int(quarter_to_index(q)) + 1)
        if nxt in q_set:
            out.append((q, nxt))
    return out


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
    return re.sub(r"\s+", " ", s).strip()


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


def clean_join(values: Any, sep: str = "|") -> str:
    vals = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            continue
        vals.append(s)
    return sep.join(sorted(set(vals)))


def row_id_series(df: pd.DataFrame, prefix: str) -> pd.Series:
    source = df["source_file"].astype(str) if "source_file" in df.columns else pd.Series([""] * len(df), index=df.index)
    if "source_row_id" in df.columns:
        rid = df["source_row_id"].astype(str)
    elif "row_id" in df.columns:
        rid = df["row_id"].astype(str)
    elif "original_row_id" in df.columns:
        rid = df["original_row_id"].astype(str)
    else:
        rid = pd.Series(df.index.astype(str), index=df.index)
    return prefix + "::" + source + "::row=" + rid


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


def save_table(df: pd.DataFrame, path: Path, write_csv: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"SAVED {path} rows={len(df):,} cols={len(df.columns):,}")
    if write_csv:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"SAVED {csv_path} rows={len(df):,} cols={len(df.columns):,}")


def read_hf_parquet(repo_id: str, filename: str, revision: str) -> tuple[pd.DataFrame, str]:
    """Download one Parquet file from a Hugging Face dataset repo cache and read it with pandas."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Missing dependency huggingface_hub. Install with: pip install huggingface_hub"
        ) from e

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    df = pd.read_parquet(local_path)
    # Keep both a reproducible HF pointer and the local cache path.
    df["parquet_source"] = f"hf://datasets/{repo_id}/{filename}@{revision}"
    df["parquet_cache_path"] = str(local_path)
    return df, local_path


def load_inputs(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = []

    if args.hf_dataset:
        repo_id = args.hf_dataset
        revision = args.hf_revision

        if args.hf_combined_file:
            df, local_path = read_hf_parquet(repo_id, args.hf_combined_file, revision)
            if "record_type" not in df.columns:
                raise ValueError("HF combined parquet must have record_type column.")
            df["record_type"] = df["record_type"].astype(str).str.lower().str.strip()
            outlook = df[df["record_type"].eq("outlook")].copy()
            rel = df[df["record_type"].isin(["relationships", "relationship"])].copy()
            manifest.append({
                "kind": "hf_combined_parquet",
                "repo_id": repo_id,
                "revision": revision,
                "filename": args.hf_combined_file,
                "path": f"hf://datasets/{repo_id}/{args.hf_combined_file}@{revision}",
                "local_cache_path": str(local_path),
            })
        else:
            outlook, outlook_cache = read_hf_parquet(repo_id, args.hf_outlook_file, revision)
            rel, rel_cache = read_hf_parquet(repo_id, args.hf_relationships_file, revision)
            manifest += [
                {
                    "kind": "hf_outlook_parquet",
                    "repo_id": repo_id,
                    "revision": revision,
                    "filename": args.hf_outlook_file,
                    "path": f"hf://datasets/{repo_id}/{args.hf_outlook_file}@{revision}",
                    "local_cache_path": str(outlook_cache),
                },
                {
                    "kind": "hf_relationships_parquet",
                    "repo_id": repo_id,
                    "revision": revision,
                    "filename": args.hf_relationships_file,
                    "path": f"hf://datasets/{repo_id}/{args.hf_relationships_file}@{revision}",
                    "local_cache_path": str(rel_cache),
                },
            ]

    elif args.llm_parquet:
        p = Path(args.llm_parquet)
        df = pd.read_parquet(p)
        if "record_type" not in df.columns:
            raise ValueError("Combined parquet must have record_type column.")
        df["record_type"] = df["record_type"].astype(str).str.lower().str.strip()
        outlook = df[df["record_type"].eq("outlook")].copy()
        rel = df[df["record_type"].isin(["relationships", "relationship"])].copy()
        outlook["parquet_source"] = str(p)
        rel["parquet_source"] = str(p)
        manifest.append({"kind": "combined_parquet", "path": str(p)})

    elif args.llm_parquet_dir:
        d = Path(args.llm_parquet_dir)
        outlook_path = d / "llm_outlook_all.parquet"
        rel_path = d / "llm_relationships_all.parquet"
        if not outlook_path.exists():
            raise FileNotFoundError(outlook_path)
        if not rel_path.exists():
            raise FileNotFoundError(rel_path)
        outlook = pd.read_parquet(outlook_path)
        rel = pd.read_parquet(rel_path)
        outlook["parquet_source"] = str(outlook_path)
        rel["parquet_source"] = str(rel_path)
        manifest += [{"kind": "outlook_parquet", "path": str(outlook_path)}, {"kind": "relationships_parquet", "path": str(rel_path)}]

    else:
        raise ValueError("Provide --hf-dataset, --llm-parquet, or --llm-parquet-dir.")

    if outlook.empty:
        raise RuntimeError("No outlook rows found.")
    if rel.empty:
        raise RuntimeError("No relationship rows found.")

    outlook = outlook.reset_index(drop=True)
    rel = rel.reset_index(drop=True)
    outlook["llm_row_id"] = row_id_series(outlook, "outlook")
    rel["llm_row_id"] = row_id_series(rel, "relationship")
    return outlook, rel, pd.DataFrame(manifest)


def ensure_cols(df: pd.DataFrame, cols: list[str], fill: str = "") -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


def clean_outlook(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    required = ["doc_id", "ticker", "current_company", "quarter", "signal", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Outlook missing required columns: {missing}")

    out = df.copy()
    for c in required:
        out[c] = out[c].astype(str).str.strip()

    out["signal_raw"] = out["signal"].astype(str).str.strip().str.lower()
    out["signal"] = out["signal_raw"].map(lambda x: SIGNAL_MAP.get(x, x))
    out["signal_mapping"] = out["signal_raw"] + " -> " + out["signal"]
    out = out[out["signal"].isin(STANDARD_SIGNALS)].copy()

    out["label_raw"] = out["label"].astype(str).str.strip()
    out["label"] = out["label_raw"].str.lower().replace({"nan": "not_mentioned"})
    out["score"] = out["label"].map(LABEL_SCORE)

    out["company_norm"] = out["current_company"].map(norm_text)
    out["ticker_norm"] = out["ticker"].map(norm_text)
    out["company_node"] = np.where(out["ticker"].astype(str).str.strip().ne(""), "COMPANY::" + out["ticker"].astype(str).str.strip(), "COMPANY::" + out["company_norm"])
    out["quarter_index"] = out["quarter"].map(quarter_to_index)
    out = filter_quarter_range(out, "quarter", start_q, end_q)

    optional = ["source_file", "source_folder", "parquet_source", "chunk_id", "chunk_uid", "evidence_id", "evidence_doc_id", "transcript_id"]
    out = ensure_cols(out, optional)

    group_cols = ["company_node", "ticker", "current_company", "company_norm", "ticker_norm", "quarter", "quarter_index", "signal"]
    agg = {
        "score": "mean",
        "label": lambda x: ";".join(sorted(set(str(v) for v in x if str(v) and str(v) != "nan"))),
        "label_raw": lambda x: clean_join(x),
        "signal_raw": lambda x: clean_join(x),
        "signal_mapping": lambda x: clean_join(x),
        "doc_id": lambda x: clean_join(x),
        "llm_row_id": lambda x: clean_join(x),
        "source_file": lambda x: clean_join(x),
        "source_folder": lambda x: clean_join(x),
        "parquet_source": lambda x: clean_join(x),
        "chunk_id": lambda x: clean_join(x),
        "chunk_uid": lambda x: clean_join(x),
        "evidence_id": lambda x: clean_join(x),
        "evidence_doc_id": lambda x: clean_join(x),
        "transcript_id": lambda x: clean_join(x),
    }
    out = out.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out = out.rename(columns={
        "doc_id": "outlook_doc_ids",
        "llm_row_id": "outlook_row_ids",
        "source_file": "outlook_source_files",
        "source_folder": "outlook_source_folders",
        "parquet_source": "outlook_parquet_sources",
        "chunk_id": "outlook_chunk_ids",
        "chunk_uid": "outlook_chunk_uids",
        "evidence_id": "outlook_evidence_ids",
        "evidence_doc_id": "outlook_evidence_doc_ids",
        "transcript_id": "outlook_transcript_ids",
        "signal_raw": "signal_raw_values",
        "signal_mapping": "signal_mapping_values",
        "label_raw": "label_raw_values",
    })
    out["direction"] = out.apply(lambda r: label_direction(r["label"], r["score"]), axis=1)
    out["is_active"] = out["score"].notna() & (out["score"].abs() > 0)
    return out


def clean_relationships(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    required = ["doc_id", "ticker", "current_company", "quarter", "relation_group", "entity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Relationships missing required columns: {missing}")

    rel = df.copy()
    for c in rel.columns:
        if rel[c].dtype == "object":
            rel[c] = rel[c].astype(str).str.strip()

    rel = rel[rel["entity"].fillna("").astype(str).str.strip().ne("")].copy()
    rel = rel[rel["relation_group"].fillna("").astype(str).str.lower().ne("none")].copy()
    rel = ensure_cols(rel, ["source_file", "source_folder", "parquet_source", "chunk_id", "chunk_uid", "evidence_id", "evidence_doc_id", "transcript_id", "relationship_type", "entity_type", "confidence"])

    rel["source_company_node"] = np.where(rel["ticker"].astype(str).str.strip().ne(""), "COMPANY::" + rel["ticker"].astype(str).str.strip(), "COMPANY::" + rel["current_company"].map(norm_text))
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
    rel = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()
    if rel.empty:
        return rel

    rel = rel.rename(columns={
        "doc_id": "relationship_doc_id",
        "llm_row_id": "relationship_row_id",
        "source_file": "relationship_source_file",
        "source_folder": "relationship_source_folder",
        "parquet_source": "relationship_parquet_source",
        "chunk_id": "relationship_chunk_id",
        "chunk_uid": "relationship_chunk_uid",
        "evidence_id": "relationship_evidence_id",
        "evidence_doc_id": "relationship_evidence_doc_id",
        "transcript_id": "relationship_transcript_id",
    })
    return rel


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
        meta[node] = {"ticker": str(r["ticker"]), "company": str(r["current_company"])}
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


def prepare_matched_relationships(relationships: pd.DataFrame, outlook: pd.DataFrame, include_self_edges: bool):
    company_map, ticker_map, meta = build_company_lookup(outlook)
    rel = relationships.copy()
    rel["target_company_node"] = rel["target_entity_norm"].map(lambda x: match_entity_to_company(x, company_map, ticker_map))
    unmatched = rel[rel["target_company_node"].fillna("").eq("")].copy()
    matched = rel[rel["target_company_node"].fillna("").ne("")].copy()
    if not include_self_edges:
        matched = matched[matched["source_company_node"] != matched["target_company_node"]].copy()
    subset = [c for c in ["source_company_node", "target_company_node", "relation_group_clean", "relationship_type", "quarter", "relationship_doc_id", "relationship_row_id"] if c in matched.columns]
    matched = matched.drop_duplicates(subset=subset)
    return matched, unmatched, meta


def make_outlook_lookup(outlook: pd.DataFrame):
    return {(r.company_node, r.quarter, r.signal): r for r in outlook.itertuples(index=False)}


def ntget(row, attr: str, default: Any = "") -> Any:
    return getattr(row, attr, default)


def select_relationships_for_window(matched_rel: pd.DataFrame, source_q: str, target_q: str, use_quarter_specific: bool) -> pd.DataFrame:
    if not use_quarter_specific:
        return matched_rel
    return matched_rel[matched_rel["quarter"].isin([source_q, target_q])].copy()


def build_events_for_pair(outlook_lookup: dict, meta: dict, matched_rel: pd.DataFrame, source_q: str, target_q: str, mode: str) -> pd.DataFrame:
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
            source_label = str(ntget(srow, "label", ""))
            target_label = str(ntget(trow, "label", ""))
            source_score = float(ntget(srow, "score", np.nan)) if not pd.isna(ntget(srow, "score", np.nan)) else np.nan
            target_score = float(ntget(trow, "score", np.nan)) if not pd.isna(ntget(trow, "score", np.nan)) else np.nan
            source_direction = label_direction(source_label, source_score)
            target_direction = label_direction(target_label, target_score)
            source_active = is_active_score(source_score)
            target_active = is_active_score(target_score)
            exact_match = source_active and target_active and source_label == target_label
            direction_match = source_active and target_active and source_direction == target_direction
            predicted_positive = source_active
            actual_positive = target_active and target_direction == source_direction
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
                "source_signal_raw_values": ntget(srow, "signal_raw_values", ""),
                "target_signal_raw_values": ntget(trow, "signal_raw_values", ""),
                "source_signal_mapping_values": ntget(srow, "signal_mapping_values", ""),
                "target_signal_mapping_values": ntget(trow, "signal_mapping_values", ""),
                "source_label": source_label,
                "target_label": target_label,
                "source_label_raw_values": ntget(srow, "label_raw_values", ""),
                "target_label_raw_values": ntget(trow, "label_raw_values", ""),
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
                "source_doc_ids": ntget(srow, "outlook_doc_ids", ""),
                "target_doc_ids": ntget(trow, "outlook_doc_ids", ""),
                "relationship_doc_id": str(edge.get("relationship_doc_id", "")),
                "source_outlook_row_ids": ntget(srow, "outlook_row_ids", ""),
                "target_outlook_row_ids": ntget(trow, "outlook_row_ids", ""),
                "relationship_row_id": str(edge.get("relationship_row_id", "")),
                "source_outlook_source_files": ntget(srow, "outlook_source_files", ""),
                "target_outlook_source_files": ntget(trow, "outlook_source_files", ""),
                "relationship_source_file": str(edge.get("relationship_source_file", "")),
                "source_outlook_parquet_sources": ntget(srow, "outlook_parquet_sources", ""),
                "target_outlook_parquet_sources": ntget(trow, "outlook_parquet_sources", ""),
                "relationship_parquet_source": str(edge.get("relationship_parquet_source", "")),
                "source_outlook_chunk_ids": ntget(srow, "outlook_chunk_ids", ""),
                "target_outlook_chunk_ids": ntget(trow, "outlook_chunk_ids", ""),
                "relationship_chunk_id": str(edge.get("relationship_chunk_id", "")),
                "source_outlook_chunk_uids": ntget(srow, "outlook_chunk_uids", ""),
                "target_outlook_chunk_uids": ntget(trow, "outlook_chunk_uids", ""),
                "relationship_chunk_uid": str(edge.get("relationship_chunk_uid", "")),
            })
    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    exp = events[events["source_active"]].copy()
    if exp.empty:
        return pd.DataFrame()
    group_cols = ["analysis_mode", "source_quarter", "target_quarter", "signal", "source_label", "source_direction", "relation_group"]
    rows = []
    for keys, g in exp.groupby(group_cols, dropna=False):
        analysis_mode, source_q, target_q, signal, source_label, source_direction, relation_group = keys
        exposed = len(g)
        target_active = int(g["target_active"].sum())
        exact = int(g["exact_match"].sum())
        direction = int(g["direction_match"].sum())
        valid_pred = g[g["prediction_correct"].notna()]
        correct = int(valid_pred["prediction_correct"].sum()) if not valid_pred.empty else 0
        rows.append({
            "analysis_mode": analysis_mode,
            "source_quarter": source_q,
            "target_quarter": target_q,
            "signal": signal,
            "source_label": source_label,
            "source_direction": source_direction,
            "relation_group": relation_group,
            "source_signal_raw_values": clean_join(g["source_signal_raw_values"]),
            "target_signal_raw_values": clean_join(g["target_signal_raw_values"]),
            "source_signal_mapping_values": clean_join(g["source_signal_mapping_values"]),
            "target_signal_mapping_values": clean_join(g["target_signal_mapping_values"]),
            "source_doc_ids": clean_join(g["source_doc_ids"]),
            "target_doc_ids": clean_join(g["target_doc_ids"]),
            "relationship_doc_ids": clean_join(g["relationship_doc_id"]),
            "source_outlook_row_ids": clean_join(g["source_outlook_row_ids"]),
            "target_outlook_row_ids": clean_join(g["target_outlook_row_ids"]),
            "relationship_row_ids": clean_join(g["relationship_row_id"]),
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
            "prediction_accuracy": correct / len(valid_pred) if len(valid_pred) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["analysis_mode", "source_quarter", "target_quarter", "signal", "relation_group"])


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
            "source_signal_raw_values": clean_join(g["source_signal_raw_values"]),
            "target_signal_raw_values": clean_join(g["target_signal_raw_values"]),
            "source_signal_mapping_values": clean_join(g["source_signal_mapping_values"]),
            "target_signal_mapping_values": clean_join(g["target_signal_mapping_values"]),
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
    return pd.DataFrame(rows).sort_values(["analysis_mode", "prediction_accuracy", "exposed_edges"], ascending=[True, False, False])


def plot_rate_by_group(agg: pd.DataFrame, mode: str, group_col: str, rate_col: str, out_png: Path, min_exposed: int):
    d = agg[(agg["analysis_mode"] == mode) & (agg["exposed_edges"] >= min_exposed)].copy()
    if d.empty:
        return
    g = d.groupby(group_col, as_index=False).agg(exposed_edges=("exposed_edges", "sum"), correct_edges=("prediction_correct_edges", "sum"), direction_edges=("direction_match_edges", "sum"))
    g[rate_col] = g["correct_edges"] / g["exposed_edges"] if rate_col == "prediction_accuracy" else g["direction_edges"] / g["exposed_edges"]
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
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")

    print("=" * 90)
    print("Two-part network prediction analysis with Parquet provenance")
    print("out_dir:", out_dir)
    print("quarter range:", args.start_quarter or "ALL", "to", args.end_quarter or "ALL")
    print("=" * 90)

    raw_outlook, raw_rel, manifest = load_inputs(args)
    save_table(manifest, out_dir / "input_file_manifest.parquet", args.write_csv)
    save_table(raw_outlook, out_dir / "raw_outlook_input_with_row_ids.parquet", args.write_csv)
    save_table(raw_rel, out_dir / "raw_relationships_input_with_row_ids.parquet", args.write_csv)

    outlook = clean_outlook(raw_outlook, args.start_quarter, args.end_quarter)
    relationships = clean_relationships(raw_rel, args.start_quarter, args.end_quarter)
    save_table(outlook, out_dir / "cleaned_outlook_all.parquet", args.write_csv)
    save_table(relationships, out_dir / "cleaned_relationships_all.parquet", args.write_csv)

    matched_rel, unmatched_rel, meta = prepare_matched_relationships(relationships, outlook, args.include_self_edges)
    save_table(matched_rel, out_dir / "matched_company_relationships.parquet", args.write_csv)
    save_table(unmatched_rel, out_dir / "unmatched_relationship_entities.parquet", args.write_csv)

    outlook_lookup = make_outlook_lookup(outlook)
    quarters = sorted(outlook["quarter"].dropna().unique(), key=quarter_to_index)
    pairs = adjacent_pairs(quarters)

    print("Available quarters:", quarters)
    print("Adjacent pairs:", pairs)

    cross_events_list = []
    for source_q, target_q in pairs:
        rel_for_pair = select_relationships_for_window(matched_rel, source_q, target_q, args.use_quarter_specific_relationships)
        if rel_for_pair.empty:
            continue
        events = build_events_for_pair(outlook_lookup, meta, rel_for_pair, source_q, target_q, "cross_quarter")
        if not events.empty:
            cross_events_list.append(events)
        print(f"cross_quarter {source_q}->{target_q}: rel={len(rel_for_pair):,}, events={len(events):,}")

    cross_events = pd.concat(cross_events_list, ignore_index=True).drop_duplicates() if cross_events_list else pd.DataFrame()
    cross_summary = summarize_events(cross_events)
    cross_agg = aggregate_summary(cross_summary)
    save_table(cross_events, out_dir / "cross_quarter_events.parquet", args.write_csv)
    save_table(cross_summary, out_dir / "cross_quarter_summary_by_window_signal_relation.parquet", args.write_csv)
    save_table(cross_agg, out_dir / "cross_quarter_prediction_accuracy.parquet", args.write_csv)

    same_events_list = []
    for q in quarters:
        rel_for_q = matched_rel[matched_rel["quarter"].eq(q)].copy() if args.use_quarter_specific_relationships else matched_rel
        if rel_for_q.empty:
            continue
        events = build_events_for_pair(outlook_lookup, meta, rel_for_q, q, q, "same_quarter")
        if not events.empty:
            same_events_list.append(events)
        print(f"same_quarter {q}: rel={len(rel_for_q):,}, events={len(events):,}")

    same_events = pd.concat(same_events_list, ignore_index=True).drop_duplicates() if same_events_list else pd.DataFrame()
    same_summary = summarize_events(same_events)
    same_agg = aggregate_summary(same_summary)
    save_table(same_events, out_dir / "same_quarter_events.parquet", args.write_csv)
    save_table(same_summary, out_dir / "same_quarter_summary_by_quarter_signal_relation.parquet", args.write_csv)
    save_table(same_agg, out_dir / "same_quarter_correlation_by_signal_relation.parquet", args.write_csv)

    combined_events = pd.concat([cross_events, same_events], ignore_index=True).drop_duplicates()
    combined_summary = pd.concat([cross_summary, same_summary], ignore_index=True).drop_duplicates()
    combined_agg = pd.concat([cross_agg, same_agg], ignore_index=True).drop_duplicates()
    save_table(combined_events, out_dir / "combined_events_cross_and_same_quarter.parquet", args.write_csv)
    save_table(combined_summary, out_dir / "combined_summary_cross_and_same_quarter.parquet", args.write_csv)
    save_table(combined_agg, out_dir / "combined_accuracy_correlation_summary.parquet", args.write_csv)

    # Audit raw -> mapped signals.
    mapping_rows = []
    for raw_values, mapped in outlook[["signal_raw_values", "signal"]].drop_duplicates().itertuples(index=False):
        for raw in str(raw_values).split("|"):
            raw = raw.strip()
            if raw:
                mapping_rows.append({"signal_raw": raw, "signal": mapped, "mapping": f"{raw} -> {mapped}"})
    signal_mapping_audit = pd.DataFrame(mapping_rows).drop_duplicates().sort_values(["signal", "signal_raw"])
    save_table(signal_mapping_audit, out_dir / "signal_mapping_audit.parquet", args.write_csv)

    plot_window_counts(cross_events, "cross_quarter", fig_dir / "cross_quarter_event_rows_by_window.png")
    plot_window_counts(same_events, "same_quarter", fig_dir / "same_quarter_event_rows_by_quarter.png")
    plot_rate_by_group(cross_agg, "cross_quarter", "signal", "prediction_accuracy", fig_dir / "cross_quarter_accuracy_by_signal.png", args.min_exposed_for_plot)
    plot_rate_by_group(cross_agg, "cross_quarter", "relation_group", "prediction_accuracy", fig_dir / "cross_quarter_accuracy_by_relation.png", args.min_exposed_for_plot)
    plot_rate_by_group(same_agg, "same_quarter", "signal", "direction_match_rate", fig_dir / "same_quarter_similarity_by_signal.png", args.min_exposed_for_plot)
    plot_rate_by_group(same_agg, "same_quarter", "relation_group", "direction_match_rate", fig_dir / "same_quarter_similarity_by_relation.png", args.min_exposed_for_plot)

    report = []
    report.append("# Two-Part Network Prediction Analysis with Parquet Provenance")
    report.append("")
    report.append("## Data")
    report.append(f"- Cleaned outlook rows: {len(outlook):,}")
    report.append(f"- Cleaned relationship rows: {len(relationships):,}")
    report.append(f"- Matched company relationships: {len(matched_rel):,}")
    report.append(f"- Unmatched relationship entities: {len(unmatched_rel):,}")
    report.append(f"- Available quarters: {', '.join(quarters)}")
    report.append(f"- Adjacent quarter windows: {len(pairs)}")
    report.append(f"- Cross-quarter event rows: {len(cross_events):,}")
    report.append(f"- Same-quarter event rows: {len(same_events):,}")
    report.append("")
    report.append("## Provenance columns")
    report.append("Event outputs preserve `source_doc_ids`, `target_doc_ids`, `relationship_doc_id`, `source_outlook_row_ids`, `target_outlook_row_ids`, `relationship_row_id`, raw signal names, signal mappings, source files, parquet sources, and chunk ids where present.")
    report.append("")
    report.append("## Signal mapping audit")
    report.append("See `signal_mapping_audit.parquet`, for example `loan_growth_outlook -> demand_outlook`.")
    report_path = out_dir / "two_part_analysis_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"SAVED {report_path}")
    print("DONE")


if __name__ == "__main__":
    main()
