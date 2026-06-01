#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4 HF: Clustering Method Comparison for Network Diffusion Analysis
==================================================================

This is the Hugging Face / Parquet version of run_cluster_method_comparison_v4.py.

Main changes compared with the local CSV/JSONL version
------------------------------------------------------
1. Reads two-part analysis outputs from Hugging Face Parquet files.
2. Reads LLM concept outputs from Hugging Face Parquet files.
3. Reads release-date metadata from Hugging Face evidence metadata Parquet.
4. Writes all major tabular outputs as Parquet by default.
5. Keeps provenance columns already produced by the parquet two-part pipeline.

Default HF sources
------------------
Two-part / LLM outputs:
    soysouce/earningALZ_twopart

LLM concept outputs:
    soysouce/earningALZ

RAG evidence metadata:
    soysouce/earningALZ_SBERT_evidence

Expected two-part Parquet files, either at repo root or under --hf-two-part-prefix:
    cleaned_outlook_all.parquet
    matched_company_relationships.parquet
    cross_quarter_events.parquet
    same_quarter_events.parquet

Expected LLM concept Parquet file, either at repo root or under --hf-llm-prefix:
    llm_concepts_all.parquet
or shard files matching:
    concepts_*.parquet

Expected evidence metadata file:
    rag_evidence_package_metadata_full_gpu_direct.parquet

Run example
-----------
cd E:/Projects/EearningAlz/RAG

python run_cluster_method_comparison_v4_hf.py ^
  --hf-two-part-dataset soysouce/earningALZ_twopart_twopart ^
  --hf-evidence-dataset soysouce/earningALZ_SBERT_evidence ^
  --out-dir rag_chroma_output/cluster_method_comparison_v4_hf ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2 ^
  --min-k 5 ^
  --max-k 20 ^
  --min-cluster-size 20 ^
  --min-exposed 10 ^
  --cooccur-min-weight 2 ^
  --write-csv-copy

Linux / AAU:
-----------
python run_cluster_method_comparison_v4_hf.py \
  --hf-two-part-dataset soysouce/earningALZ_twopart_twopart \
  --hf-evidence-dataset soysouce/earningALZ_SBERT_evidence \
  --out-dir rag_chroma_output/cluster_method_comparison_v4_hf \
  --start-quarter 2019Q2 \
  --end-quarter 2026Q2 \
  --min-k 5 \
  --max-k 20 \
  --min-cluster-size 20 \
  --min-exposed 10 \
  --cooccur-min-weight 2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from huggingface_hub import hf_hub_download, list_repo_files

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


# ============================================================
# Constants
# ============================================================

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

INDUSTRY_KEYWORDS = {
    "Banking / Financial Services": [
        "PJT", "GS", "EVR", "APO", "EQBK", "NBHC", "EBC", "BAC", "BCS", "UBS",
        "JPM", "MA", "PIPR", "CPAY", "PAYO",
        "BANCORP", "BANK", "FINANCIAL", "CREDIT", "INSURANCE",
        "MORGAN", "GOLDMAN", "WELLS", "CITI", "VISA", "MASTERCARD",
    ],
    "Cloud / AI / Digital Platforms": [
        "AMZN", "MSFT", "GOOG", "GOOGL", "META", "ORCL", "IBM", "SHOP", "CRM",
        "ADBE", "NOW", "SNOW", "OPRA", "YOU", "XPER",
        "CLOUD", "SOFTWARE", "SAAS",
    ],
    "Semiconductors / Hardware": [
        "NVDA", "INTC", "AMD", "MU", "TSM", "ASML", "AVGO", "QCOM", "SIMO",
        "AAPL", "DELL", "ANET", "IONQ", "ADEA", "GFS", "GLW",
        "CHIP", "SEMICONDUCTOR", "WAFER", "FOUNDRY",
    ],
    "Retail / Consumer / Restaurants": [
        "TGT", "WMT", "HD", "CVS", "YUMC", "COST", "LOW", "MCD", "SBUX",
        "BURL", "NKE", "JYNT",
        "RETAIL", "CONSUMER", "RESTAURANT",
    ],
    "Automotive / Mobility": [
        "GM", "F", "HMC", "TSLA", "ADNT",
        "FORD", "MOTORS", "AUTOMOTIVE", "VEHICLE",
    ],
    "Energy / Utilities": [
        "BP", "EPD", "AEP", "AQN", "GEL", "LBRT", "XOM", "CVX", "SHEL",
        "BSM", "DINO", "EOG",
        "ENERGY", "OIL", "GAS", "POWER", "UTILITY", "UTILITIES", "HAL",
    ],
    "Healthcare / Pharma": [
        "SNY", "IART", "BTMD", "PFE", "LLY", "UNH", "MRK", "BMY", "RVMD",
        "TSHA", "ALVO", "CDXS", "HIMS", "MGNX",
        "HEALTH", "MEDICAL", "PHARMA", "BIOTECH", "THERAPEUTICS", "HOSPITAL",
    ],
    "Industrial / Manufacturing / Aerospace": [
        "CAT", "AERO", "ROP", "CARR", "PKOH", "STRL", "EML", "GEV", "WRK",
        "HON", "AIR", "IDCC", "PCT", "OKLO", "ASTS", "ATOM", "RCAT",
        "INDUSTRIAL", "MANUFACTURING", "AEROSPACE", "MACHINERY", "CONSTRUCTION",
    ],
    "Airlines / Travel / Leisure": [
        "UAL", "DAL", "AAL", "SGHC", "MAR",
        "SOUTHWEST", "TRAVEL", "AIRLINES", "HOTEL", "LEISURE",
    ],
    "Telecom / Media / Cable": [
        "CHTR", "CMCSA", "VZ", "T", "TMUS", "SIRI",
        "TELECOM", "CABLE", "MEDIA", "COMMUNICATIONS",
    ],
}

CONCEPT_RULES = {
    "Semiconductors / Hardware": (["semiconductor_supply", "chip_supply", "supplier_constraint"], 0.18, 0.35),
    "Cloud / AI / Digital Platforms": (["cloud_infrastructure", "data_center_capacity"], 0.12, 0.25),
    "Retail / Consumer / Restaurants": (["customer_demand", "inventory_pressure", "pricing_pressure", "logistics_shipping"], 0.45, 1.10),
    "Energy / Utilities": (["oil_energy_supply"], 0.15, 0.15),
    "Industrial / Manufacturing / Aerospace": (["manufacturing_capacity", "production_capacity", "capex_expansion"], 0.65, 1.30),
}

SIGNAL_TYPES = [
    "demand_outlook",
    "supply_outlook",
    "margin_outlook",
    "capex_outlook",
    "inventory_outlook",
    "pricing_outlook",
]


# ============================================================
# Args and basic utilities
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # HF sources
    p.add_argument("--hf-two-part-dataset", default="soysouce/earningALZ_twopart")
    p.add_argument("--hf-two-part-revision", default="main")
    p.add_argument("--hf-two-part-prefix", default="", help="Optional folder prefix for two-part outputs on HF.")

    p.add_argument("--hf-llm-dataset", default="soysouce/earningALZ", help="Dataset containing concept parquet outputs.")
    p.add_argument("--hf-llm-revision", default="main")
    p.add_argument("--hf-llm-prefix", default="", help="Optional folder prefix for LLM output parquets on HF.")

    p.add_argument("--hf-evidence-dataset", default="soysouce/earningALZ_SBERT_evidence")
    p.add_argument("--hf-evidence-revision", default="main")
    p.add_argument("--hf-evidence-metadata-file", default="rag_evidence_package_metadata_full_gpu_direct.parquet")

    # Optional explicit HF filenames
    p.add_argument("--hf-outlook-file", default="cleaned_outlook_all.parquet")
    p.add_argument("--hf-relationships-file", default="matched_company_relationships.parquet")
    p.add_argument("--hf-cross-events-file", default="cross_quarter_events.parquet")
    p.add_argument("--hf-same-events-file", default="same_quarter_events.parquet")
    p.add_argument("--hf-concepts-file", default="llm_concepts_all.parquet")
    p.add_argument("--auto-discover-files", action="store_true", default=True,
                   help="Automatically discover required parquet files by basename anywhere in the HF repos.")
    p.add_argument("--no-auto-discover-files", dest="auto_discover_files", action="store_false")

    # Analysis controls
    p.add_argument("--out-dir", default="rag_chroma_output/cluster_method_comparison_v4_hf")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--min-k", type=int, default=5)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--min-cluster-size", type=int, default=20)
    p.add_argument("--min-exposed", type=int, default=10)
    p.add_argument("--winsor-quantile", type=float, default=0.99)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cooccur-min-weight", type=float, default=2.0)
    p.add_argument("--skip-spectral", action="store_true")
    p.add_argument("--run-optional-hdbscan", action="store_true")
    p.add_argument("--write-csv-copy", action="store_true")

    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table(df: pd.DataFrame, path: Path, write_csv_copy: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"SAVED {path} rows={len(df):,} cols={len(df.columns):,}")
    if write_csv_copy:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"SAVED {csv_path} rows={len(df):,}")


def clean_node_value(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "0"}:
        return ""
    return s


def quarter_to_index(q: str) -> float:
    m = re.match(r"^(\d{4})Q([1-4])$", str(q).strip())
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def filter_quarter_range(df: pd.DataFrame, q_cols: list[str], start_q: str, end_q: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in q_cols:
        if col not in out.columns:
            continue
        idx = out[col].map(quarter_to_index)
        mask = idx.notna()
        if start_q:
            mask &= idx >= quarter_to_index(start_q)
        if end_q:
            mask &= idx <= quarter_to_index(end_q)
        out = out[mask].copy()
    return out


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9&.\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    replacements = {
        " corporation": " corp", " incorporated": " inc",
        " company": " co", " limited": " ltd",
        " technologies": " tech", " technology": " tech",
        " international": " intl", " holdings": "",
        " holding": "", " group": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return re.sub(r"\s+", " ", s).strip()


def company_node_from(ticker: str, company: str) -> str:
    ticker = clean_node_value(ticker)
    company = clean_node_value(company)
    if ticker:
        return "COMPANY::" + ticker
    return "COMPANY::" + norm_text(company)


# ============================================================
# HF loading
# ============================================================

def prefixed(prefix: str, filename: str) -> str:
    prefix = prefix.strip().strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def list_hf_parquet_files(repo_id: str, revision: str) -> list[str]:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    return sorted([f for f in files if f.endswith(".parquet")])


def select_hf_file(
    repo_id: str,
    filename: str,
    prefix: str,
    revision: str,
    role: str,
    required_columns: set[str] | None = None,
) -> str:
    """
    Select a parquet file from a HF dataset.

    Priority:
    1. exact prefix/filename
    2. exact filename at repo root
    3. common two-part subfolders
    4. any parquet whose basename equals filename
    5. any parquet whose basename contains the filename stem and passes required-column check
    """
    files = list_hf_parquet_files(repo_id, revision)
    prefix = prefix.strip().strip("/")
    stem = Path(filename).stem

    candidates = [
        prefixed(prefix, filename),
        filename,
        prefixed(prefix, f"two_part_network_prediction_analysis_hf/{filename}"),
        prefixed(prefix, f"two_part_network_prediction_analysis_parquet/{filename}"),
        prefixed(prefix, f"two_part_network_prediction_analysis/{filename}"),
        prefixed(prefix, f"network_prediction_analysis/{filename}"),
        prefixed(prefix, f"data/{filename}"),
        prefixed(prefix, f"results/{filename}"),
    ]
    candidates += [f for f in files if Path(f).name == filename]
    candidates += [f for f in files if stem in Path(f).stem]

    # Deduplicate while preserving order.
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    errors = []
    for c in candidates:
        if c not in files:
            continue
        if required_columns:
            try:
                local = hf_hub_download(repo_id=repo_id, filename=c, repo_type="dataset", revision=revision)
                cols = set(pd.read_parquet(local).columns)
                if not required_columns.issubset(cols):
                    errors.append(f"{c}: missing {sorted(required_columns - cols)}")
                    continue
            except Exception as e:
                errors.append(f"{c}: {repr(e)}")
                continue
        print(f"HF SELECT [{role}]: {repo_id}/{c}")
        return c

    available = "\n".join(f"  - {f}" for f in files[:120])
    detail = "\nColumn-check failures:\n" + "\n".join(errors[:20]) if errors else ""
    raise FileNotFoundError(
        f"Could not find required HF parquet for role={role} in dataset={repo_id}.\n"
        f"Target filename={filename}, prefix={prefix or '[root]'}\n"
        f"Tried candidate paths/basenames. Available parquet files include:\n{available}{detail}"
    )


def read_hf_parquet(
    repo_id: str,
    filename: str,
    prefix: str,
    revision: str,
    role: str,
    required_columns: set[str] | None = None,
) -> pd.DataFrame:
    remote_file = select_hf_file(repo_id, filename, prefix, revision, role, required_columns)
    local = hf_hub_download(repo_id=repo_id, filename=remote_file, repo_type="dataset", revision=revision)
    df = pd.read_parquet(local)
    df["_hf_dataset"] = repo_id
    df["_hf_file"] = remote_file
    df["_hf_requested_file"] = filename
    print(f"HF LOAD [{role}]: rows={len(df):,}, cols={len(df.columns):,}")
    return df


def read_hf_concepts(args) -> pd.DataFrame:
    """
    Load concept features from HF.

    First tries --hf-llm-dataset, then tries --hf-two-part-dataset. This is useful
    when concepts are uploaded together with two-part results.
    """
    repos = [
        (args.hf_llm_dataset, args.hf_llm_revision, args.hf_llm_prefix),
    ]
    if args.hf_two_part_dataset != args.hf_llm_dataset:
        repos.append((args.hf_two_part_dataset, args.hf_two_part_revision, args.hf_two_part_prefix))

    for repo_id, revision, prefix_arg in repos:
        try:
            files = set(list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))
        except Exception as e:
            print(f"WARNING: cannot list concepts repo {repo_id}: {e}")
            continue

        preferred = [
            prefixed(prefix_arg, args.hf_concepts_file),
            args.hf_concepts_file,
            prefixed(prefix_arg, f"merged_parquet_outputs/{args.hf_concepts_file}"),
            prefixed(prefix_arg, f"llm_parquet_outputs_hf/{args.hf_concepts_file}"),
        ]
        for c in preferred:
            if c in files:
                path = hf_hub_download(repo_id, filename=c, repo_type="dataset", revision=revision)
                df = pd.read_parquet(path)
                df["_hf_dataset"] = repo_id
                df["_hf_file"] = c
                print(f"HF LOAD [concepts]: {repo_id}/{c} rows={len(df):,}")
                return df

        prefix = prefix_arg.strip().strip("/")
        concept_files = [
            f for f in files
            if f.endswith(".parquet")
            and Path(f).name.startswith("concepts_")
            and (not prefix or f.startswith(prefix + "/"))
        ]
        if concept_files:
            frames = []
            for f in sorted(concept_files):
                path = hf_hub_download(repo_id, filename=f, repo_type="dataset", revision=revision)
                d = pd.read_parquet(path)
                d["_hf_dataset"] = repo_id
                d["_hf_file"] = f
                frames.append(d)
                print(f"HF LOAD [concept shard]: {repo_id}/{f} rows={len(d):,}")
            return pd.concat(frames, ignore_index=True).drop_duplicates()

    print("WARNING: no concepts parquet found on HF. Continuing with empty concepts.")
    return pd.DataFrame()


def read_hf_evidence_metadata(args) -> pd.DataFrame:
    files = set(list_repo_files(repo_id=args.hf_evidence_dataset, repo_type="dataset", revision=args.hf_evidence_revision))
    c = args.hf_evidence_metadata_file
    if c not in files:
        print(f"WARNING: evidence metadata file not found: {c}. Continuing without evidence dates.")
        return pd.DataFrame()
    path = hf_hub_download(args.hf_evidence_dataset, filename=c, repo_type="dataset", revision=args.hf_evidence_revision)
    df = pd.read_parquet(path)
    df["_hf_dataset"] = args.hf_evidence_dataset
    df["_hf_file"] = c
    print(f"HF LOAD [evidence metadata]: {args.hf_evidence_dataset}/{c} rows={len(df):,}")
    return df


# ============================================================
# Release date loading from HF metadata
# ============================================================

def find_date_column(df: pd.DataFrame) -> str:
    candidates = [
        "release_date", "earnings_call_date", "call_date", "date", "published_date",
        "publish_date", "publication_date", "transcript_date", "datetime"
    ]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    return ""


def load_dates_from_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])

    dc = find_date_column(df)
    if not dc or "quarter" not in df.columns:
        print("WARNING: evidence metadata has no usable date/quarter columns.")
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])

    ticker_col = "ticker" if "ticker" in df.columns else ""
    company_col = next((c for c in ["current_company", "company", "company_name", "name"] if c in df.columns), "")
    if not ticker_col and not company_col:
        print("WARNING: evidence metadata has no company/ticker columns.")
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])

    d = df.copy()
    d["release_date"] = pd.to_datetime(d[dc], errors="coerce")
    d = d[d["release_date"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])

    d["company_node"] = d.apply(
        lambda r: company_node_from(r[ticker_col] if ticker_col else "", r[company_col] if company_col else ""),
        axis=1
    )
    d["quarter"] = d["quarter"].astype(str).str.strip()
    return d.groupby(["company_node", "quarter"], as_index=False).agg(
        release_date=("release_date", "min"),
        release_date_count=("release_date", "count")
    )


# ============================================================
# Graph and feature helpers
# ============================================================

def build_relationship_graph(rel: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    if rel.empty:
        return G

    s_col = "source_company_node"
    t_col = "target_company_node"
    if s_col not in rel.columns or t_col not in rel.columns:
        raise ValueError("Relationship table must contain source_company_node and target_company_node.")

    for _, r in rel.iterrows():
        s = clean_node_value(r.get(s_col, ""))
        t = clean_node_value(r.get(t_col, ""))
        if not s or not t or s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += 1
        else:
            G.add_edge(s, t, weight=1)
    return G


def build_signal_cooccurrence_graph(cross_events: pd.DataFrame, same_events: pd.DataFrame, min_weight: float = 2.0) -> nx.Graph:
    pair_weights: dict[tuple[str, str], float] = defaultdict(float)

    for events in [cross_events, same_events]:
        if events.empty:
            continue
        ev = events.copy()
        if "source_active" in ev.columns:
            ev = ev[ev["source_active"].astype(bool)]
        if ev.empty:
            continue

        for _, row in ev.iterrows():
            s = clean_node_value(str(row.get("source_node", "")))
            t = clean_node_value(str(row.get("target_node", "")))
            if not s or not t or s == t:
                continue
            direction_match = bool(row.get("direction_match", False))
            w = 2.0 if direction_match else 0.5
            pair_weights[(min(s, t), max(s, t))] += w

    G = nx.Graph()
    for (s, t), w in pair_weights.items():
        if w >= min_weight:
            G.add_edge(s, t, weight=w)

    print(f"Signal co-occurrence graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def company_metadata(outlook: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if {"company_node", "ticker", "current_company"}.issubset(outlook.columns):
        frames.append(outlook[["company_node", "ticker", "current_company"]].rename(columns={"current_company": "company"}))

    for side in ["source", "target"]:
        node_col = f"{side}_company_node"
        ticker_col = f"{side}_ticker"
        company_col = f"{side}_company"
        if node_col in rel.columns:
            cols = [node_col]
            if ticker_col in rel.columns:
                cols.append(ticker_col)
            if company_col in rel.columns:
                cols.append(company_col)
            x = rel[cols].rename(columns={node_col: "company_node", ticker_col: "ticker", company_col: "company"})
            frames.append(x)

    if not frames:
        return pd.DataFrame(columns=["company_node", "ticker", "company"])

    meta = pd.concat(frames, ignore_index=True).fillna("")
    if "ticker" not in meta.columns:
        meta["ticker"] = ""
    if "company" not in meta.columns:
        meta["company"] = ""

    meta["company_node"] = meta["company_node"].map(clean_node_value)
    meta = meta[meta["company_node"].ne("")].copy()
    return meta.groupby("company_node", as_index=False).agg(
        ticker=("ticker", lambda x: next((str(v) for v in x if clean_node_value(v)), "")),
        company=("company", lambda x: next((str(v) for v in x if clean_node_value(v)), "")),
    )


def compute_lead_lag_features(cross_events: pd.DataFrame, same_events: pd.DataFrame) -> pd.DataFrame:
    all_events = pd.concat([ev for ev in [cross_events, same_events] if not ev.empty], ignore_index=True)
    if all_events.empty or "source_before_target" not in all_events.columns:
        return pd.DataFrame()

    ev = all_events.copy()
    if "source_active" in ev.columns:
        ev = ev[ev["source_active"].astype(bool)]
    ev["source_before_target"] = pd.to_numeric(ev["source_before_target"], errors="coerce")

    rows = []
    for node, g in ev.groupby("source_node"):
        row = {"company_node": node}
        overall = g["source_before_target"].dropna()
        row["lead_ratio_overall"] = float(overall.mean()) if len(overall) else 0.0
        row["lead_count_total"] = int(len(overall))

        per_signal = []
        for sig in SIGNAL_TYPES:
            sg = g[g.get("signal", pd.Series(dtype=str)).eq(sig)]["source_before_target"].dropna()
            val = float(sg.mean()) if len(sg) >= 3 else np.nan
            row[f"lead_ratio_{sig}"] = val
            if not np.isnan(val):
                per_signal.append(val)

        row["lead_lag_consistency"] = 1.0 / (1.0 + float(np.std(per_signal))) if len(per_signal) >= 2 else 0.0
        rows.append(row)

    return pd.DataFrame(rows).fillna(0)


def compute_signal_homophily_features(cross_events: pd.DataFrame, same_events: pd.DataFrame, rel_graph: nx.Graph) -> pd.DataFrame:
    all_events = pd.concat([ev for ev in [cross_events, same_events] if not ev.empty], ignore_index=True)
    if all_events.empty:
        return pd.DataFrame()

    ev = all_events.copy()
    if "source_active" in ev.columns:
        ev = ev[ev["source_active"].astype(bool)]

    rows = []
    for node in rel_graph.nodes():
        neighbors = set(rel_graph.neighbors(node))
        if not neighbors:
            rows.append({"company_node": node, "signal_homophily_overall": 0.0, "neighbor_active_ratio": 0.0})
            continue

        src_ev = ev[ev["source_node"].eq(node)]
        nb_ev = ev[ev["source_node"].isin(neighbors)]
        row = {"company_node": node}

        row["signal_homophily_overall"] = float(src_ev["direction_match"].mean()) if "direction_match" in src_ev.columns and not src_ev.empty else 0.0
        active_neighbors = set(nb_ev["source_node"].unique()) if not nb_ev.empty else set()
        row["neighbor_active_ratio"] = len(active_neighbors) / len(neighbors) if neighbors else 0.0

        for sig in SIGNAL_TYPES:
            sig_ev = src_ev[src_ev.get("signal", pd.Series(dtype=str)).eq(sig)] if not src_ev.empty else pd.DataFrame()
            row[f"signal_homophily_{sig}"] = float(sig_ev["direction_match"].mean()) if not sig_ev.empty and "direction_match" in sig_ev.columns else 0.0

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)


def compute_cooccur_centrality_features(cooccur_graph: nx.Graph, feat_nodes: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"company_node": feat_nodes})
    if cooccur_graph.number_of_nodes() == 0:
        return df

    df["cooccur_degree"] = df["company_node"].map(dict(cooccur_graph.degree())).fillna(0)
    df["cooccur_strength"] = df["company_node"].map(dict(cooccur_graph.degree(weight="weight"))).fillna(0)
    if cooccur_graph.number_of_edges() > 0:
        df["cooccur_pagerank"] = df["company_node"].map(nx.pagerank(cooccur_graph, weight="weight")).fillna(0)
        try:
            df["cooccur_clustering"] = df["company_node"].map(nx.clustering(cooccur_graph, weight="weight")).fillna(0)
        except Exception:
            df["cooccur_clustering"] = 0.0
    else:
        df["cooccur_pagerank"] = 0.0
        df["cooccur_clustering"] = 0.0
    return df


def build_base_features(outlook: pd.DataFrame, rel: pd.DataFrame, concepts: pd.DataFrame, cross_events: pd.DataFrame, same_events: pd.DataFrame, cooccur_graph: nx.Graph):
    outlook = outlook.copy()
    rel = rel.copy()

    outlook["company_node"] = outlook["company_node"].map(clean_node_value)
    outlook = outlook[outlook["company_node"].ne("")].copy()

    for c in ["source_company_node", "target_company_node"]:
        rel[c] = rel[c].map(clean_node_value)
    rel = rel[
        rel["source_company_node"].ne("") &
        rel["target_company_node"].ne("") &
        rel["source_company_node"].ne(rel["target_company_node"])
    ].copy()

    G_rel = build_relationship_graph(rel)

    nodes = sorted(set(G_rel.nodes()) | set(outlook["company_node"].unique()) | set(cooccur_graph.nodes()))
    feat = pd.DataFrame({"company_node": nodes})
    feat = feat.merge(company_metadata(outlook, rel), on="company_node", how="left")

    # Relationship graph centrality
    feat["rel_degree"] = feat["company_node"].map(dict(G_rel.degree())).fillna(0)
    feat["rel_weighted_degree"] = feat["company_node"].map(dict(G_rel.degree(weight="weight"))).fillna(0)
    feat["rel_pagerank"] = feat["company_node"].map(nx.pagerank(G_rel, weight="weight") if G_rel.number_of_edges() else {}).fillna(0)

    if G_rel.number_of_nodes() > 1000:
        rel_betweenness = nx.betweenness_centrality(G_rel, k=min(500, G_rel.number_of_nodes()), seed=42, weight="weight")
    elif G_rel.number_of_nodes() > 0:
        rel_betweenness = nx.betweenness_centrality(G_rel, weight="weight")
    else:
        rel_betweenness = {}
    feat["rel_betweenness"] = feat["company_node"].map(rel_betweenness).fillna(0)

    outcnt = rel.groupby("source_company_node").size().rename("out_edge_rows").reset_index().rename(columns={"source_company_node": "company_node"})
    incnt = rel.groupby("target_company_node").size().rename("in_edge_rows").reset_index().rename(columns={"target_company_node": "company_node"})
    feat = feat.merge(outcnt, on="company_node", how="left").merge(incnt, on="company_node", how="left")

    rg = "relation_group" if "relation_group" in rel.columns else "relation_group_clean"
    if rg in rel.columns:
        piv = rel.groupby(["source_company_node", rg]).size().reset_index(name="n").pivot_table(
            index="source_company_node", columns=rg, values="n", fill_value=0
        ).reset_index()
        piv = piv.rename(columns={"source_company_node": "company_node"})
        piv.columns = ["company_node" if c == "company_node" else f"rel_count_{c}" for c in piv.columns]
        feat = feat.merge(piv, on="company_node", how="left")

    # Outlook behavior
    outlook["score"] = pd.to_numeric(outlook["score"], errors="coerce")
    score = outlook.pivot_table(index="company_node", columns="signal", values="score", aggfunc="mean", fill_value=0).reset_index()
    score.columns = ["company_node" if c == "company_node" else f"mean_{c}_score" for c in score.columns]

    tmp = outlook.assign(active=outlook["score"].abs() > 0, pos=outlook["score"] > 0, neg=outlook["score"] < 0)
    agg = tmp.groupby("company_node", as_index=False).agg(
        active_signal_rows=("active", "sum"),
        total_outlook_rows=("signal", "count"),
        positive_signal_rows=("pos", "sum"),
        negative_signal_rows=("neg", "sum"),
        active_quarters=("quarter", "nunique"),
        mean_outlook_score=("score", "mean"),
        std_outlook_score=("score", "std"),
    )
    agg["active_signal_ratio"] = agg["active_signal_rows"] / agg["total_outlook_rows"].replace(0, np.nan)
    agg["positive_signal_ratio"] = agg["positive_signal_rows"] / agg["total_outlook_rows"].replace(0, np.nan)
    agg["negative_signal_ratio"] = agg["negative_signal_rows"] / agg["total_outlook_rows"].replace(0, np.nan)
    feat = feat.merge(agg, on="company_node", how="left").merge(score, on="company_node", how="left")

    # Signal dynamics
    dyn_rows = []
    for node, g in outlook.sort_values("quarter").groupby("company_node"):
        pivot = g.pivot_table(index="quarter", columns="signal", values="score", aggfunc="mean").sort_index()
        vals = pivot.values.flatten()
        vals = vals[~pd.isna(vals)]
        changes = []
        for c in pivot.columns:
            s = pivot[c].dropna()
            if len(s) > 1:
                changes.extend(np.abs(np.diff(s.values)).tolist())
        dyn_rows.append({
            "company_node": node,
            "signal_score_std_all": float(np.std(vals)) if len(vals) else 0,
            "mean_abs_signal_change": float(np.mean(changes)) if changes else 0,
            "signal_observation_count": len(vals),
        })
    if dyn_rows:
        feat = feat.merge(pd.DataFrame(dyn_rows), on="company_node", how="left")

    # Concepts
    if not concepts.empty:
        concepts = concepts.copy()
        ticker_col = "ticker" if "ticker" in concepts.columns else ""
        company_col = next((c for c in ["current_company", "company", "company_name"] if c in concepts.columns), "")
        if "company_node" not in concepts.columns:
            concepts["company_node"] = concepts.apply(
                lambda r: company_node_from(r[ticker_col] if ticker_col else "", r[company_col] if company_col else ""),
                axis=1
            )
        cols = [c for c in CONCEPT_COLUMNS if c in concepts.columns]
        for c in cols:
            concepts[c] = pd.to_numeric(concepts[c], errors="coerce").fillna(0)
        if cols:
            con = concepts.groupby("company_node")[cols].mean().reset_index()
            con.columns = ["company_node" if c == "company_node" else f"concept_rate_{c}" for c in con.columns]

            raw = concepts.groupby("company_node")[cols].mean()
            arr = raw.values
            row_sum = arr.sum(axis=1)
            p = np.divide(arr, row_sum[:, None], out=np.zeros_like(arr, dtype=float), where=row_sum[:, None] != 0)
            entropy = -(p * np.where(p > 0, np.log(p), 0)).sum(axis=1)
            comp = pd.DataFrame({
                "company_node": raw.index,
                "concept_entropy": entropy,
                "supply_chain_intensity": raw[[c for c in ["chip_supply", "semiconductor_supply", "supplier_constraint", "raw_material_supply", "logistics_shipping"] if c in raw.columns]].sum(axis=1).values,
                "capex_intensity": raw[[c for c in ["capex_expansion", "production_capacity", "manufacturing_capacity"] if c in raw.columns]].sum(axis=1).values,
                "cloud_ai_intensity": raw[[c for c in ["cloud_infrastructure", "data_center_capacity"] if c in raw.columns]].sum(axis=1).values,
                "customer_pricing_intensity": raw[[c for c in ["customer_demand", "pricing_pressure", "inventory_pressure"] if c in raw.columns]].sum(axis=1).values,
            })
            feat = feat.merge(con, on="company_node", how="left").merge(comp, on="company_node", how="left")

    # V4 co-occurrence centrality
    feat = feat.merge(compute_cooccur_centrality_features(cooccur_graph, feat["company_node"].tolist()), on="company_node", how="left")

    return feat.fillna(0), G_rel


def add_timing_features(feat: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    if dates.empty:
        return feat
    d = dates.copy()
    d["release_date"] = pd.to_datetime(d["release_date"], errors="coerce")
    d = d[d["release_date"].notna()].copy()
    if d.empty:
        return feat
    d["day_of_year"] = d["release_date"].dt.dayofyear
    agg = d.groupby("company_node", as_index=False).agg(
        avg_release_day_of_year=("day_of_year", "mean"),
        std_release_day_of_year=("day_of_year", "std"),
        release_date_obs=("release_date", "count"),
    ).fillna(0)
    return feat.merge(agg, on="company_node", how="left").fillna(0)


def add_propagation_history_features(feat: pd.DataFrame, cross_events: pd.DataFrame) -> pd.DataFrame:
    if cross_events.empty:
        return feat
    e = cross_events[cross_events["source_active"].astype(bool)].copy()
    if e.empty:
        return feat
    src = e.groupby("source_node", as_index=False).agg(
        source_exposure_count=("signal", "count"),
        source_direction_success=("direction_match", "mean"),
        source_exact_success=("exact_match", "mean"),
    ).rename(columns={"source_node": "company_node"})
    tgt = e.groupby("target_node", as_index=False).agg(
        target_received_count=("signal", "count"),
        target_received_active_rate=("target_active", "mean"),
        target_received_direction_rate=("direction_match", "mean"),
    ).rename(columns={"target_node": "company_node"})
    return feat.merge(src, on="company_node", how="left").merge(tgt, on="company_node", how="left").fillna(0)


def add_v4_propagation_features(feat: pd.DataFrame, cross_events: pd.DataFrame, same_events: pd.DataFrame, G_rel: nx.Graph) -> pd.DataFrame:
    ll = compute_lead_lag_features(cross_events, same_events)
    if not ll.empty:
        feat = feat.merge(ll, on="company_node", how="left")

    homo = compute_signal_homophily_features(cross_events, same_events, G_rel)
    if not homo.empty:
        feat = feat.merge(homo, on="company_node", how="left")

    return feat.fillna(0)


def preprocess_features(feat: pd.DataFrame, feature_cols: list[str], winsor_q: float):
    X = feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    keep = [c for c in X.columns if X[c].std() > 1e-12]
    X = X[keep]

    Xp = X.copy()
    for c in Xp.columns:
        s = pd.to_numeric(Xp[c], errors="coerce").fillna(0)
        if s.min() >= 0:
            s = np.log1p(s)
        hi = s.quantile(winsor_q)
        lo = s.quantile(1 - winsor_q) if s.min() < 0 else s.min()
        Xp[c] = s.clip(lo, hi)

    X_scaled = StandardScaler().fit_transform(Xp.values)
    return X_scaled, list(Xp.columns), Xp


# ============================================================
# Cluster labeling
# ============================================================

def parse_top_concepts(s: str) -> dict:
    out = {}
    if pd.isna(s):
        return out
    for part in str(s).split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            out[k.strip()] = float(v)
        except Exception:
            pass
    return out


def score_company_keywords(text: str):
    text_u = "" if pd.isna(text) else str(text).upper()
    tokens = set(re.findall(r"[A-Z0-9&.\-]+", text_u))
    scores: Counter = Counter()
    evidence = defaultdict(list)

    for label, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            kw_u = kw.upper()
            if len(kw_u) <= 6:
                hit = kw_u in tokens
                weight = 3.0
            else:
                hit = kw_u in text_u
                weight = 1.0
            if hit:
                scores[label] += weight
                evidence[label].append(kw)

    return scores, evidence


def score_concepts(concepts: dict):
    scores: Counter = Counter()
    evidence = defaultdict(list)
    for label, (cols, max_th, sum_th) in CONCEPT_RULES.items():
        vals = {c: concepts.get(c, 0.0) for c in cols}
        max_val = max(vals.values()) if vals else 0
        sum_val = sum(vals.values())
        if max_val >= max_th or sum_val >= sum_th:
            score = 1.5 + 5.0 * max_val + sum_val
            scores[label] += score
            evidence[label].append(f"concept_max={max_val:.3f}, concept_sum={sum_val:.3f}")
    return scores, evidence


def infer_label(top_companies: str, top_concepts: str, num_companies: int):
    company_scores, company_ev = score_company_keywords(top_companies)
    concept_scores, concept_ev = score_concepts(parse_top_concepts(top_concepts))

    scores: Counter = Counter()
    scores.update(company_scores)
    for k, v in concept_scores.items():
        scores[k] += 0.65 * v

    if not scores:
        return "Mixed / Other", "no signal"

    top = scores.most_common(2)
    first_label, first_score = top[0]
    second_score = top[1][1] if len(top) > 1 else 0.0

    if first_score >= 1.5 * second_score or num_companies <= 2:
        label = first_label
    else:
        second_label = top[1][0]
        label = f"Mixed / {first_label[:18]} + {second_label[:18]}"

    diag = []
    for lab, sc in scores.most_common():
        ev = list(company_ev.get(lab, []))[:4] + list(concept_ev.get(lab, []))[:4]
        diag.append(f"{lab}:{sc:.1f} [{' | '.join(ev)}]")
    return label, "; ".join(diag)


def cluster_summary(assign: pd.DataFrame, feat: pd.DataFrame, rel: pd.DataFrame, G_rel: nx.Graph):
    concept_cols = [c for c in feat.columns if c.startswith("concept_rate_")]
    rows = []
    for cid, g in assign.groupby("cluster_id", dropna=False):
        nodes = set(g["company_node"])
        sub = G_rel.subgraph(nodes)

        central = sorted([(n, G_rel.degree(n) if n in G_rel else 0) for n in nodes], key=lambda x: x[1], reverse=True)[:15]
        top_companies = []
        for n, _ in central:
            r = g[g["company_node"].eq(n)].iloc[0]
            ticker = clean_node_value(r.get("ticker", ""))
            company = clean_node_value(r.get("company", ""))
            top_companies.append(ticker if ticker else company)

        tmp = feat[feat["company_node"].isin(nodes)]
        cm = tmp[concept_cols].mean().to_dict() if concept_cols else {}
        top_concepts = "; ".join(
            f"{k.replace('concept_rate_', '')}:{v:.3f}"
            for k, v in sorted(cm.items(), key=lambda x: x[1], reverse=True)[:6]
            if v > 0
        )

        label, diag = infer_label("; ".join(top_companies), top_concepts, len(nodes))

        external_edges = 0
        for n in nodes:
            if n in G_rel:
                external_edges += sum(1 for nb in G_rel.neighbors(n) if nb not in nodes)

        rg = "relation_group" if "relation_group" in rel.columns else "relation_group_clean"
        rsub = rel[rel["source_company_node"].isin(nodes)]
        top_rels = ""
        if rg in rsub.columns and not rsub.empty:
            top_rels = "; ".join(f"{idx}:{val}" for idx, val in rsub[rg].value_counts().head(6).items())

        rows.append({
            "cluster_id": int(cid),
            "cluster_theme_label": label,
            "num_companies": len(nodes),
            "internal_edges": sub.number_of_edges(),
            "external_edges": external_edges,
            "internal_edge_ratio": sub.number_of_edges() / (sub.number_of_edges() + external_edges) if sub.number_of_edges() + external_edges else 0,
            "density": nx.density(sub) if sub.number_of_nodes() > 1 else 0,
            "top_companies": "; ".join(top_companies),
            "top_relation_groups": top_rels,
            "top_concepts": top_concepts,
            "label_diagnostics": diag,
            "mean_cooccur_degree": float(tmp["cooccur_degree"].mean()) if "cooccur_degree" in tmp else 0,
            "mean_lead_ratio_overall": float(tmp["lead_ratio_overall"].mean()) if "lead_ratio_overall" in tmp else 0,
            "mean_signal_homophily": float(tmp["signal_homophily_overall"].mean()) if "signal_homophily_overall" in tmp else 0,
        })

    return pd.DataFrame(rows).sort_values("num_companies", ascending=False)


# ============================================================
# Metrics and evaluation
# ============================================================

def modularity_score(G: nx.Graph, feat: pd.DataFrame, labels: np.ndarray):
    if G.number_of_edges() == 0:
        return np.nan
    comm = defaultdict(set)
    for n, lab in zip(feat["company_node"], labels):
        if int(lab) == -1:
            continue
        if n in G:
            comm[int(lab)].add(n)
    communities = [v for v in comm.values() if v]
    if not communities:
        return np.nan
    try:
        return nx.algorithms.community.quality.modularity(G, communities, weight="weight")
    except Exception:
        return np.nan


def internal_edge_ratio(G: nx.Graph, feat: pd.DataFrame, labels: np.ndarray):
    lab = dict(zip(feat["company_node"], labels))
    internal = 0
    total = 0
    for u, v in G.edges():
        if u not in lab or v not in lab:
            continue
        if lab[u] == -1 or lab[v] == -1:
            continue
        total += 1
        if lab[u] == lab[v]:
            internal += 1
    return internal / total if total else np.nan


def label_balance_metrics(labels: np.ndarray, min_cluster_size: int):
    s = pd.Series(labels)
    non_noise = s[s != -1]
    if len(non_noise) == 0:
        return {"num_clusters": 0, "noise_ratio": 1.0, "largest_cluster_share": np.nan, "smallest_cluster_size": 0, "tiny_cluster_count": 0}
    counts = non_noise.value_counts()
    return {
        "num_clusters": int(len(counts)),
        "noise_ratio": float((s == -1).mean()),
        "largest_cluster_share": float(counts.max() / len(non_noise)),
        "smallest_cluster_size": int(counts.min()),
        "tiny_cluster_count": int((counts < min_cluster_size).sum()),
    }


def geometric_metrics(X_scaled: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels)
    mask = labels != -1
    uniq = np.unique(labels[mask])
    if mask.sum() < 5 or len(uniq) < 2:
        return {"silhouette_score": np.nan, "calinski_harabasz_score": np.nan, "davies_bouldin_score": np.nan}
    X = X_scaled[mask]
    y = labels[mask]
    out = {}
    try:
        out["silhouette_score"] = silhouette_score(X, y)
    except Exception:
        out["silhouette_score"] = np.nan
    try:
        out["calinski_harabasz_score"] = calinski_harabasz_score(X, y)
    except Exception:
        out["calinski_harabasz_score"] = np.nan
    try:
        out["davies_bouldin_score"] = davies_bouldin_score(X, y)
    except Exception:
        out["davies_bouldin_score"] = np.nan
    return out


def attach_clusters(events: pd.DataFrame, assign: pd.DataFrame, dates: pd.DataFrame):
    cmap = assign.set_index("company_node")["cluster_id"].to_dict()
    tmap = assign.set_index("company_node")["cluster_theme_label"].to_dict()
    out = events.copy()
    out["source_cluster_id"] = out["source_node"].map(cmap)
    out["target_cluster_id"] = out["target_node"].map(cmap)
    out["source_cluster_theme_label"] = out["source_node"].map(tmap)
    out["target_cluster_theme_label"] = out["target_node"].map(tmap)
    out["same_cluster"] = (
        out["source_cluster_id"].notna()
        & out["target_cluster_id"].notna()
        & out["source_cluster_id"].eq(out["target_cluster_id"])
        & out["source_cluster_id"].ne(-1)
    )
    if not dates.empty:
        d = dates.copy()
        d["release_date"] = pd.to_datetime(d["release_date"], errors="coerce")
        src = d.rename(columns={"company_node": "source_node", "quarter": "source_quarter", "release_date": "source_release_date"})
        tgt = d.rename(columns={"company_node": "target_node", "quarter": "target_quarter", "release_date": "target_release_date"})
        out = out.merge(src[["source_node", "source_quarter", "source_release_date"]], on=["source_node", "source_quarter"], how="left")
        out = out.merge(tgt[["target_node", "target_quarter", "target_release_date"]], on=["target_node", "target_quarter"], how="left")
        out["source_release_date"] = pd.to_datetime(out["source_release_date"], errors="coerce")
        out["target_release_date"] = pd.to_datetime(out["target_release_date"], errors="coerce")
        out["release_date_gap_days"] = (out["target_release_date"] - out["source_release_date"]).dt.days
        out["abs_release_date_gap_days"] = out["release_date_gap_days"].abs()
        out["source_before_target"] = out["release_date_gap_days"] > 0
    else:
        out["release_date_gap_days"] = np.nan
        out["abs_release_date_gap_days"] = np.nan
        out["source_before_target"] = np.nan
    return out


def summarize_cluster_events(events: pd.DataFrame, mode: str):
    e = events[
        events["analysis_mode"].eq(mode)
        & events["source_active"].astype(bool)
        & events["same_cluster"].astype(bool)
    ].copy()
    if e.empty:
        return pd.DataFrame()

    baseline = e.groupby(["target_quarter", "signal", "source_direction"]).apply(
        lambda g: ((g["target_active"].astype(bool)) & (g["target_direction"].astype(str).eq(g.name[2]))).mean()
    ).rename("baseline_rate").reset_index()

    cols = [
        "analysis_mode", "source_quarter", "target_quarter", "source_cluster_id",
        "source_cluster_theme_label", "signal", "source_label", "source_direction", "relation_group"
    ]
    rows = []
    for key, g in e.groupby(cols, dropna=False):
        d = dict(zip(cols, key))
        exposed = len(g)
        direction = int(g["direction_match"].sum())
        exact = int(g["exact_match"].sum())
        active = int(g["target_active"].sum())
        rate = direction / exposed if exposed else np.nan

        b = baseline[
            baseline["target_quarter"].eq(d["target_quarter"])
            & baseline["signal"].eq(d["signal"])
            & baseline["source_direction"].eq(d["source_direction"])
        ]
        br = float(b["baseline_rate"].iloc[0]) if not b.empty else np.nan
        lift = rate - br if not pd.isna(br) else np.nan

        d.update({
            "cluster_id": d.pop("source_cluster_id"),
            "cluster_theme_label": d.pop("source_cluster_theme_label"),
            "exposed_edges": exposed,
            "target_active_edges": active,
            "exact_match_edges": exact,
            "direction_match_edges": direction,
            "target_active_rate": active / exposed if exposed else np.nan,
            "exact_match_rate": exact / exposed if exposed else np.nan,
            "direction_match_rate": rate,
            "falsification_rate": 1 - rate if not pd.isna(rate) else np.nan,
            "baseline_rate": br,
            "prediction_lift": lift,
            "relative_lift": rate / br if br and br > 0 else np.nan,
            "effectiveness_score": lift * math.log1p(exposed) if not pd.isna(lift) else np.nan,
            "mean_release_date_gap_days": g["release_date_gap_days"].mean(),
            "median_release_date_gap_days": g["release_date_gap_days"].median(),
            "mean_abs_release_date_gap_days": g["abs_release_date_gap_days"].mean(),
            "median_abs_release_date_gap_days": g["abs_release_date_gap_days"].median(),
            "share_source_before_target": float(g["source_before_target"].mean()) if "source_before_target" in g.columns else np.nan,
            "date_observation_count": int(g["release_date_gap_days"].notna().sum()),
        })
        rows.append(d)

    return pd.DataFrame(rows).sort_values(
        ["effectiveness_score", "prediction_lift", "direction_match_rate", "exposed_edges"],
        ascending=[False, False, False, False]
    )


def propagation_scores(cross_summary: pd.DataFrame, same_summary: pd.DataFrame, min_exposed: int):
    def wavg(df, col):
        if df.empty or col not in df.columns:
            return np.nan
        d = df[(df["exposed_edges"] >= min_exposed) & df[col].notna()].copy()
        if d.empty:
            return np.nan
        return np.average(d[col], weights=d["exposed_edges"])

    return {
        "next_quarter_prediction_lift_weighted": wavg(cross_summary, "prediction_lift"),
        "next_quarter_direction_match_rate_weighted": wavg(cross_summary, "direction_match_rate"),
        "same_quarter_prediction_lift_weighted": wavg(same_summary, "prediction_lift"),
        "same_quarter_direction_match_rate_weighted": wavg(same_summary, "direction_match_rate"),
        "next_quarter_event_groups": int((cross_summary["exposed_edges"] >= min_exposed).sum()) if not cross_summary.empty else 0,
        "same_quarter_event_groups": int((same_summary["exposed_edges"] >= min_exposed).sum()) if not same_summary.empty else 0,
        "next_quarter_lead_lag_consistency": float(wavg(cross_summary, "share_source_before_target") or 0.0),
        "same_quarter_lead_lag_consistency": float(wavg(same_summary, "share_source_before_target") or 0.0),
    }


# ============================================================
# Clustering methods
# ============================================================

def run_graph_greedy(G: nx.Graph, feat: pd.DataFrame):
    communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    label_map = {}
    for i, comm in enumerate(communities):
        for n in comm:
            label_map[n] = i
    noise_id = len(communities)
    return np.array([label_map.get(n, noise_id) for n in feat["company_node"]], dtype=int)


def run_kmeans(X, k, seed):
    return KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(X)


def run_agglomerative(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def run_spectral(X, k, seed):
    n_neighbors = min(30, max(5, X.shape[0] // 100))
    return SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=seed,
        assign_labels="kmeans",
    ).fit_predict(X)


def run_optional_hdbscan(X, seed):
    try:
        import umap
        import hdbscan
    except Exception as e:
        print("Optional HDBSCAN/UMAP skipped:", e)
        return None
    X_umap = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.05, random_state=seed).fit_transform(X)
    return hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10).fit_predict(X_umap)


# ============================================================
# Plotting
# ============================================================

def plot_method_comparison(df: pd.DataFrame, out_png: Path):
    d = df.sort_values("propagation_score", ascending=False).head(30).copy()
    if d.empty:
        return
    d["label"] = d["method"].astype(str) + " k=" + d["k"].astype(str)
    ax = d.sort_values("propagation_score").plot(kind="barh", x="label", y="propagation_score", legend=False, figsize=(11, 8))
    ax.set_title("Clustering method comparison by propagation score")
    ax.set_xlabel("composite information-propagation score")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def plot_pca(assign: pd.DataFrame, X: np.ndarray, out_png: Path):
    if len(assign) < 3:
        return
    xy = PCA(n_components=2, random_state=42).fit_transform(X)
    d = assign.copy()
    d["pc1"] = xy[:, 0]
    d["pc2"] = xy[:, 1]
    largest = d[d["cluster_id"] != -1]["cluster_id"].value_counts().head(10).index.tolist()
    plt.figure(figsize=(12, 8))
    for cid, g in d.groupby("cluster_id"):
        plt.scatter(g["pc1"], g["pc2"], s=15, alpha=0.55, label=f"C{cid}" if cid in largest else None)
    for cid in largest:
        g = d[d["cluster_id"] == cid]
        if g.empty:
            continue
        label = str(g["cluster_theme_label"].iloc[0])
        plt.text(g["pc1"].mean(), g["pc2"].mean(), f"C{cid}\n{label}", fontsize=8, ha="center", bbox=dict(fc="white", alpha=0.70))
    plt.title("Best clustering PCA projection")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def plot_cooccur_degree_distribution(cooccur_graph: nx.Graph, out_png: Path):
    if cooccur_graph.number_of_nodes() == 0:
        return
    degrees = [d for _, d in cooccur_graph.degree()]
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Co-occurrence degree")
    plt.ylabel("Count")
    plt.title("Signal co-occurrence graph: degree distribution")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def bar_clean(df: pd.DataFrame, label_col: str, value_col: str, title: str, out_png: Path, top=30):
    if df.empty or value_col not in df.columns:
        return
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col]).copy()
    if d.empty:
        return
    d = d.sort_values(value_col, ascending=False).head(top)
    ax = d.sort_values(value_col).plot(kind="barh", x=label_col, y=value_col, legend=False, figsize=(12, 8))
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def clean_date_gap_plot_data(date_summary: pd.DataFrame, mode: str, min_exposed: int):
    d = date_summary[date_summary["mode"].eq(mode) & (date_summary["date_observation_count"] >= min_exposed)].copy()
    if d.empty:
        return d
    d["plot_label"] = (
        "C" + d["cluster_id"].astype(str)
        + " | " + d["cluster_theme_label"].astype(str)
        + " | " + d["signal"].astype(str)
        + " | " + d["relation_group"].astype(str)
    )
    return d


def make_date_summary(cross_e: pd.DataFrame, same_e: pd.DataFrame):
    frames = [cross_e.assign(mode="cross_quarter"), same_e.assign(mode="same_quarter")]
    all_dates = pd.concat(frames, ignore_index=True)
    all_dates = all_dates[all_dates["same_cluster"].astype(bool) & all_dates["source_active"].astype(bool)].copy()
    if all_dates.empty:
        return pd.DataFrame()
    out = all_dates.groupby(
        ["mode", "source_cluster_id", "source_cluster_theme_label", "signal", "relation_group"],
        dropna=False
    ).agg(
        event_count=("signal", "count"),
        mean_release_date_gap_days=("release_date_gap_days", "mean"),
        median_release_date_gap_days=("release_date_gap_days", "median"),
        mean_abs_release_date_gap_days=("abs_release_date_gap_days", "mean"),
        median_abs_release_date_gap_days=("abs_release_date_gap_days", "median"),
        share_source_before_target=("source_before_target", "mean"),
        date_observation_count=("release_date_gap_days", lambda x: int(x.notna().sum())),
    ).reset_index()
    return out.rename(columns={"source_cluster_id": "cluster_id", "source_cluster_theme_label": "cluster_theme_label"})


# ============================================================
# Evaluation loop
# ============================================================

def evaluate_candidate(method_name, k, labels, feat, X_scaled, G_rel, cooccur_graph, rel, cross, same, dates, min_exposed):
    assign = feat[["company_node", "ticker", "company"]].copy()
    assign["cluster_id"] = labels.astype(int)

    csum = cluster_summary(assign, feat, rel, G_rel)
    theme_map = csum.set_index("cluster_id")["cluster_theme_label"].to_dict()
    assign["cluster_theme_label"] = assign["cluster_id"].map(theme_map).fillna("Mixed / Other")

    cross_e = attach_clusters(cross, assign, dates)
    same_e = attach_clusters(same, assign, dates)
    cross_s = summarize_cluster_events(cross_e, "cross_quarter")
    same_s = summarize_cluster_events(same_e, "same_quarter")

    bal = label_balance_metrics(labels, min_exposed)
    geo = geometric_metrics(X_scaled, labels)
    mod_rel = modularity_score(G_rel, feat, labels)
    mod_cooccur = modularity_score(cooccur_graph, feat, labels)
    ier = internal_edge_ratio(G_rel, feat, labels)
    prop = propagation_scores(cross_s, same_s, min_exposed)

    lift1 = prop.get("next_quarter_prediction_lift_weighted") or 0.0
    lift2 = prop.get("same_quarter_prediction_lift_weighted") or 0.0
    if pd.isna(lift1): lift1 = 0.0
    if pd.isna(lift2): lift2 = 0.0

    mod_rel_score = 0.0 if pd.isna(mod_rel) else mod_rel
    mod_cooccur_score = 0.0 if pd.isna(mod_cooccur) else mod_cooccur
    ier_score = 0.0 if pd.isna(ier) else ier
    lead_lag_next = prop.get("next_quarter_lead_lag_consistency", 0.0) or 0.0
    lead_lag_same = prop.get("same_quarter_lead_lag_consistency", 0.0) or 0.0
    if pd.isna(lead_lag_next): lead_lag_next = 0.0
    if pd.isna(lead_lag_same): lead_lag_same = 0.0

    propagation_score = (
        2.0 * lift1
        + 1.0 * lift2
        + 0.5 * ier_score
        + 0.5 * mod_rel_score
        + 1.0 * mod_cooccur_score
        + 0.3 * lead_lag_next
        + 0.2 * lead_lag_same
        - 0.25 * bal["largest_cluster_share"]
        - 0.03 * bal["tiny_cluster_count"]
        - 0.20 * bal["noise_ratio"]
    )

    row = {
        "method": method_name,
        "k": int(k) if k is not None else int(bal["num_clusters"]),
        "propagation_score": propagation_score,
        "research_score": propagation_score,  # backward-compatible alias
        "modularity_rel": mod_rel,
        "modularity_cooccur": mod_cooccur,
        "internal_edge_ratio": ier,
        **bal,
        **geo,
        **prop,
    }

    return row, assign, csum, cross_e, same_e, cross_s, same_s


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")

    print("=" * 100)
    print("V4 HF clustering method comparison")
    print("HF two-part dataset:", args.hf_two_part_dataset)
    print("HF LLM dataset:", args.hf_llm_dataset)
    print("HF evidence dataset:", args.hf_evidence_dataset)
    print("out_dir:", out_dir)
    print("=" * 100)

    # Save visible HF parquet file manifest for audit.
    manifest_rows = []
    for role, repo_id, revision in [
        ("two_part", args.hf_two_part_dataset, args.hf_two_part_revision),
        ("llm_concepts", args.hf_llm_dataset, args.hf_llm_revision),
        ("evidence", args.hf_evidence_dataset, args.hf_evidence_revision),
    ]:
        try:
            for f in list_hf_parquet_files(repo_id, revision):
                manifest_rows.append({"role": role, "hf_dataset": repo_id, "revision": revision, "path": f})
        except Exception as e:
            manifest_rows.append({"role": role, "hf_dataset": repo_id, "revision": revision, "path": f"LIST_FAILED: {e}"})
    save_table(pd.DataFrame(manifest_rows), out_dir / "hf_input_file_manifest.parquet", args.write_csv_copy)

    # ---- Load HF data ----
    outlook = read_hf_parquet(
        args.hf_two_part_dataset,
        args.hf_outlook_file,
        args.hf_two_part_prefix,
        args.hf_two_part_revision,
        "cleaned_outlook",
        required_columns={"company_node", "ticker", "current_company", "quarter", "signal", "score"},
    )
    rel = read_hf_parquet(
        args.hf_two_part_dataset,
        args.hf_relationships_file,
        args.hf_two_part_prefix,
        args.hf_two_part_revision,
        "matched_relationships",
        required_columns={"source_company_node", "target_company_node"},
    )
    cross = read_hf_parquet(
        args.hf_two_part_dataset,
        args.hf_cross_events_file,
        args.hf_two_part_prefix,
        args.hf_two_part_revision,
        "cross_events",
        required_columns={"analysis_mode", "source_node", "target_node", "source_quarter", "target_quarter", "signal"},
    )
    same = read_hf_parquet(
        args.hf_two_part_dataset,
        args.hf_same_events_file,
        args.hf_two_part_prefix,
        args.hf_two_part_revision,
        "same_events",
        required_columns={"analysis_mode", "source_node", "target_node", "source_quarter", "target_quarter", "signal"},
    )
    concepts = read_hf_concepts(args)
    evidence_meta = read_hf_evidence_metadata(args)

    # ---- Filter quarters ----
    outlook = filter_quarter_range(outlook, ["quarter"], args.start_quarter, args.end_quarter)
    rel = filter_quarter_range(rel, ["quarter"], args.start_quarter, args.end_quarter)
    cross = filter_quarter_range(cross, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)
    same = filter_quarter_range(same, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)

    if not concepts.empty and "quarter" in concepts.columns:
        concepts = filter_quarter_range(concepts, ["quarter"], args.start_quarter, args.end_quarter)

    dates = load_dates_from_metadata(evidence_meta)
    dates = filter_quarter_range(dates, ["quarter"], args.start_quarter, args.end_quarter)
    save_table(dates, out_dir / "company_quarter_release_dates.parquet", args.write_csv_copy)

    # ---- Build graphs and features ----
    cooccur_graph = build_signal_cooccurrence_graph(cross, same, min_weight=args.cooccur_min_weight)
    plot_cooccur_degree_distribution(cooccur_graph, fig_dir / "cooccur_degree_distribution.png")

    feat, G_rel = build_base_features(outlook, rel, concepts, cross, same, cooccur_graph)
    feat = add_timing_features(feat, dates)
    feat = add_propagation_history_features(feat, cross)
    feat = add_v4_propagation_features(feat, cross, same, G_rel)

    feature_cols = [
        c for c in feat.columns
        if c not in {"company_node", "ticker", "company"}
        and pd.api.types.is_numeric_dtype(feat[c])
    ]

    save_table(feat, out_dir / "company_feature_matrix_v4_raw.parquet", args.write_csv_copy)
    print(f"Feature matrix: {len(feat):,} companies, {len(feature_cols):,} numeric features")

    v4_features = [c for c in feature_cols if c.startswith(("cooccur_", "lead_ratio_", "lead_lag_", "signal_homophily_", "neighbor_"))]
    print(f"V4 propagation features: {len(v4_features)}")

    X_scaled, used_features, X_processed = preprocess_features(feat, feature_cols, args.winsor_quantile)
    processed = pd.DataFrame(X_processed, columns=used_features)
    processed.insert(0, "company_node", feat["company_node"].values)
    save_table(processed, out_dir / "company_feature_matrix_v4_processed.parquet", args.write_csv_copy)

    # ---- Candidate clustering methods ----
    candidates = []

    try:
        candidates.append(("relationship_graph_greedy_modularity", None, run_graph_greedy(G_rel, feat)))
        print("relationship_graph_greedy_modularity: OK")
    except Exception as e:
        print("relationship_graph_greedy_modularity failed:", e)

    try:
        if cooccur_graph.number_of_edges() > 0:
            candidates.append(("signal_cooccurrence_graph_greedy_modularity", None, run_graph_greedy(cooccur_graph, feat)))
            print("signal_cooccurrence_graph_greedy_modularity: OK")
    except Exception as e:
        print("signal_cooccurrence_graph_greedy_modularity failed:", e)

    for k in range(args.min_k, args.max_k + 1):
        try:
            candidates.append(("kmeans", k, run_kmeans(X_scaled, k, args.random_state)))
        except Exception as e:
            print(f"kmeans k={k} failed:", e)

        try:
            candidates.append(("agglomerative_ward", k, run_agglomerative(X_scaled, k)))
        except Exception as e:
            print(f"agglomerative k={k} failed:", e)

        if not args.skip_spectral:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates.append(("spectral", k, run_spectral(X_scaled, k, args.random_state)))
            except Exception as e:
                print(f"spectral k={k} failed:", e)

    if args.run_optional_hdbscan:
        labels = run_optional_hdbscan(X_scaled, args.random_state)
        if labels is not None:
            candidates.append(("hdbscan_umap_optional", None, labels))

    print(f"Total candidates to evaluate: {len(candidates)}")

    # ---- Evaluate candidates ----
    rows = []
    cache = {}

    for method, k, labels in candidates:
        print(f"Evaluating: {method}, k={k}")
        row, assign, csum, cross_e, same_e, cross_s, same_s = evaluate_candidate(
            method, k, labels, feat, X_scaled, G_rel, cooccur_graph, rel, cross, same, dates, args.min_exposed
        )
        rows.append(row)
        cache[(method, row["k"])] = (labels, assign, csum, cross_e, same_e, cross_s, same_s)

    comparison = pd.DataFrame(rows).sort_values("propagation_score", ascending=False)
    save_table(comparison, out_dir / "method_comparison_summary.parquet", args.write_csv_copy)
    plot_method_comparison(comparison, fig_dir / "method_comparison_research_score.png")

    if comparison.empty:
        raise RuntimeError("No clustering candidates succeeded.")

    # ---- Save best method outputs ----
    best = comparison.iloc[0]
    best_key = (best["method"], int(best["k"]))
    labels, assign, csum, cross_e, same_e, cross_s, same_s = cache[best_key]

    save_table(assign, out_dir / "best_company_cluster_assignment.parquet", args.write_csv_copy)
    save_table(csum, out_dir / "best_cluster_summary.parquet", args.write_csv_copy)
    save_table(cross_e, out_dir / "best_cross_quarter_events_with_cluster_dates.parquet", args.write_csv_copy)
    save_table(same_e, out_dir / "best_same_quarter_events_with_cluster_dates.parquet", args.write_csv_copy)
    save_table(cross_s, out_dir / "best_next_quarter_cluster_contagion.parquet", args.write_csv_copy)
    save_table(same_s, out_dir / "best_same_quarter_cluster_correlation.parquet", args.write_csv_copy)

    date_summary = make_date_summary(cross_e, same_e)
    save_table(date_summary, out_dir / "best_release_date_gap_by_cluster_clean.parquet", args.write_csv_copy)

    plot_pca(assign, X_scaled, fig_dir / "best_cluster_pca.png")

    same_date_clean = clean_date_gap_plot_data(date_summary, "same_quarter", args.min_exposed)
    cross_date_clean = clean_date_gap_plot_data(date_summary, "cross_quarter", args.min_exposed)
    save_table(same_date_clean, out_dir / "best_same_quarter_date_gap_plot_data.parquet", args.write_csv_copy)
    save_table(cross_date_clean, out_dir / "best_cross_quarter_date_gap_plot_data.parquet", args.write_csv_copy)

    bar_clean(same_date_clean, "plot_label", "mean_abs_release_date_gap_days",
              "Same-quarter date gap by cluster / signal / relation",
              fig_dir / "best_same_quarter_date_gap_clean.png", top=30)
    bar_clean(same_date_clean, "plot_label", "share_source_before_target",
              "Same-quarter source-before-target ratio",
              fig_dir / "best_source_before_target_ratio.png", top=30)

    lift = cross_s[cross_s["exposed_edges"] >= args.min_exposed].copy()
    if not lift.empty:
        lift["plot_label"] = (
            "C" + lift["cluster_id"].astype(str)
            + " | " + lift["cluster_theme_label"].astype(str)
            + " | " + lift["signal"].astype(str)
            + " | " + lift["relation_group"].astype(str)
        )
    bar_clean(lift, "plot_label", "prediction_lift",
              "Best method: next-quarter prediction lift",
              fig_dir / "best_next_quarter_prediction_lift.png", top=30)

    # Feature diagnostic
    try:
        fi = pd.DataFrame({
            "feature": used_features,
            "mean_abs_processed_value": np.abs(processed[used_features].values).mean(axis=0),
        }).sort_values("mean_abs_processed_value", ascending=False)
        save_table(fi, out_dir / "feature_diagnostic_mean_abs_processed_value.parquet", args.write_csv_copy)
    except Exception as e:
        print("Feature diagnostic failed:", e)

    # ---- Markdown report ----
    report = []
    report.append("# V4 HF Clustering Method Comparison")
    report.append("")
    report.append("## Data sources")
    report.append("")
    report.append(f"- Two-part HF dataset: `{args.hf_two_part_dataset}`")
    report.append(f"- LLM concept HF dataset: `{args.hf_llm_dataset}`")
    report.append(f"- Evidence metadata HF dataset: `{args.hf_evidence_dataset}`")
    report.append("")
    report.append("## Data scale")
    report.append("")
    report.append(f"- Companies: {len(feat):,}")
    report.append(f"- Raw numeric features: {len(feature_cols):,}")
    report.append(f"- Used features after preprocessing: {len(used_features):,}")
    report.append(f"- V4 propagation features: {len(v4_features):,}")
    report.append(f"- Signal co-occurrence graph nodes: {cooccur_graph.number_of_nodes():,}")
    report.append(f"- Signal co-occurrence graph edges: {cooccur_graph.number_of_edges():,}")
    report.append(f"- Matched relationship rows: {len(rel):,}")
    report.append(f"- Cross-quarter events: {len(cross):,}")
    report.append(f"- Same-quarter events: {len(same):,}")
    report.append(f"- Release-date company-quarter rows: {len(dates):,}")
    report.append("")
    report.append("## Best method")
    report.append("")
    report.append(best.to_frame().T.to_markdown(index=False))
    report.append("")
    report.append("## Method comparison")
    report.append("")
    report.append(comparison.head(30).to_markdown(index=False))
    report.append("")
    report.append("## Best cluster summary")
    report.append("")
    report.append(csum.head(30).to_markdown(index=False))
    report.append("")
    report.append("## Top next-quarter prediction lift")
    report.append("")
    report.append(lift.sort_values("prediction_lift", ascending=False).head(30).to_markdown(index=False) if not lift.empty else "No results.")
    report.append("")

    report_path = out_dir / "v4_hf_clustering_method_comparison_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"SAVED {report_path}")

    print("=" * 100)
    print("DONE")
    print(f"Best method: {best['method']}  k={best['k']}  propagation_score={best['propagation_score']:.4f}")
    print(f"Report: {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
