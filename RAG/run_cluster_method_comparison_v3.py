#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3: Clustering Method Comparison for Network Diffusion Analysis

Purpose
-------
This script replaces the previous single KMeans clustering design.

It compares several clustering methods:

1. graph_greedy_modularity
   - graph/community based
   - uses company relationship network directly
   - good for detecting network communities

2. kmeans
   - feature based baseline
   - tested across k range

3. agglomerative_ward
   - feature based hierarchical clustering
   - tested across k range
   - often more stable than KMeans when clusters are not spherical

4. spectral
   - feature based nonlinear clustering
   - tested across k range
   - useful when clusters are not linearly separable

5. hdbscan_umap_optional
   - optional, only runs if umap-learn and hdbscan are installed
   - can identify noise / outliers instead of forcing every company into a cluster

Selection principle
-------------------
The best method is NOT selected only by silhouette.
It is selected by a combined research score:

    research_score =
        prediction_lift_score
        + same_quarter_similarity_score
        + internal_edge_ratio_score
        + modularity_score
        - balance_penalties

This is because the research question is information propagation, not pure geometric clustering.

Main outputs
------------
method_comparison_summary.csv
best_company_cluster_assignment.csv
best_cluster_summary.csv
best_next_quarter_cluster_contagion.csv
best_same_quarter_cluster_correlation.csv
best_release_date_gap_by_cluster_clean.csv
figures/method_comparison_research_score.png
figures/best_cluster_pca.png
figures/best_same_quarter_date_gap_clean.png
figures/best_source_before_target_ratio.png
figures/best_next_quarter_prediction_lift.png

Run
---
cd ~/sem2/RAG

python run_cluster_method_comparison_v3.py \
  --rag-output-dir rag_chroma_output \
  --two-part-dir rag_chroma_output/two_part_network_prediction_analysis \
  --combined-transcripts ../data/combined_transcript_data/combined_transcripts_deduplicated.csv \
  --evidence-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl \
  --out-dir rag_chroma_output/cluster_method_comparison_v3 \
  --start-quarter 2019Q2 \
  --end-quarter 2026Q2 \
  --min-k 5 \
  --max-k 20 \
  --min-cluster-size 20 \
  --min-exposed 10

Windows CMD
-----------
python run_cluster_method_comparison_v3.py ^
  --rag-output-dir rag_chroma_output ^
  --two-part-dir rag_chroma_output\two_part_network_prediction_analysis ^
  --combined-transcripts ..\data\combined_transcript_data\combined_transcripts_deduplicated.csv ^
  --evidence-jsonl rag_chroma_output\rag_evidence_packages_full_gpu_direct.jsonl ^
  --out-dir rag_chroma_output\cluster_method_comparison_v3 ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2 ^
  --min-k 5 ^
  --max-k 20 ^
  --min-cluster-size 20 ^
  --min-exposed 10
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


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
        "JPM", "MA", "BANCORP", "BANK", "FINANCIAL", "CAPITAL", "CREDIT",
        "INSURANCE", "MORGAN", "GOLDMAN", "WELLS", "CITI", "VISA", "MASTERCARD"
    ],
    "Cloud / AI / Digital Platforms": [
        "AMZN", "MSFT", "GOOG", "GOOGL", "META", "ORCL", "IBM", "SHOP", "CRM",
        "ADBE", "NOW", "SNOW", "OPRA", "YOU", "XPER", "CLOUD", "SOFTWARE",
        "AI", "DATA CENTER", "DATACENTER"
    ],
    "Semiconductors / Hardware": [
        "NVDA", "INTC", "AMD", "MU", "TSM", "ASML", "AVGO", "QCOM", "SIMO",
        "AAPL", "DELL", "ANET", "IONQ", "VOXX", "ADEA", "GFS", "GLW", "CHIP",
        "SEMICONDUCTOR", "HARDWARE", "WAFER", "FOUNDRY"
    ],
    "Retail / Consumer / Restaurants": [
        "TGT", "WMT", "HD", "CVS", "YUMC", "COST", "LOW", "MCD", "SBUX",
        "RETAIL", "CONSUMER", "STORE", "RESTAURANT", "BURL", "NKE"
    ],
    "Automotive / Mobility": [
        "GM", "F", "HMC", "TSLA", "ADNT", "FORD", "MOTORS", "AUTOMOTIVE", "VEHICLE"
    ],
    "Energy / Utilities": [
        "BP", "EPD", "AEP", "AQN", "GEL", "LBRT", "XOM", "CVX", "SHEL",
        "ENERGY", "OIL", "GAS", "POWER", "UTILITY", "UTILITIES", "HAL", "EOG"
    ],
    "Healthcare / Pharma": [
        "HEALTH", "SNY", "IART", "BTMD", "PFE", "LLY", "UNH", "MRK", "BMY",
        "MEDICAL", "PHARMA", "BIOTECH", "THERAPEUTICS", "HOSPITAL"
    ],
    "Industrial / Manufacturing / Aerospace": [
        "CAT", "AERO", "ROP", "CARR", "PKOH", "STRL", "EML", "INDUSTRIAL",
        "MANUFACTURING", "AEROSPACE", "MACHINERY", "CONSTRUCTION", "GEV", "WRK"
    ],
    "Airlines / Travel / Leisure": [
        "UAL", "SOUTHWEST", "DAL", "AAL", "TRAVEL", "AIRLINES", "HOTEL", "LEISURE", "SGHC", "MAR"
    ],
    "Telecom / Media / Cable": [
        "CHTR", "CMCSA", "VZ", "T", "TMUS", "TELECOM", "CABLE", "MEDIA", "COMMUNICATIONS"
    ],
}

CONCEPT_RULES = {
    "Semiconductors / Hardware": (["semiconductor_supply", "chip_supply", "supplier_constraint"], 0.18, 0.35),
    "Cloud / AI / Digital Platforms": (["cloud_infrastructure", "data_center_capacity"], 0.12, 0.25),
    "Retail / Consumer / Restaurants": (["customer_demand", "inventory_pressure", "pricing_pressure", "logistics_shipping"], 0.45, 1.10),
    "Energy / Utilities": (["oil_energy_supply"], 0.15, 0.15),
    "Industrial / Manufacturing / Aerospace": (["manufacturing_capacity", "production_capacity", "capex_expansion"], 0.65, 1.30),
}


# ============================================================
# Generic utilities
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag-output-dir", default="rag_chroma_output")
    p.add_argument("--two-part-dir", default="rag_chroma_output/two_part_network_prediction_analysis")
    p.add_argument("--combined-transcripts", default="../data/combined_transcript_data/combined_transcripts_deduplicated.csv")
    p.add_argument("--evidence-jsonl", default="rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl")
    p.add_argument("--out-dir", default="rag_chroma_output/cluster_method_comparison_v3")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--min-k", type=int, default=5)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--min-cluster-size", type=int, default=20)
    p.add_argument("--min-exposed", type=int, default=10)
    p.add_argument("--winsor-quantile", type=float, default=0.99)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--run-optional-hdbscan", action="store_true")
    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"SAVED {path} rows={len(df):,}")


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
    return re.sub(r"\s+", " ", s).strip()


def company_node_from(ticker: str, company: str) -> str:
    ticker = clean_node_value(ticker)
    company = clean_node_value(company)
    if ticker:
        return "COMPANY::" + ticker
    return "COMPANY::" + norm_text(company)


# ============================================================
# Release date loading
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


def load_dates_from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    dc = find_date_column(df)
    if not dc or "quarter" not in df.columns:
        return pd.DataFrame()
    ticker_col = "ticker" if "ticker" in df.columns else ""
    company_col = next((c for c in ["current_company", "company", "company_name", "name"] if c in df.columns), "")
    if not ticker_col and not company_col:
        return pd.DataFrame()
    df["release_date"] = pd.to_datetime(df[dc], errors="coerce")
    df = df[df["release_date"].notna()].copy()
    df["company_node"] = df.apply(lambda r: company_node_from(r[ticker_col] if ticker_col else "", r[company_col] if company_col else ""), axis=1)
    df["quarter"] = df["quarter"].astype(str).str.strip()
    return df.groupby(["company_node", "quarter"], as_index=False).agg(
        release_date=("release_date", "min"),
        release_date_count=("release_date", "count")
    )


def load_dates_from_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    keys = ["release_date", "earnings_call_date", "call_date", "date", "published_date", "publish_date", "publication_date", "transcript_date"]
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            d = next((obj.get(k) for k in keys if obj.get(k)), None)
            if not d:
                continue
            q = str(obj.get("quarter", "")).strip()
            if not q:
                continue
            rows.append({
                "company_node": company_node_from(obj.get("ticker", ""), obj.get("current_company", obj.get("company", ""))),
                "quarter": q,
                "release_date": d,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df[df["release_date"].notna()].copy()
    return df.groupby(["company_node", "quarter"], as_index=False).agg(
        release_date=("release_date", "min"),
        release_date_count=("release_date", "count")
    )


def combine_dates(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a.empty and b.empty:
        return pd.DataFrame(columns=["company_node", "quarter", "release_date", "release_date_count"])
    df = pd.concat([x for x in [a, b] if not x.empty], ignore_index=True)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df[df["release_date"].notna()].copy()
    return df.groupby(["company_node", "quarter"], as_index=False).agg(
        release_date=("release_date", "min"),
        release_date_count=("release_date_count", "sum")
    )


# ============================================================
# Input features
# ============================================================

def discover_concepts(rag_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(rag_dir.rglob("concepts_*.csv")):
        try:
            if f.stat().st_size:
                d = pd.read_csv(f)
                d["source_file"] = str(f)
                frames.append(d)
        except Exception as e:
            print("WARNING concept read failed", f, e)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def build_graph(rel: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    if rel.empty:
        return G
    for _, r in rel.iterrows():
        s = clean_node_value(r.get("source_company_node", ""))
        t = clean_node_value(r.get("target_company_node", ""))
        if not s or not t or s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += 1
        else:
            G.add_edge(s, t, weight=1)
    return G


def company_metadata(outlook: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not outlook.empty and {"company_node", "ticker", "current_company"}.issubset(outlook.columns):
        frames.append(outlook[["company_node", "ticker", "current_company"]].rename(columns={"current_company": "company"}))

    if "source_company_node" in rel.columns:
        cols = ["source_company_node"]
        if "source_ticker" in rel.columns:
            cols.append("source_ticker")
        if "source_company" in rel.columns:
            cols.append("source_company")
        x = rel[cols].rename(columns={"source_company_node": "company_node", "source_ticker": "ticker", "source_company": "company"})
        frames.append(x)

    if "target_company_node" in rel.columns:
        cols = ["target_company_node"]
        if "target_ticker" in rel.columns:
            cols.append("target_ticker")
        if "target_company" in rel.columns:
            cols.append("target_company")
        x = rel[cols].rename(columns={"target_company_node": "company_node", "target_ticker": "ticker", "target_company": "company"})
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


def build_base_features(outlook: pd.DataFrame, rel: pd.DataFrame, concepts: pd.DataFrame):
    outlook = outlook.copy()
    rel = rel.copy()

    outlook["company_node"] = outlook["company_node"].map(clean_node_value)
    outlook = outlook[outlook["company_node"].ne("")].copy()

    for c in ["source_company_node", "target_company_node"]:
        rel[c] = rel[c].map(clean_node_value)
    rel = rel[rel["source_company_node"].ne("") & rel["target_company_node"].ne("") & rel["source_company_node"].ne(rel["target_company_node"])].copy()

    G = build_graph(rel)
    nodes = sorted(set(G.nodes()) | set(outlook["company_node"].unique()))
    feat = pd.DataFrame({"company_node": nodes})

    meta = company_metadata(outlook, rel)
    feat = feat.merge(meta, on="company_node", how="left")

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() else {}

    # Approximate betweenness for speed.
    if G.number_of_nodes() > 1000:
        betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), seed=42, weight="weight")
    elif G.number_of_nodes() > 0:
        betweenness = nx.betweenness_centrality(G, weight="weight")
    else:
        betweenness = {}

    feat["degree"] = feat["company_node"].map(degree).fillna(0)
    feat["weighted_degree"] = feat["company_node"].map(weighted_degree).fillna(0)
    feat["pagerank"] = feat["company_node"].map(pagerank).fillna(0)
    feat["betweenness"] = feat["company_node"].map(betweenness).fillna(0)

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

    # Outlook behavior features
    outlook["score"] = pd.to_numeric(outlook["score"], errors="coerce")
    score = outlook.pivot_table(index="company_node", columns="signal", values="score", aggfunc="mean", fill_value=0).reset_index()
    score.columns = ["company_node" if c == "company_node" else f"mean_{c}_score" for c in score.columns]

    tmp = outlook.assign(
        active=outlook["score"].abs() > 0,
        pos=outlook["score"] > 0,
        neg=outlook["score"] < 0,
    )
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
        g = g.copy()
        pivot = g.pivot_table(index="quarter", columns="signal", values="score", aggfunc="mean").sort_index()
        vals = pivot.values.flatten()
        vals = vals[~pd.isna(vals)]
        if len(vals) == 0:
            continue
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
    dyn = pd.DataFrame(dyn_rows)
    if not dyn.empty:
        feat = feat.merge(dyn, on="company_node", how="left")

    # Concept features
    if not concepts.empty:
        concepts = concepts.copy()
        ticker_col = "ticker" if "ticker" in concepts.columns else ""
        company_col = next((c for c in ["current_company", "company", "company_name"] if c in concepts.columns), "")
        if "company_node" not in concepts.columns:
            concepts["company_node"] = concepts.apply(lambda r: company_node_from(r[ticker_col] if ticker_col else "", r[company_col] if company_col else ""), axis=1)
        cols = [c for c in CONCEPT_COLUMNS if c in concepts.columns]
        for c in cols:
            concepts[c] = pd.to_numeric(concepts[c], errors="coerce").fillna(0)
        if cols:
            con = concepts.groupby("company_node")[cols].mean().reset_index()
            con.columns = ["company_node" if c == "company_node" else f"concept_rate_{c}" for c in con.columns]

            # Add concept entropy and intensity composites.
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

    feat = feat.fillna(0)
    feat["ticker"] = feat.get("ticker", "").replace(0, "").astype(str)
    feat["company"] = feat.get("company", "").replace(0, "").astype(str)

    feature_cols = [
        c for c in feat.columns
        if c not in {"company_node", "ticker", "company"}
        and pd.api.types.is_numeric_dtype(feat[c])
    ]
    return feat, feature_cols, G


def add_timing_features(feat: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    if dates.empty:
        return feat

    d = dates.copy()
    d["release_date"] = pd.to_datetime(d["release_date"], errors="coerce")
    d = d[d["release_date"].notna()].copy()
    if d.empty:
        return feat

    # Approximate day in quarter by sorting inside company-quarter.
    d["year"] = d["release_date"].dt.year
    d["month"] = d["release_date"].dt.month
    d["day_of_year"] = d["release_date"].dt.dayofyear

    agg = d.groupby("company_node", as_index=False).agg(
        avg_release_day_of_year=("day_of_year", "mean"),
        std_release_day_of_year=("day_of_year", "std"),
        release_date_obs=("release_date", "count"),
    )
    agg = agg.fillna(0)
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


def preprocess_features(feat: pd.DataFrame, feature_cols: list[str], winsor_q: float):
    X = feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()

    # Remove near-zero variance features.
    keep = []
    for c in X.columns:
        if X[c].std() > 1e-12:
            keep.append(c)
    X = X[keep]

    Xp = X.copy()
    for c in Xp.columns:
        s = pd.to_numeric(Xp[c], errors="coerce").fillna(0)
        if s.min() >= 0:
            s = np.log1p(s)
        hi = s.quantile(winsor_q)
        lo = s.quantile(1 - winsor_q) if s.min() < 0 else s.min()
        s = s.clip(lo, hi)
        Xp[c] = s

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xp.values)
    return X_scaled, list(Xp.columns), Xp


# ============================================================
# Labeling
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
    scores = Counter()
    evidence = defaultdict(list)

    for label, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            kw_u = kw.upper()
            hit = kw_u in tokens if len(kw_u) <= 5 else kw_u in text_u
            if hit:
                scores[label] += 3.0 if len(kw_u) <= 5 else 2.0
                evidence[label].append(kw)
    return scores, evidence


def score_concepts(concepts: dict):
    scores = Counter()
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

    scores = Counter()
    scores.update(company_scores)
    for k, v in concept_scores.items():
        scores[k] += 0.65 * v

    if num_companies <= 2 and company_scores:
        label = company_scores.most_common(1)[0][0]
    elif scores:
        label = scores.most_common(1)[0][0]
    else:
        label = "Mixed / Other"

    diag = []
    for lab, sc in scores.most_common():
        ev = []
        ev.extend(company_ev.get(lab, []))
        ev.extend(concept_ev.get(lab, []))
        diag.append(f"{lab}:{sc:.2f} [{'|'.join(ev[:8])}]")
    return label, "; ".join(diag)


def cluster_summary(assign: pd.DataFrame, feat: pd.DataFrame, rel: pd.DataFrame, G: nx.Graph):
    concept_cols = [c for c in feat.columns if c.startswith("concept_rate_")]
    rows = []

    for cid, g in assign.groupby("cluster_id", dropna=False):
        if int(cid) == -1:
            label = "Noise / Outliers"
            diag = "HDBSCAN noise or unassigned nodes"
        nodes = set(g["company_node"])
        sub = G.subgraph(nodes)

        central = sorted([(n, G.degree(n) if n in G else 0) for n in nodes], key=lambda x: x[1], reverse=True)[:15]
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

        if int(cid) != -1:
            label, diag = infer_label("; ".join(top_companies), top_concepts, len(nodes))

        external_edges = 0
        for n in nodes:
            if n in G:
                external_edges += sum(1 for nb in G.neighbors(n) if nb not in nodes)

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
        })

    return pd.DataFrame(rows).sort_values("num_companies", ascending=False)


# ============================================================
# Metrics and evaluation
# ============================================================

def modularity_score(G: nx.Graph, feat: pd.DataFrame, labels: np.ndarray):
    comm = defaultdict(set)
    for n, lab in zip(feat["company_node"], labels):
        if int(lab) == -1:
            continue
        if n in G:
            comm[int(lab)].add(n)
    communities = [v for v in comm.values() if v]
    if not communities or G.number_of_edges() == 0:
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
        return {
            "num_clusters": 0,
            "noise_ratio": 1.0,
            "largest_cluster_share": np.nan,
            "smallest_cluster_size": 0,
            "tiny_cluster_count": 0,
        }
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
            "share_source_before_target": g["source_before_target"].mean(),
            "date_observation_count": int(g["release_date_gap_days"].notna().sum()),
        })
        rows.append(d)

    return pd.DataFrame(rows).sort_values(["effectiveness_score", "prediction_lift", "direction_match_rate", "exposed_edges"], ascending=[False, False, False, False])


def propagation_scores(cross_summary: pd.DataFrame, same_summary: pd.DataFrame, min_exposed: int):
    def wavg(df, col):
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

    labels = []
    noise_id = len(communities)
    for n in feat["company_node"]:
        labels.append(label_map.get(n, noise_id))
    return np.array(labels, dtype=int)


def run_kmeans(X, k, seed):
    return KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(X)


def run_agglomerative(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def run_spectral(X, k, seed):
    # Use nearest_neighbors affinity for nonlinear structure.
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

    reducer = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.05, random_state=seed)
    X_umap = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
    labels = clusterer.fit_predict(X_umap)
    return labels


# ============================================================
# Plotting
# ============================================================

def plot_method_comparison(df: pd.DataFrame, out_png: Path):
    d = df.sort_values("research_score", ascending=False).head(30).copy()
    if d.empty:
        return
    d["label"] = d["method"].astype(str) + " k=" + d["k"].astype(str)
    ax = d.sort_values("research_score").plot(kind="barh", x="label", y="research_score", legend=False, figsize=(11, 8))
    ax.set_title("Clustering method comparison by research score")
    ax.set_xlabel("research_score")
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

    # Only label largest clusters to avoid overlapping text.
    largest = d[d["cluster_id"] != -1]["cluster_id"].value_counts().head(10).index.tolist()

    plt.figure(figsize=(12, 8))
    for cid, g in d.groupby("cluster_id"):
        plt.scatter(g["pc1"], g["pc2"], s=15, alpha=0.55, label=f"C{cid}" if cid in largest else None)

    for cid in largest:
        g = d[d["cluster_id"] == cid]
        if g.empty:
            continue
        label = str(g["cluster_theme_label"].iloc[0])
        plt.text(
            g["pc1"].mean(), g["pc2"].mean(),
            f"C{cid}\n{label}",
            fontsize=8, ha="center",
            bbox=dict(fc="white", alpha=0.70)
        )

    plt.title("Best clustering PCA projection")
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


# ============================================================
# Main comparison loop
# ============================================================

def evaluate_candidate(method_name, k, labels, feat, X_scaled, G, rel, cross, same, dates, out_dir, min_exposed, save_candidate=False):
    assign = feat[["company_node", "ticker", "company"]].copy()
    assign["cluster_id"] = labels.astype(int)

    csum = cluster_summary(assign, feat, rel, G)
    theme_map = csum.set_index("cluster_id")["cluster_theme_label"].to_dict()
    assign["cluster_theme_label"] = assign["cluster_id"].map(theme_map).fillna("Mixed / Other")

    cross_e = attach_clusters(cross, assign, dates)
    same_e = attach_clusters(same, assign, dates)
    cross_s = summarize_cluster_events(cross_e, "cross_quarter")
    same_s = summarize_cluster_events(same_e, "same_quarter")

    bal = label_balance_metrics(labels, min_exposed)
    geo = geometric_metrics(X_scaled, labels)
    mod = modularity_score(G, feat, labels)
    ier = internal_edge_ratio(G, feat, labels)
    prop = propagation_scores(cross_s, same_s, min_exposed)

    # Combined research score.
    lift1 = prop["next_quarter_prediction_lift_weighted"]
    lift2 = prop["same_quarter_prediction_lift_weighted"]
    if pd.isna(lift1):
        lift1 = 0
    if pd.isna(lift2):
        lift2 = 0
    mod_score = 0 if pd.isna(mod) else mod
    ier_score = 0 if pd.isna(ier) else ier

    research_score = (
        2.0 * lift1
        + 1.0 * lift2
        + 0.5 * ier_score
        + 0.5 * mod_score
        - 0.25 * bal["largest_cluster_share"]
        - 0.03 * bal["tiny_cluster_count"]
        - 0.20 * bal["noise_ratio"]
    )

    row = {
        "method": method_name,
        "k": int(k) if k is not None else int(bal["num_clusters"]),
        "research_score": research_score,
        "modularity": mod,
        "internal_edge_ratio": ier,
        **bal,
        **geo,
        **prop,
    }

    if save_candidate:
        prefix = f"{method_name}_k{row['k']}".replace("/", "_")
        save_csv(assign, out_dir / f"{prefix}_company_cluster_assignment.csv")
        save_csv(csum, out_dir / f"{prefix}_cluster_summary.csv")
        save_csv(cross_s, out_dir / f"{prefix}_next_quarter_cluster_contagion.csv")
        save_csv(same_s, out_dir / f"{prefix}_same_quarter_cluster_correlation.csv")

    return row, assign, csum, cross_e, same_e, cross_s, same_s


def clean_date_gap_plot_data(date_summary: pd.DataFrame, mode: str, min_exposed: int):
    d = date_summary[
        date_summary["mode"].eq(mode)
        & (date_summary["date_observation_count"] >= min_exposed)
    ].copy()
    if d.empty:
        return d
    # Clean non-duplicate label: cluster + theme + signal + relation.
    d["plot_label"] = (
        "C" + d["cluster_id"].astype(str)
        + " | " + d["cluster_theme_label"].astype(str)
        + " | " + d["signal"].astype(str)
        + " | " + d["relation_group"].astype(str)
    )
    return d


def make_date_summary(cross_e: pd.DataFrame, same_e: pd.DataFrame):
    frames = [
        cross_e.assign(mode="cross_quarter"),
        same_e.assign(mode="same_quarter")
    ]
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

    out = out.rename(columns={
        "source_cluster_id": "cluster_id",
        "source_cluster_theme_label": "cluster_theme_label",
    })
    return out


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")

    rag_dir = Path(args.rag_output_dir)
    two_dir = Path(args.two_part_dir)

    print("=" * 100)
    print("V3 clustering method comparison")
    print("two_part_dir:", two_dir)
    print("out_dir:", out_dir)
    print("=" * 100)

    outlook = pd.read_csv(two_dir / "cleaned_outlook_all.csv")
    rel = pd.read_csv(two_dir / "matched_company_relationships.csv")
    cross = pd.read_csv(two_dir / "cross_quarter_events.csv")
    same = pd.read_csv(two_dir / "same_quarter_events.csv")
    concepts = discover_concepts(rag_dir)

    outlook = filter_quarter_range(outlook, ["quarter"], args.start_quarter, args.end_quarter)
    rel = filter_quarter_range(rel, ["quarter"], args.start_quarter, args.end_quarter)
    cross = filter_quarter_range(cross, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)
    same = filter_quarter_range(same, ["source_quarter", "target_quarter"], args.start_quarter, args.end_quarter)

    dates = combine_dates(
        load_dates_from_csv(Path(args.combined_transcripts)),
        load_dates_from_jsonl(Path(args.evidence_jsonl)),
    )
    dates = filter_quarter_range(dates, ["quarter"], args.start_quarter, args.end_quarter)
    save_csv(dates, out_dir / "company_quarter_release_dates.csv")

    feat, feature_cols, G = build_base_features(outlook, rel, concepts)
    feat = add_timing_features(feat, dates)
    feat = add_propagation_history_features(feat, cross)

    # Refresh feature columns after adding new features.
    feature_cols = [
        c for c in feat.columns
        if c not in {"company_node", "ticker", "company"}
        and pd.api.types.is_numeric_dtype(feat[c])
    ]

    save_csv(feat, out_dir / "company_feature_matrix_v3_raw.csv")

    X_scaled, used_features, X_processed = preprocess_features(feat, feature_cols, args.winsor_quantile)
    processed = pd.DataFrame(X_processed, columns=used_features)
    processed.insert(0, "company_node", feat["company_node"].values)
    save_csv(processed, out_dir / "company_feature_matrix_v3_processed.csv")

    candidates = []

    # 1. Graph greedy modularity
    try:
        labels = run_graph_greedy(G, feat)
        candidates.append(("graph_greedy_modularity", None, labels))
    except Exception as e:
        print("graph_greedy_modularity failed:", e)

    # 2. KMeans, Agglomerative, Spectral across k
    for k in range(args.min_k, args.max_k + 1):
        try:
            candidates.append(("kmeans", k, run_kmeans(X_scaled, k, args.random_state)))
        except Exception as e:
            print("kmeans failed k", k, e)

        try:
            candidates.append(("agglomerative_ward", k, run_agglomerative(X_scaled, k)))
        except Exception as e:
            print("agglomerative failed k", k, e)

        try:
            # Spectral can be slower; still try.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidates.append(("spectral", k, run_spectral(X_scaled, k, args.random_state)))
        except Exception as e:
            print("spectral failed k", k, e)

    # 3. Optional HDBSCAN
    if args.run_optional_hdbscan:
        labels = run_optional_hdbscan(X_scaled, args.random_state)
        if labels is not None:
            candidates.append(("hdbscan_umap_optional", None, labels))

    rows = []
    cache = {}

    for method, k, labels in candidates:
        print(f"Evaluating candidate: {method}, k={k}")
        row, assign, csum, cross_e, same_e, cross_s, same_s = evaluate_candidate(
            method, k, labels, feat, X_scaled, G, rel, cross, same, dates,
            out_dir, args.min_exposed, save_candidate=False
        )
        rows.append(row)
        key = (method, row["k"])
        cache[key] = (labels, assign, csum, cross_e, same_e, cross_s, same_s)

    comparison = pd.DataFrame(rows).sort_values("research_score", ascending=False)
    save_csv(comparison, out_dir / "method_comparison_summary.csv")
    plot_method_comparison(comparison, fig_dir / "method_comparison_research_score.png")

    if comparison.empty:
        raise RuntimeError("No clustering candidates succeeded.")

    best = comparison.iloc[0]
    best_key = (best["method"], int(best["k"]))
    labels, assign, csum, cross_e, same_e, cross_s, same_s = cache[best_key]

    save_csv(assign, out_dir / "best_company_cluster_assignment.csv")
    save_csv(csum, out_dir / "best_cluster_summary.csv")
    save_csv(cross_e, out_dir / "best_cross_quarter_events_with_cluster_dates.csv")
    save_csv(same_e, out_dir / "best_same_quarter_events_with_cluster_dates.csv")
    save_csv(cross_s, out_dir / "best_next_quarter_cluster_contagion.csv")
    save_csv(same_s, out_dir / "best_same_quarter_cluster_correlation.csv")

    date_summary = make_date_summary(cross_e, same_e)
    save_csv(date_summary, out_dir / "best_release_date_gap_by_cluster_clean.csv")

    plot_pca(assign, X_scaled, fig_dir / "best_cluster_pca.png")

    # Clean date gap plots: use non-duplicate plot label.
    same_date_clean = clean_date_gap_plot_data(date_summary, "same_quarter", args.min_exposed)
    cross_date_clean = clean_date_gap_plot_data(date_summary, "cross_quarter", args.min_exposed)

    save_csv(same_date_clean, out_dir / "best_same_quarter_date_gap_plot_data.csv")
    save_csv(cross_date_clean, out_dir / "best_cross_quarter_date_gap_plot_data.csv")

    bar_clean(
        same_date_clean, "plot_label", "mean_abs_release_date_gap_days",
        "Same-quarter date gap by cluster / signal / relation",
        fig_dir / "best_same_quarter_date_gap_clean.png",
        top=30
    )

    bar_clean(
        same_date_clean, "plot_label", "share_source_before_target",
        "Same-quarter source-before-target ratio",
        fig_dir / "best_source_before_target_ratio.png",
        top=30
    )

    # Lift plots
    lift = cross_s[cross_s["exposed_edges"] >= args.min_exposed].copy()
    if not lift.empty:
        lift["plot_label"] = (
            "C" + lift["cluster_id"].astype(str)
            + " | " + lift["cluster_theme_label"].astype(str)
            + " | " + lift["signal"].astype(str)
            + " | " + lift["relation_group"].astype(str)
        )
    bar_clean(
        lift, "plot_label", "prediction_lift",
        "Best method: next-quarter prediction lift",
        fig_dir / "best_next_quarter_prediction_lift.png",
        top=30
    )

    # Markdown report
    report = []
    report.append("# V3 Clustering Method Comparison")
    report.append("")
    report.append("## Why V3 was needed")
    report.append("")
    report.append("The previous KMeans/PCA result showed heavy overlap and repeated cluster labels in the timing plot. V3 compares graph-based, hierarchical, spectral, and KMeans clustering, adds more timing and propagation-history features, and selects the best method using information-propagation metrics rather than silhouette alone.")
    report.append("")
    report.append("## Data scale")
    report.append("")
    report.append(f"- Companies: {len(feat):,}")
    report.append(f"- Raw numeric features: {len(feature_cols):,}")
    report.append(f"- Used features after preprocessing: {len(used_features):,}")
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
    report.append("## Clean same-quarter date gap plot data")
    report.append("")
    report.append(same_date_clean.head(30).to_markdown(index=False) if not same_date_clean.empty else "No results.")
    report.append("")

    report_path = out_dir / "v3_clustering_method_comparison_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"SAVED {report_path}")

    print("DONE")
    print("Best method:", best["method"], "k=", best["k"])
    print("Report:", report_path)


if __name__ == "__main__":
    main()
