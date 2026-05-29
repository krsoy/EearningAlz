#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustered Network Diffusion Analysis V2

Why V2
------
The previous cluster run produced nearly all clusters labelled as
"Semiconductors / Hardware". That happened because the industry-labelling logic
over-weighted weak/non-zero semiconductor concepts. It also used raw count-like
features where extreme nodes/supernodes can dominate KMeans.

This V2 script reruns the clustering and produces diagnostic files explaining:
1. whether the feature matrix is dominated by outliers;
2. whether KMeans creates singleton clusters;
3. which features drive each cluster;
4. which top companies and concepts support each industry label;
5. why each cluster receives its final industry/thematic label.

Major changes from V1
---------------------
1. Robust feature preprocessing:
   - log1p transform for count/centrality/rate-like positive features
   - winsorization / clipping at a chosen upper quantile
   - removal of zero-variance features
   - StandardScaler after robust preprocessing

2. Better k selection:
   - silhouette, Calinski-Harabasz, Davies-Bouldin, modularity
   - largest cluster share
   - singleton cluster count
   - minimum cluster size
   - combined ranking with penalties for bad balance

3. Safer industry labelling:
   - company/ticker evidence first
   - concept evidence only if strong enough
   - no cluster is labelled semiconductor just because chip_supply is non-zero
   - output label-score diagnostics for manual inspection

4. Same-quarter and next-quarter cluster analysis:
   - same-quarter cluster correlation
   - next-quarter cluster contagion
   - baseline rate
   - prediction lift
   - effectiveness score
   - release-date gap statistics if dates exist

Run example
-----------
cd ~/sem2/RAG

python run_clustered_network_diffusion_analysis_v2.py \
  --rag-output-dir rag_chroma_output \
  --two-part-dir rag_chroma_output/two_part_network_prediction_analysis \
  --combined-transcripts ../data/combined_transcript_data/combined_transcripts_deduplicated.csv \
  --evidence-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl \
  --out-dir rag_chroma_output/clustered_network_diffusion_analysis_v2 \
  --start-quarter 2019Q2 \
  --end-quarter 2026Q2 \
  --min-k 3 \
  --max-k 20 \
  --min-cluster-size 10 \
  --min-exposed 5

Windows
-------
python run_clustered_network_diffusion_analysis_v2.py ^
  --rag-output-dir rag_chroma_output ^
  --two-part-dir rag_chroma_output\two_part_network_prediction_analysis ^
  --combined-transcripts ..\data\combined_transcript_data\combined_transcripts_deduplicated.csv ^
  --evidence-jsonl rag_chroma_output\rag_evidence_packages_full_gpu_direct.jsonl ^
  --out-dir rag_chroma_output\clustered_network_diffusion_analysis_v2 ^
  --start-quarter 2019Q2 ^
  --end-quarter 2026Q2 ^
  --min-k 3 ^
  --max-k 20 ^
  --min-cluster-size 10 ^
  --min-exposed 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


STANDARD_SIGNALS = [
    "demand_outlook",
    "supply_outlook",
    "margin_outlook",
    "capex_outlook",
    "inventory_outlook",
    "pricing_outlook",
]

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
        "JPM", "MA", "BANCORP", "BANK", "FINANCIAL", "CAPITAL", "CREDIT", "INSURANCE",
        "MORGAN", "GOLDMAN", "WELLS", "CITI", "VISA", "MASTERCARD"
    ],
    "Cloud / AI / Digital Platforms": [
        "AMZN", "MSFT", "GOOG", "GOOGL", "META", "ORCL", "IBM", "SHOP", "CRM",
        "ADBE", "NOW", "SNOW", "OPRA", "YOU", "XPER", "CLOUD", "SOFTWARE", "AI",
        "DATA CENTER", "DATACENTER"
    ],
    "Semiconductors / Hardware": [
        "NVDA", "INTC", "AMD", "MU", "TSM", "ASML", "AVGO", "QCOM", "SIMO",
        "AAPL", "DELL", "ANET", "IONQ", "VOXX", "ADEA", "GFS", "GLW", "CHIP",
        "SEMICONDUCTOR", "HARDWARE", "WAFER", "FOUNDRY"
    ],
    "Retail / Consumer / Restaurants": [
        "TGT", "WMT", "HD", "CVS", "YUMC", "COST", "LOW", "MCD", "SBUX",
        "RETAIL", "CONSUMER", "STORE", "RESTAURANT"
    ],
    "Automotive / Mobility": [
        "GM", "F", "HMC", "TSLA", "ADNT", "FORD", "MOTORS", "AUTOMOTIVE", "VEHICLE"
    ],
    "Energy / Utilities": [
        "BP", "EPD", "AEP", "AQN", "GEL", "LBRT", "XOM", "CVX", "SHEL",
        "ENERGY", "OIL", "GAS", "POWER", "UTILITY", "UTILITIES"
    ],
    "Healthcare / Pharma": [
        "HEALTH", "SNY", "IART", "BTMD", "PFE", "LLY", "UNH", "MRK", "BMY",
        "MEDICAL", "PHARMA", "BIOTECH", "THERAPEUTICS", "HOSPITAL"
    ],
    "Industrial / Manufacturing / Aerospace": [
        "CAT", "AERO", "ROP", "CARR", "PKOH", "STRL", "EML", "INDUSTRIAL",
        "MANUFACTURING", "AEROSPACE", "MACHINERY", "CONSTRUCTION"
    ],
    "Airlines / Travel / Leisure": [
        "UAL", "SOUTHWEST", "DAL", "AAL", "TRAVEL", "AIRLINES", "HOTEL", "LEISURE", "SGHC"
    ],
    "Telecom / Media / Cable": [
        "CHTR", "CMCSA", "VZ", "T", "TMUS", "TELECOM", "CABLE", "MEDIA", "COMMUNICATIONS"
    ],
}


CONCEPT_RULES = {
    # Concepts only apply when they are strong enough.
    # These are deliberately stricter than V1.
    "Semiconductors / Hardware": {
        "concepts": ["semiconductor_supply", "chip_supply", "supplier_constraint"],
        "max_threshold": 0.18,
        "sum_threshold": 0.35,
    },
    "Cloud / AI / Digital Platforms": {
        "concepts": ["cloud_infrastructure", "data_center_capacity"],
        "max_threshold": 0.12,
        "sum_threshold": 0.25,
    },
    "Retail / Consumer / Restaurants": {
        "concepts": ["customer_demand", "inventory_pressure", "pricing_pressure", "logistics_shipping"],
        "max_threshold": 0.45,
        "sum_threshold": 1.10,
    },
    "Energy / Utilities": {
        "concepts": ["oil_energy_supply"],
        "max_threshold": 0.15,
        "sum_threshold": 0.15,
    },
    "Industrial / Manufacturing / Aerospace": {
        "concepts": ["manufacturing_capacity", "production_capacity", "capex_expansion"],
        "max_threshold": 0.65,
        "sum_threshold": 1.30,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag-output-dir", default="rag_chroma_output")
    p.add_argument("--two-part-dir", default="rag_chroma_output/two_part_network_prediction_analysis")
    p.add_argument("--combined-transcripts", default="../data/combined_transcript_data/combined_transcripts_deduplicated.csv")
    p.add_argument("--evidence-jsonl", default="rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl")
    p.add_argument("--out-dir", default="rag_chroma_output/clustered_network_diffusion_analysis_v2")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--min-k", type=int, default=3)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--selected-k", type=int, default=0)
    p.add_argument("--min-cluster-size", type=int, default=10)
    p.add_argument("--max-largest-cluster-share", type=float, default=0.45)
    p.add_argument("--winsor-quantile", type=float, default=0.99)
    p.add_argument("--min-exposed", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"SAVED {path} rows={len(df):,}")


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


def clean_node_value(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "0"}:
        return ""
    return s


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


def discover_concepts(rag_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(rag_dir.rglob("concepts_*.csv")):
        try:
            if f.stat().st_size:
                df = pd.read_csv(f)
                df["source_file"] = str(f)
                frames.append(df)
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
        tmp = rel[cols].copy()
        tmp = tmp.rename(columns={"source_company_node": "company_node", "source_ticker": "ticker", "source_company": "company"})
        frames.append(tmp)
    if "target_company_node" in rel.columns:
        cols = ["target_company_node"]
        if "target_ticker" in rel.columns:
            cols.append("target_ticker")
        if "target_company" in rel.columns:
            cols.append("target_company")
        tmp = rel[cols].copy()
        tmp = tmp.rename(columns={"target_company_node": "company_node", "target_ticker": "ticker", "target_company": "company"})
        frames.append(tmp)

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


def build_feature_matrix(outlook: pd.DataFrame, rel: pd.DataFrame, concepts: pd.DataFrame):
    outlook = outlook.copy()
    rel = rel.copy()
    if "company_node" in outlook.columns:
        outlook["company_node"] = outlook["company_node"].map(clean_node_value)
        outlook = outlook[outlook["company_node"].ne("")].copy()
    for c in ["source_company_node", "target_company_node"]:
        if c in rel.columns:
            rel[c] = rel[c].map(clean_node_value)
    rel = rel[rel["source_company_node"].ne("") & rel["target_company_node"].ne("") & rel["source_company_node"].ne(rel["target_company_node"])].copy()

    G = build_graph(rel)
    nodes = sorted(set(G.nodes()) | set(outlook["company_node"].dropna().map(clean_node_value).unique()))
    feat = pd.DataFrame({"company_node": nodes})

    meta = company_metadata(outlook, rel)
    feat = feat.merge(meta, on="company_node", how="left")

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() else {}
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

    outlook["score"] = pd.to_numeric(outlook.get("score", 0), errors="coerce")
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
            feat = feat.merge(con, on="company_node", how="left")

    feat = feat.fillna(0)
    feat["company_node"] = feat["company_node"].map(clean_node_value)
    feat = feat[feat["company_node"].ne("")].copy()
    feat["ticker"] = feat.get("ticker", "").replace(0, "").astype(str)
    feat["company"] = feat.get("company", "").replace(0, "").astype(str)

    feature_cols = [
        c for c in feat.columns
        if c not in {"company_node", "ticker", "company"}
        and pd.api.types.is_numeric_dtype(feat[c])
    ]
    return feat, feature_cols, G


def feature_diagnostics(feat: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        s = pd.to_numeric(feat[c], errors="coerce")
        rows.append({
            "feature": c,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p50": s.quantile(0.50),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "zero_share": (s.fillna(0) == 0).mean(),
            "missing_share": s.isna().mean(),
            "max_to_mean": s.max() / s.mean() if s.mean() not in [0, np.nan] and s.mean() != 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("max_to_mean", ascending=False)


def preprocess_features(feat: pd.DataFrame, feature_cols: list[str], winsor_q: float):
    X = feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()

    # Remove zero-variance features.
    keep = []
    for c in feature_cols:
        if X[c].std() > 0:
            keep.append(c)
    X = X[keep]

    # log1p + winsorize. For non-negative engineered features this is safe.
    X2 = X.copy()
    for c in X2.columns:
        s = pd.to_numeric(X2[c], errors="coerce").fillna(0)
        if s.min() >= 0:
            s = np.log1p(s)
        lo = s.quantile(1 - winsor_q) if winsor_q < 1 else s.min()
        hi = s.quantile(winsor_q)
        s = s.clip(lower=lo, upper=hi)
        X2[c] = s

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X2.values)
    return X_scaled, list(X2.columns), X2


def modularity_score(G: nx.Graph, feat: pd.DataFrame, labels: np.ndarray) -> float:
    comm = defaultdict(set)
    for n, lab in zip(feat["company_node"], labels):
        if n in G:
            comm[int(lab)].add(n)
    communities = [v for v in comm.values() if v]
    if not communities or G.number_of_edges() == 0:
        return np.nan
    return nx.algorithms.community.quality.modularity(G, communities, weight="weight")


def evaluate_k(feat, X_scaled, G, min_k, max_k, seed, min_cluster_size, max_largest_share):
    rows = []
    labels_by_k = {}
    n = len(feat)
    max_k = min(max_k, n - 1)

    for k in range(max(2, min_k), max_k + 1):
        print(f"Evaluating k={k}")
        km = KMeans(n_clusters=k, random_state=seed, n_init=30)
        lab = km.fit_predict(X_scaled)
        labels_by_k[k] = lab
        counts = pd.Series(lab).value_counts()

        sil = silhouette_score(X_scaled, lab) if k < n else np.nan
        ch = calinski_harabasz_score(X_scaled, lab)
        db = davies_bouldin_score(X_scaled, lab)
        mod = modularity_score(G, feat, lab)

        singleton_count = int((counts < min_cluster_size).sum())
        largest_share = counts.max() / n

        rows.append({
            "k": k,
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
            "modularity": mod,
            "largest_cluster_share": largest_share,
            "smallest_cluster_size": int(counts.min()),
            "singleton_or_tiny_cluster_count": singleton_count,
            "num_companies": n,
            "passes_balance_filter": (singleton_count == 0 and largest_share <= max_largest_share),
        })

    m = pd.DataFrame(rows)
    m["rank_silhouette"] = m["silhouette_score"].rank(ascending=False)
    m["rank_ch"] = m["calinski_harabasz_score"].rank(ascending=False)
    m["rank_db"] = m["davies_bouldin_score"].rank(ascending=True)
    m["rank_modularity"] = m["modularity"].rank(ascending=False)
    m["rank_largest_share"] = m["largest_cluster_share"].rank(ascending=True)
    m["rank_tiny_clusters"] = m["singleton_or_tiny_cluster_count"].rank(ascending=True)
    m["combined_rank"] = m[[
        "rank_silhouette", "rank_ch", "rank_db", "rank_modularity",
        "rank_largest_share", "rank_tiny_clusters"
    ]].mean(axis=1)

    return m, labels_by_k


def select_k(metrics: pd.DataFrame, selected_k: int):
    if selected_k > 0:
        return selected_k
    valid = metrics[metrics["passes_balance_filter"]].copy()
    if valid.empty:
        valid = metrics.copy()
    return int(valid.sort_values(["combined_rank", "singleton_or_tiny_cluster_count", "largest_cluster_share"]).iloc[0]["k"])


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


def score_company_keywords(text: str) -> Counter:
    text_u = "" if pd.isna(text) else str(text).upper()
    tokens = set(re.findall(r"[A-Z0-9&.\-]+", text_u))
    scores = Counter()
    evidence = defaultdict(list)

    for label, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            kw_u = kw.upper()
            hit = False
            if len(kw_u) <= 5:
                hit = kw_u in tokens
            else:
                hit = kw_u in text_u
            if hit:
                scores[label] += 3.0 if len(kw_u) <= 5 else 2.0
                evidence[label].append(kw)
    return scores, evidence


def score_concepts(concepts: dict) -> tuple[Counter, dict]:
    scores = Counter()
    evidence = defaultdict(list)
    for label, rule in CONCEPT_RULES.items():
        vals = {c: concepts.get(c, 0.0) for c in rule["concepts"]}
        max_val = max(vals.values()) if vals else 0
        sum_val = sum(vals.values())
        if max_val >= rule["max_threshold"] or sum_val >= rule["sum_threshold"]:
            score = 1.5 + 5.0 * max_val + 1.0 * sum_val
            scores[label] += score
            evidence[label].append(f"concept_max={max_val:.3f}, concept_sum={sum_val:.3f}")
    return scores, evidence


def infer_cluster_label(top_companies: str, top_concepts: str, num_companies: int):
    company_scores, company_evidence = score_company_keywords(top_companies)
    concept_dict = parse_top_concepts(top_concepts)
    concept_scores, concept_evidence = score_concepts(concept_dict)

    scores = Counter()
    scores.update(company_scores)

    # Company evidence is primary. Concept evidence is secondary.
    for k, v in concept_scores.items():
        scores[k] += 0.65 * v

    if num_companies <= 2 and company_scores:
        label = company_scores.most_common(1)[0][0]
    elif scores:
        label = scores.most_common(1)[0][0]
    else:
        label = "Mixed / Other"

    diagnostics = []
    for lab, sc in scores.most_common():
        ev = []
        ev.extend(company_evidence.get(lab, []))
        ev.extend(concept_evidence.get(lab, []))
        diagnostics.append(f"{lab}:{sc:.2f} [{'|'.join(ev[:8])}]")

    return label, "; ".join(diagnostics)


def make_cluster_summary(assign: pd.DataFrame, feat: pd.DataFrame, rel: pd.DataFrame, G: nx.Graph):
    concept_cols = [c for c in feat.columns if c.startswith("concept_rate_")]
    rows = []

    for cid, g in assign.groupby("cluster_id"):
        nodes = set(g["company_node"])
        sub = G.subgraph(nodes)

        central = sorted([(n, G.degree(n) if n in G else 0) for n in nodes], key=lambda x: x[1], reverse=True)[:15]
        top_companies = []
        for n, d in central:
            row = g[g["company_node"].eq(n)].iloc[0]
            ticker = clean_node_value(row.get("ticker", ""))
            company = clean_node_value(row.get("company", ""))
            top_companies.append(ticker if ticker else company)

        tmp = feat[feat["company_node"].isin(nodes)]
        concept_means = tmp[concept_cols].mean().to_dict() if concept_cols else {}
        top_concepts = "; ".join(
            f"{k.replace('concept_rate_', '')}:{v:.3f}"
            for k, v in sorted(concept_means.items(), key=lambda x: x[1], reverse=True)[:6]
            if v > 0
        )

        top_text = "; ".join(top_companies)
        label, label_diagnostics = infer_cluster_label(top_text, top_concepts, len(nodes))

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
            "industry_label": label,
            "num_companies": len(nodes),
            "internal_edges": sub.number_of_edges(),
            "external_edges": external_edges,
            "internal_edge_ratio": sub.number_of_edges() / (sub.number_of_edges() + external_edges) if sub.number_of_edges() + external_edges else 0,
            "density": nx.density(sub) if sub.number_of_nodes() > 1 else 0,
            "top_companies": top_text,
            "top_relation_groups": top_rels,
            "top_concepts": top_concepts,
            "label_diagnostics": label_diagnostics,
        })

    return pd.DataFrame(rows).sort_values("num_companies", ascending=False)


def enrich_events(events: pd.DataFrame, assign: pd.DataFrame, dates: pd.DataFrame):
    if events.empty:
        return events
    cmap = assign.set_index("company_node")["cluster_id"].to_dict()
    imap = assign.set_index("company_node")["industry_label"].to_dict()

    out = events.copy()
    out["source_cluster_id"] = out["source_node"].map(cmap)
    out["target_cluster_id"] = out["target_node"].map(cmap)
    out["source_industry_label"] = out["source_node"].map(imap)
    out["target_industry_label"] = out["target_node"].map(imap)
    out["same_cluster"] = out["source_cluster_id"].notna() & out["source_cluster_id"].eq(out["target_cluster_id"])

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


def summarize_events(events: pd.DataFrame, mode: str):
    if events.empty:
        return pd.DataFrame()
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
        "source_industry_label", "signal", "source_label", "source_direction", "relation_group"
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
            "industry_label": d.pop("source_industry_label"),
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


def aggregate_industry(summary: pd.DataFrame, mode: str, min_exposed: int):
    if summary.empty:
        return pd.DataFrame()
    s = summary[(summary["analysis_mode"].eq(mode)) & (summary["exposed_edges"] >= min_exposed)].copy()
    if s.empty:
        return pd.DataFrame()

    rows = []
    for key, g in s.groupby(["industry_label", "signal", "source_direction"], dropna=False):
        exposed = g["exposed_edges"].sum()
        direction = g["direction_match_edges"].sum()
        exact = g["exact_match_edges"].sum()
        w = g["exposed_edges"]
        baseline = np.average(g["baseline_rate"].fillna(0), weights=w) if w.sum() else np.nan
        rate = direction / exposed if exposed else np.nan
        rows.append({
            "analysis_mode": mode,
            "industry_label": key[0],
            "signal": key[1],
            "source_direction": key[2],
            "exposed_edges": exposed,
            "exact_match_rate": exact / exposed if exposed else np.nan,
            "direction_match_rate": rate,
            "baseline_rate": baseline,
            "prediction_lift": rate - baseline if not pd.isna(baseline) else np.nan,
            "falsification_rate": 1 - rate if not pd.isna(rate) else np.nan,
            "most_effective_relation_group": g.sort_values("effectiveness_score", ascending=False)["relation_group"].iloc[0],
        })
    return pd.DataFrame(rows).sort_values(["prediction_lift", "direction_match_rate", "exposed_edges"], ascending=[False, False, False])


def plot_line(df, x, y, title, path):
    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"SAVED {path}")


def bar(df, x, y, title, path, top=30):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y]).sort_values(y, ascending=False).head(top)
    if d.empty:
        return
    ax = d.sort_values(y).plot(kind="barh", x=x, y=y, legend=False, figsize=(11, 7))
    ax.set_title(title)
    ax.set_xlabel(y)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"SAVED {path}")


def pca_plot(assign, X, path):
    if len(assign) < 3:
        return
    xy = PCA(n_components=2, random_state=42).fit_transform(X)
    d = assign.copy()
    d["pc1"] = xy[:, 0]
    d["pc2"] = xy[:, 1]
    plt.figure(figsize=(12, 8))
    for cid, g in d.groupby("cluster_id"):
        plt.scatter(g["pc1"], g["pc2"], s=18, alpha=0.65, label=f"C{cid}")
        plt.text(g["pc1"].mean(), g["pc2"].mean(), f"C{cid}\n{g['industry_label'].iloc[0]}", fontsize=8, ha="center", bbox=dict(fc="white", alpha=0.65))
    plt.title("Company network clusters, PCA projection")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"SAVED {path}")


def main():
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")
    rag_dir = Path(args.rag_output_dir)
    two_dir = Path(args.two_part_dir)

    print("=" * 100)
    print("Clustered Network Diffusion Analysis V2")
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

    feat, feature_cols, G = build_feature_matrix(outlook, rel, concepts)
    save_csv(feat, out_dir / "company_feature_matrix_raw.csv")
    diag = feature_diagnostics(feat, feature_cols)
    save_csv(diag, out_dir / "feature_diagnostics_raw.csv")

    X_scaled, used_features, X_processed = preprocess_features(feat, feature_cols, args.winsor_quantile)
    processed = pd.DataFrame(X_processed, columns=used_features)
    processed.insert(0, "company_node", feat["company_node"].values)
    save_csv(processed, out_dir / "company_feature_matrix_processed.csv")

    metrics, labels_by_k = evaluate_k(
        feat, X_scaled, G, args.min_k, args.max_k, args.random_state,
        args.min_cluster_size, args.max_largest_cluster_share
    )
    best_k = select_k(metrics, args.selected_k)
    metrics["selected_flag"] = metrics["k"].eq(best_k)
    save_csv(metrics, out_dir / "cluster_k_selection_metrics_v2.csv")

    for col in ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "modularity", "largest_cluster_share", "singleton_or_tiny_cluster_count"]:
        plot_line(metrics, "k", col, col + " by k", fig_dir / f"{col}_by_k.png")

    labels = labels_by_k[best_k]
    assign = feat[["company_node", "ticker", "company"]].copy()
    assign["cluster_id"] = labels

    cluster_summary = make_cluster_summary(assign, feat, rel, G)
    industry_map = cluster_summary.set_index("cluster_id")["industry_label"].to_dict()
    assign["industry_label"] = assign["cluster_id"].map(industry_map)

    save_csv(assign, out_dir / "company_cluster_assignment_v2.csv")
    save_csv(cluster_summary, out_dir / "cluster_summary_v2.csv")
    pca_plot(assign, X_scaled, fig_dir / "network_cluster_pca_v2.png")

    cross_e = enrich_events(cross, assign, dates)
    same_e = enrich_events(same, assign, dates)
    save_csv(cross_e, out_dir / "cross_quarter_events_with_cluster_dates_v2.csv")
    save_csv(same_e, out_dir / "same_quarter_events_with_cluster_dates_v2.csv")

    cross_s = summarize_events(cross_e, "cross_quarter")
    same_s = summarize_events(same_e, "same_quarter")
    save_csv(cross_s, out_dir / "next_quarter_cluster_contagion_v2.csv")
    save_csv(same_s, out_dir / "same_quarter_cluster_correlation_v2.csv")

    icross = aggregate_industry(cross_s, "cross_quarter", args.min_exposed)
    isame = aggregate_industry(same_s, "same_quarter", args.min_exposed)
    save_csv(icross, out_dir / "industry_level_next_quarter_contagion_v2.csv")
    save_csv(isame, out_dir / "industry_level_same_quarter_correlation_v2.csv")

    eff = cross_s[cross_s["exposed_edges"] >= args.min_exposed].copy()
    eff = eff.sort_values(["effectiveness_score", "prediction_lift", "direction_match_rate", "exposed_edges"], ascending=[False, False, False, False])
    save_csv(eff, out_dir / "cluster_effectiveness_ranking_v2.csv")

    all_dates = pd.concat([cross_e.assign(mode="cross_quarter"), same_e.assign(mode="same_quarter")], ignore_index=True)
    all_dates = all_dates[all_dates["same_cluster"].astype(bool) & all_dates["source_active"].astype(bool)].copy()
    if not all_dates.empty:
        date_summary = all_dates.groupby(["mode", "source_cluster_id", "source_industry_label", "signal", "relation_group"], dropna=False).agg(
            event_count=("signal", "count"),
            mean_release_date_gap_days=("release_date_gap_days", "mean"),
            median_release_date_gap_days=("release_date_gap_days", "median"),
            mean_abs_release_date_gap_days=("abs_release_date_gap_days", "mean"),
            median_abs_release_date_gap_days=("abs_release_date_gap_days", "median"),
            share_source_before_target=("source_before_target", "mean"),
            date_observation_count=("release_date_gap_days", lambda x: int(x.notna().sum())),
        ).reset_index()
    else:
        date_summary = pd.DataFrame()
    save_csv(date_summary, out_dir / "release_date_gap_by_cluster_v2.csv")

    bar(cluster_summary, "industry_label", "num_companies", "Cluster size by industry label", fig_dir / "cluster_size_by_industry_v2.png")
    bar(eff, "industry_label", "effectiveness_score", "Next-quarter cluster effectiveness", fig_dir / "cluster_effectiveness_ranking_v2.png")
    bar(icross, "industry_label", "prediction_lift", "Next-quarter prediction lift by industry", fig_dir / "next_quarter_prediction_lift_by_industry_v2.png")
    bar(icross, "signal", "prediction_lift", "Next-quarter prediction lift by signal", fig_dir / "next_quarter_prediction_lift_by_signal_v2.png")
    bar(isame, "industry_label", "direction_match_rate", "Same-quarter similarity by industry", fig_dir / "same_quarter_similarity_by_industry_v2.png")

    if not date_summary.empty:
        bar(date_summary[date_summary["mode"].eq("same_quarter")], "source_industry_label", "mean_abs_release_date_gap_days", "Same-quarter date gap by cluster", fig_dir / "same_quarter_date_gap_by_cluster_v2.png")
        bar(date_summary[date_summary["mode"].eq("cross_quarter")], "source_industry_label", "mean_abs_release_date_gap_days", "Cross-quarter date gap by cluster", fig_dir / "cross_quarter_date_gap_by_cluster_v2.png")

    report = []
    report.append("# Clustered Network Diffusion Analysis V2")
    report.append("")
    report.append("## Why V2 was needed")
    report.append("")
    report.append("The previous version over-labelled clusters as Semiconductors / Hardware because weak semiconductor concepts were given too much label weight. V2 uses company/ticker evidence first, applies stricter concept thresholds, robustly transforms skewed numeric features, and penalizes tiny/singleton clusters during k selection.")
    report.append("")
    report.append("## Data scale")
    report.append("")
    report.append(f"- Companies: {len(feat):,}")
    report.append(f"- Raw numeric features: {len(feature_cols):,}")
    report.append(f"- Used numeric features after zero-variance removal: {len(used_features):,}")
    report.append(f"- Matched relationship rows: {len(rel):,}")
    report.append(f"- Cross-quarter events: {len(cross):,}")
    report.append(f"- Same-quarter events: {len(same):,}")
    report.append(f"- Release-date company-quarter rows: {len(dates):,}")
    report.append(f"- Selected k: {best_k}")
    report.append("")
    report.append("## Cluster selection metrics")
    report.append("")
    report.append(metrics.to_markdown(index=False))
    report.append("")
    report.append("## Cluster summary")
    report.append("")
    report.append(cluster_summary.to_markdown(index=False))
    report.append("")
    report.append("## Feature outlier diagnostics, top 30")
    report.append("")
    report.append(diag.head(30).to_markdown(index=False))
    report.append("")
    report.append("## Top next-quarter effectiveness")
    report.append("")
    report.append(eff.head(30).to_markdown(index=False) if not eff.empty else "No results.")
    report.append("")
    report.append("## Top same-quarter correlation")
    report.append("")
    st = same_s[same_s["exposed_edges"] >= args.min_exposed].head(30) if not same_s.empty else pd.DataFrame()
    report.append(st.to_markdown(index=False) if not st.empty else "No results.")
    report.append("")

    report_path = out_dir / "clustered_network_diffusion_summary_v2.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"SAVED {report_path}")
    print("DONE")


if __name__ == "__main__":
    main()
