import re
import json
import pandas as pd
import networkx as nx
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "kunhanw/earning_call_transcript_dataset_with_volatility_analysis"

OUT = Path("intel_hf_validation_strict")
OUT.mkdir(exist_ok=True)

TARGET_COMPANY = "Intel"
TARGET_TICKER = "INTC"

MAX_CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

ds = load_dataset(DATASET_NAME, split="train")
df = ds.to_pandas()

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(2))


# ============================================================
# 2. EXPAND META
# ============================================================

def parse_meta(x):
    if isinstance(x, dict):
        return x

    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}

    return {}


meta_df = pd.json_normalize(df["meta"].apply(parse_meta))
df = pd.concat([df.drop(columns=["meta"]), meta_df], axis=1)

print("\nAfter meta expansion:")
print(df.columns.tolist())


# ============================================================
# 3. NORMALIZE BASIC FIELDS
# ============================================================

df["text"] = df["text"].astype(str)

if "company" in df.columns:
    df["company"] = df["company"].astype(str)
else:
    df["company"] = ""

if "title" in df.columns:
    df["title"] = df["title"].astype(str)
else:
    df["title"] = ""

# Date: use utc=True to avoid mixed timezone error
if "publishOn" in df.columns:
    df["publish_date"] = pd.to_datetime(df["publishOn"], errors="coerce", utc=True)
    df["publish_date"] = df["publish_date"].dt.tz_convert(None)
elif "volatility_analysis.publish_date" in df.columns:
    df["publish_date"] = pd.to_datetime(
        df["volatility_analysis.publish_date"],
        errors="coerce",
        utc=True
    )
    df["publish_date"] = df["publish_date"].dt.tz_convert(None)
else:
    df["publish_date"] = pd.NaT

# Ticker: dataset uses "name" and also volatility_analysis.ticker
if "name" in df.columns:
    df["ticker"] = df["name"].astype(str).str.upper()
elif "volatility_analysis.ticker" in df.columns:
    df["ticker"] = df["volatility_analysis.ticker"].astype(str).str.upper()
else:
    df["ticker"] = ""

# Remove weird empty tickers
df["ticker"] = df["ticker"].replace(["NONE", "NAN", "NULL"], "")

# Save expanded raw file
df.to_csv(OUT / "raw_hf_transcripts_expanded.csv", index=False)


# ============================================================
# 4. REMOVE INTEL ITSELF TO AVOID LEAKAGE
# ============================================================

before = len(df)

df = df[df["ticker"].astype(str).str.upper() != TARGET_TICKER].copy()

after = len(df)

print(f"\nRemoved Intel own transcripts: {before - after}")
print("Remaining rows:", len(df))


# ============================================================
# 5. INTEL RELEVANCE KEYWORDS
# ============================================================

intel_keywords = [
    # Direct Intel references
    "Intel", "INTC", "Xeon", "Core Ultra", "Lunar Lake", "Meteor Lake",
    "Arrow Lake", "Gaudi", "Intel Foundry", "18A", "20A",

    # Client computing / PC
    "PC", "PCs", "notebook", "desktop", "laptop",
    "commercial PC", "consumer PC", "AI PC",
    "Windows", "Windows refresh", "enterprise refresh",

    # Data center / server / CPU
    "server", "servers", "CPU", "CPUs", "processor", "processors",
    "data center", "datacenter", "cloud", "enterprise infrastructure",
    "inference", "accelerator", "compute",

    # Foundry / manufacturing
    "foundry", "semiconductor manufacturing", "wafer", "fab",
    "advanced packaging", "process node", "chip manufacturing",

    # Pressure signals
    "inventory", "component", "shortage", "memory", "pricing",
    "margin pressure", "gross margin", "capex", "capital expenditure"
]

keyword_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in intel_keywords) + r")\b",
    flags=re.IGNORECASE
)


# ============================================================
# 6. CHUNK TRANSCRIPTS
# ============================================================

def split_text_into_chunks(text, max_words=220, overlap=40):
    words = str(text).split()

    if len(words) <= max_words:
        return [str(text)]

    chunks = []
    step = max_words - overlap

    for start in range(0, len(words), step):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


chunk_rows = []

print("\nChunking transcripts and extracting Intel-relevant chunks...")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking transcripts"):
    chunks = split_text_into_chunks(
        row["text"],
        max_words=MAX_CHUNK_WORDS,
        overlap=CHUNK_OVERLAP_WORDS
    )

    for chunk_id, chunk in enumerate(chunks):
        matches = keyword_pattern.findall(chunk)

        if matches:
            chunk_rows.append({
                "doc_id": idx,
                "chunk_id": chunk_id,
                "ticker": row.get("ticker"),
                "company": row.get("company"),
                "title": row.get("title"),
                "publish_date": row.get("publish_date"),
                "matched_keywords": sorted(list(set([m.lower() for m in matches]))),
                "matched_keyword_count": len(matches),
                "chunk_text": chunk
            })


intel_chunks = pd.DataFrame(chunk_rows)

if intel_chunks.empty:
    raise RuntimeError("No Intel-relevant chunks found. Check keyword list or dataset content.")

intel_chunks = intel_chunks.sort_values(
    "matched_keyword_count",
    ascending=False
).reset_index(drop=True)

intel_chunks.to_csv(OUT / "intel_relevant_chunks_keyword.csv", index=False)

print("\nIntel-relevant chunks:", len(intel_chunks))
print(intel_chunks[
    ["ticker", "company", "publish_date", "matched_keyword_count", "matched_keywords"]
].head(20))


# ============================================================
# 7. STRICT INTEL-AWARE SIGNAL EXTRACTION
# ============================================================

def has_any(text, terms):
    t = str(text).lower()
    return any(term.lower() in t for term in terms)


def count_any(text, terms):
    t = str(text).lower()
    return sum(t.count(term.lower()) for term in terms)


def extract_intel_aware_signals(text):
    """
    More strict rules:
    Avoid counting generic words like cloud, margin, supply alone.
    Require Intel-relevant business context.
    """
    t = str(text).lower()
    signals = {}

    # PC demand
    pc_terms = [
        "pc", "pcs", "notebook", "desktop", "laptop",
        "commercial pc", "consumer pc", "ai pc"
    ]
    pc_context_terms = [
        "commercial", "consumer", "notebook", "desktop", "laptop",
        "windows", "ai pc", "refresh", "device", "devices"
    ]

    signals["pc_demand"] = (
        count_any(t, pc_terms)
        if has_any(t, pc_terms) and has_any(t, pc_context_terms)
        else 0
    )

    # Server CPU demand
    server_terms = [
        "server", "servers", "data center", "datacenter",
        "enterprise infrastructure"
    ]
    cpu_terms = [
        "cpu", "cpus", "xeon", "processor", "processors", "compute"
    ]

    signals["server_cpu_demand"] = (
        count_any(t, server_terms + cpu_terms)
        if has_any(t, server_terms) and has_any(t, cpu_terms)
        else 0
    )

    # Cloud capacity
    cloud_terms = [
        "cloud", "data center", "datacenter"
    ]
    capacity_terms = [
        "capacity", "capex", "capital expenditure",
        "infrastructure", "buildout", "build-out", "investment"
    ]

    signals["cloud_capacity"] = (
        count_any(t, cloud_terms + capacity_terms)
        if has_any(t, cloud_terms) and has_any(t, capacity_terms)
        else 0
    )

    # Foundry / manufacturing
    foundry_terms = [
        "foundry", "fab", "wafer", "advanced packaging",
        "process node", "18a", "20a", "semiconductor manufacturing"
    ]

    signals["foundry_manufacturing"] = (
        count_any(t, foundry_terms)
        if has_any(t, foundry_terms)
        else 0
    )

    # Supply pressure
    pressure_terms = [
        "shortage", "constraint", "constrained",
        "component", "memory", "inventory", "supply"
    ]
    semiconductor_context_terms = [
        "pc", "server", "data center", "datacenter",
        "semiconductor", "chip", "cpu", "memory", "component",
        "processor", "processors"
    ]

    signals["supply_pressure"] = (
        count_any(t, pressure_terms)
        if has_any(t, pressure_terms) and has_any(t, semiconductor_context_terms)
        else 0
    )

    # Margin pressure
    margin_terms = [
        "margin", "gross margin", "pricing",
        "cost pressure", "cost", "price"
    ]

    signals["margin_pressure"] = (
        count_any(t, margin_terms)
        if has_any(t, margin_terms) and has_any(t, semiconductor_context_terms)
        else 0
    )

    return signals


signal_names = [
    "pc_demand",
    "server_cpu_demand",
    "cloud_capacity",
    "foundry_manufacturing",
    "supply_pressure",
    "margin_pressure"
]

signal_rows = []

for _, row in tqdm(intel_chunks.iterrows(), total=len(intel_chunks), desc="Extracting strict signals"):
    signals = extract_intel_aware_signals(row["chunk_text"])
    signal_rows.append(signals)

signal_df = pd.DataFrame(signal_rows)

for col in signal_names:
    intel_chunks[col + "_count"] = signal_df[col].fillna(0).astype(int)

signal_count_cols = [s + "_count" for s in signal_names]

intel_chunks["signal_total_count"] = intel_chunks[signal_count_cols].sum(axis=1)

# Keep only chunks with strict Intel-aware signals
intel_chunks = intel_chunks[intel_chunks["signal_total_count"] > 0].copy()

intel_chunks["intel_relevance_score_rule"] = (
    intel_chunks["matched_keyword_count"] +
    intel_chunks["signal_total_count"]
)

intel_chunks.to_csv(OUT / "intel_relevant_chunks_with_strict_signals.csv", index=False)

print("\nStrict Intel-aware chunks:", len(intel_chunks))
print(intel_chunks[
    ["ticker", "company", "matched_keyword_count", "signal_total_count", "intel_relevance_score_rule"]
].head(20))


# ============================================================
# 8. COMPANY WEIGHTS
# ============================================================

important_company_weights = {
    # PC OEM / server vendors
    "DELL": 3.0,
    "HPQ": 3.0,
    "HPE": 2.5,
    "SMCI": 2.5,
    "LEN": 2.5,
    "LNVGY": 2.5,

    # Semiconductor / memory / equipment
    "AMD": 2.0,
    "MU": 2.5,
    "NVDA": 2.0,
    "ASML": 2.5,
    "AMAT": 2.5,
    "KLAC": 2.5,
    "LRCX": 2.5,
    "TSM": 2.5,
    "TSMC": 2.5,

    # Cloud / enterprise / networking
    "MSFT": 2.5,
    "AMZN": 2.5,
    "GOOGL": 2.5,
    "GOOG": 2.5,
    "META": 2.0,
    "ORCL": 2.0,
    "CSCO": 2.0,
    "ANET": 2.0,
    "CRM": 1.5,

    # Storage / memory ecosystem
    "WDC": 1.8,
    "STX": 1.8,
    "NTAP": 1.8,

    # Optical / data center infrastructure
    "CIEN": 1.8,
    "LITE": 1.8,
    "EQIX": 1.8,
}


def company_weight(ticker):
    return important_company_weights.get(str(ticker).upper(), 1.0)


# ============================================================
# 9. BUILD KG EDGES
# ============================================================

business_target_map = {
    "pc_demand": "Intel_CCG",
    "server_cpu_demand": "Intel_DCAI",
    "cloud_capacity": "Intel_DCAI",
    "foundry_manufacturing": "Intel_Foundry",
    "supply_pressure": "Intel_CCG_or_DCAI",
    "margin_pressure": "Intel_Margin"
}

edges = []

for _, row in intel_chunks.iterrows():
    source = row["ticker"] if pd.notna(row["ticker"]) and str(row["ticker"]).strip() else row["company"]
    source = str(source).upper()
    w_company = company_weight(source)

    for signal in signal_names:
        raw_count = int(row[signal + "_count"])

        if raw_count > 0:
            weighted_count = raw_count * w_company

            edges.append({
                "source_company": source,
                "source_company_name": row["company"],
                "relation": "mentions_signal",
                "target_signal": signal,
                "target_business_node": business_target_map[signal],
                "raw_weight": raw_count,
                "company_weight": w_company,
                "weight": weighted_count,
                "publish_date": row["publish_date"],
                "matched_keywords": json.dumps(row["matched_keywords"], ensure_ascii=False),
                "evidence": row["chunk_text"][:800]
            })

kg_edges = pd.DataFrame(edges)

if kg_edges.empty:
    raise RuntimeError("No KG edges generated. Check strict signal rules.")

kg_edges.to_csv(OUT / "kg_edges_strict_weighted.csv", index=False)

print("\nStrict weighted KG edges:", len(kg_edges))
print(kg_edges.head(20))


# ============================================================
# 10. COMPANY SIGNAL SUMMARY
# ============================================================

company_signal_summary = (
    kg_edges
    .groupby(
        ["source_company", "source_company_name", "target_signal", "target_business_node"],
        dropna=False
    )
    .agg(
        total_weight=("weight", "sum"),
        raw_weight=("raw_weight", "sum"),
        mention_count=("weight", "count"),
        first_date=("publish_date", "min"),
        last_date=("publish_date", "max")
    )
    .reset_index()
    .sort_values("total_weight", ascending=False)
)

company_signal_summary.to_csv(OUT / "company_signal_summary_strict_weighted.csv", index=False)

print("\nCompany signal summary:")
print(company_signal_summary.head(30))


# ============================================================
# 11. BUILD NETWORKX KG
# ============================================================

G = nx.DiGraph()

for _, e in kg_edges.iterrows():
    company_node = f"COMPANY::{e['source_company']}"
    signal_node = f"SIGNAL::{e['target_signal']}"
    business_node = f"BUSINESS::{e['target_business_node']}"

    G.add_node(
        company_node,
        node_type="company",
        name=e["source_company_name"]
    )

    G.add_node(
        signal_node,
        node_type="signal"
    )

    G.add_node(
        business_node,
        node_type="intel_business_node"
    )

    # Company -> Signal
    if G.has_edge(company_node, signal_node):
        G[company_node][signal_node]["weight"] += float(e["weight"])
    else:
        G.add_edge(
            company_node,
            signal_node,
            relation="mentions",
            weight=float(e["weight"])
        )

    # Signal -> Intel business node
    if G.has_edge(signal_node, business_node):
        G[signal_node][business_node]["weight"] += float(e["weight"])
    else:
        G.add_edge(
            signal_node,
            business_node,
            relation="affects",
            weight=float(e["weight"])
        )

print("\nKG nodes:", G.number_of_nodes())
print("KG edges:", G.number_of_edges())


# ============================================================
# 12. PAGERANK
# ============================================================

pagerank = nx.pagerank(G, weight="weight")

pr_df = pd.DataFrame([
    {
        "node": node,
        "pagerank": score,
        "node_type": G.nodes[node].get("node_type")
    }
    for node, score in pagerank.items()
]).sort_values("pagerank", ascending=False)

pr_df.to_csv(OUT / "kg_pagerank_strict_weighted.csv", index=False)

print("\nTop KG nodes:")
print(pr_df.head(30))


# ============================================================
# 13. NEXT IMPACTED INTEL NODES
# ============================================================

next_impact = (
    kg_edges
    .groupby(["target_business_node", "target_signal"], dropna=False)
    .agg(
        total_signal_weight=("weight", "sum"),
        raw_signal_weight=("raw_weight", "sum"),
        source_company_count=("source_company", "nunique"),
        evidence_count=("evidence", "count")
    )
    .reset_index()
)

next_impact["impact_score"] = (
    next_impact["total_signal_weight"] *
    next_impact["source_company_count"]
)

next_impact = next_impact.sort_values("impact_score", ascending=False)

next_impact.to_csv(OUT / "next_impacted_intel_nodes_strict_weighted.csv", index=False)

print("\nLikely next impacted Intel nodes:")
print(next_impact)


# ============================================================
# 14. TOP EVIDENCE BY SIGNAL
# ============================================================

top_evidence_rows = []

for signal in signal_names:
    temp = kg_edges[kg_edges["target_signal"] == signal].copy()

    if temp.empty:
        continue

    temp = temp.sort_values("weight", ascending=False).head(10)

    for _, r in temp.iterrows():
        top_evidence_rows.append({
            "target_signal": signal,
            "target_business_node": r["target_business_node"],
            "source_company": r["source_company"],
            "source_company_name": r["source_company_name"],
            "weight": r["weight"],
            "publish_date": r["publish_date"],
            "evidence": r["evidence"]
        })

top_evidence = pd.DataFrame(top_evidence_rows)
top_evidence.to_csv(OUT / "top_evidence_by_signal.csv", index=False)

print("\nTop evidence by signal:")
print(top_evidence.head(30))


# ============================================================
# 15. SAVE GRAPH EDGE LIST AND NODE LIST
# ============================================================

nx_edges = []

for u, v, data in G.edges(data=True):
    nx_edges.append({
        "source": u,
        "target": v,
        "relation": data.get("relation"),
        "weight": data.get("weight")
    })

nx_edges_df = pd.DataFrame(nx_edges)
nx_edges_df.to_csv(OUT / "networkx_edge_list.csv", index=False)

nx_nodes = []

for node, data in G.nodes(data=True):
    nx_nodes.append({
        "node": node,
        "node_type": data.get("node_type"),
        "name": data.get("name")
    })

nx_nodes_df = pd.DataFrame(nx_nodes)
nx_nodes_df.to_csv(OUT / "networkx_node_list.csv", index=False)


# ============================================================
# DONE
# ============================================================

print("\nDONE.")
print(f"Output folder: {OUT.resolve()}")

print("\nGenerated files:")
for p in OUT.glob("*.csv"):
    print("-", p.name)