import re
import json
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASETS = [
    "kunhanw/earning_call_transcript_dataset_with_volatility_analysis",
    "hfmlsoc/sp500_dataset_earnings_calls",
]

OUT = Path("combined_hf_earnings_analysis")
PLOT_DIR = OUT / "plots"
OUT.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

MIN_TEXT_LENGTH = 500

TARGET_TICKER = "INTC"

MAX_CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40


# ============================================================
# 1. GENERAL HELPERS
# ============================================================

def safe_json_loads(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def normalize_text_for_hash(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_hash(text):
    clean = normalize_text_for_hash(text)
    return hashlib.md5(clean.encode("utf-8")).hexdigest()


def word_count(text):
    return len(str(text).split())


def first_existing(row, keys, default=None):
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]).strip() not in ["", "nan", "None", "NULL"]:
            return row[k]
    return default


def parse_date_any(x):
    if x is None or pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce", utc=True).tz_convert(None)
    except Exception:
        try:
            return pd.to_datetime(x, errors="coerce", utc=True).tz_localize(None)
        except Exception:
            return pd.NaT


def flatten_meta_if_exists(df):
    """
    For kunhanw dataset:
    columns = text, meta
    meta is dict-like and contains company, ticker-like name, publishOn, etc.
    """
    if "meta" in df.columns:
        meta_df = pd.json_normalize(df["meta"].apply(safe_json_loads))
        df = pd.concat([df.drop(columns=["meta"]), meta_df], axis=1)
    return df


# ============================================================
# 2. LOAD AND INSPECT DATASETS
# ============================================================

def load_dataset_to_pandas(dataset_name):
    print(f"\nLoading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    schema_info = {
        "dataset_name": dataset_name,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns}
    }

    safe_name = dataset_name.replace("/", "__")
    with open(OUT / f"schema__{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2, ensure_ascii=False)

    return df


# ============================================================
# 3. RECURSIVE TEXT EXTRACTION FOR UNKNOWN / NESTED FORMATS
# ============================================================

def extract_long_text_objects(obj, parent_key=""):
    """
    Recursively extract transcript-like text from nested dict/list structures.
    This is used because hfmlsoc dataset may have a different schema.
    """
    results = []

    if isinstance(obj, str):
        if len(obj) >= MIN_TEXT_LENGTH:
            results.append({
                "text": obj,
                "field_path": parent_key
            })

    elif isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            results.extend(extract_long_text_objects(v, new_key))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            results.extend(extract_long_text_objects(item, new_key))

    return results


def normalize_kunhanw(df, dataset_name):
    """
    Known format:
    - text
    - meta dict expanded to company, name, publishOn, title, volatility_analysis.*
    """
    df = flatten_meta_if_exists(df)

    rows = []

    for idx, row in df.iterrows():
        text = row.get("text", "")

        if not isinstance(text, str) or len(text) < MIN_TEXT_LENGTH:
            continue

        ticker = first_existing(
            row,
            ["name", "ticker", "symbol", "primaryTickers", "volatility_analysis.ticker"],
            default=""
        )

        company = first_existing(
            row,
            ["company", "volatility_analysis.company", "company_name", "name"],
            default=""
        )

        publish_date_raw = first_existing(
            row,
            ["publishOn", "publish_date", "date", "volatility_analysis.publish_date", "volatility_analysis.base_date"],
            default=None
        )

        title = first_existing(
            row,
            ["title"],
            default=""
        )

        rows.append({
            "source_dataset": dataset_name,
            "source_row_id": idx,
            "source_field_path": "text",
            "ticker": str(ticker).upper().strip(),
            "company": str(company).strip(),
            "title": str(title).strip(),
            "publish_date": parse_date_any(publish_date_raw),
            "text": text,
        })

    return pd.DataFrame(rows)


def normalize_generic_dataset(df, dataset_name):
    """
    Generic normalizer for hfmlsoc/sp500_dataset_earnings_calls.
    It does not assume exact columns.
    It tries:
    1. common transcript columns
    2. long string columns
    3. nested dict/list fields
    """
    df = flatten_meta_if_exists(df)

    common_text_cols = [
        "text", "content", "transcript", "call_transcript",
        "earnings_call", "earnings_calls", "earning_call",
        "transcripts", "full_text", "document", "body"
    ]

    ticker_cols = [
        "ticker", "symbol", "name", "primaryTickers",
        "stock", "company_ticker", "volatility_analysis.ticker"
    ]

    company_cols = [
        "company", "company_name", "name", "issuer",
        "registrant_name", "volatility_analysis.company"
    ]

    date_cols = [
        "publishOn", "publish_date", "date", "call_date",
        "earnings_date", "fiscal_date", "period_date",
        "volatility_analysis.publish_date", "volatility_analysis.base_date"
    ]

    title_cols = [
        "title", "headline", "document_title", "call_title"
    ]

    rows = []

    for idx, row in df.iterrows():
        base_ticker = first_existing(row, ticker_cols, default="")
        base_company = first_existing(row, company_cols, default="")
        base_date = first_existing(row, date_cols, default=None)
        base_title = first_existing(row, title_cols, default="")

        # 1. Direct common text columns
        found_direct = False

        for col in common_text_cols:
            if col in df.columns:
                val = row.get(col)

                if isinstance(val, str) and len(val) >= MIN_TEXT_LENGTH:
                    rows.append({
                        "source_dataset": dataset_name,
                        "source_row_id": idx,
                        "source_field_path": col,
                        "ticker": str(base_ticker).upper().strip(),
                        "company": str(base_company).strip(),
                        "title": str(base_title).strip(),
                        "publish_date": parse_date_any(base_date),
                        "text": val,
                    })
                    found_direct = True

                elif isinstance(val, (dict, list)):
                    extracted = extract_long_text_objects(val, parent_key=col)
                    for item in extracted:
                        rows.append({
                            "source_dataset": dataset_name,
                            "source_row_id": idx,
                            "source_field_path": item["field_path"],
                            "ticker": str(base_ticker).upper().strip(),
                            "company": str(base_company).strip(),
                            "title": str(base_title).strip(),
                            "publish_date": parse_date_any(base_date),
                            "text": item["text"],
                        })
                    if extracted:
                        found_direct = True

        if found_direct:
            continue

        # 2. Search all object columns for long strings or nested objects
        for col in df.columns:
            val = row.get(col)

            if isinstance(val, str) and len(val) >= MIN_TEXT_LENGTH:
                rows.append({
                    "source_dataset": dataset_name,
                    "source_row_id": idx,
                    "source_field_path": col,
                    "ticker": str(base_ticker).upper().strip(),
                    "company": str(base_company).strip(),
                    "title": str(base_title).strip(),
                    "publish_date": parse_date_any(base_date),
                    "text": val,
                })

            elif isinstance(val, (dict, list)):
                extracted = extract_long_text_objects(val, parent_key=col)
                for item in extracted:
                    rows.append({
                        "source_dataset": dataset_name,
                        "source_row_id": idx,
                        "source_field_path": item["field_path"],
                        "ticker": str(base_ticker).upper().strip(),
                        "company": str(base_company).strip(),
                        "title": str(base_title).strip(),
                        "publish_date": parse_date_any(base_date),
                        "text": item["text"],
                    })

    return pd.DataFrame(rows)


def normalize_dataset(df, dataset_name):
    if dataset_name == "kunhanw/earning_call_transcript_dataset_with_volatility_analysis":
        norm = normalize_kunhanw(df, dataset_name)
    else:
        norm = normalize_generic_dataset(df, dataset_name)

    if norm.empty:
        print(f"WARNING: No transcript text extracted from {dataset_name}")
        return norm

    norm["ticker"] = norm["ticker"].replace(["NONE", "NAN", "NULL", "NA"], "")
    norm["company"] = norm["company"].replace(["None", "nan", "NULL"], "")
    norm["word_count"] = norm["text"].apply(word_count)
    norm["content_hash"] = norm["text"].apply(text_hash)

    return norm


# ============================================================
# 4. COMBINE DATASETS
# ============================================================

normalized_frames = []

for dataset_name in DATASETS:
    raw_df = load_dataset_to_pandas(dataset_name)
    norm_df = normalize_dataset(raw_df, dataset_name)

    print(f"Extracted transcript rows from {dataset_name}: {len(norm_df)}")

    if not norm_df.empty:
        print(norm_df[[
            "source_dataset", "ticker", "company", "publish_date",
            "word_count", "source_field_path"
        ]].head(10))

    normalized_frames.append(norm_df)

combined_df = pd.concat(
    [x for x in normalized_frames if not x.empty],
    ignore_index=True
)

if combined_df.empty:
    raise RuntimeError("No transcripts extracted from either dataset.")

combined_df.to_csv(OUT / "combined_raw_normalized_before_dedup.csv", index=False)

print("\nCombined before dedup:", len(combined_df))
print("Rows by source before dedup:")
print(combined_df["source_dataset"].value_counts())


# ============================================================
# 5. DEDUPLICATION
# ============================================================

before_dedup = len(combined_df)

# Exact content duplicate removal
combined_df = combined_df.sort_values(
    ["content_hash", "word_count"],
    ascending=[True, False]
).drop_duplicates(
    subset=["content_hash"],
    keep="first"
).copy()

after_exact = len(combined_df)

# Secondary duplicate rule:
# Same ticker + same publish date + very similar title often means same call.
# Keep the longer transcript.
combined_df["date_only"] = combined_df["publish_date"].dt.date.astype(str)

combined_df["title_norm"] = (
    combined_df["title"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

combined_df = combined_df.sort_values(
    ["ticker", "date_only", "title_norm", "word_count"],
    ascending=[True, True, True, False]
).drop_duplicates(
    subset=["ticker", "date_only", "title_norm"],
    keep="first"
).copy()

after_secondary = len(combined_df)

dedup_summary = pd.DataFrame([
    {
        "stage": "before_dedup",
        "rows": before_dedup
    },
    {
        "stage": "after_exact_content_hash_dedup",
        "rows": after_exact
    },
    {
        "stage": "after_ticker_date_title_dedup",
        "rows": after_secondary
    },
    {
        "stage": "duplicates_removed_total",
        "rows": before_dedup - after_secondary
    }
])

dedup_summary.to_csv(OUT / "dedup_summary.csv", index=False)

combined_df = combined_df.reset_index(drop=True)
combined_df["doc_id"] = combined_df.index

combined_df.to_csv(OUT / "combined_transcripts_deduplicated.csv", index=False)

print("\nDedup summary:")
print(dedup_summary)

print("\nRows by source after dedup:")
print(combined_df["source_dataset"].value_counts())


# ============================================================
# 6. COVERAGE ANALYSIS
# ============================================================

coverage_df = combined_df.dropna(subset=["publish_date"]).copy()

coverage_df["year"] = coverage_df["publish_date"].dt.year
coverage_df["quarter"] = coverage_df["publish_date"].dt.quarter
coverage_df["year_quarter"] = coverage_df["publish_date"].dt.to_period("Q").astype(str)
coverage_df["quarter_start"] = coverage_df["publish_date"].dt.to_period("Q").dt.start_time

yearly_counts = (
    coverage_df.groupby("year")
    .agg(
        transcript_count=("doc_id", "count"),
        unique_tickers=("ticker", "nunique"),
        unique_companies=("company", "nunique"),
        avg_word_count=("word_count", "mean"),
        total_words=("word_count", "sum")
    )
    .reset_index()
    .sort_values("year")
)

quarterly_counts = (
    coverage_df.groupby(["quarter_start", "year_quarter"])
    .agg(
        transcript_count=("doc_id", "count"),
        unique_tickers=("ticker", "nunique"),
        unique_companies=("company", "nunique"),
        avg_word_count=("word_count", "mean"),
        median_word_count=("word_count", "median"),
        total_words=("word_count", "sum")
    )
    .reset_index()
    .sort_values("quarter_start")
)

quarterly_counts["transcripts_per_ticker"] = (
    quarterly_counts["transcript_count"] /
    quarterly_counts["unique_tickers"].replace(0, pd.NA)
)

ticker_coverage = (
    coverage_df.groupby(["ticker", "company"], dropna=False)
    .agg(
        transcript_count=("doc_id", "count"),
        first_date=("publish_date", "min"),
        last_date=("publish_date", "max"),
        unique_quarters=("year_quarter", "nunique"),
        avg_word_count=("word_count", "mean"),
        total_words=("word_count", "sum")
    )
    .reset_index()
    .sort_values(["transcript_count", "unique_quarters"], ascending=False)
)

quarter_ticker_matrix = (
    coverage_df.pivot_table(
        index="year_quarter",
        columns="ticker",
        values="doc_id",
        aggfunc="count",
        fill_value=0
    )
)

yearly_counts.to_csv(OUT / "yearly_transcript_coverage.csv", index=False)
quarterly_counts.to_csv(OUT / "quarterly_transcript_coverage.csv", index=False)
ticker_coverage.to_csv(OUT / "ticker_level_coverage.csv", index=False)
quarter_ticker_matrix.to_csv(OUT / "quarter_ticker_transcript_matrix.csv")

print("\nYearly coverage:")
print(yearly_counts)

print("\nQuarterly coverage tail:")
print(quarterly_counts.tail(20))

print("\nTop tickers by transcript count:")
print(ticker_coverage.head(30))


# ============================================================
# 7. INFORMATION DENSITY ANALYSIS
# ============================================================

supply_chain_terms = {
    "supply_general": [
        "supply chain", "supply", "supplier", "suppliers", "vendor", "vendors",
        "sourcing", "procurement", "purchase order", "orders", "backlog"
    ],
    "manufacturing": [
        "manufacturing", "production", "factory", "plant", "fab", "fabs",
        "wafer", "foundry", "assembly", "packaging", "advanced packaging",
        "process node", "capacity", "utilization", "yield"
    ],
    "inventory": [
        "inventory", "inventories", "stock", "channel inventory",
        "destocking", "restocking", "inventory correction"
    ],
    "logistics": [
        "logistics", "shipment", "shipments", "shipping", "delivery",
        "lead time", "lead times", "freight", "transportation"
    ],
    "constraints": [
        "shortage", "shortages", "constraint", "constraints", "constrained",
        "bottleneck", "bottlenecks", "delay", "delays", "allocation"
    ],
    "cost_pricing": [
        "input cost", "cost pressure", "pricing pressure", "raw material",
        "component cost", "memory pricing", "freight cost", "tariff", "tariffs"
    ],
    "semiconductor_specific": [
        "semiconductor", "chip", "chips", "CPU", "GPU", "processor",
        "memory", "DRAM", "NAND", "HBM", "substrate", "CoWoS",
        "lithography", "EUV", "equipment"
    ],
    "capex_infrastructure": [
        "capex", "capital expenditure", "capital expenditures",
        "data center", "datacenter", "infrastructure", "buildout",
        "capacity expansion"
    ]
}

company_entities = {
    "Intel": ["Intel", "INTC", "Xeon", "Intel Foundry"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "NVIDIA": ["NVIDIA", "NVDA"],
    "Dell": ["Dell", "Dell Technologies", "DELL"],
    "HP": ["HP", "HP Inc", "HPQ", "Hewlett Packard"],
    "Lenovo": ["Lenovo", "LNVGY"],
    "Microsoft": ["Microsoft", "MSFT", "Azure"],
    "Amazon": ["Amazon", "AMZN", "AWS"],
    "Google": ["Google", "Alphabet", "GOOG", "GOOGL", "Google Cloud"],
    "Meta": ["Meta", "Facebook", "META"],
    "Oracle": ["Oracle", "ORCL"],
    "HPE": ["HPE", "Hewlett Packard Enterprise"],
    "Supermicro": ["Super Micro", "Supermicro", "SMCI"],
    "TSMC": ["TSMC", "Taiwan Semiconductor", "TSM"],
    "Samsung": ["Samsung"],
    "SK Hynix": ["SK Hynix", "Hynix"],
    "Micron": ["Micron", "MU"],
    "Broadcom": ["Broadcom", "AVGO"],
    "Qualcomm": ["Qualcomm", "QCOM"],
    "ASML": ["ASML"],
    "Applied Materials": ["Applied Materials", "AMAT"],
    "Lam Research": ["Lam Research", "LRCX"],
    "KLA": ["KLA", "KLAC"],
    "Cisco": ["Cisco", "CSCO"],
    "Arista": ["Arista", "ANET"],
    "Western Digital": ["Western Digital", "WDC"],
    "Seagate": ["Seagate", "STX"],
    "NetApp": ["NetApp", "NTAP"],
    "Salesforce": ["Salesforce", "CRM"],
    "SAP": ["SAP"],
    "Baidu": ["Baidu", "BIDU"],
    "Alibaba": ["Alibaba", "BABA"],
    "Tencent": ["Tencent"],
}

all_supply_terms = []
for group, terms in supply_chain_terms.items():
    all_supply_terms.extend(terms)
all_supply_terms = sorted(set(all_supply_terms), key=len, reverse=True)


def count_term_occurrences(text, terms):
    text = str(text)
    counts = {}

    for term in terms:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        counts[term] = len(matches)

    return counts


def count_supply_terms_by_group(text):
    result = {}
    total_count = 0
    unique_terms = set()

    for group, terms in supply_chain_terms.items():
        counts = count_term_occurrences(text, terms)
        group_total = sum(counts.values())

        result[f"{group}_count"] = group_total
        total_count += group_total

        for term, c in counts.items():
            if c > 0:
                unique_terms.add(term)

    result["supply_chain_term_total_count"] = total_count
    result["supply_chain_unique_term_count"] = len(unique_terms)
    result["supply_chain_unique_terms"] = ", ".join(sorted(unique_terms))

    return result


def count_company_entities(text):
    text = str(text)
    matched_companies = {}
    total_mentions = 0

    for canonical_name, aliases in company_entities.items():
        alias_count = 0

        for alias in aliases:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"
            alias_count += len(re.findall(pattern, text, flags=re.IGNORECASE))

        if alias_count > 0:
            matched_companies[canonical_name] = alias_count
            total_mentions += alias_count

    return {
        "company_entity_unique_count": len(matched_companies),
        "company_entity_total_mentions": total_mentions,
        "matched_company_entities": ", ".join(sorted(matched_companies.keys())),
        "matched_company_entity_counts": json.dumps(matched_companies, ensure_ascii=False)
    }


density_records = []

for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Information density"):
    text = row["text"]
    wc = int(row["word_count"])

    supply_stats = count_supply_terms_by_group(text)
    company_stats = count_company_entities(text)

    record = {
        "doc_id": row["doc_id"],
        "source_dataset": row["source_dataset"],
        "ticker": row["ticker"],
        "company": row["company"],
        "title": row["title"],
        "publish_date": row["publish_date"],
        "word_count": wc,
        **supply_stats,
        **company_stats
    }

    if wc > 0:
        record["supply_chain_terms_per_1000_words"] = record["supply_chain_term_total_count"] / wc * 1000
        record["unique_supply_terms_per_1000_words"] = record["supply_chain_unique_term_count"] / wc * 1000
        record["company_mentions_per_1000_words"] = record["company_entity_total_mentions"] / wc * 1000
        record["unique_companies_per_1000_words"] = record["company_entity_unique_count"] / wc * 1000
    else:
        record["supply_chain_terms_per_1000_words"] = 0
        record["unique_supply_terms_per_1000_words"] = 0
        record["company_mentions_per_1000_words"] = 0
        record["unique_companies_per_1000_words"] = 0

    record["information_density_score"] = (
        record["supply_chain_terms_per_1000_words"] * 0.5
        + record["unique_supply_terms_per_1000_words"] * 2.0
        + record["company_mentions_per_1000_words"] * 0.3
        + record["unique_companies_per_1000_words"] * 3.0
    )

    density_records.append(record)

density_df = pd.DataFrame(density_records)
density_df = density_df.sort_values("information_density_score", ascending=False)

density_df.to_csv(OUT / "transcript_information_density.csv", index=False)

summary_stats = density_df[[
    "word_count",
    "supply_chain_term_total_count",
    "supply_chain_unique_term_count",
    "supply_chain_terms_per_1000_words",
    "unique_supply_terms_per_1000_words",
    "company_entity_unique_count",
    "company_entity_total_mentions",
    "company_mentions_per_1000_words",
    "unique_companies_per_1000_words",
    "information_density_score"
]].describe()

summary_stats.to_csv(OUT / "information_density_summary_stats.csv")

company_density_summary = (
    density_df.groupby(["ticker", "company"], dropna=False)
    .agg(
        transcript_count=("doc_id", "count"),
        avg_word_count=("word_count", "mean"),
        avg_supply_chain_terms=("supply_chain_term_total_count", "mean"),
        avg_supply_chain_terms_per_1000_words=("supply_chain_terms_per_1000_words", "mean"),
        avg_unique_supply_terms=("supply_chain_unique_term_count", "mean"),
        avg_company_entity_unique_count=("company_entity_unique_count", "mean"),
        avg_company_mentions_per_1000_words=("company_mentions_per_1000_words", "mean"),
        avg_information_density_score=("information_density_score", "mean")
    )
    .reset_index()
    .sort_values("avg_information_density_score", ascending=False)
)

company_density_summary.to_csv(OUT / "company_level_information_density.csv", index=False)

term_global_rows = []

for term in tqdm(all_supply_terms, desc="Global supply term frequency"):
    total_count = 0
    transcript_count = 0

    for text in combined_df["text"]:
        c = count_term_occurrences(text, [term])[term]
        total_count += c
        if c > 0:
            transcript_count += 1

    term_global_rows.append({
        "term": term,
        "total_occurrences": total_count,
        "transcript_count_with_term": transcript_count,
        "avg_occurrences_per_transcript": total_count / len(combined_df),
        "transcript_coverage_pct": transcript_count / len(combined_df) * 100
    })

term_freq_df = pd.DataFrame(term_global_rows).sort_values("total_occurrences", ascending=False)
term_freq_df.to_csv(OUT / "supply_chain_term_global_frequency.csv", index=False)

company_global_rows = []

for canonical_name, aliases in tqdm(company_entities.items(), desc="Global company entity frequency"):
    total_mentions = 0
    transcript_count = 0

    for text in combined_df["text"]:
        alias_total = 0

        for alias in aliases:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"
            alias_total += len(re.findall(pattern, str(text), flags=re.IGNORECASE))

        total_mentions += alias_total

        if alias_total > 0:
            transcript_count += 1

    company_global_rows.append({
        "company_entity": canonical_name,
        "total_mentions": total_mentions,
        "transcript_count_with_company": transcript_count,
        "avg_mentions_per_transcript": total_mentions / len(combined_df),
        "transcript_coverage_pct": transcript_count / len(combined_df) * 100
    })

company_freq_df = pd.DataFrame(company_global_rows).sort_values("total_mentions", ascending=False)
company_freq_df.to_csv(OUT / "company_entity_global_frequency.csv", index=False)

print("\nInformation density summary:")
print(summary_stats)

print("\nTop supply-chain terms:")
print(term_freq_df.head(30))

print("\nTop company entities:")
print(company_freq_df.head(30))


# ============================================================
# 8. INTEL-RELATED KG PROTOTYPE
# ============================================================

intel_keywords = [
    "Intel", "INTC", "Xeon", "Core Ultra", "Lunar Lake", "Meteor Lake",
    "Arrow Lake", "Gaudi", "Intel Foundry", "18A", "20A",
    "PC", "PCs", "notebook", "desktop", "laptop",
    "commercial PC", "consumer PC", "AI PC",
    "Windows", "Windows refresh", "enterprise refresh",
    "server", "servers", "CPU", "CPUs", "processor", "processors",
    "data center", "datacenter", "cloud", "enterprise infrastructure",
    "inference", "accelerator", "compute",
    "foundry", "semiconductor manufacturing", "wafer", "fab",
    "advanced packaging", "process node", "chip manufacturing",
    "inventory", "component", "shortage", "memory", "pricing",
    "margin pressure", "gross margin", "capex", "capital expenditure"
]

keyword_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in intel_keywords) + r")\b",
    flags=re.IGNORECASE
)


def split_text_into_chunks(text, max_words=220, overlap=40):
    words = str(text).split()

    if len(words) <= max_words:
        return [str(text)]

    chunks = []
    step = max_words - overlap

    for start in range(0, len(words), step):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

    return chunks


def has_any(text, terms):
    t = str(text).lower()
    return any(term.lower() in t for term in terms)


def count_any(text, terms):
    t = str(text).lower()
    return sum(t.count(term.lower()) for term in terms)


def extract_intel_aware_signals(text):
    t = str(text).lower()
    signals = {}

    pc_terms = ["pc", "pcs", "notebook", "desktop", "laptop", "commercial pc", "consumer pc", "ai pc"]
    pc_context_terms = ["commercial", "consumer", "notebook", "desktop", "laptop", "windows", "ai pc", "refresh", "device", "devices"]

    signals["pc_demand"] = (
        count_any(t, pc_terms)
        if has_any(t, pc_terms) and has_any(t, pc_context_terms)
        else 0
    )

    server_terms = ["server", "servers", "data center", "datacenter", "enterprise infrastructure"]
    cpu_terms = ["cpu", "cpus", "xeon", "processor", "processors", "compute"]

    signals["server_cpu_demand"] = (
        count_any(t, server_terms + cpu_terms)
        if has_any(t, server_terms) and has_any(t, cpu_terms)
        else 0
    )

    cloud_terms = ["cloud", "data center", "datacenter"]
    capacity_terms = ["capacity", "capex", "capital expenditure", "infrastructure", "buildout", "build-out", "investment"]

    signals["cloud_capacity"] = (
        count_any(t, cloud_terms + capacity_terms)
        if has_any(t, cloud_terms) and has_any(t, capacity_terms)
        else 0
    )

    foundry_terms = ["foundry", "fab", "wafer", "advanced packaging", "process node", "18a", "20a", "semiconductor manufacturing"]

    signals["foundry_manufacturing"] = (
        count_any(t, foundry_terms)
        if has_any(t, foundry_terms)
        else 0
    )

    pressure_terms = ["shortage", "constraint", "constrained", "component", "memory", "inventory", "supply"]
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

    margin_terms = ["margin", "gross margin", "pricing", "cost pressure", "cost", "price"]

    signals["margin_pressure"] = (
        count_any(t, margin_terms)
        if has_any(t, margin_terms) and has_any(t, semiconductor_context_terms)
        else 0
    )

    return signals


chunk_rows = []

kg_source_df = combined_df[
    combined_df["ticker"].astype(str).str.upper() != TARGET_TICKER
].copy()

for _, row in tqdm(kg_source_df.iterrows(), total=len(kg_source_df), desc="Intel chunk extraction"):
    chunks = split_text_into_chunks(
        row["text"],
        max_words=MAX_CHUNK_WORDS,
        overlap=CHUNK_OVERLAP_WORDS
    )

    for chunk_id, chunk in enumerate(chunks):
        matches = keyword_pattern.findall(chunk)

        if matches:
            signals = extract_intel_aware_signals(chunk)
            signal_total = sum(signals.values())

            if signal_total <= 0:
                continue

            item = {
                "doc_id": row["doc_id"],
                "chunk_id": chunk_id,
                "source_dataset": row["source_dataset"],
                "ticker": row["ticker"],
                "company": row["company"],
                "title": row["title"],
                "publish_date": row["publish_date"],
                "matched_keywords": sorted(list(set([m.lower() for m in matches]))),
                "matched_keyword_count": len(matches),
                "signal_total_count": signal_total,
                "chunk_text": chunk
            }

            for k, v in signals.items():
                item[k + "_count"] = v

            chunk_rows.append(item)

intel_chunks = pd.DataFrame(chunk_rows)

if not intel_chunks.empty:
    intel_chunks.to_csv(OUT / "intel_relevant_chunks_combined.csv", index=False)
else:
    print("No Intel-relevant chunks found after strict filtering.")

important_company_weights = {
    "DELL": 3.0,
    "HPQ": 3.0,
    "HPE": 2.5,
    "SMCI": 2.5,
    "LEN": 2.5,
    "LNVGY": 2.5,
    "AMD": 2.0,
    "MU": 2.5,
    "NVDA": 2.0,
    "ASML": 2.5,
    "AMAT": 2.5,
    "KLAC": 2.5,
    "LRCX": 2.5,
    "TSM": 2.5,
    "TSMC": 2.5,
    "MSFT": 2.5,
    "AMZN": 2.5,
    "GOOGL": 2.5,
    "GOOG": 2.5,
    "META": 2.0,
    "ORCL": 2.0,
    "CSCO": 2.0,
    "ANET": 2.0,
    "CRM": 1.5,
    "WDC": 1.8,
    "STX": 1.8,
    "NTAP": 1.8,
    "CIEN": 1.8,
    "LITE": 1.8,
    "EQIX": 1.8,
}


def company_weight(ticker):
    return important_company_weights.get(str(ticker).upper(), 1.0)


signal_names = [
    "pc_demand",
    "server_cpu_demand",
    "cloud_capacity",
    "foundry_manufacturing",
    "supply_pressure",
    "margin_pressure"
]

business_target_map = {
    "pc_demand": "Intel_CCG",
    "server_cpu_demand": "Intel_DCAI",
    "cloud_capacity": "Intel_DCAI",
    "foundry_manufacturing": "Intel_Foundry",
    "supply_pressure": "Intel_CCG_or_DCAI",
    "margin_pressure": "Intel_Margin"
}

edges = []

if not intel_chunks.empty:
    for _, row in intel_chunks.iterrows():
        source = str(row["ticker"]).upper().strip()
        if not source:
            source = str(row["company"]).upper().strip()

        w_company = company_weight(source)

        for signal in signal_names:
            raw_count = int(row.get(signal + "_count", 0))

            if raw_count > 0:
                edges.append({
                    "source_company": source,
                    "source_company_name": row["company"],
                    "source_dataset": row["source_dataset"],
                    "relation": "mentions_signal",
                    "target_signal": signal,
                    "target_business_node": business_target_map[signal],
                    "raw_weight": raw_count,
                    "company_weight": w_company,
                    "weight": raw_count * w_company,
                    "publish_date": row["publish_date"],
                    "matched_keywords": json.dumps(row["matched_keywords"], ensure_ascii=False),
                    "evidence": row["chunk_text"][:800]
                })

kg_edges = pd.DataFrame(edges)

if not kg_edges.empty:
    kg_edges.to_csv(OUT / "kg_edges_combined_strict_weighted.csv", index=False)

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

    company_signal_summary.to_csv(OUT / "company_signal_summary_combined.csv", index=False)

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
    next_impact.to_csv(OUT / "next_impacted_intel_nodes_combined.csv", index=False)

    G = nx.DiGraph()

    for _, e in kg_edges.iterrows():
        company_node = f"COMPANY::{e['source_company']}"
        signal_node = f"SIGNAL::{e['target_signal']}"
        business_node = f"BUSINESS::{e['target_business_node']}"

        G.add_node(company_node, node_type="company", name=e["source_company_name"])
        G.add_node(signal_node, node_type="signal")
        G.add_node(business_node, node_type="intel_business_node")

        if G.has_edge(company_node, signal_node):
            G[company_node][signal_node]["weight"] += float(e["weight"])
        else:
            G.add_edge(company_node, signal_node, relation="mentions", weight=float(e["weight"]))

        if G.has_edge(signal_node, business_node):
            G[signal_node][business_node]["weight"] += float(e["weight"])
        else:
            G.add_edge(signal_node, business_node, relation="affects", weight=float(e["weight"]))

    pagerank = nx.pagerank(G, weight="weight")

    pr_df = pd.DataFrame([
        {
            "node": node,
            "pagerank": score,
            "node_type": G.nodes[node].get("node_type")
        }
        for node, score in pagerank.items()
    ]).sort_values("pagerank", ascending=False)

    pr_df.to_csv(OUT / "kg_pagerank_combined.csv", index=False)

    print("\nKG edges:", len(kg_edges))
    print("\nNext impacted Intel nodes:")
    print(next_impact)

else:
    print("No KG edges generated.")


# ============================================================
# 9. PLOTS
# ============================================================

plt.figure(figsize=(10, 6))
plt.bar(yearly_counts["year"].astype(str), yearly_counts["transcript_count"])
plt.title("Combined Dataset: Transcripts per Year")
plt.xlabel("Year")
plt.ylabel("Transcript Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_transcripts_per_year.png", dpi=300)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(quarterly_counts["year_quarter"], quarterly_counts["transcript_count"], marker="o")
plt.title("Combined Dataset: Transcripts per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Transcript Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_transcripts_per_quarter.png", dpi=300)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(quarterly_counts["year_quarter"], quarterly_counts["unique_tickers"], marker="o")
plt.title("Combined Dataset: Unique Tickers per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Unique Tickers")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_unique_tickers_per_quarter.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(density_df["supply_chain_terms_per_1000_words"].dropna(), bins=40)
plt.title("Supply Chain Terms per 1,000 Words")
plt.xlabel("Supply Chain Terms per 1,000 Words")
plt.ylabel("Number of Transcripts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_supply_chain_density_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(density_df["company_mentions_per_1000_words"].dropna(), bins=40)
plt.title("Company Mentions per 1,000 Words")
plt.xlabel("Company Mentions per 1,000 Words")
plt.ylabel("Number of Transcripts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_company_mentions_density_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(
    density_df["supply_chain_terms_per_1000_words"],
    density_df["company_mentions_per_1000_words"],
    alpha=0.45
)
plt.title("Transcript Information Density")
plt.xlabel("Supply Chain Terms per 1,000 Words")
plt.ylabel("Company Mentions per 1,000 Words")
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_supply_chain_vs_company_mentions.png", dpi=300)
plt.show()

top_terms = term_freq_df.head(30)

plt.figure(figsize=(12, 9))
plt.barh(top_terms["term"][::-1], top_terms["total_occurrences"][::-1])
plt.title("Top Supply Chain Terms by Total Occurrences")
plt.xlabel("Total Occurrences")
plt.ylabel("Supply Chain Term")
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_top_supply_chain_terms.png", dpi=300)
plt.show()

top_companies = company_density_summary.head(30).copy()
top_companies["label"] = (
    top_companies["ticker"].astype(str)
    + " | "
    + top_companies["company"].astype(str).str.slice(0, 35)
)

plt.figure(figsize=(12, 9))
plt.barh(
    top_companies["label"][::-1],
    top_companies["avg_information_density_score"][::-1]
)
plt.title("Top Companies by Average Information Density")
plt.xlabel("Average Information Density Score")
plt.ylabel("Company")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_top_companies_information_density.png", dpi=300)
plt.show()

if not kg_edges.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(
        next_impact["target_business_node"] + " | " + next_impact["target_signal"],
        next_impact["impact_score"]
    )
    plt.title("Likely Next Impacted Intel Business Nodes")
    plt.xlabel("Impact Score")
    plt.ylabel("Intel Business Node / Signal")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "09_next_impacted_intel_nodes.png", dpi=300)
    plt.show()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

final_summary = {
    "combined_rows_before_dedup": int(before_dedup),
    "combined_rows_after_dedup": int(after_secondary),
    "duplicates_removed": int(before_dedup - after_secondary),
    "unique_tickers": int(combined_df["ticker"].nunique()),
    "unique_companies": int(combined_df["company"].nunique()),
    "total_words": int(combined_df["word_count"].sum()),
    "avg_words_per_transcript": float(combined_df["word_count"].mean()),
    "date_min": str(coverage_df["publish_date"].min()) if not coverage_df.empty else None,
    "date_max": str(coverage_df["publish_date"].max()) if not coverage_df.empty else None,
    "quarters_covered": int(coverage_df["year_quarter"].nunique()) if not coverage_df.empty else 0,
    "avg_supply_chain_terms_per_transcript": float(density_df["supply_chain_term_total_count"].mean()),
    "avg_supply_chain_terms_per_1000_words": float(density_df["supply_chain_terms_per_1000_words"].mean()),
    "avg_company_mentions_per_transcript": float(density_df["company_entity_total_mentions"].mean()),
    "avg_company_mentions_per_1000_words": float(density_df["company_mentions_per_1000_words"].mean()),
    "kg_edges": int(len(kg_edges)) if not kg_edges.empty else 0
}

with open(OUT / "final_summary.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=2, ensure_ascii=False)

print("\n================ FINAL SUMMARY ================")
for k, v in final_summary.items():
    print(f"{k}: {v}")

print("\nDONE.")
print("Output folder:", OUT.resolve())

print("\nGenerated files:")
for p in OUT.glob("*.csv"):
    print("-", p.name)

print("\nGenerated plots:")
for p in PLOT_DIR.glob("*.png"):
    print("-", p.name)