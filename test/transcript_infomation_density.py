import re
import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "kunhanw/earning_call_transcript_dataset_with_volatility_analysis"

OUT = Path("transcript_information_density")
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD DATASET
# ============================================================

ds = load_dataset(DATASET_NAME, split="train")
df = ds.to_pandas()

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

df["text"] = df["text"].astype(str)

if "company" not in df.columns:
    df["company"] = ""

if "name" in df.columns:
    df["ticker"] = df["name"].astype(str).str.upper()
elif "volatility_analysis.ticker" in df.columns:
    df["ticker"] = df["volatility_analysis.ticker"].astype(str).str.upper()
else:
    df["ticker"] = ""

if "publishOn" in df.columns:
    df["publish_date"] = pd.to_datetime(df["publishOn"], errors="coerce", utc=True).dt.tz_convert(None)
else:
    df["publish_date"] = pd.NaT

# ============================================================
# 2. DEFINE SUPPLY CHAIN VOCABULARY
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

# flatten term list
all_supply_terms = []
for group, terms in supply_chain_terms.items():
    all_supply_terms.extend(terms)

# Sort longer terms first to avoid "supply chain" being swallowed by "supply"
all_supply_terms = sorted(set(all_supply_terms), key=len, reverse=True)

# ============================================================
# 3. DEFINE COMPANY NAME DICTIONARY
# ============================================================
# 这里先用一个可扩展 company dictionary。
# 后面可以换成全市场 ticker/company list。

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

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def count_term_occurrences(text, terms):
    """
    Count occurrences of terms in text.
    Uses word boundary for simple alphanumeric terms.
    For multi-word phrases, uses case-insensitive phrase search.
    """
    text = str(text)
    counts = {}

    for term in terms:
        term_clean = term.strip()
        if not term_clean:
            continue

        # For terms with symbols or spaces, use escaped phrase search
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term_clean) + r"(?![A-Za-z0-9])"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        counts[term_clean] = len(matches)

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
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            alias_count += len(matches)

        if alias_count > 0:
            matched_companies[canonical_name] = alias_count
            total_mentions += alias_count

    return {
        "company_entity_unique_count": len(matched_companies),
        "company_entity_total_mentions": total_mentions,
        "matched_company_entities": ", ".join(sorted(matched_companies.keys())),
        "matched_company_entity_counts": json.dumps(matched_companies, ensure_ascii=False)
    }


def word_count(text):
    return len(str(text).split())


# ============================================================
# 5. TRANSCRIPT-LEVEL INFORMATION DENSITY
# ============================================================

records = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating transcript density"):
    text = row["text"]

    wc = word_count(text)
    supply_stats = count_supply_terms_by_group(text)
    company_stats = count_company_entities(text)

    record = {
        "doc_id": idx,
        "ticker": row.get("ticker"),
        "company": row.get("company"),
        "title": row.get("title"),
        "publish_date": row.get("publish_date"),
        "word_count": wc,
        **supply_stats,
        **company_stats
    }

    # Normalize per 1,000 words
    if wc > 0:
        record["supply_chain_terms_per_1000_words"] = (
            record["supply_chain_term_total_count"] / wc * 1000
        )
        record["unique_supply_terms_per_1000_words"] = (
            record["supply_chain_unique_term_count"] / wc * 1000
        )
        record["company_mentions_per_1000_words"] = (
            record["company_entity_total_mentions"] / wc * 1000
        )
        record["unique_companies_per_1000_words"] = (
            record["company_entity_unique_count"] / wc * 1000
        )
    else:
        record["supply_chain_terms_per_1000_words"] = 0
        record["unique_supply_terms_per_1000_words"] = 0
        record["company_mentions_per_1000_words"] = 0
        record["unique_companies_per_1000_words"] = 0

    # Composite information density score
    record["information_density_score"] = (
        record["supply_chain_terms_per_1000_words"] * 0.5
        + record["unique_supply_terms_per_1000_words"] * 2.0
        + record["company_mentions_per_1000_words"] * 0.3
        + record["unique_companies_per_1000_words"] * 3.0
    )

    records.append(record)

density_df = pd.DataFrame(records)

density_df = density_df.sort_values(
    "information_density_score",
    ascending=False
).reset_index(drop=True)

density_df.to_csv(OUT / "transcript_information_density.csv", index=False)

print("\nTop transcripts by information density:")
print(density_df[[
    "ticker",
    "company",
    "publish_date",
    "word_count",
    "supply_chain_term_total_count",
    "supply_chain_terms_per_1000_words",
    "company_entity_unique_count",
    "company_mentions_per_1000_words",
    "information_density_score",
    "matched_company_entities",
    "supply_chain_unique_terms"
]].head(30))


# ============================================================
# 6. OVERALL DESCRIPTIVE STATISTICS
# ============================================================

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

print("\nSummary statistics:")
print(summary_stats)


# ============================================================
# 7. SUPPLY CHAIN TERM FREQUENCY ACROSS ALL TRANSCRIPTS
# ============================================================

term_global_rows = []

for term in all_supply_terms:
    total_count = 0
    transcript_count = 0

    for text in df["text"]:
        c = count_term_occurrences(text, [term])[term]
        total_count += c
        if c > 0:
            transcript_count += 1

    term_global_rows.append({
        "term": term,
        "total_occurrences": total_count,
        "transcript_count_with_term": transcript_count,
        "avg_occurrences_per_transcript": total_count / len(df),
        "transcript_coverage_pct": transcript_count / len(df) * 100
    })

term_freq_df = pd.DataFrame(term_global_rows).sort_values(
    "total_occurrences",
    ascending=False
)

term_freq_df.to_csv(OUT / "supply_chain_term_global_frequency.csv", index=False)

print("\nTop supply chain terms:")
print(term_freq_df.head(40))


# ============================================================
# 8. COMPANY ENTITY FREQUENCY ACROSS ALL TRANSCRIPTS
# ============================================================

company_global_rows = []

for canonical_name, aliases in company_entities.items():
    total_mentions = 0
    transcript_count = 0

    for text in df["text"]:
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
        "avg_mentions_per_transcript": total_mentions / len(df),
        "transcript_coverage_pct": transcript_count / len(df) * 100
    })

company_freq_df = pd.DataFrame(company_global_rows).sort_values(
    "total_mentions",
    ascending=False
)

company_freq_df.to_csv(OUT / "company_entity_global_frequency.csv", index=False)

print("\nTop company entities:")
print(company_freq_df.head(40))


# ============================================================
# 9. COMPANY / TICKER LEVEL AVERAGE INFORMATION DENSITY
# ============================================================

company_density_summary = (
    density_df
    .groupby(["ticker", "company"], dropna=False)
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

print("\nTop companies by average transcript information density:")
print(company_density_summary.head(40))


# ============================================================
# DONE
# ============================================================

print("\nDONE.")
print("Output files:")
for p in OUT.glob("*.csv"):
    print("-", p.name)