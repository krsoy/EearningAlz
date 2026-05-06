import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
import os

# Option 1: read from a plain text file, e.g. api_key.txt
key_path = Path("api_key.txt")

if key_path.exists():
    FMP_API_KEY = key_path.read_text(encoding="utf-8").strip()
else:
    # Option 2: fallback to environment variable
    FMP_API_KEY = os.getenv("FMP_API_KEY")

if not FMP_API_KEY:
    raise ValueError("FMP API key not found. Please create api_key.txt or set FMP_API_KEY environment variable.")
BASE = "https://financialmodelingprep.com/stable"

OUT = Path("intel_earnings_project")
OUT.mkdir(exist_ok=True)

TARGET_TICKER = "INTC"
TARGET_QUARTER = "2026Q1"
TARGET_EARNINGS_DATE = pd.Timestamp("2026-04-23")
LOOKBACK_DAYS = 180
START_DATE = TARGET_EARNINGS_DATE - pd.Timedelta(days=LOOKBACK_DAYS)

# 先放第一梯队，后面你可以直接扩展到全市场 ticker list
UNIVERSE = pd.DataFrame([
    {"symbol": "DELL", "company": "Dell Technologies", "role": "pc_oem_server_vendor", "intel_group": "CCG_DCAI"},
    {"symbol": "HPQ", "company": "HP Inc.", "role": "pc_oem", "intel_group": "CCG"},
    {"symbol": "LNVGY", "company": "Lenovo Group ADR", "role": "pc_oem_server_vendor", "intel_group": "CCG_DCAI"},
])

# ============================================================
# BASIC API FUNCTION
# ============================================================

def fmp_get(endpoint, params=None, sleep=0.25):
    if params is None:
        params = {}

    params = dict(params)
    params["apikey"] = FMP_API_KEY

    url = f"{BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, params=params, timeout=30)
    time.sleep(sleep)

    if r.status_code != 200:
        print("URL:", r.url)
        print("Status:", r.status_code)
        print("Text:", r.text[:500])
        r.raise_for_status()

    try:
        return r.json()
    except Exception:
        print("URL:", r.url)
        print("Raw text:", r.text[:500])
        raise


# ============================================================
# 1. GET LATEST TRANSCRIPT METADATA
# 这是你截图里的 API
# Endpoint:
# /stable/earning-call-transcript-latest?limit=100&page=0
# ============================================================

def get_latest_transcripts_all(max_pages=100, limit=100):
    rows = []

    for page in tqdm(range(max_pages), desc="Fetching latest transcript pages"):
        data = fmp_get(
            "earning-call-transcript-latest",
            params={
                "limit": limit,
                "page": page
            }
        )

        if not data:
            break

        if isinstance(data, dict):
            data = [data]

        rows.extend(data)

        if len(data) < limit:
            break

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


latest_df = get_latest_transcripts_all(max_pages=100, limit=100)
latest_df.to_csv(OUT / "fmp_latest_transcript_metadata_all.csv", index=False)

print("Latest metadata rows:", len(latest_df))
print(latest_df.head())


# ============================================================
# 2. GET TRANSCRIPT DATES BY SYMBOL
# Endpoint:
# /stable/earning-call-transcript-dates?symbol=AAPL
# 返回某个公司所有可用 transcript 的 period / fiscalYear / date
# ============================================================

def get_transcript_dates_by_symbol(symbol):
    data = fmp_get(
        "earning-call-transcript-dates",
        params={"symbol": symbol}
    )

    if not data:
        return pd.DataFrame()

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)
    df["symbol"] = symbol

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


all_dates = []

for _, row in tqdm(UNIVERSE.iterrows(), total=len(UNIVERSE), desc="Fetching transcript dates"):
    symbol = row["symbol"]

    df_dates = get_transcript_dates_by_symbol(symbol)

    if df_dates.empty:
        print(f"No transcript dates found for {symbol}")
        continue

    df_dates["company"] = row["company"]
    df_dates["role"] = row["role"]
    df_dates["intel_group"] = row["intel_group"]

    all_dates.append(df_dates)

dates_df = pd.concat(all_dates, ignore_index=True) if all_dates else pd.DataFrame()
dates_df.to_csv(OUT / "first_tier_transcript_dates.csv", index=False)

print("Transcript date rows:", len(dates_df))
print(dates_df.head())


# ============================================================
# 3. FILTER BEFORE INTEL EARNINGS DATE
# 只保留 Intel 财报日前已经公开的 transcript，避免 data leakage
# ============================================================

pre_target_df = dates_df[
    (dates_df["date"] >= START_DATE) &
    (dates_df["date"] < TARGET_EARNINGS_DATE)
].copy()

pre_target_df["target_ticker"] = TARGET_TICKER
pre_target_df["target_quarter"] = TARGET_QUARTER
pre_target_df["target_earnings_date"] = TARGET_EARNINGS_DATE.date().isoformat()
pre_target_df["days_before_target"] = (TARGET_EARNINGS_DATE - pre_target_df["date"]).dt.days

pre_target_df = pre_target_df.sort_values(["symbol", "date"], ascending=[True, False])
pre_target_df.to_csv(OUT / "first_tier_pre_intel_transcript_metadata.csv", index=False)

print("Pre-target transcript rows:", len(pre_target_df))
print(pre_target_df)


# ============================================================
# 4. GET FULL TRANSCRIPT TEXT
# Endpoint:
# /stable/earning-call-transcript?symbol=AAPL&period=Q3&year=2025
#
# 你截图里的 metadata 字段是:
# symbol, period, fiscalYear, date
# 所以这里用 symbol + period + fiscalYear 去拿全文
# ============================================================

def get_full_transcript(symbol, period, fiscal_year):
    data = fmp_get(
        "earning-call-transcript",
        params={
            "symbol": symbol,
            "period": period,
            "year": int(fiscal_year)
        }
    )

    if not data:
        return None

    if isinstance(data, list):
        if len(data) == 0:
            return None
        return data[0]

    if isinstance(data, dict):
        return data

    return None


full_records = []

for _, row in tqdm(pre_target_df.iterrows(), total=len(pre_target_df), desc="Fetching full transcripts"):
    symbol = row["symbol"]

    # FMP stable metadata 常见字段：period, fiscalYear
    period = row.get("period")
    fiscal_year = row.get("fiscalYear")

    if pd.isna(period) or pd.isna(fiscal_year):
        print(f"Missing period/fiscalYear for {symbol}: {row.to_dict()}")
        continue

    try:
        obj = get_full_transcript(symbol, period, fiscal_year)

        if obj is None:
            print(f"No full transcript returned: {symbol} {period} {fiscal_year}")
            continue

        content = obj.get("content") or obj.get("transcript") or obj.get("text") or ""

        full_records.append({
            "target_ticker": TARGET_TICKER,
            "target_quarter": TARGET_QUARTER,
            "target_earnings_date": TARGET_EARNINGS_DATE.date().isoformat(),

            "source_symbol": symbol,
            "source_company": row.get("company"),
            "source_role": row.get("role"),
            "intel_group": row.get("intel_group"),

            "source_period": period,
            "source_fiscal_year": fiscal_year,
            "source_transcript_date": row["date"].date().isoformat(),
            "days_before_target": int(row["days_before_target"]),

            "content": content,
            "content_length": len(content),
            "raw_json": json.dumps(obj, ensure_ascii=False)
        })

    except Exception as e:
        print(f"FAILED {symbol} {period} {fiscal_year}: {e}")

full_df = pd.DataFrame(full_records)

if not full_df.empty:
    full_df = full_df[full_df["content_length"] > 500].copy()

full_df.to_csv(OUT / "first_tier_pre_intel_full_transcripts.csv", index=False)

print("Full transcript rows:", len(full_df))
print(full_df[[
    "source_symbol",
    "source_company",
    "source_period",
    "source_fiscal_year",
    "source_transcript_date",
    "days_before_target",
    "content_length"
]])


# ============================================================
# 5. CREATE LLM INPUT TABLE
# 后面直接把这个 CSV 喂给 LLM 做语义抽取
# ============================================================

def build_prompt_context(row):
    return f"""
Target company: Intel ({TARGET_TICKER})
Target quarter: {TARGET_QUARTER}
Target earnings date: {TARGET_EARNINGS_DATE.date()}

Source company: {row['source_company']} ({row['source_symbol']})
Source company role: {row['source_role']}
Source transcript period: {row['source_period']} {row['source_fiscal_year']}
Source transcript date: {row['source_transcript_date']}
Days before Intel earnings: {row['days_before_target']}

Analyze this transcript as a public demand proxy for Intel.

Intel business mapping:
- PC OEM demand affects Intel Client Computing Group (CCG).
- Commercial PC refresh affects Intel CCG.
- AI PC adoption affects Intel CCG.
- Server and enterprise infrastructure demand affects Intel Data Center and AI (DCAI).
- AI server demand is indirectly relevant to Intel through host CPU and server platform demand.
- Component shortages, memory pricing, inventory pressure, and pricing pressure may hurt demand or margins.
- Do not assume the source company is a confirmed Intel customer. Treat it only as a demand proxy.

Return JSON fields later:
intel_relevance_score,
pc_demand_signal,
commercial_pc_signal,
ai_pc_signal,
server_infrastructure_signal,
cloud_capacity_signal,
ai_server_indirect_signal,
inventory_pressure_signal,
component_cost_pressure_signal,
margin_pressure_signal,
likely_impact_on_intel_ccg,
likely_impact_on_intel_dcai,
short_summary,
key_evidence,
risk_notes.
""".strip()


if not full_df.empty:
    llm_input_df = full_df.copy()
    llm_input_df["prompt_context"] = llm_input_df.apply(build_prompt_context, axis=1)
    llm_input_df["transcript_text"] = llm_input_df["content"].str[:60000]

    llm_input_df = llm_input_df[[
        "target_ticker",
        "target_quarter",
        "target_earnings_date",
        "source_symbol",
        "source_company",
        "source_role",
        "intel_group",
        "source_period",
        "source_fiscal_year",
        "source_transcript_date",
        "days_before_target",
        "content_length",
        "prompt_context",
        "transcript_text"
    ]]

    llm_input_df.to_csv(OUT / "llm_input_first_tier_pre_intel.csv", index=False)

    print("LLM input rows:", len(llm_input_df))
    print(llm_input_df[[
        "source_symbol",
        "source_company",
        "source_period",
        "source_fiscal_year",
        "source_transcript_date",
        "days_before_target",
        "content_length"
    ]])
else:
    print("No full transcripts downloaded. Check API key, subscription permission, or symbol coverage.")