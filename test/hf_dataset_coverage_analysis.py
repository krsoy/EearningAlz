import json
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "kunhanw/earning_call_transcript_dataset_with_volatility_analysis"

OUT = Path("hf_dataset_coverage_analysis")
PLOT_DIR = OUT / "plots"

OUT.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

ds = load_dataset(DATASET_NAME, split="train")
df = ds.to_pandas()

print("Raw rows:", len(df))
print("Raw columns:", df.columns.tolist())


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

print("Expanded columns:")
print(df.columns.tolist())


# ============================================================
# 3. NORMALIZE DATE / TICKER / COMPANY
# ============================================================

df["text"] = df["text"].astype(str)

if "company" not in df.columns:
    df["company"] = ""

df["company"] = df["company"].astype(str)

if "name" in df.columns:
    df["ticker"] = df["name"].astype(str).str.upper()
elif "volatility_analysis.ticker" in df.columns:
    df["ticker"] = df["volatility_analysis.ticker"].astype(str).str.upper()
else:
    df["ticker"] = ""

df["ticker"] = df["ticker"].replace(["NONE", "NAN", "NULL"], "")

# Prefer publishOn, fallback to volatility_analysis.publish_date
if "publishOn" in df.columns:
    df["publish_date"] = pd.to_datetime(
        df["publishOn"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)
elif "volatility_analysis.publish_date" in df.columns:
    df["publish_date"] = pd.to_datetime(
        df["volatility_analysis.publish_date"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)
else:
    df["publish_date"] = pd.NaT

df = df.dropna(subset=["publish_date"]).copy()

df["year"] = df["publish_date"].dt.year
df["quarter"] = df["publish_date"].dt.quarter
df["year_quarter"] = df["publish_date"].dt.to_period("Q").astype(str)
df["quarter_start"] = df["publish_date"].dt.to_period("Q").dt.start_time

df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

print("\nRows with valid publish_date:", len(df))
print("Date range:", df["publish_date"].min(), "to", df["publish_date"].max())
print("Unique tickers:", df["ticker"].nunique())
print("Unique companies:", df["company"].nunique())


# ============================================================
# 4. BASIC COVERAGE TABLES
# ============================================================

yearly_counts = (
    df.groupby("year")
    .agg(
        transcript_count=("text", "count"),
        unique_tickers=("ticker", "nunique"),
        unique_companies=("company", "nunique"),
        avg_word_count=("word_count", "mean"),
        total_words=("word_count", "sum")
    )
    .reset_index()
    .sort_values("year")
)

quarterly_counts = (
    df.groupby(["quarter_start", "year_quarter"])
    .agg(
        transcript_count=("text", "count"),
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

yearly_counts.to_csv(OUT / "yearly_transcript_coverage.csv", index=False)
quarterly_counts.to_csv(OUT / "quarterly_transcript_coverage.csv", index=False)

print("\nYearly coverage:")
print(yearly_counts)

print("\nQuarterly coverage:")
print(quarterly_counts.tail(20))


# ============================================================
# 5. PAST 10 YEARS COVERAGE
# ============================================================

latest_date = df["publish_date"].max()
start_10y = latest_date - pd.DateOffset(years=10)

df_10y = df[df["publish_date"] >= start_10y].copy()

quarterly_10y = (
    df_10y.groupby(["quarter_start", "year_quarter"])
    .agg(
        transcript_count=("text", "count"),
        unique_tickers=("ticker", "nunique"),
        unique_companies=("company", "nunique"),
        avg_word_count=("word_count", "mean"),
        total_words=("word_count", "sum")
    )
    .reset_index()
    .sort_values("quarter_start")
)

quarterly_10y["transcripts_per_ticker"] = (
    quarterly_10y["transcript_count"] /
    quarterly_10y["unique_tickers"].replace(0, pd.NA)
)

quarterly_10y.to_csv(OUT / "quarterly_transcript_coverage_past_10_years.csv", index=False)

print("\nPast 10 years start date:", start_10y)
print("\nPast 10 years quarterly coverage:")
print(quarterly_10y)


# ============================================================
# 6. TICKER-LEVEL COVERAGE
# ============================================================

ticker_coverage = (
    df.groupby(["ticker", "company"], dropna=False)
    .agg(
        transcript_count=("text", "count"),
        first_date=("publish_date", "min"),
        last_date=("publish_date", "max"),
        unique_quarters=("year_quarter", "nunique"),
        avg_word_count=("word_count", "mean"),
        total_words=("word_count", "sum")
    )
    .reset_index()
    .sort_values(["transcript_count", "unique_quarters"], ascending=False)
)

ticker_coverage.to_csv(OUT / "ticker_level_coverage.csv", index=False)

print("\nTop tickers by transcript count:")
print(ticker_coverage.head(30))


# ============================================================
# 7. QUARTER x TICKER MATRIX
# ============================================================

quarter_ticker_matrix = (
    df.pivot_table(
        index="year_quarter",
        columns="ticker",
        values="text",
        aggfunc="count",
        fill_value=0
    )
)

quarter_ticker_matrix.to_csv(OUT / "quarter_ticker_transcript_matrix.csv")

print("\nQuarter x ticker matrix shape:", quarter_ticker_matrix.shape)


# ============================================================
# 8. PLOTS
# ============================================================

# ------------------------------------------------------------
# Plot 1: transcripts per year
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.bar(
    yearly_counts["year"].astype(str),
    yearly_counts["transcript_count"]
)
plt.title("Number of Transcripts per Year")
plt.xlabel("Year")
plt.ylabel("Transcript Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_transcripts_per_year.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 2: unique tickers per year
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.bar(
    yearly_counts["year"].astype(str),
    yearly_counts["unique_tickers"]
)
plt.title("Unique Tickers per Year")
plt.xlabel("Year")
plt.ylabel("Unique Tickers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_unique_tickers_per_year.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 3: transcripts per quarter
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(
    quarterly_counts["year_quarter"],
    quarterly_counts["transcript_count"],
    marker="o"
)
plt.title("Number of Transcripts per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Transcript Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_transcripts_per_quarter.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 4: unique tickers per quarter
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(
    quarterly_counts["year_quarter"],
    quarterly_counts["unique_tickers"],
    marker="o"
)
plt.title("Unique Tickers per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Unique Tickers")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_unique_tickers_per_quarter.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 5: past 10 years transcripts per quarter
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.bar(
    quarterly_10y["year_quarter"],
    quarterly_10y["transcript_count"]
)
plt.title("Past 10 Years: Transcripts per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Transcript Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_past_10y_transcripts_per_quarter.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 6: past 10 years unique tickers per quarter
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.bar(
    quarterly_10y["year_quarter"],
    quarterly_10y["unique_tickers"]
)
plt.title("Past 10 Years: Unique Tickers per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Unique Tickers")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_past_10y_unique_tickers_per_quarter.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 7: avg word count per quarter
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(
    quarterly_counts["year_quarter"],
    quarterly_counts["avg_word_count"],
    marker="o"
)
plt.title("Average Transcript Word Count per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Average Word Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_avg_word_count_per_quarter.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 8: total words per quarter
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(
    quarterly_counts["year_quarter"],
    quarterly_counts["total_words"],
    marker="o"
)
plt.title("Total Transcript Words per Quarter")
plt.xlabel("Quarter")
plt.ylabel("Total Words")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_total_words_per_quarter.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 9: top tickers by transcript count
# ------------------------------------------------------------

top_tickers = ticker_coverage.head(30).copy()
top_tickers["label"] = (
    top_tickers["ticker"].astype(str)
    + " | "
    + top_tickers["company"].astype(str).str.slice(0, 35)
)

plt.figure(figsize=(12, 9))
plt.barh(
    top_tickers["label"][::-1],
    top_tickers["transcript_count"][::-1]
)
plt.title("Top Tickers by Transcript Count")
plt.xlabel("Transcript Count")
plt.ylabel("Ticker")
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_top_tickers_by_transcript_count.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 10: transcript count distribution by ticker
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.hist(
    ticker_coverage["transcript_count"],
    bins=30
)
plt.title("Distribution of Transcript Count per Ticker")
plt.xlabel("Transcript Count per Ticker")
plt.ylabel("Number of Tickers")
plt.tight_layout()
plt.savefig(PLOT_DIR / "10_transcript_count_distribution_by_ticker.png", dpi=300)
plt.show()


# ============================================================
# 9. SIMPLE CONCLUSION PRINT
# ============================================================

print("\n================ DATASET COVERAGE SUMMARY ================")
print(f"Total transcripts: {len(df)}")
print(f"Date range: {df['publish_date'].min()} to {df['publish_date'].max()}")
print(f"Unique tickers: {df['ticker'].nunique()}")
print(f"Unique companies: {df['company'].nunique()}")
print(f"Number of quarters covered: {df['year_quarter'].nunique()}")
print(f"Average transcripts per quarter: {quarterly_counts['transcript_count'].mean():.2f}")
print(f"Median transcripts per quarter: {quarterly_counts['transcript_count'].median():.2f}")
print(f"Average unique tickers per quarter: {quarterly_counts['unique_tickers'].mean():.2f}")
print(f"Median unique tickers per quarter: {quarterly_counts['unique_tickers'].median():.2f}")

print("\nGenerated CSV files:")
for p in OUT.glob("*.csv"):
    print("-", p.name)

print("\nGenerated plots:")
for p in PLOT_DIR.glob("*.png"):
    print("-", p.name)