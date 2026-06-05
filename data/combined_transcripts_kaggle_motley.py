import re
import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

# Kaggle source:
# https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts
KAGGLE_MOTLEY_DATASET = "tpotterer/motley-fool-scraped-earnings-call-transcripts"

HF_DATASET_NAME = "Rogersurf/earnings-call-transcripts"

OUT_DIR = Path("combined_transcript_data")
OUT_DIR.mkdir(exist_ok=True)

RAW_COMBINED_PATH = OUT_DIR / "combined_raw_before_dedup.csv"
DEDUP_PATH = OUT_DIR / "combined_transcripts_deduplicated.csv"
DEDUP_SUMMARY_PATH = OUT_DIR / "dedup_summary.json"
KAGGLE_SCHEMA_PATH = OUT_DIR / "kaggle_motley_schema_preview.json"
HF_SCHEMA_PATH = OUT_DIR / "hf_schema_preview.json"


# ============================================================
# BASIC CLEANING FUNCTIONS
# ============================================================

def normalize_space(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_for_hash(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def md5_hash(text: str) -> str:
    return hashlib.md5(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(normalize_space(text).split())


def clean_ticker(x: str) -> str:
    """
    Handles examples:
    NASDAQ: BILI -> BILI
    NYSE: GFF -> GFF
    (NYSE: CIB) -> CIB
    BILI -> BILI
    """
    if pd.isna(x):
        return ""

    x = str(x).strip()
    x = x.replace("(", "").replace(")", "").strip()

    if ":" in x:
        x = x.split(":")[-1].strip()

    x = re.sub(r"[^A-Za-z0-9.\-]", "", x)
    return x.upper()


def parse_motley_date_series(date_series: pd.Series) -> pd.Series:
    """
    Motley raw date example:
    Nov 18, 2021, 12:00 p.m. ET
    """
    s = date_series.astype(str)

    cleaned = (
        s.str.replace(r"\s+ET$", "", regex=True)
         .str.replace(r"\.(?=\s)", "", regex=True)
         .str.replace("a.m.", "AM", regex=False)
         .str.replace("p.m.", "PM", regex=False)
    )

    return pd.to_datetime(cleaned, errors="coerce", format="mixed")


def parse_any_date(x: Any):
    if x is None:
        return pd.NaT

    if isinstance(x, (dict, list, tuple)):
        return pd.NaT

    try:
        if pd.isna(x):
            return pd.NaT
    except Exception:
        pass

    return pd.to_datetime(x, errors="coerce", utc=True).tz_convert(None)


def parse_motley_publish_dates(date_series: pd.Series) -> pd.Series:
    """
    First try Motley-specific format, then fallback to general pandas parsing.
    """
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce")

    parsed = parse_motley_date_series(date_series)
    fallback = date_series.apply(parse_any_date)

    # Fill Motley-specific parsing failures with general parsing results.
    return parsed.fillna(fallback)


def make_quarter_from_date(date_value):
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return ""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def normalize_title(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# GENERAL COLUMN HELPERS
# ============================================================

def flatten_dict(d, parent_key="", sep="."):
    items = []

    if not isinstance(d, dict):
        return {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]):
    """
    Case-insensitive column finder.
    Returns the actual column name in df.
    """
    exact = {col: col for col in df.columns}
    lower_map = {str(col).lower(): col for col in df.columns}

    for col in candidates:
        if col in exact:
            return exact[col]

        lowered = str(col).lower()
        if lowered in lower_map:
            return lower_map[lowered]

    return None


def find_best_text_column(df: pd.DataFrame):
    candidates = [
        "transcript",
        "text",
        "content",
        "body",
        "full_text",
        "call_transcript",
        "earnings_call",
        "document"
    ]

    for col in candidates:
        real_col = find_first_existing_column(df, [col])
        if real_col is not None:
            avg_len = df[real_col].astype(str).str.len().mean()
            if avg_len > 500:
                return real_col

    # fallback: choose object column with longest average length
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not object_cols:
        return None

    avg_lengths = {
        col: df[col].astype(str).str.len().mean()
        for col in object_cols
    }

    best_col = max(avg_lengths, key=avg_lengths.get)

    if avg_lengths[best_col] < 500:
        return None

    return best_col


def safe_json_sample_row(row: dict, max_chars: int = 800) -> dict:
    """
    Prevent schema preview JSON from becoming huge because transcripts are long.
    """
    safe = {}
    for k, v in row.items():
        text = str(v)
        safe[k] = text[:max_chars] + ("..." if len(text) > max_chars else "")
    return safe


# ============================================================
# LOAD MOTLEY FOOL DATA DIRECTLY FROM KAGGLE
# ============================================================

def download_kaggle_dataset(dataset_name: str) -> Path:
    """
    Downloads the latest Kaggle dataset version into KaggleHub's local cache.

    Install first:
        pip install kagglehub

    For some environments you may also need Kaggle credentials:
        ~/.kaggle/kaggle.json
    """
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "Missing dependency: kagglehub. Install it with:\n"
            "    pip install kagglehub\n"
        ) from e

    dataset_dir = kagglehub.dataset_download(dataset_name)
    return Path(dataset_dir)


def read_dataframe_from_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        return pd.DataFrame(obj).copy()

    if suffix == ".json":
        return pd.read_json(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported data file type: {path}")


def list_supported_data_files(dataset_dir: Path) -> list[Path]:
    supported_suffixes = {".csv", ".parquet", ".pq", ".pkl", ".pickle", ".json", ".jsonl"}

    files = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_suffixes
    ]

    def file_score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        score = 0

        # Prefer files that look like the main transcript data.
        for keyword, weight in [
            ("motley", 10),
            ("fool", 10),
            ("earnings", 5),
            ("transcript", 5),
            ("data", 2),
        ]:
            if keyword in name:
                score += weight

        # The Kaggle dataset is commonly distributed as a pickle-like data file.
        suffix_priority = {
            ".pkl": 5,
            ".pickle": 5,
            ".csv": 4,
            ".parquet": 3,
            ".pq": 3,
            ".jsonl": 2,
            ".json": 1,
        }
        score += suffix_priority.get(path.suffix.lower(), 0)

        # Put larger likely data files first if scores tie.
        try:
            score += min(int(path.stat().st_size // 10_000_000), 20)
        except OSError:
            pass

        return (-score, str(path))

    return sorted(files, key=file_score)


def validate_motley_columns(df: pd.DataFrame) -> dict:
    """
    The Kaggle Motley dataset is expected to have:
    date, exchange, quarter/q, ticker, transcript.

    This function supports q/quarter/period variants so the script is not brittle.
    """
    text_col = find_first_existing_column(
        df,
        ["transcript", "text", "content", "body", "full_text"]
    )

    date_col = find_first_existing_column(
        df,
        ["date", "publish_date", "published_date", "call_date", "created_at"]
    )

    exchange_col = find_first_existing_column(
        df,
        ["exchange", "market", "stock_exchange"]
    )

    ticker_col = find_first_existing_column(
        df,
        ["ticker", "symbol", "company_ticker", "stock"]
    )

    quarter_col = find_first_existing_column(
        df,
        ["q", "quarter", "fiscal_quarter", "period"]
    )

    missing = []
    if text_col is None:
        missing.append("transcript/text")
    if date_col is None:
        missing.append("date")
    if ticker_col is None and exchange_col is None:
        missing.append("ticker or exchange")
    if quarter_col is None:
        missing.append("q/quarter/period")

    if missing:
        raise ValueError(
            "Not a valid Motley transcript table. Missing: "
            + ", ".join(missing)
            + f". Columns: {df.columns.tolist()}"
        )

    return {
        "text_col": text_col,
        "date_col": date_col,
        "exchange_col": exchange_col,
        "ticker_col": ticker_col,
        "quarter_col": quarter_col,
    }


def load_motley_data() -> pd.DataFrame:
    """
    Replaces the old local pickle reader:

        MOTLEY_PKL_PATH = Path("../data/motley-fool-data.pkl")

    with direct Kaggle loading:

        kagglehub.dataset_download(
            "tpotterer/motley-fool-scraped-earnings-call-transcripts"
        )
    """
    dataset_dir = download_kaggle_dataset(KAGGLE_MOTLEY_DATASET)
    candidate_files = list_supported_data_files(dataset_dir)

    if not candidate_files:
        raise FileNotFoundError(
            f"No supported data files found in Kaggle dataset directory: {dataset_dir}"
        )

    errors = []

    for data_file in candidate_files:
        try:
            df = read_dataframe_from_file(data_file)
            df = df.copy()
            cols = validate_motley_columns(df)
            break
        except Exception as e:
            errors.append(f"{data_file}: {repr(e)}")
    else:
        raise ValueError(
            "Could not find a usable Motley Fool transcript file in the Kaggle dataset.\n"
            + "\n".join(errors[:10])
        )

    schema_preview = {
        "dataset": KAGGLE_MOTLEY_DATASET,
        "dataset_dir": str(dataset_dir),
        "selected_file": str(data_file),
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "detected_columns": cols,
        "sample_row": safe_json_sample_row(df.head(1).to_dict("records")[0]) if len(df) > 0 else {}
    }

    with open(KAGGLE_SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema_preview, f, indent=2, ensure_ascii=False, default=str)

    text_col = cols["text_col"]
    date_col = cols["date_col"]
    exchange_col = cols["exchange_col"]
    ticker_col = cols["ticker_col"]
    quarter_col = cols["quarter_col"]

    publish_dates = parse_motley_publish_dates(df[date_col])

    out = pd.DataFrame()
    out["source_dataset"] = f"kaggle:{KAGGLE_MOTLEY_DATASET}"
    out["source_row_id"] = df.index.astype(str)
    out["source_field_path"] = f"{data_file.name}:{text_col}"

    if ticker_col:
        out["ticker"] = df[ticker_col].apply(clean_ticker)
    else:
        out["ticker"] = ""

    # exchange sometimes has ticker too; fallback if ticker missing.
    if exchange_col:
        fallback_ticker = df[exchange_col].apply(clean_ticker)
        out["ticker"] = out["ticker"].where(out["ticker"].str.len() > 0, fallback_ticker)
        out["exchange"] = df[exchange_col].astype(str)
    else:
        out["exchange"] = ""

    out["company"] = ""
    out["title"] = ""
    out["period"] = df[quarter_col].astype(str)
    out["publish_date"] = publish_dates
    out["quarter"] = out["publish_date"].apply(make_quarter_from_date)

    # If date parsing fails for some rows, keep the original q/quarter/period value.
    out["quarter"] = out["quarter"].where(
        out["quarter"].astype(str).str.len() > 0,
        out["period"].astype(str)
    )

    out["text"] = df[text_col].astype(str).apply(normalize_space)

    return out


# ============================================================
# LOAD HUGGING FACE DATA
# ============================================================

def load_hf_data() -> pd.DataFrame:
    ds = load_dataset(HF_DATASET_NAME)

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    raw_df = ds[split_name].to_pandas()

    # Flatten nested dict columns if any.
    records = []

    for _, row in raw_df.iterrows():
        row_dict = row.to_dict()
        flat = {}

        for k, v in row_dict.items():
            if isinstance(v, dict):
                flat.update(flatten_dict(v, k))
            else:
                flat[k] = v

        records.append(flat)

    df = pd.DataFrame(records)

    schema_preview = {
        "dataset": HF_DATASET_NAME,
        "split": split_name,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "sample_row": safe_json_sample_row(df.head(1).to_dict("records")[0]) if len(df) > 0 else {}
    }

    with open(HF_SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema_preview, f, indent=2, ensure_ascii=False, default=str)

    text_col = find_best_text_column(df)

    if text_col is None:
        raise ValueError(
            "Could not find transcript text column in HF dataset. "
            f"Check schema file: {HF_SCHEMA_PATH}"
        )

    ticker_col = find_first_existing_column(
        df,
        [
            "ticker",
            "symbol",
            "company_ticker",
            "stock",
            "exchange",
            "meta.ticker",
            "metadata.ticker"
        ]
    )

    company_col = find_first_existing_column(
        df,
        [
            "company",
            "company_name",
            "name",
            "companyName",
            "meta.company",
            "metadata.company"
        ]
    )

    date_col = find_first_existing_column(
        df,
        [
            "date",
            "publish_date",
            "published_date",
            "call_date",
            "quarter_date",
            "fiscal_date",
            "created_at"
        ]
    )

    quarter_col = find_first_existing_column(
        df,
        [
            "quarter",
            "q",
            "fiscal_quarter",
            "period"
        ]
    )

    title_col = find_first_existing_column(
        df,
        [
            "title",
            "headline",
            "event_title"
        ]
    )

    out = pd.DataFrame()
    out["source_dataset"] = HF_DATASET_NAME
    out["source_row_id"] = df.index.astype(str)
    out["source_field_path"] = text_col

    if ticker_col:
        out["ticker"] = df[ticker_col].apply(clean_ticker)
    else:
        out["ticker"] = ""

    if company_col:
        out["company"] = df[company_col].astype(str)
    else:
        out["company"] = ""

    if title_col:
        out["title"] = df[title_col].astype(str)
    else:
        out["title"] = ""

    if quarter_col:
        out["period"] = df[quarter_col].astype(str)
    else:
        out["period"] = ""

    if date_col:
        out["publish_date"] = df[date_col].apply(parse_any_date)
    else:
        out["publish_date"] = pd.NaT

    out["quarter"] = out["publish_date"].apply(make_quarter_from_date)

    # If quarter missing from date, fallback to provided period.
    out["quarter"] = out["quarter"].where(
        out["quarter"].astype(str).str.len() > 0,
        out["period"].astype(str)
    )

    out["exchange"] = ""
    out["text"] = df[text_col].astype(str).apply(normalize_space)

    return out


# ============================================================
# FINAL NORMALIZATION + DEDUP
# ============================================================

def finalize_normalized_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "source_dataset",
        "source_row_id",
        "source_field_path",
        "ticker",
        "exchange",
        "company",
        "title",
        "period",
        "quarter",
        "text"
    ]:
        if col not in df.columns:
            df[col] = ""

    if "publish_date" not in df.columns:
        df["publish_date"] = pd.NaT

    df["ticker"] = df["ticker"].apply(clean_ticker)
    df["company"] = df["company"].astype(str).apply(normalize_space)
    df["title"] = df["title"].astype(str).apply(normalize_space)
    df["period"] = df["period"].astype(str).apply(normalize_space)
    df["quarter"] = df["quarter"].astype(str).apply(normalize_space)
    df["text"] = df["text"].astype(str).apply(normalize_space)

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["date_only"] = df["publish_date"].dt.date.astype(str)

    df["word_count"] = df["text"].apply(word_count)
    df["content_hash"] = df["text"].apply(md5_hash)

    df["title_norm"] = df["title"].apply(normalize_title)

    # Stable document id.
    df["doc_id"] = (
        df["source_dataset"].astype(str)
        + "__"
        + df["source_row_id"].astype(str)
        + "__"
        + df["content_hash"].str.slice(0, 10)
    )

    # Remove empty or too short transcripts.
    df = df[df["word_count"] >= 200].copy()

    return df


def deduplicate_transcripts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(df)

    # Prefer longer text if duplicate.
    df = df.sort_values(
        by=["word_count"],
        ascending=False
    ).copy()

    # 1. Exact text dedup by content hash.
    before_hash = len(df)
    df_hash = df.drop_duplicates(subset=["content_hash"], keep="first").copy()
    after_hash = len(df_hash)

    # 2. Secondary dedup by ticker + date + quarter.
    # Only apply when ticker and date are usable.
    df_hash["has_secondary_key"] = (
        df_hash["ticker"].astype(str).str.strip().ne("")
        & df_hash["date_only"].astype(str).str.strip().ne("")
        & df_hash["date_only"].astype(str).str.strip().ne("NaT")
        & df_hash["date_only"].astype(str).str.strip().ne("nan")
    )

    keyed = df_hash[df_hash["has_secondary_key"]].copy()
    unkeyed = df_hash[~df_hash["has_secondary_key"]].copy()

    before_secondary = len(keyed)

    keyed = keyed.sort_values(
        by=["ticker", "date_only", "quarter", "word_count"],
        ascending=[True, True, True, False]
    )

    keyed = keyed.drop_duplicates(
        subset=["ticker", "date_only", "quarter"],
        keep="first"
    )

    after_secondary = len(keyed)

    final_df = pd.concat([keyed, unkeyed], ignore_index=True)
    final_df = final_df.drop(columns=["has_secondary_key"], errors="ignore")

    # Sort final output.
    final_df = final_df.sort_values(
        by=["publish_date", "ticker"],
        ascending=[True, True],
        na_position="last"
    ).reset_index(drop=True)

    summary = {
        "rows_before_dedup": int(before),
        "rows_before_hash_dedup": int(before_hash),
        "rows_after_hash_dedup": int(after_hash),
        "exact_hash_duplicates_removed": int(before_hash - after_hash),
        "rows_before_secondary_dedup_keyed_only": int(before_secondary),
        "rows_after_secondary_dedup_keyed_only": int(after_secondary),
        "secondary_duplicates_removed": int(before_secondary - after_secondary),
        "rows_after_all_dedup": int(len(final_df)),
        "total_duplicates_removed": int(before - len(final_df)),
        "unique_tickers": int(final_df["ticker"].nunique()),
        "date_min": str(final_df["publish_date"].min()),
        "date_max": str(final_df["publish_date"].max()),
        "total_words": int(final_df["word_count"].sum()),
        "avg_words_per_transcript": float(final_df["word_count"].mean())
    }

    return final_df, summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading Kaggle Motley Fool data...")
    motley_df = load_motley_data()
    print("Motley rows:", len(motley_df))

    print("\nLoading Hugging Face data...")
    hf_df = load_hf_data()
    print("HF rows:", len(hf_df))

    print("\nCombining...")
    combined = pd.concat([motley_df, hf_df], ignore_index=True)
    combined = finalize_normalized_df(combined)

    print("Combined rows before dedup:", len(combined))

    combined.to_csv(RAW_COMBINED_PATH, index=False)

    print("\nDeduplicating...")
    dedup_df, summary = deduplicate_transcripts(combined)

    dedup_df.to_csv(DEDUP_PATH, index=False)

    with open(DEDUP_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDONE.")
    print("Raw combined:", RAW_COMBINED_PATH.resolve())
    print("Deduplicated:", DEDUP_PATH.resolve())
    print("Summary:", DEDUP_SUMMARY_PATH.resolve())
    print("Kaggle schema preview:", KAGGLE_SCHEMA_PATH.resolve())
    print("HF schema preview:", HF_SCHEMA_PATH.resolve())

    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSource counts after dedup:")
    print(dedup_df["source_dataset"].value_counts())

    print("\nPreview:")
    print(
        dedup_df[
            [
                "source_dataset",
                "ticker",
                "company",
                "period",
                "quarter",
                "publish_date",
                "word_count"
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()
