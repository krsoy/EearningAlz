import os
import re
import argparse
from pathlib import Path

import pandas as pd
import matplotlib

# 正常 .py 环境下推荐使用 Agg，避免没有图形界面时报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq


DATASET_REPO = "Rogersurf/earnings-call-transcripts"


def find_parquet_file(repo_id: str) -> str:
    """
    自动寻找 Hugging Face dataset repo 里的 parquet 文件。
    """
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]

    if not parquet_files:
        raise FileNotFoundError("No parquet file found in the dataset repository.")

    print("Found parquet files:")
    for f in parquet_files:
        print(" -", f)

    # 如果只有一个 parquet 文件，直接使用
    # 如果有多个，默认使用第一个
    return parquet_files[0]


def download_dataset(repo_id: str, filename: str = None) -> str:
    """
    下载 parquet 文件。
    """
    if filename is None:
        filename = find_parquet_file(repo_id)

    print(f"\nDownloading file: {filename}")

    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )

    print(f"Downloaded to: {parquet_path}")
    return parquet_path


def get_existing_columns(parquet_path: str):
    """
    读取 parquet schema，获取实际存在的列名。
    """
    pf = pq.ParquetFile(parquet_path)
    return pf.schema.names


def safe_read_parquet(parquet_path: str, wanted_cols: list) -> pd.DataFrame:
    """
    只读取实际存在的列，避免列名不一致时报错。
    """
    existing_cols = get_existing_columns(parquet_path)
    cols = [c for c in wanted_cols if c in existing_cols]

    missing_cols = [c for c in wanted_cols if c not in existing_cols]
    if missing_cols:
        print("\nWarning: these columns are missing in the dataset:")
        for c in missing_cols:
            print(" -", c)

    print("\nReading columns:")
    for c in cols:
        print(" -", c)

    return pd.read_parquet(parquet_path, columns=cols)


def clean_quarter_value(x):
    """
    把 quarter 字段清洗成 Q1/Q2/Q3/Q4 格式。
    """
    if pd.isna(x):
        return None

    s = str(x).strip().upper()

    # already like Q1
    m = re.search(r"Q([1-4])", s)
    if m:
        return f"Q{m.group(1)}"

    # like 1, 2, 3, 4
    m = re.search(r"\b([1-4])\b", s)
    if m:
        return f"Q{m.group(1)}"

    return s


def save_bar_plot(df, x_col, y_col, title, xlabel, ylabel, output_path, rotation=45):
    plt.figure(figsize=(12, 6))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_line_plot(df, x_col, y_col, title, xlabel, ylabel, output_path, rotation=90):
    plt.figure(figsize=(14, 6))
    plt.plot(df[x_col].astype(str), df[y_col], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_horizontal_bar_plot(df, x_col, y_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(12, 8))
    plot_df = df.iloc[::-1]
    plt.barh(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def basic_eda(parquet_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = [
        "ticker",
        "company",
        "quarter",
        "earnings_year",
        "call_date",
        "title",
        "source_url",
        "scraped_at"
    ]

    df = safe_read_parquet(parquet_path, meta_cols)

    print("\n==============================")
    print("Basic Dataset Information")
    print("==============================")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head(5).to_string())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    # 保存基础信息
    with open(output_dir / "basic_info.txt", "w", encoding="utf-8") as f:
        f.write("Shape:\n")
        f.write(str(df.shape))
        f.write("\n\nColumns:\n")
        f.write(str(df.columns.tolist()))
        f.write("\n\nData types:\n")
        f.write(str(df.dtypes))
        f.write("\n\nMissing values:\n")
        f.write(df.isna().sum().sort_values(ascending=False).to_string())
        f.write("\n\nFirst 5 rows:\n")
        f.write(df.head(5).to_string())

    # 日期处理
    if "call_date" in df.columns:
        df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
        df["call_year"] = df["call_date"].dt.year
        df["call_month"] = df["call_date"].dt.month
        df["call_month_period"] = df["call_date"].dt.to_period("M").astype(str)
    else:
        df["call_year"] = pd.NA
        df["call_month"] = pd.NA
        df["call_month_period"] = pd.NA

    if "scraped_at" in df.columns:
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce")

    if "quarter" in df.columns:
        df["quarter_clean"] = df["quarter"].apply(clean_quarter_value)
    else:
        df["quarter_clean"] = pd.NA

    # earnings_year 处理
    if "earnings_year" in df.columns:
        df["earnings_year"] = pd.to_numeric(df["earnings_year"], errors="coerce").astype("Int64")

    # 保存清洗后的 metadata
    df.to_csv(output_dir / "metadata_cleaned.csv", index=False)

    # 每年数量：earnings_year
    if "earnings_year" in df.columns:
        year_count_earnings = (
            df.dropna(subset=["earnings_year"])
            .groupby("earnings_year")
            .size()
            .reset_index(name="count")
            .sort_values("earnings_year")
        )

        year_count_earnings.to_csv(output_dir / "year_count_by_earnings_year.csv", index=False)

        print("\n==============================")
        print("Count by earnings_year")
        print("==============================")
        print(year_count_earnings.to_string(index=False))

        save_bar_plot(
            year_count_earnings,
            x_col="earnings_year",
            y_col="count",
            title="Number of Earnings Call Transcripts by Earnings Year",
            xlabel="Earnings Year",
            ylabel="Number of Transcripts",
            output_path=output_dir / "year_count_by_earnings_year.png"
        )

    # 每年数量：call_year
    if "call_year" in df.columns:
        year_count_call = (
            df.dropna(subset=["call_year"])
            .groupby("call_year")
            .size()
            .reset_index(name="count")
            .sort_values("call_year")
        )

        year_count_call["call_year"] = year_count_call["call_year"].astype(int)
        year_count_call.to_csv(output_dir / "year_count_by_call_year.csv", index=False)

        print("\n==============================")
        print("Count by call_year")
        print("==============================")
        print(year_count_call.to_string(index=False))

        save_bar_plot(
            year_count_call,
            x_col="call_year",
            y_col="count",
            title="Number of Earnings Call Transcripts by Call Year",
            xlabel="Call Year",
            ylabel="Number of Transcripts",
            output_path=output_dir / "year_count_by_call_year.png"
        )

    # 年份 × 季度分布
    if "earnings_year" in df.columns and "quarter_clean" in df.columns:
        year_quarter = (
            df.dropna(subset=["earnings_year", "quarter_clean"])
            .groupby(["earnings_year", "quarter_clean"])
            .size()
            .reset_index(name="count")
            .sort_values(["earnings_year", "quarter_clean"])
        )

        year_quarter.to_csv(output_dir / "year_quarter_distribution.csv", index=False)

        year_quarter_pivot = (
            year_quarter
            .pivot(index="earnings_year", columns="quarter_clean", values="count")
            .fillna(0)
            .astype(int)
        )

        year_quarter_pivot.to_csv(output_dir / "year_quarter_pivot.csv")

        print("\n==============================")
        print("Year x Quarter Distribution")
        print("==============================")
        print(year_quarter_pivot.to_string())

        plt.figure(figsize=(12, 6))
        year_quarter_pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
        plt.title("Earnings Call Transcript Distribution by Year and Quarter")
        plt.xlabel("Earnings Year")
        plt.ylabel("Number of Transcripts")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "year_quarter_distribution.png", dpi=200)
        plt.close()

    # 月度分布
    if "call_month_period" in df.columns:
        monthly_count = (
            df.dropna(subset=["call_month_period"])
            .groupby("call_month_period")
            .size()
            .reset_index(name="count")
            .sort_values("call_month_period")
        )

        monthly_count.to_csv(output_dir / "monthly_distribution.csv", index=False)

        print("\n==============================")
        print("Monthly Distribution")
        print("==============================")
        print(monthly_count.head(20).to_string(index=False))

        save_line_plot(
            monthly_count,
            x_col="call_month_period",
            y_col="count",
            title="Monthly Distribution of Earnings Call Transcripts",
            xlabel="Call Month",
            ylabel="Number of Transcripts",
            output_path=output_dir / "monthly_distribution.png"
        )

    # ticker 分布
    if "ticker" in df.columns:
        ticker_count = (
            df["ticker"]
            .value_counts(dropna=False)
            .reset_index()
        )
        ticker_count.columns = ["ticker", "count"]
        ticker_count.to_csv(output_dir / "ticker_count.csv", index=False)

        top_tickers = ticker_count.head(30)

        print("\n==============================")
        print("Top 30 Tickers")
        print("==============================")
        print(top_tickers.to_string(index=False))

        save_horizontal_bar_plot(
            top_tickers,
            x_col="ticker",
            y_col="count",
            title="Top 30 Tickers by Number of Transcripts",
            xlabel="Number of Transcripts",
            ylabel="Ticker",
            output_path=output_dir / "top_30_tickers.png"
        )

        print("\nUnique tickers:", df["ticker"].nunique(dropna=True))

    # company 分布
    if "company" in df.columns:
        company_count = (
            df["company"]
            .value_counts(dropna=False)
            .reset_index()
        )
        company_count.columns = ["company", "count"]
        company_count.to_csv(output_dir / "company_count.csv", index=False)

        top_companies = company_count.head(30)

        print("\nUnique companies:", df["company"].nunique(dropna=True))

        save_horizontal_bar_plot(
            top_companies,
            x_col="company",
            y_col="count",
            title="Top 30 Companies by Number of Transcripts",
            xlabel="Number of Transcripts",
            ylabel="Company",
            output_path=output_dir / "top_30_companies.png"
        )

    # 公司年份覆盖
    if "ticker" in df.columns and "earnings_year" in df.columns:
        agg_dict = {
            "transcript_count": ("ticker", "size"),
            "first_year": ("earnings_year", "min"),
            "last_year": ("earnings_year", "max"),
            "n_years": ("earnings_year", "nunique"),
        }

        if "quarter_clean" in df.columns:
            agg_dict["n_quarters_type"] = ("quarter_clean", "nunique")

        company_year_coverage = (
            df.groupby("ticker", dropna=False)
            .agg(**agg_dict)
            .reset_index()
            .sort_values(["transcript_count", "n_years"], ascending=False)
        )

        company_year_coverage.to_csv(output_dir / "company_year_coverage.csv", index=False)

        print("\n==============================")
        print("Company Year Coverage - Top 30")
        print("==============================")
        print(company_year_coverage.head(30).to_string(index=False))

    # ticker-period 覆盖
    if "ticker" in df.columns and "earnings_year" in df.columns and "quarter_clean" in df.columns:
        df_period = df.copy()
        df_period["period"] = (
            df_period["earnings_year"].astype(str)
            + "-"
            + df_period["quarter_clean"].astype(str)
        )

        ticker_period_count = (
            df_period.dropna(subset=["ticker", "period"])
            .groupby("ticker")["period"]
            .nunique()
            .reset_index(name="unique_periods")
            .sort_values("unique_periods", ascending=False)
        )

        ticker_period_count.to_csv(output_dir / "ticker_period_count.csv", index=False)

        print("\n==============================")
        print("Ticker Period Coverage - Top 30")
        print("==============================")
        print(ticker_period_count.head(30).to_string(index=False))

    return df


def transcript_length_eda(parquet_path: str, output_dir: str):
    """
    读取 transcript 字段并计算文本长度。
    注意：这一步会读取大文本列，内存占用更高。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_cols = get_existing_columns(parquet_path)

    if "transcript" not in existing_cols:
        print("\nNo transcript column found. Skipping transcript length EDA.")
        return None

    wanted_cols = [
        "ticker",
        "company",
        "earnings_year",
        "quarter",
        "call_date",
        "transcript"
    ]

    cols = [c for c in wanted_cols if c in existing_cols]

    print("\n==============================")
    print("Reading transcript column for text length EDA")
    print("==============================")
    print("Columns:", cols)

    text_df = pd.read_parquet(parquet_path, columns=cols)

    text_df["transcript"] = text_df["transcript"].fillna("").astype(str)
    text_df["char_len"] = text_df["transcript"].str.len()
    text_df["word_len"] = text_df["transcript"].str.split().str.len()

    if "earnings_year" in text_df.columns:
        text_df["earnings_year"] = pd.to_numeric(text_df["earnings_year"], errors="coerce").astype("Int64")

    if "quarter" in text_df.columns:
        text_df["quarter_clean"] = text_df["quarter"].apply(clean_quarter_value)

    text_len_cols = [c for c in [
        "ticker",
        "company",
        "earnings_year",
        "quarter_clean",
        "call_date",
        "char_len",
        "word_len"
    ] if c in text_df.columns]

    text_df[text_len_cols].to_csv(output_dir / "transcript_text_length.csv", index=False)

    print("\n==============================")
    print("Transcript Length Summary")
    print("==============================")
    print(text_df[["char_len", "word_len"]].describe().to_string())

    with open(output_dir / "transcript_length_summary.txt", "w", encoding="utf-8") as f:
        f.write(text_df[["char_len", "word_len"]].describe().to_string())

    # 文本长度分布
    plt.figure(figsize=(12, 6))
    plt.hist(text_df["word_len"], bins=50)
    plt.title("Transcript Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Transcripts")
    plt.tight_layout()
    plt.savefig(output_dir / "transcript_word_count_distribution.png", dpi=200)
    plt.close()

    # 按年份看文本长度
    if "earnings_year" in text_df.columns:
        year_text_len = (
            text_df.dropna(subset=["earnings_year"])
            .groupby("earnings_year")
            .agg(
                count=("word_len", "size"),
                avg_word_len=("word_len", "mean"),
                median_word_len=("word_len", "median"),
                min_word_len=("word_len", "min"),
                max_word_len=("word_len", "max"),
            )
            .reset_index()
            .sort_values("earnings_year")
        )

        year_text_len.to_csv(output_dir / "year_text_length.csv", index=False)

        print("\n==============================")
        print("Text Length by Earnings Year")
        print("==============================")
        print(year_text_len.to_string(index=False))

        plt.figure(figsize=(12, 6))
        plt.plot(year_text_len["earnings_year"], year_text_len["avg_word_len"], marker="o", label="Average")
        plt.plot(year_text_len["earnings_year"], year_text_len["median_word_len"], marker="o", label="Median")
        plt.title("Average and Median Transcript Length by Earnings Year")
        plt.xlabel("Earnings Year")
        plt.ylabel("Word Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "year_text_length.png", dpi=200)
        plt.close()

    return text_df


def main():
    parser = argparse.ArgumentParser(
        description="EDA for Hugging Face earnings call transcript dataset."
    )

    parser.add_argument(
        "--repo",
        type=str,
        default=DATASET_REPO,
        help="Hugging Face dataset repo id."
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Parquet filename in the Hugging Face dataset repo. If not given, auto-detect first parquet file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results_earnings_eda",
        help="Output folder for EDA results."
    )

    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip transcript text length EDA to save memory."
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = download_dataset(
        repo_id=args.repo,
        filename=args.filename
    )

    print("\n==============================")
    print("Available Columns")
    print("==============================")
    existing_cols = get_existing_columns(parquet_path)
    for col in existing_cols:
        print(" -", col)

    basic_eda(parquet_path, output_dir)

    if not args.skip_text:
        transcript_length_eda(parquet_path, output_dir)
    else:
        print("\nSkipped transcript text length EDA.")

    print("\n==============================")
    print("EDA Finished")
    print("==============================")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()