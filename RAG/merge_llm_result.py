# merge_llm_csv_outputs_to_parquet.py

from pathlib import Path
import pandas as pd

BASE_DIR = Path("rag_chroma_output/llm_csv_outputs_balanced_time_range")
OUT_DIR = Path("rag_chroma_output/merged_parquet_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_TYPES = {
    "concepts": "concepts_*.csv",
    "relationships": "relationships_*.csv",
    "outlook": "outlook_*.csv",
    "failed": "failed_*.csv",
}

all_frames = []

for record_type, pattern in FILE_TYPES.items():
    files = sorted(BASE_DIR.rglob(pattern))
    print(f"\nLoading {record_type}: {len(files)} files")

    type_frames = []

    for f in files:
        print(f"  reading {f}")
        df = pd.read_csv(f)

        df["record_type"] = record_type
        df["source_file"] = str(f)
        df["source_folder"] = f.parent.name

        type_frames.append(df)

    if type_frames:
        type_df = pd.concat(type_frames, ignore_index=True, sort=False)

        typed_out = OUT_DIR / f"llm_{record_type}_all.parquet"
        type_df.to_parquet(typed_out, index=False)
        print(f"SAVED {typed_out} rows={len(type_df):,} cols={len(type_df.columns):,}")

        all_frames.append(type_df)

if not all_frames:
    raise RuntimeError("No CSV files found.")

all_df = pd.concat(all_frames, ignore_index=True, sort=False)

# Optional: put important metadata columns first
front_cols = ["record_type", "source_folder", "source_file"]
other_cols = [c for c in all_df.columns if c not in front_cols]
all_df = all_df[front_cols + other_cols]

out_path = OUT_DIR / "llm_csv_outputs_balanced_time_range_all.parquet"
all_df.to_parquet(out_path, index=False)

print("\nDONE")
print(f"SAVED {out_path}")
print(f"Total rows: {len(all_df):,}")
print(f"Total columns: {len(all_df.columns):,}")
print("\nRows by record_type:")
print(all_df["record_type"].value_counts())