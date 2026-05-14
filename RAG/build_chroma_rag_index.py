import re
import json
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path("combined_hf_earnings_analysis/combined_transcripts_deduplicated.csv")

OUT_DIR = Path("rag_chroma_output")
CHROMA_DIR = OUT_DIR / "chroma_db"
CHUNK_INDEX_PATH = OUT_DIR / "chunk_index.csv"
CHUNK_INDEX_PARQUET_PATH = OUT_DIR / "chunk_index.parquet"
BUILD_SUMMARY_PATH = OUT_DIR / "build_summary.json"

COLLECTION_NAME = "earnings_call_chunks"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
MIN_CHUNK_WORDS = 40

# Set None for all transcripts.
MAX_TRANSCRIPTS = None

# Embedding batch size.
EMBED_BATCH_SIZE = 64

# Chroma insertion batch size.
CHROMA_ADD_BATCH_SIZE = 1000

# If True, delete old collection and rebuild from zero.
RESET_COLLECTION = True


# ============================================================
# HELPERS
# ============================================================

def normalize_space(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    return normalize_space(text).lower()


def md5_hash(text: str) -> str:
    return hashlib.md5(normalize_for_hash(text).encode("utf-8")).hexdigest()


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def make_quarter_from_date(date_value: str) -> str:
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return ""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def split_into_chunks(text: str, chunk_words=220, overlap=40):
    words = normalize_space(text).split()

    if len(words) < MIN_CHUNK_WORDS:
        return []

    if len(words) <= chunk_words:
        return [normalize_space(text)]

    chunks = []
    step = chunk_words - overlap

    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk_words_list = words[start:end]

        if len(chunk_words_list) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words_list))

        if end >= len(words):
            break

    return chunks


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Missing required column: text")

    default_columns = [
        "doc_id",
        "source_dataset",
        "ticker",
        "company",
        "title",
        "publish_date",
        "word_count",
        "content_hash"
    ]

    for col in default_columns:
        if col not in df.columns:
            df[col] = ""

    if df["doc_id"].astype(str).str.strip().eq("").all():
        df["doc_id"] = [f"doc_{i}" for i in range(len(df))]

    if df["word_count"].astype(str).str.strip().eq("").all():
        df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    if df["content_hash"].astype(str).str.strip().eq("").all():
        df["content_hash"] = df["text"].astype(str).apply(md5_hash)

    return df


# ============================================================
# BUILD CHUNK TABLE
# ============================================================

def build_chunk_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking transcripts"):
        doc_id = safe_str(row["doc_id"]).strip()
        if not doc_id:
            doc_id = f"doc_{row_idx}"

        text = safe_str(row["text"])
        chunks = split_into_chunks(text, CHUNK_WORDS, CHUNK_OVERLAP)

        publish_date = safe_str(row.get("publish_date", ""))
        quarter = make_quarter_from_date(publish_date)

        for chunk_id, chunk_text in enumerate(chunks):
            chunk_uid = f"{doc_id}__chunk_{chunk_id}"

            rows.append({
                "chunk_uid": chunk_uid,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source_dataset": safe_str(row.get("source_dataset", "")),
                "ticker": safe_str(row.get("ticker", "")),
                "company": safe_str(row.get("company", "")),
                "title": safe_str(row.get("title", "")),
                "publish_date": publish_date,
                "quarter": quarter,
                "transcript_word_count": safe_int(row.get("word_count", 0)),
                "content_hash": safe_str(row.get("content_hash", "")),
                "chunk_word_count": len(chunk_text.split()),
                "chunk_text": chunk_text,
                "chunk_hash": md5_hash(chunk_text)
            })

    chunk_df = pd.DataFrame(rows)

    if chunk_df.empty:
        raise ValueError("No chunks were created. Check transcript text and MIN_CHUNK_WORDS.")

    # Remove exact duplicate chunks.
    chunk_df = chunk_df.drop_duplicates(subset=["chunk_hash"]).reset_index(drop=True)

    # Ensure unique Chroma ids after duplicate removal.
    chunk_df["chunk_uid"] = [
        f"{row.doc_id}__chunk_{int(row.chunk_id)}__{i}"
        for i, row in enumerate(chunk_df.itertuples(index=False))
    ]

    return chunk_df


# ============================================================
# CHROMA INDEX
# ============================================================

def get_chroma_collection():
    OUT_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted old collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def make_chroma_metadata(row) -> dict:
    """
    Chroma metadata values should be simple scalar types:
    str, int, float, bool.
    Avoid None, list, dict.
    """
    return {
        "doc_id": str(row["doc_id"]),
        "chunk_id": int(row["chunk_id"]),
        "source_dataset": str(row["source_dataset"]),
        "ticker": str(row["ticker"]),
        "company": str(row["company"]),
        "title": str(row["title"])[:500],
        "publish_date": str(row["publish_date"]),
        "quarter": str(row["quarter"]),
        "transcript_word_count": int(row["transcript_word_count"]),
        "chunk_word_count": int(row["chunk_word_count"]),
        "content_hash": str(row["content_hash"]),
        "chunk_hash": str(row["chunk_hash"])
    }


def add_chunks_to_chroma(chunk_df: pd.DataFrame, model: SentenceTransformer, collection):
    documents = chunk_df["chunk_text"].astype(str).tolist()
    ids = chunk_df["chunk_uid"].astype(str).tolist()
    metadatas = [make_chroma_metadata(row) for _, row in chunk_df.iterrows()]

    total = len(documents)

    for start in tqdm(range(0, total, CHROMA_ADD_BATCH_SIZE), desc="Embedding + adding to Chroma"):
        end = min(start + CHROMA_ADD_BATCH_SIZE, total)

        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_metadatas = metadatas[start:end]

        embeddings = model.encode(
            batch_docs,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metadatas,
            embeddings=embeddings
        )


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}\n"
            "Please first create combined_transcripts_deduplicated.csv."
        )

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading transcripts...")
    df = pd.read_csv(INPUT_CSV)
    df = ensure_required_columns(df)

    df["word_count"] = df["word_count"].apply(lambda x: safe_int(x, 0))
    df = df[df["word_count"] >= 200].copy()

    if MAX_TRANSCRIPTS is not None:
        df = df.head(MAX_TRANSCRIPTS).copy()

    print("Transcripts used:", len(df))

    chunk_df = build_chunk_table(df)

    print("Chunks created:", len(chunk_df))
    print("Saving chunk index...")

    chunk_df.to_csv(CHUNK_INDEX_PATH, index=False)
    chunk_df.to_parquet(CHUNK_INDEX_PARQUET_PATH, index=False)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    collection = get_chroma_collection()

    add_chunks_to_chroma(chunk_df, model, collection)

    summary = {
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUT_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "transcripts_used": int(len(df)),
        "chunks_created": int(len(chunk_df)),
        "chroma_collection_count": int(collection.count()),
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
        "min_chunk_words": MIN_CHUNK_WORDS
    }

    with open(BUILD_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDONE.")
    print("Chroma DB:", CHROMA_DIR.resolve())
    print("Collection:", COLLECTION_NAME)
    print("Chunk index CSV:", CHUNK_INDEX_PATH.resolve())
    print("Chunk index Parquet:", CHUNK_INDEX_PARQUET_PATH.resolve())
    print("Build summary:", BUILD_SUMMARY_PATH.resolve())
    print("Chroma collection count:", collection.count())


if __name__ == "__main__":
    main()