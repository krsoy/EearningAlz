import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# GPU CONFIG
# ============================================================

REQUIRE_CUDA = True

if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required but not available. "
        "Check CUDA torch installation, GPU driver, or Slurm GPU allocation."
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")

print("Using device:", DEVICE)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = Path("rag_chroma_output")
CHROMA_DIR = OUT_DIR / "chroma_db"
CHUNK_INDEX_PATH = OUT_DIR / "chunk_index.csv"

COLLECTION_NAME = "earnings_call_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Full-data mode
PROCESS_FULL_DATA = True
OUTPUT_SUFFIX = "full_gpu_direct"

EVIDENCE_JSONL_PATH = OUT_DIR / f"rag_evidence_packages_{OUTPUT_SUFFIX}.jsonl"
EVIDENCE_FLAT_CSV_PATH = OUT_DIR / f"rag_evidence_chunks_flat_{OUTPUT_SUFFIX}.csv"
SAMPLE_AGENT_INPUT_PATH = OUT_DIR / f"sample_agent_input_{OUTPUT_SUFFIX}.json"
RETRIEVAL_SUMMARY_PATH = OUT_DIR / f"retrieval_summary_{OUTPUT_SUFFIX}.json"

TOP_K_PER_GROUP = 5
CANDIDATE_K = 25

# Set None for all documents.
# For testing, use for example: MAX_DOCS = 100
MAX_DOCS = None

# ChromaDB is used only as embedding storage.
# If memory or Chroma read has problems, reduce this to 5000 or 10000.
CHROMA_READ_BATCH_SIZE = 20000

QUERY_EMBED_BATCH_SIZE = 32


# ============================================================
# QUERY GROUPS
# ============================================================

QUERY_GROUPS = {
    "relationship": [
        "supplier vendor upstream supply partner manufacturer component provider",
        "customer client downstream buyer OEM distributor channel reseller",
        "parent company holding company subsidiary business unit segment division acquisition acquired merger",
        "competitor partner ecosystem relationship contract agreement"
    ],
    "supply_chain": [
        "chip supply semiconductor supply component shortage supplier constraint constrained bottleneck",
        "raw material oil energy fuel natural gas commodity input cost",
        "inventory stock channel inventory destocking restocking backlog orders",
        "manufacturing capacity production capacity factory plant fab wafer foundry utilization yield",
        "logistics shipping freight delivery transportation lead time delay",
        "pricing pressure cost pressure gross margin margin input cost",
        "capital expenditure capex infrastructure data center cloud capacity buildout"
    ],
    "expectation": [
        "expect outlook guidance forecast anticipate next quarter coming quarter fiscal year",
        "we expect we believe we see we anticipate going forward",
        "demand outlook customer demand demand environment order trend",
        "supply outlook supply constraint expected capacity expected shortage",
        "margin outlook pricing outlook cost outlook gross margin outlook",
        "capex outlook investment plan capacity expansion future spending"
    ]
}

GROUP_TO_OUTPUT_KEY = {
    "relationship": "relationship_chunks",
    "supply_chain": "supply_chain_chunks",
    "expectation": "expectation_chunks"
}


# ============================================================
# HELPERS
# ============================================================

def normalize_space(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_meta(metadata: dict, key: str, default=""):
    if metadata is None:
        return default
    value = metadata.get(key, default)
    if value is None:
        return default
    return value


def parse_int(value, default=-1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def keyword_score_chunk(chunk: str, query_terms: list[str]) -> int:
    text = str(chunk).lower()
    score = 0

    for term in query_terms:
        term = term.lower().strip()
        if not term:
            continue

        if " " in term:
            score += text.count(term)
        else:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
            score += len(re.findall(pattern, text, flags=re.IGNORECASE))

    return score


def get_query_terms(query_list):
    terms = []

    for query in query_list:
        query = query.lower()
        terms.extend(query.split())

    phrases = [
        "supply chain",
        "gross margin",
        "data center",
        "raw material",
        "capital expenditure",
        "business unit",
        "parent company",
        "holding company",
        "going forward",
        "next quarter"
    ]

    terms.extend(phrases)

    return sorted(set(terms))


def get_chroma_collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {CHROMA_DIR}\n"
            "Run build_chroma_rag_index.py first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


# ============================================================
# LOAD FULL CHROMA STORAGE BY chunk_uid
# ============================================================

def load_full_chunks_from_chroma(collection):
    """
    Full-data version based on build_chroma_rag_index.py.

    Build script stores ChromaDB ids as chunk_df["chunk_uid"].
    Therefore retrieval must use chunk_index.csv["chunk_uid"] directly.

    ChromaDB is used only as embedding storage:
    - chunk_index.csv provides all chunk_uid ids
    - collection.get(ids=chunk_uid) fetches embeddings/documents/metadatas
    - PyTorch GPU performs similarity search later
    """

    if not CHUNK_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"chunk_index.csv not found: {CHUNK_INDEX_PATH}\n"
            "Run build_chroma_rag_index.py first."
        )

    print("Reading chunk index:", CHUNK_INDEX_PATH)
    chunk_df = pd.read_csv(CHUNK_INDEX_PATH)

    print("Chunk index rows:", len(chunk_df))
    print("Chunk index columns:", chunk_df.columns.tolist())

    if "chunk_uid" not in chunk_df.columns:
        raise RuntimeError(
            "chunk_index.csv does not contain 'chunk_uid'. "
            "Your build_chroma_rag_index.py should save chunk_uid and use it as ChromaDB id."
        )

    required_cols = [
        "chunk_uid",
        "doc_id",
        "chunk_id",
        "source_dataset",
        "ticker",
        "company",
        "title",
        "publish_date",
        "quarter",
        "transcript_word_count",
        "chunk_word_count",
        "chunk_text",
        "content_hash",
        "chunk_hash"
    ]

    for col in required_cols:
        if col not in chunk_df.columns:
            chunk_df[col] = ""

    # ========================================================
    # Full data: no quarter/date filtering
    # ========================================================

    target_df = chunk_df.copy()
    target_df["chunk_uid"] = target_df["chunk_uid"].astype(str)
    target_df["doc_id"] = target_df["doc_id"].astype(str)

    if MAX_DOCS is not None:
        keep_doc_ids = (
            target_df["doc_id"]
            .drop_duplicates()
            .head(MAX_DOCS)
            .tolist()
        )
        target_df = target_df[target_df["doc_id"].isin(keep_doc_ids)].copy()

    print("Full-data mode:", PROCESS_FULL_DATA)
    print("Target chunks to load:", len(target_df))
    print("Target documents:", target_df["doc_id"].nunique())

    if "quarter" in target_df.columns:
        print("\nQuarter distribution from chunk_index:")
        print(target_df["quarter"].value_counts(dropna=False).sort_index())

    target_ids = target_df["chunk_uid"].astype(str).tolist()

    print("\nTarget Chroma IDs to load:", len(target_ids))
    print("Example chunk_uid:")
    print(target_ids[:5])

    records = []
    embeddings = []

    total_requested = 0
    total_returned = 0

    for start in tqdm(
        range(0, len(target_ids), CHROMA_READ_BATCH_SIZE),
        desc="Reading full embeddings from ChromaDB by chunk_uid"
    ):
        batch_ids = target_ids[start:start + CHROMA_READ_BATCH_SIZE]
        total_requested += len(batch_ids)

        batch = collection.get(
            ids=batch_ids,
            include=["embeddings", "documents", "metadatas"]
        )

        ids = batch.get("ids", [])
        docs = batch.get("documents", [])
        metas = batch.get("metadatas", [])
        embs = batch.get("embeddings", [])

        total_returned += len(ids)

        for chroma_id, text, meta, emb in zip(ids, docs, metas, embs):
            if emb is None:
                continue

            embeddings.append(np.asarray(emb, dtype=np.float32))

            records.append({
                "chroma_id": str(chroma_id),
                "doc_id": str(safe_meta(meta, "doc_id", "")),
                "chunk_id": parse_int(safe_meta(meta, "chunk_id", -1)),
                "chunk_text": normalize_space(text),
                "source_dataset": str(safe_meta(meta, "source_dataset", "")),
                "ticker": str(safe_meta(meta, "ticker", "")),
                "company": str(safe_meta(meta, "company", "")),
                "title": str(safe_meta(meta, "title", "")),
                "publish_date": str(safe_meta(meta, "publish_date", "")),
                "quarter": str(safe_meta(meta, "quarter", "")),
                "transcript_word_count": parse_int(
                    safe_meta(meta, "transcript_word_count", 0),
                    default=0
                ),
                "metadata": meta
            })

    print("\nChroma read check:")
    print("Requested ids:", total_requested)
    print("Returned ids:", total_returned)

    if total_returned != total_requested:
        print(
            "WARNING: Returned ids != requested ids. "
            "This usually means chroma_db and chunk_index.csv are not from the same build run."
        )

    if not records:
        raise RuntimeError(
            "No embeddings loaded from ChromaDB by chunk_uid. "
            "This means chunk_index.csv['chunk_uid'] does not match ChromaDB stored ids. "
            "Check whether chroma_db and chunk_index.csv come from the same build run."
        )

    meta_df = pd.DataFrame(records)
    emb_matrix = np.vstack(embeddings).astype(np.float32)

    print("\nLoaded chunks:", len(meta_df))
    print("Embedding matrix shape:", emb_matrix.shape)
    print("Loaded documents:", meta_df["doc_id"].nunique())

    if "quarter" in meta_df.columns:
        print("\nLoaded quarter distribution:")
        print(meta_df["quarter"].value_counts(dropna=False).sort_index())

    return meta_df, emb_matrix


# ============================================================
# DOC GROUPING
# ============================================================

def build_doc_groups(meta_df: pd.DataFrame):
    """
    Build mapping:
    doc_id -> row indices in meta_df / embedding matrix
    """

    doc_to_indices = defaultdict(list)

    for idx, doc_id in enumerate(meta_df["doc_id"].astype(str).tolist()):
        doc_to_indices[doc_id].append(idx)

    doc_ids = list(doc_to_indices.keys())

    print("Documents selected for retrieval:", len(doc_ids))
    print("Average chunks per document:", len(meta_df) / max(1, len(doc_ids)))

    print("\nChunks per document statistics:")
    print(
        meta_df.groupby("doc_id")
        .size()
        .describe()
    )

    return doc_ids, doc_to_indices


def build_doc_metadata(meta_df: pd.DataFrame, doc_ids: list[str]) -> dict:
    doc_meta = {}

    grouped = meta_df.groupby("doc_id", sort=False)

    for doc_id in doc_ids:
        rows = grouped.get_group(doc_id)
        first = rows.iloc[0]

        doc_meta[doc_id] = {
            "doc_id": str(doc_id),
            "source_dataset": str(first.get("source_dataset", "")),
            "ticker": str(first.get("ticker", "")),
            "company": str(first.get("company", "")),
            "title": str(first.get("title", "")),
            "publish_date": str(first.get("publish_date", "")),
            "quarter": str(first.get("quarter", "")),
            "word_count": int(first.get("transcript_word_count", 0))
        }

    return doc_meta


# ============================================================
# QUERY EMBEDDINGS
# ============================================================

def load_embedding_model():
    print(f"Loading embedding model on {DEVICE}: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=DEVICE
    )

    print("Model loaded.")
    print("Model device:", model.device)

    return model


def precompute_query_embeddings(model):
    """
    Encode query groups once on GPU.
    """

    query_cache = {}

    print("Precomputing query embeddings on GPU...")

    for group_name, query_list in QUERY_GROUPS.items():
        with torch.inference_mode():
            q_emb = model.encode(
                query_list,
                batch_size=QUERY_EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=DEVICE
            )

        q_emb = q_emb.to(DEVICE)
        q_emb = F.normalize(q_emb.float(), p=2, dim=1)

        query_cache[group_name] = {
            "query_list": query_list,
            "query_embeddings": q_emb,
            "query_terms": get_query_terms(query_list)
        }

        print(
            f"Query group: {group_name}, "
            f"queries={len(query_list)}, "
            f"dim={q_emb.shape[1]}"
        )

    return query_cache


# ============================================================
# GPU SIMILARITY SEARCH
# ============================================================

def gpu_search_one_group_for_doc(
    meta_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    doc_indices: list[int],
    query_embeddings: torch.Tensor,
    query_terms: list[str],
    group_name: str
):
    """
    True similarity search with PyTorch GPU.

    For one document:
    - take its chunk embeddings
    - move them to GPU
    - compute query_embeddings @ chunk_embeddings.T
    - take max similarity over query variants
    - take candidate top-k chunks
    - rerank by hybrid score
    """

    if not doc_indices:
        return []

    doc_emb_np = emb_matrix[doc_indices]

    with torch.inference_mode():
        chunk_emb = torch.from_numpy(doc_emb_np).to(DEVICE)
        chunk_emb = F.normalize(chunk_emb.float(), p=2, dim=1)

        scores = query_embeddings @ chunk_emb.T

        best_scores, best_query_idx = torch.max(scores, dim=0)

        candidate_k = min(CANDIDATE_K, best_scores.numel())

        top_scores, top_local_indices = torch.topk(
            best_scores,
            k=candidate_k,
            largest=True
        )

        top_scores = top_scores.detach().cpu().numpy()
        top_local_indices = top_local_indices.detach().cpu().numpy()

    ranked = []

    for score, local_idx in zip(top_scores, top_local_indices):
        global_idx = doc_indices[int(local_idx)]
        row = meta_df.iloc[global_idx]

        chunk_text = row["chunk_text"]
        keyword_count = keyword_score_chunk(chunk_text, query_terms)

        semantic_similarity = float(score)

        keyword_boost = min(keyword_count, 10) / 10.0

        # This direct GPU version ranks by max similarity over query variants.
        # matched_query_count is kept for schema compatibility.
        matched_query_count = 1
        query_hit_boost = min(matched_query_count, 5) / 5.0

        hybrid_score = (
            semantic_similarity * 0.75
            + keyword_boost * 0.15
            + query_hit_boost * 0.10
        )

        ranked.append({
            "chroma_id": str(row["chroma_id"]),
            "doc_id": str(row["doc_id"]),
            "chunk_id": int(row["chunk_id"]),
            "chunk_text": chunk_text,
            "metadata": row["metadata"],
            "best_similarity": semantic_similarity,
            "keyword_count": int(keyword_count),
            "matched_query_count": int(matched_query_count),
            "hybrid_score": float(hybrid_score),
            "query_group": group_name
        })

    ranked = sorted(ranked, key=lambda x: x["hybrid_score"], reverse=True)

    return ranked[:TOP_K_PER_GROUP]


def build_evidence_package_for_doc(
    meta_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    doc_id: str,
    doc_indices: list[int],
    doc_meta: dict,
    query_cache: dict
):
    info = doc_meta[doc_id]

    package = {
        "doc_id": str(doc_id),
        "source_dataset": str(info.get("source_dataset", "")),
        "current_company": str(info.get("company", "")),
        "ticker": str(info.get("ticker", "")),
        "title": str(info.get("title", "")),
        "publish_date": str(info.get("publish_date", "")),
        "quarter": str(info.get("quarter", "")),
        "word_count": int(info.get("word_count", 0)),
        "retrieved_evidence": {
            "relationship_chunks": [],
            "supply_chain_chunks": [],
            "expectation_chunks": []
        }
    }

    for group_name in QUERY_GROUPS.keys():
        cache_item = query_cache[group_name]

        retrieved = gpu_search_one_group_for_doc(
            meta_df=meta_df,
            emb_matrix=emb_matrix,
            doc_indices=doc_indices,
            query_embeddings=cache_item["query_embeddings"],
            query_terms=cache_item["query_terms"],
            group_name=group_name
        )

        output_key = GROUP_TO_OUTPUT_KEY[group_name]

        for item in retrieved:
            meta = item["metadata"]

            package["retrieved_evidence"][output_key].append({
                "chroma_id": item["chroma_id"],
                "chunk_id": item["chunk_id"],
                "hybrid_score": round(float(item["hybrid_score"]), 6),
                "semantic_similarity": round(float(item["best_similarity"]), 6),
                "keyword_count": int(item["keyword_count"]),
                "matched_query_count": int(item["matched_query_count"]),
                "text": item["chunk_text"],
                "metadata": {
                    "doc_id": str(safe_meta(meta, "doc_id", "")),
                    "ticker": str(safe_meta(meta, "ticker", "")),
                    "company": str(safe_meta(meta, "company", "")),
                    "publish_date": str(safe_meta(meta, "publish_date", "")),
                    "quarter": str(safe_meta(meta, "quarter", ""))
                }
            })

    return package


# ============================================================
# OUTPUT
# ============================================================

def flatten_packages(packages: list[dict]) -> pd.DataFrame:
    rows = []

    for pkg in packages:
        base = {
            "doc_id": pkg["doc_id"],
            "source_dataset": pkg["source_dataset"],
            "ticker": pkg["ticker"],
            "company": pkg["current_company"],
            "title": pkg["title"],
            "publish_date": pkg["publish_date"],
            "quarter": pkg["quarter"],
            "word_count": pkg["word_count"]
        }

        for evidence_type, chunks in pkg["retrieved_evidence"].items():
            for item in chunks:
                rows.append({
                    **base,
                    "evidence_type": evidence_type,
                    "chroma_id": item["chroma_id"],
                    "chunk_id": item["chunk_id"],
                    "hybrid_score": item["hybrid_score"],
                    "semantic_similarity": item["semantic_similarity"],
                    "keyword_count": item["keyword_count"],
                    "matched_query_count": item["matched_query_count"],
                    "chunk_text": item["text"]
                })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ChromaDB as embedding storage...")
    collection = get_chroma_collection()
    collection_count = collection.count()
    print("Chroma collection count:", collection_count)

    meta_df, emb_matrix = load_full_chunks_from_chroma(collection)

    doc_ids, doc_to_indices = build_doc_groups(meta_df)
    doc_meta = build_doc_metadata(meta_df, doc_ids)

    print("\nLoading query embedding model...")
    model = load_embedding_model()

    query_cache = precompute_query_embeddings(model)

    packages = []

    for doc_id in tqdm(doc_ids, desc="Full GPU direct similarity retrieval"):
        doc_indices = doc_to_indices[doc_id]

        pkg = build_evidence_package_for_doc(
            meta_df=meta_df,
            emb_matrix=emb_matrix,
            doc_id=doc_id,
            doc_indices=doc_indices,
            doc_meta=doc_meta,
            query_cache=query_cache
        )

        packages.append(pkg)

    print("\nWriting outputs...")

    with open(EVIDENCE_JSONL_PATH, "w", encoding="utf-8") as f:
        for pkg in packages:
            f.write(json.dumps(pkg, ensure_ascii=False) + "\n")

    flat_df = flatten_packages(packages)
    flat_df.to_csv(EVIDENCE_FLAT_CSV_PATH, index=False)

    if packages:
        with open(SAMPLE_AGENT_INPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(packages[0], f, indent=2, ensure_ascii=False)

    summary = {
        "mode": "full_gpu_direct_similarity_search_chromadb_storage_only",
        "process_full_data": bool(PROCESS_FULL_DATA),
        "collection_name": COLLECTION_NAME,
        "collection_count": int(collection_count),
        "target_chunks_loaded": int(len(meta_df)),
        "embedding_matrix_shape": list(emb_matrix.shape),
        "documents_retrieved": int(len(packages)),
        "flat_evidence_rows": int(len(flat_df)),
        "top_k_per_group": TOP_K_PER_GROUP,
        "candidate_k": CANDIDATE_K,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "output_jsonl": str(EVIDENCE_JSONL_PATH),
        "output_flat_csv": str(EVIDENCE_FLAT_CSV_PATH),
        "sample_agent_input": str(SAMPLE_AGENT_INPUT_PATH),
        "retrieval_summary": str(RETRIEVAL_SUMMARY_PATH)
    }

    with open(RETRIEVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDONE.")
    print("Evidence JSONL:", EVIDENCE_JSONL_PATH.resolve())
    print("Flat evidence CSV:", EVIDENCE_FLAT_CSV_PATH.resolve())
    print("Sample Agent input:", SAMPLE_AGENT_INPUT_PATH.resolve())
    print("Retrieval summary:", RETRIEVAL_SUMMARY_PATH.resolve())

    if not flat_df.empty:
        print("\nPreview:")
        preview_cols = [
            "ticker",
            "company",
            "quarter",
            "evidence_type",
            "chunk_id",
            "hybrid_score",
            "semantic_similarity",
            "keyword_count"
        ]
        preview_cols = [c for c in preview_cols if c in flat_df.columns]
        print(flat_df[preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()