#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG label-chunk retrieval directly from a Hugging Face hosted ChromaDB.

Purpose
-------
This script is the HF-ChromaDB version of the previous full_gpu_direct RAG
label/evidence retrieval script.

It does NOT run LLM extraction.
It uses SBERT query embeddings to select/label the most relevant chunks inside
each transcript/document for these evidence groups:

    1. relationship_chunks
    2. supply_chain_chunks
    3. expectation_chunks

Main difference from the old local version
------------------------------------------
Old version:
    - expected local rag_chroma_output/chroma_db
    - expected local rag_chroma_output/chunk_index.csv
    - loaded Chroma embeddings by chunk_uid from chunk_index.csv

This version:
    - can download chroma_db/ directly from Hugging Face dataset repo
    - reads chunks, embeddings, documents, and metadatas directly from ChromaDB
    - does not require chunk_index.csv
    - still performs per-document PyTorch/SBERT similarity search

Example
-------
Install:
    uv pip install huggingface_hub chromadb sentence-transformers pandas numpy tqdm

If using CUDA torch, install torch first, for example:
    uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
    uv pip install huggingface_hub chromadb sentence-transformers pandas numpy tqdm

Download HF chroma_db and run a small Intel test:
    uv run python rag_label_chunks_from_hf_chroma.py \
        --download-hf \
        --repo-id soysouce/earning_chroma \
        --hf-local-dir hf_earning_chroma \
        --chroma-dir hf_earning_chroma/chroma_db \
        --collection earnings_call_chunks \
        --ticker INTC \
        --quarters 2025Q2 2025Q3 \
        --max-docs 20 \
        --device cuda \
        --out-dir rag_chroma_output \
        --suffix hf_full_gpu_direct

Run after already downloaded:
    uv run python rag_label_chunks_from_hf_chroma.py \
        --chroma-dir hf_earning_chroma/chroma_db \
        --collection earnings_call_chunks \
        --ticker INTC \
        --quarters 2025Q2 2025Q3 \
        --max-docs 20 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# QUERY GROUPS
# ============================================================

QUERY_GROUPS: dict[str, list[str]] = {
    "relationship": [
        "supplier vendor upstream supply partner manufacturer component provider",
        "customer client downstream buyer OEM distributor channel reseller",
        "parent company holding company subsidiary business unit segment division acquisition acquired merger",
        "competitor partner ecosystem relationship contract agreement",
    ],
    "supply_chain": [
        "chip supply semiconductor supply component shortage supplier constraint constrained bottleneck",
        "raw material oil energy fuel natural gas commodity input cost",
        "inventory stock channel inventory destocking restocking backlog orders",
        "manufacturing capacity production capacity factory plant fab wafer foundry utilization yield",
        "logistics shipping freight delivery transportation lead time delay",
        "pricing pressure cost pressure gross margin margin input cost",
        "capital expenditure capex infrastructure data center cloud capacity buildout",
    ],
    "expectation": [
        "expect outlook guidance forecast anticipate next quarter coming quarter fiscal year",
        "we expect we believe we see we anticipate going forward",
        "demand outlook customer demand demand environment order trend",
        "supply outlook supply constraint expected capacity expected shortage",
        "margin outlook pricing outlook cost outlook gross margin outlook",
        "capex outlook investment plan capacity expansion future spending",
    ],
}

GROUP_TO_OUTPUT_KEY = {
    "relationship": "relationship_chunks",
    "supply_chain": "supply_chain_chunks",
    "expectation": "expectation_chunks",
}

DEFAULT_PHRASE_TERMS = [
    "supply chain",
    "gross margin",
    "data center",
    "raw material",
    "capital expenditure",
    "business unit",
    "parent company",
    "holding company",
    "going forward",
    "next quarter",
]


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label/retrieve evidence chunks from a Hugging Face ChromaDB source using SBERT query groups."
    )

    # HF source.
    p.add_argument("--download-hf", action="store_true", help="Download chroma_db/ from Hugging Face before running.")
    p.add_argument("--repo-id", default="soysouce/earning_chroma", help="Hugging Face dataset repo id.")
    p.add_argument("--hf-local-dir", default="hf_earning_chroma", help="Local folder for snapshot_download.")

    # Chroma.
    p.add_argument("--chroma-dir", default="hf_earning_chroma/chroma_db", help="Local path to the downloaded chroma_db folder.")
    p.add_argument("--collection", default="earnings_call_chunks", help="Chroma collection name.")
    p.add_argument("--list-collections", action="store_true", help="List collections and exit.")

    # Optional filters. These reduce source data before GPU search.
    p.add_argument("--ticker", action="append", default=[], help="Ticker filter. Can be repeated: --ticker INTC --ticker DELL")
    p.add_argument("--tickers", nargs="*", default=[], help="Ticker filter list, e.g. --tickers INTC DELL HPQ")
    p.add_argument("--quarters", nargs="*", default=[], help="Quarter filter, e.g. --quarters 2025Q2 2025Q3")
    p.add_argument("--max-docs", type=int, default=None, help="Limit number of documents/transcripts after filtering.")
    p.add_argument("--max-chunks", type=int, default=None, help="Limit number of chunks read from Chroma after filtering.")

    # Retrieval parameters.
    p.add_argument("--top-k-per-group", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=25)
    p.add_argument("--chroma-read-batch-size", type=int, default=20000)
    p.add_argument("--query-embed-batch-size", type=int, default=32)

    # Embedding/GPU.
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--require-cuda", action="store_true", help="Raise an error if CUDA is unavailable.")

    # Output.
    p.add_argument("--out-dir", default="rag_chroma_output")
    p.add_argument("--suffix", default="hf_full_gpu_direct")

    return p.parse_args()


# ============================================================
# SMALL HELPERS
# ============================================================


def normalize_space(text: Any) -> str:
    text = str(text if text is not None else "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_meta(metadata: dict | None, key: str, default: Any = "") -> Any:
    if metadata is None:
        return default
    value = metadata.get(key, default)
    if value is None:
        return default
    return value


def parse_int(value: Any, default: int = -1) -> int:
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


def get_query_terms(query_list: list[str]) -> list[str]:
    terms: list[str] = []
    for query in query_list:
        terms.extend(query.lower().split())
    terms.extend(DEFAULT_PHRASE_TERMS)
    return sorted(set(t for t in terms if t))


def choose_device(device_arg: str, require_cuda: bool) -> str:
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if require_cuda and device != "cuda":
        raise RuntimeError(
            "CUDA is required but not available or not selected. "
            "Install CUDA torch and/or run with a GPU allocation."
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was selected, but torch.cuda.is_available() is False.")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    print("Using device:", device)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu count:", torch.cuda.device_count())
        print("gpu name:", torch.cuda.get_device_name(0))

    return device


# ============================================================
# HUGGING FACE DOWNLOAD
# ============================================================


def download_hf_chroma_db(repo_id: str, hf_local_dir: str) -> Path:
    """Download only the ChromaDB folder from the HF dataset repo."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --download-hf. Install with: uv pip install huggingface_hub"
        ) from exc

    local_dir = Path(hf_local_dir)
    print("Downloading HF chroma_db/ ...")
    print("repo_id:", repo_id)
    print("local_dir:", local_dir)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[
            "chroma_db/**",
            "chroma_db/*",
            "chunk_index.csv",
            "rag_chroma_output/chunk_index.csv",
        ],
    )

    chroma_dir = local_dir / "chroma_db"
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Downloaded snapshot, but chroma_db was not found at: {chroma_dir}\n"
            "Check the dataset file layout."
        )

    print("Downloaded ChromaDB dir:", chroma_dir.resolve())
    return chroma_dir


# ============================================================
# CHROMA ACCESS
# ============================================================


def open_chroma_collection(chroma_dir: str, collection_name: str):
    chroma_path = Path(chroma_dir)
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {chroma_path}\n"
            "Use --download-hf first, or point --chroma-dir to the local downloaded chroma_db folder."
        )

    sqlite_path = chroma_path / "chroma.sqlite3"
    if not sqlite_path.exists():
        print(f"WARNING: chroma.sqlite3 not found at {sqlite_path}. Trying to open anyway.")

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)
    return client, collection


def list_collections(chroma_dir: str) -> None:
    chroma_path = Path(chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collections = client.list_collections()

    print("Collections:")
    for c in collections:
        name = c.name if hasattr(c, "name") else str(c)
        try:
            coll = client.get_collection(name=name)
            count = coll.count()
        except Exception:
            count = "?"
        print(f"  - {name}  count={count}")


def build_where_filter(tickers: list[str], quarters: list[str]) -> dict | None:
    filters = []

    tickers = [t.strip().upper() for t in tickers if str(t).strip()]
    quarters = [q.strip() for q in quarters if str(q).strip()]

    if tickers:
        if len(tickers) == 1:
            filters.append({"ticker": tickers[0]})
        else:
            filters.append({"ticker": {"$in": tickers}})

    if quarters:
        if len(quarters) == 1:
            filters.append({"quarter": quarters[0]})
        else:
            filters.append({"quarter": {"$in": quarters}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


# ============================================================
# LOAD CHUNKS DIRECTLY FROM CHROMA
# ============================================================


def load_chunks_directly_from_chroma(
    collection,
    where_filter: dict | None,
    read_batch_size: int,
    max_chunks: int | None,
    max_docs: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Read embeddings/documents/metadatas directly from ChromaDB.

    This replaces the old chunk_index.csv -> chunk_uid -> collection.get(ids=...) path.
    It is safer for HF shared ChromaDB because the HF source may not include the exact
    local chunk_index.csv from the original build.
    """

    total_count = collection.count()
    print("Chroma collection count:", total_count)
    print("where_filter:", where_filter)

    records: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    offset = 0
    selected_doc_ids: set[str] | None = None

    pbar_total = max_chunks if max_chunks is not None else total_count
    pbar = tqdm(total=pbar_total, desc="Reading chunks directly from ChromaDB")

    while True:
        if max_chunks is not None and len(records) >= max_chunks:
            break

        current_limit = read_batch_size
        if max_chunks is not None:
            current_limit = min(current_limit, max_chunks - len(records))
        if current_limit <= 0:
            break

        batch = collection.get(
            where=where_filter,
            include=["embeddings", "documents", "metadatas"],
            limit=current_limit,
            offset=offset,
        )

        ids = batch.get("ids", []) or []
        docs = batch.get("documents", []) or []
        metas = batch.get("metadatas", []) or []
        embs = batch.get("embeddings", []) or []

        if not ids:
            break

        for chroma_id, text, meta, emb in zip(ids, docs, metas, embs):
            if emb is None:
                continue

            doc_id = str(safe_meta(meta, "doc_id", ""))
            if not doc_id:
                # Last-resort fallback: use chroma_id prefix before chunk separator if any.
                doc_id = str(chroma_id).split("::")[0]

            if max_docs is not None:
                if selected_doc_ids is None:
                    selected_doc_ids = set()
                if doc_id not in selected_doc_ids:
                    if len(selected_doc_ids) >= max_docs:
                        # Skip chunks from new documents after max_docs is reached.
                        continue
                    selected_doc_ids.add(doc_id)

            embeddings.append(np.asarray(emb, dtype=np.float32))
            records.append(
                {
                    "chroma_id": str(chroma_id),
                    "doc_id": doc_id,
                    "chunk_id": parse_int(safe_meta(meta, "chunk_id", len(records))),
                    "chunk_text": normalize_space(text),
                    "source_dataset": str(safe_meta(meta, "source_dataset", "")),
                    "ticker": str(safe_meta(meta, "ticker", "")),
                    "company": str(safe_meta(meta, "company", "")),
                    "title": str(safe_meta(meta, "title", "")),
                    "publish_date": str(safe_meta(meta, "publish_date", "")),
                    "quarter": str(safe_meta(meta, "quarter", "")),
                    "transcript_word_count": parse_int(safe_meta(meta, "transcript_word_count", 0), default=0),
                    "metadata": meta or {},
                }
            )

        offset += len(ids)
        pbar.update(len(ids))

        # If max_docs is reached, we still continue reading current filtered result only if
        # returned batches may contain more chunks from already selected docs. To avoid scanning
        # millions of rows unnecessarily, stop once max_docs is reached and a batch adds no new
        # accepted record from selected docs. This keeps small tests fast.
        if max_docs is not None and selected_doc_ids is not None and len(selected_doc_ids) >= max_docs:
            # Good enough for testing / document-limited runs. Remove max_docs for full run.
            break

    pbar.close()

    if not records:
        raise RuntimeError(
            "No chunks loaded from ChromaDB. Check collection name, metadata keys, ticker/quarter filters, and HF download."
        )

    meta_df = pd.DataFrame(records)
    emb_matrix = np.vstack(embeddings).astype(np.float32)

    print("\nLoaded chunks:", len(meta_df))
    print("Embedding matrix shape:", emb_matrix.shape)
    print("Loaded documents:", meta_df["doc_id"].nunique())

    if "ticker" in meta_df.columns:
        print("\nTicker distribution:")
        print(meta_df["ticker"].value_counts(dropna=False).head(30).to_string())

    if "quarter" in meta_df.columns:
        print("\nQuarter distribution:")
        print(meta_df["quarter"].value_counts(dropna=False).sort_index().to_string())

    return meta_df, emb_matrix


# ============================================================
# DOC GROUPING
# ============================================================


def build_doc_groups(meta_df: pd.DataFrame) -> tuple[list[str], dict[str, list[int]]]:
    doc_to_indices: dict[str, list[int]] = defaultdict(list)

    for idx, doc_id in enumerate(meta_df["doc_id"].astype(str).tolist()):
        doc_to_indices[doc_id].append(idx)

    doc_ids = list(doc_to_indices.keys())

    print("Documents selected for retrieval:", len(doc_ids))
    print("Average chunks per document:", len(meta_df) / max(1, len(doc_ids)))
    print("\nChunks per document statistics:")
    print(meta_df.groupby("doc_id").size().describe().to_string())

    return doc_ids, doc_to_indices


def build_doc_metadata(meta_df: pd.DataFrame, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
    doc_meta: dict[str, dict[str, Any]] = {}
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
            "word_count": int(first.get("transcript_word_count", 0) or 0),
        }

    return doc_meta


# ============================================================
# QUERY EMBEDDINGS
# ============================================================


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    print(f"Loading embedding model on {device}: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print("Model loaded.")
    print("Model device:", model.device)
    return model


def precompute_query_embeddings(
    model: SentenceTransformer,
    device: str,
    query_embed_batch_size: int,
) -> dict[str, dict[str, Any]]:
    query_cache: dict[str, dict[str, Any]] = {}
    print("Precomputing query embeddings...")

    for group_name, query_list in QUERY_GROUPS.items():
        with torch.inference_mode():
            q_emb = model.encode(
                query_list,
                batch_size=query_embed_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=device,
            )

        q_emb = q_emb.to(device)
        q_emb = F.normalize(q_emb.float(), p=2, dim=1)

        query_cache[group_name] = {
            "query_list": query_list,
            "query_embeddings": q_emb,
            "query_terms": get_query_terms(query_list),
        }

        print(f"Query group: {group_name}, queries={len(query_list)}, dim={q_emb.shape[1]}")

    return query_cache


# ============================================================
# GPU / CPU SIMILARITY SEARCH
# ============================================================


def search_one_group_for_doc(
    meta_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    doc_indices: list[int],
    query_embeddings: torch.Tensor,
    query_terms: list[str],
    group_name: str,
    device: str,
    candidate_k: int,
    top_k_per_group: int,
) -> list[dict[str, Any]]:
    """
    For one document:
    - take its chunk embeddings
    - compute query_embeddings @ chunk_embeddings.T
    - take max similarity over query variants
    - rerank by hybrid score = semantic + keyword hit boost
    """
    if not doc_indices:
        return []

    doc_emb_np = emb_matrix[doc_indices]

    with torch.inference_mode():
        chunk_emb = torch.from_numpy(doc_emb_np).to(device)
        chunk_emb = F.normalize(chunk_emb.float(), p=2, dim=1)

        scores = query_embeddings @ chunk_emb.T
        best_scores, _best_query_idx = torch.max(scores, dim=0)

        k = min(candidate_k, best_scores.numel())
        top_scores, top_local_indices = torch.topk(best_scores, k=k, largest=True)

        top_scores_np = top_scores.detach().cpu().numpy()
        top_local_indices_np = top_local_indices.detach().cpu().numpy()

    ranked: list[dict[str, Any]] = []

    for score, local_idx in zip(top_scores_np, top_local_indices_np):
        global_idx = doc_indices[int(local_idx)]
        row = meta_df.iloc[global_idx]

        chunk_text = row["chunk_text"]
        keyword_count = keyword_score_chunk(chunk_text, query_terms)
        semantic_similarity = float(score)
        keyword_boost = min(keyword_count, 10) / 10.0

        matched_query_count = 1  # kept for schema compatibility
        query_hit_boost = min(matched_query_count, 5) / 5.0

        hybrid_score = semantic_similarity * 0.75 + keyword_boost * 0.15 + query_hit_boost * 0.10

        ranked.append(
            {
                "chroma_id": str(row["chroma_id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_id": int(row["chunk_id"]),
                "chunk_text": chunk_text,
                "metadata": row["metadata"],
                "best_similarity": semantic_similarity,
                "keyword_count": int(keyword_count),
                "matched_query_count": int(matched_query_count),
                "hybrid_score": float(hybrid_score),
                "query_group": group_name,
            }
        )

    ranked = sorted(ranked, key=lambda x: x["hybrid_score"], reverse=True)
    return ranked[:top_k_per_group]


def build_evidence_package_for_doc(
    meta_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    doc_id: str,
    doc_indices: list[int],
    doc_meta: dict[str, dict[str, Any]],
    query_cache: dict[str, dict[str, Any]],
    device: str,
    candidate_k: int,
    top_k_per_group: int,
) -> dict[str, Any]:
    info = doc_meta[doc_id]

    package: dict[str, Any] = {
        "doc_id": str(doc_id),
        "source_dataset": str(info.get("source_dataset", "")),
        "current_company": str(info.get("company", "")),
        "ticker": str(info.get("ticker", "")),
        "title": str(info.get("title", "")),
        "publish_date": str(info.get("publish_date", "")),
        "quarter": str(info.get("quarter", "")),
        "word_count": int(info.get("word_count", 0) or 0),
        "retrieved_evidence": {
            "relationship_chunks": [],
            "supply_chain_chunks": [],
            "expectation_chunks": [],
        },
    }

    for group_name in QUERY_GROUPS.keys():
        cache_item = query_cache[group_name]
        retrieved = search_one_group_for_doc(
            meta_df=meta_df,
            emb_matrix=emb_matrix,
            doc_indices=doc_indices,
            query_embeddings=cache_item["query_embeddings"],
            query_terms=cache_item["query_terms"],
            group_name=group_name,
            device=device,
            candidate_k=candidate_k,
            top_k_per_group=top_k_per_group,
        )

        output_key = GROUP_TO_OUTPUT_KEY[group_name]
        for item in retrieved:
            meta = item["metadata"]
            package["retrieved_evidence"][output_key].append(
                {
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
                        "quarter": str(safe_meta(meta, "quarter", "")),
                    },
                }
            )

    return package


# ============================================================
# OUTPUT
# ============================================================


def flatten_packages(packages: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for pkg in packages:
        base = {
            "doc_id": pkg["doc_id"],
            "source_dataset": pkg["source_dataset"],
            "ticker": pkg["ticker"],
            "company": pkg["current_company"],
            "title": pkg["title"],
            "publish_date": pkg["publish_date"],
            "quarter": pkg["quarter"],
            "word_count": pkg["word_count"],
        }

        for evidence_type, chunks in pkg["retrieved_evidence"].items():
            for item in chunks:
                rows.append(
                    {
                        **base,
                        "evidence_type": evidence_type,
                        "chroma_id": item["chroma_id"],
                        "chunk_id": item["chunk_id"],
                        "hybrid_score": item["hybrid_score"],
                        "semantic_similarity": item["semantic_similarity"],
                        "keyword_count": item["keyword_count"],
                        "matched_query_count": item["matched_query_count"],
                        "chunk_text": item["text"],
                    }
                )

    return pd.DataFrame(rows)


def output_paths(out_dir: str, suffix: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "evidence_jsonl": out / f"rag_evidence_packages_{suffix}.jsonl",
        "flat_csv": out / f"rag_evidence_chunks_flat_{suffix}.csv",
        "sample_agent_input": out / f"sample_agent_input_{suffix}.json",
        "summary": out / f"retrieval_summary_{suffix}.json",
    }


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    if args.download_hf:
        download_hf_chroma_db(args.repo_id, args.hf_local_dir)

    if args.list_collections:
        list_collections(args.chroma_dir)
        return

    device = choose_device(args.device, args.require_cuda)

    all_tickers = []
    all_tickers.extend(args.ticker or [])
    all_tickers.extend(args.tickers or [])
    all_tickers = [t.upper() for t in all_tickers if str(t).strip()]

    print("Opening ChromaDB as HF embedding storage...")
    _client, collection = open_chroma_collection(args.chroma_dir, args.collection)

    where_filter = build_where_filter(all_tickers, args.quarters)

    meta_df, emb_matrix = load_chunks_directly_from_chroma(
        collection=collection,
        where_filter=where_filter,
        read_batch_size=args.chroma_read_batch_size,
        max_chunks=args.max_chunks,
        max_docs=args.max_docs,
    )

    doc_ids, doc_to_indices = build_doc_groups(meta_df)
    doc_meta = build_doc_metadata(meta_df, doc_ids)

    print("\nLoading query embedding model...")
    model = load_embedding_model(args.embedding_model, device)
    query_cache = precompute_query_embeddings(model, device, args.query_embed_batch_size)

    packages: list[dict[str, Any]] = []

    for doc_id in tqdm(doc_ids, desc="RAG label-chunk retrieval"):
        pkg = build_evidence_package_for_doc(
            meta_df=meta_df,
            emb_matrix=emb_matrix,
            doc_id=doc_id,
            doc_indices=doc_to_indices[doc_id],
            doc_meta=doc_meta,
            query_cache=query_cache,
            device=device,
            candidate_k=args.candidate_k,
            top_k_per_group=args.top_k_per_group,
        )
        packages.append(pkg)

    paths = output_paths(args.out_dir, args.suffix)

    print("\nWriting outputs...")
    with paths["evidence_jsonl"].open("w", encoding="utf-8") as f:
        for pkg in packages:
            f.write(json.dumps(pkg, ensure_ascii=False) + "\n")

    flat_df = flatten_packages(packages)
    flat_df.to_csv(paths["flat_csv"], index=False)

    if packages:
        with paths["sample_agent_input"].open("w", encoding="utf-8") as f:
            json.dump(packages[0], f, indent=2, ensure_ascii=False)

    summary = {
        "mode": "hf_chromadb_direct_sbert_label_chunk_retrieval",
        "hf_repo_id": args.repo_id,
        "chroma_dir": str(args.chroma_dir),
        "collection_name": args.collection,
        "where_filter": where_filter,
        "tickers": all_tickers,
        "quarters": args.quarters,
        "target_chunks_loaded": int(len(meta_df)),
        "embedding_matrix_shape": list(emb_matrix.shape),
        "documents_retrieved": int(len(packages)),
        "flat_evidence_rows": int(len(flat_df)),
        "top_k_per_group": int(args.top_k_per_group),
        "candidate_k": int(args.candidate_k),
        "embedding_model": args.embedding_model,
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "output_jsonl": str(paths["evidence_jsonl"]),
        "output_flat_csv": str(paths["flat_csv"]),
        "sample_agent_input": str(paths["sample_agent_input"]),
        "retrieval_summary": str(paths["summary"]),
    }

    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDONE.")
    print("Evidence JSONL:", paths["evidence_jsonl"].resolve())
    print("Flat evidence CSV:", paths["flat_csv"].resolve())
    print("Sample Agent input:", paths["sample_agent_input"].resolve())
    print("Retrieval summary:", paths["summary"].resolve())

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
            "keyword_count",
        ]
        preview_cols = [c for c in preview_cols if c in flat_df.columns]
        print(flat_df[preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
