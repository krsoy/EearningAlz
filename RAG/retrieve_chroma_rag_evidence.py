import re
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = Path("rag_chroma_output")
CHROMA_DIR = OUT_DIR / "chroma_db"
CHUNK_INDEX_PATH = OUT_DIR / "chunk_index.csv"

COLLECTION_NAME = "earnings_call_chunks"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EVIDENCE_JSONL_PATH = OUT_DIR / "rag_evidence_packages.jsonl"
EVIDENCE_FLAT_CSV_PATH = OUT_DIR / "rag_evidence_chunks_flat.csv"
SAMPLE_AGENT_INPUT_PATH = OUT_DIR / "sample_agent_input.json"
RETRIEVAL_SUMMARY_PATH = OUT_DIR / "retrieval_summary.json"

TOP_K_PER_GROUP = 5

# Query more candidates first, then rerank.
CANDIDATE_K = 25

# Set None for all documents.
MAX_DOCS = None


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

    # Add several useful phrases manually.
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


def safe_meta(metadata: dict, key: str, default=""):
    if metadata is None:
        return default
    value = metadata.get(key, default)
    if value is None:
        return default
    return value


def get_chroma_collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {CHROMA_DIR}\n"
            "Run build_chroma_rag_index.py first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def load_doc_metadata() -> pd.DataFrame:
    if not CHUNK_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Chunk index not found: {CHUNK_INDEX_PATH}\n"
            "Run build_chroma_rag_index.py first."
        )

    chunks = pd.read_csv(CHUNK_INDEX_PATH)

    doc_cols = [
        "doc_id",
        "source_dataset",
        "ticker",
        "company",
        "title",
        "publish_date",
        "quarter",
        "transcript_word_count",
        "content_hash"
    ]

    for col in doc_cols:
        if col not in chunks.columns:
            chunks[col] = ""

    docs = (
        chunks[doc_cols]
        .drop_duplicates(subset=["doc_id"])
        .reset_index(drop=True)
    )

    if MAX_DOCS is not None:
        docs = docs.head(MAX_DOCS).copy()

    return docs


# ============================================================
# RETRIEVAL
# ============================================================

def query_one_group_for_doc(collection, model, doc_id: str, group_name: str, query_list: list[str]):
    """
    Retrieve only chunks from one transcript using Chroma where filter:
    where={"doc_id": doc_id}

    This is important because Agent 1 should extract information for one transcript,
    not mix evidence from other companies.
    """

    query_embeddings = model.encode(
        query_list,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=CANDIDATE_K,
        where={"doc_id": str(doc_id)},
        include=["documents", "metadatas", "distances"]
    )

    query_terms = get_query_terms(query_list)

    merged = {}

    ids_by_query = result.get("ids", [])
    docs_by_query = result.get("documents", [])
    metas_by_query = result.get("metadatas", [])
    dists_by_query = result.get("distances", [])

    for q_idx in range(len(docs_by_query)):
        ids = ids_by_query[q_idx]
        docs = docs_by_query[q_idx]
        metas = metas_by_query[q_idx]
        dists = dists_by_query[q_idx]

        for item_id, doc_text, meta, distance in zip(ids, docs, metas, dists):
            # For cosine space, Chroma returns distance.
            # Approximate similarity = 1 - distance.
            similarity = 1.0 - float(distance)

            keyword_count = keyword_score_chunk(doc_text, query_terms)

            if item_id not in merged:
                merged[item_id] = {
                    "chroma_id": item_id,
                    "chunk_id": int(safe_meta(meta, "chunk_id", -1)),
                    "chunk_text": normalize_space(doc_text),
                    "metadata": meta,
                    "best_similarity": similarity,
                    "keyword_count": keyword_count,
                    "matched_query_count": 1
                }
            else:
                merged[item_id]["best_similarity"] = max(
                    merged[item_id]["best_similarity"],
                    similarity
                )
                merged[item_id]["keyword_count"] = max(
                    merged[item_id]["keyword_count"],
                    keyword_count
                )
                merged[item_id]["matched_query_count"] += 1

    ranked = []

    for item in merged.values():
        # Simple rerank:
        # semantic similarity is main score;
        # keyword count and multiple query hits give small boosts.
        keyword_boost = min(item["keyword_count"], 10) / 10.0
        query_hit_boost = min(item["matched_query_count"], 5) / 5.0

        hybrid_score = (
            item["best_similarity"] * 0.75
            + keyword_boost * 0.15
            + query_hit_boost * 0.10
        )

        item["hybrid_score"] = hybrid_score
        item["query_group"] = group_name
        ranked.append(item)

    ranked = sorted(ranked, key=lambda x: x["hybrid_score"], reverse=True)

    return ranked[:TOP_K_PER_GROUP]


def build_evidence_package_for_doc(collection, model, doc_row) -> dict:
    doc_id = str(doc_row["doc_id"])

    package = {
        "doc_id": doc_id,
        "source_dataset": str(doc_row.get("source_dataset", "")),
        "current_company": str(doc_row.get("company", "")),
        "ticker": str(doc_row.get("ticker", "")),
        "title": str(doc_row.get("title", "")),
        "publish_date": str(doc_row.get("publish_date", "")),
        "quarter": str(doc_row.get("quarter", "")),
        "word_count": int(doc_row.get("transcript_word_count", 0)),
        "retrieved_evidence": {
            "relationship_chunks": [],
            "supply_chain_chunks": [],
            "expectation_chunks": []
        }
    }

    for group_name, query_list in QUERY_GROUPS.items():
        retrieved = query_one_group_for_doc(
            collection=collection,
            model=model,
            doc_id=doc_id,
            group_name=group_name,
            query_list=query_list
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
    print("Loading Chroma collection...")
    collection = get_chroma_collection()
    print("Collection count:", collection.count())

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    docs = load_doc_metadata()
    print("Documents to retrieve:", len(docs))

    packages = []

    for _, doc_row in tqdm(docs.iterrows(), total=len(docs), desc="Retrieving evidence"):
        pkg = build_evidence_package_for_doc(collection, model, doc_row)
        packages.append(pkg)

    with open(EVIDENCE_JSONL_PATH, "w", encoding="utf-8") as f:
        for pkg in packages:
            f.write(json.dumps(pkg, ensure_ascii=False) + "\n")

    flat_df = flatten_packages(packages)
    flat_df.to_csv(EVIDENCE_FLAT_CSV_PATH, index=False)

    if packages:
        with open(SAMPLE_AGENT_INPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(packages[0], f, indent=2, ensure_ascii=False)

    summary = {
        "collection_name": COLLECTION_NAME,
        "collection_count": int(collection.count()),
        "documents_retrieved": int(len(packages)),
        "flat_evidence_rows": int(len(flat_df)),
        "top_k_per_group": TOP_K_PER_GROUP,
        "candidate_k": CANDIDATE_K,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "output_jsonl": str(EVIDENCE_JSONL_PATH),
        "output_flat_csv": str(EVIDENCE_FLAT_CSV_PATH),
        "sample_agent_input": str(SAMPLE_AGENT_INPUT_PATH)
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
        print(flat_df[
            [
                "ticker",
                "company",
                "quarter",
                "evidence_type",
                "chunk_id",
                "hybrid_score",
                "keyword_count"
            ]
        ].head(30))


if __name__ == "__main__":
    main()