# RAG Evidence Parquet Data Dictionary

Generated from `rag_chroma_output` evidence retrieval outputs.

## Files

### `rag_evidence_packages_full_gpu_direct.parquet`

One row per RAG evidence package. This is the Parquet version of:

```text
rag_evidence_packages_full_gpu_direct.jsonl
```

Nested evidence fields are serialized as JSON strings. Use this file if you need to reconstruct the full package sent to LLM agents.

### `rag_evidence_package_metadata_full_gpu_direct.parquet`

One row per RAG evidence package, but large nested evidence fields are removed. This is useful for joining with LLM outputs using document-level identifiers such as `doc_id`, `ticker`, `current_company`, `quarter`, or similar fields depending on the package schema.

### `rag_evidence_chunks_flat_full_gpu_direct.parquet`

One row per retrieved evidence chunk. This is the Parquet version of:

```text
rag_evidence_chunks_flat_full_gpu_direct.csv
```

This is the most useful file for evidence-chain auditing because it usually contains document id, chunk id, query group, rank, similarity score, and retrieved text or chunk metadata.

### `retrieval_summary_full_gpu_direct.parquet`

Parquet version of retrieval summary JSON.

### `sample_agent_input_full_gpu_direct.parquet`

Parquet version of the sample agent input JSON.

## Recommended evidence-chain joins

Use these identifiers where available:

```text
doc_id
ticker
current_company
quarter
chunk_id
chunk_uid
evidence_id
source_file
```

For public sharing, consider removing raw transcript text or long chunk text fields if source licenses do not allow redistribution.
