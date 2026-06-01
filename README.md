# EearningAlz

This project studies information propagation across listed-company earnings-call transcripts and uses the propagation structure to support same-quarter or next-quarter narrative inference.

## Current Data Sources
- Earnings call transcript: https://www.earningscall.biz/
- Financial data reference: https://www.alphavantage.co/

## Existing Core Flow
1. Scrape or collect transcripts into `data/{TICKER}/Qx_YYYY.txt`.
2. Build Chroma index with `data/build_chroma_index_full.py`.
3. Retrieve related chunks with `RAG/rag.py`.
4. Generate answers with `LLM_run/run.py`.

## New Propagation Module
A dedicated generation module is now provided with separate code/results folders:
- Code: `/tmp/workspace/krsoy/EearningAlz/propagation_code`
- Results: `/tmp/workspace/krsoy/EearningAlz/propagation_results`

### New capabilities
- Cross-company information propagation extraction and scoring.
- Signal/entity extraction for each transcript node.
- Dark-theme dashboard with company/domain/time filtering and dynamic propagation paths.
- CrewAI-ready chunk validation pipeline for propagation explanation and grading.

## Quick Start
```bash
bash /tmp/workspace/krsoy/EearningAlz/setup.sh
uv run python /tmp/workspace/krsoy/EearningAlz/propagation_code/transcript_propagation_pipeline.py
uv run python /tmp/workspace/krsoy/EearningAlz/propagation_code/crewai_propagation_validator.py
uv run streamlit run /tmp/workspace/krsoy/EearningAlz/propagation_code/dashboard_app.py
```
