# Current System Understanding (English)

## Existing Project Purpose
The current repository aims to use listed-company earnings-call transcripts to study information spread across firms and support transcript inference for the same quarter or next quarter.

## Existing Workflow (Before This Update)
1. Transcript scraping scripts pull earnings-call text from earningscall.biz.
2. Transcripts are stored under `data/{TICKER}/Qx_YYYY.txt`.
3. `data/build_chroma_index_full.py` chunks and embeds transcript text into ChromaDB.
4. `RAG/rag.py` retrieves relevant transcript chunks from Chroma by embedding similarity.
5. `LLM_run/run.py` generates an answer using a local instruction LLM with selected transcript context.

## Existing Data Sources
- Earnings call transcripts: https://www.earningscall.biz/
- Financial data intent noted in README: https://www.alphavantage.co/

## Gaps Identified
- No explicit propagation graph between companies.
- No dashboard for dynamic diffusion exploration.
- No chunk-level propagation validation workflow.
- Existing test script is interactive and not CI-safe.

## New Additions in This Task
- Dedicated generation code folder: `/tmp/workspace/krsoy/EearningAlz/propagation_code`
- Dedicated generation results folder: `/tmp/workspace/krsoy/EearningAlz/propagation_results`
- Propagation extraction + scoring pipeline.
- Dark-theme dashboard for company/domain/time filtering and path display.
- CrewAI-ready chunk validation and explanation workflow.
