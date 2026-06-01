# Propagation Code Package

This folder contains new generation code for cross-company earnings-call information propagation analysis.

## Files
- `transcript_propagation_pipeline.py`: extracts narrative signals/entities, builds a quarter-aware propagation graph, and writes analysis artifacts.
- `dashboard_app.py`: dark-theme Streamlit dashboard with company/domain/time filtering and dynamic path visualization.
- `crewai_propagation_validator.py`: CrewAI-ready validation workflow for chunk-level evidence checks, path explanation, and scoring.

## Run
```bash
uv run python /tmp/workspace/krsoy/EearningAlz/propagation_code/transcript_propagation_pipeline.py
uv run python /tmp/workspace/krsoy/EearningAlz/propagation_code/crewai_propagation_validator.py
uv run streamlit run /tmp/workspace/krsoy/EearningAlz/propagation_code/dashboard_app.py
```

## Input Data
- Reads transcripts from `/tmp/workspace/krsoy/EearningAlz/data/{TICKER}/Qx_YYYY.txt`.

## Output Data
- Writes only to `/tmp/workspace/krsoy/EearningAlz/propagation_results`.
