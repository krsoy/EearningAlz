# Date-Aware Two-Part Network Analysis

This is the updated version of `run_two_part_network_prediction_analysis.py`.

The old script created same-quarter events using quarter labels only. It could not verify:

```text
source_publish_date < target_publish_date
```

This version adds publish-date joining and creates an ordered same-quarter subset.

## Date source

The script uses:

```text
soysouce/earningALZ_SBERT_evidence
rag_evidence_package_metadata_full_gpu_direct.parquet
```

Required columns:

```text
ticker
quarter
publish_date
```

The date join is:

```text
source_node + source_quarter -> source_publish_date
target_node + target_quarter -> target_publish_date
```

Then:

```text
publish_gap_days = target_publish_date - source_publish_date
source_before_target = publish_gap_days > 0
```

## Recommended command

```bash
python run_two_part_network_prediction_analysis_date_aware.py \
  --rag-output-dir rag_chroma_output \
  --out-dir rag_chroma_output/two_part_network_prediction_analysis_date_aware \
  --start-quarter 2019Q2 \
  --end-quarter 2026Q2 \
  --date-source hf \
  --evidence-dataset soysouce/earningALZ_SBERT_evidence \
  --evidence-metadata-file rag_evidence_package_metadata_full_gpu_direct.parquet \
  --write-parquet
```

Windows PowerShell:

```powershell
python run_two_part_network_prediction_analysis_date_aware.py `
  --rag-output-dir rag_chroma_output `
  --out-dir rag_chroma_output\two_part_network_prediction_analysis_date_aware `
  --start-quarter 2019Q2 `
  --end-quarter 2026Q2 `
  --date-source hf `
  --evidence-dataset soysouce/earningALZ_SBERT_evidence `
  --evidence-metadata-file rag_evidence_package_metadata_full_gpu_direct.parquet `
  --write-parquet
```

## Local metadata alternative

```bash
python run_two_part_network_prediction_analysis_date_aware.py \
  --rag-output-dir rag_chroma_output \
  --out-dir rag_chroma_output/two_part_network_prediction_analysis_date_aware \
  --metadata-parquet rag_evidence_package_metadata_full_gpu_direct.parquet \
  --write-parquet
```

## New outputs

```text
company_quarter_publish_dates.csv
date_coverage_report.csv

cross_quarter_events.csv
cross_quarter_summary_by_window_signal_relation.csv
cross_quarter_prediction_accuracy.csv

same_quarter_events.csv
same_quarter_summary_by_quarter_signal_relation.csv
same_quarter_correlation_by_signal_relation.csv

same_quarter_events_ordered.csv
same_quarter_ordered_summary_by_quarter_signal_relation.csv
same_quarter_ordered_prediction_by_signal_relation.csv

two_part_analysis_date_aware_summary.md
```

## Interpretation

### Cross-quarter

Cross-quarter remains quarter-level lead-lag.

The publish-date gap should not be interpreted as diffusion speed because the events are already separated by quarter.

### Same-quarter

All same-quarter rows measure network correlation / co-movement.

Only rows in:

```text
same_quarter_events_ordered.csv
```

are within-quarter prediction candidates because these satisfy:

```text
source_publish_date < target_publish_date
```
