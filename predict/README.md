# EarningALZ Prediction Pipeline with Cluster Assignment Support

This version fixes the "cluster label not found" problem.

## Why the old version failed

The previous prediction builder only looked for a community/cluster assignment file inside the Hugging Face two-part dataset. It did not properly support your local V4 clustering output.

Your V4 clustering script writes the selected assignment here:

```text
rag_chroma_output/cluster_method_comparison_v4/best_company_cluster_assignment.csv
```

That file is exactly what the prediction model should use.

## Run with local V4 cluster output

```bash
python run_prediction_pipeline.py \
  --out-dir prediction_model_outputs \
  --test-start-quarter 2024Q1 \
  --community-local-file rag_chroma_output/cluster_method_comparison_v4/best_company_cluster_assignment.csv \
  --write-csv-copy
```

Windows PowerShell:

```powershell
python run_prediction_pipeline.py `
  --out-dir prediction_model_outputs `
  --test-start-quarter 2024Q1 `
  --community-local-file rag_chroma_output\cluster_method_comparison_v4\best_company_cluster_assignment.csv `
  --write-csv-copy
```

## Direct build command

```bash
python 01_build_prediction_dataset.py \
  --out-dir prediction_model_outputs \
  --community-local-file rag_chroma_output/cluster_method_comparison_v4/best_company_cluster_assignment.csv \
  --write-csv-copy
```

## Accepted cluster assignment columns

The script supports common V4 columns:

```text
company_node
ticker
company
cluster_id
cluster_theme_label
```

It normalizes them to:

```text
company_node
community_id
community_label
```

Then it joins them to prediction events:

```text
source_node -> source_community_id
target_node -> target_community_id
```

and creates:

```text
same_community
community_pair
target_community_event_count
target_community_active_count
target_community_positive_share
target_community_negative_share
target_community_signal_balance
target_community_signal_score_mean
```

## Check whether cluster join worked

After running, inspect:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("prediction_model_outputs/prediction_dataset_cross_quarter.parquet")
print(df[["source_community_id","target_community_id","same_community"]].head())
print("source community coverage:", (df["source_community_id"]!="unknown").mean())
print("target community coverage:", (df["target_community_id"]!="unknown").mean())
print("same community rate:", df["same_community"].mean())
PY
```

If coverage is still near zero, the cluster assignment and event tables are using different company_node formats.
