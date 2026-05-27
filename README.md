# Earnings Call RAG–LLM–Network Analysis Python Workflow Summary

This document summarizes the Python workflow developed so far in the project. It covers the full process from raw earnings call transcript processing, RAG evidence construction, LLM-agent feature extraction, network analysis, cross-quarter propagation analysis, same-quarter correlation analysis, and the data outputs used for Overleaf writing.

---

## 0. Overall Research Goal

The project is not simply a sentiment analysis of individual earnings call transcripts. Instead, the goal is to transform earnings call transcripts into a **corporate intelligence network** and study:

1. whether inter-firm relationships can be extracted from earnings call evidence;
2. whether business signals expressed in earnings calls propagate through the relationship network;
3. whether a signal from a company in quarter `t` can predict a same-direction signal in a related company in quarter `t+1`;
4. whether connected firms show similar outlook signals within the same quarter;
5. which signals are transmitted and which are not, producing falsification / non-transmission evidence.

The final research framework is divided into two parts:

```text
Part A: Cross-quarter lead-lag prediction
source firm signal in quarter t
        ↓ network relationship
target firm same-direction signal in quarter t+1

Part B: Same-quarter network correlation
connected firms in the same quarter
        ↓
same-direction signal co-movement / contemporaneous correlation
```

---

# 1. Raw Data Integration and Deduplication

## 1.1 Input Data

The project uses multiple earnings call transcript sources, including:

```text
Motley Fool earnings call transcripts
Hugging Face earnings call transcript datasets
Other earnings-call-related datasets
```

The goal is to standardize transcripts from different sources into a unified format.

---

## 1.2 Standard Fields

The cleaned transcript dataset should contain at least the following fields:

```text
doc_id
ticker
company / current_company
date / publish_date
year
quarter
transcript text
```

The `quarter` field is standardized as:

```text
YYYYQ1
YYYYQ2
YYYYQ3
YYYYQ4
```

For example:

```text
2025Q2
2025Q3
2026Q1
```

---

## 1.3 Output

The cleaned and deduplicated main transcript file is:

```text
data/combined_transcript_data/combined_transcripts_deduplicated.csv
```

This file is the input for building the RAG index.

---

# 2. RAG ChromaDB Index Construction

Main script:

```text
RAG/build_chroma_rag_index.py
```

---

## 2.1 Goal

The goal is to split full transcripts into chunks, generate embeddings for each chunk, and store them in ChromaDB.

---

## 2.2 Main Steps

```text
1. Read combined_transcripts_deduplicated.csv
2. Split each transcript into chunks
3. Generate a unique chunk_uid for every chunk
4. Generate embeddings with SentenceTransformer
5. Store embeddings and metadata in ChromaDB
6. Save chunk_index.csv / chunk_index.parquet
7. Save build_summary.json
```

---

## 2.3 Role of ChromaDB

ChromaDB is mainly used as persistent storage for embedding chunks and metadata.

```text
ChromaDB = storage layer for embedded transcript chunks
```

It is not the core research model. Its role is to store, manage, and retrieve chunk-level transcript evidence.

---

## 2.4 Current Scale

The completed RAG index has the following scale:

```text
Transcript-level documents: 25,795
Total chunks: 1,151,208
Embedding dimension: 384
```

Main output directory:

```text
RAG/rag_chroma_output/
├── chroma_db/
├── chunk_index.csv
├── chunk_index.parquet
├── build_summary.json
```

---

# 3. RAG Evidence Retrieval

Main script:

```text
RAG/retrieve_chroma_rag_evidence_full_gpu_direct.py
```

Older script:

```text
RAG/retrieve_chroma_rag_evidence.py
```

---

## 3.1 Goal

The goal is to read chunk embeddings from ChromaDB and retrieve the most relevant evidence chunks for each transcript. These evidence chunks are then used as input for LLM-agent extraction.

---

## 3.2 Query Groups

Retrieval is organized into three query groups:

```text
relationship
supply_chain
expectation / outlook
```

Meaning:

```text
relationship:
    evidence about company relationships, upstream/downstream links,
    customers, suppliers, parent-subsidiary structures, partners, etc.

supply_chain:
    evidence about supply chains, inventory, capacity, raw materials,
    logistics, chips, cloud infrastructure, etc.

expectation / outlook:
    evidence about forward-looking expectations, demand, margin,
    capex, pricing, inventory outlook, etc.
```

---

## 3.3 GPU Direct Similarity Retrieval

The final retrieval method uses GPU-based direct similarity search:

```text
Load full embeddings from ChromaDB
        ↓
Load query embedding model on GPU
        ↓
Precompute query embeddings
        ↓
Group chunks by document
        ↓
Compute cosine similarity directly on GPU
        ↓
Output top evidence chunks for each transcript
```

This avoids the high overhead of repeatedly calling ChromaDB search for every document.

---

## 3.4 Output Files

Main outputs:

```text
RAG/rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl
RAG/rag_chroma_output/rag_evidence_chunks_flat_full_gpu_direct.csv
RAG/rag_chroma_output/retrieval_summary_full_gpu_direct.json
RAG/rag_chroma_output/sample_agent_input_full_gpu_direct.json
```

The most important file is:

```text
rag_evidence_packages_full_gpu_direct.jsonl
```

This is the input for LLM-agent extraction.

---

# 4. LLM Multi-Agent Extraction

Main script:

```text
RAG/extract_llm_agents_csv_vllm.py
```

---

## 4.1 Goal

The goal is to use a locally deployed vLLM server with Qwen2.5-14B-Instruct to convert RAG evidence packages into structured information.

---

## 4.2 Model Deployment

Model setup:

```text
vLLM
Qwen/Qwen2.5-14B-Instruct
4 × NVIDIA L4
tensor parallel size = 4
```

Each Slurm job uses:

```text
#SBATCH --gres=gpu:4
```

---

## 4.3 One Model Acting as Multiple Agents

The current design does not start multiple LLM models. Instead, it uses:

```text
single vLLM server
        ↓
different prompts
        ↓
different agent tasks
```

The three agents are:

```text
concepts agent
relationships agent
outlook agent
```

---

## 4.4 Concepts Agent

The concepts agent extracts binary supply-chain and operational concept features.

Typical fields:

```text
chip_supply
semiconductor_supply
raw_material_supply
oil_energy_supply
manufacturing_capacity
production_capacity
inventory_pressure
logistics_shipping
supplier_constraint
customer_demand
pricing_pressure
capex_expansion
data_center_capacity
cloud_infrastructure
labor_constraint
geopolitical_risk
```

Output files:

```text
concepts_*.csv
```

---

## 4.5 Relationships Agent

The relationships agent extracts company and entity relationships.

Typical relation groups:

```text
upstream
downstream
parent
subsidiary
partner
competitor
related
acquirer
acquired_company
customer_group
supplier_group
provider
internal
```

Typical output fields:

```text
doc_id
ticker
current_company
quarter
relation_group
entity
entity_type
relationship_type
confidence
evidence_chunk_ids
notes
```

Output files:

```text
relationships_*.csv
```

---

## 4.6 Outlook Agent

The outlook agent extracts forward-looking business signals.

The six standardized signals are:

```text
demand_outlook
supply_outlook
margin_outlook
capex_outlook
inventory_outlook
pricing_outlook
```

Typical labels:

```text
positive
negative
mixed
neutral
improving
worsening
increase
decrease
stable
not_mentioned
```

Output files:

```text
outlook_*.csv
```

---

## 4.7 Output Directories

Early Q2/Q3 test outputs:

```text
RAG/rag_chroma_output/llm_csv_outputs_2025Q2_Q3/
```

Later workload-balanced time-range outputs:

```text
RAG/rag_chroma_output/llm_csv_outputs_balanced_time_range/
```

Typical outputs:

```text
concepts_*.csv
relationships_*.csv
outlook_*.csv
failed_*.csv
progress_*.json
```

---

# 5. Slurm + vLLM Automation

Main Slurm script:

```text
run_llm_agents_balanced_time_range_4l4.slurm
```

Task table generation script:

```text
RAG/make_balanced_time_range_tasks.py
```

---

## 5.1 Why a Task Table Is Needed

The LLM extraction workload is too large to run in one job. Therefore, the workflow uses:

```text
Specify time range
        ↓
Count evidence packages in that range
        ↓
Split automatically by max-docs-per-task
        ↓
Generate Slurm array task file
```

---

## 5.2 Task Table Generation

Example:

```bash
cd ~/sem2/RAG

python make_balanced_time_range_tasks.py   --input-jsonl rag_chroma_output/rag_evidence_packages_full_gpu_direct.jsonl   --out-tsv ../llm_tasks_2024Q1_2026Q2_2000docs.tsv   --start-quarter 2024Q1   --end-quarter 2026Q2   --max-docs-per-task 2000   --run-prefix y2024q1_2026q2_2000docs
```

---

## 5.3 Sharding Logic

The current recommended setting is:

```text
max-docs-per-task = 2000
```

Reason:

```text
Earlier 144 docs took about 54 minutes.
Based on that speed, 2000 docs should run for around 9.5–10.5 hours.
```

This matches the target of approximately 10 hours per shard.

---

## 5.4 Job Submission

Example:

```bash
cd ~/sem2

N=$(tail -n +2 llm_tasks_2024Q1_2026Q2_2000docs.tsv | wc -l)

TASK_FILE=~/sem2/llm_tasks_2024Q1_2026Q2_2000docs.tsv sbatch --array=0-$((N-1))%2 run_llm_agents_balanced_time_range_4l4.slurm
```

Where:

```text
%2 = run at most two 4×L4 jobs at the same time
```

If the job shows:

```text
QOSMaxGRESPerUser
```

it means the current GPU/GRES limit has been reached. If the job is still in the queue, it does not need to be resubmitted. It will start automatically after previous GPU jobs finish and release resources.

---

# 6. Task Range Planning

Several time ranges have been planned so far.

---

## 6.1 2025Q2–2025Q3 Test Task

Used for early pipeline validation:

```text
2025Q2 + 2025Q3
NUM_SHARDS = 6
```

Main purpose:

```text
LLM extraction
network visualization
contagion analysis
falsification analysis
```

---

## 6.2 2024Q1–2026Q2 Task

Used to expand more recent time periods:

```text
2024Q1 to 2026Q2
max-docs-per-task = 2000
```

---

## 6.3 Pre-2024 Task

Used to process earlier historical data:

```text
2019Q1 to 2023Q4
Actual available data: 2019Q2 to 2023Q1
selected_doc_count = 17,172
num_shards = 9
max-docs-per-task = 2000
```

The generated task tags are:

```text
pre2024_2000docs_s000_of009
...
pre2024_2000docs_s008_of009
```

---

# 7. Master Network + Contagion Analysis

Main script:

```text
RAG/run_network_contagion_master_analysis.py
```

---

## 7.1 Goal

This script automatically scans existing LLM output folders, merges all:

```text
concepts_*.csv
relationships_*.csv
outlook_*.csv
```

Then it generates:

```text
cleaned tables
network nodes / edges
contagion events
falsification summaries
network figures
analysis summary
```

---

## 7.2 Input Directory

Example:

```text
RAG/rag_chroma_output/
```

The script recursively scans folders such as:

```text
llm_csv_outputs_2025Q2_Q3/
llm_csv_outputs_balanced_time_range/
other existing extraction output directories
```

---

## 7.3 Cleaning Logic

### Outlook Cleaning

The script maps LLM schema drift back to standardized signals.

Examples:

```text
supply_chain_outlook → supply_outlook
production_outlook → supply_outlook
loan_growth_outlook → demand_outlook
credit_quality_outlook → margin_outlook
capital_generation_outlook → capex_outlook
```

Labels are mapped to scores:

```text
positive / improving / increase = +1
negative / worsening / decrease = -1
mixed = 0.5
neutral / stable = 0
not_mentioned = NaN
```

---

### Relationship Cleaning

The script handles schema drift such as:

```text
relation_group = upstream|downstream|parent|subsidiary|related
```

and splits it into separate relationship categories:

```text
upstream
downstream
parent
subsidiary
related
```

---

## 7.4 Current Master Analysis Results

Current results:

```text
Concepts files: 13
Relationships files: 13
Outlook files: 14
Failed files: 6

Cleaned concepts rows: 11,087
Cleaned relationship rows: 40,070
Cleaned outlook rows: 58,584

Network nodes: 16,914
Network edges: 23,189

Contagion event rows for 2025Q2 → 2025Q3: 498
Unmatched relationship entity rows: 4,049
```

---

## 7.5 Outputs

```text
rag_chroma_output/network_contagion_master_analysis/
├── input_file_manifest.csv
├── cleaned_concepts_all.csv
├── cleaned_relationships_all.csv
├── cleaned_outlook_all.csv
├── company_quarter_signal_matrix.csv
├── network_nodes.csv
├── network_edges.csv
├── network_centrality_degree.csv
├── contagion_events.csv
├── contagion_summary_by_signal_label_relation.csv
├── falsification_cases_margin_outlook_improving.csv
├── falsification_summary_margin_outlook_improving.csv
├── relationship_network_interactive.html
├── analysis_summary.md
└── figures/
```

---

# 8. Network Visualization

Early scripts:

```text
RAG/visualize_information_flow_network.py
RAG/visualize_information_flow_network_v2.py
```

---

## 8.1 Goal

These scripts convert extracted:

```text
relationships
outlook signals
```

into network visualizations.

---

## 8.2 Issue with V1 Network

V1 used a strict definition:

```text
source company in 2025Q2 has signal
relationship target matches another transcript company
target company in 2025Q3 has same signal
```

Because of this strict requirement, early output contained only:

```text
propagation_events.csv rows = 6
network_nodes.csv rows = 4
network_edges.csv rows = 2
```

The reason was not lack of data. Many target entities were generic entities such as:

```text
cloud customers
suppliers
OEM customers
partners
```

These could not be matched to listed companies.

---

## 8.3 V2 Network Improvement

V2 defines three types of edges:

```text
relationship_signal_flow:
    source company signal → extracted relationship entity

matched_temporal_flow:
    source company signal → matched target company next-quarter signal

same_company_temporal_flow:
    same company signal continuation across quarters
```

This version is better for visualizing information flow.

---

## 8.4 Outputs

```text
information_flow_network_v2.html
information_flow_network_v2_static.png
flow_edges_v2.csv
flow_nodes_v2.csv
```

---

# 9. Contagion / Transmission Analysis

Main script:

```text
RAG/analyze_signal_contagion.py
```

---

## 9.1 Goal

This script answers questions such as:

```text
If a source company has margin_outlook = improving in 2025Q2,
do its upstream / downstream / related / parent / subsidiary target companies
also show margin_outlook = improving in 2025Q3?
```

---

## 9.2 Metrics

```text
exposed_edges:
    the source has an active signal and a relationship edge to a matched target company

exact_label_transmission_rate:
    the target shows the exact same label in the next quarter

same_direction_transmission_rate:
    the target shows a label with the same direction in the next quarter

target_active_rate:
    the target has any active state for the same signal in the next quarter
```

---

## 9.3 Early Result for margin_outlook = improving

In the 2025Q2 → 2025Q3 pilot window:

```text
downstream:
    exposed_edges = 18
    exact_transmission_rate = 44.4%
    falsification_rate = 55.6%

upstream:
    exposed_edges = 8
    exact_transmission_rate = 50.0%
    falsification_rate = 50.0%

related:
    exposed_edges = 13
    exact_transmission_rate = 61.5%
    falsification_rate = 38.5%
```

This shows that:

```text
Signal propagation is selective and relationship-dependent rather than automatic.
```

---

# 10. Falsification / Non-Transmission Analysis

Main script:

```text
RAG/analyze_signal_falsification.py
```

---

## 10.1 Goal

The goal is not only to identify successful transmission, but also to identify falsification cases:

```text
source firm: margin_outlook = improving
target firm: does not show improving / does not show same-direction signal in the next quarter
```

---

## 10.2 Outputs

```text
falsification_cases_margin_outlook_improving.csv
falsification_summary_margin_outlook_improving.csv
falsification_reason_counts_margin_outlook_improving.csv
falsification_summary.md
```

---

## 10.3 Falsification Types

```text
target_not_mentioned
target_neutral_or_stable
opposite_negative
target_mixed
not_transmitted
not_exact_but_same_direction
transmitted_exact
```

---

## 10.4 Research Value

This part is important because it prevents the analysis from becoming a forced narrative.

The framework identifies both:

```text
transmission
```

and:

```text
non-transmission / falsification
```

Therefore, the project can argue that:

```text
information propagation is selective rather than automatic
```

---

# 11. Rolling Full-Range Contagion Analysis

Main script:

```text
RAG/run_rolling_full_range_contagion.py
```

---

## 11.1 Why This Script Is Needed

The early master analysis only computed:

```text
2025Q2 → 2025Q3
```

However, the final research should not rely on a single window. Therefore, rolling analysis is needed for all adjacent quarters:

```text
2019Q2 → 2019Q3
2019Q3 → 2019Q4
...
2026Q1 → 2026Q2
```

---

## 11.2 Function

The script automatically computes all adjacent-quarter contagion windows:

```text
source quarter t
        ↓
target quarter t+1
```

Outputs:

```text
rolling_contagion_events_all.csv
rolling_contagion_summary_by_window_signal_relation.csv
rolling_contagion_summary_aggregated.csv
rolling_falsification_cases_margin_outlook_improving.csv
rolling_falsification_summary_margin_outlook_improving.csv
rolling_signal_quarter_label_counts.csv
rolling_window_summary.csv
rolling_analysis_summary.md
```

---

## 11.3 Figure Outputs

```text
figures/rolling_event_rows_by_window.png
figures/rolling_transmission_rate_by_relation.png
figures/rolling_transmission_rate_by_signal.png
figures/rolling_falsification_rate_margin_outlook_improving.png
```

---

## 11.4 Interpretation

This script is used for the long-term complete analysis:

```text
calculate rolling signal transmission across all completed quarters
```

---

# 12. Two-Part Network Prediction Analysis

Main script:

```text
RAG/run_two_part_network_prediction_analysis.py
```

This is the latest and most important analysis workflow.

---

## 12.1 Why the Analysis Is Split into Two Parts

A key research issue is:

```text
earnings call events for companies in the relationship network occur at different times
```

Therefore, the final analysis is split into:

```text
Part A: Cross-quarter lead-lag prediction
Part B: Same-quarter network correlation
```

---

# 12A. Part A: Cross-Quarter Lead-Lag Prediction

## 12A.1 Research Question

```text
source firm has a signal in quarter t
        ↓
relationship edge
target firm shows same-direction signal in quarter t+1?
```

---

## 12A.2 Outputs

```text
cross_quarter_events.csv
cross_quarter_summary_by_window_signal_relation.csv
cross_quarter_prediction_accuracy.csv
```

Figures:

```text
figures/cross_quarter_event_rows_by_window.png
figures/cross_quarter_accuracy_by_signal.png
figures/cross_quarter_accuracy_by_relation.png
```

---

## 12A.3 Current Scale

```text
Cross-quarter event rows: 75,282
Adjacent quarter windows: 19
```

---

## 12A.4 Current Important Results

Top cross-quarter results include:

```text
demand_outlook = positive, partner:
    exposed_edges = 1,124
    direction_match_rate = 73.2%

demand_outlook = positive, upstream:
    exposed_edges = 980
    direction_match_rate = 73.2%

demand_outlook = positive, customer_group:
    exposed_edges = 52
    direction_match_rate = 88.5%

margin_outlook = improving, acquirer:
    exposed_edges = 112
    direction_match_rate = 73.2%
```

---

## 12A.5 Metric Interpretation

An earlier table included both:

```text
Direction match rate
Prediction accuracy
```

However, under the current definition, these two are mathematically identical:

```text
prediction_accuracy = direction_match_edges / exposed_edges
direction_match_rate = direction_match_edges / exposed_edges
```

Therefore, the Overleaf table should avoid showing both columns. It should keep:

```text
Exposed edges
Target active rate
Exact match rate
Direction match rate
```

The explanation should be:

```text
Direction match rate is a preliminary transcript-signal prediction measure.
It is not a full machine-learning forecasting accuracy metric.
```

---

# 12B. Part B: Same-Quarter Network Correlation

## 12B.1 Research Question

```text
Within the same quarter, do connected firms show same-direction signals?
```

---

## 12B.2 Why It Should Not Be Called Direct Prediction Yet

Companies report earnings on different dates within the same quarter.

Without exact earnings call dates:

```text
same-quarter analysis = network correlation / co-movement
```

If exact earnings call dates are added later, this can be upgraded to:

```text
within-quarter lead-lag prediction
```

---

## 12B.3 Outputs

```text
same_quarter_events.csv
same_quarter_summary_by_quarter_signal_relation.csv
same_quarter_correlation_by_signal_relation.csv
```

Figures:

```text
figures/same_quarter_event_rows_by_quarter.png
figures/same_quarter_similarity_by_signal.png
figures/same_quarter_similarity_by_relation.png
```

---

## 12B.4 Current Scale

```text
Same-quarter event rows: 99,090
```

---

## 12B.5 Current Important Results

```text
demand_outlook = positive, customer_group:
    exposed_edges = 74
    direction_match_rate = 85.1%

demand_outlook = mixed;positive, partner:
    exposed_edges = 17
    direction_match_rate = 88.2%

margin_outlook = improving;stable, partner:
    exposed_edges = 12
    direction_match_rate = 83.3%

margin_outlook = improving, acquired_company:
    exposed_edges = 117
    direction_match_rate = 77.8%
```

---

# 13. Overleaf Writing Workflow

## 13.1 Problem with the Old Version

The old Overleaf version was based on an early feasibility study and contained outdated statistics:

```text
1,831 transcripts
1,703 tickers
2024Q3 / 2024Q4 only
supply-chain dictionary counts
11 million words
```

These are no longer suitable for the current study and should be removed.

---

## 13.2 Updated Overleaf Version

The latest Overleaf file is:

```text
updated_two_part_network_prediction_overleaf.tex
```

Core structure:

```text
Abstract
Introduction
Literature Review
Data
Method
Results
Discussion
Conclusion
References
```

---

## 13.3 Current Overleaf Focus

The paper has been updated to focus on:

```text
RAG-LLM pipeline
corporate intelligence network
cross-quarter lead-lag prediction
same-quarter network correlation
transmission and non-transmission
```

---

## 13.4 Figures to Insert

Two-part analysis figures:

```text
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_event_rows_by_window.png
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_accuracy_by_signal.png
rag_chroma_output/two_part_network_prediction_analysis/figures/cross_quarter_accuracy_by_relation.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_event_rows_by_quarter.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_similarity_by_signal.png
rag_chroma_output/two_part_network_prediction_analysis/figures/same_quarter_similarity_by_relation.png
```

---

# 14. Current Final Research Results

## 14.1 Data Scale

```text
Cleaned outlook rows: 58,584
Cleaned relationship rows: 40,070
Matched company relationships: 9,833
Unmatched relationship entities: 28,556
Available quarters: 21
Adjacent-quarter windows: 19
```

---

## 14.2 Cross-Quarter Lead-Lag Prediction

```text
Cross-quarter event rows: 75,282
```

Main findings:

```text
Positive demand_outlook signals show strong lead-lag structure.
Partner and upstream relationships show large-sample direction match rates around 73%.
Customer_group relationships show a higher direction match rate, around 88.5%, with fewer exposures.
Improving margin_outlook also shows meaningful cross-quarter structure in some relationship groups.
```

---

## 14.3 Same-Quarter Network Correlation

```text
Same-quarter event rows: 99,090
```

Main findings:

```text
Connected firms often show same-direction outlook signals within the same quarter.
demand_outlook and margin_outlook show strong contemporaneous co-movement.
Same-quarter results should be treated as network correlation unless exact earnings call dates are used.
```

---

## 14.4 Research Conclusion

Current results support the following conclusions:

```text
Earnings call signals are not isolated firm-level text features.
They can be structured into a corporate intelligence network.
Cross-quarter signals provide preliminary lead-lag predictive information.
Same-quarter connected firms show meaningful network co-movement.
Information propagation is selective and relationship-dependent.
```

---

# 15. Future Python Workflow Improvements

## 15.1 Add Exact Earnings Call Dates

Currently, same-quarter analysis should only be called:

```text
same-quarter network correlation
```

If exact call dates are added, it can be upgraded to:

```text
within-quarter lead-lag prediction
```

Example:

```text
Company A call date: 2025-07-20
Company B call date: 2025-08-05

A signal before B signal
        ↓
same-quarter intra-quarter prediction
```

---

## 15.2 Add Negative Cases and Baselines

The current cross-quarter measure is based on source-active exposure cases.

A more complete prediction model requires:

```text
positive exposure cases
negative exposure cases
randomized network baseline
sector-controlled baseline
company fixed effects
quarter fixed effects
```

---

## 15.3 Add Financial Outcomes

The final prediction target should move from transcript signals to real financial variables:

```text
revenue growth
gross margin change
inventory change
capex change
cash flow change
stock return
volatility
earnings surprise
```

---

## 15.4 Improve Entity Resolution

There are still many unmatched entities:

```text
Unmatched relationship entities: 28,556
```

Future improvements should include:

```text
company alias dictionary
ticker mapping
subsidiary-parent mapping
supply-chain database matching
manual high-frequency entity normalization
```

---

## 15.5 Randomized Network Tests

To verify that the observed patterns are not random, future tests should include:

```text
shuffle edges
shuffle quarters
shuffle source-target pairs
compare observed transmission rate against random baseline
```

---

# 16. Recommended Final Python File List

```text
RAG/build_chroma_rag_index.py
RAG/retrieve_chroma_rag_evidence_full_gpu_direct.py
RAG/extract_llm_agents_csv_vllm.py

RAG/make_balanced_time_range_tasks.py
run_llm_agents_balanced_time_range_4l4.slurm

RAG/run_network_contagion_master_analysis.py
RAG/run_rolling_full_range_contagion.py
RAG/run_two_part_network_prediction_analysis.py

RAG/analyze_signal_contagion.py
RAG/analyze_signal_falsification.py
RAG/visualize_information_flow_network_v2.py
```

---

# 17. One-Sentence Summary

The current Python pipeline has completed the following workflow:

```text
Raw earnings call transcripts
        ↓
RAG chunk embedding storage
        ↓
RAG evidence retrieval
        ↓
vLLM multi-agent extraction
        ↓
structured concepts / relationships / outlook signals
        ↓
corporate intelligence network
        ↓
cross-quarter prediction analysis
        ↓
same-quarter network correlation analysis
        ↓
Overleaf research results
```

The core research conclusion is:

```text
Earnings call transcripts can be transformed into a structured corporate intelligence network.
Cross-quarter network signals provide preliminary lead-lag predictive value.
Same-quarter connected firms show meaningful signal co-movement.
Information propagation is selective and relationship-dependent.
```
