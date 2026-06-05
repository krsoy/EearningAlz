# EarningALZ JavaScript Dynamic Information Flow Plot V6

V6 fixes the time-gap interpretation.

## Key change

Cross-quarter and same-quarter flows are now treated differently.

### Cross-quarter mode

```text
source quarter t → target quarter t+1
```

This is a quarter-level lead-lag flow. The dashboard does **not** use publish-date gap as diffusion speed.

It displays:

```text
time interpretation: quarter-level lead-lag flow
window: 2024Q1 → 2024Q2
source publish date
target publish date
publish gap: not used for cross-quarter flow
```

### Same-quarter mode

```text
source quarter t → target quarter t
```

This is a date-level within-quarter flow. Here publish-date gap is meaningful.

It displays:

```text
same_quarter_publish_gap_days
source_before_target_rate
source_publish_date
target_publish_date
```

## Build cross-quarter AAPL JSON

```bash
python build_stock_network_json.py \
  --ticker AAPL \
  --mode cross_quarter \
  --signal All \
  --hop-depth 2 \
  --max-nodes 120 \
  --max-links 260 \
  --out data/aapl_network.json
```

## Build same-quarter ordered AAPL JSON

This is the most meaningful version for date-level information flow.

```bash
python build_stock_network_json.py \
  --ticker AAPL \
  --mode same_quarter \
  --ordered-same-quarter-only \
  --signal All \
  --hop-depth 2 \
  --max-nodes 120 \
  --max-links 260 \
  --out data/aapl_network.json
```

## Run

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000/aapl_dynamic_network_v6.html
```
