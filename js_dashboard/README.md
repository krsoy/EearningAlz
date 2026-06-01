# EarningALZ JavaScript Dynamic Information Flow Plot

This is a lightweight JavaScript/D3 version of the information-flow network view.

It is faster and smoother than a Streamlit network plot because the browser directly animates SVG elements.

## Files

```text
build_stock_network_json.py
aapl_dynamic_network.html
requirements.txt
data/
```

## Install

```bash
pip install -r requirements.txt
```

## Build AAPL JSON from Hugging Face

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

For a specific signal:

```bash
python build_stock_network_json.py \
  --ticker AAPL \
  --mode cross_quarter \
  --signal demand_outlook \
  --hop-depth 2 \
  --max-nodes 120 \
  --max-links 260 \
  --out data/aapl_network.json
```

## Run locally

Do not open the HTML by double clicking, because browser security may block local JSON loading.

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000/aapl_dynamic_network.html
```

## Change stock

Regenerate the JSON:

```bash
python build_stock_network_json.py --ticker NVDA --out data/aapl_network.json
```

The HTML will still load `data/aapl_network.json`, but the content will be centered on NVDA.
