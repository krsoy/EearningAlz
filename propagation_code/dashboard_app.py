import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.getenv("PROPAGATION_RESULTS_DIR", str(PROJECT_ROOT / "propagation_results")))
NODE_FILE = ROOT / "propagation_nodes.csv"
EDGE_FILE = ROOT / "propagation_edges.csv"


def load_nodes() -> List[Dict[str, str]]:
    if not NODE_FILE.exists():
        st.error(f"Missing node file: {NODE_FILE}. Run transcript_propagation_pipeline.py first.")
        st.stop()
    with NODE_FILE.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_edges() -> List[Dict[str, str]]:
    if not EDGE_FILE.exists():
        st.error(f"Missing edge file: {EDGE_FILE}. Run transcript_propagation_pipeline.py first.")
        st.stop()
    with EDGE_FILE.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["score"] = float(row["score"])
        row["temporal_distance"] = int(row["temporal_distance"])
    return rows


def quarter_to_index(quarter: str) -> Optional[int]:
    if "_" not in quarter or not quarter.startswith("Q"):
        return None
    q, y = quarter.split("_")
    try:
        return int(y) * 4 + int(q[1:])
    except Exception:
        return None


def index_to_quarter(idx: int) -> str:
    year, rem = divmod(idx - 1, 4)
    return f"Q{rem + 1}_{year}"


def build_positions(node_ids: List[str]) -> Dict[str, Tuple[float, float]]:
    quarter_groups = defaultdict(list)
    for node in node_ids:
        _, quarter = node.split("|")
        q_idx = quarter_to_index(quarter)
        if q_idx is None:
            continue
        quarter_groups[q_idx].append(node)

    positions = {}
    max_per_q = max((len(v) for v in quarter_groups.values()), default=1)
    for q_idx in sorted(quarter_groups):
        nodes = sorted(quarter_groups[q_idx])
        for i, node in enumerate(nodes):
            x = q_idx
            y = (i + 1) / (max_per_q + 1)
            positions[node] = (x, y)
    return positions


def build_figure(nodes: List[Dict[str, str]], edges: List[Dict[str, str]]) -> go.Figure:
    node_ids = [n["node_id"] for n in nodes]
    pos = build_positions(node_ids)

    fig = go.Figure()

    for edge in edges:
        src = edge["source_id"]
        dst = edge["target_id"]
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": max(edge["score"] * 4, 0.7), "color": "rgba(0, 191, 255, 0.45)"},
                hoverinfo="text",
                text=f"{src} ➜ {dst}<br>score={edge['score']:.3f}<br>signals={edge['shared_signals']}",
                showlegend=False,
            )
        )

    xs, ys, labels, text = [], [], [], []
    for node in nodes:
        node_id = node["node_id"]
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        xs.append(x)
        ys.append(y)
        labels.append(node["ticker"])
        text.append(
            f"{node_id}<br>domain={node['dominant_domain']}<br>signals={node['top_signals'][:140]}"
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker={"size": 10, "color": "#7CFC00", "line": {"width": 1, "color": "#FFFFFF"}},
            text=labels,
            textposition="top center",
            hoverinfo="text",
            hovertext=text,
            name="Transcripts",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#060916",
        paper_bgcolor="#060916",
        title="Dynamic Information Propagation Path",
        xaxis_title="Quarter Index",
        yaxis_title="Relative Node Position",
        font={"color": "#E8EEFF"},
        height=720,
    )
    return fig


def filter_data(
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, str]],
    tickers: List[str],
    domains: List[str],
    quarter_start: int,
    quarter_end: int,
    min_score: float,
):
    filtered_nodes = []
    allowed_ids = set()
    for n in nodes:
        q_idx = int(n["quarter_index"])
        if q_idx < quarter_start or q_idx > quarter_end:
            continue
        if tickers and n["ticker"] not in tickers:
            continue
        if domains and n["dominant_domain"] not in domains:
            continue
        filtered_nodes.append(n)
        allowed_ids.add(n["node_id"])

    filtered_edges = [
        e
        for e in edges
        if e["source_id"] in allowed_ids and e["target_id"] in allowed_ids and e["score"] >= min_score
    ]
    return filtered_nodes, filtered_edges


def main() -> None:
    st.set_page_config(page_title="Earnings Propagation Dashboard", layout="wide")
    st.markdown(
        """
        <style>
            .stApp { background: linear-gradient(160deg, #060916 0%, #0D132A 55%, #111C3D 100%); color: #D8E6FF; }
            .stMetric { background: rgba(7,18,45,0.45); border: 1px solid #1E90FF44; padding: 12px; border-radius: 10px; }
            .stSelectbox, .stMultiSelect, .stSlider { background: rgba(7,18,45,0.35); border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Earnings Call Information Propagation Dashboard")
    st.caption("Dark-tech dashboard for cross-company narrative diffusion and propagation path exploration.")

    nodes = load_nodes()
    edges = load_edges()

    tickers = sorted({n["ticker"] for n in nodes})
    domains = sorted({n["dominant_domain"] for n in nodes})
    quarter_indices = [int(n["quarter_index"]) for n in nodes if str(n.get("quarter_index", "")).isdigit()]
    q_min = min(quarter_indices)
    q_max = max(quarter_indices)

    with st.sidebar:
        st.header("Filters")
        selected_tickers = st.multiselect("Companies", tickers)
        selected_domains = st.multiselect("Domains", domains)
        q_range = st.slider("Quarter Range", min_value=q_min, max_value=q_max, value=(q_min, q_max))
        min_score = st.slider("Minimum Propagation Score", min_value=0.30, max_value=1.00, value=0.45, step=0.01)

    f_nodes, f_edges = filter_data(
        nodes,
        edges,
        selected_tickers,
        selected_domains,
        q_range[0],
        q_range[1],
        min_score,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Transcripts", len(f_nodes))
    c2.metric("Propagation Edges", len(f_edges))
    c3.metric("Avg Edge Score", f"{(sum(e['score'] for e in f_edges) / max(len(f_edges), 1)):.3f}")

    st.plotly_chart(build_figure(f_nodes, f_edges), use_container_width=True)

    st.subheader("Top Propagation Paths")
    ranked = sorted(f_edges, key=lambda x: x["score"], reverse=True)[:40]
    st.dataframe(ranked, use_container_width=True)

    st.subheader("Timeline Window")
    st.write(f"Showing {index_to_quarter(q_range[0])} to {index_to_quarter(q_range[1])}")


if __name__ == "__main__":
    main()
