#!/usr/bin/env python3
"""
Information flow network visualizer v2.

Why v2:
- The strict version only creates an edge when:
  source company has signal in 2025Q2
  AND extracted relationship target can be matched to another transcript company
  AND that target company has the same signal in 2025Q3.
  This is often too sparse.

This version creates richer demo networks with three edge types:
1. relationship_signal_flow:
   source company signal -> extracted relationship entity
   This shows where the information could flow, even when entity is generic
   such as "cloud customers", "suppliers", "OEM customers".

2. matched_temporal_flow:
   source company signal -> matched target company signal in next quarter.
   This is the stricter propagation evidence.

3. same_company_temporal_flow:
   same company source-quarter signal -> same company target-quarter signal.
   This shows persistence / continuation of signal inside the same firm.

Outputs:
  information_flow_network_v2.html
  information_flow_network_v2_static.png
  flow_edges_v2.csv
  flow_nodes_v2.csv
  flow_summary_v2.md

Run:
  cd E:/Projects/EearningAlz/RAG
  python visualize_information_flow_network_v2.py ^
    --input-dir rag_chroma_output/llm_csv_outputs_2025Q2_Q3 ^
    --out-dir rag_chroma_output/information_flow_network_demo_v2 ^
    --source-quarter 2025Q2 ^
    --target-quarter 2025Q3 ^
    --top-relationship-edges 200 ^
    --top-temporal-edges 200
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


STANDARD_SIGNALS = [
    "demand_outlook",
    "supply_outlook",
    "margin_outlook",
    "capex_outlook",
    "inventory_outlook",
    "pricing_outlook",
]

SIGNAL_MAP = {
    "demand_outlook": "demand_outlook",
    "supply_outlook": "supply_outlook",
    "margin_outlook": "margin_outlook",
    "capex_outlook": "capex_outlook",
    "inventory_outlook": "inventory_outlook",
    "pricing_outlook": "pricing_outlook",
    "supply_chain_outlook": "supply_outlook",
    "production_outlook": "supply_outlook",
    "manufacturing_outlook": "supply_outlook",
    "capacity_outlook": "supply_outlook",
    "loan_growth_outlook": "demand_outlook",
    "revenue_outlook": "demand_outlook",
    "sales_outlook": "demand_outlook",
    "credit_quality_outlook": "margin_outlook",
    "noninterest_income_outlook": "margin_outlook",
    "capital_generation_outlook": "capex_outlook",
}

LABEL_SCORE = {
    "positive": 1.0,
    "improving": 1.0,
    "increase": 1.0,
    "negative": -1.0,
    "worsening": -1.0,
    "decrease": -1.0,
    "mixed": 0.5,
    "neutral": 0.0,
    "stable": 0.0,
    "not_mentioned": np.nan,
    "": np.nan,
    "nan": np.nan,
}

ACTIVE_LABELS = {
    "positive",
    "improving",
    "increase",
    "negative",
    "worsening",
    "decrease",
    "mixed",
}


def parse_args():
    _SCRIPT_DIR = Path(__file__).resolve().parent   # .../test/
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-dir",
        default=str(_PROJECT_ROOT / "RAG" / "rag_chroma_output" / "llm_csv_outputs_2025Q2_Q3"),  # ← 修改
    )
    p.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "RAG" / "rag_chroma_output" / "information_flow_network_demo_v2"),  # ← 修改
    )
    p.add_argument("--source-quarter", default="2025Q2")
    p.add_argument("--target-quarter", default="2025Q3")

    p.add_argument("--top-relationship-edges", type=int, default=200)
    p.add_argument("--top-temporal-edges", type=int, default=200)
    p.add_argument("--top-nodes", type=int, default=180)

    p.add_argument(
        "--min-source-abs-score",
        type=float,
        default=0.5,
        help="Source signal must be at least this active.",
    )
    p.add_argument(
        "--include-neutral-temporal",
        action="store_true",
        help="Include stable/neutral target signals in temporal flows.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_shards(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern))
    frames = []
    print(f"Searching {pattern}: {len(files)} files")

    for f in files:
        if f.exists() and f.stat().st_size > 0:
            try:
                df = pd.read_csv(f)
                df["source_file"] = f.name
                frames.append(df)
                print(f"  loaded {f.name}: {len(df):,} rows")
            except Exception as e:
                print(f"  WARNING skipped {f}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop_duplicates()


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9&.\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    replacements = {
        " corporation": " corp",
        " incorporated": " inc",
        " company": " co",
        " limited": " ltd",
        " technologies": " tech",
        " technology": " tech",
        " international": " intl",
        " holdings": "",
        " holding": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"\bthe\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def short_name(name: str, ticker: str = "") -> str:
    name = "" if pd.isna(name) else str(name).strip()
    ticker = "" if pd.isna(ticker) else str(ticker).strip()

    if ticker and ticker.lower() not in {"nan", "none", ""}:
        return ticker[:18]

    if not name:
        return "UNKNOWN"

    n = re.sub(
        r"\b(inc|corp|corporation|company|co|ltd|plc|holdings|holding|group)\b\.?",
        "",
        name,
        flags=re.I,
    )
    n = re.sub(r"\s+", " ", n).strip()
    return n[:30] if n else name[:30]


def clean_outlook(df: pd.DataFrame) -> pd.DataFrame:
    required = ["doc_id", "ticker", "current_company", "quarter", "signal", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Outlook CSV missing columns: {missing}")

    out = df.copy()
    for c in required:
        out[c] = out[c].astype(str).str.strip()

    out["signal_raw"] = out["signal"]
    out["signal"] = out["signal"].str.lower().map(lambda x: SIGNAL_MAP.get(x, x))
    out = out[out["signal"].isin(STANDARD_SIGNALS)].copy()

    out["label"] = out["label"].str.lower().replace({"nan": "not_mentioned"})
    out["score"] = out["label"].map(LABEL_SCORE)

    out["company_norm"] = out["current_company"].map(norm_text)
    out["ticker_norm"] = out["ticker"].map(norm_text)
    out["company_node"] = np.where(
        out["ticker"].astype(str).str.strip().ne(""),
        "COMPANY::" + out["ticker"].astype(str).str.strip(),
        "COMPANY::" + out["company_norm"],
    )

    group_cols = [
        "company_node",
        "ticker",
        "current_company",
        "company_norm",
        "ticker_norm",
        "quarter",
        "signal",
    ]

    agg = {
        "score": "mean",
        "label": lambda x: ";".join(sorted(set(str(v) for v in x if str(v) != "nan"))),
    }

    if "evidence_chunk_ids" in out.columns:
        agg["evidence_chunk_ids"] = lambda x: "|".join(
            sorted(set(str(v) for v in x.dropna() if str(v).strip()))
        )

    if "notes" in out.columns:
        agg["notes"] = lambda x: " || ".join(str(v) for v in x.dropna() if str(v).strip())

    out = out.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out["is_active"] = out["score"].notna() & (out["score"].abs() > 0)
    return out


def clean_relationships(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    required = ["ticker", "current_company", "quarter", "relation_group", "entity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Relationships CSV missing columns: {missing}")

    rel = df.copy()
    for c in rel.columns:
        if rel[c].dtype == "object":
            rel[c] = rel[c].astype(str).str.strip()

    rel = rel[rel["entity"].fillna("").astype(str).str.strip().ne("")].copy()
    rel = rel[rel["relation_group"].str.lower().ne("none")].copy()

    rel["source_company_node"] = np.where(
        rel["ticker"].astype(str).str.strip().ne(""),
        "COMPANY::" + rel["ticker"].astype(str).str.strip(),
        "COMPANY::" + rel["current_company"].map(norm_text),
    )
    rel["source_company_norm"] = rel["current_company"].map(norm_text)
    rel["source_ticker_norm"] = rel["ticker"].map(norm_text)
    rel["target_entity_norm"] = rel["entity"].map(norm_text)
    rel["target_entity_node"] = "ENTITY::" + rel["target_entity_norm"]

    # Split schema-drift groups like "upstream|downstream|parent|subsidiary|related"
    rows = []
    for _, r in rel.iterrows():
        groups = [g.strip() for g in str(r["relation_group"]).split("|") if g.strip()]
        if not groups:
            groups = [str(r["relation_group"]).strip()]
        for g in groups:
            rr = r.copy()
            rr["relation_group_clean"] = g
            rows.append(rr)

    return pd.DataFrame(rows).drop_duplicates() if rows else rel


def build_company_lookup(outlook: pd.DataFrame):
    company_map = {}
    ticker_map = {}
    display_map = {}

    base = outlook[
        ["company_node", "ticker", "current_company", "company_norm", "ticker_norm"]
    ].drop_duplicates()

    for _, r in base.iterrows():
        node = str(r["company_node"])
        cname = str(r["company_norm"])
        ticker = str(r["ticker_norm"])

        if cname:
            company_map[cname] = node
        if ticker:
            ticker_map[ticker] = node

        display_map[node] = {
            "label": short_name(r["current_company"], r["ticker"]),
            "company": str(r["current_company"]),
            "ticker": str(r["ticker"]),
            "node_type": "company",
        }

    return company_map, ticker_map, display_map


def match_entity_to_company(entity_norm: str, company_map: dict, ticker_map: dict) -> str:
    if not entity_norm:
        return ""

    if entity_norm in ticker_map:
        return ticker_map[entity_norm]
    if entity_norm in company_map:
        return company_map[entity_norm]

    if len(entity_norm) >= 5:
        candidates = []
        for cname, node in company_map.items():
            if not cname or len(cname) < 5:
                continue
            if entity_norm in cname or cname in entity_norm:
                candidates.append((abs(len(cname) - len(entity_norm)), node))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

    return ""


def signal_direction(label: str, score: float) -> str:
    if pd.isna(score):
        return "not_mentioned"

    labels = set(str(label).split(";"))
    if labels & {"positive", "improving", "increase"}:
        return "positive"
    if labels & {"negative", "worsening", "decrease"}:
        return "negative"
    if labels & {"mixed"}:
        return "mixed"
    if labels & {"stable", "neutral"}:
        return "neutral"

    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def make_outlook_lookup(outlook: pd.DataFrame):
    lookup = {}
    for r in outlook.itertuples(index=False):
        lookup[(r.company_node, r.quarter, r.signal)] = r
    return lookup


def build_flow_edges(
    outlook: pd.DataFrame,
    relationships: pd.DataFrame,
    source_q: str,
    target_q: str,
    min_source_abs_score: float,
    include_neutral_temporal: bool,
    top_relationship_edges: int,
    top_temporal_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns node table and edge table.
    """

    company_map, ticker_map, display_map = build_company_lookup(outlook)
    outlook_lookup = make_outlook_lookup(outlook)

    node_info = dict(display_map)
    edge_rows = []

    src_outlook = outlook[
        (outlook["quarter"] == source_q)
        & (outlook["score"].notna())
        & (outlook["score"].abs() >= min_source_abs_score)
    ].copy()

    tgt_outlook = outlook[
        (outlook["quarter"] == target_q)
        & (outlook["score"].notna())
    ].copy()

    # ------------------------------------------------------------
    # A. Relationship signal flows:
    # source company signal in Q2 -> extracted relationship entity
    # ------------------------------------------------------------

    if not relationships.empty and not src_outlook.empty:
        rel = relationships.copy()

        # Prefer relationships extracted in the observed quarters, but keep structural rows.
        rel_q = rel[rel["quarter"].isin([source_q, target_q])].copy()
        if not rel_q.empty:
            rel = rel_q

        # Attach each active source signal to each relationship from the source firm.
        merged = rel.merge(
            src_outlook,
            left_on="source_company_node",
            right_on="company_node",
            how="inner",
            suffixes=("_rel", "_sig"),
        )

        for _, r in merged.iterrows():
            source = str(r["source_company_node"])
            entity_node = str(r["target_entity_node"])
            entity_label = short_name(str(r.get("entity", "")))

            if entity_node not in node_info:
                node_info[entity_node] = {
                    "label": entity_label,
                    "company": str(r.get("entity", "")),
                    "ticker": "",
                    "node_type": "entity",
                }

            weight = 1.0 + abs(float(r["score"]))

            edge_rows.append({
                "source": source,
                "target": entity_node,
                "edge_type": "relationship_signal_flow",
                "signal": str(r["signal"]),
                "source_quarter": source_q,
                "target_quarter": "",
                "source_label": str(r["label"]),
                "target_label": "",
                "source_score": float(r["score"]),
                "target_score": np.nan,
                "weight": weight,
                "relation_group": str(r.get("relation_group_clean", r.get("relation_group", ""))),
                "relationship_type": str(r.get("relationship_type", "")),
                "entity_type": str(r.get("entity_type", "")),
                "confidence": str(r.get("confidence", "")),
                "source_company": str(r.get("current_company_rel", r.get("current_company", ""))),
                "target_company": str(r.get("entity", "")),
                "matched_target_company": "",
            })

            matched_node = match_entity_to_company(str(r["target_entity_norm"]), company_map, ticker_map)

            # If the relationship entity can be matched to a company node,
            # add stricter temporal edge if target has same signal in Q3.
            if matched_node:
                trow = outlook_lookup.get((matched_node, target_q, str(r["signal"])))
                if trow is not None and not pd.isna(trow.score):
                    target_active = abs(float(trow.score)) > 0
                    if include_neutral_temporal or target_active:
                        same_direction = np.sign(float(r["score"])) == np.sign(float(trow.score))
                        temporal_weight = abs(float(r["score"])) * (0.5 + abs(float(trow.score)))
                        if same_direction:
                            temporal_weight += 0.5

                        edge_rows.append({
                            "source": source,
                            "target": matched_node,
                            "edge_type": "matched_temporal_flow",
                            "signal": str(r["signal"]),
                            "source_quarter": source_q,
                            "target_quarter": target_q,
                            "source_label": str(r["label"]),
                            "target_label": str(trow.label),
                            "source_score": float(r["score"]),
                            "target_score": float(trow.score),
                            "weight": temporal_weight,
                            "relation_group": str(r.get("relation_group_clean", r.get("relation_group", ""))),
                            "relationship_type": str(r.get("relationship_type", "")),
                            "entity_type": str(r.get("entity_type", "")),
                            "confidence": str(r.get("confidence", "")),
                            "source_company": str(r.get("current_company_rel", r.get("current_company", ""))),
                            "target_company": str(trow.current_company),
                            "matched_target_company": str(trow.current_company),
                        })

    # ------------------------------------------------------------
    # B. Same-company temporal flow:
    # company Q2 signal -> same company Q3 signal
    # ------------------------------------------------------------

    src_keys = set((r.company_node, r.signal) for r in src_outlook.itertuples(index=False))

    for company_node, signal in src_keys:
        srow = outlook_lookup.get((company_node, source_q, signal))
        trow = outlook_lookup.get((company_node, target_q, signal))

        if srow is None or trow is None:
            continue
        if pd.isna(trow.score):
            continue

        target_active = abs(float(trow.score)) > 0
        if not include_neutral_temporal and not target_active:
            continue

        temporal_node = f"{company_node}::{target_q}"
        source_node = f"{company_node}::{source_q}"

        # Create quarter-specific company nodes for clearer left-right flow
        base = node_info.get(company_node, {})
        node_info[source_node] = {
            "label": f"{base.get('label', company_node.replace('COMPANY::',''))}\n{source_q}",
            "company": base.get("company", ""),
            "ticker": base.get("ticker", ""),
            "node_type": "company_quarter_source",
        }
        node_info[temporal_node] = {
            "label": f"{base.get('label', company_node.replace('COMPANY::',''))}\n{target_q}",
            "company": base.get("company", ""),
            "ticker": base.get("ticker", ""),
            "node_type": "company_quarter_target",
        }

        same_direction = np.sign(float(srow.score)) == np.sign(float(trow.score))
        weight = abs(float(srow.score)) * (0.5 + abs(float(trow.score)))
        if same_direction:
            weight += 0.5

        edge_rows.append({
            "source": source_node,
            "target": temporal_node,
            "edge_type": "same_company_temporal_flow",
            "signal": signal,
            "source_quarter": source_q,
            "target_quarter": target_q,
            "source_label": str(srow.label),
            "target_label": str(trow.label),
            "source_score": float(srow.score),
            "target_score": float(trow.score),
            "weight": weight,
            "relation_group": "same_company",
            "relationship_type": "temporal_continuation",
            "entity_type": "company",
            "confidence": "observed",
            "source_company": str(srow.current_company),
            "target_company": str(trow.current_company),
            "matched_target_company": str(trow.current_company),
        })

    edges = pd.DataFrame(edge_rows)

    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Cap each edge family so the figure is not unreadable.
    capped = []
    for edge_type, cap in [
        ("matched_temporal_flow", top_temporal_edges),
        ("same_company_temporal_flow", top_temporal_edges),
        ("relationship_signal_flow", top_relationship_edges),
    ]:
        part = edges[edges["edge_type"] == edge_type].copy()
        if not part.empty:
            capped.append(part.sort_values("weight", ascending=False).head(cap))

    edges = pd.concat(capped, ignore_index=True).drop_duplicates() if capped else edges

    used_nodes = set(edges["source"]).union(set(edges["target"]))

    nodes = []
    for node in used_nodes:
        info = node_info.get(node, {})
        nodes.append({
            "node": node,
            "label": info.get("label", node.replace("COMPANY::", "").replace("ENTITY::", "")[:30]),
            "company": info.get("company", ""),
            "ticker": info.get("ticker", ""),
            "node_type": info.get("node_type", "unknown"),
        })

    nodes = pd.DataFrame(nodes)

    return nodes, edges


def make_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, r in nodes.iterrows():
        G.add_node(
            r["node"],
            label=r["label"],
            company=r.get("company", ""),
            ticker=r.get("ticker", ""),
            node_type=r.get("node_type", ""),
        )

    for _, r in edges.iterrows():
        u = r["source"]
        v = r["target"]

        if G.has_edge(u, v):
            G[u][v]["weight"] += float(r["weight"])
            G[u][v]["signals"] += "," + str(r["signal"])
            G[u][v]["edge_types"] += "," + str(r["edge_type"])
        else:
            G.add_edge(
                u,
                v,
                weight=float(r["weight"]),
                signal=str(r["signal"]),
                signals=str(r["signal"]),
                edge_type=str(r["edge_type"]),
                edge_types=str(r["edge_type"]),
                relation_group=str(r.get("relation_group", "")),
                relationship_type=str(r.get("relationship_type", "")),
                source_label=str(r.get("source_label", "")),
                target_label=str(r.get("target_label", "")),
            )

    return G


def layered_layout(G: nx.DiGraph):
    """
    Put source-quarter nodes on the left, generic entities in the middle,
    target-quarter nodes on the right.
    """

    source_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company_quarter_source"]
    target_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company_quarter_target"]
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
    company_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "company" and n not in source_nodes and n not in target_nodes
    ]
    other_nodes = [
        n for n in G.nodes()
        if n not in set(source_nodes + target_nodes + entity_nodes + company_nodes)
    ]

    pos = {}

    def place(nodes, x, y_min=-1.0, y_max=1.0):
        nodes = list(nodes)
        if not nodes:
            return
        if len(nodes) == 1:
            ys = [0.0]
        else:
            ys = np.linspace(y_max, y_min, len(nodes))
        for node, y in zip(nodes, ys):
            pos[node] = (x, float(y))

    # Sort by degree for readability
    source_nodes = sorted(source_nodes, key=lambda n: G.degree(n), reverse=True)
    entity_nodes = sorted(entity_nodes, key=lambda n: G.degree(n), reverse=True)
    company_nodes = sorted(company_nodes, key=lambda n: G.degree(n), reverse=True)
    target_nodes = sorted(target_nodes, key=lambda n: G.degree(n), reverse=True)

    place(source_nodes, 0.0)
    place(company_nodes, 0.35)
    place(entity_nodes, 0.55)
    place(target_nodes, 1.0)
    place(other_nodes, 0.75)

    # Fallback for any missing nodes
    missing = [n for n in G.nodes() if n not in pos]
    if missing:
        spring = nx.spring_layout(G.subgraph(missing), seed=42)
        for n, p in spring.items():
            pos[n] = (0.5 + 0.2 * float(p[0]), float(p[1]))

    return pos


def save_static(G: nx.DiGraph, out_png: Path):
    if G.number_of_nodes() == 0:
        return

    pos = layered_layout(G)

    plt.figure(figsize=(18, 12))

    sizes = [250 + 120 * math.sqrt(max(1, G.degree(n))) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.85)

    widths = [0.6 + min(4.0, float(d.get("weight", 1))) for _, _, d in G.edges(data=True)]

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=widths,
        alpha=0.35,
        connectionstyle="arc3,rad=0.06",
    )

    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title("Information Flow Network Demo: Relationship Signals and Temporal Continuation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def save_html(G: nx.DiGraph, out_html: Path, source_q: str, target_q: str):
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skip HTML.")
        return

    if G.number_of_nodes() == 0:
        return

    pos = layered_layout(G)

    edge_x = []
    edge_y = []

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        name="information-flow edges",
    )

    annotations = []
    for idx, (u, v, d) in enumerate(G.edges(data=True)):
        if idx >= 250:
            break
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        annotations.append(
            dict(
                ax=x0,
                ay=y0,
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1,
                opacity=0.35,
            )
        )

    node_x = []
    node_y = []
    node_text = []
    hover_text = []
    node_size = []

    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(d.get("label", n))
        node_size.append(12 + 5 * math.sqrt(max(1, G.degree(n))))

        hover_text.append(
            f"<b>{d.get('label', n)}</b><br>"
            f"node: {n}<br>"
            f"type: {d.get('node_type', '')}<br>"
            f"company/entity: {d.get('company', '')}<br>"
            f"in-degree: {G.in_degree(n)}<br>"
            f"out-degree: {G.out_degree(n)}<br>"
            f"degree: {G.degree(n)}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(size=node_size, line=dict(width=1)),
        name="nodes",
    )

    subtitle = (
        "Left: source-quarter company signals. "
        "Middle: extracted relationship entities / matched companies. "
        "Right: target-quarter same-company signal continuation. "
        "This is a descriptive candidate information-flow map, not causal proof."
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Information Flow Network Demo: {source_q} → {target_q}<br><sup>{subtitle}</sup>",
                x=0.02,
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=100),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"SAVED {out_html}")


def save_bar_charts(edges: pd.DataFrame, out_dir: Path):
    if edges.empty:
        return

    # Edge type counts
    edge_type_counts = edges["edge_type"].value_counts().reset_index()
    edge_type_counts.columns = ["edge_type", "count"]
    edge_type_counts.to_csv(out_dir / "edge_type_counts.csv", index=False)

    ax = edge_type_counts.plot(kind="bar", x="edge_type", y="count", legend=False, figsize=(10, 5))
    ax.set_title("Information-flow edge types")
    ax.set_xlabel("Edge type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "edge_type_counts.png", dpi=220)
    plt.close()

    # Signal counts
    signal_counts = edges["signal"].value_counts().reset_index()
    signal_counts.columns = ["signal", "count"]
    signal_counts.to_csv(out_dir / "signal_flow_counts.csv", index=False)

    ax = signal_counts.plot(kind="bar", x="signal", y="count", legend=False, figsize=(10, 5))
    ax.set_title("Information-flow edges by signal")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "signal_flow_counts.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    print("=" * 90)
    print("Information Flow Network Visualization V2")
    print("input_dir:", input_dir)
    print("out_dir:", out_dir)
    print("source_quarter:", args.source_quarter)
    print("target_quarter:", args.target_quarter)
    print("=" * 90)

    outlook_raw = read_shards(input_dir, "outlook_q2q3_shard*_of006.csv")
    relationships_raw = read_shards(input_dir, "relationships_q2q3_shard*_of006.csv")
    concepts_raw = read_shards(input_dir, "concepts_q2q3_shard*_of006.csv")

    if outlook_raw.empty:
        raise FileNotFoundError("No outlook CSV found.")

    outlook = clean_outlook(outlook_raw)
    relationships = clean_relationships(relationships_raw) if not relationships_raw.empty else pd.DataFrame()

    print("\nLoaded rows:")
    print("outlook_raw:", len(outlook_raw))
    print("relationships_raw:", len(relationships_raw))
    print("concepts_raw:", len(concepts_raw))

    print("\nCleaned rows:")
    print("outlook:", len(outlook))
    print("relationships:", len(relationships))

    print("\nQuarter distribution:")
    print(outlook["quarter"].value_counts(dropna=False).sort_index())

    print("\nSignal distribution:")
    print(outlook["signal"].value_counts(dropna=False))

    if not relationships.empty:
        print("\nRelationship group distribution:")
        print(relationships["relation_group_clean"].value_counts(dropna=False).head(30))

    nodes, edges = build_flow_edges(
        outlook=outlook,
        relationships=relationships,
        source_q=args.source_quarter,
        target_q=args.target_quarter,
        min_source_abs_score=args.min_source_abs_score,
        include_neutral_temporal=args.include_neutral_temporal,
        top_relationship_edges=args.top_relationship_edges,
        top_temporal_edges=args.top_temporal_edges,
    )

    if edges.empty:
        raise RuntimeError("No flow edges created. Try lowering --min-source-abs-score or check labels.")

    # Cap most connected graph nodes if needed
    G = make_graph(nodes, edges)

    if G.number_of_nodes() > args.top_nodes:
        # Keep top degree nodes and related edges
        top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[: args.top_nodes]
        keep_nodes = {n for n, _ in top}
        edges = edges[edges["source"].isin(keep_nodes) & edges["target"].isin(keep_nodes)].copy()
        nodes = nodes[nodes["node"].isin(keep_nodes)].copy()
        G = make_graph(nodes, edges)

    nodes_out = nodes.copy()
    if not nodes_out.empty:
        nodes_out["in_degree"] = nodes_out["node"].map(dict(G.in_degree()))
        nodes_out["out_degree"] = nodes_out["node"].map(dict(G.out_degree()))
        nodes_out["degree"] = nodes_out["node"].map(dict(G.degree()))

    nodes_out.to_csv(out_dir / "flow_nodes_v2.csv", index=False)
    edges.to_csv(out_dir / "flow_edges_v2.csv", index=False)

    print(f"\nSAVED {out_dir / 'flow_nodes_v2.csv'} rows={len(nodes_out):,}")
    print(f"SAVED {out_dir / 'flow_edges_v2.csv'} rows={len(edges):,}")
    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())

    print("\nEdge type distribution:")
    print(edges["edge_type"].value_counts(dropna=False))

    print("\nSignal distribution in edges:")
    print(edges["signal"].value_counts(dropna=False))

    save_static(G, out_dir / "information_flow_network_v2_static.png")
    save_html(G, out_dir / "information_flow_network_v2.html", args.source_quarter, args.target_quarter)
    save_bar_charts(edges, out_dir)

    summary = [
        "# Information Flow Network Demo V2",
        "",
        f"- Source quarter: `{args.source_quarter}`",
        f"- Target quarter: `{args.target_quarter}`",
        f"- Raw outlook rows: {len(outlook_raw):,}",
        f"- Raw relationship rows: {len(relationships_raw):,}",
        f"- Cleaned outlook rows: {len(outlook):,}",
        f"- Cleaned relationship rows: {len(relationships):,}",
        f"- Flow nodes: {G.number_of_nodes():,}",
        f"- Flow edges: {G.number_of_edges():,}",
        "",
        "## Edge types",
        "",
        "- `relationship_signal_flow`: active source-company signal points to extracted relationship entity.",
        "- `matched_temporal_flow`: relationship entity was matched to another observed company, and that company had the same signal in the target quarter.",
        "- `same_company_temporal_flow`: same company had the signal in both source and target quarter.",
        "",
        "## Important note",
        "",
        "This is a descriptive visualization of candidate information flow. "
        "It should be interpreted as a map of possible diffusion channels and signal persistence, not as causal proof.",
    ]
    (out_dir / "README_information_flow_v2.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"SAVED {out_dir / 'README_information_flow_v2.md'}")
    print("\nDONE. Open:")
    print(out_dir / "information_flow_network_v2.html")


if __name__ == "__main__":
    main()
