#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master analysis for existing LLM extracted earnings-call results.

It recursively reads extracted CSVs under rag_chroma_output:
  concepts_*.csv
  relationships_*.csv
  outlook_*.csv

Then generates:
  cleaned tables
  company-quarter signal matrix
  relationship network nodes/edges
  contagion events for source_quarter -> target_quarter
  transmission summary
  falsification summary for focus signal/label
  static PNG figures
  optional interactive Plotly HTML network

No LLM call. No GPU required.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

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
    "demand_outlook", "supply_outlook", "margin_outlook",
    "capex_outlook", "inventory_outlook", "pricing_outlook",
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
    "positive": 1.0, "improving": 1.0, "increase": 1.0,
    "negative": -1.0, "worsening": -1.0, "decrease": -1.0,
    "mixed": 0.5,
    "neutral": 0.0, "stable": 0.0,
    "not_mentioned": np.nan, "": np.nan, "nan": np.nan,
}

POSITIVE_LABELS = {"positive", "improving", "increase"}
NEGATIVE_LABELS = {"negative", "worsening", "decrease"}
NEUTRAL_LABELS = {"neutral", "stable"}
MIXED_LABELS = {"mixed"}

CONCEPT_COLUMNS = [
    "chip_supply", "semiconductor_supply", "raw_material_supply", "oil_energy_supply",
    "manufacturing_capacity", "production_capacity", "inventory_pressure",
    "logistics_shipping", "supplier_constraint", "customer_demand", "pricing_pressure",
    "capex_expansion", "data_center_capacity", "cloud_infrastructure",
    "labor_constraint", "geopolitical_risk",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag-output-dir", default="rag_chroma_output")
    p.add_argument("--out-dir", default="rag_chroma_output/network_contagion_master_analysis")
    p.add_argument("--start-quarter", default="")
    p.add_argument("--end-quarter", default="")
    p.add_argument("--source-quarter", default="2025Q2")
    p.add_argument("--target-quarter", default="2025Q3")
    p.add_argument("--focus-signal", default="margin_outlook")
    p.add_argument("--focus-label", default="improving")
    p.add_argument("--top-network-nodes", type=int, default=120)
    p.add_argument("--include-self-edges", action="store_true")
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def quarter_to_index(q: str) -> float:
    q = str(q).strip()
    m = re.match(r"^(\d{4})Q([1-4])$", q)
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2))


def in_quarter_range(df: pd.DataFrame, q_col: str, start_q: str, end_q: str) -> pd.DataFrame:
    if df.empty or q_col not in df.columns:
        return df
    if not start_q and not end_q:
        return df
    out = df.copy()
    out["_qidx"] = out[q_col].map(quarter_to_index)
    if start_q:
        out = out[out["_qidx"] >= quarter_to_index(start_q)].copy()
    if end_q:
        out = out[out["_qidx"] <= quarter_to_index(end_q)].copy()
    return out.drop(columns=["_qidx"])


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9&.\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    repl = {
        " corporation": " corp", " incorporated": " inc", " company": " co",
        " limited": " ltd", " technologies": " tech", " technology": " tech",
        " international": " intl", " holdings": "", " holding": "", " group": "",
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    s = re.sub(r"\bthe\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def short_name(name: str, ticker: str = "") -> str:
    name = "" if pd.isna(name) else str(name).strip()
    ticker = "" if pd.isna(ticker) else str(ticker).strip()
    if ticker and ticker.lower() not in {"nan", "none", ""}:
        return ticker[:18]
    if not name:
        return "UNKNOWN"
    n = re.sub(r"\b(inc|corp|corporation|company|co|ltd|plc|holdings|holding|group)\b\.?,?", "", name, flags=re.I)
    n = re.sub(r"\s+", " ", n).strip()
    return n[:30] if n else name[:30]


def label_set(x) -> set[str]:
    if pd.isna(x):
        return set()
    return {v.strip().lower() for v in str(x).split(";") if v.strip()}


def label_direction(label: str, score=None) -> str:
    labels = label_set(label)
    if labels & POSITIVE_LABELS:
        return "positive"
    if labels & NEGATIVE_LABELS:
        return "negative"
    if labels & MIXED_LABELS:
        return "mixed"
    if labels & NEUTRAL_LABELS:
        return "neutral"
    if score is not None and not pd.isna(score):
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"
    return "not_mentioned"


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"SAVED {path} rows={len(df):,}")


def discover_csvs(rag_dir: Path):
    all_csvs = sorted(rag_dir.rglob("*.csv"))
    concepts, relationships, outlook, failed = [], [], [], []
    for f in all_csvs:
        name = f.name.lower()
        # only extraction outputs, not analysis outputs
        if name.startswith("concepts_"):
            concepts.append(f)
        elif name.startswith("relationships_"):
            relationships.append(f)
        elif name.startswith("outlook_"):
            outlook.append(f)
        elif name.startswith("failed_"):
            failed.append(f)
    return concepts, relationships, outlook, failed


def read_many_csv(files: list[Path], kind: str) -> pd.DataFrame:
    frames = []
    print(f"\nLoading {kind}: {len(files)} files")
    for f in files:
        try:
            if f.stat().st_size == 0:
                print("  skip empty:", f)
                continue
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            frames.append(df)
            print(f"  loaded {f} rows={len(df):,}")
        except Exception as e:
            print(f"  WARNING failed reading {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def clean_outlook(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    if df.empty:
        return df
    required = ["doc_id", "ticker", "current_company", "quarter", "signal", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Outlook missing columns: {missing}")
    out = df.copy()
    for c in required:
        out[c] = out[c].astype(str).str.strip()
    out["signal_raw"] = out["signal"]
    out["signal"] = out["signal"].str.lower().map(lambda x: SIGNAL_MAP.get(x, x))
    out = out[out["signal"].isin(STANDARD_SIGNALS)].copy()
    out["label"] = out["label"].astype(str).str.strip().str.lower().replace({"nan": "not_mentioned"})
    out["score"] = out["label"].map(LABEL_SCORE)
    out["company_norm"] = out["current_company"].map(norm_text)
    out["ticker_norm"] = out["ticker"].map(norm_text)
    out["company_node"] = np.where(out["ticker"].str.strip().ne(""), "COMPANY::" + out["ticker"], "COMPANY::" + out["company_norm"])
    out["quarter_index"] = out["quarter"].map(quarter_to_index)
    out = in_quarter_range(out, "quarter", start_q, end_q)
    group_cols = ["company_node", "ticker", "current_company", "company_norm", "ticker_norm", "quarter", "quarter_index", "signal"]
    agg = {
        "score": "mean",
        "label": lambda x: ";".join(sorted(set(str(v) for v in x if str(v) and str(v) != "nan"))),
        "source_file": lambda x: "|".join(sorted(set(str(v) for v in x.dropna()))),
    }
    if "evidence_chunk_ids" in out.columns:
        agg["evidence_chunk_ids"] = lambda x: "|".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))
    if "notes" in out.columns:
        agg["notes"] = lambda x: " || ".join(str(v) for v in x.dropna() if str(v).strip())
    out = out.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out["direction"] = out.apply(lambda r: label_direction(r["label"], r["score"]), axis=1)
    out["is_active"] = out["score"].notna() & (out["score"].abs() > 0)
    return out


def clean_relationships(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    if df.empty:
        return df
    required = ["ticker", "current_company", "quarter", "relation_group", "entity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Relationships missing columns: {missing}")
    rel = df.copy()
    for c in rel.columns:
        if rel[c].dtype == "object":
            rel[c] = rel[c].astype(str).str.strip()
    rel = rel[rel["entity"].fillna("").astype(str).str.strip().ne("")].copy()
    rel = rel[rel["relation_group"].fillna("").astype(str).str.lower().ne("none")].copy()
    rel["source_company_node"] = np.where(rel["ticker"].astype(str).str.strip().ne(""), "COMPANY::" + rel["ticker"].astype(str).str.strip(), "COMPANY::" + rel["current_company"].map(norm_text))
    rel["source_company_norm"] = rel["current_company"].map(norm_text)
    rel["source_ticker_norm"] = rel["ticker"].map(norm_text)
    rel["target_entity_norm"] = rel["entity"].map(norm_text)
    rel["target_entity_node"] = "ENTITY::" + rel["target_entity_norm"]
    rel["quarter_index"] = rel["quarter"].map(quarter_to_index)
    rel = in_quarter_range(rel, "quarter", start_q, end_q)
    rows = []
    for _, r in rel.iterrows():
        groups = [g.strip() for g in str(r["relation_group"]).split("|") if g.strip()]
        if not groups:
            groups = [str(r["relation_group"]).strip()]
        for g in groups:
            rr = r.copy()
            rr["relation_group_clean"] = g
            rows.append(rr)
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def clean_concepts(df: pd.DataFrame, start_q: str, end_q: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in ["doc_id", "ticker", "current_company", "quarter"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    if "current_company" in out.columns:
        out["company_norm"] = out["current_company"].map(norm_text)
    else:
        out["company_norm"] = ""
    if "ticker" in out.columns:
        out["ticker_norm"] = out["ticker"].map(norm_text)
        out["company_node"] = np.where(out["ticker"].astype(str).str.strip().ne(""), "COMPANY::" + out["ticker"].astype(str).str.strip(), "COMPANY::" + out["company_norm"])
    else:
        out["ticker_norm"] = ""
        out["company_node"] = "COMPANY::" + out["company_norm"]
    if "quarter" in out.columns:
        out["quarter_index"] = out["quarter"].map(quarter_to_index)
        out = in_quarter_range(out, "quarter", start_q, end_q)
    for c in CONCEPT_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out.drop_duplicates()


def build_company_lookup(outlook: pd.DataFrame):
    company_map, ticker_map, meta = {}, {}, {}
    if outlook.empty:
        return company_map, ticker_map, meta
    base = outlook[["company_node", "ticker", "current_company", "company_norm", "ticker_norm"]].drop_duplicates()
    for _, r in base.iterrows():
        node = str(r["company_node"])
        cname = str(r["company_norm"])
        ticker = str(r["ticker_norm"])
        if cname:
            company_map[cname] = node
        if ticker:
            ticker_map[ticker] = node
        meta[node] = {"ticker": str(r["ticker"]), "company": str(r["current_company"]), "label": short_name(r["current_company"], r["ticker"])}
    return company_map, ticker_map, meta


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


def build_relationship_network(relationships: pd.DataFrame, outlook: pd.DataFrame):
    company_map, ticker_map, meta = build_company_lookup(outlook)
    rel = relationships.copy()
    if rel.empty:
        return nx.DiGraph(), pd.DataFrame(), pd.DataFrame()
    rel["matched_target_company_node"] = rel["target_entity_norm"].map(lambda x: match_entity_to_company(x, company_map, ticker_map))
    G = nx.DiGraph()
    edge_rows = []
    for _, r in rel.iterrows():
        s = str(r["source_company_node"])
        t_match = str(r.get("matched_target_company_node", "")).strip()
        t = t_match if t_match else str(r["target_entity_node"])
        if not s or not t:
            continue
        smeta = meta.get(s, {})
        G.add_node(s, node_type="company", label=smeta.get("label", short_name(r.get("current_company", ""), r.get("ticker", ""))), company=smeta.get("company", str(r.get("current_company", ""))), ticker=smeta.get("ticker", str(r.get("ticker", ""))))
        if t_match:
            tmeta = meta.get(t_match, {})
            G.add_node(t, node_type="matched_company", label=tmeta.get("label", t.replace("COMPANY::", "")), company=tmeta.get("company", ""), ticker=tmeta.get("ticker", ""))
            matched = True
        else:
            G.add_node(t, node_type="entity", label=short_name(str(r.get("entity", ""))), company=str(r.get("entity", "")), ticker="")
            matched = False
        relation = str(r.get("relation_group_clean", r.get("relation_group", "")))
        if G.has_edge(s, t):
            G[s][t]["weight"] += 1
            G[s][t]["relations"] += "," + relation
        else:
            G.add_edge(s, t, weight=1, relation_group=relation, relations=relation, relationship_type=str(r.get("relationship_type", "")), matched_target_company=matched)
        edge_rows.append({
            "source_node": s, "target_node": t,
            "source_company": str(r.get("current_company", "")), "source_ticker": str(r.get("ticker", "")),
            "target_entity": str(r.get("entity", "")), "matched_target_company_node": t_match,
            "relation_group": relation, "relationship_type": str(r.get("relationship_type", "")),
            "entity_type": str(r.get("entity_type", "")), "confidence": str(r.get("confidence", "")),
            "quarter": str(r.get("quarter", "")), "matched_target_company": matched,
        })
    node_rows = []
    for n, d in G.nodes(data=True):
        node_rows.append({"node": n, "label": d.get("label", n), "node_type": d.get("node_type", ""), "company": d.get("company", ""), "ticker": d.get("ticker", ""), "in_degree": G.in_degree(n), "out_degree": G.out_degree(n), "degree": G.degree(n)})
    return G, pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def make_outlook_lookup(outlook: pd.DataFrame):
    return {(r.company_node, r.quarter, r.signal): r for r in outlook.itertuples(index=False)}


def build_contagion_events(outlook: pd.DataFrame, relationships: pd.DataFrame, source_q: str, target_q: str, include_self_edges: bool):
    company_map, ticker_map, meta = build_company_lookup(outlook)
    lookup = make_outlook_lookup(outlook)
    rel = relationships.copy()
    if rel.empty:
        return pd.DataFrame(), pd.DataFrame()
    rel_q = rel[rel["quarter"].isin([source_q, target_q])].copy()
    if not rel_q.empty:
        rel = rel_q
    rel["target_company_node"] = rel["target_entity_norm"].map(lambda x: match_entity_to_company(x, company_map, ticker_map))
    unmatched = rel[rel["target_company_node"].fillna("").eq("")].copy()
    matched = rel[rel["target_company_node"].fillna("").ne("")].copy()
    if not include_self_edges:
        matched = matched[matched["source_company_node"] != matched["target_company_node"]].copy()
    events = []
    for _, edge in matched.iterrows():
        source_node = str(edge["source_company_node"])
        target_node = str(edge["target_company_node"])
        smeta = meta.get(source_node, {})
        tmeta = meta.get(target_node, {})
        for signal in STANDARD_SIGNALS:
            srow = lookup.get((source_node, source_q, signal))
            trow = lookup.get((target_node, target_q, signal))
            if srow is None or trow is None:
                continue
            source_label = str(srow.label)
            target_label = str(trow.label)
            source_score = float(srow.score) if not pd.isna(srow.score) else np.nan
            target_score = float(trow.score) if not pd.isna(trow.score) else np.nan
            source_direction = label_direction(source_label, source_score)
            target_direction = label_direction(target_label, target_score)
            source_active = not pd.isna(source_score) and abs(source_score) > 0
            target_active = not pd.isna(target_score) and abs(target_score) > 0
            exact_adoption = source_active and target_active and source_label == target_label
            direction_adoption = source_active and target_active and source_direction == target_direction
            events.append({
                "source_node": source_node, "source_ticker": smeta.get("ticker", ""), "source_company": smeta.get("company", ""),
                "target_node": target_node, "target_ticker": tmeta.get("ticker", ""), "target_company": tmeta.get("company", ""),
                "source_quarter": source_q, "target_quarter": target_q, "signal": signal,
                "source_label": source_label, "target_label": target_label,
                "source_score": source_score, "target_score": target_score,
                "source_direction": source_direction, "target_direction": target_direction,
                "source_active": source_active, "target_active": target_active,
                "exact_adoption": exact_adoption, "direction_adoption": direction_adoption,
                "relation_group": str(edge.get("relation_group_clean", edge.get("relation_group", ""))),
                "relationship_type": str(edge.get("relationship_type", "")), "entity_type": str(edge.get("entity_type", "")),
                "confidence": str(edge.get("confidence", "")), "extracted_entity": str(edge.get("entity", "")),
            })
    return pd.DataFrame(events), unmatched


def summarize_contagion(events: pd.DataFrame):
    if events.empty:
        return pd.DataFrame()
    exp = events[events["source_active"]].copy()
    if exp.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in exp.groupby(["signal", "source_label", "source_direction", "relation_group"], dropna=False):
        signal, source_label, source_direction, relation_group = keys
        exposed = len(g)
        target_active = int(g["target_active"].sum())
        exact = int(g["exact_adoption"].sum())
        direction = int(g["direction_adoption"].sum())
        rows.append({
            "signal": signal, "source_label": source_label, "source_direction": source_direction, "relation_group": relation_group,
            "exposed_edges": exposed, "target_active_edges": target_active, "exact_adopted_edges": exact, "direction_adopted_edges": direction,
            "exact_transmission_rate": exact / exposed if exposed else np.nan,
            "direction_transmission_rate": direction / exposed if exposed else np.nan,
            "target_active_rate": target_active / exposed if exposed else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["direction_transmission_rate", "exact_transmission_rate", "exposed_edges"], ascending=[False, False, False])


def focus_falsification(events: pd.DataFrame, focus_signal: str, focus_label: str):
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    focus_label = focus_label.strip().lower()
    f = events[(events["signal"] == focus_signal) & (events["source_active"]) & (events["source_label"].astype(str).str.lower().str.contains(rf"\b{re.escape(focus_label)}\b", regex=True, na=False))].copy()
    if f.empty:
        return pd.DataFrame(), pd.DataFrame()
    source_dir = label_direction(focus_label)
    f["target_direction_calc"] = f["target_label"].map(label_direction)
    f["exact_transmitted"] = f["target_label"].astype(str).str.lower().str.contains(rf"\b{re.escape(focus_label)}\b", regex=True, na=False)
    f["direction_transmitted"] = f["target_direction_calc"].eq(source_dir)
    f["falsified_exact"] = ~f["exact_transmitted"]
    f["falsified_direction"] = ~f["direction_transmitted"]
    def reason(row):
        if row["exact_transmitted"]:
            return "transmitted_exact"
        if row["direction_transmitted"]:
            return "not_exact_but_same_direction"
        if str(row.get("target_label", "")).lower() in {"not_mentioned", "nan", ""}:
            return "target_not_mentioned"
        if row["target_direction_calc"] == "neutral":
            return "target_neutral_or_stable"
        if row["target_direction_calc"] == "negative":
            return "opposite_negative"
        if row["target_direction_calc"] == "positive":
            return "opposite_positive"
        if row["target_direction_calc"] == "mixed":
            return "target_mixed"
        return "not_transmitted"
    f["falsification_reason"] = f.apply(reason, axis=1)
    rows = []
    for relation, g in f.groupby("relation_group", dropna=False):
        exposed = len(g)
        exact = int(g["exact_transmitted"].sum())
        direction = int(g["direction_transmitted"].sum())
        rows.append({
            "focus_signal": focus_signal, "focus_label": focus_label, "relation_group": relation,
            "exposed_edges": exposed, "exact_transmitted_edges": exact, "direction_transmitted_edges": direction,
            "falsified_exact_edges": exposed - exact, "falsified_direction_edges": exposed - direction,
            "exact_transmission_rate": exact / exposed if exposed else np.nan,
            "direction_transmission_rate": direction / exposed if exposed else np.nan,
            "exact_falsification_rate": (exposed - exact) / exposed if exposed else np.nan,
            "direction_falsification_rate": (exposed - direction) / exposed if exposed else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values(["direction_falsification_rate", "exact_falsification_rate", "exposed_edges"], ascending=[False, False, False])
    return f, summary


def plot_outlook_distribution(outlook: pd.DataFrame, out_png: Path):
    if outlook.empty:
        return
    dist = outlook.groupby(["signal", "label"]).size().reset_index(name="count")
    pivot = dist.pivot(index="signal", columns="label", values="count").fillna(0)
    pivot = pivot.reindex([s for s in STANDARD_SIGNALS if s in pivot.index])
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title("Outlook label distribution by signal")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def plot_concept_distribution(concepts: pd.DataFrame, out_png: Path):
    if concepts.empty:
        return
    rows = []
    for c in [c for c in CONCEPT_COLUMNS if c in concepts.columns]:
        rows.append({"concept": c, "mentions": int(pd.to_numeric(concepts[c], errors="coerce").fillna(0).sum())})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("mentions", ascending=False)
    ax = df.sort_values("mentions").plot(kind="barh", x="concept", y="mentions", legend=False, figsize=(10, 6))
    ax.set_title("Supply-chain concept mentions")
    ax.set_xlabel("Mentions")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def save_network_png(G: nx.DiGraph, out_png: Path, top_nodes: int = 120):
    if G.number_of_nodes() == 0:
        return
    if G.number_of_nodes() > top_nodes:
        keep = {n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_nodes]}
        G = G.subgraph(keep).copy()
    pos = nx.spring_layout(G, seed=42, k=1.0, iterations=80)
    plt.figure(figsize=(16, 11))
    sizes = [250 + 80 * np.sqrt(max(1, G.degree(n))) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, alpha=0.30, width=0.8)
    labels = {n: d.get("label", n.replace("COMPANY::", "").replace("ENTITY::", "")[:18]) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
    plt.title("Inter-firm / Entity Relationship Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def save_network_html(G: nx.DiGraph, out_html: Path, top_nodes: int = 120):
    if not PLOTLY_AVAILABLE or G.number_of_nodes() == 0:
        return
    if G.number_of_nodes() > top_nodes:
        keep = {n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_nodes]}
        G = G.subgraph(keep).copy()
    pos = nx.spring_layout(G, seed=42, k=1.0, iterations=80)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none", name="relationship edges")
    node_x, node_y, node_text, hover, sizes = [], [], [], [], []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        label = d.get("label", n.replace("COMPANY::", "").replace("ENTITY::", "")[:20])
        node_x.append(x); node_y.append(y); node_text.append(label)
        deg = G.degree(n)
        sizes.append(10 + 4 * np.sqrt(max(1, deg)))
        hover.append(f"<b>{label}</b><br>node: {n}<br>type: {d.get('node_type','')}<br>company/entity: {d.get('company','')}<br>in-degree: {G.in_degree(n)}<br>out-degree: {G.out_degree(n)}<br>degree: {deg}")
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center", hovertext=hover, hoverinfo="text", marker=dict(size=sizes, line=dict(width=1)), name="nodes")
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="Inter-firm / Entity Relationship Network", showlegend=True, hovermode="closest", margin=dict(b=20, l=10, r=10, t=60), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"SAVED {out_html}")


def plot_rate_bar(df: pd.DataFrame, name_col: str, value_col: str, title: str, out_png: Path, top_n: int = 30):
    if df.empty or value_col not in df.columns:
        return
    d = df.copy()
    if name_col not in d.columns:
        d[name_col] = d["signal"].astype(str) + " | " + d.get("source_label", pd.Series([""] * len(d))).astype(str) + " | " + d.get("relation_group", pd.Series([""] * len(d))).astype(str)
    d = d.sort_values(value_col, ascending=False).head(top_n)
    ax = d.sort_values(value_col).plot(kind="barh", x=name_col, y=value_col, legend=False, figsize=(12, 8))
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"SAVED {out_png}")


def main():
    args = parse_args()
    rag_dir = Path(args.rag_output_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir / "figures")
    print("=" * 90)
    print("Network + contagion master analysis")
    print("rag_output_dir:", rag_dir)
    print("out_dir:", out_dir)
    print("quarter range:", args.start_quarter or "ALL", "to", args.end_quarter or "ALL")
    print("contagion:", args.source_quarter, "->", args.target_quarter)
    print("focus:", args.focus_signal, args.focus_label)
    print("=" * 90)

    concept_files, rel_files, outlook_files, failed_files = discover_csvs(rag_dir)
    manifest = pd.DataFrame({
        "kind": (["concepts"] * len(concept_files)) + (["relationships"] * len(rel_files)) + (["outlook"] * len(outlook_files)) + (["failed"] * len(failed_files)),
        "path": [str(x) for x in concept_files + rel_files + outlook_files + failed_files],
    })
    save_csv(manifest, out_dir / "input_file_manifest.csv")

    raw_concepts = read_many_csv(concept_files, "concepts")
    raw_relationships = read_many_csv(rel_files, "relationships")
    raw_outlook = read_many_csv(outlook_files, "outlook")

    concepts = clean_concepts(raw_concepts, args.start_quarter, args.end_quarter)
    relationships = clean_relationships(raw_relationships, args.start_quarter, args.end_quarter)
    outlook = clean_outlook(raw_outlook, args.start_quarter, args.end_quarter)

    save_csv(concepts, out_dir / "cleaned_concepts_all.csv")
    save_csv(relationships, out_dir / "cleaned_relationships_all.csv")
    save_csv(outlook, out_dir / "cleaned_outlook_all.csv")

    if not outlook.empty:
        outlook_dist = outlook.groupby(["quarter", "signal", "label"], dropna=False).size().reset_index(name="count").sort_values(["quarter", "signal", "count"], ascending=[True, True, False])
        save_csv(outlook_dist, out_dir / "outlook_signal_label_distribution.csv")
        signal_matrix = outlook.pivot_table(index=["company_node", "ticker", "current_company", "quarter", "quarter_index"], columns="signal", values="score", aggfunc="mean").reset_index()
        for s in STANDARD_SIGNALS:
            if s not in signal_matrix.columns:
                signal_matrix[s] = np.nan
        signal_matrix["active_signal_count"] = signal_matrix[STANDARD_SIGNALS].apply(lambda r: np.sum(r.fillna(0).abs() > 0), axis=1)
        save_csv(signal_matrix, out_dir / "company_quarter_signal_matrix.csv")

    if not concepts.empty:
        concept_rows = []
        for c in [c for c in CONCEPT_COLUMNS if c in concepts.columns]:
            vals = pd.to_numeric(concepts[c], errors="coerce").fillna(0)
            concept_rows.append({"concept": c, "mentions": int(vals.sum()), "rows": int(len(concepts)), "mention_rate": float(vals.mean())})
        if concept_rows:
            save_csv(pd.DataFrame(concept_rows).sort_values("mentions", ascending=False), out_dir / "concept_distribution.csv")

    G, network_nodes, network_edges = build_relationship_network(relationships, outlook)
    save_csv(network_nodes, out_dir / "network_nodes.csv")
    save_csv(network_edges, out_dir / "network_edges.csv")
    if not network_nodes.empty:
        save_csv(network_nodes.sort_values(["degree", "out_degree", "in_degree"], ascending=False), out_dir / "network_centrality_degree.csv")

    contagion_events, unmatched = build_contagion_events(outlook, relationships, args.source_quarter, args.target_quarter, args.include_self_edges)
    save_csv(contagion_events, out_dir / "contagion_events.csv")
    save_csv(unmatched, out_dir / "unmatched_relationship_entities.csv")
    contagion_summary = summarize_contagion(contagion_events)
    save_csv(contagion_summary, out_dir / "contagion_summary_by_signal_label_relation.csv")
    falsification_cases, falsification_summary = focus_falsification(contagion_events, args.focus_signal, args.focus_label)
    save_csv(falsification_cases, out_dir / f"falsification_cases_{args.focus_signal}_{args.focus_label}.csv")
    save_csv(falsification_summary, out_dir / f"falsification_summary_{args.focus_signal}_{args.focus_label}.csv")

    plot_outlook_distribution(outlook, fig_dir / "outlook_label_distribution_by_signal.png")
    plot_concept_distribution(concepts, fig_dir / "concept_distribution.png")
    save_network_png(G, fig_dir / "relationship_network_static.png", top_nodes=args.top_network_nodes)
    save_network_html(G, out_dir / "relationship_network_interactive.html", top_nodes=args.top_network_nodes)
    if not contagion_summary.empty:
        tmp = contagion_summary.copy()
        tmp["name"] = tmp["signal"].astype(str) + " | " + tmp["source_label"].astype(str) + " | " + tmp["relation_group"].astype(str)
        plot_rate_bar(tmp, "name", "direction_transmission_rate", "Candidate same-direction transmission rate", fig_dir / "top_direction_transmission_rates.png")
    if not falsification_summary.empty:
        plot_rate_bar(falsification_summary, "relation_group", "direction_falsification_rate", f"Falsification rate for {args.focus_signal}={args.focus_label}", fig_dir / f"falsification_rate_{args.focus_signal}_{args.focus_label}.png")

    lines = []
    lines.append("# Network and Contagion Master Analysis")
    lines.append("")
    lines.append(f"- RAG output directory: `{rag_dir}`")
    lines.append(f"- Output directory: `{out_dir}`")
    lines.append(f"- Quarter range: `{args.start_quarter or 'ALL'}` to `{args.end_quarter or 'ALL'}`")
    lines.append(f"- Contagion pair: `{args.source_quarter}` → `{args.target_quarter}`")
    lines.append(f"- Focus falsification: `{args.focus_signal} = {args.focus_label}`")
    lines.append("")
    lines.append("## Loaded extraction files")
    lines.append(f"- Concepts files: {len(concept_files)}")
    lines.append(f"- Relationships files: {len(rel_files)}")
    lines.append(f"- Outlook files: {len(outlook_files)}")
    lines.append(f"- Failed files: {len(failed_files)}")
    lines.append("")
    lines.append("## Cleaned rows")
    lines.append(f"- Concepts rows: {len(concepts):,}")
    lines.append(f"- Relationship rows: {len(relationships):,}")
    lines.append(f"- Outlook rows: {len(outlook):,}")
    lines.append("")
    lines.append("## Network")
    lines.append(f"- Network nodes: {G.number_of_nodes():,}")
    lines.append(f"- Network edges: {G.number_of_edges():,}")
    lines.append("")
    lines.append("## Contagion")
    lines.append(f"- Contagion event rows: {len(contagion_events):,}")
    lines.append(f"- Unmatched relationship entity rows: {len(unmatched):,}")
    lines.append("")
    if not falsification_summary.empty:
        lines.append("## Focus falsification summary")
        lines.append(falsification_summary.to_markdown(index=False))
        lines.append("")
    lines.append("## Key files")
    lines.append("- `relationship_network_interactive.html`")
    lines.append("- `network_nodes.csv`")
    lines.append("- `network_edges.csv`")
    lines.append("- `contagion_events.csv`")
    lines.append("- `contagion_summary_by_signal_label_relation.csv`")
    lines.append("- `company_quarter_signal_matrix.csv`")
    (out_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SAVED {out_dir / 'analysis_summary.md'}")
    print("\nDONE. Open:")
    print(out_dir / "relationship_network_interactive.html")


if __name__ == "__main__":
    main()
