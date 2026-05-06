import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "intel_hf_validation_strict"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 200


def safe_read_csv(path: Path):
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None


# ===========================================================
# PLOT 1: Next Impacted Intel Nodes
# ===========================================================
next_impact = safe_read_csv(DATA_DIR / "next_impacted_intel_nodes_strict_weighted.csv")
if next_impact is not None and not next_impact.empty:
    top_n = 12
    plot_df = next_impact.head(top_n).copy()
    if {"target_business_node", "target_signal"}.issubset(plot_df.columns):
        plot_df["label"] = plot_df["target_business_node"].astype(str) + " | " + plot_df["target_signal"].astype(str)
    else:
        plot_df["label"] = plot_df.index.astype(str)
    x_col = "impact_score" if "impact_score" in plot_df.columns else plot_df.columns[0]

    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, y="label", x=x_col, palette="viridis")
    plt.title("Top Next Impacted Intel Nodes (Strict Weighted)")
    plt.xlabel(x_col)
    plt.ylabel("Business Node | Signal")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_next_impacted_nodes_top12.png")
    plt.close()
    print("[OK] Plot 1 saved.")


# ===========================================================
# PLOT 2: Company-Signal Heatmap
# ===========================================================
summary = safe_read_csv(DATA_DIR / "company_signal_summary_strict_weighted.csv")
if summary is not None and not summary.empty:
    if {"source_company", "target_signal", "total_weight"}.issubset(summary.columns):
        top_companies = (
            summary.groupby("source_company", as_index=False)["total_weight"]
            .sum().sort_values("total_weight", ascending=False)
            .head(15)["source_company"].tolist()
        )
        heat_df = summary[summary["source_company"].isin(top_companies)].copy()
        pivot_df = heat_df.pivot_table(
            index="source_company", columns="target_signal",
            values="total_weight", aggfunc="sum", fill_value=0
        )
        plt.figure(figsize=(11, 7))
        sns.heatmap(pivot_df, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.3)
        plt.title("Company vs Signal Total Weight (Top 15 Companies)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "02_company_signal_heatmap.png")
        plt.close()
        print("[OK] Plot 2 saved.")


# ===========================================================
# PLOT 3: KG PageRank Bar
# ===========================================================
pr = safe_read_csv(DATA_DIR / "kg_pagerank_strict_weighted.csv")
if pr is not None and not pr.empty:
    if {"node", "pagerank", "node_type"}.issubset(pr.columns):
        plot_df = pr.sort_values("pagerank", ascending=False).head(20).copy()
        plot_df = plot_df.sort_values("pagerank", ascending=True)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=plot_df, y="node", x="pagerank",
                    hue="node_type", dodge=False, palette="Set2")
        plt.title("Top 20 KG Nodes by PageRank")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "03_kg_pagerank_top20.png")
        plt.close()
        print("[OK] Plot 3 saved.")


# ===========================================================
# PLOT 4: Signal Trend Over Time
# ===========================================================
edges_df = safe_read_csv(DATA_DIR / "kg_edges_strict_weighted.csv")
if edges_df is not None and not edges_df.empty:
    if {"publish_date", "target_signal", "weight"}.issubset(edges_df.columns):
        edges_df["publish_date"] = pd.to_datetime(edges_df["publish_date"], errors="coerce")
        trend_df = edges_df.dropna(subset=["publish_date"]).copy()
        trend_df["month"] = trend_df["publish_date"].dt.to_period("M").dt.to_timestamp()
        line_df = (
            trend_df.groupby(["month", "target_signal"], as_index=False)["weight"]
            .sum().sort_values("month")
        )
        if not line_df.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=line_df, x="month", y="weight",
                         hue="target_signal", marker="o")
            plt.title("Signal Weight Trend by Month")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "04_signal_trend_monthly.png")
            plt.close()
            print("[OK] Plot 4 saved.")


# ===========================================================
# PLOT 5a: Knowledge Graph — Full Layout (spring)
# ===========================================================
nx_edges_df = safe_read_csv(DATA_DIR / "networkx_edge_list.csv")
nx_nodes_df = safe_read_csv(DATA_DIR / "networkx_node_list.csv")

if nx_edges_df is not None and nx_nodes_df is not None:
    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in nx_nodes_df.iterrows():
        G.add_node(row["node"],
                   node_type=row.get("node_type", "unknown"),
                   label=str(row.get("name", "")) or str(row["node"]))

    # Add edges
    for _, row in nx_edges_df.iterrows():
        G.add_edge(row["source"], row["target"],
                   relation=row.get("relation", ""),
                   weight=float(row.get("weight", 1.0)))

    print(f"[KG] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # ------------------------------------------
    # Color scheme by node type
    # ------------------------------------------
    type_colors = {
        "company":            "#4E79A7",
        "signal":             "#F28E2B",
        "intel_business_node":"#E15759",
        "unknown":            "#BAB0AC",
    }

    node_types = [G.nodes[n].get("node_type", "unknown") for n in G.nodes()]
    node_colors = [type_colors.get(t, "#BAB0AC") for t in node_types]

    # Node size: scale by pagerank if available
    if pr is not None and not pr.empty and "node" in pr.columns:
        pr_dict = dict(zip(pr["node"], pr["pagerank"]))
        min_size, max_size = 200, 2000
        pr_vals = [pr_dict.get(n, 0) for n in G.nodes()]
        if max(pr_vals) > 0:
            pr_norm = [(v / max(pr_vals)) for v in pr_vals]
        else:
            pr_norm = [0.1] * len(pr_vals)
        node_sizes = [min_size + (max_size - min_size) * v for v in pr_norm]
    else:
        node_sizes = 400

    # Edge width: scale by weight
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    edge_widths = [0.3 + 2.5 * (w / max_w) for w in weights]

    # ------------------------------------------
    # Layout: spring, filtered to top-N by weight
    # ------------------------------------------
    # Show top 60 nodes by total edge weight to keep readable
    node_total_weight = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        node_total_weight[u] = node_total_weight.get(u, 0) + d.get("weight", 1)
        node_total_weight[v] = node_total_weight.get(v, 0) + d.get("weight", 1)

    top_nodes = sorted(node_total_weight, key=node_total_weight.get, reverse=True)[:60]
    G_sub = G.subgraph(top_nodes).copy()

    sub_node_types   = [G.nodes[n].get("node_type", "unknown") for n in G_sub.nodes()]
    sub_node_colors  = [type_colors.get(t, "#BAB0AC") for t in sub_node_types]
    sub_weights_raw  = [G_sub[u][v].get("weight", 1.0) for u, v in G_sub.edges()]
    sub_max_w        = max(sub_weights_raw) if sub_weights_raw else 1.0
    sub_edge_widths  = [0.3 + 2.5 * (w / sub_max_w) for w in sub_weights_raw]

    if pr is not None and not pr.empty:
        pr_dict = dict(zip(pr["node"], pr["pagerank"]))
        pr_vals_sub = [pr_dict.get(n, 0) for n in G_sub.nodes()]
        mx = max(pr_vals_sub) if max(pr_vals_sub) > 0 else 1
        sub_node_sizes = [200 + 2000 * (v / mx) for v in pr_vals_sub]
    else:
        sub_node_sizes = 400

    pos = nx.spring_layout(G_sub, seed=42, k=2.2, iterations=60)

    # Labels: shorten long names
    def short_label(n):
        n = str(n)
        for prefix in ("COMPANY::", "SIGNAL::", "BUSINESS::"):
            if n.startswith(prefix):
                n = n[len(prefix):]
        return n[:18]

    labels = {n: short_label(n) for n in G_sub.nodes()}

    fig, ax = plt.subplots(figsize=(18, 13))

    nx.draw_networkx_edges(
        G_sub, pos, ax=ax,
        width=sub_edge_widths,
        alpha=0.45,
        edge_color="#888888",
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.08"
    )

    nx.draw_networkx_nodes(
        G_sub, pos, ax=ax,
        node_color=sub_node_colors,
        node_size=sub_node_sizes,
        alpha=0.92
    )

    nx.draw_networkx_labels(
        G_sub, pos, labels=labels, ax=ax,
        font_size=7, font_color="black"
    )

    # Legend
    patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in type_colors.items() if t != "unknown"
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=10, title="Node Type")
    ax.set_title("Intel Earnings Knowledge Graph\n(Top 60 nodes by edge weight, sized by PageRank)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_knowledge_graph.png", bbox_inches="tight")
    plt.close()
    print("[OK] Plot 5a (KG full spring layout) saved.")

    # ------------------------------------------
    # PLOT 5b: Ego-graph centred on Intel business nodes
    # ------------------------------------------
    intel_biz_nodes = [n for n in G.nodes()
                       if G.nodes[n].get("node_type") == "intel_business_node"]

    if intel_biz_nodes:
        ego = nx.DiGraph()
        for center in intel_biz_nodes:
            ego.add_nodes_from(G.pred[center])   # companies -> signals -> biz
            ego.add_nodes_from(G.succ[center])
            ego.add_node(center)
            ego.add_edges_from(G.in_edges(center, data=True))
            ego.add_edges_from(G.out_edges(center, data=True))
            # one more hop: signal predecessors
            for sig in list(G.pred[center]):
                ego.add_edges_from(G.in_edges(sig, data=True))
                ego.add_nodes_from(G.pred[sig])

        # carry node attributes
        for n in ego.nodes():
            if n in G.nodes:
                ego.nodes[n].update(G.nodes[n])

        ego_colors  = [type_colors.get(ego.nodes[n].get("node_type","unknown"), "#BAB0AC") for n in ego.nodes()]
        ego_weights = [ego[u][v].get("weight", 1.0) for u, v in ego.edges()]
        ego_max_w   = max(ego_weights) if ego_weights else 1.0
        ego_widths  = [0.3 + 2.8 * (w / ego_max_w) for w in ego_weights]

        if pr is not None and not pr.empty:
            pr_vals_ego = [pr_dict.get(n, 0) for n in ego.nodes()]
            mx = max(pr_vals_ego) if max(pr_vals_ego) > 0 else 1
            ego_sizes = [300 + 2500 * (v / mx) for v in pr_vals_ego]
        else:
            ego_sizes = 500

        pos_ego = nx.spring_layout(ego, seed=7, k=3.0, iterations=80)
        ego_labels = {n: short_label(n) for n in ego.nodes()}

        fig, ax = plt.subplots(figsize=(16, 11))
        nx.draw_networkx_edges(
            ego, pos_ego, ax=ax,
            width=ego_widths, alpha=0.5, edge_color="#666666",
            arrows=True, arrowsize=14,
            connectionstyle="arc3,rad=0.06"
        )
        nx.draw_networkx_nodes(
            ego, pos_ego, ax=ax,
            node_color=ego_colors, node_size=ego_sizes, alpha=0.93
        )
        nx.draw_networkx_labels(
            ego, pos_ego, labels=ego_labels, ax=ax,
            font_size=7.5, font_color="black"
        )
        ax.legend(handles=patches, loc="upper left", fontsize=10, title="Node Type")
        ax.set_title("Intel Business Node Ego-Graph\n(Company → Signal → Intel Business Unit, edge width = weighted mentions)", fontsize=13)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "06_intel_ego_graph.png", bbox_inches="tight")
        plt.close()
        print("[OK] Plot 5b (Intel ego-graph) saved.")

# ===========================================================
print(f"\nAll plots saved to: {PLOT_DIR.resolve()}")
for p in sorted(PLOT_DIR.glob("*.png")):
    print(f"  {p.name}")
