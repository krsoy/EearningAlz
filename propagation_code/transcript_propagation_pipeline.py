import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


QUARTER_RE = re.compile(r"Q([1-4])[_-]?(20\\d{2})", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")

DOMAIN_KEYWORDS: Dict[str, Sequence[str]] = {
    "AI Infrastructure": ("ai", "gpu", "inference", "training", "model", "cluster", "accelerator", "llm"),
    "Enterprise Software": ("enterprise", "software", "subscription", "saas", "crm", "productivity", "platform"),
    "Cloud and Data": ("cloud", "datacenter", "data center", "storage", "compute", "workload"),
    "Consumer and Commerce": ("consumer", "ecommerce", "ad", "advertising", "retail", "travel", "booking"),
    "Financial Services": ("credit", "loan", "bank", "payment", "deposit", "insurance"),
    "Supply Chain and Manufacturing": ("supply", "inventory", "manufacturing", "capacity", "shipment", "yield"),
}

SIGNAL_TERMS = {
    "guidance", "demand", "pricing", "margin", "capex", "opex", "pipeline", "adoption",
    "deployment", "productivity", "efficiency", "headwind", "tailwind", "macro", "enterprise",
    "competition", "partner", "customer", "expansion", "growth", "utilization", "backlog",
}


@dataclass
class TranscriptRecord:
    ticker: str
    quarter: str
    year: int
    quarter_num: int
    quarter_index: int
    path: Path
    text: str
    top_signals: List[str]
    top_entities: List[str]
    dominant_domain: str
    domain_distribution: Dict[str, int]


@dataclass
class EdgeRecord:
    source_id: str
    target_id: str
    temporal_distance: int
    similarity: float
    theme_overlap: float
    score: float
    shared_signals: List[str]


def parse_quarter(name: str) -> Tuple[str, int, int, int]:
    match = QUARTER_RE.search(name)
    if not match:
        return "UNKNOWN", 0, 0, -1
    q_num = int(match.group(1))
    year = int(match.group(2))
    quarter = f"Q{q_num}_{year}"
    return quarter, year, q_num, year * 4 + q_num


def iter_transcripts(data_dir: Path) -> List[TranscriptRecord]:
    records: List[TranscriptRecord] = []
    if not data_dir.exists() or not data_dir.is_dir():
        return records
    for ticker_dir in sorted(data_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        for fp in sorted(ticker_dir.glob("*.txt")):
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue
            quarter, year, q_num, q_idx = parse_quarter(fp.name)
            if quarter == "UNKNOWN":
                continue
            domain_counts = score_domains(text)
            dominant_domain = max(domain_counts.items(), key=lambda kv: kv[1])[0]
            top_signals = extract_top_signals(text)
            top_entities = extract_top_entities(text)
            records.append(
                TranscriptRecord(
                    ticker=ticker,
                    quarter=quarter,
                    year=year,
                    quarter_num=q_num,
                    quarter_index=q_idx,
                    path=fp,
                    text=text,
                    top_signals=top_signals,
                    top_entities=top_entities,
                    dominant_domain=dominant_domain,
                    domain_distribution=domain_counts,
                )
            )
    return records


def score_domains(text: str) -> Dict[str, int]:
    lower_text = text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(lower_text.count(k.lower()) for k in keywords)
    if not any(scores.values()):
        scores["Unclassified"] = 1
    return scores


def extract_top_signals(text: str, limit: int = 12) -> List[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    counts = Counter(t for t in tokens if t in SIGNAL_TERMS)
    return [w for w, _ in counts.most_common(limit)]


def extract_top_entities(text: str, limit: int = 12) -> List[str]:
    tokens = TOKEN_RE.findall(text)
    counts = Counter(t for t in tokens if t[0].isupper() and len(t) > 2)
    stop = {"Operator", "Question", "Answer", "Prepared", "Remarks", "Call", "Quarter"}
    filtered = [(k, v) for k, v in counts.items() if k not in stop]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in filtered[:limit]]


def build_edges(records: Sequence[TranscriptRecord], threshold: float = 0.36) -> List[EdgeRecord]:
    corpus = [r.text[:4000] for r in records]
    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(matrix)

    edges: List[EdgeRecord] = []
    for i, src in enumerate(records):
        for j, dst in enumerate(records):
            if i == j or src.ticker == dst.ticker:
                continue
            if src.quarter_index < 0 or dst.quarter_index < 0:
                continue
            temporal_distance = dst.quarter_index - src.quarter_index
            if temporal_distance < 0 or temporal_distance > 1:
                continue

            similarity = float(sim[i, j])
            shared_signals = sorted(set(src.top_signals) & set(dst.top_signals))
            theme_overlap = len(shared_signals) / max(len(set(src.top_signals) | set(dst.top_signals)), 1)
            temporal_weight = 1.0 if temporal_distance == 0 else 0.8
            score = 0.65 * similarity + 0.25 * theme_overlap + 0.10 * temporal_weight
            if score >= threshold:
                edges.append(
                    EdgeRecord(
                        source_id=f"{src.ticker}|{src.quarter}",
                        target_id=f"{dst.ticker}|{dst.quarter}",
                        temporal_distance=temporal_distance,
                        similarity=similarity,
                        theme_overlap=theme_overlap,
                        score=score,
                        shared_signals=shared_signals[:8],
                    )
                )

    edges.sort(key=lambda e: e.score, reverse=True)
    return edges


def summarize(records: Sequence[TranscriptRecord], edges: Sequence[EdgeRecord], top_n: int = 20) -> Dict[str, object]:
    incoming = Counter(e.target_id for e in edges)
    outgoing = Counter(e.source_id for e in edges)
    by_domain = Counter(r.dominant_domain for r in records)
    strongest_edges = [
        {
            "source": e.source_id,
            "target": e.target_id,
            "score": round(e.score, 4),
            "similarity": round(e.similarity, 4),
            "theme_overlap": round(e.theme_overlap, 4),
            "shared_signals": e.shared_signals,
        }
        for e in list(edges)[:top_n]
    ]
    return {
        "total_transcripts": len(records),
        "total_edges": len(edges),
        "most_influential_nodes": outgoing.most_common(12),
        "most_infected_nodes": incoming.most_common(12),
        "domain_distribution": dict(by_domain),
        "strongest_edges": strongest_edges,
    }


def write_nodes(records: Sequence[TranscriptRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_id",
                "ticker",
                "quarter",
                "quarter_index",
                "dominant_domain",
                "top_signals",
                "top_entities",
                "source_path",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    f"{r.ticker}|{r.quarter}",
                    r.ticker,
                    r.quarter,
                    r.quarter_index,
                    r.dominant_domain,
                    ";".join(r.top_signals),
                    ";".join(r.top_entities),
                    str(r.path),
                ]
            )


def write_edges(edges: Sequence[EdgeRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_id",
                "target_id",
                "temporal_distance",
                "similarity",
                "theme_overlap",
                "score",
                "shared_signals",
            ]
        )
        for e in edges:
            writer.writerow(
                [
                    e.source_id,
                    e.target_id,
                    e.temporal_distance,
                    f"{e.similarity:.6f}",
                    f"{e.theme_overlap:.6f}",
                    f"{e.score:.6f}",
                    ";".join(e.shared_signals),
                ]
            )


def write_extractions(records: Sequence[TranscriptRecord], output: Path) -> None:
    data = []
    for r in records:
        snippets = pick_signal_snippets(r.text, r.top_signals)
        data.append(
            {
                "node_id": f"{r.ticker}|{r.quarter}",
                "ticker": r.ticker,
                "quarter": r.quarter,
                "dominant_domain": r.dominant_domain,
                "top_signals": r.top_signals,
                "top_entities": r.top_entities,
                "signal_snippets": snippets,
            }
        )
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def pick_signal_snippets(text: str, signals: Sequence[str], limit: int = 5) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if len(s.strip()) > 35]
    selected: List[str] = []
    signal_set = {s.lower() for s in signals}
    for sentence in sentences:
        lower = sentence.lower()
        if any(sig in lower for sig in signal_set):
            selected.append(sentence)
            if len(selected) >= limit:
                break
    return selected


def write_report(summary: Dict[str, object], output: Path) -> None:
    influential = "\n".join(f"- {node}: {count}" for node, count in summary["most_influential_nodes"])
    infected = "\n".join(f"- {node}: {count}" for node, count in summary["most_infected_nodes"])
    edges = "\n".join(
        f"- {e['source']} -> {e['target']} | score={e['score']} | shared signals: {', '.join(e['shared_signals']) or 'none'}"
        for e in summary["strongest_edges"][:12]
    )
    domain_dist = "\n".join(f"- {k}: {v}" for k, v in summary["domain_distribution"].items())

    content = f"""# Information Propagation Report

## Snapshot
- Total transcripts analyzed: {summary['total_transcripts']}
- Propagation edges discovered: {summary['total_edges']}

## Most Influential Nodes (Out-degree)
{influential}

## Most Infected Nodes (In-degree)
{infected}

## Domain Mix
{domain_dist}

## Strongest Propagation Paths
{edges}

## Additional Analysis Ideas
- Add event-study overlays linking propagation spikes to post-call return volatility.
- Estimate directional lag effects by sector using panel regression on quarterly similarity scores.
- Track narrative drift between management prepared remarks and Q&A sections.
- Measure executive-level influence by mapping speaker-level language to cross-company adoption.
- Build contradiction detection to identify when downstream companies negate upstream narratives.

## Additional Information Extraction Opportunities
- Fine-grained guidance decomposition (revenue, margin, capex, hiring, geography).
- Competitor mention polarity and response strategy extraction.
- Supply chain dependency statements and risk level grading.
- Customer cohort adoption stage extraction (trial, pilot, production, scale).
- Product launch timeline extraction and confidence scoring.
"""
    output.write_text(content, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build information propagation artifacts from earnings transcripts.")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", str(project_root / "data")))
    parser.add_argument("--output-dir", default=os.getenv("PROPAGATION_RESULTS_DIR", str(project_root / "propagation_results")))
    parser.add_argument("--edge-threshold", type=float, default=0.36)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = iter_transcripts(data_dir)
    if not records:
        raise SystemExit(f"No transcripts found in {data_dir}")

    edges = build_edges(records, threshold=args.edge_threshold)
    summary = summarize(records, edges)

    write_nodes(records, output_dir / "propagation_nodes.csv")
    write_edges(edges, output_dir / "propagation_edges.csv")
    write_extractions(records, output_dir / "extraction_samples.json")
    (output_dir / "propagation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, output_dir / "analysis_report.md")

    print(f"[done] Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
