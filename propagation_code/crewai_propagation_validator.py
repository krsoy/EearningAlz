import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+")


@dataclass
class EdgeInput:
    source_id: str
    target_id: str
    score: float
    shared_signals: List[str]


def load_edges(edge_file: Path, top_n: int) -> List[EdgeInput]:
    rows: List[EdgeInput] = []
    with edge_file.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                EdgeInput(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    score=float(row["score"]),
                    shared_signals=[s for s in row["shared_signals"].split(";") if s],
                )
            )
    rows.sort(key=lambda r: r.score, reverse=True)
    return rows[:top_n]


def transcript_path(base_data_dir: Path, node_id: str) -> Path:
    ticker, quarter = node_id.split("|")
    return base_data_dir / ticker / f"{quarter}.txt"


def extract_supporting_sentences(text: str, signals: Sequence[str], limit: int = 4) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if len(s.strip()) > 30]
    supports: List[str] = []
    signal_set = {s.lower() for s in signals}
    for sentence in sentences:
        lower = sentence.lower()
        if any(sig in lower for sig in signal_set):
            supports.append(sentence)
            if len(supports) >= limit:
                break
    return supports


def rule_based_validate(edge: EdgeInput, source_text: str, target_text: str) -> Dict[str, object]:
    src_support = extract_supporting_sentences(source_text, edge.shared_signals)
    dst_support = extract_supporting_sentences(target_text, edge.shared_signals)

    shared_signal_strength = min(len(edge.shared_signals) / 6.0, 1.0)
    evidence_strength = min((len(src_support) + len(dst_support)) / 8.0, 1.0)
    confidence = 0.55 * edge.score + 0.25 * shared_signal_strength + 0.20 * evidence_strength

    if confidence >= 0.75:
        grade = "A"
    elif confidence >= 0.62:
        grade = "B"
    elif confidence >= 0.50:
        grade = "C"
    else:
        grade = "D"

    explanation = (
        f"The propagation path from {edge.source_id} to {edge.target_id} is supported by "
        f"{len(edge.shared_signals)} shared narrative signals ({', '.join(edge.shared_signals) or 'none'}), "
        f"source evidence snippets={len(src_support)}, target evidence snippets={len(dst_support)}, "
        f"and base path score={edge.score:.3f}."
    )

    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "path_score": round(edge.score, 4),
        "validator_confidence": round(confidence, 4),
        "grade": grade,
        "shared_signals": edge.shared_signals,
        "source_evidence": src_support,
        "target_evidence": dst_support,
        "explanation": explanation,
        "method": "rule-based fallback (CrewAI-ready schema)",
    }


def try_crewai_validation(payload: Dict[str, object]) -> Dict[str, object]:
    try:
        from crewai import Agent, Crew, Process, Task  # type: ignore
    except Exception:
        return payload

    if not payload.get("source_evidence") and not payload.get("target_evidence"):
        return payload

    prompt = (
        "Validate whether the information propagation path is plausible based on evidence chunks. "
        "Return JSON with keys: validator_confidence(0-1), grade(A-D), explanation.\n"
        f"Candidate path: {payload['source_id']} -> {payload['target_id']}\n"
        f"Path score: {payload['path_score']}\n"
        f"Shared signals: {payload['shared_signals']}\n"
        f"Source evidence: {payload['source_evidence']}\n"
        f"Target evidence: {payload['target_evidence']}"
    )

    validator = Agent(
        role="Propagation Validation Analyst",
        goal="Evaluate if a claimed propagation path is supported by textual evidence chunks.",
        backstory="You are strict, evidence-first, and produce concise machine-readable outputs.",
        verbose=False,
    )
    task = Task(description=prompt, expected_output="JSON only", agent=validator)
    crew = Crew(agents=[validator], tasks=[task], process=Process.sequential, verbose=False)

    try:
        output = crew.kickoff()
        payload["method"] = "crewai"
        payload["crewai_raw_output"] = str(output)
    except Exception as exc:
        payload["crewai_error"] = str(exc)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate high-score propagation paths with a CrewAI-ready workflow.")
    parser.add_argument("--results-dir", default="/tmp/workspace/krsoy/EearningAlz/propagation_results")
    parser.add_argument("--data-dir", default="/tmp/workspace/krsoy/EearningAlz/data")
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    edges = load_edges(results_dir / "propagation_edges.csv", args.top_n)
    validations = []

    for edge in edges:
        src_path = transcript_path(data_dir, edge.source_id)
        dst_path = transcript_path(data_dir, edge.target_id)
        if not src_path.exists() or not dst_path.exists():
            continue

        source_text = src_path.read_text(encoding="utf-8", errors="ignore")
        target_text = dst_path.read_text(encoding="utf-8", errors="ignore")

        payload = rule_based_validate(edge, source_text, target_text)
        payload = try_crewai_validation(payload)
        validations.append(payload)

    out_file = results_dir / "crew_validation_report.json"
    out_file.write_text(json.dumps(validations, indent=2), encoding="utf-8")
    print(f"[done] Wrote {len(validations)} validation records to {out_file}")


if __name__ == "__main__":
    main()
