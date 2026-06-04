#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrewAI Multi-Agent LLM-as-Judge Workflow.

For each judge case (sampled from earnings call events), three agents work together:

  1. EvidenceAnalyst   — reads the source transcript chunks and explains what signal
                         the LLM originally extracted (e.g. "supply_outlook: negative").
  2. RelationshipAuditor — reads both source and target chunks and evaluates whether
                           the inferred relationship (e.g. AAPL → CARR upstream) is
                           plausible and grounded in the text.
  3. JudgeAgent        — LLM-as-judge: gives a verdict (VALID / PARTIALLY_VALID / INVALID)
                         with a confidence score and explanation.

Outputs: judge_results.jsonl  (one result per case)

Usage (local vLLM):
    # Start vLLM first:
    # python -m vllm.entrypoints.openai.api_server \
    #   --model Qwen/Qwen2.5-14B-Instruct --port 8000 --tensor-parallel-size 4

    python crewai_judge_workflow.py --cases judge_cases.jsonl --out judge_results.jsonl

Usage (cloud SLURM):
    See run_judge.slurm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

# ── Patch CrewAI's SQLite task-output cache before importing Crew ──────────
# CrewAI persists kickoff task outputs in a SQLite DB (~/.crewai/).
# On shared HPC nodes the DB can become corrupted across jobs.
# We replace the storage with a no-op in-memory stub so the DB is never touched.
try:
    import crewai.memory.storage.kickoff_task_outputs_storage as _ktos_mod

    class _NoOpTaskOutputStorage:
        """Drop-in replacement that keeps everything in memory only."""
        def __init__(self, *a, **kw): self._store: dict = {}
        def add(self, task, output, task_index):     self._store[task_index] = output
        def update(self, task_index, **kw):          self._store.setdefault(task_index, {}).update(kw)
        def load(self):                              return list(self._store.values())
        def delete(self):                            self._store.clear()

    _ktos_mod.KickoffTaskOutputsSQLiteStorage = _NoOpTaskOutputStorage
except Exception:
    pass  # older crewai versions may not have this module
# ──────────────────────────────────────────────────────────────────────────

# CrewAI imports — install with: uv add crewai
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM


# ============================================================
# Config
# ============================================================

DEFAULT_VLLM_URL  = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL     = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_CASES     = "judge_cases.jsonl"
DEFAULT_OUT       = "judge_results.jsonl"
MAX_CHUNK_CHARS   = 600   # truncate very long chunks for prompt efficiency
MAX_CHUNKS_PER_DOC = 5    # top N chunks per doc fed into prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases",    default=os.environ.get("JUDGE_CASES", DEFAULT_CASES))
    p.add_argument("--out",      default=os.environ.get("JUDGE_OUT",   DEFAULT_OUT))
    p.add_argument("--vllm-url", default=os.environ.get("VLLM_URL",    DEFAULT_VLLM_URL))
    p.add_argument("--model",    default=os.environ.get("LLM_MODEL",   DEFAULT_MODEL))
    p.add_argument("--max-cases",type=int, default=None,
                   help="Limit number of cases processed (for testing).")
    p.add_argument("--offset",   type=int, default=0,
                   help="Skip first N cases (for sharding across jobs).")
    p.add_argument("--resume",       action="store_true",
                   help="Skip cases already present in --out file.")
    p.add_argument("--retry-errors", action="store_true",
                   help="When resuming, re-process cases that previously resulted in ERROR.")
    p.add_argument("--max-retries",  type=int, default=2,
                   help="Number of retries on transient failure per case.")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens",  type=int,   default=1200)
    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def format_chunks(chunks: list[dict], max_n: int = MAX_CHUNKS_PER_DOC) -> str:
    if not chunks:
        return "(no evidence chunks available)"
    lines = []
    for i, c in enumerate(chunks[:max_n]):
        cid   = c.get("chunk_id", f"chunk_{i}")
        score = c.get("hybrid_score", "")
        group = c.get("evidence_group", "")
        text  = truncate(c.get("text", ""))
        lines.append(f"[{group} | chunk_id={cid} | score={score}]\n{text}")
    return "\n\n".join(lines)


def format_case_context(case: dict) -> str:
    return f"""
=== CASE {case['case_id']} ===
Analysis mode : {case['analysis_mode']}
Signal        : {case['signal']}
Relation group: {case['relation_group']}
Source direction (claimed): {case['source_direction']}
Target direction (claimed): {case['target_direction']}
Direction match: {case['direction_match']}

--- SOURCE TRANSCRIPT ---
Ticker  : {case['source_ticker']}  ({case['source_company']})
Quarter : {case['source_quarter']}
Date    : {case['source_publish_date']}

{format_chunks(case.get('source_chunks', []))}

--- TARGET TRANSCRIPT ---
Ticker  : {case['target_ticker']}  ({case['target_company']})
Quarter : {case['target_quarter']}
Date    : {case['target_publish_date']}

{format_chunks(case.get('target_chunks', []))}
""".strip()


def load_already_done(out_path: Path, skip_errors: bool = False) -> set[int]:
    done: set[int] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if skip_errors and obj.get("verdict") == "ERROR":
                        continue   # will be retried
                    done.add(int(obj["case_id"]))
                except Exception:
                    pass
    return done


# ============================================================
# CrewAI setup
# ============================================================

def build_llm(args) -> LLM:
    """Build a CrewAI LLM object pointing at the local vLLM server.

    CrewAI natively supports the 'hosted_vllm' provider without requiring
    litellm.  Using 'openai/' prefix triggers the litellm fallback path which
    requires an extra install.
    """
    return LLM(
        model=f"hosted_vllm/{args.model}",
        base_url=args.vllm_url,
        api_key="not-needed",          # vLLM does not require a real key
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def build_crew(llm: LLM) -> tuple[Agent, Agent, Agent, Crew]:
    # ---- Agent 1: Evidence Analyst ----
    evidence_analyst = Agent(
        role="Evidence Analyst",
        goal=(
            "Read the source earnings call transcript evidence chunks and explain "
            "what signal information the LLM likely extracted. Focus on the exact "
            "phrases that support or contradict the claimed signal and direction."
        ),
        backstory=(
            "You are an expert financial text analyst specialising in earnings call "
            "transcripts. You identify supply-chain signals, demand outlooks, and "
            "directional cues (positive/negative/neutral) from raw transcript text."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ---- Agent 2: Relationship Auditor ----
    relationship_auditor = Agent(
        role="Relationship Auditor",
        goal=(
            "Audit whether the inter-company relationship (source → target, with the "
            "given relation_group e.g. upstream/downstream/competitor) is plausible "
            "given what both transcripts actually say. Identify supporting and "
            "contradicting evidence."
        ),
        backstory=(
            "You are a supply-chain analyst who cross-references earnings call "
            "transcripts from two companies in the same or adjacent quarters to "
            "validate whether one company's signal propagates to or from the other."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ---- Agent 3: Judge ----
    judge_agent = Agent(
        role="LLM Judge",
        goal=(
            "Issue a final verdict on whether the extracted relationship is valid, "
            "partially valid, or invalid based on the analyses provided by the "
            "Evidence Analyst and the Relationship Auditor."
        ),
        backstory=(
            "You are an impartial LLM judge. You synthesise analyses from other "
            "agents and issue structured JSON verdicts. Your verdicts will be used "
            "to measure LLM extraction reliability."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    crew = Crew(
        agents=[evidence_analyst, relationship_auditor, judge_agent],
        process=Process.sequential,
        verbose=True,
        memory=False,           # disable RAG/long-term memory (avoids SQLite issues)
        output_log_file=False,  # don't write per-run log files
    )

    return evidence_analyst, relationship_auditor, judge_agent, crew


def run_case(case: dict, evidence_analyst: Agent, relationship_auditor: Agent,
             judge_agent: Agent, crew: Crew, max_retries: int = 2) -> dict:
    context = format_case_context(case)

    task1 = Task(
        description=(
            f"Analyse the SOURCE transcript evidence chunks below and explain what "
            f"signal '{case['signal']}' with direction '{case['source_direction']}' "
            f"the original LLM likely extracted, and whether the text genuinely "
            f"supports that signal.\n\n{context}"
        ),
        expected_output=(
            "A paragraph (3-6 sentences) explaining: (1) what phrases in the source "
            "transcript support or contradict the claimed signal and direction, "
            "(2) overall quality of evidence for this signal."
        ),
        agent=evidence_analyst,
    )

    task2 = Task(
        description=(
            f"Given the evidence analysis above, examine BOTH the source "
            f"({case['source_ticker']}, {case['source_quarter']}) and target "
            f"({case['target_ticker']}, {case['target_quarter']}) transcript chunks. "
            f"Evaluate whether the relation_group='{case['relation_group']}' "
            f"(source → target) is plausible and textually grounded.\n\n{context}"
        ),
        expected_output=(
            "A paragraph (3-6 sentences) covering: (1) whether both transcripts "
            "mention each other or shared themes, (2) whether the relation direction "
            "is plausible, (3) any contradicting signals."
        ),
        agent=relationship_auditor,
    )

    task3 = Task(
        description=(
            "Based on the Evidence Analyst's and Relationship Auditor's findings, "
            "issue a final verdict as a JSON object with this exact schema:\n"
            "{\n"
            '  "verdict": "VALID" | "PARTIALLY_VALID" | "INVALID",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "signal_evidence_quality": "strong" | "moderate" | "weak" | "absent",\n'
            '  "relation_plausibility": "high" | "medium" | "low",\n'
            '  "key_supporting_phrases": ["phrase1", "phrase2"],\n'
            '  "key_contradicting_phrases": ["phrase1"],\n'
            '  "judge_reasoning": "one sentence summary"\n'
            "}\n"
            "Return ONLY the JSON object, no markdown, no extra text."
        ),
        expected_output="A valid JSON object matching the schema above.",
        agent=judge_agent,
    )

    crew.tasks = [task1, task2, task3]

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = crew.kickoff()
            raw_output = str(result)
            break
        except Exception as e:
            last_error = str(e)
            print(f"  [attempt {attempt+1}/{max_retries+1}] ERROR: {last_error[:300]}")
            if attempt < max_retries:
                time.sleep(5 * (attempt + 1))
    else:
        # All retries exhausted
        return {
            "case_id":          case["case_id"],
            "signal":           case["signal"],
            "relation_group":   case["relation_group"],
            "source_ticker":    case["source_ticker"],
            "target_ticker":    case["target_ticker"],
            "analysis_mode":    case["analysis_mode"],
            "error":            last_error,
            "verdict":          "ERROR",
            "confidence":       0.0,
            "raw_judge_output": "",
        }

    # Parse judge JSON from output
    verdict_data: dict[str, Any] = {}
    json_match = re.search(r"\{[\s\S]*?\}", raw_output)
    if json_match:
        try:
            verdict_data = json.loads(json_match.group())
        except Exception:
            verdict_data = {"parse_error": raw_output[:500]}

    return {
        "case_id":                  case["case_id"],
        "signal":                   case["signal"],
        "relation_group":           case["relation_group"],
        "source_direction":         case["source_direction"],
        "analysis_mode":            case["analysis_mode"],
        "source_ticker":            case["source_ticker"],
        "source_quarter":           case["source_quarter"],
        "target_ticker":            case["target_ticker"],
        "target_quarter":           case["target_quarter"],
        "original_direction_match": case["direction_match"],
        **verdict_data,
        "raw_judge_output":         raw_output,
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    cases_path = Path(args.cases)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}\n"
                                f"Run sample_cases.py first.")

    # Load cases
    cases: list[dict] = []
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))


    # Apply shard offset BEFORE resume filtering so case_ids stay stable
    if args.offset > 0:
        cases = cases[args.offset:]
        print(f"Shard offset: skipping first {args.offset} cases, {len(cases)} remaining.")
    if args.max_cases:
        cases = cases[:args.max_cases]
        print(f"Shard limit: capped at {args.max_cases} cases.")

    already_done: set[int] = set()
    if args.resume:
        already_done = load_already_done(out_path, skip_errors=args.retry_errors)
        skipped = len(already_done)
        print(f"Resuming: {skipped} cases already done, skipping."
              + (" (ERROR cases will be retried)" if args.retry_errors else ""))
        cases = [c for c in cases if c["case_id"] not in already_done]

    print("=" * 60)
    print("CrewAI LLM Judge Workflow")
    print(f"  cases file : {cases_path}")
    print(f"  vLLM URL   : {args.vllm_url}")
    print(f"  model      : {args.model}")
    print(f"  to process : {len(cases)}")
    print(f"  output     : {out_path}")
    print("=" * 60)

    # Wait for vLLM server
    models_url = args.vllm_url.rstrip("/") + "/models"
    for attempt in range(60):
        try:
            resp = requests.get(models_url, timeout=5)
            if resp.status_code == 200:
                print("vLLM server is ready.")
                break
        except Exception:
            pass
        print(f"  waiting for vLLM... ({attempt + 1}/60)")
        time.sleep(10)

    # Build agents
    llm = build_llm(args)
    evidence_analyst, relationship_auditor, judge_agent, crew = build_crew(llm)

    # Process cases
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as out_f:
        for i, case in enumerate(cases):
            print(f"\n[{i+1}/{len(cases)}] case_id={case['case_id']} "
                  f"{case['source_ticker']}→{case['target_ticker']} "
                  f"signal={case['signal']} relation={case['relation_group']}")

            result = run_case(case, evidence_analyst, relationship_auditor,
                              judge_agent, crew, max_retries=args.max_retries)

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            verdict = result.get("verdict", "ERROR")
            conf    = result.get("confidence", "?")
            print(f"  → verdict={verdict}  confidence={conf}")
            print(f"  → reasoning: {result.get('judge_reasoning', '')}")
            print(f"  → supporting: {result.get('key_supporting_phrases', [])}")
            print(f"  → raw judge output:\n{result.get('raw_judge_output', '')[:1500]}")
            print("-" * 60)

    print(f"\nAll done. Results saved to {out_path}")
    print(f"Total processed: {len(cases)}")


if __name__ == "__main__":
    main()

