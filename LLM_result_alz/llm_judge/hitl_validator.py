
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-in-the-Loop (HITL) Validator for LLM Judge Results.

Presents each LLM judge verdict to a human reviewer in the terminal.
Human confirms, overrides, or flags each case.
Outputs a final statistics report on LLM extraction reliability.

Usage:
    python hitl_validator.py --results judge_results.jsonl
    python hitl_validator.py --results judge_results.jsonl --out hitl_validated.jsonl
    python hitl_validator.py --stats-only --validated hitl_validated.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================
# Arg parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results",   default="judge_results.jsonl",
                   help="LLM judge results JSONL from crewai_judge_workflow.py")
    p.add_argument("--out",       default="hitl_validated.jsonl",
                   help="Output path for human-validated results")
    p.add_argument("--stats-only", action="store_true",
                   help="Skip HITL review, just print stats on an existing validated file")
    p.add_argument("--validated", default="hitl_validated.jsonl",
                   help="Validated file to use with --stats-only")
    p.add_argument("--resume",    action="store_true",
                   help="Skip cases already present in --out")
    return p.parse_args()


# ============================================================
# Display helpers
# ============================================================

COLORS = {
    "VALID":           "\033[92m",   # green
    "PARTIALLY_VALID": "\033[93m",   # yellow
    "INVALID":         "\033[91m",   # red
    "ERROR":           "\033[95m",   # magenta
    "RESET":           "\033[0m",
}

def color(text: str, key: str) -> str:
    if sys.stdout.isatty():
        return COLORS.get(key, "") + str(text) + COLORS["RESET"]
    return str(text)


def print_case(result: dict, idx: int, total: int):
    sep = "=" * 70
    verdict = result.get("verdict", "ERROR")
    conf    = result.get("confidence", "?")

    print(f"\n{sep}")
    print(f"  Case {idx}/{total}  |  case_id={result.get('case_id')}")
    print(sep)
    print(f"  Source  : {result.get('source_ticker')} ({result.get('source_quarter')})")
    print(f"  Target  : {result.get('target_ticker')} ({result.get('target_quarter')})")
    print(f"  Signal  : {result.get('signal')}")
    print(f"  Relation: {result.get('relation_group')}")
    print(f"  Direction (claimed): {result.get('source_direction')}")
    print(f"  Mode    : {result.get('analysis_mode')}")
    print()
    print(f"  LLM Verdict    : {color(verdict, verdict)}   confidence={conf}")
    print(f"  Signal quality : {result.get('signal_evidence_quality', 'N/A')}")
    print(f"  Relation plaus.: {result.get('relation_plausibility', 'N/A')}")

    supporting = result.get("key_supporting_phrases", [])
    if supporting:
        print(f"\n  Supporting phrases:")
        for p in supporting[:3]:
            print(f"    + \"{p}\"")

    contradicting = result.get("key_contradicting_phrases", [])
    if contradicting:
        print(f"\n  Contradicting phrases:")
        for p in contradicting[:2]:
            print(f"    - \"{p}\"")

    reasoning = result.get("judge_reasoning", "")
    if reasoning:
        print(f"\n  Judge reasoning: {reasoning}")

    if result.get("error"):
        print(f"\n  ERROR: {result['error']}")

    print()


def prompt_human(result: dict) -> dict:
    """
    Present LLM verdict and ask human to confirm or override.
    Returns updated result dict with human judgment fields.
    """
    llm_verdict = result.get("verdict", "ERROR")

    print("  Your decision:")
    print("    [Enter]  = Agree with LLM verdict")
    print("    [v]      = VALID")
    print("    [p]      = PARTIALLY_VALID")
    print("    [i]      = INVALID")
    print("    [s]      = Skip (exclude from stats)")
    print("    [q]      = Quit and save progress")

    while True:
        try:
            raw = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted. Saving progress.")
            return {**result, "human_verdict": None, "human_skipped": False, "quit": True}

        if raw == "":
            human_verdict = llm_verdict
            break
        elif raw == "v":
            human_verdict = "VALID"
            break
        elif raw == "p":
            human_verdict = "PARTIALLY_VALID"
            break
        elif raw == "i":
            human_verdict = "INVALID"
            break
        elif raw == "s":
            print("  Skipped.")
            return {**result, "human_verdict": None, "human_skipped": True, "quit": False}
        elif raw == "q":
            return {**result, "human_verdict": None, "human_skipped": False, "quit": True}
        else:
            print("  Invalid input. Try again.")

    # Optional comment
    try:
        comment = input("  Comment (optional, Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        comment = ""

    agrees = (human_verdict == llm_verdict)
    print(f"  ✓ Recorded: {color(human_verdict, human_verdict)}"
          + ("  [same as LLM]" if agrees else f"  [OVERRIDE from {llm_verdict}]"))

    return {
        **result,
        "human_verdict":   human_verdict,
        "human_comment":   comment,
        "human_agreed":    agrees,
        "human_skipped":   False,
        "quit":            False,
    }


# ============================================================
# Statistics
# ============================================================

def compute_stats(records: list[dict]) -> dict:
    active = [r for r in records if not r.get("human_skipped") and r.get("human_verdict")]

    total = len(active)
    if total == 0:
        return {"total": 0, "note": "no valid records"}

    # Agreement rate
    agreed = sum(1 for r in active if r.get("human_agreed"))
    agreement_rate = agreed / total

    # Verdict distribution (LLM)
    llm_dist: dict[str, int] = defaultdict(int)
    for r in active:
        llm_dist[r.get("verdict", "UNKNOWN")] += 1

    # Verdict distribution (human)
    human_dist: dict[str, int] = defaultdict(int)
    for r in active:
        human_dist[r.get("human_verdict", "UNKNOWN")] += 1

    # LLM reliability: fraction human marked VALID or PARTIALLY_VALID
    human_valid = sum(1 for r in active
                      if r.get("human_verdict") in ("VALID", "PARTIALLY_VALID"))
    human_valid_rate = human_valid / total

    # By signal
    by_signal: dict[str, dict] = defaultdict(lambda: {"total": 0, "human_valid": 0, "llm_agreed": 0})
    for r in active:
        sig = r.get("signal", "unknown")
        by_signal[sig]["total"] += 1
        if r.get("human_verdict") in ("VALID", "PARTIALLY_VALID"):
            by_signal[sig]["human_valid"] += 1
        if r.get("human_agreed"):
            by_signal[sig]["llm_agreed"] += 1

    # By relation_group
    by_relation: dict[str, dict] = defaultdict(lambda: {"total": 0, "human_valid": 0, "llm_agreed": 0})
    for r in active:
        rel = r.get("relation_group", "unknown")
        by_relation[rel]["total"] += 1
        if r.get("human_verdict") in ("VALID", "PARTIALLY_VALID"):
            by_relation[rel]["human_valid"] += 1
        if r.get("human_agreed"):
            by_relation[rel]["llm_agreed"] += 1

    # Override analysis
    overrides = [r for r in active if not r.get("human_agreed")]
    override_directions: dict[str, int] = defaultdict(int)
    for r in overrides:
        key = f"{r.get('verdict')}→{r.get('human_verdict')}"
        override_directions[key] += 1

    return {
        "total_reviewed":            total,
        "llm_human_agreement_rate":  round(agreement_rate, 3),
        "human_valid_or_partial_rate": round(human_valid_rate, 3),
        "llm_verdict_distribution":  dict(llm_dist),
        "human_verdict_distribution":dict(human_dist),
        "override_count":            len(overrides),
        "override_directions":       dict(override_directions),
        "by_signal": {
            sig: {
                "total":       v["total"],
                "human_valid_rate": round(v["human_valid"] / v["total"], 3) if v["total"] else 0,
                "llm_agree_rate":   round(v["llm_agreed"] / v["total"], 3) if v["total"] else 0,
            }
            for sig, v in by_signal.items()
        },
        "by_relation_group": {
            rel: {
                "total":       v["total"],
                "human_valid_rate": round(v["human_valid"] / v["total"], 3) if v["total"] else 0,
                "llm_agree_rate":   round(v["llm_agreed"] / v["total"], 3) if v["total"] else 0,
            }
            for rel, v in by_relation.items()
        },
    }


def print_stats(stats: dict):
    print("\n" + "=" * 70)
    print("  LLM EXTRACTION RELIABILITY REPORT")
    print("=" * 70)
    print(f"  Total reviewed       : {stats.get('total_reviewed')}")
    print(f"  LLM↔Human agreement : {stats.get('llm_human_agreement_rate', 0)*100:.1f}%")
    print(f"  Human valid/partial  : {stats.get('human_valid_or_partial_rate', 0)*100:.1f}%")
    print()

    print("  LLM Verdict Distribution:")
    for k, v in sorted(stats.get("llm_verdict_distribution", {}).items()):
        bar = "█" * v
        print(f"    {k:20s}: {v:4d}  {bar}")
    print()

    print("  Human Verdict Distribution:")
    for k, v in sorted(stats.get("human_verdict_distribution", {}).items()):
        bar = "█" * v
        print(f"    {k:20s}: {v:4d}  {bar}")
    print()

    ov = stats.get("override_count", 0)
    total = stats.get("total_reviewed", 1)
    print(f"  Human overrides      : {ov} ({ov/total*100:.1f}%)")
    for k, v in sorted(stats.get("override_directions", {}).items()):
        print(f"    {k}: {v}")
    print()

    print("  Reliability by Signal:")
    header = f"    {'Signal':<28} {'N':>4}  {'HumanValid%':>11}  {'LLM-agree%':>10}"
    print(header)
    for sig, row in sorted(stats.get("by_signal", {}).items()):
        print(f"    {sig:<28} {row['total']:>4}  "
              f"{row['human_valid_rate']*100:>10.1f}%  "
              f"{row['llm_agree_rate']*100:>9.1f}%")
    print()

    print("  Reliability by Relation Group:")
    header = f"    {'Relation':<24} {'N':>4}  {'HumanValid%':>11}  {'LLM-agree%':>10}"
    print(header)
    for rel, row in sorted(stats.get("by_relation_group", {}).items()):
        print(f"    {rel:<24} {row['total']:>4}  "
              f"{row['human_valid_rate']*100:>10.1f}%  "
              f"{row['llm_agree_rate']*100:>9.1f}%")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def main():
    args = parse_args()

    # Stats-only mode
    if args.stats_only:
        path = Path(args.validated)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        records = load_jsonl(path)
        stats = compute_stats(records)
        print_stats(stats)
        stats_path = path.with_name(path.stem + "_stats.json")
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"\nStats saved to {stats_path}")
        return

    results_path = Path(args.results)
    out_path     = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run crewai_judge_workflow.py first.")
        sys.exit(1)

    results = load_jsonl(results_path)

    # Resume support
    already_done: set[int] = set()
    existing: list[dict] = []
    if args.resume and out_path.exists():
        existing = load_jsonl(out_path)
        already_done = {r["case_id"] for r in existing
                        if r.get("human_verdict") is not None or r.get("human_skipped")}
        print(f"Resuming: {len(already_done)} cases already validated.")

    to_review = [r for r in results
                 if r.get("case_id") not in already_done
                 and r.get("verdict") != "ERROR"]

    print("=" * 70)
    print("  HITL Validator")
    print(f"  Total cases to review: {len(to_review)}")
    print("=" * 70)

    validated = list(existing)

    with out_path.open("a" if args.resume else "w", encoding="utf-8") as out_f:
        for idx, result in enumerate(to_review, start=1):
            print_case(result, idx, len(to_review))
            updated = prompt_human(result)

            if not updated.get("human_skipped") and updated.get("human_verdict"):
                validated.append(updated)

            out_line = {k: v for k, v in updated.items() if k != "quit"}
            out_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            out_f.flush()

            if updated.get("quit"):
                print("\nSaving and exiting.")
                break

    # Compute and print final stats
    stats = compute_stats(validated)
    print_stats(stats)

    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nValidated results : {out_path}")
    print(f"Statistics saved  : {stats_path}")


if __name__ == "__main__":
    main()

