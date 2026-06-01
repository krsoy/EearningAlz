#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM agent extraction from Hugging Face RAG evidence Parquet.

Data source
-----------
Default HF repo:
    soysouce/earningALZ_SBERT_evidence

Default evidence file:
    rag_evidence_packages_full_gpu_direct.parquet

Why this version
----------------
The old extractor used INPUT_JSONL, for example:
    rag_chroma_output/rag_evidence_packages_2025Q2_Q3_gpu_direct.jsonl

This version reads the evidence packages directly from Hugging Face Parquet, keeps
package-level provenance, and writes shard outputs as Parquet.

Evidence-chain fields kept in every output row:
    evidence_package_row_id
    evidence_package_doc_id
    evidence_package_source
    evidence_package_file
    evidence_chunk_ids

Run locally
-----------
python extract_llm_agents_hf_evidence_vllm.py ^
  --hf-dataset soysouce/earningALZ_SBERT_evidence ^
  --hf-evidence-file rag_evidence_packages_full_gpu_direct.parquet ^
  --output-dir rag_chroma_output/llm_parquet_outputs_hf/2024Q1_2026Q2 ^
  --target-quarters 2024Q1,2024Q2,2024Q3,2024Q4,2025Q1,2025Q2,2025Q3,2025Q4,2026Q1,2026Q2 ^
  --num-shards 5 ^
  --shard-id 0 ^
  --run-name y2024q1_2026q2_s000_of005

SLURM normally passes these values through environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download


# ============================================================
# Schemas
# ============================================================

CONCEPT_KEYS = [
    "chip_supply",
    "semiconductor_supply",
    "raw_material_supply",
    "oil_energy_supply",
    "manufacturing_capacity",
    "production_capacity",
    "inventory_pressure",
    "logistics_shipping",
    "supplier_constraint",
    "customer_demand",
    "pricing_pressure",
    "capex_expansion",
    "data_center_capacity",
    "cloud_infrastructure",
    "labor_constraint",
    "geopolitical_risk",
]

OUTLOOK_KEYS = [
    "demand_outlook",
    "supply_outlook",
    "margin_outlook",
    "capex_outlook",
    "inventory_outlook",
    "pricing_outlook",
]

CONCEPT_COLUMNS = [
    "doc_id", "ticker", "current_company", "quarter", "publish_date",
    *CONCEPT_KEYS,
    "overall_supply_chain_relevance", "evidence_chunk_ids", "notes",
    "evidence_package_row_id", "evidence_package_doc_id", "evidence_package_source", "evidence_package_file",
]

RELATIONSHIP_COLUMNS = [
    "doc_id", "ticker", "current_company", "quarter", "publish_date",
    "relation_group", "entity", "entity_type", "relationship_type", "confidence", "evidence_chunk_ids",
    "evidence_package_row_id", "evidence_package_doc_id", "evidence_package_source", "evidence_package_file",
]

OUTLOOK_COLUMNS = [
    "doc_id", "ticker", "current_company", "quarter", "publish_date",
    "signal", "label", "evidence_chunk_ids", "notes",
    "evidence_package_row_id", "evidence_package_doc_id", "evidence_package_source", "evidence_package_file",
]

FAILED_COLUMNS = [
    "doc_id", "ticker", "current_company", "quarter", "agent_task", "error",
    "evidence_package_row_id", "evidence_package_doc_id", "evidence_package_source", "evidence_package_file",
]


# ============================================================
# Argument/env handling
# ============================================================

def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--hf-dataset", default=env("HF_DATASET", "soysouce/earningALZ_SBERT_evidence"))
    p.add_argument("--hf-evidence-file", default=env("HF_EVIDENCE_FILE", "rag_evidence_packages_full_gpu_direct.parquet"))
    p.add_argument("--hf-revision", default=env("HF_REVISION", "main"))

    p.add_argument("--output-dir", default=env("OUTPUT_DIR", "rag_chroma_output/llm_parquet_outputs_hf"))
    p.add_argument("--run-name", default=env("RUN_NAME", ""))

    p.add_argument("--vllm-url", default=env("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    p.add_argument("--model", default=env("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
    p.add_argument("--agent-tasks", default=env("AGENT_TASKS", "concepts,relationships,outlook"))

    p.add_argument("--target-quarters", default=env("TARGET_QUARTERS", ""))
    p.add_argument("--num-shards", type=int, default=int(env("NUM_SHARDS", "1")))
    p.add_argument("--shard-id", type=int, default=int(env("SHARD_ID", "0")))

    max_docs = env("MAX_DOCS", "")
    limit_docs = env("LIMIT_DOCS", "")
    p.add_argument("--max-docs", type=int, default=int(max_docs) if max_docs.strip() else None)
    p.add_argument("--start-offset", type=int, default=int(env("START_OFFSET", "0")))
    p.add_argument("--limit-docs", type=int, default=int(limit_docs) if limit_docs.strip() else None)

    p.add_argument("--max-workers", type=int, default=int(env("LLM_MAX_WORKERS", "6")))
    p.add_argument("--request-timeout", type=int, default=int(env("LLM_REQUEST_TIMEOUT", "240")))
    p.add_argument("--max-retries", type=int, default=int(env("LLM_MAX_RETRIES", "3")))
    p.add_argument("--temperature", type=float, default=float(env("LLM_TEMPERATURE", "0.0")))
    p.add_argument("--top-p", type=float, default=float(env("LLM_TOP_P", "1.0")))
    p.add_argument("--max-tokens", type=int, default=int(env("LLM_MAX_TOKENS", "1600")))

    p.add_argument("--write-csv-copy", action="store_true", default=env("WRITE_CSV_COPY", "0") == "1")
    return p.parse_args()


# ============================================================
# HF Parquet loading
# ============================================================

def maybe_json_load(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def load_hf_evidence_packages(args) -> list[dict]:
    local_path = hf_hub_download(
        repo_id=args.hf_dataset,
        filename=args.hf_evidence_file,
        repo_type="dataset",
        revision=args.hf_revision,
    )
    df = pd.read_parquet(local_path).reset_index(drop=True)

    packages = []
    for i, row in df.iterrows():
        obj = {}
        for k, v in row.to_dict().items():
            if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                obj[k] = ""
            else:
                obj[k] = maybe_json_load(v)

        # Stable provenance fields.
        obj["_evidence_package_row_id"] = int(obj.get("_jsonl_row_id", i) if str(obj.get("_jsonl_row_id", "")).strip() != "" else i)
        obj["_evidence_package_source"] = args.hf_dataset
        obj["_evidence_package_file"] = args.hf_evidence_file
        obj["_local_hf_cache_file"] = str(local_path)
        packages.append(obj)

    return packages


def count_quarters(packages: list[dict]) -> dict:
    counter = {}
    for pkg in packages:
        q = str(pkg.get("quarter", "")).strip()
        counter[q] = counter.get(q, 0) + 1
    return counter


def filter_target_quarters(packages: list[dict], target_quarters: list[str]) -> list[dict]:
    if not target_quarters:
        return packages
    target_set = set(target_quarters)
    return [pkg for pkg in packages if str(pkg.get("quarter", "")).strip() in target_set]


def select_shard(packages: list[dict], args) -> list[dict]:
    target_quarters = [x.strip() for x in args.target_quarters.split(",") if x.strip()]
    packages = filter_target_quarters(packages, target_quarters)

    if args.max_docs is not None:
        packages = packages[:args.max_docs]

    if args.num_shards <= 1:
        selected = packages
    else:
        selected = [pkg for i, pkg in enumerate(packages) if i % args.num_shards == args.shard_id]

    if args.start_offset > 0:
        selected = selected[args.start_offset:]

    if args.limit_docs is not None:
        selected = selected[:args.limit_docs]

    return selected


# ============================================================
# Prompt construction
# ============================================================

def evidence_chunks_to_text(pkg: dict, agent_task: str) -> str:
    evidence = pkg.get("retrieved_evidence", {})
    evidence = maybe_json_load(evidence)
    if not isinstance(evidence, dict):
        evidence = {}

    if agent_task == "concepts":
        group_order = [
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE"),
            ("relationship_chunks", "RELATIONSHIP EVIDENCE"),
            ("expectation_chunks", "EXPECTATION EVIDENCE"),
        ]
    elif agent_task == "relationships":
        group_order = [
            ("relationship_chunks", "RELATIONSHIP EVIDENCE"),
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE"),
        ]
    elif agent_task == "outlook":
        group_order = [
            ("expectation_chunks", "EXPECTATION / OUTLOOK EVIDENCE"),
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE"),
        ]
    else:
        group_order = [
            ("relationship_chunks", "RELATIONSHIP EVIDENCE"),
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE"),
            ("expectation_chunks", "EXPECTATION EVIDENCE"),
        ]

    sections = []
    for key, title in group_order:
        chunks = evidence.get(key, [])
        if isinstance(chunks, str):
            chunks = maybe_json_load(chunks)
        if not isinstance(chunks, list) or not chunks:
            continue

        lines = [f"\n## {title}"]
        for item in chunks:
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("chunk_id", "")
            score = item.get("hybrid_score", "")
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            lines.append(f"\n[chunk_id={chunk_id}, score={score}]\n{text}")

        sections.append("\n".join(lines))

    return "\n".join(sections)


def build_prompt(pkg: dict, agent_task: str) -> list[dict]:
    doc_id = str(pkg.get("doc_id", ""))
    ticker = str(pkg.get("ticker", ""))
    company = str(pkg.get("current_company", ""))
    quarter = str(pkg.get("quarter", ""))
    publish_date = str(pkg.get("publish_date", ""))
    title = str(pkg.get("title", ""))

    evidence_text = evidence_chunks_to_text(pkg, agent_task)

    system_msg = """
You are a financial information extraction agent.

Use only the provided earnings call evidence chunks.
Do not use outside knowledge.
Do not invent companies, relationships, or signals.
Return valid JSON only.
No markdown.
No explanation outside JSON.
""".strip()

    if agent_task == "concepts":
        user_msg = f"""
Metadata:
doc_id: {doc_id}
ticker: {ticker}
current_company: {company}
quarter: {quarter}
publish_date: {publish_date}
title: {title}

Evidence:
{evidence_text}

Task:
Extract binary supply-chain concept features.

Return strict JSON with this schema:

{{
  "doc_id": "{doc_id}",
  "ticker": "{ticker}",
  "current_company": "{company}",
  "quarter": "{quarter}",
  "publish_date": "{publish_date}",
  "chip_supply": 0,
  "semiconductor_supply": 0,
  "raw_material_supply": 0,
  "oil_energy_supply": 0,
  "manufacturing_capacity": 0,
  "production_capacity": 0,
  "inventory_pressure": 0,
  "logistics_shipping": 0,
  "supplier_constraint": 0,
  "customer_demand": 0,
  "pricing_pressure": 0,
  "capex_expansion": 0,
  "data_center_capacity": 0,
  "cloud_infrastructure": 0,
  "labor_constraint": 0,
  "geopolitical_risk": 0,
  "overall_supply_chain_relevance": "high|medium|low|none",
  "evidence_chunk_ids": [],
  "notes": ""
}}

Rules:
- All concept values must be 0 or 1.
- evidence_chunk_ids must only use chunk_id values shown above.
- If evidence is weak or absent, use 0.
- JSON only.
""".strip()

    elif agent_task == "relationships":
        user_msg = f"""
Metadata:
doc_id: {doc_id}
ticker: {ticker}
current_company: {company}
quarter: {quarter}
publish_date: {publish_date}
title: {title}

Evidence:
{evidence_text}

Task:
Extract company or entity relationships.

Return strict JSON with this schema:

{{
  "doc_id": "{doc_id}",
  "ticker": "{ticker}",
  "current_company": "{company}",
  "quarter": "{quarter}",
  "publish_date": "{publish_date}",
  "relationships": [
    {{
      "relation_group": "upstream|downstream|parent|subsidiary|related",
      "entity": "",
      "entity_type": "company|supplier_group|customer_group|industry_group|business_unit|unknown",
      "relationship_type": "supplier|vendor|component_provider|manufacturer|customer|buyer|OEM|distributor|parent|holding_company|subsidiary|business_unit|partner|competitor|acquirer|acquired_company|other",
      "confidence": "high|medium|low",
      "evidence_chunk_ids": []
    }}
  ]
}}

Rules:
- Extract only relationships supported by evidence.
- If no relationship is found, return "relationships": [].
- Do not invent entity names.
- Generic groups such as "cloud customers" are allowed only if explicitly mentioned.
- JSON only.
""".strip()

    elif agent_task == "outlook":
        user_msg = f"""
Metadata:
doc_id: {doc_id}
ticker: {ticker}
current_company: {company}
quarter: {quarter}
publish_date: {publish_date}
title: {title}

Evidence:
{evidence_text}

Task:
Extract forward-looking expectation signals.

Return strict JSON with this schema:

{{
  "doc_id": "{doc_id}",
  "ticker": "{ticker}",
  "current_company": "{company}",
  "quarter": "{quarter}",
  "publish_date": "{publish_date}",
  "outlook": [
    {{
      "signal": "demand_outlook|supply_outlook|margin_outlook|capex_outlook|inventory_outlook|pricing_outlook",
      "label": "positive|negative|mixed|neutral|increase|decrease|stable|improving|worsening|not_mentioned",
      "evidence_chunk_ids": [],
      "notes": ""
    }}
  ]
}}

Rules:
- Use not_mentioned if no evidence exists for a signal.
- evidence_chunk_ids must only use chunk_id values shown above.
- JSON only.
""".strip()
    else:
        raise ValueError(f"Unknown agent_task: {agent_task}")

    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


# ============================================================
# vLLM call
# ============================================================

def extract_json_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def call_vllm(pkg: dict, agent_task: str, args) -> dict:
    payload = {
        "model": args.model,
        "messages": build_prompt(pkg, agent_task),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(args.vllm_url, json=payload, timeout=args.request_timeout)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = extract_json_from_text(content)

            parsed["doc_id"] = str(pkg.get("doc_id", ""))
            parsed["ticker"] = str(pkg.get("ticker", ""))
            parsed["current_company"] = str(pkg.get("current_company", ""))
            parsed["quarter"] = str(pkg.get("quarter", ""))
            parsed["publish_date"] = str(pkg.get("publish_date", ""))
            parsed["_evidence_package_row_id"] = pkg.get("_evidence_package_row_id", "")
            parsed["_evidence_package_doc_id"] = str(pkg.get("doc_id", ""))
            parsed["_evidence_package_source"] = pkg.get("_evidence_package_source", "")
            parsed["_evidence_package_file"] = pkg.get("_evidence_package_file", "")
            return parsed
        except Exception as e:
            last_error = repr(e)
            time.sleep(2 * attempt)

    raise RuntimeError(last_error)


# ============================================================
# Convert model output to row tables
# ============================================================

def stringify_list(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return "|".join(str(i) for i in x)
    return str(x)


def add_provenance(row: dict, obj: dict) -> dict:
    row["evidence_package_row_id"] = obj.get("_evidence_package_row_id", "")
    row["evidence_package_doc_id"] = obj.get("_evidence_package_doc_id", obj.get("doc_id", ""))
    row["evidence_package_source"] = obj.get("_evidence_package_source", "")
    row["evidence_package_file"] = obj.get("_evidence_package_file", "")
    return row


def concept_result_to_rows(obj: dict) -> list[dict]:
    row = {
        "doc_id": obj.get("doc_id", ""),
        "ticker": obj.get("ticker", ""),
        "current_company": obj.get("current_company", ""),
        "quarter": obj.get("quarter", ""),
        "publish_date": obj.get("publish_date", ""),
        "overall_supply_chain_relevance": obj.get("overall_supply_chain_relevance", "none"),
        "evidence_chunk_ids": stringify_list(obj.get("evidence_chunk_ids", [])),
        "notes": obj.get("notes", ""),
    }
    for key in CONCEPT_KEYS:
        try:
            row[key] = int(obj.get(key, 0))
        except Exception:
            row[key] = 0
    return [add_provenance(row, obj)]


def relationship_result_to_rows(obj: dict) -> list[dict]:
    rels = obj.get("relationships", [])
    if not isinstance(rels, list):
        rels = []

    base = {
        "doc_id": obj.get("doc_id", ""),
        "ticker": obj.get("ticker", ""),
        "current_company": obj.get("current_company", ""),
        "quarter": obj.get("quarter", ""),
        "publish_date": obj.get("publish_date", ""),
    }

    rows = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        rows.append(add_provenance({
            **base,
            "relation_group": r.get("relation_group", ""),
            "entity": r.get("entity", ""),
            "entity_type": r.get("entity_type", ""),
            "relationship_type": r.get("relationship_type", ""),
            "confidence": r.get("confidence", ""),
            "evidence_chunk_ids": stringify_list(r.get("evidence_chunk_ids", [])),
        }, obj))

    if not rows:
        rows.append(add_provenance({
            **base,
            "relation_group": "none",
            "entity": "",
            "entity_type": "",
            "relationship_type": "",
            "confidence": "",
            "evidence_chunk_ids": "",
        }, obj))

    return rows


def outlook_result_to_rows(obj: dict) -> list[dict]:
    outlook = obj.get("outlook", [])
    if not isinstance(outlook, list):
        outlook = []

    base = {
        "doc_id": obj.get("doc_id", ""),
        "ticker": obj.get("ticker", ""),
        "current_company": obj.get("current_company", ""),
        "quarter": obj.get("quarter", ""),
        "publish_date": obj.get("publish_date", ""),
    }

    rows = []
    seen = set()
    for o in outlook:
        if not isinstance(o, dict):
            continue
        signal = str(o.get("signal", "")).strip()
        if not signal:
            continue
        seen.add(signal)
        rows.append(add_provenance({
            **base,
            "signal": signal,
            "label": o.get("label", "not_mentioned"),
            "evidence_chunk_ids": stringify_list(o.get("evidence_chunk_ids", [])),
            "notes": o.get("notes", ""),
        }, obj))

    for signal in OUTLOOK_KEYS:
        if signal not in seen:
            rows.append(add_provenance({
                **base,
                "signal": signal,
                "label": "not_mentioned",
                "evidence_chunk_ids": "",
                "notes": "",
            }, obj))

    return rows


def failed_to_row(pkg: dict, agent_task: str, error: str) -> dict:
    return {
        "doc_id": str(pkg.get("doc_id", "")),
        "ticker": str(pkg.get("ticker", "")),
        "current_company": str(pkg.get("current_company", "")),
        "quarter": str(pkg.get("quarter", "")),
        "agent_task": agent_task,
        "error": error,
        "evidence_package_row_id": pkg.get("_evidence_package_row_id", ""),
        "evidence_package_doc_id": str(pkg.get("doc_id", "")),
        "evidence_package_source": pkg.get("_evidence_package_source", ""),
        "evidence_package_file": pkg.get("_evidence_package_file", ""),
    }


def save_parquet_and_optional_csv(rows: list[dict], columns: list[str], path: Path, write_csv: bool):
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)

    for c in columns:
        if c not in df.columns:
            df[c] = ""
    df = df[columns]

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"SAVED {path} rows={len(df):,}")

    if write_csv:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"SAVED {csv_path} rows={len(df):,}")


def write_progress(path: Path, args, total, done, success, failed):
    obj = {
        "run_name": args.run_name,
        "hf_dataset": args.hf_dataset,
        "hf_evidence_file": args.hf_evidence_file,
        "hf_revision": args.hf_revision,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "target_quarters": [x.strip() for x in args.target_quarters.split(",") if x.strip()],
        "agent_tasks": [x.strip() for x in args.agent_tasks.split(",") if x.strip()],
        "total_items": total,
        "done_items": done,
        "success_items": success,
        "failed_items": failed,
        "output_dir": args.output_dir,
        "model": args.model,
        "vllm_url": args.vllm_url,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_done_keys(output_dir: Path, run_name: str) -> set[tuple[str, str]]:
    done = set()
    for agent, filename in [
        ("concepts", f"concepts_{run_name}.parquet"),
        ("relationships", f"relationships_{run_name}.parquet"),
        ("outlook", f"outlook_{run_name}.parquet"),
    ]:
        p = output_dir / filename
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["doc_id"])
            for doc_id in df["doc_id"].dropna().astype(str).unique():
                if doc_id:
                    done.add((agent, doc_id))
        except Exception:
            pass
    return done


def process_one(pkg: dict, agent_task: str, args):
    result = call_vllm(pkg, agent_task, args)
    return agent_task, result


def main():
    args = parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"Invalid shard_id={args.shard_id}, num_shards={args.num_shards}")

    agent_tasks = [x.strip() for x in args.agent_tasks.split(",") if x.strip()]
    for task in agent_tasks:
        if task not in {"concepts", "relationships", "outlook"}:
            raise ValueError(f"Unknown agent task: {task}")

    if not args.run_name:
        args.run_name = f"hf_shard{args.shard_id:03d}_of{args.num_shards:03d}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts_path = out_dir / f"concepts_{args.run_name}.parquet"
    relationships_path = out_dir / f"relationships_{args.run_name}.parquet"
    outlook_path = out_dir / f"outlook_{args.run_name}.parquet"
    failed_path = out_dir / f"failed_{args.run_name}.parquet"
    progress_path = out_dir / f"progress_{args.run_name}.json"

    print("=" * 90)
    print("LLM Agent Extraction from Hugging Face Evidence Parquet")
    print("HF dataset:", args.hf_dataset)
    print("HF evidence file:", args.hf_evidence_file)
    print("Output dir:", out_dir)
    print("Run name:", args.run_name)
    print("Model:", args.model)
    print("vLLM URL:", args.vllm_url)
    print("Agent tasks:", agent_tasks)
    print("Target quarters:", args.target_quarters)
    print("NUM_SHARDS:", args.num_shards)
    print("SHARD_ID:", args.shard_id)
    print("=" * 90)

    packages = load_hf_evidence_packages(args)
    print("Total input packages:", len(packages))
    print("Input quarter distribution:")
    for q, n in sorted(count_quarters(packages).items()):
        print(q, n)

    selected = select_shard(packages, args)
    print("Selected packages for this shard:", len(selected))
    print("Selected quarter distribution:")
    for q, n in sorted(count_quarters(selected).items()):
        print(q, n)

    done_keys = load_done_keys(out_dir, args.run_name)
    jobs = []
    for pkg in selected:
        doc_id = str(pkg.get("doc_id", ""))
        for task in agent_tasks:
            if (task, doc_id) not in done_keys:
                jobs.append((pkg, task))

    print("Already done agent-doc pairs:", len(done_keys))
    print("Todo agent-doc pairs:", len(jobs))

    concept_rows, relationship_rows, outlook_rows, failed_rows = [], [], [], []

    if not jobs:
        write_progress(progress_path, args, total=0, done=0, success=0, failed=0)
        print("Nothing to do.")
        return

    success = 0
    failed = 0
    write_progress(progress_path, args, total=len(jobs), done=0, success=0, failed=0)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_job = {
            executor.submit(process_one, pkg, task, args): (pkg, task)
            for pkg, task in jobs
        }

        for i, future in enumerate(tqdm(as_completed(future_to_job), total=len(future_to_job), desc="LLM agents"), start=1):
            pkg, task = future_to_job[future]
            try:
                agent_task, result = future.result()
                if agent_task == "concepts":
                    concept_rows.extend(concept_result_to_rows(result))
                elif agent_task == "relationships":
                    relationship_rows.extend(relationship_result_to_rows(result))
                elif agent_task == "outlook":
                    outlook_rows.extend(outlook_result_to_rows(result))
                success += 1
            except Exception as e:
                failed_rows.append(failed_to_row(pkg, task, repr(e)))
                failed += 1

            if i % 10 == 0 or i == len(jobs):
                write_progress(progress_path, args, total=len(jobs), done=i, success=success, failed=failed)

    save_parquet_and_optional_csv(concept_rows, CONCEPT_COLUMNS, concepts_path, args.write_csv_copy)
    save_parquet_and_optional_csv(relationship_rows, RELATIONSHIP_COLUMNS, relationships_path, args.write_csv_copy)
    save_parquet_and_optional_csv(outlook_rows, OUTLOOK_COLUMNS, outlook_path, args.write_csv_copy)
    save_parquet_and_optional_csv(failed_rows, FAILED_COLUMNS, failed_path, args.write_csv_copy)

    write_progress(progress_path, args, total=len(jobs), done=len(jobs), success=success, failed=failed)

    print("DONE.")
    print("Success:", success)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
