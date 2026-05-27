import os
import re
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


# ============================================================
# ENV CONFIG
# ============================================================

INPUT_JSONL = Path(os.environ.get(
    "INPUT_JSONL",
    "rag_chroma_output/rag_evidence_packages_2025Q2_Q3_gpu_direct.jsonl"
))

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "rag_chroma_output/llm_csv_outputs_2025Q2_Q3"
))

VLLM_URL = os.environ.get(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions"
)

MODEL_NAME = os.environ.get(
    "LLM_MODEL",
    "Qwen/Qwen2.5-14B-Instruct"
)

AGENT_TASKS = [
    x.strip()
    for x in os.environ.get("AGENT_TASKS", "concepts,relationships,outlook").split(",")
    if x.strip()
]

# Controlled from bash/slurm
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "6"))
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))

# Only extract selected quarters
TARGET_QUARTERS = [
    x.strip()
    for x in os.environ.get("TARGET_QUARTERS", "2025Q2,2025Q3").split(",")
    if x.strip()
]

# Optional testing controls
MAX_DOCS = os.environ.get("MAX_DOCS", "")
MAX_DOCS = int(MAX_DOCS) if MAX_DOCS.strip() else None

START_OFFSET = int(os.environ.get("START_OFFSET", "0"))

LIMIT_DOCS = os.environ.get("LIMIT_DOCS", "")
LIMIT_DOCS = int(LIMIT_DOCS) if LIMIT_DOCS.strip() else None

# LLM request controls
MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "6"))
REQUEST_TIMEOUT = int(os.environ.get("LLM_REQUEST_TIMEOUT", "240"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))

TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("LLM_TOP_P", "1.0"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1600"))

RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"q2q3_shard{SHARD_ID:03d}_of{NUM_SHARDS:03d}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONCEPTS_CSV = OUTPUT_DIR / f"concepts_{RUN_NAME}.csv"
RELATIONSHIPS_CSV = OUTPUT_DIR / f"relationships_{RUN_NAME}.csv"
OUTLOOK_CSV = OUTPUT_DIR / f"outlook_{RUN_NAME}.csv"
FAILED_CSV = OUTPUT_DIR / f"failed_{RUN_NAME}.csv"
PROGRESS_JSON = OUTPUT_DIR / f"progress_{RUN_NAME}.json"


# ============================================================
# CSV SCHEMA
# ============================================================

CONCEPT_COLUMNS = [
    "doc_id",
    "ticker",
    "current_company",
    "quarter",
    "publish_date",
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
    "overall_supply_chain_relevance",
    "evidence_chunk_ids",
    "notes"
]

RELATIONSHIP_COLUMNS = [
    "doc_id",
    "ticker",
    "current_company",
    "quarter",
    "publish_date",
    "relation_group",
    "entity",
    "entity_type",
    "relationship_type",
    "confidence",
    "evidence_chunk_ids"
]

OUTLOOK_COLUMNS = [
    "doc_id",
    "ticker",
    "current_company",
    "quarter",
    "publish_date",
    "signal",
    "label",
    "evidence_chunk_ids",
    "notes"
]

FAILED_COLUMNS = [
    "doc_id",
    "ticker",
    "current_company",
    "quarter",
    "agent_task",
    "error"
]

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
    "geopolitical_risk"
]

OUTLOOK_KEYS = [
    "demand_outlook",
    "supply_outlook",
    "margin_outlook",
    "capex_outlook",
    "inventory_outlook",
    "pricing_outlook"
]


# ============================================================
# FILE HELPERS
# ============================================================

def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def ensure_csv_header(path: Path, columns: list[str]):
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv(path: Path, columns: list[str], rows: list[dict]):
    ensure_csv_header(path, columns)

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        for row in rows:
            writer.writerow(row)


def load_done_keys():
    """
    Used for resume.
    If a CSV already contains a doc_id for an agent task,
    that agent-doc pair will be skipped.
    """

    done = set()

    for path, agent in [
        (CONCEPTS_CSV, "concepts"),
        (RELATIONSHIPS_CSV, "relationships"),
        (OUTLOOK_CSV, "outlook"),
    ]:
        if not path.exists() or path.stat().st_size == 0:
            continue

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = str(row.get("doc_id", "")).strip()
                if doc_id:
                    done.add((agent, doc_id))

    return done


def write_progress(total, done, success, failed):
    obj = {
        "run_name": RUN_NAME,
        "num_shards": NUM_SHARDS,
        "shard_id": SHARD_ID,
        "target_quarters": TARGET_QUARTERS,
        "agent_tasks": AGENT_TASKS,
        "total_items": total,
        "done_items": done,
        "success_items": success,
        "failed_items": failed,
        "input_jsonl": str(INPUT_JSONL),
        "output_dir": str(OUTPUT_DIR),
        "model": MODEL_NAME,
        "vllm_url": VLLM_URL,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with PROGRESS_JSON.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stringify_list(x):
    if x is None:
        return ""

    if isinstance(x, list):
        return "|".join(str(i) for i in x)

    return str(x)


# ============================================================
# FILTERING AND SHARDING
# ============================================================

def filter_target_quarters(packages: list[dict]) -> list[dict]:
    if not TARGET_QUARTERS:
        return packages

    target_set = set(TARGET_QUARTERS)

    filtered = [
        pkg for pkg in packages
        if str(pkg.get("quarter", "")).strip() in target_set
    ]

    return filtered


def select_shard(packages: list[dict]) -> list[dict]:
    # 1. keep only 2025Q2 / 2025Q3
    packages = filter_target_quarters(packages)

    # 2. optional debugging cap
    if MAX_DOCS is not None:
        packages = packages[:MAX_DOCS]

    # 3. shard by index
    if NUM_SHARDS <= 1:
        selected = packages
    else:
        selected = [
            pkg for i, pkg in enumerate(packages)
            if i % NUM_SHARDS == SHARD_ID
        ]

    # 4. optional offset / limit inside this shard
    if START_OFFSET > 0:
        selected = selected[START_OFFSET:]

    if LIMIT_DOCS is not None:
        selected = selected[:LIMIT_DOCS]

    return selected


def count_quarters(packages: list[dict]) -> dict:
    counter = {}

    for pkg in packages:
        q = str(pkg.get("quarter", "")).strip()
        counter[q] = counter.get(q, 0) + 1

    return counter


# ============================================================
# PROMPT HELPERS
# ============================================================

def evidence_chunks_to_text(pkg: dict, agent_task: str) -> str:
    evidence = pkg.get("retrieved_evidence", {})

    if agent_task == "concepts":
        group_order = [
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE"),
            ("relationship_chunks", "RELATIONSHIP EVIDENCE"),
            ("expectation_chunks", "EXPECTATION EVIDENCE")
        ]
    elif agent_task == "relationships":
        group_order = [
            ("relationship_chunks", "RELATIONSHIP EVIDENCE"),
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE")
        ]
    elif agent_task == "outlook":
        group_order = [
            ("expectation_chunks", "EXPECTATION / OUTLOOK EVIDENCE"),
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE")
        ]
    else:
        group_order = [
            ("relationship_chunks", "RELATIONSHIP EVIDENCE"),
            ("supply_chain_chunks", "SUPPLY CHAIN EVIDENCE"),
            ("expectation_chunks", "EXPECTATION EVIDENCE")
        ]

    sections = []

    for key, title in group_order:
        chunks = evidence.get(key, [])
        if not chunks:
            continue

        lines = [f"\n## {title}"]

        for item in chunks:
            chunk_id = item.get("chunk_id", "")
            score = item.get("hybrid_score", "")
            text = str(item.get("text", ""))
            text = re.sub(r"\s+", " ", text).strip()

            lines.append(
                f"\n[chunk_id={chunk_id}, score={score}]\n{text}"
            )

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

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


# ============================================================
# LLM CALL
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


def call_vllm(pkg: dict, agent_task: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": build_prompt(pkg, agent_task),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                VLLM_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            parsed = extract_json_from_text(content)

            # Keep original metadata stable
            parsed["doc_id"] = str(pkg.get("doc_id", ""))
            parsed["ticker"] = str(pkg.get("ticker", ""))
            parsed["current_company"] = str(pkg.get("current_company", ""))
            parsed["quarter"] = str(pkg.get("quarter", ""))
            parsed["publish_date"] = str(pkg.get("publish_date", ""))

            return parsed

        except Exception as e:
            last_error = repr(e)
            time.sleep(2 * attempt)

    raise RuntimeError(last_error)


# ============================================================
# PARSE MODEL OUTPUT TO CSV ROWS
# ============================================================

def concept_result_to_rows(obj: dict) -> list[dict]:
    row = {
        "doc_id": obj.get("doc_id", ""),
        "ticker": obj.get("ticker", ""),
        "current_company": obj.get("current_company", ""),
        "quarter": obj.get("quarter", ""),
        "publish_date": obj.get("publish_date", ""),
        "overall_supply_chain_relevance": obj.get("overall_supply_chain_relevance", "none"),
        "evidence_chunk_ids": stringify_list(obj.get("evidence_chunk_ids", [])),
        "notes": obj.get("notes", "")
    }

    for key in CONCEPT_KEYS:
        try:
            row[key] = int(obj.get(key, 0))
        except Exception:
            row[key] = 0

    return [row]


def relationship_result_to_rows(obj: dict) -> list[dict]:
    rows = []

    rels = obj.get("relationships", [])
    if not isinstance(rels, list):
        rels = []

    base = {
        "doc_id": obj.get("doc_id", ""),
        "ticker": obj.get("ticker", ""),
        "current_company": obj.get("current_company", ""),
        "quarter": obj.get("quarter", ""),
        "publish_date": obj.get("publish_date", "")
    }

    for r in rels:
        if not isinstance(r, dict):
            continue

        rows.append({
            **base,
            "relation_group": r.get("relation_group", ""),
            "entity": r.get("entity", ""),
            "entity_type": r.get("entity_type", ""),
            "relationship_type": r.get("relationship_type", ""),
            "confidence": r.get("confidence", ""),
            "evidence_chunk_ids": stringify_list(r.get("evidence_chunk_ids", []))
        })

    if not rows:
        rows.append({
            **base,
            "relation_group": "none",
            "entity": "",
            "entity_type": "",
            "relationship_type": "",
            "confidence": "",
            "evidence_chunk_ids": ""
        })

    return rows


def outlook_result_to_rows(obj: dict) -> list[dict]:
    rows = []

    outlook = obj.get("outlook", [])
    if not isinstance(outlook, list):
        outlook = []

    base = {
        "doc_id": obj.get("doc_id", ""),
        "ticker": obj.get("ticker", ""),
        "current_company": obj.get("current_company", ""),
        "quarter": obj.get("quarter", ""),
        "publish_date": obj.get("publish_date", "")
    }

    seen = set()

    for o in outlook:
        if not isinstance(o, dict):
            continue

        signal = str(o.get("signal", "")).strip()
        if not signal:
            continue

        seen.add(signal)

        rows.append({
            **base,
            "signal": signal,
            "label": o.get("label", "not_mentioned"),
            "evidence_chunk_ids": stringify_list(o.get("evidence_chunk_ids", [])),
            "notes": o.get("notes", "")
        })

    # Ensure every doc has every outlook signal
    for signal in OUTLOOK_KEYS:
        if signal not in seen:
            rows.append({
                **base,
                "signal": signal,
                "label": "not_mentioned",
                "evidence_chunk_ids": "",
                "notes": ""
            })

    return rows


def write_agent_result(agent_task: str, result: dict):
    if agent_task == "concepts":
        rows = concept_result_to_rows(result)
        append_csv(CONCEPTS_CSV, CONCEPT_COLUMNS, rows)

    elif agent_task == "relationships":
        rows = relationship_result_to_rows(result)
        append_csv(RELATIONSHIPS_CSV, RELATIONSHIP_COLUMNS, rows)

    elif agent_task == "outlook":
        rows = outlook_result_to_rows(result)
        append_csv(OUTLOOK_CSV, OUTLOOK_COLUMNS, rows)

    else:
        raise ValueError(agent_task)


def write_failed(pkg: dict, agent_task: str, error: str):
    row = {
        "doc_id": str(pkg.get("doc_id", "")),
        "ticker": str(pkg.get("ticker", "")),
        "current_company": str(pkg.get("current_company", "")),
        "quarter": str(pkg.get("quarter", "")),
        "agent_task": agent_task,
        "error": error
    }

    append_csv(FAILED_CSV, FAILED_COLUMNS, [row])


# ============================================================
# MAIN
# ============================================================

def process_one(pkg: dict, agent_task: str):
    result = call_vllm(pkg, agent_task)
    return agent_task, result


def main():
    if SHARD_ID < 0 or SHARD_ID >= NUM_SHARDS:
        raise ValueError(f"Invalid SHARD_ID={SHARD_ID}, NUM_SHARDS={NUM_SHARDS}")

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL}")

    for task in AGENT_TASKS:
        if task not in {"concepts", "relationships", "outlook"}:
            raise ValueError(f"Unknown agent task: {task}")

    print("============================================================")
    print("LLM Agent CSV Extraction")
    print("============================================================")
    print("Input:", INPUT_JSONL)
    print("Output dir:", OUTPUT_DIR)
    print("Run name:", RUN_NAME)
    print("Model:", MODEL_NAME)
    print("vLLM URL:", VLLM_URL)
    print("Agent tasks:", AGENT_TASKS)
    print("Target quarters:", TARGET_QUARTERS)
    print("NUM_SHARDS:", NUM_SHARDS)
    print("SHARD_ID:", SHARD_ID)
    print("MAX_DOCS:", MAX_DOCS)
    print("START_OFFSET:", START_OFFSET)
    print("LIMIT_DOCS:", LIMIT_DOCS)
    print("MAX_WORKERS:", MAX_WORKERS)
    print("============================================================")

    packages = read_jsonl(INPUT_JSONL)

    print("Total input packages:", len(packages))

    print("\nInput quarter distribution:")
    for q, n in sorted(count_quarters(packages).items()):
        print(q, n)

    filtered = filter_target_quarters(packages)

    print("\nAfter target quarter filter:", len(filtered))
    print("Filtered quarter distribution:")
    for q, n in sorted(count_quarters(filtered).items()):
        print(q, n)

    selected = select_shard(packages)

    print("\nSelected packages for this shard:", len(selected))
    print("Selected quarter distribution:")
    for q, n in sorted(count_quarters(selected).items()):
        print(q, n)

    done_keys = load_done_keys()

    jobs = []

    for pkg in selected:
        doc_id = str(pkg.get("doc_id", ""))
        for task in AGENT_TASKS:
            if (task, doc_id) not in done_keys:
                jobs.append((pkg, task))

    print("\nAlready done agent-doc pairs:", len(done_keys))
    print("Todo agent-doc pairs:", len(jobs))

    if not jobs:
        print("Nothing to do.")
        write_progress(total=0, done=0, success=0, failed=0)
        return

    success = 0
    failed = 0

    write_progress(
        total=len(jobs),
        done=0,
        success=success,
        failed=failed
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {
            executor.submit(process_one, pkg, task): (pkg, task)
            for pkg, task in jobs
        }

        for i, future in enumerate(
            tqdm(as_completed(future_to_job), total=len(future_to_job), desc="LLM agents"),
            start=1
        ):
            pkg, task = future_to_job[future]

            try:
                agent_task, result = future.result()
                write_agent_result(agent_task, result)
                success += 1

            except Exception as e:
                write_failed(pkg, task, repr(e))
                failed += 1

            if i % 10 == 0 or i == len(jobs):
                write_progress(
                    total=len(jobs),
                    done=i,
                    success=success,
                    failed=failed
                )

    write_progress(
        total=len(jobs),
        done=len(jobs),
        success=success,
        failed=failed
    )

    print("\nDONE.")
    print("Success:", success)
    print("Failed:", failed)
    print("Concepts CSV:", CONCEPTS_CSV.resolve())
    print("Relationships CSV:", RELATIONSHIPS_CSV.resolve())
    print("Outlook CSV:", OUTLOOK_CSV.resolve())
    print("Failed CSV:", FAILED_CSV.resolve())
    print("Progress JSON:", PROGRESS_JSON.resolve())


if __name__ == "__main__":
    main()