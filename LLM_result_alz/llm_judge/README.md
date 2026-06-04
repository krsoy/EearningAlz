# LLM Judge Workflow — README

## 整体流程

```
sample_cases.py          ← 从 HF 采样 10例/组
       ↓
crewai_judge_workflow.py ← CrewAI 3-Agent LLM-as-Judge (vLLM)
       ↓
judge_results.jsonl
       ↓
hitl_validator.py        ← 本地人工复核 (HITL)
       ↓
hitl_validated.jsonl + _stats.json
```

---

## 三个 Agent 的分工

| Agent | 职责 |
|---|---|
| **EvidenceAnalyst** | 读 source transcript chunks，解释 LLM 为何提取出该 signal/direction |
| **RelationshipAuditor** | 跨读 source + target chunks，判断 upstream/downstream 关系是否文本可支撑 |
| **JudgeAgent** | 综合两者分析，输出结构化 verdict JSON |

判断结果 verdict 三档：
- `VALID` — 文本明确支撑
- `PARTIALLY_VALID` — 有一定依据但不充分
- `INVALID` — 文本不支撑该关系

---

## 本地运行步骤

```bash
# 0. 安装依赖
cd E:\Projects\EearningAlz
.venv\Scripts\python.exe -m pip install crewai
# 或
uv add crewai

# 1. 采样案例（全量，10例/组）
cd LLM_result_alz\llm_judge
..\..\\.venv\Scripts\python.exe sample_cases.py --n-per-group 10 --out judge_cases.jsonl

# 1b. 只看 AAPL→CARR 这条关系
..\..\\.venv\Scripts\python.exe sample_cases.py ^
  --filter-source AAPL --filter-target CARR ^
  --filter-signal supply_outlook --filter-direction negative ^
  --n-per-group 10 --out judge_cases_aapl_carr.jsonl

# 2. 启动本地 vLLM (需要 GPU)
python -m vllm.entrypoints.openai.api_server ^
  --model Qwen/Qwen2.5-14B-Instruct ^
  --port 8000 --tensor-parallel-size 1

# 3. 运行 CrewAI 判断工作流
..\..\\.venv\Scripts\python.exe crewai_judge_workflow.py ^
  --cases judge_cases.jsonl ^
  --out judge_results.jsonl ^
  --vllm-url http://127.0.0.1:8000/v1 ^
  --resume

# 4. 本地 HITL 人工复核
..\..\\.venv\Scripts\python.exe hitl_validator.py ^
  --results judge_results.jsonl ^
  --out hitl_validated.jsonl

# 4b. 只看统计，不复核
..\..\\.venv\Scripts\python.exe hitl_validator.py ^
  --stats-only --validated hitl_validated.jsonl
```

---

## 云端 SLURM 运行

```bash
cd ~/sem2

# 全量
sbatch LLM_result_alz/llm_judge/run_judge.slurm

# 只看 AAPL→CARR
JUDGE_FILTER_SOURCE=AAPL JUDGE_FILTER_TARGET=CARR \
sbatch LLM_result_alz/llm_judge/run_judge.slurm
```

SLURM 跑完后，`judge_results.jsonl` 会在 `LLM_result_alz/llm_judge/` 下，  
然后在**本地**运行 `hitl_validator.py` 做人工复核。

---

## 输出统计指标

`_stats.json` 包含：

```json
{
  "total_reviewed": 80,
  "llm_human_agreement_rate": 0.74,
  "human_valid_or_partial_rate": 0.68,
  "by_signal": {
    "supply_outlook": {"total": 10, "human_valid_rate": 0.70, "llm_agree_rate": 0.80},
    ...
  },
  "by_relation_group": {
    "upstream": {"total": 15, "human_valid_rate": 0.65, "llm_agree_rate": 0.73},
    ...
  }
}
```

这些数字直接回答："这个 LLM pipeline 提取的关系，有多少比例是真实可靠的？"

