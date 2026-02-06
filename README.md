
# Production-Grade LLM Observability & Drift Detection Pipeline

This project implements an **end-to-end, production-grade LLM observability pipeline** that begins with **carefully curated evaluation data** and ends with **behavioral drift detection** under real-world conditions.

Unlike typical demo pipelines that start directly from prompts, this system **intentionally begins with data curation** to mirror how real evaluation pipelines are built in industry.

---

## How the Story Begins - Evaluation Data Matters

Before any model evaluation, **the quality of prompts must be trusted**.

Production LLM failures are often misdiagnosed because:
- evaluation prompts are noisy,
- questions are context-dependent,
- or datasets are not frozen reproducibly.

This project therefore starts with **Stage 1: controlled prompt dataset construction**, using two well-known QA datasets:
- **SQuAD v1.1**
- **Natural Questions (NQ)**

Both datasets are **filtered, cleaned, categorized, sampled, and frozen** before any LLM inference occurs.

This ensures that **any detected drift is due to model behavior — not data noise**.

---

## Environment Configuration (.env)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Stage 1 — Prompt Dataset Construction (FOUNDATION)

### Data Sources
- **SQuAD v1.1** (Stanford Question Answering Dataset)
- **Natural Questions** (Google NQ)

### Cleaning & Filtering Principles
Only high-quality, standalone questions are retained:

- Length-constrained (avoids trivial or verbose prompts)
- Context-independent (no “according to the passage”)
- Proper question form
- Deduplicated aggressively

### Categorization
Questions are categorized using lightweight, rule-based logic:
- `factual` (who / what / when / where)
- `explanatory` (why / how)
- `ambiguous` (other well-formed questions)

### Sampling Strategy
Each dataset is sampled deterministically:

| Dataset | Factual | Explanatory | Ambiguous | Total |
|------|--------|-------------|-----------|-------|
| SQuAD   |   60    |      60     |     30    | 150 |
| NQ      |  100    |      50     |     —     | 150 |

### Freezing the Dataset
Both datasets are merged and **frozen** into a single canonical file:

```
data/prompts.csv
```

This file is never modified downstream.

> From this point onward, **all drift is attributed to model behavior**, not data instability.

---

## Problem Statement

In production, LLM failures rarely look like obvious accuracy drops.

Instead, models drift subtly due to:
- instruction or prompt changes
- model updates
- decoding behavior
- latency and infrastructure shifts

Traditional evaluation fails because:
- Accuracy ≠ behavioral stability
- Benchmarks ≠ live traffic
- Offline tests ≠ long-running systems

This project addresses that gap.

---

## End-to-End Pipeline Overview

```
Stage 1 — Prompt Dataset Construction (SQuAD + NQ)
          ↓
data/prompts.csv (FROZEN)
          ↓
Stage 2 — LLM Inference (controlled, run-scoped)
          ↓
Stage 3 — Feature Extraction (deterministic)
          ↓
Stage 4 — Quality & Reliability Scoring (config-driven)
          ↓
Stage 5 — Baseline Drift Detection
          ↓
Stage 6 — Time-Window Drift Detection
```

---

## Pipeline Stages

### **Stage 2 - LLM Inference**
- Controlled inference across models, temperatures, and instruction variants
- Resume-safe JSONL logging
- Per-run artifact isolation
- Automatic JSONL → CSV conversion

### **Stage 3 - Feature Extraction**
- Deterministic textual and behavioral features
- No LLM-based judging
- Full lineage preserved (`run_id`, timestamps)

### **Stage 4 - Quality & Reliability Scoring**
- Config-driven weighted scoring (`mlops/config.yaml`)
- Measures:
  - Instructional stability
  - Verbosity balance
  - Confidence signals
  - Formatting consistency
  - Latency behavior

### **Stage 5 - Baseline Drift Detection**
- Baseline vs current comparison
- Instruction-aware (concise vs detailed prompts)
- Distribution drift (KS test)
- Mean-shift drift detection

### **Stage 6 - Time-Window Drift Detection**
- Earlier vs recent window comparison
- Detects gradual behavioral degradation
- Production-correct temporal logic

---

## Project Structure

```
.
├── data/
│   └── prompts.csv        # Frozen evaluation dataset
│
├── artifacts/
│   ├── runs/              # Full historical runs
│   └── latest/            # Latest production pointers
│
├── mlops/
│   ├── config.yaml        # Scoring weights & thresholds
│   └── policies.yaml      # Alerting rules
│
├── utilities/
│   └── run_utils.py
│
├── stages/
│   ├── stage_2_inference.py
│   ├── stage_3_features.py
│   ├── stage_4_scoring.py
│   ├── stage_5_drift_baseline.py
│   └── stage_6_drift_timewindow.py
│
├── pipeline.py
├── README.md
└── requirements.txt
```

---

## System Architecture

![LLM Drift Detection Pipeline](/assets/Architectural-Diagram.png)

## How to Run

```bash
python pipeline.py
```

This executes the full pipeline with:
- one shared `run_id`
- strict stage ordering
- failure safety
- full artifact lineage

---

## Artifacts Produced

Each pipeline run generates:

```
artifacts/runs/run_YYYYMMDD_HHMMSS/
├── raw_llm_outputs.jsonl
├── raw_llm_outputs.csv
├── llm_features.csv
├── llm_scores.csv
├── llm_drift_report.csv
└── llm_drift_timewindow.csv
```

`artifacts/latest/` always points to the most recent run.

---

## Why This Is Production-Grade

- Evaluation data is curated and frozen
- Deterministic, reproducible scoring
- No LLM-as-judge dependency
- Resume-safe, append-only logging
- Config-driven behavior
- Run-level lineage & auditability
- Baseline + time-window drift detection

This architecture mirrors how **real LLM observability systems** are built.

---

## What This Project Intentionally Avoids

- RLHF or fine-tuning
- LLM-based evaluators
- Dashboard-heavy tooling
- Benchmark-only metrics

The focus is **operational reliability**, not research metrics.

---

## Use Cases

- Production LLM monitoring
- Prompt or instruction change validation
- Model version comparison
- Reliability audits
- LLMOps / MLOps portfolio demonstration

---

## License

MIT

