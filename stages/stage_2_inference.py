"""
STAGE 2 — OpenAI Inference & JSONL Logging (COST-SAFE + CONTROLLED DRIFT)
+ RUN ID + ARTIFACTS STRUCTURE + AUTO JSONL→CSV + LATEST POINTERS

OpenAI SDK: openai==2.16.0
API: Responses API (client.responses.create)

PROTECTIONS:
- Hard RPM limiter
- Output token cap
- Append-only JSONL (resume-safe within the same run)
- Timestamped (for time-window drift)

PRODUCTION ADDITIONS:
- run_id per pipeline execution
- writes to artifacts/runs/{run_id}/
- updates artifacts/latest/
- converts JSONL -> CSV at end of run
"""

# ===============================
# 1. IMPORTS
# ===============================
import argparse

import os
import time
import json
import csv
import hashlib
import shutil
from typing import Dict, List
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.run_utils import generate_run_id, create_run_dirs, resolve_run_id


load_dotenv()

# ===============================
# 2. CLIENT
# ===============================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ===============================
# 3. RATE LIMITER
# ===============================

MAX_RPM = 30
MIN_DELAY = 60 / MAX_RPM
_last_call = 0.0

def rate_limit():
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < MIN_DELAY:
        time.sleep(MIN_DELAY - elapsed)
    _last_call = time.time()

# ===============================
# ARGUMENT PARSING (PIPELINE SUPPORT)
# ===============================

parser = argparse.ArgumentParser(description="Stage 2 — OpenAI Inference")
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help="Optional pipeline run_id (if not provided, one will be generated)"
)
args = parser.parse_args()


# ===============================
# 4. RUN CONTEXT + CONFIG
# ===============================

RUN_ID = resolve_run_id(args.run_id)
RUN_DIR = create_run_dirs(RUN_ID)

######################### Actual settings ##########################

MODELS = [
    "gpt-4o-mini",         # primary
    "gpt-4.1-mini",        # secondary
    "gpt-3.5-turbo-0125"   # legacy baseline
]

# Portfolio-friendly drift (valid, controlled)
TEMPERATURES = [0.2, 0.8]
RUNS = [1, 2]  # Run 1 = concise instruction, Run 2 = detailed instruction


######################### Testing settings ##########################

# MODELS = [
#     "gpt-4o-mini",        # secondary
# ]

# # Portfolio-friendly drift (valid, controlled)
# TEMPERATURES = [0.2]
# RUNS = [1]  # Run 1 = concise instruction, Run 2 = detailed instruction



MAX_OUTPUT_TOKENS = 300

# Run-scoped artifacts
JSONL_PATH = f"{RUN_DIR}/raw_llm_outputs.jsonl"
CSV_PATH = f"{RUN_DIR}/raw_llm_outputs.csv"

# Latest pointers (what production reads)
LATEST_JSONL_PATH = "artifacts/latest/raw_llm_outputs.jsonl"
LATEST_CSV_PATH = "artifacts/latest/raw_llm_outputs.csv"

# Ensure latest directory exists
os.makedirs(os.path.dirname(LATEST_JSONL_PATH), exist_ok=True)

# ===============================
# 5. LOAD PROMPTS
# ===============================

# If your prompts are in data/prompts.csv (as your screenshot shows), keep this:
PROMPTS_PATH = "data/prompts.csv"

df = pd.read_csv(PROMPTS_PATH)
assert len(df) == 300, f"Expected exactly 300 prompts, got {len(df)}"

# ===============================
# 6. RESUME SUPPORT (PER-RUN)
# ===============================

def load_completed_hashes(jsonl_path: str) -> set:
    hashes = set()
    if not os.path.exists(jsonl_path):
        return hashes

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if "task_hash" in record:
                    hashes.add(record["task_hash"])
            except Exception:
                # Skip corrupted lines gracefully
                continue
    return hashes

def make_task_hash(task: Dict) -> str:
    raw = f'{task["prompt_id"]}|{task["model"]}|{task["temperature"]}|{task["run"]}'
    return hashlib.sha256(raw.encode()).hexdigest()

# ===============================
# 7. CONTROLLED DRIFT: INSTRUCTIONS
# ===============================

def get_instruction(run: int) -> str:
    """
    Controlled 'prompt-policy drift' across runs.
    Run 1: concise (stable/short)
    Run 2: detailed (more verbose/contextual)
    """
    if run == 1:
        return "Answer clearly and concisely. Use 1–2 sentences."
    return "Answer clearly with helpful context. Use 4–7 sentences and include brief reasoning."

def get_instruction_version(run: int) -> str:
    return "v1_concise" if run == 1 else "v2_detailed"

# ===============================
# 8. OPENAI CALL
# ===============================

def call_openai(prompt: str, model: str, temperature: float, run: int):
    rate_limit()

    start = time.time()

    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        instructions=get_instruction(run),
        input=prompt,
    )

    latency = time.time() - start
    text = response.output_text.strip() if response and response.output_text else ""

    return text, latency

# ===============================
# 9. TASK GENERATION
# ===============================

def create_tasks() -> List[Dict]:
    tasks: List[Dict] = []
    for _, row in df.iterrows():
        for model in MODELS:
            for temp in TEMPERATURES:
                for run in RUNS:
                    task = {
                        "prompt_id": row["prompt_id"],
                        "source": row["source"],
                        "category": row["category"],
                        "prompt_text": row["prompt_text"],
                        "model": model,
                        "temperature": float(temp),
                        "run": int(run),
                    }
                    task["task_hash"] = make_task_hash(task)
                    tasks.append(task)
    return tasks

# ===============================
# 10. JSONL → CSV CONVERTER
# ===============================

FIELDNAMES = [
    "run_id",
    "prompt_id",
    "source",
    "category",
    "model",
    "temperature",
    "run",
    "instruction_version",
    "response_text",
    "response_length",
    "latency_sec",
    "status",
    "error",
    "task_hash",
    "run_timestamp",
]

def jsonl_to_csv(jsonl_path: str, csv_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(csv_path, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=FIELDNAMES,
            extrasaction="ignore"
        )
        writer.writeheader()

        for line_num, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON at line {line_num}: {e}")
                continue

            row = {field: record.get(field) for field in FIELDNAMES}
            writer.writerow(row)

    print(f"Conversion complete → {csv_path}")

# ===============================
# 11. MAIN LOOP (JSONL APPEND)
# ===============================

def main():
    print(f"\n=== STAGE 2 START ===")
    print(f"RUN_ID: {RUN_ID}")
    print(f"RUN_DIR: {RUN_DIR}")
    print(f"JSONL_PATH: {JSONL_PATH}")

    completed = load_completed_hashes(JSONL_PATH)
    tasks = create_tasks()

    print(f"Total tasks: {len(tasks)}")
    print(f"Already completed (this run): {len(completed)}")
    print(f"Will run: {len(tasks) - len(completed)}\n")

    # Append-only JSONL (resume-safe)
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for task in tqdm(tasks):

            if task["task_hash"] in completed:
                continue

            now_iso = datetime.now(timezone.utc).isoformat()

            try:
                text, latency = call_openai(
                    task["prompt_text"],
                    task["model"],
                    task["temperature"],
                    task["run"],
                )

                record = {
                    "run_id": RUN_ID,
                    "prompt_id": task["prompt_id"],
                    "source": task["source"],
                    "category": task["category"],
                    "model": task["model"],
                    "temperature": task["temperature"],
                    "run": task["run"],
                    "instruction_version": get_instruction_version(task["run"]),
                    "response_text": text,
                    "response_length": len(text) if text else 0,
                    "latency_sec": latency,
                    "status": "ok",
                    "error": None,
                    "task_hash": task["task_hash"],
                    "run_timestamp": now_iso,
                }

            except Exception as e:
                record = {
                    "run_id": RUN_ID,
                    "prompt_id": task["prompt_id"],
                    "source": task["source"],
                    "category": task["category"],
                    "model": task["model"],
                    "temperature": task["temperature"],
                    "run": task["run"],
                    "instruction_version": get_instruction_version(task["run"]),
                    "response_text": None,
                    "response_length": 0,
                    "latency_sec": None,
                    "status": "failed",
                    "error": str(e),
                    "task_hash": task["task_hash"],
                    "run_timestamp": now_iso,
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    # ===============================
    # FINALIZE: JSONL → CSV + UPDATE LATEST
    # ===============================

    print("\nFinalizing artifacts...")
    jsonl_to_csv(JSONL_PATH, CSV_PATH)

    shutil.copyfile(JSONL_PATH, LATEST_JSONL_PATH)
    shutil.copyfile(CSV_PATH, LATEST_CSV_PATH)

    print(f"Latest JSONL updated → {LATEST_JSONL_PATH}")
    print(f"Latest CSV updated  → {LATEST_CSV_PATH}")
    print(f"=== STAGE 2 COMPLETE: {RUN_ID} ===\n")

if __name__ == "__main__":
    main()
