"""
STAGE 5 — Drift Detection (PRODUCTION-GRADE)
INSTRUCTION-AWARE, BASELINE vs CURRENT

INPUT:
- artifacts/latest/llm_scores.csv

OUTPUT:
- artifacts/runs/{run_id}/llm_drift_report.csv
- artifacts/latest/llm_drift_report.csv

NOTES:
- Baseline vs current comparison only
- Instruction-aware (v1_concise vs v2_detailed)
- Config-driven thresholds
"""

# ===============================
# 1. IMPORTS
# ===============================

import os
import sys
import shutil
import yaml
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utilities.run_utils import create_run_dirs, resolve_run_id



import argparse

# ===============================
# ARGUMENT PARSING (PIPELINE SUPPORT)
# ===============================

parser = argparse.ArgumentParser(description="Stage 4 — Quality Scoring")
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help="Optional pipeline run_id (if not provided, one will be generated)"
)
args = parser.parse_args()



# ===============================
# 2. RUN CONTEXT (MLOps)
# ===============================

RUN_ID = resolve_run_id(args.run_id)
RUN_DIR = create_run_dirs(RUN_ID)

INPUT_PATH = "artifacts/latest/llm_scores.csv"
OUTPUT_PATH = f"{RUN_DIR}/llm_drift_report.csv"
LATEST_PATH = "artifacts/latest/llm_drift_report.csv"

os.makedirs(os.path.dirname(LATEST_PATH), exist_ok=True)

print(f"RUN_ID: {RUN_ID}")

# ===============================
# 3. LOAD CONFIG (THRESHOLDS)
# ===============================

with open("mlops/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

drift_cfg = config["drift"]

KS_THRESHOLD = drift_cfg["ks_threshold"]
MEAN_THRESHOLD = drift_cfg["mean_threshold"]
MIN_ROWS = drift_cfg["min_rows"]

# ===============================
# 4. LOAD SCORES
# ===============================

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} total rows from {INPUT_PATH}")

# ===============================
# 5. REQUIRED SCORE COLUMNS
# ===============================

SCORE_COLS = [
    "overall_quality_score",
    "verbosity_score",
    "instructional_stability_score",
    "latency_score",
]

df[SCORE_COLS] = df[SCORE_COLS].apply(
    pd.to_numeric, errors="coerce"
)

print("\nNaN counts before filtering:")
print(df[SCORE_COLS].isna().sum())

# ===============================
# 6. FILTER INCOMPLETE ROWS
# ===============================

clean_df = df.dropna(subset=SCORE_COLS).copy()
print(f"\nRows after completeness filter: {len(clean_df)}")

if len(clean_df) < MIN_ROWS:
    print("Insufficient data for drift detection — exiting safely.")
    clean_df.to_csv(OUTPUT_PATH, index=False)
    sys.exit(0)

# ===============================
# 7. BASELINE DEFINITION (PER MODEL + INSTRUCTION)
# ===============================

clean_df = clean_df.sort_values("prompt_id")
clean_df["is_baseline"] = 0

for (model, instr) in (
    clean_df[["model", "instruction_version"]]
    .drop_duplicates()
    .itertuples(index=False)
):

    subset = clean_df[
        (clean_df["model"] == model) &
        (clean_df["instruction_version"] == instr) &
        (clean_df["temperature"] == 0.2)
    ]

    prompt_ids = subset["prompt_id"].unique()
    cutoff = int(0.3 * len(prompt_ids))

    baseline_ids = prompt_ids[:cutoff]

    clean_df.loc[
        (clean_df["model"] == model) &
        (clean_df["instruction_version"] == instr) &
        (clean_df["prompt_id"].isin(baseline_ids)),
        "is_baseline"
    ] = 1

print("\nBaseline size per (model, instruction_version):")
print(
    clean_df[clean_df["is_baseline"] == 1]
    .groupby(["model", "instruction_version"])
    .size()
)

# ===============================
# 8. DRIFT METRIC FUNCTIONS
# ===============================

def ks_drift(baseline, current):
    if len(baseline) < MIN_ROWS or len(current) < MIN_ROWS:
        return 0.0
    return float(ks_2samp(baseline, current).statistic)

def mean_shift(baseline, current):
    if len(baseline) < MIN_ROWS:
        return 0.0

    base_mean = baseline.mean()
    if base_mean == 0 or np.isnan(base_mean):
        return 0.0

    return float(abs(current.mean() - base_mean) / abs(base_mean))

# ===============================
# 9. DRIFT COMPUTATION
# ===============================

records = []

for (model, instr) in (
    clean_df[["model", "instruction_version"]]
    .drop_duplicates()
    .itertuples(index=False)
):

    baseline = clean_df[
        (clean_df["model"] == model) &
        (clean_df["instruction_version"] == instr) &
        (clean_df["is_baseline"] == 1)
    ]

    current = clean_df[
        (clean_df["model"] == model) &
        (clean_df["instruction_version"] == instr) &
        (clean_df["is_baseline"] == 0)
    ]

    if len(baseline) < MIN_ROWS or len(current) < MIN_ROWS:
        print(f"Skipping {model} | {instr} (insufficient data)")
        continue

    records.append({
        "run_id": RUN_ID,
        "model": model,
        "instruction_version": instr,

        "quality_drift": ks_drift(
            baseline["overall_quality_score"],
            current["overall_quality_score"]
        ),
        "verbosity_drift": ks_drift(
            baseline["verbosity_score"],
            current["verbosity_score"]
        ),
        "instructional_stability_drift": mean_shift(
            baseline["instructional_stability_score"],
            current["instructional_stability_score"]
        ),
        "latency_drift": mean_shift(
            baseline["latency_score"],
            current["latency_score"]
        ),
    })

drift_df = pd.DataFrame(records)

# ===============================
# 10. EMPTY SAFE EXIT
# ===============================

if drift_df.empty:
    print("\nNo drift computed — no valid model/instruction pairs")
    drift_df.to_csv(OUTPUT_PATH, index=False)
    shutil.copyfile(OUTPUT_PATH, LATEST_PATH)
    sys.exit(0)

# ===============================
# 11. DRIFT FLAGS (CONFIG-DRIVEN)
# ===============================

drift_df["quality_drift_flag"] = (drift_df["quality_drift"] > KS_THRESHOLD).astype(int)
drift_df["verbosity_drift_flag"] = (drift_df["verbosity_drift"] > KS_THRESHOLD).astype(int)

drift_df["instructional_stability_drift_flag"] = (
    drift_df["instructional_stability_drift"] > MEAN_THRESHOLD
).astype(int)

drift_df["latency_drift_flag"] = (
    drift_df["latency_drift"] > MEAN_THRESHOLD
).astype(int)

drift_df["drift_detected"] = (
    drift_df[
        [
            "quality_drift_flag",
            "verbosity_drift_flag",
            "instructional_stability_drift_flag",
            "latency_drift_flag",
        ]
    ].sum(axis=1) >= 2
).astype(int)

# ===============================
# 12. SAVE ARTIFACTS
# ===============================

drift_df.to_csv(OUTPUT_PATH, index=False)
shutil.copyfile(OUTPUT_PATH, LATEST_PATH)

print("\n=> STAGE 5 COMPLETE — Drift Report Generated")
print(drift_df)
print(f"\nRun artifact  → {OUTPUT_PATH}")
print(f"Latest alias  → {LATEST_PATH}")
print(f"RUN_ID       → {RUN_ID}")
