"""
STAGE 6 — Time-Window Drift Detection (PRODUCTION-GRADE)

Baseline → earlier time window
Current  → recent time window

INPUT:
- artifacts/latest/llm_scores.csv

OUTPUT:
- artifacts/runs/{run_id}/llm_drift_timewindow.csv
- artifacts/latest/llm_drift_timewindow.csv

NOTES:
- Instruction-aware (per model + instruction_version)
- Config-driven thresholds from mlops/config.yaml
- Preserves run lineage via run_id
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
OUTPUT_PATH = f"{RUN_DIR}/llm_drift_timewindow.csv"
LATEST_PATH = "artifacts/latest/llm_drift_timewindow.csv"

os.makedirs(os.path.dirname(LATEST_PATH), exist_ok=True)

TIME_COL = "run_timestamp"

print(f"RUN_ID: {RUN_ID}")

# ===============================
# 3. LOAD CONFIG (THRESHOLDS)
# ===============================

with open("mlops/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

drift_cfg = config["drift"]

KS_THRESHOLD = float(drift_cfg["ks_threshold"])
MEAN_THRESHOLD = float(drift_cfg["mean_threshold"])
MIN_ROWS_PER_WINDOW = int(drift_cfg["min_rows"])

# ===============================
# 4. LOAD DATA
# ===============================

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} total rows from {INPUT_PATH}")

# ===============================
# 5. TIME PARSING
# ===============================

if TIME_COL not in df.columns:
    print(f"Missing time column: {TIME_COL}")
    sys.exit(1)

df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
df = df.dropna(subset=[TIME_COL])

# ===============================
# 6. SCORE SANITIZATION
# ===============================

SCORE_COLS = [
    "overall_quality_score",
    "verbosity_score",
    "instructional_stability_score",
    "latency_score",
]

df[SCORE_COLS] = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=SCORE_COLS)

print(f"Rows after score filtering: {len(df)}")

# ===============================
# 7. DRIFT FUNCTIONS
# ===============================

def ks_drift(b, c):
    if len(b) < MIN_ROWS_PER_WINDOW or len(c) < MIN_ROWS_PER_WINDOW:
        return 0.0
    return float(ks_2samp(b, c).statistic)

def mean_shift(b, c):
    if len(b) < MIN_ROWS_PER_WINDOW:
        return 0.0
    m = b.mean()
    if m == 0 or np.isnan(m):
        return 0.0
    return float(abs(c.mean() - m) / abs(m))

# ===============================
# 8. TIME-WINDOW DRIFT (PER MODEL + INSTRUCTION)
# ===============================

records = []

pairs = df[["model", "instruction_version"]].drop_duplicates()

for _, pair in pairs.iterrows():
    model = pair["model"]
    instr = pair["instruction_version"]

    subset = (
        df[
            (df["model"] == model) &
            (df["instruction_version"] == instr)
        ]
        .sort_values(TIME_COL)
    )

    # Need enough data to split into two windows
    if len(subset) < 2 * MIN_ROWS_PER_WINDOW:
        print(f"Skipping {model} | {instr} (insufficient time data)")
        continue

    midpoint = int(len(subset) * 0.5)

    baseline = subset.iloc[:midpoint]
    current = subset.iloc[midpoint:]

    if len(baseline) < MIN_ROWS_PER_WINDOW or len(current) < MIN_ROWS_PER_WINDOW:
        print(f"Skipping {model} | {instr} (window too small after split)")
        continue

    records.append({
        "run_id": RUN_ID,
        "model": model,
        "instruction_version": instr,

        "baseline_start": baseline[TIME_COL].min(),
        "baseline_end": baseline[TIME_COL].max(),
        "current_start": current[TIME_COL].min(),
        "current_end": current[TIME_COL].max(),

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
# 9. EMPTY SAFETY EXIT
# ===============================

if drift_df.empty:
    print("No time-window drift computed")
    drift_df.to_csv(OUTPUT_PATH, index=False)
    shutil.copyfile(OUTPUT_PATH, LATEST_PATH)
    sys.exit(0)

# ===============================
# 10. DRIFT FLAGS (CONFIG-DRIVEN)
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
# 11. SAVE ARTIFACTS
# ===============================

drift_df.to_csv(OUTPUT_PATH, index=False)
shutil.copyfile(OUTPUT_PATH, LATEST_PATH)

print("\n=> STAGE 6 COMPLETE — Time-Window Drift Report Generated")
print(drift_df)

print(f"\nRun artifact  → {OUTPUT_PATH}")
print(f"Latest alias  → {LATEST_PATH}")
print(f"RUN_ID       → {RUN_ID}\n")
