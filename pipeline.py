import subprocess
import sys
import os
import pandas as pd
from datetime import datetime, timezone

# ===============================
# PIPELINE CONFIG
# ===============================

STAGES = [

    "stages/stage_2_inference.py",
    "stages/stage_3_features.py",
    "stages/stage_4_scoring.py",
    "stages/stage_5_drift_baseline.py",
    "stages/stage_6_drift_timewindow.py",
    
]

RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

print("=" * 60)
print("PIPELINE STARTED")
print(f"RUN_ID: {RUN_ID}")
print("=" * 60)

# ===============================
# RUN STAGES
# ===============================

for stage in STAGES:
    print(f"\nRunning {stage}")
    result = subprocess.run(
        [sys.executable, stage, "--run_id", RUN_ID]
    )

    if result.returncode != 0:
        print(f"\nPIPELINE FAILED at {stage}")
        sys.exit(1)

    print(f"Completed {stage}")

# ===============================
# PIPELINE SUMMARY
# ===============================

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)

run_dir = os.path.join("artifacts", "runs", RUN_ID)

try:
    scores_path = os.path.join(run_dir, "llm_scores.csv")
    drift_base_path = os.path.join(run_dir, "llm_drift_report.csv")
    drift_time_path = os.path.join(run_dir, "llm_drift_timewindow.csv")

    if os.path.exists(scores_path):
        scores_df = pd.read_csv(scores_path)
        rows_scored = len(scores_df)
        models = sorted(scores_df["model"].unique().tolist())
    else:
        rows_scored = "N/A"
        models = []

    if os.path.exists(drift_base_path):
        base_df = pd.read_csv(drift_base_path)
        baseline_drift = bool(base_df["drift_detected"].max())
    else:
        baseline_drift = "N/A"

    if os.path.exists(drift_time_path):
        time_df = pd.read_csv(drift_time_path)
        time_window_drift = bool(time_df["drift_detected"].max())
    else:
        time_window_drift = "N/A"

    print(f"Run ID: {RUN_ID}")
    print(f"Artifacts directory: {run_dir}")
    print(f"Rows evaluated: {rows_scored}")
    print(f"Models evaluated: {', '.join(models) if models else 'N/A'}")
    print(f"Baseline drift detected: {'YES' if baseline_drift else 'NO'}")
    print(f"Time-window drift detected: {'YES' if time_window_drift else 'NO'}")

except Exception as e:
    print("Failed to generate pipeline summary")
    print(e)

# ===============================
# FINAL STATUS
# ===============================

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print(f"RUN_ID: {RUN_ID}")
print("=" * 60)
