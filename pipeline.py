import subprocess
import sys
import os
import pandas as pd
import mlflow
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
RUN_DIR = os.path.join("artifacts", "runs", RUN_ID)

# Optional but recommended
mlflow.set_experiment("llm_output_quality_pipeline")

print("=" * 60)
print("PIPELINE STARTED")
print(f"RUN_ID: {RUN_ID}")
print("=" * 60)

# ===============================
# MLflow RUN (PIPELINE LEVEL)
# ===============================

with mlflow.start_run(run_name=RUN_ID):

    # ---- Log static pipeline context
    mlflow.set_tag("pipeline", "llm_output_quality")
    mlflow.set_tag("run_id", RUN_ID)

    if os.path.exists("mlops/config.yaml"):
        mlflow.log_artifact("mlops/config.yaml")

    if os.path.exists("mlops/policies.yaml"):
        mlflow.log_artifact("mlops/policies.yaml")

    mlflow.log_param("num_stages", len(STAGES))

    # ===============================
    # RUN STAGES
    # ===============================

    for stage in STAGES:
        print(f"\nRunning {stage}")
        result = subprocess.run(
            [sys.executable, stage, "--run_id", RUN_ID]
        )

        if result.returncode != 0:
            mlflow.set_tag("pipeline_status", "FAILED")
            mlflow.set_tag("failed_stage", stage)
            print(f"\nPIPELINE FAILED at {stage}")
            sys.exit(1)

        print(f"Completed {stage}")

    # ===============================
    # PIPELINE SUMMARY + METRICS
    # ===============================

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    try:
        scores_path = os.path.join(RUN_DIR, "llm_scores.csv")
        drift_base_path = os.path.join(RUN_DIR, "llm_drift_report.csv")
        drift_time_path = os.path.join(RUN_DIR, "llm_drift_timewindow.csv")

        rows_scored = 0
        baseline_drift = 0
        time_window_drift = 0
        models = []

        # ===============================
        # LOAD SCORES + LOG RICH METRICS
        # ===============================
        if os.path.exists(scores_path):
            scores_df = pd.read_csv(scores_path)
            rows_scored = len(scores_df)
            models = sorted(scores_df["model"].astype(str).unique().tolist())

            # ---- Choose the best available quality column
            quality_col_candidates = ["quality_score", "overall_quality_score"]
            quality_col = next((c for c in quality_col_candidates if c in scores_df.columns), None)

            if quality_col:
                avg_quality = float(scores_df[quality_col].mean())
                mlflow.log_metric("avg_quality_score", avg_quality)

            mlflow.log_metric("rows_evaluated", rows_scored)

            # ---- Optional richer metrics (only if columns exist)
            if "latency_sec" in scores_df.columns:
                mlflow.log_metric("avg_latency_sec", float(scores_df["latency_sec"].mean()))
            elif "latency_score" in scores_df.columns:
                mlflow.log_metric("avg_latency_score", float(scores_df["latency_score"].mean()))

            if "verbosity_score" in scores_df.columns:
                mlflow.log_metric("avg_verbosity", float(scores_df["verbosity_score"].mean()))

            if "consistency_score" in scores_df.columns:
                mlflow.log_metric("avg_consistency", float(scores_df["consistency_score"].mean()))

            if "instructional_stability_score" in scores_df.columns:
                mlflow.log_metric("avg_instructional_stability", float(scores_df["instructional_stability_score"].mean()))

            # ---- Per-model metrics (kept clean + safe metric names)
            # If you want to limit clutter, change None -> 5 (top 5 models by rows)
            TOP_N_MODELS = None

            if TOP_N_MODELS:
                top_models = scores_df["model"].value_counts().head(TOP_N_MODELS).index.tolist()
                model_groups = [(m, scores_df[scores_df["model"] == m]) for m in top_models]
            else:
                model_groups = list(scores_df.groupby("model"))

            for model, dfm in model_groups:
                safe_model = str(model).replace("/", "_").replace(" ", "_")

                if quality_col:
                    mlflow.log_metric(f"model__{safe_model}__avg_quality", float(dfm[quality_col].mean()))

                if "latency_sec" in dfm.columns:
                    mlflow.log_metric(f"model__{safe_model}__avg_latency_sec", float(dfm["latency_sec"].mean()))
                elif "latency_score" in dfm.columns:
                    mlflow.log_metric(f"model__{safe_model}__avg_latency_score", float(dfm["latency_score"].mean()))

                if "verbosity_score" in dfm.columns:
                    mlflow.log_metric(f"model__{safe_model}__avg_verbosity", float(dfm["verbosity_score"].mean()))

                if "consistency_score" in dfm.columns:
                    mlflow.log_metric(f"model__{safe_model}__avg_consistency", float(dfm["consistency_score"].mean()))

        # ===============================
        # DRIFT METRICS
        # ===============================
        if os.path.exists(drift_base_path):
            base_df = pd.read_csv(drift_base_path)
            baseline_drift = int(base_df["drift_detected"].max())
            mlflow.log_metric("baseline_drift_detected", baseline_drift)

        if os.path.exists(drift_time_path):
            time_df = pd.read_csv(drift_time_path)
            time_window_drift = int(time_df["drift_detected"].max())
            mlflow.log_metric("timewindow_drift_detected", time_window_drift)

        # ---- Log artifacts directory
        if os.path.exists(RUN_DIR):
            mlflow.log_artifacts(RUN_DIR, artifact_path="pipeline_outputs")

        # ---- Console summary
        print(f"Run ID: {RUN_ID}")
        print(f"Artifacts directory: {RUN_DIR}")
        print(f"Rows evaluated: {rows_scored}")
        print(f"Models evaluated: {', '.join(models) if models else 'N/A'}")
        print(f"Baseline drift detected: {'YES' if baseline_drift else 'NO'}")
        print(f"Time-window drift detected: {'YES' if time_window_drift else 'NO'}")

        mlflow.set_tag("pipeline_status", "SUCCESS")

    except Exception as e:
        mlflow.set_tag("pipeline_status", "SUMMARY_FAILED")
        print("Failed to generate pipeline summary")
        print(e)

# ===============================
# FINAL STATUS
# ===============================

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print(f"RUN_ID: {RUN_ID}")
print("=" * 60)
