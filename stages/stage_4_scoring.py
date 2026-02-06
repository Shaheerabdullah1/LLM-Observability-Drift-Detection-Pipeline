"""
STAGE 4 — Quality & Reliability Scoring (PRODUCTION-GRADE)

INPUT:
- artifacts/latest/llm_features.csv

OUTPUT:
- artifacts/runs/{run_id}/llm_scores.csv
- artifacts/latest/llm_scores.csv

NOTES:
- With 2 runs, we measure INSTRUCTIONAL STABILITY (concise vs detailed)
- Uses config-driven weights from mlops/config.yaml
- Preserves lineage via run_id
"""

# ===============================
# 1. IMPORTS
# ===============================

import os
import shutil
import yaml
import pandas as pd
import numpy as np

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


INPUT_PATH = "artifacts/latest/llm_features.csv"

OUTPUT_PATH = f"{RUN_DIR}/llm_scores.csv"
LATEST_PATH = "artifacts/latest/llm_scores.csv"

os.makedirs(os.path.dirname(LATEST_PATH), exist_ok=True)

# ===============================
# 3. LOAD CONFIG (WEIGHTS)
# ===============================

CONFIG_PATH = "mlops/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

weights = config["scoring"]["weights"]

REQUIRED_WEIGHT_KEYS = [
    "instructional_stability",
    "confidence",
    "verbosity",
    "format",
    "latency",
]

missing = [k for k in REQUIRED_WEIGHT_KEYS if k not in weights]
if missing:
    raise KeyError(
        f"Missing keys in mlops/config.yaml scoring.weights: {missing}. "
        f"Expected keys: {REQUIRED_WEIGHT_KEYS}"
    )

# Optional sanity check (warn only)
weight_sum = float(sum(weights[k] for k in REQUIRED_WEIGHT_KEYS))
if not (0.95 <= weight_sum <= 1.05):
    print(f"⚠️ Warning: scoring weight sum is {weight_sum:.3f} (expected ~1.0)")

# ===============================
# 4. LOAD FEATURES
# ===============================

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} feature rows from {INPUT_PATH}")
print(f"RUN_ID: {RUN_ID}")

# Basic schema checks (fail fast)
REQUIRED_COLS = [
    "run_id",
    "prompt_id",
    "source",
    "category",
    "model",
    "temperature",
    "run",
    "instruction_version",
    "run_timestamp",
    "char_len",
    "word_len",
    "sentence_count",
    "contains_disclaimer",
    "contains_uncertainty",
    "has_markdown",
    "has_bullets",
    "latency_sec",
]
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in llm_features.csv: {missing_cols}")

# ===============================
# 5. INSTRUCTIONAL STABILITY SCORE
# ===============================

def compute_instructional_stability(group: pd.DataFrame) -> float:
    g1 = group[group["run"] == 1]
    g2 = group[group["run"] == 2]

    if len(g1) == 0 or len(g2) == 0:
        return 0.5  # neutral if one side missing

    g1 = g1.iloc[0]
    g2 = g2.iloc[0]

    diffs = [
        abs(float(g1["char_len"]) - float(g2["char_len"])),
        abs(float(g1["word_len"]) - float(g2["word_len"])),
        abs(float(g1["sentence_count"]) - float(g2["sentence_count"])),
    ]

    norm_diff = float(np.mean(diffs))

    # Smooth decay: small difference → high score
    score = float(np.exp(-norm_diff / 20.0))
    return score

# Compute once per (prompt, model, temperature)
stability_df = (
    df.groupby(["prompt_id", "model", "temperature"], dropna=False)
      .apply(compute_instructional_stability)
      .reset_index()
      .rename(columns={0: "instructional_stability_score"})
)

# Merge back
df = df.merge(
    stability_df,
    on=["prompt_id", "model", "temperature"],
    how="left"
)

# Fill in edge cases
df["instructional_stability_score"] = df["instructional_stability_score"].fillna(0.5).clip(0, 1)

# ===============================
# 6. VERBOSITY SCORE
# ===============================

low, high = np.percentile(df["char_len"].astype(float), [10, 90])

def verbosity_score(x: float) -> float:
    x = float(x)
    if low <= 0:
        return 1.0
    if x < low:
        return float(x / low)
    if x > high and x > 0:
        return float(high / x)
    return 1.0

df["verbosity_score"] = df["char_len"].apply(verbosity_score).clip(0, 1)

# ===============================
# 7. CONFIDENCE SCORE
# ===============================

df["confidence_score"] = 1.0
df.loc[df["contains_disclaimer"] == 1, "confidence_score"] -= 0.4
df.loc[df["contains_uncertainty"] == 1, "confidence_score"] -= 0.3
df["confidence_score"] = df["confidence_score"].clip(0, 1)

# ===============================
# 8. FORMAT SCORE
# ===============================

df["format_score"] = 1.0
df.loc[df["has_markdown"] == 1, "format_score"] -= 0.3
df.loc[df["has_bullets"] == 1, "format_score"] -= 0.3
df["format_score"] = df["format_score"].clip(0, 1)

# ===============================
# 9. LATENCY SCORE (PER MODEL)
# ===============================

df["latency_score"] = 0.0

for model in df["model"].unique():
    mask = df["model"] == model
    max_latency = df.loc[mask, "latency_sec"].astype(float).max()

    if pd.notna(max_latency) and max_latency > 0:
        df.loc[mask, "latency_score"] = (
            1.0 - (df.loc[mask, "latency_sec"].astype(float) / float(max_latency))
        )
    else:
        df.loc[mask, "latency_score"] = 1.0

df["latency_score"] = df["latency_score"].clip(0, 1)

# ===============================
# 10. OVERALL QUALITY SCORE (CONFIG-DRIVEN)
# ===============================

df["overall_quality_score"] = (
    float(weights["instructional_stability"]) * df["instructional_stability_score"]
    + float(weights["confidence"]) * df["confidence_score"]
    + float(weights["verbosity"]) * df["verbosity_score"]
    + float(weights["format"]) * df["format_score"]
    + float(weights["latency"]) * df["latency_score"]
)

df["overall_quality_score"] = df["overall_quality_score"].clip(0, 1)

# ===============================
# 11. SAVE FINAL SCORES
# ===============================

SCORE_COLUMNS = [
    "run_id",

    "prompt_id",
    "source",
    "category",
    "model",
    "temperature",
    "run",
    "instruction_version",
    "run_timestamp",

    "instructional_stability_score",
    "verbosity_score",
    "confidence_score",
    "format_score",
    "latency_score",
    "overall_quality_score",
]

scores_df = df[SCORE_COLUMNS].copy()

scores_df.to_csv(OUTPUT_PATH, index=False)
shutil.copyfile(OUTPUT_PATH, LATEST_PATH)

print(f"\nSTAGE 4 COMPLETE")
print(f"Run artifact  → {OUTPUT_PATH}")
print(f"Latest alias  → {LATEST_PATH}")
print(f"Rows written → {len(scores_df)}")
print(f"RUN_ID       → {RUN_ID}\n")
