"""
STAGE 3 — Feature Extraction (PRODUCTION-GRADE)

INPUT:
- artifacts/latest/raw_llm_outputs.csv

OUTPUT:
- artifacts/runs/{run_id}/llm_features.csv
- artifacts/latest/llm_features.csv

RULES:
- DO NOT modify raw outputs
- Feature logic must remain deterministic
"""

# ===============================
# 1. IMPORTS
# ===============================

import os
import re
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.run_utils import create_run_dirs, resolve_run_id


tqdm.pandas()

import argparse

# ===============================
# ARGUMENT PARSING (PIPELINE SUPPORT)
# ===============================

parser = argparse.ArgumentParser(description="Stage 3 — Feature Extraction")
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

INPUT_PATH = "artifacts/latest/raw_llm_outputs.csv"

OUTPUT_PATH = f"{RUN_DIR}/llm_features.csv"
LATEST_PATH = "artifacts/latest/llm_features.csv"

os.makedirs(os.path.dirname(LATEST_PATH), exist_ok=True)

# ===============================
# 3. LOAD DATA
# ===============================

df = pd.read_csv(INPUT_PATH)

# Keep only successful generations
df = df[df["status"] == "ok"].copy()

print(f"Loaded {len(df)} valid rows from {INPUT_PATH}")
print(f"RUN_ID: {RUN_ID}")

# ===============================
# 4. TEXT FEATURE FUNCTIONS
# ===============================

def char_len(text):
    return len(text)

def word_len(text):
    return len(text.split())

def sentence_count(text):
    return len(re.findall(r"[.!?]", text))

def avg_word_len(text):
    words = text.split()
    return np.mean([len(w) for w in words]) if words else 0

# ===============================
# 5. FORMAT / BEHAVIOR FEATURES
# ===============================

def has_markdown(text):
    return int(bool(re.search(r"(\*\*|##|\|---|\|)", text)))

def has_bullets(text):
    return int(bool(re.search(r"(^|\n)[\-\•\*]\s+", text)))

def starts_with_answer(text):
    first_word = text.strip().split(" ")[0].lower()
    return int(first_word in {
        "the", "first", "it", "this", "he", "she", "they"
    })

def contains_disclaimer(text):
    patterns = [
        "as of my knowledge",
        "i may be wrong",
        "cannot guarantee",
        "may have changed",
        "verify",
        "check official",
    ]
    t = text.lower()
    return int(any(p in t for p in patterns))

def contains_uncertainty(text):
    patterns = [
        "might",
        "possibly",
        "likely",
        "uncertain",
        "unclear",
        "not sure",
        "unknown"
    ]
    t = text.lower()
    return int(any(p in t for p in patterns))

def contains_date(text):
    return int(bool(re.search(r"\b(18|19|20)\d{2}\b", text)))

# ===============================
# 6. APPLY FEATURE EXTRACTION
# ===============================

print("Extracting features...")

df["char_len"] = df["response_text"].progress_apply(char_len)
df["word_len"] = df["response_text"].progress_apply(word_len)
df["sentence_count"] = df["response_text"].progress_apply(sentence_count)
df["avg_word_len"] = df["response_text"].progress_apply(avg_word_len)

df["has_markdown"] = df["response_text"].progress_apply(has_markdown)
df["has_bullets"] = df["response_text"].progress_apply(has_bullets)
df["starts_with_answer"] = df["response_text"].progress_apply(starts_with_answer)
df["contains_disclaimer"] = df["response_text"].progress_apply(contains_disclaimer)
df["contains_uncertainty"] = df["response_text"].progress_apply(contains_uncertainty)
df["contains_date"] = df["response_text"].progress_apply(contains_date)

# ===============================
# 7. SELECT FINAL FEATURE SET
# ===============================

FEATURE_COLUMNS = [
    # Lineage
    "run_id",

    # Identity & context
    "prompt_id",
    "source",
    "category",
    "model",
    "temperature",
    "run",
    "instruction_version",
    "run_timestamp",

    # Infra
    "latency_sec",
    "response_length",

    # Textual features
    "char_len",
    "word_len",
    "sentence_count",
    "avg_word_len",

    # Behavioral / format features
    "has_markdown",
    "has_bullets",
    "starts_with_answer",
    "contains_disclaimer",
    "contains_uncertainty",
    "contains_date",
]

features_df = df[FEATURE_COLUMNS].copy()

# ===============================
# 8. SAVE FEATURES
# ===============================

features_df.to_csv(OUTPUT_PATH, index=False)
shutil.copyfile(OUTPUT_PATH, LATEST_PATH)

print(f"\nSTAGE 3 COMPLETE")
print(f"Run artifact  → {OUTPUT_PATH}")
print(f"Latest alias  → {LATEST_PATH}")
print(f"Rows written → {len(features_df)}")
print(f"RUN_ID       → {RUN_ID}\n")
