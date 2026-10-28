#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-verl}"
fi

RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT_DIR}/processed_data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/processed_data}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
STEPS="${STEPS:-1 2 3 4 5 6}"   # e.g. STEPS="4 5 6" to rebuild labels, episodes and RL parquets (step 4 needs the scoring step done)
SCORE_MODEL_NAME="${SCORE_MODEL_NAME:-qwen3_4b_instruct}"   # folder under processed_data/score/ with scores.jsonl
WRONG_PENALTY="${WRONG_PENALTY:-1.0}"
REQUEST_COST="${REQUEST_COST:-0.05}"
TAGGER_MODEL="${TAGGER_MODEL:-gpt-4.1}"
WORKERS="${WORKERS:-16}"
LIMIT="${LIMIT:-0}"   # e.g. LIMIT=20 for a cheap test run of step 1

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

run_step() { [[ " ${STEPS} " == *" $1 "* ]]; }

# Step 1: sort facts into history / exam / tests, keep patient-specific questions
if run_step 1; then
  : "${OPENAI_API_KEY:?OPENAI_API_KEY must be set in .env}"
  export OPENAI_API_KEY
  python preprocess/build_data/01_tag_facts.py \
    --input-jsonl "${RAW_DATA_DIR}/mediq/all_dev_good.jsonl" \
    --output-jsonl "${PROCESSED_DIR}/tagged/cases_tagged.jsonl" \
    --cache-jsonl "${PROCESSED_DIR}/tagged/api_cache.jsonl" \
    --review-json "${PROCESSED_DIR}/tagged/review_sample.json" \
    --stats-json "${PROCESSED_DIR}/tagged/tag_stats.json" \
    --prompt-file preprocess/prompts/tag_facts.txt \
    --model "${TAGGER_MODEL}" \
    --workers "${WORKERS}" \
    --limit "${LIMIT}" \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/01_tag_facts.log"
fi

# Step 2: stratified train / val / test split of the kept cases
if run_step 2; then
  python preprocess/build_data/02_split_cases.py \
    --tagged-jsonl "${PROCESSED_DIR}/tagged/cases_tagged.jsonl" \
    --output-csv "${PROCESSED_DIR}/splits/case_split.csv" \
    --val-ratio 0.1 \
    --test-ratio 0.2 \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/02_split_cases.log"
fi

# Step 3: scoring prompts, one per case x fact set (H, H+E, H+T, H+E+T) x option order
if run_step 3; then
  python preprocess/build_data/03_build_score_prompts.py \
    --tagged-jsonl "${PROCESSED_DIR}/tagged/cases_tagged.jsonl" \
    --split-csv "${PROCESSED_DIR}/splits/case_split.csv" \
    --instruction-prompt-file preprocess/prompts/score_instruction.txt \
    --output-jsonl "${PROCESSED_DIR}/score/score_prompts.jsonl" \
    --orders rot0,rot1,rot2,rot3 \
    2>&1 | tee "${LOG_DIR}/03_build_score_prompts.log"
fi

# Step 4: best action and best path for every (case, availability version), from the scored p values
if run_step 4; then
  python preprocess/build_data/04_make_labels.py \
    --tagged-jsonl "${PROCESSED_DIR}/tagged/cases_tagged.jsonl" \
    --split-csv "${PROCESSED_DIR}/splits/case_split.csv" \
    --scores-jsonl "${PROCESSED_DIR}/score/${SCORE_MODEL_NAME}/scores.jsonl" \
    --output-jsonl "${PROCESSED_DIR}/labels/${SCORE_MODEL_NAME}/labels.jsonl" \
    --stats-json "${PROCESSED_DIR}/labels/${SCORE_MODEL_NAME}/label_stats.json" \
    --skipped-json "${PROCESSED_DIR}/labels/${SCORE_MODEL_NAME}/skipped_cases.json" \
    --wrong-penalty "${WRONG_PENALTY}" \
    --request-cost "${REQUEST_COST}" \
    --heldout-versions "unavailable:obtainable,obtainable:unavailable" \
    2>&1 | tee "${LOG_DIR}/04_make_labels.log"
fi

# Step 5: episode items (train / val / test), one plain QA item per (case, version), no chat roles
if run_step 5; then
  python preprocess/build_data/05_build_episodes.py \
    --tagged-jsonl "${PROCESSED_DIR}/tagged/cases_tagged.jsonl" \
    --labels-jsonl "${PROCESSED_DIR}/labels/${SCORE_MODEL_NAME}/labels.jsonl" \
    --output-dir "${PROCESSED_DIR}/episodes/${SCORE_MODEL_NAME}" \
    2>&1 | tee "${LOG_DIR}/05_build_episodes.log"
fi

# Step 6: verl GRPO parquets (train / val, seen versions only) with the same prompt as the episode evaluation
if run_step 6; then
  python preprocess/build_data/06_build_rl_parquet.py \
    --episodes-dir "${PROCESSED_DIR}/episodes/${SCORE_MODEL_NAME}" \
    --instruction-prompt-file preprocess/prompts/instruction_prompt.txt \
    --output-dir "${PROCESSED_DIR}/rl_parquet/${SCORE_MODEL_NAME}" \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/06_build_rl_parquet.log"
fi
