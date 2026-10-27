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

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/processed_data}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
API_MODEL="${API_MODEL:-base}"                 # must match SERVED_MODEL_NAME in host_model.sh
MODEL_NAME="${MODEL_NAME:-${API_MODEL}}"       # label for the output folder, e.g. qwen3_4b_instruct
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
WORKERS="${WORKERS:-64}"
LIMIT_CASES="${LIMIT_CASES:-0}"                # e.g. LIMIT_CASES=100 for the pilot
PROMPTS_JSONL="${PROMPTS_JSONL:-${PROCESSED_DIR}/score/score_prompts.jsonl}"
OUT_DIR="${OUT_DIR:-${PROCESSED_DIR}/score/${MODEL_NAME}}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

declare -a ARGS=(
  --prompts-jsonl "${PROMPTS_JSONL}"
  --out-dir "${OUT_DIR}"
  --api-model "${API_MODEL}"
  --base-url "${BASE_URL}"
  --workers "${WORKERS}"
  --limit-cases "${LIMIT_CASES}"
)

if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_EVAL_ARGS} )
  ARGS+=("${EXTRA_ARGS[@]}")
fi

python eval/score_p_vllm.py "${ARGS[@]}" 2>&1 | tee "${LOG_DIR}/score_p_${MODEL_NAME}.log"
