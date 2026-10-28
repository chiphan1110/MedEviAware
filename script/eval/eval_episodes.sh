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
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
API_MODEL="${API_MODEL:-base}"                 # must match SERVED_MODEL_NAME in host_model.sh
MODEL_NAME="${MODEL_NAME:-${API_MODEL}}"       # label for the output folder
LABEL_NAME="${LABEL_NAME:-qwen3_4b_instruct}"  # which labels the episodes were built from (processed_data/episodes/<LABEL_NAME>/)
SPLIT="${SPLIT:-test}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
WORKERS="${WORKERS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_TURNS="${MAX_TURNS:-3}"
LIMIT="${LIMIT:-0}"                            # e.g. LIMIT=20 for a quick check
INSTRUCTION_PROMPT_FILE="${INSTRUCTION_PROMPT_FILE:-preprocess/prompts/instruction_prompt.txt}"
EPISODES_JSONL="${EPISODES_JSONL:-${PROCESSED_DIR}/episodes/${LABEL_NAME}/${SPLIT}.jsonl}"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/eval_episodes/${MODEL_NAME}/${SPLIT}}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

# Fail fast if the vLLM server is not up or does not serve API_MODEL
if ! MODELS_JSON="$(curl -sf --noproxy '*' "${BASE_URL}/models")"; then
  echo "No vLLM server at ${BASE_URL}. Start it with script/eval/host_model.sh and wait for 'Application startup complete'." >&2
  exit 1
fi
if [[ "${MODELS_JSON}" != *"\"id\":\"${API_MODEL}\""* ]]; then
  echo "Server at ${BASE_URL} does not serve '${API_MODEL}'. It serves: ${MODELS_JSON}" >&2
  exit 1
fi

declare -a ARGS=(
  --episodes-jsonl "${EPISODES_JSONL}"
  --instruction-prompt-file "${INSTRUCTION_PROMPT_FILE}"
  --out-dir "${OUT_DIR}"
  --api-model "${API_MODEL}"
  --base-url "${BASE_URL}"
  --workers "${WORKERS}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --max-turns "${MAX_TURNS}"
  --limit "${LIMIT}"
)

if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_EVAL_ARGS} )
  ARGS+=("${EXTRA_ARGS[@]}")
fi

python eval/eval_episodes_vllm.py "${ARGS[@]}" 2>&1 | tee "${LOG_DIR}/eval_episodes_${MODEL_NAME}_${SPLIT}.log"
