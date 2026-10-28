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
  conda activate "${CONDA_ENV_NAME:-vllm}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_DIR}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-base}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-4}"   # one replica per 4090
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"   # multi-turn episodes with reasoning
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_LOGPROBS="${MAX_LOGPROBS:-20}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"   # variants of a case share the same history prefix
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

declare -a CMD=(
  vllm serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-logprobs "${MAX_LOGPROBS}"
  --trust-remote-code
)

# Data parallel: one replica per GPU behind one port (CUDA_VISIBLE_DEVICES must list DP x TP GPUs)
if [[ "${DATA_PARALLEL_SIZE}" != "1" ]]; then
  CMD+=(--data-parallel-size "${DATA_PARALLEL_SIZE}")
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  CMD+=(--enable-prefix-caching)
fi

if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_VLLM_ARGS} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

exec "${CMD[@]}"
