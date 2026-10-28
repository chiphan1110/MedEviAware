#!/usr/bin/env bash
# Make a verl checkpoint servable by the vllm env: <step>/actor/huggingface_vllm holds links to the trained
# weights plus the base model's config and tokenizer files. The training env's newer transformers writes
# config.json / tokenizer_config.json in a layout the vllm env misreads (wrong RoPE setting -> looping replies).
# Training only changes the weights, so the base model's files are the right ones.
#   bash script/eval/make_vllm_ckpt.sh results/checkpoints/medeviaware_rl/<experiment>/global_step_20
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

STEP_DIR="${1:?usage: make_vllm_ckpt.sh <checkpoint global_step_N folder>}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:?BASE_MODEL_DIR must be set in .env}"
SRC_DIR="$(cd "${STEP_DIR}/actor/huggingface" && pwd)"
OUT_DIR="${STEP_DIR}/actor/huggingface_vllm"

shopt -s nullglob
WEIGHTS=("${SRC_DIR}"/*.safetensors)
if [[ ${#WEIGHTS[@]} -eq 0 ]]; then
  echo "No *.safetensors in ${SRC_DIR} (was the checkpoint saved with hf_model?)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
# Trained weights (and their shard index, if verl split them), linked rather than copied
for f in "${WEIGHTS[@]}" "${SRC_DIR}"/*.safetensors.index.json; do
  ln -sf "${f}" "${OUT_DIR}/$(basename "${f}")"
done
# Base model config and tokenizer; never its weight index, which names shards that are not here
for name in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  if [[ -f "${BASE_MODEL_DIR}/${name}" ]]; then
    cp "${BASE_MODEL_DIR}/${name}" "${OUT_DIR}/${name}"
  fi
done

echo "vLLM-ready checkpoint: $(cd "${OUT_DIR}" && pwd)"
ls -l "${OUT_DIR}"
