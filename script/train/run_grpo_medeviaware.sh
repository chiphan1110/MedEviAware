#!/usr/bin/env bash
# GRPO for MedEviAware: answer / request / defer with reasoning, single turn, from Qwen3-4B-Instruct-2507.
# Reward: verl/verl/utils/reward_score/medeviaware_reward.py. Data: processed_data/rl_parquet (step 6 of build_data.sh).
set -xeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Repo-root .env (ROOT, BASE_MODEL_DIR, RESULTS_DIR, LOG_DIR, WANDB_DIR)
source "${ROOT_DIR}/.env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-verl}"

# Run verl from the repo copy, which holds the MedEviAware reward
export PYTHONPATH="${ROOT_DIR}/verl:${PYTHONPATH:-}"
export WANDB_MODE=offline
export WANDB_DIR=$WANDB_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export HYDRA_FULL_ERROR=1

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-"medeviaware_rl"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_qwen3_4b_instruct"}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}
# The FSDP actor of a 4B model takes ~16 GB per GPU over 4 GPUs; vLLM gets this share of each 48 GB card
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL_DIR}"}

CKPTS_DIR=${CKPTS_DIR:-"$RESULTS_DIR/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
ROLLOUT_DIR=${ROLLOUT_DIR:-"$RESULTS_DIR/rollouts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
VAL_DIR=${VAL_DIR:-"$RESULTS_DIR/val-generation/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
PARQUET_DIR=${PARQUET_DIR:-"${ROOT_DIR}/processed_data/rl_parquet/qwen3_4b_instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$PARQUET_DIR/train.parquet"}
TEST_FILE=${TEST_FILE:-"$PARQUET_DIR/val.parquet"}

REWARD_FILE=${REWARD_FILE:-"${ROOT_DIR}/verl/verl/utils/reward_score/medeviaware_reward.py"}
REWARD_FN=${REWARD_FN:-compute_score}

# Longest prompt is ~1,700 tokens (step 6 prints the max); responses use the same 2,048 limit as scoring and eval
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKENS=${PPO_MAX_TOKENS:-16384}
LOGPROB_MAX_TOKENS=${LOGPROB_MAX_TOKENS:-32768}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
########################### end user-adjustable ###########################

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p "$LOG_DIR"

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # Dr. GRPO advantages: no per-group std normalization
    algorithm.norm_adv_by_std_in_grpo=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.seed=42
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    # No KL: the reference model is not built
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKENS}
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NDEVICES_PER_NODE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # Also save a HuggingFace copy, which the episode evaluation hosts with vLLM
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']
    actor_rollout_ref.actor.checkpoint.load_contents=['model','optimizer','extra']
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKENS}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    # Unset, verl uses the model's max_position_embeddings (262,144 for Qwen3-4B-Instruct-2507): 36 GiB of KV cache
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    # Greedy validation, as the episode evaluation (temperature 0)
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.n=1
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console','wandb']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=True
    trainer.resume_mode=auto
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.rollout_data_dir=${ROLLOUT_DIR}
    trainer.validation_data_dir=${VAL_DIR}
    trainer.val_before_train=True
    trainer.log_val_generations=10
    trainer.save_freq=10
    trainer.test_freq=10
    trainer.max_actor_ckpt_to_keep=3
    trainer.total_epochs=${TOTAL_EPOCHS}
)

REWARD=(
    reward.custom_reward_function.path="${REWARD_FILE}"
    reward.custom_reward_function.name="${REWARD_FN}"
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${TRAINER[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "$@" 2>&1 | tee "$LOG_DIR/${EXPERIMENT_NAME}-${start_time}.log"
