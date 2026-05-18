#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

cd "${BEACON_DIR:-$(autodl_repo_root)}"
autodl_validate_paths

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

ENGINE="${ENGINE:-vllm}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-autodl_migpo_qwen2_5_1p5b_sft_lora_seed${SEED:-1025}}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-3}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-50}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
LOGGER="${LOGGER:-['console','wandb']}"
RESUME_MODE="${RESUME_MODE:-disable}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export SEED="${SEED:-1025}"
export LR="${LR:-2e-6}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export MAX_ENV_STEPS="${MAX_ENV_STEPS:-15}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
export VERL_SEPARATE_LORA_ADAPTER_DIR="${VERL_SEPARATE_LORA_ADAPTER_DIR:-${LORA_ADAPTER_ROOT}/${EXPERIMENT_NAME}}"
export VERL_WANDB_METRIC_ALLOWLIST="${VERL_WANDB_METRIC_ALLOWLIST:-actor/entropy_loss,actor/pg_loss,actor/clipfrac,actor/kl_loss,critic/score/mean,critic/rewards/mean,env/success_rate,env/webshop_task_score*,val/success_rate,train/global_step}"

mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}" "${LORA_ADAPTER_ROOT}"
autodl_prepare_data "${TRAIN_DATA_SIZE}" "$((VAL_DATA_SIZE * 2))"

mapfile -t OVERRIDES < <(autodl_common_ppo_overrides \
  "${ENGINE}" \
  "${EXPERIMENT_NAME}" \
  "${TRAIN_DATA_SIZE}" \
  "${VAL_DATA_SIZE}" \
  "${GROUP_SIZE}" \
  "${TOTAL_TRAINING_STEPS}" \
  "${LOGGER}" \
  "${RESUME_MODE}" \
  "${SAVE_FREQ}" \
  "${TEST_FREQ}" \
  "${VAL_BEFORE_TRAIN}")

"${BEACON_DIR}/.venv/bin/python" -m verl.trainer.main_ppo "${OVERRIDES[@]}" "$@"
