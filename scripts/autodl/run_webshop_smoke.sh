#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

cd "${BEACON_DIR:-$(autodl_repo_root)}"
autodl_validate_paths

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VERL_SEPARATE_LORA_ADAPTER_DIR="${VERL_SEPARATE_LORA_ADAPTER_DIR:-${LORA_ADAPTER_ROOT}/smoke}"

ENGINE="${ENGINE:-vllm}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-autodl_webshop_smoke}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-1}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-1}"
GROUP_SIZE="${GROUP_SIZE:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
LOGGER="${LOGGER:-['console']}"
RESUME_MODE="${RESUME_MODE:-disable}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"

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
