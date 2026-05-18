#!/usr/bin/env bash

set -euo pipefail

autodl_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/../.." && pwd
}

autodl_load_env() {
  export BEACON_DIR="${BEACON_DIR:-$(autodl_repo_root)}"
  if [[ -f "${BEACON_DIR}/.env.autodl" ]]; then
    # shellcheck disable=SC1091
    source "${BEACON_DIR}/.env.autodl"
  fi

  export BEACON_DIR="${BEACON_DIR:-$(autodl_repo_root)}"
  export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
  export SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${BEACON_DIR}/saves/qwen2_5_1p5b_lora_step_sft}"
  export DATA_ROOT="${DATA_ROOT:-${BEACON_DIR}/data/autodl/verl-agent}"
  export RUN_ROOT="${RUN_ROOT:-${BEACON_DIR}/checkpoints/autodl}"
  export OUTPUT_ROOT="${OUTPUT_ROOT:-${BEACON_DIR}/outputs/autodl}"
  export LORA_ADAPTER_ROOT="${LORA_ADAPTER_ROOT:-${BEACON_DIR}/saves/autodl_lora_adapters}"
  export WEBSHOP_INDEX_PATH="${WEBSHOP_INDEX_PATH:-${BEACON_DIR}/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k}"
  export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/root/autodl-tmp/cache/wandb}"
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
}

autodl_fail() {
  echo "[autodl][error] $*" >&2
  exit 1
}

autodl_require_file() {
  [[ -f "$1" ]] || autodl_fail "missing file: $1"
}

autodl_require_dir() {
  [[ -d "$1" ]] || autodl_fail "missing directory: $1"
}

autodl_reject_host_path() {
  local name="$1"
  local value="$2"
  case "${value}" in
    *"/home/fengshuwen"*|*"/mnt/dataset"*|*"/home/shenyl"*)
      autodl_fail "${name} points to a non-AutoDL host path: ${value}"
      ;;
  esac
}

autodl_assert_under() {
  local name="$1"
  local path="$2"
  local root="$3"
  case "$(realpath -m "${path}")" in
    "$(realpath -m "${root}")"/*|"$(realpath -m "${root}")") ;;
    *) autodl_fail "${name} must stay under ${root}, got ${path}" ;;
  esac
}

autodl_validate_paths() {
  autodl_load_env

  [[ "$(pwd)" == "$(realpath -m "${BEACON_DIR}")" ]] || autodl_fail "run from BEACON_DIR: ${BEACON_DIR}"

  for pair in \
    "BEACON_DIR=${BEACON_DIR}" \
    "MODEL_PATH=${MODEL_PATH}" \
    "SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH}" \
    "DATA_ROOT=${DATA_ROOT}" \
    "RUN_ROOT=${RUN_ROOT}" \
    "OUTPUT_ROOT=${OUTPUT_ROOT}" \
    "LORA_ADAPTER_ROOT=${LORA_ADAPTER_ROOT}" \
    "WEBSHOP_INDEX_PATH=${WEBSHOP_INDEX_PATH}"; do
    autodl_reject_host_path "${pair%%=*}" "${pair#*=}"
  done

  autodl_assert_under "DATA_ROOT" "${DATA_ROOT}" "${BEACON_DIR}/data"
  autodl_assert_under "RUN_ROOT" "${RUN_ROOT}" "${BEACON_DIR}/checkpoints"
  autodl_assert_under "OUTPUT_ROOT" "${OUTPUT_ROOT}" "${BEACON_DIR}/outputs"
  autodl_assert_under "LORA_ADAPTER_ROOT" "${LORA_ADAPTER_ROOT}" "${BEACON_DIR}/saves"

  autodl_require_dir "${MODEL_PATH}"
  autodl_require_file "${MODEL_PATH}/config.json"
  autodl_require_file "${MODEL_PATH}/tokenizer_config.json"
  autodl_require_dir "${SFT_ADAPTER_PATH}"
  autodl_require_file "${SFT_ADAPTER_PATH}/adapter_config.json"
  autodl_require_file "${SFT_ADAPTER_PATH}/adapter_model.safetensors"
  autodl_require_dir "${WEBSHOP_INDEX_PATH}"
}

autodl_prepare_data() {
  local train_size="$1"
  local val_size="$2"
  mkdir -p "${DATA_ROOT}"
  "${BEACON_DIR}/.venv/bin/python" -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "${DATA_ROOT}" \
    --train_data_size "${train_size}" \
    --val_data_size "${val_size}"
}

autodl_common_ppo_overrides() {
  local engine="$1"
  local experiment_name="$2"
  local train_size="$3"
  local val_size="$4"
  local group_size="$5"
  local total_steps="$6"
  local logger="$7"
  local resume_mode="$8"
  local save_freq="$9"
  local test_freq="${10}"
  local val_before_train="${11}"

  cat <<EOF
algorithm.adv_estimator=migpo
data.train_files=${DATA_ROOT}/text/train.parquet
data.val_files=${DATA_ROOT}/text/test.parquet
data.train_batch_size=${train_size}
data.val_batch_size=${val_size}
data.max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
data.max_response_length=${MAX_RESPONSE_LENGTH:-512}
data.filter_overlong_prompts=True
data.truncation=error
data.return_raw_chat=True
actor_rollout_ref.model.path=${MODEL_PATH}
actor_rollout_ref.model.lora_adapter_path=${SFT_ADAPTER_PATH}
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=64
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.optim.lr=${LR:-2e-6}
actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-1}
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS:-False}
actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0.0}
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:-0.001}
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.rollout.name=${engine}
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.55}
actor_rollout_ref.rollout.enable_chunked_prefill=False
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.free_cache_engine=False
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
actor_rollout_ref.rollout.val_kwargs.temperature=0.4
actor_rollout_ref.rollout.val_kwargs.do_sample=True
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
actor_rollout_ref.ref.fsdp_config.param_offload=True
actor_rollout_ref.actor.use_invalid_action_penalty=True
actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF:-1}
algorithm.use_kl_in_reward=False
algorithm.gamma=0.95
+algorithm.migpo.step_advantage_w=1
+algorithm.migpo.mode=mean_norm
+algorithm.migpo.gamma=0.95
+algorithm.migpo.threshold=0.95
env.env_name=Webshop
env.seed=${SEED:-1025}
env.max_steps=${MAX_ENV_STEPS:-15}
env.rollout.n=${group_size}
env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER:-0.1}
env.webshop.use_small=True
env.webshop.human_goals=False
env.webshop.train_start_idx=500
env.webshop.val_start_idx=0
env.webshop.val_end_idx=500
env.webshop.exclude_goal_indices_path=null
trainer.critic_warmup=0
trainer.logger=${logger}
trainer.project_name=${WANDB_PROJECT:-verl_agent_webshop}
trainer.experiment_name=${experiment_name}
trainer.n_gpus_per_node=1
trainer.nnodes=1
trainer.save_freq=${save_freq}
trainer.test_freq=${test_freq}
trainer.total_training_steps=${total_steps}
trainer.total_epochs=200
trainer.val_before_train=${val_before_train}
trainer.resume_mode=${resume_mode}
trainer.default_local_dir=${RUN_ROOT}/${experiment_name}
trainer.default_hdfs_dir=null
trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1}
trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-1}
EOF
}
