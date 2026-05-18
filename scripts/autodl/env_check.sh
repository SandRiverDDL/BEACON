#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

cd "${BEACON_DIR:-$(autodl_repo_root)}"
autodl_validate_paths

echo "== paths =="
printf 'BEACON_DIR=%s\n' "${BEACON_DIR}"
printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"
printf 'SFT_ADAPTER_PATH=%s\n' "${SFT_ADAPTER_PATH}"
printf 'DATA_ROOT=%s\n' "${DATA_ROOT}"
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
printf 'OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf 'LORA_ADAPTER_ROOT=%s\n' "${LORA_ADAPTER_ROOT}"
printf 'WEBSHOP_INDEX_PATH=%s\n' "${WEBSHOP_INDEX_PATH}"

echo
echo "== gpu =="
nvidia-smi || true

echo
echo "== python/imports =="
"${BEACON_DIR}/.venv/bin/python" - <<'PY'
import os
import torch
import flash_attn
import vllm
from peft import PeftConfig
from transformers import AutoConfig, AutoTokenizer

model_path = os.environ["MODEL_PATH"]
adapter_path = os.environ["SFT_ADAPTER_PATH"]
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
peft_cfg = PeftConfig.from_pretrained(adapter_path)

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
print("flash_attn", flash_attn.__version__)
print("vllm", vllm.__version__)
print("tokenizer", tok.__class__.__name__, tok.vocab_size)
print("model", cfg.model_type, getattr(cfg, "hidden_size", None))
print("adapter", peft_cfg.peft_type, "rank", getattr(peft_cfg, "r", None))
PY

echo
echo "== webshop index =="
du -sh "${WEBSHOP_INDEX_PATH}"
find "${WEBSHOP_INDEX_PATH}" -maxdepth 1 -type f | wc -l

echo
echo "== disk =="
df -h "${BEACON_DIR}" /root/autodl-tmp 2>/dev/null || df -h "${BEACON_DIR}"
