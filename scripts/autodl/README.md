# AutoDL WebShop Launchers

这些脚本只用于 AutoDL 服务器，目的是把训练输入、checkpoint、LoRA adapter 和日志固定在当前 BEACON checkout 内，避免误写实验室服务器路径。

## 运行顺序

```bash
cd /root/autodl-tmp/projects/BEACON
source .env.autodl

scripts/autodl/env_check.sh
scripts/autodl/run_webshop_smoke.sh
scripts/autodl/run_webshop_rl.sh
```

## 关键路径

- `MODEL_PATH`：远端 HF cache 里的 Qwen2.5-1.5B-Instruct snapshot。
- `SFT_ADAPTER_PATH`：`$BEACON_DIR/saves/qwen2_5_1p5b_lora_step_sft`。
- `DATA_ROOT`：`$BEACON_DIR/data/autodl/verl-agent`。
- `RUN_ROOT`：`$BEACON_DIR/checkpoints/autodl`。
- `LORA_ADAPTER_ROOT`：`$BEACON_DIR/saves/autodl_lora_adapters`。

脚本会拒绝 `/home/fengshuwen`、`/mnt/dataset`、`/home/shenyl` 等非 AutoDL 路径。

## 常用覆盖

```bash
TOTAL_TRAINING_STEPS=3 scripts/autodl/run_webshop_rl.sh
RESUME_MODE=auto TOTAL_TRAINING_STEPS=25 scripts/autodl/run_webshop_rl.sh
LOGGER="['console']" scripts/autodl/run_webshop_rl.sh
```
