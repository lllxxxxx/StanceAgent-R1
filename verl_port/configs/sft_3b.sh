#!/usr/bin/env bash
# SFT cold start with LoRA. Usage: sft_3b.sh {csd|baseline} <save_dir> [overrides...]
set -euo pipefail

PIPELINE="${1:?need pipeline arg: csd or baseline}"
SAVE_DIR="${2:?need save dir, e.g. outputs/baseline_sft}"
shift 2 || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/verl_port/data}"

case "$PIPELINE" in
  csd)      MAX_LEN=3072; LORA_RANK=64; LORA_ALPHA=64; LR=5e-5 ;;
  baseline) MAX_LEN=2048; LORA_RANK=32; LORA_ALPHA=64; LR=5e-5 ;;
  *) echo "unknown pipeline: $PIPELINE" >&2; exit 1 ;;
esac

NPROC="${NPROC:-1}"
# Reduce allocator fragmentation on a single GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.sft_trainer \
    data.train_files="$DATA_DIR/${PIPELINE}_sft_train.parquet" \
    data.val_files="$DATA_DIR/${PIPELINE}_sft_train.parquet" \
    data.messages_key=messages \
    data.max_length=$MAX_LEN \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=$((MAX_LEN * 2)) \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=$LR \
    engine=fsdp \
    engine.param_offload=True \
    engine.optimizer_offload=True \
    model.path="$MODEL_PATH" \
    model.use_remove_padding=False \
    +model.override_config.attn_implementation=sdpa \
    model.lora_rank=$LORA_RANK \
    model.lora_alpha=$LORA_ALPHA \
    model.target_modules=all-linear \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.project_name=stance_agent_r1 \
    trainer.experiment_name="verl_${PIPELINE}_sft_3b" \
    trainer.logger=console \
    trainer.total_epochs=1 \
    "$@"
