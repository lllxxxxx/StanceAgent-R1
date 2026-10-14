#!/usr/bin/env bash
# GRPO with LoRA. Usage: grpo_3b.sh {csd|baseline} [overrides...]
set -euo pipefail

PIPELINE="${1:?need pipeline arg: csd or baseline, then optional hydra overrides}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/verl_port/data}"

case "$PIPELINE" in
  csd)
    REWARD_FILE="$REPO_ROOT/verl_port/csd_reward.py"
    MAX_PROMPT=2048; MAX_RESP=896
    LORA_RANK=32; LORA_ALPHA=64
    ;;
  baseline)
    REWARD_FILE="$REPO_ROOT/verl_port/baseline_reward.py"
    MAX_PROMPT=1536; MAX_RESP=320
    LORA_RANK=32; LORA_ALPHA=64
    ;;
  *) echo "unknown pipeline: $PIPELINE" >&2; exit 1 ;;
esac

export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
# expandable_segments is incompatible with vLLM's colocated memory pool.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/${PIPELINE}_grpo_train.parquet" \
    data.val_files="$DATA_DIR/${PIPELINE}_dev.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESP \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESP)) \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="$REWARD_FILE" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=stance_agent_r1 \
    trainer.experiment_name="verl_${PIPELINE}_grpo_3b" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    "$@"
