#!/usr/bin/env bash
# DAPO variant: asymmetric clipping and no KL loss.
# Usage: dapo_3b.sh {csd|baseline} [overrides...]
set -euo pipefail

PIPELINE="${1:?need pipeline arg: csd or baseline}"
shift || true
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$DIR/grpo_3b.sh" "$PIPELINE" \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    trainer.experiment_name="verl_${PIPELINE}_dapo_3b" \
    "$@"
