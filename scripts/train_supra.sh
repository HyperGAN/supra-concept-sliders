#!/usr/bin/env bash
# Call the Supra2-IMG slider trainer in a particle-sliders checkout.
# This product repo does not contain the DiT trainer.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRODUCT="$(cd "$HERE/.." && pwd)"
BACKEND="${PARTICLE_SLIDERS_ROOT:?Set PARTICLE_SLIDERS_ROOT to a checkout of HyperGAN/particle-sliders at the commit in backend.lock.json (PR 130).}"
TRAINER="$BACKEND/conceptmod/textsliders/train_lora_supra.py"

if [[ ! -f "$TRAINER" ]]; then
  echo "train_lora_supra.py not found at $TRAINER" >&2
  echo "Clone HyperGAN/particle-sliders and check out the commit pinned in backend.lock.json." >&2
  exit 1
fi

PROMPTS="${PROMPTS_FILE:-$PRODUCT/data/prompts-supra.yaml}"
NAME="${NAME:-lighting-supra}"
SAVE_DIR="${SAVE_DIR:-$PRODUCT/artifacts/lighting-supra}"

args=(
  --name "$NAME"
  --prompts_file "$PROMPTS"
  --model_id SupraLabs/Supra2-IMG
  --lora_targets cross
  --rank 16
  --resolution 256
  --sample_steps 50
  --cfg 3
  --lr 1e-4
  --lm_target trajectory
  --traj_steps 4
  --sample_every 100
  --save_dir "$SAVE_DIR"
)

if [[ "${DUMMY:-}" == "1" ]]; then
  args+=(--dummy --steps 8 --device cpu)
else
  args+=(--device "${DEVICE:-cuda:0}")
  if [[ -n "${CHECKPOINT:-}" ]]; then
    args+=(--checkpoint "$CHECKPOINT")
  fi
fi

cd "$BACKEND"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${BACKEND}${PYTHONPATH:+:$PYTHONPATH}"
exec python "$TRAINER" "${args[@]}" "$@"
