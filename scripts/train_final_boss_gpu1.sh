#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec "${SUPRA_PYTHON:-/ml2/ntc-image-studio/.venv-anima/bin/python}" \
  -u scripts/train_final_boss.py --device cuda:0 "$@"
