#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

exec "$PYTHON_BIN" scripts/train_horuseye_tr.py \
  --data-root data \
  --volume-glob "001-Femur/*.zarr" \
  --reg-subpath REG/0 \
  --hr-subpath HR/2 \
  --output-root output/train_001_femur_x012_c005_hr_denoiser_p256_e50 \
  --device cuda \
  --batch-size 32 \
  --patch-size 256 \
  --samples-per-epoch 4096 \
  --epochs-stage1 4 \
  --epochs-stage2 1 \
  --epochs-stage3 50 \
  --lr-denoiser 2e-4 \
  --lr-predictor 3e-5 \
  --xi-low 0.12 \
  --xi-corr 0.05 \
  --structure-blur-kernel-size 9 \
  --structure-blur-sigma 2.0 \
  --highpass-kernel-size 9 \
  --highpass-sigma 2.0 \
  --gate-blur-kernel-size 11 \
  --gate-blur-sigma 2.5 \
  --denoiser-target hr \
  --num-workers 8 \
  --save-sample-every 1
