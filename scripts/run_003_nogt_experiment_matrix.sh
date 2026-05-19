#!/usr/bin/env bash
set -euo pipefail

cd /home/ice/workspace/HorusEye-TR-v2

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_ROOT="data"
VOLUME_GLOB="003-COVID-19/*.zarr"
REG_SUBPATH="REG/0"
HR_SUBPATH="HR/0"
SLICES=(128 192 256 320 384)
MAX_EVAL_VOLUMES=64
SAVE_COMPARISONS=24

COMMON_TRAIN_ARGS=(
  --data-root "$DATA_ROOT"
  --volume-glob "$VOLUME_GLOB"
  --reg-subpath "$REG_SUBPATH"
  --hr-subpath "$HR_SUBPATH"
  --device cuda
  --batch-size 32
  --patch-size 256
  --samples-per-epoch 4096
  --epochs-stage1 10
  --epochs-stage2 50
  --epochs-stage3 50
  --lr-denoiser 1e-4
  --lr-predictor 1e-4
  --xi-low 0.12
  --xi-corr 0.05
  --structure-blur-kernel-size 9
  --structure-blur-sigma 2.0
  --highpass-kernel-size 9
  --highpass-sigma 2.0
  --gate-blur-kernel-size 11
  --gate-blur-sigma 2.5
  --num-workers 8
  --save-sample-every 10
)

CONTROL_OUT="output/train_003_covid19_x012_c005_predboost50_p256_s2e50"
SELFSUP_OUT="output/train_003_covid19_x012_c005_selfsup_regtarget_p256_s2e50"
MIXED_OUT="output/train_003_covid19_x012_c005_mixed_hrref010_p256_s2e50"
MATRIX_ROOT="output/experiment_003_nogt_matrix"

mkdir -p "$MATRIX_ROOT/logs"

run_train_if_needed() {
  local name="$1"
  local out="$2"
  shift 2
  if [[ -f "$out/checkpoints/horuseye_tr_final.pt" ]]; then
    echo "[$name] final checkpoint exists, skipping training: $out"
    return
  fi
  if [[ -e "$out" ]]; then
    echo "[$name] output exists without final checkpoint; refusing to overwrite: $out" >&2
    exit 1
  fi
  echo "[$name] training -> $out"
  "$PYTHON_BIN" scripts/train_horuseye_tr.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --output-root "$out" \
    "$@" \
    2>&1 | tee "$MATRIX_ROOT/logs/${name}_train.log"
}

run_evals() {
  local name="$1"
  local out="$2"
  local ckpt="$out/checkpoints/horuseye_tr_final.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$name] missing final checkpoint: $ckpt" >&2
    exit 1
  fi

  echo "[$name] no-GT evaluation"
  "$PYTHON_BIN" scripts/eval_horuseye_tr_nogt.py \
    --checkpoint "$ckpt" \
    --data-root "$DATA_ROOT" \
    --volume-glob "$VOLUME_GLOB" \
    --reg-subpath "$REG_SUBPATH" \
    --output-dir "$out/eval/nogt_reg_m64" \
    --device cuda \
    --slices "${SLICES[@]}" \
    --max-volumes "$MAX_EVAL_VOLUMES" \
    --save-comparisons "$SAVE_COMPARISONS" \
    2>&1 | tee "$MATRIX_ROOT/logs/${name}_eval_nogt.log"

  echo "[$name] HR-reference similarity evaluation"
  "$PYTHON_BIN" scripts/eval_horuseye_tr_hr.py \
    --checkpoint "$ckpt" \
    --data-root "$DATA_ROOT" \
    --volume-glob "$VOLUME_GLOB" \
    --reg-subpath "$REG_SUBPATH" \
    --hr-subpath "$HR_SUBPATH" \
    --output-dir "$out/eval/hr_reference_m64" \
    --device cuda \
    --slices "${SLICES[@]}" \
    --max-volumes "$MAX_EVAL_VOLUMES" \
    --save-comparisons "$SAVE_COMPARISONS" \
    2>&1 | tee "$MATRIX_ROOT/logs/${name}_eval_hr_reference.log"
}

run_train_if_needed "selfsup_regtarget" "$SELFSUP_OUT" \
  --denoiser-target reg \
  --hr-reference-weight 0.0

run_train_if_needed "mixed_hrref010" "$MIXED_OUT" \
  --denoiser-target reg \
  --hr-reference-weight 0.10

run_evals "control_hrtarget" "$CONTROL_OUT"
run_evals "selfsup_regtarget" "$SELFSUP_OUT"
run_evals "mixed_hrref010" "$MIXED_OUT"

"$PYTHON_BIN" - <<'PY'
import csv
import json
from pathlib import Path

experiments = {
    "control_hrtarget": Path("output/train_003_covid19_x012_c005_predboost50_p256_s2e50"),
    "selfsup_regtarget": Path("output/train_003_covid19_x012_c005_selfsup_regtarget_p256_s2e50"),
    "mixed_hrref010": Path("output/train_003_covid19_x012_c005_mixed_hrref010_p256_s2e50"),
}
matrix_root = Path("output/experiment_003_nogt_matrix")
rows = []
for name, root in experiments.items():
    nogt = json.loads((root / "eval/nogt_reg_m64/nogt_eval_summary.json").read_text())
    href = json.loads((root / "eval/hr_reference_m64/hr_eval_summary.json").read_text())
    rows.append({
        "experiment": name,
        "checkpoint": str(root / "checkpoints/horuseye_tr_final.pt"),
        "nogt_edge_retention_den": nogt["edge_retention_den"]["mean"],
        "nogt_laplacian_std_ratio_den": nogt["laplacian_std_ratio_den"]["mean"],
        "nogt_residual_lowfreq_fraction_den": nogt["residual_lowfreq_fraction_den"]["mean"],
        "nogt_residual_edge_corr_den": nogt["residual_edge_corr_den"]["mean"],
        "nogt_mean_abs_change_den": nogt["mean_abs_change_den"]["mean"],
        "nogt_edge_retention_joint": nogt["edge_retention_joint"]["mean"],
        "nogt_laplacian_std_ratio_joint": nogt["laplacian_std_ratio_joint"]["mean"],
        "hrref_delta_psnr_den": href["delta_psnr_den"]["mean"],
        "hrref_delta_ssim_den": href["delta_ssim_den"]["mean"],
        "hrref_delta_psnr_joint": href["delta_psnr_joint"]["mean"],
        "hrref_delta_ssim_joint": href["delta_ssim_joint"]["mean"],
    })

matrix_root.mkdir(parents=True, exist_ok=True)
with (matrix_root / "summary.json").open("w") as f:
    json.dump(rows, f, indent=2)
with (matrix_root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY

echo "Experiment matrix complete: $MATRIX_ROOT"
