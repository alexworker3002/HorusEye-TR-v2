# HorusEye-TR v2

Executable implementation of the HorusEye-TR stable dual-feedback Noisier-to-Noisy self-supervised denoising model described in `references/horuseye_tr_theory_zh_v2.0.tex`.

## Directory layout

- `scripts/`: training and inference entry points.
- `data/`: local CT zarr volumes. Each CT domain can use its own volume naming convention.
- `output/<training_script>/`: default run root for each training entry point.
- `output/<training_script>/checkpoints/`: model checkpoints.
- `output/<training_script>/logs/`: CSV metrics and run configuration.
- `output/<training_script>/figures/`: loss/metric curves and qualitative training samples.
- `output/<training_script>/inference_scripts/`: snapshots of the training, inference, and HR-evaluation scripts used by the run, plus run-specific launchers.
- `output/<training_script>/inference/<inference_script>/`: denoised slices and comparison figures from an inference script.

## Repository usage logic

The repository is organized around repeatable training runs:

1. Put CT data under `data/`.
2. Select a domain by passing a volume glob such as `--volume-glob "001-Femur/*.zarr"` or `--volume-glob "002-Vertebrae/*.zarr"`.
3. Run a training entry point from `scripts/`. By default, the script stem becomes the output run directory. For example, `scripts/train_horuseye_tr.py` writes to `output/train_horuseye_tr/`.
4. Keep run artifacts inside that output directory: checkpoints, logs, figures, script snapshots, and inference results.
5. Run inference from the saved checkpoint, writing results under the same run directory, for example `output/train_horuseye_tr/inference/infer_horuseye_tr/`.

This means a new domain or experiment can either reuse `scripts/train_horuseye_tr.py` with a custom `--output-root`, or get its own training script name so it naturally writes to `output/<that_script_name>/`.

## Data layout and domains

The default loader expects each volume root to contain a noisy/REG zarr array:

```text
data/
  001-Femur/
    Femur_15_80kV_ome.zarr/
      REG/0/
        zarr.json
      HR/2/
        zarr.json
  002-Vertebrae/
    Vertebrae_A_80kV_ome.zarr/
      REG/0/
        zarr.json
      HR/2/
        zarr.json
```

Some volumes also contain other scales, for example:

```text
Femur_15_80kV_ome.zarr/
  HR/0/
    zarr.json
  HR/1/
    zarr.json
  HR/2/
    zarr.json
  HR/3/
    zarr.json
  REG/0/
    zarr.json
  REG/1/
    zarr.json
```

The default discovery pattern is:

```text
--data-root data
--volume-glob "**/*.zarr"
--reg-subpath "REG/0"
--hr-subpath "HR/2"
```

For the current organized datasets, `REG/0` matches `HR/2`, and `REG/1` matches `HR/3`. The predictor always samples its three adjacent slices from the selected REG training set. The denoiser target defaults to the matching HR array, and HR evaluation uses the same matching HR array as supervised reference.

For a specific current labeled dataset, select that label folder:

```bash
python scripts/train_horuseye_tr.py \
  --data-root data \
  --volume-glob "001-Femur/*.zarr" \
  --reg-subpath "REG/0" \
  --hr-subpath "HR/2" \
  --denoiser-target hr \
  --output-root output/train_001_femur_horuseye
```

For another CT domain, keep it under `data/` and adjust the glob/subpaths to that domain's naming scheme.

Training needs at least two REG volumes because predictor residuals are injected across different volumes. With the default `--denoiser-target hr`, every selected REG volume must have a matching HR array with the same shape. Use `--denoiser-target reg` only for a REG-only self-supervised ablation.

## Quick start

```bash
pip install -r requirements.txt
python scripts/train_horuseye_tr.py \
  --data-root data \
  --volume-glob "001-Femur/*.zarr" \
  --reg-subpath "REG/0" \
  --hr-subpath "HR/2" \
  --denoiser-target hr \
  --output-root output/train_horuseye_tr_strict_gate \
  --device cuda \
  --batch-size 4 \
  --patch-size 128 \
  --samples-per-epoch 2048 \
  --epochs-stage1 2 \
  --epochs-stage2 1 \
  --epochs-stage3 10 \
  --lr-denoiser 2e-4 \
  --lr-predictor 5e-5 \
  --xi-low 0.10 \
  --xi-corr 0.04 \
  --structure-blur-kernel-size 9 \
  --structure-blur-sigma 2.0 \
  --highpass-kernel-size 9 \
  --highpass-sigma 2.0 \
  --gate-blur-kernel-size 11 \
  --gate-blur-sigma 2.5 \
  --num-workers 0 \
  --save-sample-every 1
```

By default, training uses `--num-workers 0`, which avoids CUDA/fork DataLoader worker crashes on local GPU runs. If you explicitly set `--num-workers > 0` with CUDA, the script uses spawned persistent workers and reopens zarr volumes inside each worker.

The root-level `train_horuseye_tr.py` remains as a compatibility wrapper:

```bash
python train_horuseye_tr.py --data-root data --output-root output/train_horuseye_tr_strict_gate --device cuda
```

## Outputs during training

Training writes:

- `output/<training_script>/logs/train_metrics.csv`: per-step `loss_d`, `loss_p`, accepted residual ratio, injection `alpha`, residual statistics.
- `output/<training_script>/logs/train_config.json`: CLI arguments and resolved model configuration for the run.
- `output/<training_script>/figures/train_metrics.png`: visualized loss and key statistics.
- `output/<training_script>/figures/sample_stage*_epoch*.png`: qualitative panels showing noisy input, injected input, denoised output, EMA anchor, and residual.
- `output/<training_script>/checkpoints/horuseye_tr_stage*_epoch*.pt`: resumable per-epoch checkpoints.
- `output/<training_script>/checkpoints/horuseye_tr_latest.pt`: latest resumable checkpoint.
- `output/<training_script>/checkpoints/horuseye_tr_final.pt`: final checkpoint.
- `output/<training_script>/inference_scripts/*.py`: script snapshots for later inference or evaluation from the same run context.
- `output/<training_script>/inference_scripts/run_infer_horuseye_tr.sh`: launcher for denoising with the run's final checkpoint.
- `output/<training_script>/inference_scripts/run_eval_horuseye_tr_hr.sh`: launcher for HR evaluation with the run's data-discovery settings.

## Resume Training

Resume from the latest checkpoint:

```bash
python scripts/train_horuseye_tr.py --data-root data --device cuda --resume-latest
```

Resume from a specific checkpoint:

```bash
python scripts/train_horuseye_tr.py \
  --data-root data \
  --device cuda \
  --resume-checkpoint output/train_horuseye_tr/checkpoints/horuseye_tr_stage3_epoch9.pt
```

## Inference

```bash
python scripts/infer_horuseye_tr.py \
  --checkpoint output/train_horuseye_tr/checkpoints/horuseye_tr_final.pt \
  --data-root data \
  --volume-glob "001-Femur/*.zarr" \
  --reg-subpath "REG/0" \
  --output-dir output/train_horuseye_tr/inference/infer_horuseye_tr \
  --device cuda
```

## HR Evaluation

Evaluate denoising against paired matching HR volumes before deciding whether to extend a run:

```bash
python scripts/eval_horuseye_tr_hr.py \
  --checkpoint output/train_horuseye_tr_strict_gate_256/checkpoints/horuseye_tr_final.pt \
  --data-root data \
  --volume-glob "001-Femur/*.zarr" \
  --reg-subpath "REG/0" \
  --hr-subpath "HR/2" \
  --output-dir output/train_horuseye_tr_strict_gate_256/eval/hr_final \
  --device cuda \
  --slices 160 224 256 288 352 \
  --save-comparisons 16
```

The script writes:

- `hr_eval_metrics.csv`: per-volume/slice metrics for REG baseline, predictor `P(z-1,z+1)`, denoiser `D(REG)`, and joint output `D(P)`.
- `hr_eval_summary.json`: aggregate PSNR/SSIM/MAE/RMSE deltas.
- `comparisons/*.png`: REG, predictor, denoised, joint, HR, removed signal, and error maps.

Treat a run as promising only if `delta_psnr_den`, `delta_psnr_pred`, `delta_psnr_joint` and the matching SSIM deltas are consistently positive, and the comparison figures do not show obvious anatomical structure in `REG - D(REG)`, `P - REG`, or `D(P) - D(REG)`.

The training loader discovers volumes using `--data-root`, `--volume-glob`, and `--reg-subpath`, then samples axial slice triplets. Inference uses only the denoiser `D_theta(x)`.
