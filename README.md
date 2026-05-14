# HorusEye-TR v2

Executable implementation of the HorusEye-TR stable dual-feedback Noisier-to-Noisy self-supervised denoising model described in `references/horuseye_tr_theory_zh_v2.0.tex`.

## Directory layout

- `scripts/`: training and inference entry points.
- `output/<training_script>/`: default run root for each training entry point.
- `output/train_horuseye_tr/checkpoints/`: model checkpoints.
- `output/train_horuseye_tr/logs/`: CSV metrics and run configuration.
- `output/train_horuseye_tr/figures/`: loss/metric curves and qualitative training samples.
- `output/train_horuseye_tr/inference_scripts/`: optional place to keep inference scripts/config snapshots for the run.
- `output/train_horuseye_tr/inference/infer_horuseye_tr/`: denoised slices and comparison figures from `scripts/infer_horuseye_tr.py`.

## Quick start

```bash
pip install -r requirements.txt
python scripts/train_horuseye_tr.py \
  --data-root data \
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

- `output/train_horuseye_tr/logs/train_metrics.csv`: per-step `loss_d`, `loss_p`, accepted residual ratio, injection `alpha`, residual statistics.
- `output/train_horuseye_tr/figures/train_metrics.png`: visualized loss and key statistics.
- `output/train_horuseye_tr/figures/sample_stage*_epoch*.png`: qualitative panels showing noisy input, injected input, denoised output, EMA anchor, and residual.
- `output/train_horuseye_tr/checkpoints/horuseye_tr_stage*_epoch*.pt`: resumable per-epoch checkpoints.
- `output/train_horuseye_tr/checkpoints/horuseye_tr_latest.pt`: latest resumable checkpoint.
- `output/train_horuseye_tr/checkpoints/horuseye_tr_final.pt`: final checkpoint.

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
  --volume data/covid-disk1.zarr/REG/0 \
  --output-dir output/train_horuseye_tr/inference/infer_horuseye_tr \
  --device cuda
```

The training loader automatically discovers `data/covid-*.zarr/REG/0` volumes and samples axial slice triplets. Inference uses only the denoiser `D_theta(x)`.
