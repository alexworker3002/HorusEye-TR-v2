from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from horuseye_tr import HorusEyeConfig, HorusEyeTR, ZarrTripletDataset

TRAIN_SCRIPT_NAME = Path(__file__).stem
INFERENCE_SCRIPT_NAME = "infer_horuseye_tr"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / TRAIN_SCRIPT_NAME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HorusEye-TR stable dual-feedback self-supervised denoiser.")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--volume-glob", default="**/*.zarr", help="Glob under data-root selecting volume roots for this CT domain.")
    p.add_argument("--reg-subpath", default="REG/0", help="Relative path from each volume root to the noisy/REG zarr array.")
    p.add_argument("--hr-subpath", default="HR/2", help="Relative path from each volume root to the paired HR zarr array used as denoiser target.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--samples-per-epoch", type=int, default=2048)
    p.add_argument("--epochs-stage1", type=int, default=2)
    p.add_argument("--epochs-stage2", type=int, default=1)
    p.add_argument("--epochs-stage3", type=int, default=10)
    p.add_argument("--lr-denoiser", type=float, default=2e-4)
    p.add_argument("--lr-predictor", type=float, default=5e-5)
    p.add_argument("--xi-low", type=float, default=0.10)
    p.add_argument("--xi-corr", type=float, default=0.04)
    p.add_argument("--structure-blur-kernel-size", type=int, default=9)
    p.add_argument("--structure-blur-sigma", type=float, default=2.0)
    p.add_argument("--highpass-kernel-size", type=int, default=9)
    p.add_argument("--highpass-sigma", type=float, default=2.0)
    p.add_argument("--gate-blur-kernel-size", type=int, default=11)
    p.add_argument("--gate-blur-sigma", type=float, default=2.5)
    p.add_argument(
        "--denoiser-target",
        choices=("hr", "reg"),
        default="hr",
        help="Target image used for denoiser noise-injection training. Predictor triplets always come from REG.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-sample-every", type=int, default=1)
    p.add_argument("--resume-checkpoint", "--resume", type=Path, default=None, dest="resume_checkpoint")
    p.add_argument("--resume-latest", action="store_true")
    return p.parse_args()


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True).float() for k, v in batch.items()}


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "logs": output_root / "logs",
        "figures": output_root / "figures",
        "inference": output_root / "inference" / INFERENCE_SCRIPT_NAME,
        "inference_scripts": output_root / "inference_scripts",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def snapshot_run_scripts(dirs: dict[str, Path], args: argparse.Namespace) -> None:
    script_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts" / f"{INFERENCE_SCRIPT_NAME}.py",
        ROOT / "scripts" / "eval_horuseye_tr_hr.py",
    ]
    for src in script_paths:
        if src.exists():
            shutil.copy2(src, dirs["inference_scripts"] / src.name)

    repo_root = ROOT.resolve()
    run_root = dirs["root"].resolve()
    infer_launcher = dirs["inference_scripts"] / f"run_{INFERENCE_SCRIPT_NAME}.sh"
    infer_launcher.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(repo_root))}",
                '"${PYTHON_BIN:-python3}" '
                f"scripts/{INFERENCE_SCRIPT_NAME}.py "
                f"--checkpoint {shlex.quote(str(run_root / 'checkpoints' / 'horuseye_tr_final.pt'))} "
                f"--data-root {shlex.quote(str(args.data_root.resolve()))} "
                f"--volume-glob {shlex.quote(args.volume_glob)} "
                f"--reg-subpath {shlex.quote(args.reg_subpath)} "
                f"--output-dir {shlex.quote(str(run_root / 'inference' / INFERENCE_SCRIPT_NAME))} "
                '"$@"',
                "",
            ]
        )
    )
    infer_launcher.chmod(0o755)

    eval_launcher = dirs["inference_scripts"] / "run_eval_horuseye_tr_hr.sh"
    eval_launcher.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(repo_root))}",
                '"${PYTHON_BIN:-python3}" '
                "scripts/eval_horuseye_tr_hr.py "
                f"--checkpoint {shlex.quote(str(run_root / 'checkpoints' / 'horuseye_tr_final.pt'))} "
                f"--data-root {shlex.quote(str(args.data_root.resolve()))} "
                f"--volume-glob {shlex.quote(args.volume_glob)} "
                f"--reg-subpath {shlex.quote(args.reg_subpath)} "
                f"--hr-subpath {shlex.quote(args.hr_subpath)} "
                f"--output-dir {shlex.quote(str(run_root / 'eval' / 'hr_final'))} "
                '"$@"',
                "",
            ]
        )
    )
    eval_launcher.chmod(0o755)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def trim_metrics_csv(path: Path, max_step_exclusive: int) -> None:
    if not path.exists():
        return
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return
        rows = [row for row in reader if int(float(row["step"])) < max_step_exclusive]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_resume_checkpoint(args: argparse.Namespace, checkpoint_dir: Path) -> Path | None:
    if args.resume_checkpoint is not None:
        return args.resume_checkpoint
    if not args.resume_latest:
        return None
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def capture_rng_state(device: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None, device: str) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if device.startswith("cuda") and torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def next_position_after_completed(stage: int, epoch: int, args: argparse.Namespace) -> tuple[int, int]:
    schedule = make_schedule(args)
    try:
        idx = schedule.index((stage, epoch))
    except ValueError:
        return stage, epoch + 1
    return schedule[idx + 1] if idx + 1 < len(schedule) else (0, 0)


def infer_resume_position(ckpt: dict[str, Any], ckpt_path: Path, steps_per_epoch: int, args: argparse.Namespace) -> tuple[int, int]:
    if "next_stage" in ckpt and "next_epoch" in ckpt:
        next_stage = int(ckpt["next_stage"])
        next_epoch = int(ckpt["next_epoch"])
        if (next_stage, next_epoch) != (0, 0):
            return next_stage, next_epoch
        completed_stage = int(ckpt.get("stage", 0))
        completed_epoch = int(ckpt.get("epoch", 0))
        if completed_stage > 0 and completed_epoch > 0:
            return next_position_after_completed(completed_stage, completed_epoch, args)
        return 0, 0

    match = re.search(r"stage(\d+)_epoch(\d+)", ckpt_path.stem)
    if match:
        stage = int(match.group(1))
        epoch = int(match.group(2)) + 1
        return stage, epoch

    completed_epochs = int(ckpt.get("step", 0)) // max(1, steps_per_epoch)
    if completed_epochs < args.epochs_stage1:
        return 1, completed_epochs + 1
    completed_epochs -= args.epochs_stage1
    if completed_epochs < args.epochs_stage2:
        return 2, completed_epochs + 1
    completed_epochs -= args.epochs_stage2
    if completed_epochs < args.epochs_stage3:
        return 3, completed_epochs + 1
    return 0, 0


def make_loader(args: argparse.Namespace, data: ZarrTripletDataset) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if args.device.startswith("cuda"):
            loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(data, **loader_kwargs)


def make_schedule(args: argparse.Namespace) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    schedule.extend((1, e + 1) for e in range(args.epochs_stage1))
    schedule.extend((2, e + 1) for e in range(args.epochs_stage2))
    schedule.extend((3, e + 1) for e in range(args.epochs_stage3))
    return schedule


def apply_config_overrides(cfg: HorusEyeConfig, args: argparse.Namespace) -> HorusEyeConfig:
    cfg.xi_low = args.xi_low
    cfg.xi_corr = args.xi_corr
    cfg.structure_blur_kernel_size = args.structure_blur_kernel_size
    cfg.structure_blur_sigma = args.structure_blur_sigma
    cfg.highpass_kernel_size = args.highpass_kernel_size
    cfg.highpass_sigma = args.highpass_sigma
    cfg.gate_blur_kernel_size = args.gate_blur_kernel_size
    cfg.gate_blur_sigma = args.gate_blur_sigma
    return cfg


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().squeeze().numpy()
    return np.clip(arr, 0.0, 1.0)


def save_image_grid(path: Path, images: list[tuple[str, torch.Tensor]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), constrained_layout=True)
    if len(images) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, images):
        ax.imshow(tensor_to_image(image), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metrics(csv_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows: list[dict[str, float]] = []
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if k not in {"stage"}} | {"stage": float(row["stage"])})
    if not rows:
        return

    def column(name: str, default: float = np.nan) -> np.ndarray:
        return np.asarray([r.get(name, default) for r in rows], dtype=float)

    def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
        window = max(1, min(window, values.size))
        if window == 1:
            return values
        valid = np.isfinite(values)
        if not valid.any():
            return np.full_like(values, np.nan)
        kernel = np.ones(window, dtype=float)
        sums = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
        counts = np.convolve(valid.astype(float), kernel, mode="same")
        return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    steps = column("step")
    stages = column("stage").astype(int)
    loss_d = column("loss_d")
    loss_p = column("loss_p")
    accepted = column("accepted")
    denoiser_updated = column("denoiser_updated", 0.0) > 0
    predictor_updated = loss_p > 0
    alpha = column("alpha")
    res_mu = column("res_mu")
    res_sigma = column("res_sigma")
    x_sigma = column("x_sigma")

    d_loss_updates = np.where(denoiser_updated & (loss_d > 0), loss_d, np.nan)
    p_loss_updates = np.where(predictor_updated & (loss_p > 0), loss_p, np.nan)
    smooth_window = max(5, min(101, len(rows) // 30))
    if smooth_window % 2 == 0:
        smooth_window += 1

    stage_colors = {1: "#eaf2fb", 2: "#fff2db", 3: "#e9f6ec"}

    def add_stage_spans(ax: Any, annotate: bool = False) -> None:
        start = 0
        for idx in range(1, len(stages) + 1):
            if idx == len(stages) or stages[idx] != stages[start]:
                left = steps[start]
                right = steps[idx - 1]
                stage = int(stages[start])
                ax.axvspan(left, right, color=stage_colors.get(stage, "#eeeeee"), alpha=0.55, lw=0, zorder=0)
                if annotate:
                    ax.text(
                        (left + right) * 0.5,
                        0.96,
                        f"Stage {stage}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="#444444",
                        transform=ax.get_xaxis_transform(),
                    )
                start = idx

    def plot_update_loss(ax: Any, values: np.ndarray, title: str, color: str) -> None:
        add_stage_spans(ax, annotate=ax is axes[0])
        if np.isfinite(values).any():
            ax.scatter(steps, values, s=9, color=color, alpha=0.28, linewidths=0, label="actual update")
            ax.plot(steps, rolling_nanmean(values, smooth_window), color=color, lw=2.0, label=f"rolling mean ({smooth_window} steps)")
            ax.set_yscale("log")
        else:
            ax.text(0.02, 0.5, "no positive update loss recorded", transform=ax.transAxes, color="#666666")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("loss")
        ax.grid(alpha=0.22)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right")

    recent = accepted[-min(200, accepted.size) :]
    summary = (
        f"{len(rows)} steps | D updates {int(denoiser_updated.sum())} | "
        f"P updates {int(predictor_updated.sum())} | "
        f"recent accepted {float(np.nanmean(recent)):.3f}"
    )
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True, constrained_layout=True)
    fig.suptitle(f"HorusEye-TR training diagnostics - {summary}", fontsize=14)

    plot_update_loss(axes[0], d_loss_updates, "Denoiser loss: real optimizer steps only", "#1f6fb2")
    plot_update_loss(axes[1], p_loss_updates, "Predictor loss: real optimizer steps only", "#a15c00")

    add_stage_spans(axes[2])
    axes[2].plot(steps, accepted, color="#4f5965", alpha=0.18, lw=0.8, label="accepted per step")
    axes[2].plot(steps, rolling_nanmean(accepted, smooth_window), color="#087f5b", lw=2.0, label=f"accepted rolling mean ({smooth_window} steps)")
    axes[2].set_ylim(-0.04, 1.04)
    axes[2].set_ylabel("accepted")
    axes[2].set_title("Gate acceptance and residual injection strength", loc="left", fontsize=11)
    axes[2].grid(alpha=0.22)
    ax_alpha = axes[2].twinx()
    ax_alpha.plot(steps, alpha, color="#b83280", lw=1.6, label="alpha")
    ax_alpha.set_ylabel("alpha")
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax_alpha.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="upper right")

    add_stage_spans(axes[3])
    axes[3].plot(steps, res_sigma, color="#5941a9", lw=1.8, label="residual sigma")
    axes[3].plot(steps, x_sigma, color="#c55a11", lw=1.8, label="base highpass sigma")
    axes[3].set_ylabel("sigma")
    axes[3].set_title("Normalization statistics", loc="left", fontsize=11)
    axes[3].grid(alpha=0.22)
    ax_mu = axes[3].twinx()
    ax_mu.plot(steps, res_mu, color="#57606a", lw=1.2, alpha=0.75, label="residual mean")
    ax_mu.set_ylabel("residual mean")
    lines, labels = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax_mu.get_legend_handles_labels()
    axes[3].legend(lines + lines2, labels + labels2, loc="upper right")

    add_stage_spans(axes[4])
    axes[4].step(steps, np.cumsum(denoiser_updated), where="post", color="#1f6fb2", lw=1.8, label="denoiser update count")
    axes[4].step(steps, np.cumsum(predictor_updated), where="post", color="#a15c00", lw=1.8, label="predictor update count")
    axes[4].set_title("Optimizer activity", loc="left", fontsize=11)
    axes[4].set_ylabel("updates")
    axes[4].set_xlabel("global step")
    axes[4].grid(alpha=0.22)
    axes[4].legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_visual_sample(model: HorusEyeTR, batch: dict[str, torch.Tensor], path: Path, alpha: float, denoiser_target: str) -> None:
    model.eval()
    with torch.no_grad():
        clean = batch["c_hr"] if denoiser_target == "hr" and "c_hr" in batch else batch["c"]
        residual = model.raw_residual(batch["h_m1"], batch["h"], batch["h_p1"], use_initial=False)
        low, corr = model.gate_scores(residual, batch["h"])
        mask = model.gate_mask(residual, batch["h"])
        z = model.normalize_residual(residual)
        noisy_plus = torch.clamp(clean + alpha * z, 0.0, 1.0)
        denoised = model.denoiser(noisy_plus)
        accepted = float(mask.mean().detach().cpu())
        sample_accepted = bool(mask[0].item() > 0)
        sample_low = float(low[0].detach().cpu())
        sample_corr = float(corr[0].detach().cpu())
    save_image_grid(
        path,
        [
            ("base REG x_c", batch["c"][0]),
            ("injected y_c", noisy_plus[0]),
            ("denoised D(y_c)", denoised[0]),
            ("center x_h", batch["h"][0]),
            (f"clean target {denoiser_target}", clean[0]),
            (
                f"residual used={int(sample_accepted)} batch={accepted:.2f} low={sample_low:.3f} corr={sample_corr:.3f}",
                (residual[0] - residual[0].min()) / (residual[0].amax() - residual[0].amin() + 1e-6),
            ),
        ],
    )
    model.train()


def train() -> None:
    args = parse_args()
    out = ensure_output_dirs(args.output_root)
    metrics_csv = out["logs"] / "train_metrics.csv"
    metrics_png = out["figures"] / "train_metrics.png"

    resume_path = resolve_resume_checkpoint(args, out["checkpoints"])
    resume_ckpt = load_checkpoint(resume_path, "cpu") if resume_path is not None else None

    if resume_ckpt is not None and "config" in resume_ckpt:
        cfg = HorusEyeConfig(**resume_ckpt["config"])
    else:
        cfg = HorusEyeConfig(patch_size=args.patch_size)
    cfg = apply_config_overrides(cfg, args)

    snapshot_run_scripts(out, args)

    data = ZarrTripletDataset(
        args.data_root,
        patch_size=cfg.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        include_hr=args.denoiser_target == "hr",
        volume_glob=args.volume_glob,
        reg_subpath=args.reg_subpath,
        hr_subpath=args.hr_subpath,
    )
    loader = make_loader(args, data)
    model = HorusEyeTR(cfg).to(args.device)
    opt_d = torch.optim.AdamW(model.denoiser.parameters(), lr=args.lr_denoiser, weight_decay=1e-4)
    opt_p = torch.optim.AdamW(model.predictor.parameters(), lr=args.lr_predictor, weight_decay=1e-4)
    global_step = 0
    warmup_steps = max(1, args.epochs_stage1 * len(loader))
    schedule = make_schedule(args)
    start_index = 0

    if resume_ckpt is not None and resume_path is not None:
        model.load_state_dict(resume_ckpt["model"])
        global_step = int(resume_ckpt.get("step", 0))
        if "opt_d" in resume_ckpt:
            opt_d.load_state_dict(resume_ckpt["opt_d"])
        else:
            print("Resume checkpoint has no denoiser optimizer state; optimizer will restart.")
        if "opt_p" in resume_ckpt:
            opt_p.load_state_dict(resume_ckpt["opt_p"])
        else:
            print("Resume checkpoint has no predictor optimizer state; optimizer will restart.")
        restore_rng_state(resume_ckpt.get("rng_state"), args.device)
        start_stage, start_epoch = infer_resume_position(resume_ckpt, resume_path, len(loader), args)
        if (start_stage, start_epoch) == (0, 0):
            start_index = len(schedule)
        else:
            try:
                start_index = schedule.index((start_stage, start_epoch))
            except ValueError:
                start_index = len(schedule)
        trim_metrics_csv(metrics_csv, global_step)
        print(f"Resuming from {resume_path} at step {global_step}, next stage/epoch: {start_stage}/{start_epoch}")

    args_for_json = vars(args).copy()
    for key in ("data_root", "output_root", "resume_checkpoint"):
        if args_for_json.get(key) is not None:
            args_for_json[key] = str(args_for_json[key])
    with (out["logs"] / "train_config.json").open("w") as f:
        json.dump({"args": args_for_json, "config": cfg.__dict__}, f, indent=2)

    def save_training_checkpoint(path: Path, stage: int, epoch: int, next_stage: int, next_epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "step": global_step,
                "stage": stage,
                "epoch": epoch,
                "next_stage": next_stage,
                "next_epoch": next_epoch,
                "opt_d": opt_d.state_dict(),
                "opt_p": opt_p.state_dict(),
                "rng_state": capture_rng_state(args.device),
            },
            path,
        )

    def run_epoch(stage: int, epoch: int) -> None:
        nonlocal global_step
        model.train()
        pbar = tqdm(loader, desc=f"stage {stage} epoch {epoch}")
        last_batch: dict[str, torch.Tensor] | None = None
        last_alpha = cfg.alpha_min
        for batch in pbar:
            b = to_device(batch, args.device)
            last_batch = b
            use_initial = stage == 1
            residual = model.raw_residual(b["h_m1"], b["h"], b["h_p1"], use_initial=use_initial)
            mask = model.gate_mask(residual.detach(), b["h"])
            denoiser_clean = b["c_hr"] if args.denoiser_target == "hr" else b["c"]
            if stage in (1, 3):
                model.update_stats(residual.detach(), denoiser_clean, mask)
            current_alpha = model.alpha(global_step, warmup_steps)
            last_alpha = current_alpha

            denoiser_updated = 0.0
            if stage in (1, 3):
                opt_d.zero_grad(set_to_none=True)
                loss_d_or_none = model.denoiser_loss(denoiser_clean, residual, mask, current_alpha)
                if loss_d_or_none is None:
                    loss_d = torch.zeros((), device=args.device)
                else:
                    loss_d = loss_d_or_none
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
                    opt_d.step()
                    model.update_ema()
                    denoiser_updated = 1.0
            else:
                loss_d = torch.zeros((), device=args.device)

            if stage == 2 or (stage == 3 and global_step % cfg.predictor_update_every == 0):
                opt_p.zero_grad(set_to_none=True)
                loss_p = model.predictor_loss(b["h_m1"], b["h"], b["h_p1"])
                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
                opt_p.step()
            else:
                loss_p = torch.zeros((), device=args.device)

            row = {
                "step": global_step,
                "stage": stage,
                "epoch": epoch,
                "loss_d": float(loss_d.detach().cpu()),
                "loss_p": float(loss_p.detach().cpu()),
                "accepted": float(mask.mean().detach().cpu()),
                "denoiser_updated": denoiser_updated,
                "alpha": float(current_alpha),
                "res_mu": float(model.res_mu.detach().cpu()),
                "res_sigma": float(model.res_sigma.detach().cpu()),
                "x_sigma": float(model.x_sigma.detach().cpu()),
            }
            append_csv(metrics_csv, row)
            global_step += 1
            pbar.set_postfix(loss_d=f"{row['loss_d']:.4f}", loss_p=f"{row['loss_p']:.4f}", accepted=f"{row['accepted']:.2f}", updated=f"{row['denoiser_updated']:.0f}", alpha=f"{row['alpha']:.3f}")

        plot_metrics(metrics_csv, metrics_png)
        if last_batch is not None and args.save_sample_every > 0 and epoch % args.save_sample_every == 0:
            save_visual_sample(model, last_batch, out["figures"] / f"sample_stage{stage}_epoch{epoch}.png", last_alpha, args.denoiser_target)

    for idx, (stage, epoch) in enumerate(schedule[start_index:], start=start_index):
        run_epoch(stage, epoch)
        next_stage, next_epoch = schedule[idx + 1] if idx + 1 < len(schedule) else (0, 0)
        save_training_checkpoint(out["checkpoints"] / f"horuseye_tr_stage{stage}_epoch{epoch}.pt", stage, epoch, next_stage, next_epoch)
        save_training_checkpoint(out["checkpoints"] / "horuseye_tr_latest.pt", stage, epoch, next_stage, next_epoch)

    save_training_checkpoint(out["checkpoints"] / "horuseye_tr_final.pt", 0, 0, 0, 0)
    plot_metrics(metrics_csv, metrics_png)
    print(f"Saved checkpoints to {out['checkpoints']}")
    print(f"Saved metrics CSV to {metrics_csv}")
    print(f"Saved metric plot to {metrics_png}")


if __name__ == "__main__":
    train()
