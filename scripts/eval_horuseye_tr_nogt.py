# 中文说明：本脚本编写于 2026-05-19，用于 003-COVID-19 数据集没有真实 GT 的情景。
# 测试用途：对 HorusEye-TR 模型进行 no-GT / reference-free 评估，只读取 REG 切片，不把 HR/0 当作 Ground Truth。
# 评估重点：去噪强度、高频残差低频占比、边缘保持、残差与结构边缘相关性、跨切片一致性。
# 预期结果：帮助比较不同 denoiser 训练方案是否在不依赖伪 HR 的前提下保留结构并抑制噪声。
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from horuseye_tr import HorusEyeConfig, HorusEyeTR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reference-free evaluation for HorusEye-TR on REG zarr volumes.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--volume-glob", default="**/*.zarr")
    p.add_argument("--reg-subpath", default="REG/0")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--slices", type=int, nargs="*", default=[128, 192, 256, 320, 384])
    p.add_argument("--max-volumes", type=int, default=None)
    p.add_argument("--save-comparisons", type=int, default=24)
    return p.parse_args()


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_triplet(reg_m1: np.ndarray, reg: np.ndarray, reg_p1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reg_m1 = reg_m1.astype(np.float32, copy=False)
    reg = reg.astype(np.float32, copy=False)
    reg_p1 = reg_p1.astype(np.float32, copy=False)
    lo, hi = np.percentile(reg, (0.5, 99.5))
    scale = hi - lo + 1e-6
    return (
        np.clip((reg_m1 - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False),
        np.clip((reg - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False),
        np.clip((reg_p1 - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False),
    )


def field_of_view_mask(x: np.ndarray) -> np.ndarray:
    border = np.concatenate([x[:, :16].ravel(), x[:, -16:].ravel(), x[:16, :].ravel(), x[-16:, :].ravel()])
    bg = float(np.median(border))
    mask = np.abs(x - bg) > 1e-4
    return mask if float(mask.mean()) >= 0.2 else np.ones_like(x, dtype=bool)


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    denom = math.sqrt(float((av * av).mean()) * float((bv * bv).mean())) + 1e-12
    return float((av * bv).mean() / denom)


def gradient_mag(x: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(x.astype(np.float32, copy=False))
    return np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)


def laplacian(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=False)
    out = -4.0 * y.copy()
    out[1:, :] += y[:-1, :]
    out[:-1, :] += y[1:, :]
    out[:, 1:] += y[:, :-1]
    out[:, :-1] += y[:, 1:]
    return out


def box_blur(x: np.ndarray, radius: int = 4) -> np.ndarray:
    y = x.astype(np.float32, copy=False)
    pad = radius
    padded = np.pad(y, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    k = 2 * radius + 1
    total = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return (total / float(k * k)).astype(np.float32, copy=False)


def masked_std(x: np.ndarray, mask: np.ndarray) -> float:
    return float(x[mask].astype(np.float64).std())


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(a - b)[mask].mean())


def add_no_gt_metrics(row: dict[str, Any], name: str, image: np.ndarray, reg: np.ndarray, neighbor_avg: np.ndarray, mask: np.ndarray) -> None:
    residual = reg - image
    grad_reg = gradient_mag(reg)
    grad_image = gradient_mag(image)
    lap_reg = laplacian(reg)
    lap_image = laplacian(image)
    row[f"mean_abs_change_{name}"] = masked_mae(reg, image, mask)
    row[f"p99_abs_change_{name}"] = float(np.percentile(np.abs(residual)[mask], 99.0))
    row[f"residual_std_{name}"] = masked_std(residual, mask)
    row[f"residual_lowfreq_fraction_{name}"] = masked_std(box_blur(residual), mask) / (masked_std(residual, mask) + 1e-12)
    row[f"residual_edge_corr_{name}"] = corr(residual, grad_reg, mask)
    row[f"edge_retention_{name}"] = corr(grad_reg, grad_image, mask)
    row[f"laplacian_std_ratio_{name}"] = masked_std(lap_image, mask) / (masked_std(lap_reg, mask) + 1e-12)
    row[f"neighbor_mae_{name}"] = masked_mae(image, neighbor_avg, mask)


def save_comparison(path: Path, reg_m1: np.ndarray, reg: np.ndarray, reg_p1: np.ndarray, pred: np.ndarray, den: np.ndarray, joint: np.ndarray, title: str) -> None:
    removed = reg - den
    grad_reg = gradient_mag(reg)
    grad_den = gradient_mag(den)
    vmax_removed = max(1e-4, float(np.percentile(np.abs(removed), 99.5)))
    vmax_delta = max(1e-4, float(np.percentile(np.abs(np.concatenate([(pred - reg).ravel(), (joint - reg).ravel()])), 99.5)))
    fig, axes = plt.subplots(3, 5, figsize=(22, 13), constrained_layout=True)
    fig.suptitle(title)
    panels = [
        ("REG z-1", reg_m1, "gray", 0.0, 1.0),
        ("REG center", reg, "gray", 0.0, 1.0),
        ("REG z+1", reg_p1, "gray", 0.0, 1.0),
        ("Predictor P", pred, "gray", 0.0, 1.0),
        ("P - REG", pred - reg, "coolwarm", -vmax_delta, vmax_delta),
        ("REG", reg, "gray", 0.0, 1.0),
        ("D(REG)", den, "gray", 0.0, 1.0),
        ("REG - D(REG)", removed, "coolwarm", -vmax_removed, vmax_removed),
        ("Joint D(P)", joint, "gray", 0.0, 1.0),
        ("D(P) - REG", joint - reg, "coolwarm", -vmax_delta, vmax_delta),
        ("grad REG", grad_reg, "magma", None, None),
        ("grad D(REG)", grad_den, "magma", None, None),
        ("grad D - grad REG", grad_den - grad_reg, "coolwarm", -vmax_removed, vmax_removed),
        ("lap REG", laplacian(reg), "coolwarm", -vmax_removed, vmax_removed),
        ("lap D(REG)", laplacian(den), "coolwarm", -vmax_removed, vmax_removed),
    ]
    for ax, (name, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        if cmap != "gray":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def discover_volumes(data_root: Path, volume_glob: str, reg_subpath: str) -> list[tuple[str, Path]]:
    reg_rel = Path(reg_subpath)
    volumes: list[tuple[str, Path]] = []
    for volume_root in sorted(data_root.glob(volume_glob)):
        reg = volume_root / reg_rel
        if (reg / "zarr.json").exists():
            volumes.append((volume_root.stem, reg))
    return volumes


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    summary: dict[str, Any] = {"num_slices": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = HorusEyeConfig(**ckpt.get("config", {}))
    model = HorusEyeTR(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    volumes = discover_volumes(args.data_root, args.volume_glob, args.reg_subpath)
    if args.max_volumes is not None:
        volumes = volumes[: args.max_volumes]
    if not volumes:
        raise RuntimeError(
            "No REG zarr volumes found: "
            f"data_root={args.data_root}, volume_glob={args.volume_glob}, reg_subpath={args.reg_subpath}"
        )

    rows: list[dict[str, Any]] = []
    comparison_count = 0
    with torch.no_grad():
        for volume_name, reg_path in volumes:
            reg_vol = zarr.open(str(reg_path), mode="r")
            depth = reg_vol.shape[0]
            for z in args.slices:
                if z <= 0 or z >= depth - 1:
                    continue
                reg_m1, reg, reg_p1 = normalize_triplet(
                    np.asarray(reg_vol[z - 1]),
                    np.asarray(reg_vol[z]),
                    np.asarray(reg_vol[z + 1]),
                )
                x = torch.from_numpy(reg).unsqueeze(0).unsqueeze(0).to(args.device).float()
                x_m1 = torch.from_numpy(reg_m1).unsqueeze(0).unsqueeze(0).to(args.device).float()
                x_p1 = torch.from_numpy(reg_p1).unsqueeze(0).unsqueeze(0).to(args.device).float()
                pred = model.predictor(x_m1, x_p1).squeeze().detach().cpu().numpy().astype(np.float32)
                den = model.denoiser(x).squeeze().detach().cpu().numpy().astype(np.float32)
                joint = model.denoiser(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(args.device).float()).squeeze().detach().cpu().numpy().astype(np.float32)
                pred = np.clip(pred, 0.0, 1.0)
                den = np.clip(den, 0.0, 1.0)
                joint = np.clip(joint, 0.0, 1.0)
                mask = field_of_view_mask(reg)
                neighbor_avg = 0.5 * (reg_m1 + reg_p1)

                row: dict[str, Any] = {
                    "volume": volume_name,
                    "slice": z,
                    "mask_fraction": float(mask.mean()),
                    "neighbor_mae_reg": masked_mae(reg, neighbor_avg, mask),
                    "laplacian_std_reg": masked_std(laplacian(reg), mask),
                }
                add_no_gt_metrics(row, "pred", pred, reg, neighbor_avg, mask)
                add_no_gt_metrics(row, "den", den, reg, neighbor_avg, mask)
                add_no_gt_metrics(row, "joint", joint, reg, neighbor_avg, mask)
                rows.append(row)

                if comparison_count < args.save_comparisons:
                    save_comparison(
                        comparison_dir / f"{comparison_count + 1:03d}_{volume_name}_slice_{z:04d}.png",
                        reg_m1,
                        reg,
                        reg_p1,
                        pred,
                        den,
                        joint,
                        f"{volume_name} slice {z}",
                    )
                    comparison_count += 1

    metrics_csv = args.output_dir / "nogt_eval_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["checkpoint"] = str(args.checkpoint)
    summary["data_root"] = str(args.data_root)
    summary["volume_glob"] = args.volume_glob
    summary["reg_subpath"] = args.reg_subpath
    summary["slices"] = args.slices
    with (args.output_dir / "nogt_eval_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_csv}")
    print(f"Saved comparison figures to {comparison_dir}")


if __name__ == "__main__":
    main()
