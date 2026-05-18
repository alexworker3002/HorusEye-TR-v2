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
    p = argparse.ArgumentParser(description="Evaluate HorusEye-TR denoising against paired HR zarr volumes.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--volume-glob", default="**/*.zarr", help="Glob under data-root selecting volume roots for this CT domain.")
    p.add_argument("--reg-subpath", default="REG/0", help="Relative path from each volume root to the noisy/REG zarr array.")
    p.add_argument("--hr-subpath", default="HR/2", help="Relative path from each volume root to the paired HR zarr array.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--slices", type=int, nargs="*", default=[160, 224, 256, 288, 352])
    p.add_argument("--max-volumes", type=int, default=None)
    p.add_argument("--save-comparisons", type=int, default=16)
    return p.parse_args()


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_triplet_and_hr_from_center(
    reg_m1: np.ndarray,
    reg: np.ndarray,
    reg_p1: np.ndarray,
    hr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reg_m1 = reg_m1.astype(np.float32, copy=False)
    reg = reg.astype(np.float32, copy=False)
    reg_p1 = reg_p1.astype(np.float32, copy=False)
    hr = hr.astype(np.float32, copy=False)
    lo, hi = np.percentile(reg, (0.5, 99.5))
    scale = hi - lo + 1e-6
    reg_m1_n = np.clip((reg_m1 - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)
    reg_n = np.clip((reg - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)
    reg_p1_n = np.clip((reg_p1 - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)
    hr_n = np.clip((hr - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)
    return reg_m1_n, reg_n, reg_p1_n, hr_n


def field_of_view_mask(x: np.ndarray) -> np.ndarray:
    border = np.concatenate([x[:, :16].ravel(), x[:, -16:].ravel(), x[:16, :].ravel(), x[-16:, :].ravel()])
    bg = float(np.median(border))
    mask = np.abs(x - bg) > 1e-4
    return mask if float(mask.mean()) >= 0.2 else np.ones_like(x, dtype=bool)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    err = (a - b) ** 2
    mse = float(err[mask].mean())
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(a - b)[mask].mean())


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2)[mask].mean()))


def ssim_global(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(av.mean())
    muy = float(bv.mean())
    vx = float(av.var())
    vy = float(bv.var())
    cov = float(((av - mux) * (bv - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))


def add_metrics(row: dict[str, Any], prefix: str, image: np.ndarray, hr: np.ndarray, mask: np.ndarray) -> None:
    row[f"psnr_{prefix}"] = psnr(image, hr, mask)
    row[f"mae_{prefix}"] = mae(image, hr, mask)
    row[f"rmse_{prefix}"] = rmse(image, hr, mask)
    row[f"ssim_{prefix}"] = ssim_global(image, hr, mask)


def add_delta_metrics(row: dict[str, Any], prefix: str) -> None:
    row[f"delta_psnr_{prefix}"] = row[f"psnr_{prefix}"] - row["psnr_reg"]
    row[f"delta_mae_{prefix}"] = row[f"mae_{prefix}"] - row["mae_reg"]
    row[f"delta_rmse_{prefix}"] = row[f"rmse_{prefix}"] - row["rmse_reg"]
    row[f"delta_ssim_{prefix}"] = row[f"ssim_{prefix}"] - row["ssim_reg"]


def save_comparison(
    path: Path,
    reg: np.ndarray,
    pred: np.ndarray,
    den: np.ndarray,
    joint: np.ndarray,
    hr: np.ndarray,
    title: str,
) -> None:
    removed = reg - den
    pred_err = pred - hr
    den_err = den - hr
    joint_err = joint - hr
    reg_err = reg - hr
    vmax_removed = max(1e-4, float(np.percentile(np.abs(removed), 99.5)))
    vmax_err = max(
        1e-4,
        float(np.percentile(np.abs(np.concatenate([reg_err.ravel(), pred_err.ravel(), den_err.ravel(), joint_err.ravel()])), 99.5)),
    )
    fig, axes = plt.subplots(3, 5, figsize=(22, 13), constrained_layout=True)
    fig.suptitle(title)
    panels = [
        ("REG", reg, "gray", 0.0, 1.0),
        ("HR", hr, "gray", 0.0, 1.0),
        ("REG - HR", reg_err, "coolwarm", -vmax_err, vmax_err),
        ("D(REG)", den, "gray", 0.0, 1.0),
        ("D(REG) - HR", den_err, "coolwarm", -vmax_err, vmax_err),
        ("Predictor P(z-1,z+1)", pred, "gray", 0.0, 1.0),
        ("HR", hr, "gray", 0.0, 1.0),
        ("P - HR", pred_err, "coolwarm", -vmax_err, vmax_err),
        ("Joint D(P)", joint, "gray", 0.0, 1.0),
        ("D(P) - HR", joint_err, "coolwarm", -vmax_err, vmax_err),
        ("REG", reg, "gray", 0.0, 1.0),
        ("D(REG)", den, "gray", 0.0, 1.0),
        ("REG - D(REG)", removed, "coolwarm", -vmax_removed, vmax_removed),
        ("P - REG", pred - reg, "coolwarm", -vmax_err, vmax_err),
        ("D(P) - D(REG)", joint - den, "coolwarm", -vmax_err, vmax_err),
    ]
    for ax, (name, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        if cmap != "gray":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def discover_pairs(data_root: Path, volume_glob: str, reg_subpath: str, hr_subpath: str) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    reg_rel = Path(reg_subpath)
    hr_rel = Path(hr_subpath)
    for volume_root in sorted(data_root.glob(volume_glob)):
        reg = volume_root / reg_rel
        hr = volume_root / hr_rel
        if (reg / "zarr.json").exists() and (hr / "zarr.json").exists():
            pairs.append((volume_root.stem, reg, hr))
    return pairs


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

    pairs = discover_pairs(args.data_root, args.volume_glob, args.reg_subpath, args.hr_subpath)
    if args.max_volumes is not None:
        pairs = pairs[: args.max_volumes]
    if not pairs:
        raise RuntimeError(
            "No paired REG and HR zarr volumes found: "
            f"data_root={args.data_root}, volume_glob={args.volume_glob}, "
            f"reg_subpath={args.reg_subpath}, hr_subpath={args.hr_subpath}"
        )

    rows: list[dict[str, Any]] = []
    comparison_count = 0
    with torch.no_grad():
        for volume_name, reg_path, hr_path in pairs:
            reg_vol = zarr.open(str(reg_path), mode="r")
            hr_vol = zarr.open(str(hr_path), mode="r")
            if reg_vol.shape != hr_vol.shape:
                raise RuntimeError(
                    f"REG and HR shapes must match for aligned evaluation: {reg_path} shape={reg_vol.shape}, "
                    f"{hr_path} shape={hr_vol.shape}. Use a matching --hr-subpath such as HR/2 for REG/0."
                )
            depth = min(reg_vol.shape[0], hr_vol.shape[0])
            for z in args.slices:
                if z <= 0 or z >= depth - 1:
                    continue
                reg_m1, reg, reg_p1, hr = normalize_triplet_and_hr_from_center(
                    np.asarray(reg_vol[z - 1]),
                    np.asarray(reg_vol[z]),
                    np.asarray(reg_vol[z + 1]),
                    np.asarray(hr_vol[z]),
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
                mask = field_of_view_mask(hr)

                row = {
                    "volume": volume_name,
                    "slice": z,
                    "mask_fraction": float(mask.mean()),
                    "mean_abs_change_den": float(np.abs(den - reg)[mask].mean()),
                    "p99_abs_change_den": float(np.percentile(np.abs(den - reg)[mask], 99.0)),
                    "mean_abs_change_pred": float(np.abs(pred - reg)[mask].mean()),
                    "p99_abs_change_pred": float(np.percentile(np.abs(pred - reg)[mask], 99.0)),
                    "mean_abs_change_joint": float(np.abs(joint - reg)[mask].mean()),
                    "p99_abs_change_joint": float(np.percentile(np.abs(joint - reg)[mask], 99.0)),
                }
                add_metrics(row, "reg", reg, hr, mask)
                add_metrics(row, "pred", pred, hr, mask)
                add_metrics(row, "den", den, hr, mask)
                add_metrics(row, "joint", joint, hr, mask)
                add_delta_metrics(row, "pred")
                add_delta_metrics(row, "den")
                add_delta_metrics(row, "joint")
                rows.append(row)

                if comparison_count < args.save_comparisons:
                    save_comparison(
                        comparison_dir / f"{comparison_count + 1:03d}_{volume_name}_slice_{z:04d}.png",
                        reg,
                        pred,
                        den,
                        joint,
                        hr,
                        f"{volume_name} slice {z}",
                    )
                    comparison_count += 1

    metrics_csv = args.output_dir / "hr_eval_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["checkpoint"] = str(args.checkpoint)
    summary["data_root"] = str(args.data_root)
    summary["volume_glob"] = args.volume_glob
    summary["reg_subpath"] = args.reg_subpath
    summary["hr_subpath"] = args.hr_subpath
    summary["slices"] = args.slices
    with (args.output_dir / "hr_eval_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_csv}")
    print(f"Saved comparison figures to {comparison_dir}")


if __name__ == "__main__":
    main()
