from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from horuseye_tr import HorusEyeConfig, HorusEyeTR

TRAIN_SCRIPT_NAME = "train_horuseye_tr"
INFERENCE_SCRIPT_NAME = Path(__file__).stem
DEFAULT_TRAIN_OUTPUT_ROOT = ROOT / "output" / TRAIN_SCRIPT_NAME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HorusEye-TR denoiser inference on zarr CT slices.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_TRAIN_OUTPUT_ROOT / "checkpoints" / "horuseye_tr_final.pt")
    p.add_argument("--volume", type=Path, default=None, help="Single zarr array to denoise. If omitted, volumes are discovered from data-root.")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--volume-glob", default="**/*.zarr", help="Glob under data-root selecting volume roots for batch inference.")
    p.add_argument("--reg-subpath", default="REG/0", help="Relative path from each discovered volume root to the zarr array to denoise.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_ROOT / "inference" / INFERENCE_SCRIPT_NAME)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--slice", type=int, default=None)
    p.add_argument("--num-slices", type=int, default=8)
    return p.parse_args()


def normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, (0.5, 99.5))
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32, copy=False)


def save_pair(path: Path, noisy: np.ndarray, denoised: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("input noisy")
    axes[1].imshow(denoised, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("denoised")
    axes[2].imshow(np.abs(noisy - denoised), cmap="magma")
    axes[2].set_title("removed |x-D(x)|")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_checkpoint(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def discover_volumes(data_root: Path, volume_glob: str, reg_subpath: str) -> list[tuple[str, Path]]:
    reg_rel = Path(reg_subpath)
    volumes: list[tuple[str, Path]] = []
    for volume_root in sorted(data_root.glob(volume_glob)):
        volume = volume_root / reg_rel
        if (volume / "zarr.json").exists():
            volumes.append((volume_root.stem, volume))
    return volumes


def single_volume_name(volume: Path) -> str:
    for parent in volume.parents:
        if parent.suffix == ".zarr":
            return parent.stem
    return volume.name


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = HorusEyeConfig(**ckpt.get("config", {}))
    model = HorusEyeTR(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.volume is None:
        volumes = discover_volumes(args.data_root, args.volume_glob, args.reg_subpath)
    else:
        volumes = [(single_volume_name(args.volume), args.volume)]
    if not volumes:
        raise RuntimeError(
            "No inference volumes found: "
            f"data_root={args.data_root}, volume_glob={args.volume_glob}, reg_subpath={args.reg_subpath}"
        )

    with torch.no_grad():
        for volume_name, volume_path in volumes:
            volume_output_dir = args.output_dir if args.volume is not None else args.output_dir / volume_name
            volume_output_dir.mkdir(parents=True, exist_ok=True)
            vol = zarr.open(str(volume_path), mode="r")
            depth = vol.shape[0]
            if args.slice is None:
                start = max(0, depth // 2 - args.num_slices // 2)
                slice_ids = list(range(start, min(depth, start + args.num_slices)))
            else:
                slice_ids = [args.slice]

            for z in slice_ids:
                noisy = normalize_slice(np.asarray(vol[z]))
                x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(args.device).float()
                denoised = model.denoiser(x).squeeze().cpu().numpy()
                np.save(volume_output_dir / f"slice_{z:04d}_denoised.npy", denoised.astype(np.float32))
                save_pair(volume_output_dir / f"slice_{z:04d}_comparison.png", noisy, denoised)

    print(f"Saved inference results to {args.output_dir}")


if __name__ == "__main__":
    main()
