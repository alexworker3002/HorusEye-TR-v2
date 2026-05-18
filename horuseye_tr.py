from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr


def read_zarr_shape(path: str | Path) -> tuple[int, ...]:
    metadata_path = Path(path) / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    return tuple(int(dim) for dim in metadata["shape"])


@dataclass
class HorusEyeConfig:
    channels: int = 48
    blocks: int = 6
    eta_d: float = 1.0
    alpha_min: float = 0.02
    alpha_max: float = 0.15
    xi_low: float = 0.10
    xi_corr: float = 0.04
    structure_blur_kernel_size: int = 9
    structure_blur_sigma: float = 2.0
    highpass_kernel_size: int = 9
    highpass_sigma: float = 2.0
    gate_blur_kernel_size: int = 11
    gate_blur_sigma: float = 2.5
    ema_decay: float = 0.995
    stat_decay: float = 0.99
    eps_gate: float = 1e-6
    eps_charb: float = 1e-3
    lambda_g: float = 0.1
    lambda_low: float = 0.05
    lambda_h: float = 0.1
    a_max: float = 3.0
    predictor_update_every: int = 4
    patch_size: int = 128


class ZarrTripletDataset(Dataset):
    """Sample REG triplets for the predictor and an aligned REG/HR center target for the denoiser."""

    def __init__(
        self,
        data_root: str | Path,
        patch_size: int = 128,
        samples_per_epoch: int = 4096,
        include_hr: bool = False,
        volume_glob: str = "**/*.zarr",
        reg_subpath: str = "REG/0",
        hr_subpath: str = "HR/2",
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.include_hr = include_hr
        self.volume_glob = volume_glob
        self.reg_subpath = Path(reg_subpath)
        self.hr_subpath = Path(hr_subpath)
        self.volume_roots = [p for p in sorted(self.data_root.glob(self.volume_glob)) if (p / self.reg_subpath / "zarr.json").exists()]
        self.volume_paths = [p / self.reg_subpath for p in self.volume_roots]
        if len(self.volume_paths) < 2:
            raise RuntimeError(
                "Need at least two REG volumes for cross-volume residual injection: "
                f"data_root={self.data_root}, volume_glob={self.volume_glob}, reg_subpath={self.reg_subpath}"
            )
        self.hr_volume_paths = [p / self.hr_subpath for p in self.volume_roots]
        if self.include_hr and not all((path / "zarr.json").exists() for path in self.hr_volume_paths):
            raise RuntimeError(
                "HR denoiser training needs paired HR volumes for every REG volume: "
                f"data_root={self.data_root}, volume_glob={self.volume_glob}, hr_subpath={self.hr_subpath}"
            )
        if self.include_hr:
            mismatched = []
            for reg_path, hr_path in zip(self.volume_paths, self.hr_volume_paths):
                reg_shape = read_zarr_shape(reg_path)
                hr_shape = read_zarr_shape(hr_path)
                if reg_shape != hr_shape:
                    mismatched.append((reg_path, hr_path, reg_shape, hr_shape))
            if mismatched:
                details = "; ".join(f"{reg} shape={reg_shape} vs {hr} shape={hr_shape}" for reg, hr, reg_shape, hr_shape in mismatched[:3])
                raise RuntimeError(
                    "REG and HR arrays must have identical shapes for patch-aligned HR training. "
                    "Choose a matching --hr-subpath, for example HR/2 for REG/0 or HR/3 for REG/1 in the current data. "
                    f"Mismatches: {details}"
                )
        self.volumes: list[Any] | None = None
        self.hr_volumes: list[Any] | None = None
        self._open_volumes()
        assert self.volumes is not None
        self.shapes = [tuple(v.shape) for v in self.volumes]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["volumes"] = None
        state["hr_volumes"] = None
        return state

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _open_volumes(self) -> None:
        if self.volumes is None:
            self.volumes = [zarr.open(str(path), mode="r") for path in self.volume_paths]
        if self.include_hr and self.hr_volumes is None:
            self.hr_volumes = [zarr.open(str(path), mode="r") for path in self.hr_volume_paths]

    def _sample_patch(self, vol_idx: int, z: int | None = None, include_hr: bool = False) -> tuple[Tensor, ...]:
        self._open_volumes()
        assert self.volumes is not None
        vol = self.volumes[vol_idx]
        depth, height, width = self.shapes[vol_idx]
        z = random.randint(1, depth - 2) if z is None else max(1, min(depth - 2, z))
        y0 = random.randint(0, max(0, height - self.patch_size))
        x0 = random.randint(0, max(0, width - self.patch_size))
        ys = slice(y0, y0 + self.patch_size)
        xs = slice(x0, x0 + self.patch_size)
        triplet = np.asarray(vol[z - 1 : z + 2, ys, xs], dtype=np.float32)
        lo, hi = np.percentile(triplet, (0.5, 99.5))
        scale = hi - lo + 1e-6
        triplet = np.clip((triplet - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)
        tensors = [torch.from_numpy(triplet[i]).unsqueeze(0) for i in range(3)]
        if include_hr:
            assert self.hr_volumes is not None
            hr_vol = self.hr_volumes[vol_idx]
            hr = np.asarray(hr_vol[z, ys, xs], dtype=np.float32)
            hr = np.clip((hr - lo) / scale, 0.0, 1.0).astype(np.float32, copy=False)
            tensors.append(torch.from_numpy(hr).unsqueeze(0))
        return tuple(tensors)

    def __getitem__(self, _: int) -> dict[str, Tensor]:
        self._open_volumes()
        assert self.volumes is not None
        h_idx = random.randrange(len(self.volumes))
        c_idx = random.randrange(len(self.volumes) - 1)
        if c_idx >= h_idx:
            c_idx += 1
        hm1, h, hp1 = self._sample_patch(h_idx)
        c_patch = self._sample_patch(c_idx, include_hr=self.include_hr)
        _, c, *_ = c_patch
        sample = {"h_m1": hm1, "h": h, "h_p1": hp1, "c": c}
        if self.include_hr:
            sample["c_hr"] = c_patch[3]
        return sample


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * self.net(x)


class ResidualCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, channels: int, blocks: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, channels, 3, padding=1), nn.SiLU(inplace=True)]
        layers += [ResidualBlock(channels) for _ in range(blocks)]
        layers += [nn.Conv2d(channels, out_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StructurePredictor(nn.Module):
    def __init__(self, channels: int, blocks: int):
        super().__init__()
        self.net = ResidualCNN(2, 1, channels, blocks)

    def forward(self, x_m1: Tensor, x_p1: Tensor) -> Tensor:
        return torch.sigmoid(self.net(torch.cat([x_m1, x_p1], dim=1)))


class ResidualDenoiser(nn.Module):
    def __init__(self, channels: int, blocks: int, eta_d: float):
        super().__init__()
        self.eta_d = eta_d
        self.residual = ResidualCNN(1, 1, channels, blocks)

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x - self.eta_d * self.residual(x), 0.0, 1.0)


class HorusEyeTR(nn.Module):
    def __init__(self, cfg: HorusEyeConfig):
        super().__init__()
        self.cfg = cfg
        self.predictor = StructurePredictor(cfg.channels, max(2, cfg.blocks // 2))
        self.denoiser = ResidualDenoiser(cfg.channels, cfg.blocks, cfg.eta_d)
        self.ema_denoiser = copy.deepcopy(self.denoiser).requires_grad_(False)
        self.register_buffer("res_mu", torch.zeros(1, 1, 1, 1))
        self.register_buffer("res_sigma", torch.ones(1, 1, 1, 1))
        self.register_buffer("x_sigma", torch.ones(1, 1, 1, 1))

    @staticmethod
    def _odd_kernel_size(kernel_size: int) -> int:
        kernel_size = max(3, int(kernel_size))
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    @staticmethod
    def blur(x: Tensor, kernel_size: int = 5, sigma: float = 1.0) -> Tensor:
        kernel_size = HorusEyeTR._odd_kernel_size(kernel_size)
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        k2 = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
        return F.conv2d(x, k2, padding=kernel_size // 2)

    def highpass(self, x: Tensor) -> Tensor:
        return x - self.blur(x, self.cfg.highpass_kernel_size, self.cfg.highpass_sigma)

    @staticmethod
    def charbonnier(x: Tensor, eps: float) -> Tensor:
        return torch.sqrt(x * x + eps * eps).mean()

    @staticmethod
    def weighted_charbonnier(e: Tensor, weight: Tensor, eps: float) -> Tensor:
        return (weight * torch.sqrt(e * e + eps * eps)).mean()

    def smooth_grad_loss(self, e: Tensor) -> Tensor:
        smoothed = self.blur(e)
        gx = smoothed[:, :, :, 1:] - smoothed[:, :, :, :-1]
        gy = smoothed[:, :, 1:, :] - smoothed[:, :, :-1, :]
        return self.charbonnier(gx, self.cfg.eps_charb) + self.charbonnier(gy, self.cfg.eps_charb)

    def initial_structure(self, x_m1: Tensor, x_p1: Tensor) -> Tensor:
        return self.blur((x_m1 + x_p1) * 0.5, self.cfg.structure_blur_kernel_size, self.cfg.structure_blur_sigma)

    def raw_residual(self, x_m1: Tensor, x: Tensor, x_p1: Tensor, use_initial: bool = False) -> Tensor:
        pred = self.initial_structure(x_m1, x_p1) if use_initial else self.predictor(x_m1, x_p1)
        return self.highpass(x - pred)

    def gate_scores(self, residual: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        flat_r = residual.flatten(1)
        low_r = self.blur(residual, self.cfg.gate_blur_kernel_size, self.cfg.gate_blur_sigma)
        low_x = self.blur(x, self.cfg.gate_blur_kernel_size, self.cfg.gate_blur_sigma)
        low = low_r.flatten(1).norm(dim=1) / (flat_r.norm(dim=1) + self.cfg.eps_gate)
        sx = low_x.flatten(1)
        corr = (flat_r * sx).sum(dim=1).abs() / (flat_r.norm(dim=1) * sx.norm(dim=1) + self.cfg.eps_gate)
        return low, corr

    def gate_mask(self, residual: Tensor, x: Tensor) -> Tensor:
        low, corr = self.gate_scores(residual, x)
        return ((low <= self.cfg.xi_low) & (corr <= self.cfg.xi_corr)).float().view(-1, 1, 1, 1)

    @torch.no_grad()
    def update_stats(self, residual: Tensor, x_base: Tensor, mask: Tensor) -> None:
        if mask.sum() < 1:
            return
        accepted = residual[mask.flatten() > 0]
        mu = accepted.mean().view_as(self.res_mu)
        sigma = accepted.std(unbiased=False).clamp_min(1e-6).view_as(self.res_sigma)
        xs = self.highpass(x_base).std(unbiased=False).clamp_min(1e-6).view_as(self.x_sigma)
        d = self.cfg.stat_decay
        self.res_mu.mul_(d).add_(mu, alpha=1 - d)
        self.res_sigma.mul_(d).add_(sigma, alpha=1 - d)
        self.x_sigma.mul_(d).add_(xs, alpha=1 - d)

    def normalize_residual(self, residual: Tensor) -> Tensor:
        a = torch.clamp(self.x_sigma / (self.res_sigma + 1e-6), 0.0, self.cfg.a_max)
        return a * (residual - self.res_mu)

    def alpha(self, global_step: int, warmup_steps: int) -> float:
        t = min(1.0, global_step / max(1, warmup_steps))
        return self.cfg.alpha_min + 0.5 * (1 - math.cos(math.pi * t)) * (self.cfg.alpha_max - self.cfg.alpha_min)

    @torch.no_grad()
    def update_ema(self) -> None:
        d = self.cfg.ema_decay
        for ema_p, p in zip(self.ema_denoiser.parameters(), self.denoiser.parameters()):
            ema_p.mul_(d).add_(p, alpha=1 - d)

    def denoiser_loss(self, x_c: Tensor, residual: Tensor, mask: Tensor, alpha: float) -> Tensor | None:
        accept = mask.flatten() > 0
        if accept.sum() == 0:
            return None
        x_c = x_c[accept]
        residual = residual[accept]
        z = self.normalize_residual(residual.detach())
        y = torch.clamp(x_c + alpha * z, 0.0, 1.0)
        out = self.denoiser(y)
        e = out - x_c
        high = self.highpass(x_c).abs()
        protect = torch.clamp(1.0 - 0.75 * high / (high.amax(dim=(2, 3), keepdim=True) + 1e-6), 0.1, 1.0)
        pixel = self.weighted_charbonnier(e, protect, self.cfg.eps_charb)
        grad = self.smooth_grad_loss(e)
        low = F.l1_loss(self.blur(out), self.blur(x_c))
        return pixel + self.cfg.lambda_g * grad + self.cfg.lambda_low * low

    def predictor_loss(self, x_m1: Tensor, x: Tensor, x_p1: Tensor) -> Tensor:
        pred = self.predictor(x_m1, x_p1)
        with torch.no_grad():
            anchor = self.ema_denoiser(x)
        low = F.l1_loss(self.blur(pred), self.blur(anchor))
        high = F.l1_loss(self.highpass(pred), self.highpass((x_m1 + x_p1) * 0.5))
        return low + self.cfg.lambda_h * high


def iter_parameters(modules: Sequence[nn.Module]) -> Iterable[nn.Parameter]:
    for module in modules:
        yield from module.parameters()
