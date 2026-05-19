#!/usr/bin/env python3
"""Convert paired COVID-19 npz CT volumes into the local HorusEye zarr layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.lib.format as npy_format
import zarr
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle


DEFAULT_HR_DIR = Path(
    "/mnt/Longxi_ET1/data_disk/rescaled_ct_and_semantics/"
    "rescaled_ct-denoise/COVID-19/mudanjiang"
)
DEFAULT_REG_DIR = Path(
    "/mnt/Longxi_ET1/data_disk/rescaled_ct_and_semantics/"
    "rescaled_ct/COVID-19/mudanjiang"
)
DOMAIN_RE = re.compile(r"^(?P<number>\d+)-(?P<name>.+)$")


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three integers, e.g. 16,128,128")
    try:
        shape = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunks must be integers") from exc
    if any(v <= 0 for v in shape):
        raise argparse.ArgumentTypeError("chunks must be positive")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair denoised COVID-19 npz files as HR with raw/rescaled npz files as REG, "
            "then write data/NNN-COVID-19/<sample>_ome.zarr volumes."
        )
    )
    parser.add_argument("--hr-dir", type=Path, default=DEFAULT_HR_DIR, help="Denoised npz directory treated as HR.")
    parser.add_argument("--reg-dir", "--lr-dir", dest="reg_dir", type=Path, default=DEFAULT_REG_DIR, help="Raw/rescaled npz directory treated as REG input.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Local data root. Default: data")
    parser.add_argument("--domain", default="COVID-19", help="Domain name used in NNN-domain folder. Default: COVID-19")
    parser.add_argument("--array-key", default=None, help="NPZ array key. If omitted, a single-key npz is accepted.")
    parser.add_argument("--chunks", type=parse_shape, default=(16, 128, 128), help="Zarr chunk shape z,y,x. Default: 16,128,128")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many paired volumes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing zarr volume roots.")
    parser.add_argument("--strict-pairs", action="store_true", help="Fail when HR/REG filenames are not exactly one-to-one.")
    parser.add_argument("--dry-run", action="store_true", help="Only report pairing and planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate an already converted domain without writing data.")
    parser.add_argument("--model-check", action="store_true", help="Instantiate ZarrTripletDataset to verify framework alignment.")
    return parser.parse_args()


def domain_dir(data_root: Path, domain: str) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    highest = 0
    found: Path | None = None
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        match = DOMAIN_RE.match(child.name)
        if not match:
            continue
        highest = max(highest, int(match.group("number")))
        if match.group("name") == domain:
            found = child
    if found is not None:
        return found
    return data_root / f"{highest + 1:03d}-{domain}"


def npz_files(path: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(path.glob("*.npz"))}


def pair_npz_files(hr_dir: Path, reg_dir: Path, strict: bool) -> list[tuple[str, Path, Path]]:
    hr = npz_files(hr_dir)
    reg = npz_files(reg_dir)
    missing_reg = sorted(set(hr) - set(reg))
    missing_hr = sorted(set(reg) - set(hr))
    if strict and (missing_reg or missing_hr):
        raise RuntimeError(
            "HR/REG npz filenames are not one-to-one: "
            f"missing REG for {missing_reg[:10]}, missing HR for {missing_hr[:10]}"
        )
    if missing_reg:
        print(f"WARN skipping {len(missing_reg)} HR-only files, first entries: {missing_reg[:10]}")
    if missing_hr:
        print(f"WARN skipping {len(missing_hr)} REG-only files, first entries: {missing_hr[:10]}")
    common = sorted(set(hr) & set(reg))
    return [(name, hr[name], reg[name]) for name in common]


def load_npz_array(path: Path, array_key: str | None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        key = array_key
        if key is None:
            if len(keys) != 1:
                raise RuntimeError(f"{path} contains keys {keys}; pass --array-key")
            key = keys[0]
        if key not in npz:
            raise RuntimeError(f"{path} does not contain key {key!r}; available keys: {keys}")
        array = npz[key]
        if array.ndim != 3:
            raise RuntimeError(f"{path}:{key} must be a 3D z,y,x volume, got shape={array.shape}")
        return np.asarray(array)


def inspect_npz(path: Path, array_key: str | None) -> tuple[tuple[int, int, int], str]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".npy")]
        keys = [Path(name).stem for name in names]
        key = array_key
        if key is None:
            if len(keys) != 1:
                raise RuntimeError(f"{path} contains keys {keys}; pass --array-key")
            key = keys[0]
        member = f"{key}.npy"
        if member not in names:
            raise RuntimeError(f"{path} does not contain key {key!r}; available keys: {keys}")
        with archive.open(member) as fh:
            version = npy_format.read_magic(fh)
            if version == (1, 0):
                shape, _, dtype = npy_format.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, _, dtype = npy_format.read_array_header_2_0(fh)
            else:
                shape, _, dtype = npy_format.read_array_header_2_0(fh)
        if len(shape) != 3:
            raise RuntimeError(f"{path}:{key} must be 3D, got shape={shape}")
        return tuple(int(v) for v in shape), str(dtype)



def safe_sample_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    if not stem:
        raise RuntimeError(f"Cannot derive a safe sample name from {name!r}")
    return stem


def complete_volume_root(volume_root: Path) -> bool:
    return (volume_root / "HR" / "0" / "zarr.json").exists() and (volume_root / "REG" / "0" / "zarr.json").exists()

def write_group_metadata(path: Path, name: str, dataset_paths: Iterable[str]) -> None:
    datasets = []
    for ds_path in dataset_paths:
        datasets.append(
            {
                "path": ds_path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                ],
            }
        )
    metadata = {
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "datasets": datasets,
                        "name": f"/{name}" if name else "/",
                        "axes": [
                            {"name": "z", "type": "space"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                    }
                ],
            }
        },
        "zarr_format": 3,
        "consolidated_metadata": None,
        "node_type": "group",
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(json.dumps(metadata, indent=2) + "\n")


def create_array(path: Path, array: np.ndarray, chunks: tuple[int, int, int]) -> None:
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=3, shuffle=BloscShuffle.bitshuffle)
    z = zarr.create_array(
        str(path),
        shape=array.shape,
        dtype=array.dtype,
        chunks=tuple(min(c, s) for c, s in zip(chunks, array.shape)),
        zarr_format=3,
        compressors=[compressor],
        dimension_names=("z", "y", "x"),
        overwrite=True,
    )
    z[:] = array


def convert_pair(
    name: str,
    hr_npz: Path,
    reg_npz: Path,
    output_domain: Path,
    array_key: str | None,
    chunks: tuple[int, int, int],
    overwrite: bool,
) -> None:
    sample = safe_sample_name(name)
    volume_root = output_domain / f"{sample}_ome.zarr"
    temp_root = output_domain / f".{sample}_ome.zarr.tmp"
    if volume_root.exists():
        if not overwrite and complete_volume_root(volume_root):
            print(f"SKIP {volume_root} already exists")
            return
        shutil.rmtree(volume_root)
    if temp_root.exists():
        shutil.rmtree(temp_root)

    hr = load_npz_array(hr_npz, array_key)
    reg = load_npz_array(reg_npz, array_key)
    if hr.shape != reg.shape:
        raise RuntimeError(f"Shape mismatch for {name}: HR {hr.shape} vs REG {reg.shape}")

    write_group_metadata(temp_root, "", ["HR/0", "REG/0"])
    write_group_metadata(temp_root / "HR", "HR", ["0"])
    write_group_metadata(temp_root / "REG", "REG", ["0"])
    create_array(temp_root / "HR" / "0", hr, chunks)
    create_array(temp_root / "REG" / "0", reg, chunks)
    if not complete_volume_root(temp_root):
        raise RuntimeError(f"Incomplete zarr write for {name}: {temp_root}")
    temp_root.rename(volume_root)
    print(f"WROTE {volume_root} shape={hr.shape} dtype={hr.dtype}")


def validate_pairs(pairs: list[tuple[str, Path, Path]], array_key: str | None, limit: int | None) -> None:
    selected = pairs[:limit] if limit is not None else pairs
    print(f"paired files: {len(pairs)}; validating: {len(selected)}")
    bad = []
    for name, hr_path, reg_path in selected:
        hr_shape, hr_dtype = inspect_npz(hr_path, array_key)
        reg_shape, reg_dtype = inspect_npz(reg_path, array_key)
        if hr_shape != reg_shape:
            bad.append(f"{name}: HR {hr_shape} vs REG {reg_shape}")
        if len(hr_shape) != 3 or hr_shape[0] < 3:
            bad.append(f"{name}: model needs a 3D volume with depth >= 3, got {hr_shape}")
        print(f"PAIR {name}: shape={hr_shape} HR_dtype={hr_dtype} REG_dtype={reg_dtype}")
    if bad:
        raise RuntimeError("Invalid pairs:\n" + "\n".join(bad[:20]))

def validate_zarr_domain(output_domain: Path, reg_subpath: str) -> None:
    volumes = sorted(output_domain.glob("*.zarr"))
    if len(volumes) < 2:
        raise RuntimeError(f"Model training needs at least two zarr volumes, found {len(volumes)} in {output_domain}")
    bad = []
    for volume in volumes:
        hr_meta = volume / "HR" / "0" / "zarr.json"
        reg_meta = volume / reg_subpath / "zarr.json"
        if not hr_meta.exists() or not reg_meta.exists():
            bad.append(f"{volume}: missing HR/0 or {reg_subpath}")
            continue
        hr_shape = tuple(json.loads(hr_meta.read_text())["shape"])
        reg_shape = tuple(json.loads(reg_meta.read_text())["shape"])
        if reg_shape != hr_shape:
            bad.append(f"{volume}: REG {reg_shape}, HR {hr_shape}")
        if len(hr_shape) != 3 or hr_shape[0] < 3 or hr_shape[1] < 128 or hr_shape[2] < 128:
            bad.append(f"{volume}: shape may not support default patch training, shape={hr_shape}")
    if bad:
        raise RuntimeError("Converted zarr validation failed:\n" + "\n".join(bad[:20]))
    print(f"OK zarr domain {output_domain}: {len(volumes)} aligned volumes")


def model_check(data_root: Path, volume_glob: str, reg_subpath: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from horuseye_tr import ZarrTripletDataset

    dataset = ZarrTripletDataset(
        data_root,
        patch_size=128,
        samples_per_epoch=1,
        include_hr=True,
        volume_glob=volume_glob,
        reg_subpath=reg_subpath,
        hr_subpath="HR/0",
    )
    sample = dataset[0]
    shapes = {key: tuple(value.shape) for key, value in sample.items()}
    print(f"OK model dataset check: volumes={len(dataset.volume_roots)} sample_shapes={shapes}")


def main() -> int:
    args = parse_args()
    hr_dir = args.hr_dir.expanduser().resolve()
    reg_dir = args.reg_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_domain = domain_dir(data_root, args.domain)
    reg_subpath = "REG/0"

    if args.validate_only:
        validate_zarr_domain(output_domain, reg_subpath)
        if args.model_check:
            model_check(data_root, f"{output_domain.name}/*.zarr", reg_subpath)
        return 0

    pairs = pair_npz_files(hr_dir, reg_dir, args.strict_pairs)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError(f"No paired npz files found between {hr_dir} and {reg_dir}")

    validate_pairs(pairs, args.array_key, None)
    print(f"output domain: {output_domain}")
    if args.dry_run:
        for name, _, _ in pairs[:20]:
            print(f"PLAN {name} -> {output_domain / (safe_sample_name(name) + '_ome.zarr')}")
        if len(pairs) > 20:
            print(f"... {len(pairs) - 20} more")
        return 0

    output_domain.mkdir(parents=True, exist_ok=True)
    for name, hr_npz, reg_npz in pairs:
        convert_pair(
            name,
            hr_npz,
            reg_npz,
            output_domain,
            args.array_key,
            args.chunks,
            args.overwrite,
        )
    validate_zarr_domain(output_domain, reg_subpath)
    if args.model_check:
        model_check(data_root, f"{output_domain.name}/*.zarr", reg_subpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
