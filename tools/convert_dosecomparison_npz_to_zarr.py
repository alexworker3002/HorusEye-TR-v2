#!/usr/bin/env python3
"""Convert DoseComparison npz CT volumes into HorusEye-compatible zarr datasets."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.lib.format as npy_format
import zarr
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle


DEFAULT_SOURCE = Path("/mnt/Longxi_ET1/DoseComparison")
DATASET_DIRS = {
    "FBP": "004-FBP",
    "ASIR": "005-ASIR",
}
FILENAME_RE = re.compile(r"^(?P<sample>\d+)_(?P<dose>\d+)KV_(?P<kind>FBP|ASIR)\.npz$")


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
            "Convert /mnt/Longxi_ET1/DoseComparison npz files into data/004-FBP "
            "and data/005-ASIR zarr datasets."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="DoseComparison npz directory.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Local data root. Default: data")
    parser.add_argument("--array-key", default="data", help="NPZ array key. Default: data")
    parser.add_argument("--chunks", type=parse_shape, default=(16, 128, 128), help="Zarr chunk shape z,y,x. Default: 16,128,128")
    parser.add_argument("--kind", choices=("FBP", "ASIR"), default=None, help="Convert only one reconstruction kind.")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many samples per kind.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing complete zarr volume roots.")
    parser.add_argument("--dry-run", action="store_true", help="Only report parsed mapping and planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate already converted zarr datasets without writing.")
    parser.add_argument("--model-check", action="store_true", help="Instantiate ZarrTripletDataset for converted datasets.")
    return parser.parse_args()


def inspect_npz(path: Path, array_key: str) -> tuple[tuple[int, int, int], str]:
    with zipfile.ZipFile(path) as archive:
        member = f"{array_key}.npy"
        names = archive.namelist()
        if member not in names:
            keys = [Path(name).stem for name in names if name.endswith(".npy")]
            raise RuntimeError(f"{path} does not contain key {array_key!r}; available keys: {keys}")
        with archive.open(member) as fh:
            version = npy_format.read_magic(fh)
            if version == (1, 0):
                shape, _, dtype = npy_format.read_array_header_1_0(fh)
            else:
                shape, _, dtype = npy_format.read_array_header_2_0(fh)
    if len(shape) != 3:
        raise RuntimeError(f"{path}:{array_key} must be a 3D z,y,x volume, got shape={shape}")
    return tuple(int(v) for v in shape), str(dtype)


def load_npz_array(path: Path, array_key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        if array_key not in npz:
            raise RuntimeError(f"{path} does not contain key {array_key!r}; available keys: {list(npz.keys())}")
        array = npz[array_key]
        if array.ndim != 3:
            raise RuntimeError(f"{path}:{array_key} must be a 3D z,y,x volume, got shape={array.shape}")
        return np.asarray(array)


def discover(source: Path) -> dict[str, dict[str, dict[int, Path]]]:
    groups: dict[str, dict[str, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    ignored = []
    for path in sorted(source.glob("*.npz")):
        if path.name.startswith("._"):
            ignored.append(path.name)
            continue
        match = FILENAME_RE.match(path.name)
        if not match:
            ignored.append(path.name)
            continue
        groups[match.group("kind")][match.group("sample")][int(match.group("dose"))] = path
    if ignored:
        print(f"WARN ignored {len(ignored)} non-data files, first entries: {ignored[:10]}")
    return groups


def role_mapping(kind: str, dose_paths: dict[int, Path]) -> dict[str, tuple[int, Path]]:
    doses = sorted(dose_paths)
    if len(doses) < 2:
        raise RuntimeError(f"{kind} sample needs at least two doses for REG/0 and HR/0, got {doses}")

    mapping = {
        "REG/0": (doses[0], dose_paths[doses[0]]),
        "HR/0": (doses[-1], dose_paths[doses[-1]]),
    }
    if len(doses) >= 3:
        mapping["HR/2"] = (doses[1], dose_paths[doses[1]])
    return mapping


def write_group_metadata(path: Path, name: str, dataset_paths: Iterable[str], scales: dict[str, list[float]] | None = None) -> None:
    datasets = []
    for ds_path in dataset_paths:
        datasets.append(
            {
                "path": ds_path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": (scales or {}).get(ds_path, [1.0, 1.0, 1.0])},
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


def complete_volume_root(volume_root: Path, has_hr2: bool) -> bool:
    required = [
        volume_root / "REG" / "0" / "zarr.json",
        volume_root / "HR" / "0" / "zarr.json",
    ]
    if has_hr2:
        required.append(volume_root / "HR" / "2" / "zarr.json")
    return all(path.exists() for path in required)


def convert_sample(
    sample: str,
    kind: str,
    mapping: dict[str, tuple[int, Path]],
    output_dir: Path,
    array_key: str,
    chunks: tuple[int, int, int],
    overwrite: bool,
) -> None:
    has_hr2 = "HR/2" in mapping
    volume_root = output_dir / f"{sample}_ome.zarr"
    temp_root = output_dir / f".{sample}_ome.zarr.tmp"
    if volume_root.exists():
        if not overwrite and complete_volume_root(volume_root, has_hr2):
            print(f"SKIP {volume_root} already exists")
            return
        shutil.rmtree(volume_root)
    if temp_root.exists():
        shutil.rmtree(temp_root)

    arrays = {role: load_npz_array(path, array_key) for role, (_, path) in mapping.items()}
    shapes = {role: array.shape for role, array in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"{kind} sample {sample} dose shapes are not aligned: {shapes}")

    root_paths = ["REG/0", "HR/0"] + (["HR/2"] if has_hr2 else [])
    write_group_metadata(temp_root, "", root_paths)
    write_group_metadata(temp_root / "REG", "REG", ["0"])
    hr_paths = ["0"] + (["2"] if has_hr2 else [])
    hr_scales = {"0": [1.0, 1.0, 1.0]}
    if has_hr2:
        hr_scales["2"] = [2.0, 2.0, 2.0]
    write_group_metadata(temp_root / "HR", "HR", hr_paths, hr_scales)

    create_array(temp_root / "REG" / "0", arrays["REG/0"], chunks)
    create_array(temp_root / "HR" / "0", arrays["HR/0"], chunks)
    if has_hr2:
        create_array(temp_root / "HR" / "2", arrays["HR/2"], chunks)

    if not complete_volume_root(temp_root, has_hr2):
        raise RuntimeError(f"Incomplete zarr write for {kind} sample {sample}: {temp_root}")
    temp_root.rename(volume_root)

    doses = ", ".join(f"{role}={dose}KV" for role, (dose, _) in sorted(mapping.items()))
    print(f"WROTE {volume_root} shape={next(iter(shapes.values()))} {doses}")


def validate_source(groups: dict[str, dict[str, dict[int, Path]]], kinds: Iterable[str], array_key: str) -> None:
    for kind in kinds:
        samples = groups.get(kind, {})
        if not samples:
            raise RuntimeError(f"No {kind} npz files found")
        print(f"{kind}: {len(samples)} samples")
        for sample, dose_paths in sorted(samples.items()):
            mapping = role_mapping(kind, dose_paths)
            info = {}
            for role, (dose, path) in mapping.items():
                info[role] = (dose, *inspect_npz(path, array_key))
            shapes = {role: value[1] for role, value in info.items()}
            if len(set(shapes.values())) != 1:
                raise RuntimeError(f"{kind} sample {sample} source shapes mismatch: {info}")
            details = ", ".join(f"{role}={dose}KV/{shape}/{dtype}" for role, (dose, shape, dtype) in sorted(info.items()))
            print(f"  {sample}: {details}")


def validate_zarr(output_dir: Path, require_hr2: bool) -> None:
    volumes = sorted(output_dir.glob("*.zarr"))
    if len(volumes) < 2:
        raise RuntimeError(f"Model training needs at least two zarr volumes, found {len(volumes)} in {output_dir}")
    bad = []
    for volume in volumes:
        reg_meta = volume / "REG" / "0" / "zarr.json"
        hr_meta = volume / "HR" / "0" / "zarr.json"
        hr2_meta = volume / "HR" / "2" / "zarr.json"
        if not reg_meta.exists() or not hr_meta.exists() or (require_hr2 and not hr2_meta.exists()):
            bad.append(f"{volume}: missing REG/0, HR/0, or required HR/2")
            continue
        reg_shape = tuple(json.loads(reg_meta.read_text())["shape"])
        hr_shape = tuple(json.loads(hr_meta.read_text())["shape"])
        if reg_shape != hr_shape:
            bad.append(f"{volume}: REG {reg_shape}, HR {hr_shape}")
        if len(hr_shape) != 3 or hr_shape[0] < 3 or hr_shape[1] < 128 or hr_shape[2] < 128:
            bad.append(f"{volume}: shape may not support default patch training, shape={hr_shape}")
        if hr2_meta.exists() and tuple(json.loads(hr2_meta.read_text())["shape"]) != hr_shape:
            bad.append(f"{volume}: HR/2 shape does not match HR/0")
    if bad:
        raise RuntimeError("Converted zarr validation failed:\n" + "\n".join(bad[:20]))
    print(f"OK zarr domain {output_dir}: {len(volumes)} aligned volumes")


def model_check(data_root: Path, volume_glob: str, hr_subpath: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from horuseye_tr import ZarrTripletDataset

    dataset = ZarrTripletDataset(
        data_root,
        patch_size=128,
        samples_per_epoch=1,
        include_hr=True,
        volume_glob=volume_glob,
        reg_subpath="REG/0",
        hr_subpath=hr_subpath,
    )
    sample = dataset[0]
    shapes = {key: tuple(value.shape) for key, value in sample.items()}
    print(f"OK model dataset check: glob={volume_glob} hr={hr_subpath} volumes={len(dataset.volume_roots)} sample_shapes={shapes}")


def selected_kinds(kind: str | None) -> list[str]:
    return [kind] if kind else ["FBP", "ASIR"]


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    kinds = selected_kinds(args.kind)

    if args.validate_only:
        for kind in kinds:
            output_dir = data_root / DATASET_DIRS[kind]
            validate_zarr(output_dir, require_hr2=(kind == "FBP"))
            if args.model_check:
                model_check(data_root, f"{DATASET_DIRS[kind]}/*.zarr", "HR/0")
        return 0

    groups = discover(source)
    validate_source(groups, kinds, args.array_key)

    for kind in kinds:
        output_dir = data_root / DATASET_DIRS[kind]
        samples = sorted(groups[kind])
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"output domain: {output_dir}")
        if args.dry_run:
            for sample in samples:
                mapping = role_mapping(kind, groups[kind][sample])
                doses = ", ".join(f"{role}={dose}KV" for role, (dose, _) in sorted(mapping.items()))
                print(f"PLAN {kind} sample {sample}: {doses} -> {output_dir / (sample + '_ome.zarr')}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            convert_sample(sample, kind, role_mapping(kind, groups[kind][sample]), output_dir, args.array_key, args.chunks, args.overwrite)
        validate_zarr(output_dir, require_hr2=(kind == "FBP"))
        if args.model_check:
            model_check(data_root, f"{DATASET_DIRS[kind]}/*.zarr", "HR/0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
