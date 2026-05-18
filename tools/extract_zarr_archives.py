#!/usr/bin/env python3
"""Extract zarr tarballs into numbered domain directories."""

from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path


DOMAIN_RE = re.compile(r"^(?P<number>\d+)-(?P<domain>[^/]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find *zarr.tar.gz files and extract them into domain folders "
            "named like 001-Femur."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="~/Downloads",
        help="Directory containing *zarr.tar.gz files, or one archive file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./data",
        help="Extraction root directory. Default: ./data",
    )
    return parser.parse_args()


def domain_from_archive_name(path: Path) -> str:
    name = path.name
    if name.endswith(".zarr.tar.gz"):
        name = name[: -len(".zarr.tar.gz")]
    elif name.endswith(".tar.gz"):
        name = name[: -len(".tar.gz")]
    return name.split("_", 1)[0]


def find_archives(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.name.endswith("zarr.tar.gz") else []
    return sorted(source.glob("*zarr.tar.gz"))


def existing_domain_dirs(output_root: Path) -> dict[str, Path]:
    domains: dict[str, Path] = {}
    if not output_root.exists():
        return domains

    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = DOMAIN_RE.match(child.name)
        if match:
            domains[match.group("domain")] = child
    return domains


def next_domain_number(output_root: Path) -> int:
    highest = 0
    if output_root.exists():
        for child in output_root.iterdir():
            match = DOMAIN_RE.match(child.name)
            if match:
                highest = max(highest, int(match.group("number")))
    return highest + 1


def domain_dir_for(domain: str, output_root: Path, domains: dict[str, Path]) -> Path:
    if domain not in domains:
        number = next_domain_number(output_root)
        domains[domain] = output_root / f"{number:03d}-{domain}"
        domains[domain].mkdir(parents=True, exist_ok=True)
    return domains[domain]


def top_level_name(archive: Path) -> str:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            top = Path(member.name).parts[0] if member.name else ""
            if top:
                return top
    raise ValueError(f"{archive} is empty")


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe tar member path in {archive}: {member.name}")
        tar.extractall(destination)


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    archives = find_archives(source)
    if not archives:
        print(f"No *zarr.tar.gz files found in {source}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    domains = existing_domain_dirs(output_root)

    for archive in archives:
        domain = domain_from_archive_name(archive)
        destination = domain_dir_for(domain, output_root, domains)
        zarr_name = top_level_name(archive)
        zarr_destination = destination / zarr_name

        if zarr_destination.exists():
            print(f"SKIP {archive.name} -> {destination.name}/{zarr_name}")
            continue

        print(f"EXTRACT {archive.name} -> {destination.name}/")
        safe_extract(archive, destination)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
