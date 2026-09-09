#!/usr/bin/env python3
"""Verify the exact source set and SHA-256 manifest for the AKI v3 handoff."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


FIXED_PATHS = {
    "pyproject.toml",
    "src/__init__.py",
    "src/mimic/__init__.py",
    "configs/mimic_aki_fox_h200_smd015_amendment_v3.yaml",
    "docs/mimic_aki_fox_h200.md",
    "docs/mimic_aki_protocol_amendment_v3.md",
    "scripts/mimic_aki_finalize_fox_transfer.sh",
    "scripts/mimic_aki_integrity_gate.py",
    "scripts/mimic_aki_verify_source_manifest.py",
    "scripts/mimic_aki_verify_transfer_manifest.py",
    "scripts/slurm/FOX/mimic_aki_h200_v3.slurm",
    "scripts/slurm/FOX/submit_mimic_aki_h200_v3.sh",
    "tests/test_aki_cohort.py",
}


def expected_paths(repo_root: Path) -> set[str]:
    result = set(FIXED_PATHS)
    for relative_directory in (Path("src/mimic/aki"), Path("tests/mimic/aki")):
        result.update(
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / relative_directory).rglob("*.py")
            if path.is_file() and not path.is_symlink()
        )
    return result


def parse_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid manifest row {line_number}")
        digest, filename = parts
        filename = filename.lstrip("*")
        relative = Path(filename)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != filename
            or filename in records
        ):
            raise ValueError(f"invalid manifest row {line_number}")
        records[filename] = digest.lower()
    return records


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo_root: Path, manifest: Path) -> None:
    records = parse_manifest(manifest)
    expected = expected_paths(repo_root)
    if set(records) != expected:
        raise ValueError(
            "source manifest path set differs from the required exact source set"
        )
    for relative, expected_digest in records.items():
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError("a required source path is missing or a symbolic link")
        if file_digest(source) != expected_digest:
            raise ValueError("a required source file has a SHA-256 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        verify(
            Path(args.repo_root).expanduser().resolve(),
            Path(args.manifest).expanduser().resolve(),
        )
    except (OSError, ValueError) as exc:
        print(f"source_manifest_status=FAIL error_type={type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2) from None
    print("source_manifest_status=PASS")


if __name__ == "__main__":
    main()
