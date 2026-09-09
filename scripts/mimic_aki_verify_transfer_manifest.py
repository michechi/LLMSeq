#!/usr/bin/env python3
"""Verify the exact private CPU-to-FOX AKI transfer file set and hashes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


RAW_MIMIC_TABLE_BASENAMES = {
    "labevents.csv",
    "labevents.csv.gz",
    "admissions.csv",
    "admissions.csv.gz",
    "patients.csv",
    "patients.csv.gz",
    "d_labitems.csv",
    "d_labitems.csv.gz",
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        while filename.startswith("./"):
            filename = filename[2:]
        relative = Path(filename)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or not filename
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != filename
            or filename in records
        ):
            raise ValueError(f"invalid manifest row {line_number}")
        records[filename] = digest.lower()
    return records


def verify(run_root: Path, manifest: Path) -> None:
    if run_root.is_symlink() or manifest.is_symlink():
        raise ValueError("run root and manifest must not be symbolic links")
    manifest_relative = manifest.relative_to(run_root).as_posix()
    expected: set[str] = set()
    for path in run_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("the transfer tree contains a symbolic link")
        if not path.is_dir() and not path.is_file():
            raise ValueError("the transfer tree contains an unsupported entry type")
        if path.is_file() and path.name in RAW_MIMIC_TABLE_BASENAMES:
            raise ValueError("the transfer tree contains a raw MIMIC source-table basename")
        if path.is_file() and path != manifest:
            expected.add(path.relative_to(run_root).as_posix())
    records = parse_manifest(manifest)
    if manifest_relative in records or set(records) != expected:
        raise ValueError("transfer manifest path set differs from the exact file set")
    for relative, expected_digest in records.items():
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("a transferred file is missing or a symbolic link")
        if file_digest(path) != expected_digest:
            raise ValueError("a transferred file has a SHA-256 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        verify(
            Path(args.run_root).expanduser().resolve(),
            Path(args.manifest).expanduser().resolve(),
        )
    except (OSError, ValueError) as exc:
        print(f"transfer_manifest_status=FAIL error_type={type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2) from None
    print("transfer_manifest_status=PASS")


if __name__ == "__main__":
    main()
