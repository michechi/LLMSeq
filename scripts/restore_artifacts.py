#!/usr/bin/env python3
"""Restore exact non-clinical artifacts from a local checkout or local Git objects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('synthetic-data', 'generated-figures', 'compiled-paper')


def safe_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
        raise ValueError('Manifest contains an unsafe path')
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError('Artifact path leaves the selected directory')
    return result


def validate_record(row: dict[str, str]) -> None:
    name, group = row['path'], row['group']
    safe_path(ROOT, name)
    allowed = (
        group == 'synthetic-data' and name.startswith('data/simulation/') and name.endswith('.csv')
        or group == 'generated-figures' and name.endswith('.png') and (
            name.startswith(('Bert_att_results/', 'Llama_att_results/'))
            or name.startswith('data/simulation/holes/')
            or '/' not in name and 'ngrams_diff' in name
        )
        or group == 'generated-figures' and name == 'data/simulation/holes/latex_tables.tex'
        or group == 'compiled-paper' and name in {'paper/NeurIPS.pdf', 'paper/NLDL/main.pdf'}
    )
    if not allowed or len(row['sha256']) != 64 or int(row['bytes']) < 0:
        raise ValueError('Manifest contains an unsupported artifact')


def read_artifact(row: dict[str, str], source: Path | None) -> bytes:
    if source is not None:
        payload = safe_path(source, row['path']).read_bytes()
    else:
        # Resolve the recorded path at the recorded commit before reading its blob.
        ref = f"{row['source_commit']}:{row['path']}"
        oid = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--verify', ref], stderr=subprocess.DEVNULL).decode().strip()
        if oid != row['git_blob']:
            raise ValueError(f"Git object mismatch: {row['path']}")
        payload = subprocess.check_output(['git', '-C', str(ROOT), 'cat-file', 'blob', oid], stderr=subprocess.DEVNULL)
    if len(payload) != int(row['bytes']) or hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError(f"Source checksum mismatch: {row['path']}")
    return payload


def restore(group: str, source: Path | None, destination: Path, dry_run: bool) -> dict[str, int]:
    with (ROOT / 'data/manifests/retired_artifacts.csv').open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['group'] == group]
    if not rows:
        raise ValueError('No artifacts in the selected group')
    if len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Duplicate manifest path')
    # Check the entire group before writing any file. Never replace conflicting data.
    for row in rows:
        validate_record(row)
        read_artifact(row, source)
        target = safe_path(destination, row['path'])
        if target.exists() and (not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']):
            raise ValueError(f"Refusing to replace an existing different file: {row['path']}")
    written = 0
    if not dry_run:
        for row in rows:
            target = safe_path(destination, row['path'])
            if target.exists():
                continue
            payload = read_artifact(row, source)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Link an already-complete temporary file atomically, with no overwrite.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                os.link(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            written += 1
    return {'validated': len(rows), 'written': written, 'bytes': sum(int(r['bytes']) for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-root', type=Path)
    source.add_argument('--from-git', action='store_true', help='Read local Git objects; never fetch')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        result = restore(args.group, args.source_root.resolve() if args.source_root else None,
                         args.destination.resolve(), args.dry_run)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Restore failed: {exc}\n')
    print(f"Validated {result['validated']} files ({result['bytes'] / 2**20:.2f} MiB); wrote {result['written']} files.")


if __name__ == '__main__':
    main()
