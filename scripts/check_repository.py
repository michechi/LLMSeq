#!/usr/bin/env python3
"""Check tracked files for bulk artifacts, notebook output and recognizable secrets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 5 * 1024 * 1024
SECRET_PATTERNS = (
    re.compile(rb'\bhf_[A-Za-z0-9]{20,}\b'),
    re.compile(rb'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b'),
    re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
)
ARTIFACT_SUFFIXES = {'.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.joblib',
                     '.parquet','.arrow','.npy','.npz','.gz','.zip','.tar','.sif','.bin'}
DATA_FILES_ALLOWED = {
    'data/README.md', 'data/manifests/retired_artifacts.csv',
    'data/simulation/oc_completion/dataset_manifest.json',
    'data/simulation/oc_completion/pairs/pair_generation_report.json',
    'repro/data/README.md', 'repro/data/mimic/README.md',
}


def check_file(name: str, payload: bytes) -> list[str]:
    issues = []
    path = Path(name)
    if name.startswith(('data/', 'repro/data/', 'mimic_analysis/data/')) and name not in DATA_FILES_ALLOWED:
        issues.append('data file outside the explicit documentation/manifest list')
    if path.name == 'clinical_histories.txt':
        issues.append('clinical text export')
    if path.suffix in ARTIFACT_SUFFIXES:
        issues.append('serialized data, model or bulk archive')
    if path.suffix in {'.out','.err','.log'} or name.startswith(('logs/', 'scripts/slurm/logs/')):
        issues.append('raw runtime log')
    if path.name == '.DS_Store' or path.suffix in {'.bak','.tmp','.pyc','.pyo'}:
        issues.append('generated or temporary file')
    if path.name.startswith('.env') and path.name != '.env.example':
        issues.append('local environment file')
    if name.startswith(('Bert_att_results/','Llama_att_results/')):
        issues.append('generated figure gallery')
    if len(payload) > LIMIT:
        issues.append('file exceeds 5 MiB; use an external artifact and manifest')
    if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
        issues.append('recognizable credential; value omitted')
    if path.suffix == '.ipynb':
        try:
            notebook = json.loads(payload)
            if not isinstance(notebook, dict) or not isinstance(notebook.get('cells'), list):
                raise ValueError('invalid notebook structure')
            if any(cell.get('outputs') or cell.get('execution_count') is not None
                   for cell in notebook['cells'] if cell.get('cell_type') == 'code'):
                issues.append('notebook contains saved output or execution counts')
            if notebook.get('metadata', {}).get('widgets'):
                issues.append('notebook contains saved widget output')
        except (ValueError, UnicodeDecodeError, AttributeError):
            issues.append('invalid notebook JSON or structure')
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    names = subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
    failures = []
    total = 0
    for name in filter(None, names):
        path = root / name
        if path.is_symlink() or not path.is_file():
            failures.append((name, 'tracked path is missing or a symbolic link'))
            continue
        payload = path.read_bytes()
        total += len(payload)
        failures.extend((name, issue) for issue in check_file(name, payload))
    for name, issue in failures:
        print(f'{name}: {issue}')
    print(f"{sum(bool(n) for n in names)} tracked files, {total / 2**20:.2f} MiB, {len(failures)} issues")
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
