"""Shared paths and constants. All paths resolve via env vars with sensible defaults."""
from __future__ import annotations

import os
from pathlib import Path

REPRO_ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("DATA_DIR", REPRO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPRO_ROOT / "results"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", REPRO_ROOT / "cache"))

TESTED_DIR = DATA_DIR / "simulation" / "tested"
PARITY_DIR = DATA_DIR / "simulation" / "parity_decomp"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

SEP = "\x1f"

DATASET_TAGS = {
    "tricky_det": "6",
    "tricky_rnd": "9",
    "parity": "test_just_pair",
    "naive": "alph",
}
