"""Build deterministic per-sequence shuffled copies of the KIP datasets.

Used by the shuffled-train (mode b) and shuffled-eval (mode c) controls: every
sequence gets an independent uniform permutation of its 20 tokens, labels are
kept UNCHANGED (copied byte-identically). For KIP a uniform shuffle re-randomises
the key-letter permutation, so the original label agrees with the post-shuffle
latent label only ~50% of the time -- which is exactly why these controls should
sit at chance.

Determinism: each row's permutation comes from ``random.Random(f"{seed}:{tag}:{split}:{row}")``,
so the shuffle depends only on (shuffle seed, tag, split, row index) -- stable
across machines and re-runs, and identical for every consumer.

Output: ``data/simulation/kip_shuffled/`` with the SAME filenames as the ordered
data, so training code just points ``--path_csv`` at the other directory. The
``<TAG>_rule.json`` sidecars are copied too. A ``shuffle_manifest.json`` records
the seed and the measured label-preservation rate per split.

CLI::

    DATA_DIR=/path/to/data python -m src.data.kip_shuffles
    DATA_DIR=/path/to/data python -m src.data.kip_shuffles --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.common import DATA_DIR, SEP, TESTED_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

SHUFFLE_SEED = 777
TAGS = ("kip_m4", "kip_m6")
SPLITS = ("train", "val", "test")
DEFAULT_OUT = DATA_DIR / "simulation" / "kip_shuffled"


def _tokens(raw: str) -> List[str]:
    if SEP in raw:
        return [t for t in raw.split(SEP) if t]
    return list(raw)


def shuffle_sequence(raw: str, tag: str, split: str, row: int, seed: int) -> str:
    toks = _tokens(raw)
    rng = random.Random(f"{seed}:{tag}:{split}:{row}")
    rng.shuffle(toks)
    return SEP.join(toks)


def build(src_dir: Path, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"shuffle_seed": seed, "source": str(src_dir), "splits": {}}

    for tag in TAGS:
        rule = json.loads((src_dir / f"{tag}_rule.json").read_text())
        kappa = {k: int(v) for k, v in rule["kappa"].items()}
        shutil.copy2(src_dir / f"{tag}_rule.json", out_dir / f"{tag}_rule.json")

        for split in SPLITS:
            X = pd.read_csv(src_dir / f"X_{split}_{tag}.csv")
            y = pd.read_csv(src_dir / f"y_{split}_{tag}.csv")

            shuffled = [
                shuffle_sequence(s, tag, split, i, seed)
                for i, s in enumerate(X["Sequences"].tolist())
            ]
            pd.DataFrame({"Sequences": shuffled}).to_csv(
                out_dir / f"X_{split}_{tag}.csv", index=False)
            shutil.copy2(src_dir / f"y_{split}_{tag}.csv", out_dir / f"y_{split}_{tag}.csv")

            # Measured label-preservation rate: fraction of rows where the
            # ORIGINAL label still equals the latent KIP label of the shuffled
            # sequence. Expected ~0.5.
            def _lab(toks: List[str]) -> int:
                ranks = [kappa[t] for t in toks if t in kappa]
                inv = sum(1 for a in range(len(ranks)) for b in range(a + 1, len(ranks))
                          if ranks[a] > ranks[b])
                return int(inv % 2 == 0)

            preserved = sum(
                1 for s, lab in zip(shuffled, y["Outcome"].tolist())
                if _lab(_tokens(s)) == int(lab)
            ) / len(shuffled)
            manifest["splits"][f"{tag}/{split}"] = {
                "rows": len(shuffled), "label_preserved_rate": round(preserved, 5),
            }
            logger.info("  %s/%s rows=%d label_preserved=%.4f",
                        tag, split, len(shuffled), preserved)

    (out_dir / "shuffle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(src_dir: Path, out_dir: Path, seed: int, n_check: int = 2000) -> bool:
    """Re-derive a sample of shuffled rows and compare with what is on disk."""
    ok = True
    for tag in TAGS:
        for split in SPLITS:
            src = pd.read_csv(src_dir / f"X_{split}_{tag}.csv")["Sequences"].tolist()
            got = pd.read_csv(out_dir / f"X_{split}_{tag}.csv")["Sequences"].tolist()
            if len(src) != len(got):
                logger.error("%s/%s row-count mismatch", tag, split)
                ok = False
                continue
            bad = sum(
                1 for i in range(min(n_check, len(src)))
                if shuffle_sequence(src[i], tag, split, i, seed) != got[i]
            )
            multiset_bad = sum(
                1 for i in range(min(n_check, len(src)))
                if sorted(_tokens(src[i])) != sorted(_tokens(got[i]))
            )
            status = "OK" if (bad == 0 and multiset_bad == 0) else "FAIL"
            logger.info("  %s/%s determinism_bad=%d multiset_bad=%d [%s]",
                        tag, split, bad, multiset_bad, status)
            ok = ok and bad == 0 and multiset_bad == 0
    return ok


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build shuffled KIP control datasets")
    p.add_argument("--src_dir", type=Path, default=TESTED_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    p.add_argument("--verify", action="store_true")
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        return 0 if verify(args.src_dir, args.out_dir, args.seed) else 1
    build(args.src_dir, args.out_dir, args.seed)
    return 0 if verify(args.src_dir, args.out_dir, args.seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
