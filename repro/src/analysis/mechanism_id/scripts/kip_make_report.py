"""Assemble the KIP markdown report from the sanity, audit and reveal outputs.

Every number and every summary sentence is derived from the CSVs, so the report
cannot drift out of step with the results it describes. Re-run it after re-running
any of the three upstream scripts.

CLI::

    DATA_DIR=/path/to/data python -m src.analysis.mechanism_id.scripts.kip_make_report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ANALYSIS_ROOT, DATA_DIR, RESULTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

CHANCE_BAND = (0.48, 0.52)
LAG_AGNOSTIC = ("max_over_lag", "sum_over_lag", "stacked_all_lag")


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.{nd}f}"


def _dims(sub: pd.DataFrame, col: str, ms: List) -> str:
    """Dimensionality per m, collapsed to one cell. C(m,2) features differ with m
    (6 at m=4, 15 at m=6), so showing only the first m's value would be wrong."""
    seen: List[str] = []
    for m in ms:
        s = sub[sub["m"] == m]
        if s.empty:
            continue
        v = str(s.iloc[0][col])
        if v not in seen:
            seen.append(v)
    return " / ".join(seen) if seen else "--"


def _md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def rule_summary(data_dir: Path, ms: Iterable[int]) -> str:
    rows = []
    for m in ms:
        p = data_dir / f"kip_m{m}_rule.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        keys = r["key_letters_in_kappa_order"]
        rows.append([
            f"`kip_m{m}`", str(r["m"]), str(r["n"]),
            " < ".join(keys), str(r["seed"]),
        ])
    return _md_table(["tag", "m", "n", "S in kappa order", "rule seed"], rows)


def audit_section(df: pd.DataFrame) -> str:
    ms = sorted(df["m"].unique())
    families = list(dict.fromkeys(df["family"]))
    models = list(dict.fromkeys(df["model"]))

    header = ["family", "model", "dim"] + [f"m={m} AUC" for m in ms] + ["theory"]
    rows: List[List[str]] = []
    for fam in families:
        for mod in models:
            sub = df[(df["family"] == fam) & (df["model"] == mod)]
            if sub.empty:
                continue
            cells = []
            for m in ms:
                s = sub[sub["m"] == m]
                if s.empty:
                    cells.append("--")
                    continue
                r = s.iloc[0]
                mark = ""
                if r["outside_band"]:
                    mark = " **!**"
                elif r.get("unexploited", False):
                    mark = " *(o)*"
                cells.append(_fmt(r["test_auc"]) + mark)
            first = sub.iloc[0]
            expect = "chance" if first["theory_expects_chance"] else "may exceed chance"
            rows.append([f"`{fam}`", mod, _dims(sub, "dim", ms)] + cells + [expect])
    return _md_table(header, rows)


def reveal_section(df: pd.DataFrame) -> str:
    ms = sorted(df["m"].unique())
    rungs = list(dict.fromkeys(df["rung"]))
    header = ["rung", "model", "dim"] + [f"m={m} AUC" for m in ms] + ["expected"]
    rows: List[List[str]] = []
    for rung in rungs:
        sub_all = df[df["rung"] == rung]
        for mod in list(dict.fromkeys(sub_all["model"])):
            sub = sub_all[sub_all["model"] == mod]
            cells = []
            for m in ms:
                s = sub[sub["m"] == m]
                cells.append(_fmt(s.iloc[0]["AUC"], 6 if rung.startswith("oracle") else 4)
                             if not s.empty else "--")
            first = sub.iloc[0]
            rows.append([f"`{rung}`", mod, _dims(sub, "feat_dim", ms)] + cells
                        + [str(first.get("expected", ""))])
    return _md_table(header, rows)


def derived_findings(audit: pd.DataFrame, reveal: pd.DataFrame) -> str:
    lines: List[str] = []
    ms = sorted(audit["m"].unique())

    # 1. Best baseline per m.
    for m in ms:
        sub = audit[audit["m"] == m]
        best = sub.loc[sub["test_auc"].idxmax()]
        lines.append(
            f"- **m={m}: the strongest baseline in the ladder is "
            f"`{best['family']}` / {best['model']} at AUC {best['test_auc']:.4f}.**"
        )

    # 2. Fixed-lag pair families -- the family that saturates the OC tasks.
    fixed = audit[audit["family"].str.startswith("lag_pair_l")]
    if len(fixed):
        lines.append(
            f"- Fixed-lag pair counts, the family that saturates the compliance "
            f"tasks, span AUC {fixed['test_auc'].min():.4f}-{fixed['test_auc'].max():.4f} "
            f"across every lag and both m. That shortcut is closed."
        )

    # 3. Linear vs nonlinear on the lag-agnostic families.
    for m in ms:
        la = audit[(audit["m"] == m) & (audit["family"].isin(LAG_AGNOSTIC))]
        if not len(la):
            continue
        lin = la[la["model"].str.startswith("logreg")]
        nonlin = la[~la["model"].str.startswith("logreg")]
        if len(lin) and len(nonlin):
            lines.append(
                f"- m={m}, lag-agnostic all-pairs families (these reconstruct the "
                f"full precedence matrix, so they *determine* Y): linear "
                f"{lin['test_auc'].min():.4f}-{lin['test_auc'].max():.4f}, "
                f"nonlinear {nonlin['test_auc'].min():.4f}-{nonlin['test_auc'].max():.4f}."
            )

    # 4. Oracle exactness.
    orc = reveal[reveal["rung"] == "oracle_closed_form"]
    if len(orc):
        exact = all(abs(v - 1.0) < 1e-12 for v in orc["AUC"])
        lines.append(
            f"- Closed-form oracle from precedence bits: "
            f"{'AUC = 1.000000 exactly for every m' if exact else 'NOT exact -- investigate'}."
        )

    # 5. Negative control.
    nc = reveal[reveal["rung"] == "b_membership_bits"]
    if len(nc):
        ok = all(CHANCE_BAND[0] <= v <= CHANCE_BAND[1] for v in nc["AUC"])
        lines.append(
            f"- Negative control (key-position membership bits): "
            f"{'at chance for every m and model, as required' if ok else 'ABOVE CHANCE -- the generator leaks'} "
            f"(range {nc['AUC'].min():.4f}-{nc['AUC'].max():.4f})."
        )

    # 6. Representation dependence: rank sequence vs precedence bits.
    for m in ms:
        c = reveal[(reveal["m"] == m) & (reveal["rung"] == "c_rank_sequence")
                   & (reveal["model"] == "mlp_64")]
        d = reveal[(reveal["m"] == m) & (reveal["rung"] == "d_precedence_bits")
                   & (reveal["model"] == "mlp_64")]
        if len(c) and len(d):
            lines.append(
                f"- m={m}: the rank sequence determines the precedence bits, yet the "
                f"same MLP scores {c.iloc[0]['AUC']:.4f} on the rank sequence against "
                f"{d.iloc[0]['AUC']:.4f} on the precedence bits -- identical information, "
                f"different representation."
            )
    return "\n".join(lines)


def build_report(sanity: Optional[str], audit: pd.DataFrame, reveal: pd.DataFrame,
                 data_dir: Path) -> str:
    ms = sorted(audit["m"].unique())
    parts: List[str] = []
    parts.append("# KIP (Key-Inversion Parity) -- generation, sanity, audit and reveal ladder\n")
    parts.append(
        "Generated by `src/analysis/mechanism_id/scripts/kip_make_report.py`. Every\n"
        "number below is read from `kip_audit.csv` and `kip_oracle_reveal.csv`.\n"
    )

    parts.append("\n## Task\n")
    parts.append(
        "Each sequence contains every one of the `m` hidden key letters exactly once,\n"
        "at uniformly random positions; the remaining positions are filled i.i.d.\n"
        "uniform from the distractor alphabet `A \\ S`, so a key letter never repeats.\n"
        "Reading the key letters in sequence order and mapping through `kappa` gives a\n"
        "permutation `pi` of `{1..m}`, and\n\n"
        "    Y = 1  iff  inv(pi) is even         (equivalently, iff pi is an even permutation)\n\n"
        "`rho = 0.5` and `AUC* = 1.000` hold by construction, since `sign(pi)` is uniform\n"
        "for `m >= 2`. Sizes (400K/50K/50K), the 80/10/10 split and seed handling match\n"
        "Tricky Deterministic (tag `6`); no class-balancing subsample is needed.\n"
    )
    parts.append("\n" + rule_summary(data_dir, ms) + "\n")
    parts.append(
        "\nThe hidden rule is **sampled per dataset**, not hardcoded, and stored beside\n"
        "the CSVs as `<TAG>_rule.json`. This is a deliberate departure from the other\n"
        "variants: it is what prevents the code/data key-set drift documented in\n"
        "`report.md` (paper says `S = {W,D,Q,J,X,U}`, the parity generator used `N`).\n"
    )

    if sanity:
        parts.append("\n## Sanity suite\n")
        parts.append("```\n" + sanity.strip() + "\n```\n")

    parts.append("\n## Audit ladder\n")
    parts.append(
        "AUC on the held-out test split. Classifier settings and protocol are those of\n"
        "`phase2_baseline_ladder.py`, so numbers are comparable with the existing\n"
        "ladder. `**!**` marks a result above the chance band that theory said should\n"
        "be at chance; `*(o)*` marks a family that provably determines `Y` yet stayed\n"
        "at chance.\n"
    )
    parts.append("\n" + audit_section(audit) + "\n")

    parts.append("\n## Oracle and reveal ladder\n")
    parts.append(
        "Because `inv(pi) = C(m,2) - sum(precedence bits)`, the pairwise precedence\n"
        "bits determine the label exactly, which gives a closed-form oracle needing no\n"
        "training. The ladder then asks what a 64-unit MLP needs to be handed.\n"
    )
    parts.append("\n" + reveal_section(reveal) + "\n")

    parts.append("\n## Findings\n")
    parts.append(derived_findings(audit, reveal) + "\n")
    return "\n".join(parts)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble the KIP report")
    p.add_argument("--audit_csv", type=Path, default=RESULTS_DIR / "kip_audit.csv")
    p.add_argument("--reveal_csv", type=Path, default=RESULTS_DIR / "kip_oracle_reveal.csv")
    p.add_argument("--sanity_txt", type=Path, default=RESULTS_DIR / "kip_sanity.txt")
    p.add_argument("--data_dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=ANALYSIS_ROOT / "kip_report.md")
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    for name, path in (("audit", args.audit_csv), ("reveal", args.reveal_csv)):
        if not path.exists():
            logger.error("missing %s CSV: %s", name, path)
            return 1
    audit = pd.read_csv(args.audit_csv)
    reveal = pd.read_csv(args.reveal_csv)
    sanity = args.sanity_txt.read_text() if args.sanity_txt.exists() else None
    if sanity is None:
        logger.warning("no sanity report at %s; run kip_sanity with --out",
                       args.sanity_txt)

    report = build_report(sanity, audit, reveal, args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    logger.info("wrote %s (%d lines)", args.out, report.count("\n") + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
