"""Score extracted report probabilities against the 58 expert-labelled studies.

Everything here reports a bootstrap CI alongside the point estimate. With 58
studies and as few as 9 positives for MCL, a bare AUC number is close to
meaningless on its own - the CI is the part you should actually read.

    uv run python src/evaluate.py artifacts/pseudo_qwen9b_gold.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent


def bootstrap_auc(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    if not stats:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(stats)), *np.percentile(stats, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    tr = pd.read_csv(ROOT / "data" / "train.csv")
    gold = tr[tr[LABELS].notna().all(axis=1)]
    preds = pd.read_parquet(a.preds)
    m = gold.merge(preds, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    if len(m) != len(gold):
        print(f"WARNING: matched {len(m)}/{len(gold)} gold studies")

    print(f"n = {len(m)} studies\n")
    print(f"{'finding':18} {'pos':>4} {'AUC':>6}   {'95% CI':>14}")
    print("-" * 48)
    aucs = []
    for c in LABELS:
        y = m[f"{c}_true"].to_numpy().astype(int)
        p = m[f"{c}_pred"].to_numpy().astype(float)
        auc = roc_auc_score(y, p)
        _, lo, hi = bootstrap_auc(y, p, a.n_boot)
        aucs.append(auc)
        print(f"{c:18} {y.sum():4d} {auc:6.3f}   [{lo:.3f}, {hi:.3f}]")
    print("-" * 48)
    print(f"{'MEAN':18} {'':4} {np.mean(aucs):6.3f}")
    print(
        "\nNote: the competition metric is computed on a hidden test set whose "
        "base rates almost certainly differ from these 58 enriched studies. "
        "Treat this as a relative ranking signal between variants, not as a "
        "leaderboard prediction."
    )


if __name__ == "__main__":
    main()
