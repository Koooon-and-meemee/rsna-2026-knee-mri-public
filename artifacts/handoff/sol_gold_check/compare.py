"""Paired comparison of two report-extraction teachers on the 58 gold studies.

Absolute AUC on 58 studies has a huge standard error, but the two teachers are
scored on the SAME 58 studies, so their errors cancel and the paired difference
is far better resolved than either absolute number. This reports the paired
bootstrap, which is the statistic that should decide anything.

Rule of thumb established for this dataset (2,000-rep simulation at the actual
positive counts): two genuinely different pipelines need a macro-AUC gap of
~0.04 before the 58 can call it at 80% power. Anything smaller is a coin flip.

    uv run python src/compare.py A.parquet B.parquet --names qwen9b fable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from labels import LABELS

def _root() -> Path:
    """Walk up until data/train.csv appears.

    The literal `.parent.parent` this replaced resolved to artifacts/handoff/
    from this directory, three levels down, so the script died on a missing
    train.csv before printing anything. Walking up keeps it working wherever the
    bundle is copied to, which is the point of a bundle.
    """
    for d in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if (d / "data" / "train.csv").exists():
            return d
    return Path(__file__).resolve().parent


ROOT = _root()


def load_preds(path: Path, gold: pd.DataFrame) -> pd.DataFrame:
    """Accept either a parquet keyed by StudyInstanceUID or a CSV keyed by
    positional `id` (0-57, the order gold appears in train.csv)."""
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "StudyInstanceUID" not in df.columns:
        if "id" not in df.columns:
            raise SystemExit(f"{path}: needs StudyInstanceUID or id column")
        df = df.sort_values("id").reset_index(drop=True)
        if len(df) != len(gold):
            raise SystemExit(f"{path}: {len(df)} rows, expected {len(gold)}")
        df["StudyInstanceUID"] = gold["StudyInstanceUID"].to_numpy()
    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    return df[["StudyInstanceUID"] + LABELS]


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    na, nb = args.names

    tr = pd.read_csv(ROOT / "data" / "train.csv")
    gold = tr[tr[LABELS].notna().all(axis=1)].reset_index(drop=True)
    A = load_preds(args.a, gold).set_index("StudyInstanceUID").loc[gold.StudyInstanceUID]
    B = load_preds(args.b, gold).set_index("StudyInstanceUID").loc[gold.StudyInstanceUID]

    y = gold[LABELS].to_numpy().astype(int)
    pa, pb = A[LABELS].to_numpy(float), B[LABELS].to_numpy(float)

    print(f"n = {len(gold)} studies\n")
    print(f"{'finding':18} {'pos':>4} {na:>8} {nb:>8} {'diff':>8}")
    print("-" * 50)
    for j, c in enumerate(LABELS):
        aa = roc_auc_score(y[:, j], pa[:, j])
        bb = roc_auc_score(y[:, j], pb[:, j])
        print(f"{c:18} {y[:, j].sum():4d} {aa:8.3f} {bb:8.3f} {bb - aa:+8.3f}")
    ma, mb = macro_auc(y, pa), macro_auc(y, pb)
    print("-" * 50)
    print(f"{'MACRO':18} {'':4} {ma:8.3f} {mb:8.3f} {mb - ma:+8.3f}")

    # paired bootstrap over studies
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(y), len(y))
        ys = y[idx]
        if any(len(np.unique(ys[:, j])) < 2 for j in range(ys.shape[1])):
            continue
        diffs.append(macro_auc(ys, pb[idx]) - macro_auc(ys, pa[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_b_better = float((diffs > 0).mean())

    print(f"\npaired bootstrap of ({nb} - {na}), {len(diffs)} resamples")
    print(f"  mean diff      {diffs.mean():+.4f}")
    print(f"  95% CI         [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P({nb} > {na})   {p_b_better:.3f}")

    # The verdict keys off the bootstrap CI, not the raw gap. The ~0.04
    # rule of thumb was simulated for two INDEPENDENT pipelines; when the two
    # share a base model their errors are correlated, the paired differences
    # collapse toward zero variance, and the CI resolves gaps far below 0.04
    # that the blanket rule would wrongly call a coin flip.
    gap = abs(mb - ma)
    better, worse = (nb, na) if mb > ma else (na, nb)
    print()
    if lo > 0 or hi < 0:
        print(
            f"VERDICT: {better} is ahead of {worse} by {gap:.3f}. The 95% CI\n"
            f"[{lo:+.4f}, {hi:+.4f}] excludes zero, so this holds despite the small n -\n"
            f"the paired design cancels the per-study difficulty that dominates\n"
            f"the absolute AUCs."
        )
    elif gap < 0.04:
        print(
            f"VERDICT: gap {gap:.3f}, CI [{lo:+.4f}, {hi:+.4f}] spans zero. These two\n"
            f"are indistinguishable on the 58. Decide on cost, latency and\n"
            f"scalability to all 4,407 reports instead, or get more gold labels."
        )
    else:
        print(
            f"VERDICT: {better} leads by {gap:.3f}, but the CI [{lo:+.4f}, {hi:+.4f}]\n"
            f"still spans zero - the point estimate is large and the evidence is\n"
            f"not. Treat as unresolved."
        )


if __name__ == "__main__":
    main()
