"""Merge the per-batch teacher labels into one training table, and check them.

Each batch is scored by a separate teacher invocation. Separate invocations
anchor their probability scale slightly differently, and that drift lands on
the pseudo-labels as noise. make_batches.py already round-robins languages so
every batch is a miniature of the whole pool - which means any systematic
difference between batches is drift, not content, and is measurable here.

The check that matters is per-finding mean by batch. If batch 007 puts Effusion
at 0.71 and batch 002 at 0.44 on statistically identical samples, the student
learns the batch id, not the radiology.

    uv run python src/collect_labels.py
    uv run python src/collect_labels.py --normalise   # rank-align batch scales
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "artifacts" / "batches"  # --batches 로 덮어쓸 수 있다


def load() -> pd.DataFrame:
    frames = []
    for path in sorted(BATCHES.glob("labels_[0-9][0-9][0-9].csv")):
        df = pd.read_csv(path)
        df["batch"] = path.stem.split("_")[1]
        frames.append(df)
    if not frames:
        raise SystemExit("no labels_*.csv found")
    df = pd.concat(frames, ignore_index=True)

    # The relabel pass rescored studies whose text was repaired after their
    # batch was scored. It wins wherever it overlaps.
    relabel_path = BATCHES / "labels_relabel.csv"
    if relabel_path.exists():
        rl = pd.read_csv(relabel_path)
        n_before = len(df)
        df = df[~df.id.isin(rl.id)]
        rl["batch"] = "relabel"
        df = pd.concat([df, rl], ignore_index=True)
        print(f"relabel: replaced {n_before - len(df) + len(rl)} rows\n")
    return df


def report(df: pd.DataFrame) -> None:
    print(f"{len(df)} rows, {df.id.nunique()} unique ids, {df.batch.nunique()} batches")
    dupes = df.id.duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate ids")

    print("\n=== per-batch scale (mean probability by finding) ===")
    piv = df.groupby("batch")[LABELS].mean()
    hdr = "".join(f"{b:>8}" for b in piv.index)
    print(f"{'finding':18}{hdr}{'spread':>9}")
    print("-" * (18 + 8 * len(piv) + 9))
    spreads = []
    for c in LABELS:
        row = piv[c]
        spread = row.max() - row.min()
        spreads.append(spread)
        print(f"{c:18}" + "".join(f"{v:8.2f}" for v in row) + f"{spread:9.2f}")
    print("-" * (18 + 8 * len(piv) + 9))
    print(f"{'mean spread':18}" + " " * (8 * len(piv)) + f"{np.mean(spreads):9.2f}")
    print(
        "\nSpread is max-min of the per-batch mean. This is DESCRIPTIVE ONLY - do\n"
        "not read a large spread as teacher drift. Batches also differ in content:\n"
        "batch 003's reports average 498 characters against batch 000's 2,118, and\n"
        "thin reports genuinely carry fewer findings. src/label_drift.py is the\n"
        "test that separates the two, and it found the drift here to be 0.01-0.04."
    )

    print("\n=== tie structure (what the AUC actually sees) ===")
    print(f"{'finding':18}{'unique':>8}{'mode%':>8}{'mean':>8}")
    print("-" * 42)
    for c in LABELS:
        v = df[c]
        print(f"{c:18}{v.nunique():8d}{v.value_counts(normalize=True).iloc[0] * 100:7.1f}%{v.mean():8.2f}")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map each batch's per-finding scores onto the pooled distribution.

    Rank-based, so it removes a batch's scale offset without touching the
    within-batch ordering the teacher actually decided.
    """
    out = df.copy()
    for c in LABELS:
        pooled = np.sort(df[c].to_numpy())
        for b, idx in df.groupby("batch").groups.items():
            v = df.loc[idx, c]
            q = v.rank(pct=True, method="average").to_numpy()
            out.loc[idx, c] = np.quantile(pooled, np.clip(q, 0, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalise", action="store_true", help="rank-align batch scales")
    # A second labeller's output must never land on the first's. Both the input
    # directory and the output file move together, so one flag pair keeps the
    # two runs from sharing anything.
    ap.add_argument("--batches", type=Path, default=None, help="배치 라벨 디렉터리")
    ap.add_argument("--out", type=Path, default=None, help="출력 parquet")
    a = ap.parse_args()
    global BATCHES
    if a.batches:
        BATCHES = a.batches

    df = load()
    report(df)

    if a.normalise:
        df = normalise(df)
        print("\n=== after rank alignment ===")
        report(df)

    index = pd.read_csv(ROOT / "artifacts" / "teacher_subset_index.csv")
    uid = index.StudyInstanceUID.to_numpy()
    missing = df.id[df.id >= len(uid)]
    if len(missing):
        raise SystemExit(f"{len(missing)} ids out of range for the subset index")
    df.insert(0, "StudyInstanceUID", uid[df.id.to_numpy()])

    out = a.out or (ROOT / "artifacts" / "teacher_labels.parquet")
    df.sort_values("id").to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(df)} studies)")


if __name__ == "__main__":
    main()
