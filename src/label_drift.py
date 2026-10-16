"""Separate teacher-invocation drift from language and content in the labels.

The problem this exists to answer. Each batch is one teacher invocation, and
invocations anchor their probability scale slightly differently. In batches
005-009 that is harmless: they are language-interleaved, so batch and language
are orthogonal and the drift averages out across languages. Batches 000-004
predate the interleaving and are language-segregated (000 is 100% English, 004
is 0%), so there the batch effect and the language effect are the same effect
wearing two hats. If it is large, the student learns "English reports score
higher" - which is a property of our labelling process, not of knees.

You cannot read this off per-batch means, because batches also differ in
content: batch 003's reports average 498 characters against batch 000's 2,118,
and thin reports genuinely carry fewer findings.

The identification strategy:

  1. Use the Qwen-9B pass over all 4,407 reports as a CONTENT control. It is a
     single fixed model applied to every batch, so it cannot drift by batch.
  2. Fit, per finding, on the BALANCED batches only, where language is
     orthogonal to batch:
         teacher ~ slope * q9 + language effects
     Language effects are identified here and only here.
  3. Apply that fit to the segregated batches. Whatever offset remains, after
     content and language are accounted for, is that batch's own drift.

Caveat worth stating: the 9B is an imperfect content proxy and its accuracy is
itself language-dependent, so step 2 absorbs some of the 9B's own language bias
into the "language effect" term. That makes this test CONSERVATIVE - it will
under-report drift rather than invent it.

    uv run python src/label_drift.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "artifacts" / "batches"

# a batch counts as language-balanced if no single language dominates it
BALANCED_MAX_SHARE = 0.60


def build() -> pd.DataFrame:
    index = pd.read_csv(ROOT / "artifacts" / "teacher_subset_index.csv")
    q9 = pd.read_parquet(ROOT / "artifacts" / "pseudo_qwen9b_all.parquet")
    q9 = q9.set_index("StudyInstanceUID")

    frames = []
    for path in sorted(BATCHES.glob("labels_[0-9][0-9][0-9].csv")):
        df = pd.read_csv(path)
        df["batch"] = path.stem.split("_")[1]
        frames.append(df)
    if not frames:
        raise SystemExit("no labels_*.csv found")
    df = pd.concat(frames, ignore_index=True)

    relabel = BATCHES / "labels_relabel.csv"
    if relabel.exists():
        rl = pd.read_csv(relabel)
        # Drop the relabelled studies entirely rather than folding either
        # version in. Their original scores came from corrupted text, so they
        # carry repair noise on top of drift; their new scores came from a
        # different invocation than the batch being measured. Neither answers
        # "what scale did invocation N anchor on", and 17 rows is a cheap loss.
        df = df[~df.id.isin(rl.id)]

    df["StudyInstanceUID"] = index.StudyInstanceUID.to_numpy()[df.id.to_numpy()]
    df["lang"] = index.lang.to_numpy()[df.id.to_numpy()]
    for c in LABELS:
        df[f"q9_{c}"] = q9[c].reindex(df.StudyInstanceUID).to_numpy()
    return df.dropna(subset=[f"q9_{c}" for c in LABELS])


def classify_batches(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    share = df.groupby("batch").lang.value_counts(normalize=True).groupby("batch").max()
    balanced = sorted(share[share <= BALANCED_MAX_SHARE].index)
    segregated = sorted(share[share > BALANCED_MAX_SHARE].index)
    return balanced, segregated


def fit_and_score(df: pd.DataFrame, balanced: list[str], finding: str) -> pd.Series:
    """Return each batch's residual offset for one finding, in probability units."""
    langs = sorted(df.lang.unique())
    x_q9 = df[f"q9_{finding}"].to_numpy()
    y = df[finding].to_numpy()

    # design: intercept, content slope, one dummy per language after the first
    def design(rows: np.ndarray) -> np.ndarray:
        cols = [np.ones(rows.sum()), x_q9[rows]]
        for lg in langs[1:]:
            cols.append((df.lang.to_numpy()[rows] == lg).astype(float))
        return np.column_stack(cols)

    train = df.batch.isin(balanced).to_numpy()
    if train.sum() < 30:
        return pd.Series(dtype=float)
    beta, *_ = np.linalg.lstsq(design(train), y[train], rcond=None)

    all_rows = np.ones(len(df), dtype=bool)
    resid = y - design(all_rows) @ beta
    return pd.Series(resid, index=df.index).groupby(df.batch).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="offset above which a batch is called drifted")
    a = ap.parse_args()

    df = build()
    balanced, segregated = classify_batches(df)
    print(f"{len(df)} labelled studies across {df.batch.nunique()} batches")
    print(f"  language-balanced (fit here): {balanced}")
    print(f"  language-segregated (tested): {segregated}")
    if not balanced:
        raise SystemExit(
            "\nNo balanced batch is labelled yet - language effects are not "
            "identifiable.\nRun this again once batches 005-009 are in."
        )
    print()

    table = pd.DataFrame({c: fit_and_score(df, balanced, c) for c in LABELS})
    if table.empty:
        raise SystemExit("not enough balanced data to fit")

    print("=== residual offset by batch (content + language removed) ===")
    print("positive = this invocation scored HIGHER than content and language explain\n")
    hdr = "".join(f"{b:>8}" for b in table.index)
    print(f"{'finding':18}{hdr}")
    print("-" * (18 + 8 * len(table)))
    for c in LABELS:
        row = table[c]
        print(f"{c:18}" + "".join(f"{v:+8.2f}" for v in row))
    print("-" * (18 + 8 * len(table)))
    print(f"{'mean |offset|':18}" + "".join(f"{abs(table.loc[b]).mean():8.2f}" for b in table.index))

    worst = table.abs().mean(axis=1)
    bad = [b for b in segregated if worst.get(b, 0) > a.threshold]
    print()
    if not bad:
        print(
            f"VERDICT: no segregated batch exceeds {a.threshold:.2f} mean absolute\n"
            f"offset. The language/batch confound is not measurably biasing the\n"
            f"labels. Do NOT rank-align - it would only add noise."
        )
    else:
        print(
            f"VERDICT: batches {bad} drift beyond {a.threshold:.2f} after content and\n"
            f"language are accounted for, and their language is confounded with the\n"
            f"batch. Down-weight them, or rank-align with\n"
            f"  uv run python src/collect_labels.py --normalise"
        )
    out = ROOT / "artifacts" / "label_drift.csv"
    table.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
