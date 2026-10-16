"""Add K-fold split columns to the image dataset, grouped by report text.

Why this exists, 2026-09-04. Every model in this project trains on the same
3,827 studies and validates on the same 522. That is one split, reused
twenty-eight times, and it has two costs that no experiment here has ever
priced:

  1. **522 studies - 12% of the corpus - have never been trained on by any
     member.** Not by the incumbent, not by anything in the ensemble.
  2. **Every member is trained on identical data.** All the diversity in the
     0.947 ensemble comes from recipe changes on one fixed sample. Data
     diversity, which is what fold ensembling buys, has never been in it.

Teams reporting 0.94+ on the forum describe a "single (5-fold) model" as their
unit of work. Ours is a single fold, and the four-member ensemble that scores
0.947 averages four recipes over one sample rather than five samples.

The fold columns here let `train_image.py --split-col fold0` run unchanged:
fold k's studies read "val", every other non-gold study reads "train", and the
58 gold studies stay "gold_val" in every fold so the one clean instrument is
never trained on.

**Grouping is on report text, not on the study UID.** 197 studies share a
report verbatim with another study and therefore carry an identical
pseudo-label, so splitting those apart makes validation partly a memorisation
test - the same reasoning make_image_dataset.py applies to the fixed split.
Note that src/oof_predict.py folds on md5(StudyInstanceUID) % folds despite a
docstring claiming otherwise; that is a real leak there and this file does not
repeat it.

    uv run python src/make_folds.py --folds 5
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from clean_report import clean_report
from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "artifacts" / "image_dataset.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(a.dataset)
    train = pd.read_csv(ROOT / "data" / "train.csv")
    key = dict(zip(train.StudyInstanceUID,
                   train.Report.fillna("").map(clean_report)))

    df["report_key"] = df.StudyInstanceUID.map(key)
    if df.report_key.isna().any():
        raise SystemExit("some series have no report text to group on")

    # Assign whole report groups to folds, largest first, always into the
    # currently smallest fold. Hashing the group id would be simpler and would
    # also let one 37-study template group land entirely in one fold and skew
    # it; greedy balancing keeps the folds within a few studies of each other.
    gold = set(df.loc[df.split == "gold_val", "StudyInstanceUID"])
    pool = df[~df.StudyInstanceUID.isin(gold)]
    sizes = (pool.drop_duplicates("StudyInstanceUID")
                 .groupby("report_key").size().sort_values(ascending=False))
    rng = np.random.default_rng(a.seed)
    order = []
    for n, grp in sizes.groupby(sizes.values, sort=False):
        g = list(grp.index)
        rng.shuffle(g)
        order += g

    load = [0] * a.folds
    fold_of = {}
    for g in order:
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += int(sizes[g])

    fold_col = df.report_key.map(fold_of)
    for f in range(a.folds):
        name = f"fold{f}"
        df[name] = np.where(df.split == "gold_val", "gold_val",
                            np.where(fold_col == f, "val", "train"))

    df = df.drop(columns=["report_key"])
    df.to_parquet(a.dataset, index=False)

    print(f"{a.dataset}  +{a.folds} fold columns")
    print(f"{'fold':6}{'train':>9}{'val':>7}{'gold':>7}{'val pos rate':>14}")
    for f in range(a.folds):
        c = df.drop_duplicates("StudyInstanceUID")[f"fold{f}"]
        v = df[df[f"fold{f}"] == "val"].drop_duplicates("StudyInstanceUID")
        print(f"{f:<6}{int((c=='train').sum()):9d}{int((c=='val').sum()):7d}"
              f"{int((c=='gold_val').sum()):7d}{v[LABELS].mean().mean():14.3f}")

    # The check that matters: no study, and no report group, on both sides.
    for f in range(a.folds):
        spans = df.groupby("StudyInstanceUID")[f"fold{f}"].nunique()
        if int((spans > 1).sum()):
            raise SystemExit(f"fold{f}: {int((spans>1).sum())} studies span splits")
    print("no study spans a split in any fold")


if __name__ == "__main__":
    main()
