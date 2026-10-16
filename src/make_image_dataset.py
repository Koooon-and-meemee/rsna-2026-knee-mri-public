"""Assemble the image model's training table: one row per series.

Joins the three things the image model needs and nothing else - a target, a
loss weight, and a split - so that the training script never has to reason
about where a label came from.

Decisions encoded here:

**Gold labels win, and gold studies are the validation set.** The 58
expert-labelled studies are the only clean measurement we have. Training on
them would buy 58 examples out of 4,407 and cost us the ability to tell
whether anything works, which is a bad trade. Their targets are still written
to the table so training-time code can assert it is not touching them.

**The split is grouped by study AND by report text.** A study contributes
several series - putting some in train and others in validation would let the
model recognise the knee rather than the finding. Beyond that, 197 studies
share a report verbatim with another study and therefore carry an identical
pseudo-label; splitting those apart makes validation partly a memorisation
test. Both constraints are satisfied by grouping on report text, since series
of one study trivially share it.

**Weights come from report_quality, and the pseudo-labels inherit the defects
of the report they came from.** A bilateral report produces a confidently
wrong target for whichever knee was not imaged; a 37-study template group
produces 37 identical targets. Neither is visible in the image.

    uv run python src/make_image_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clean_report import clean_report
from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "image_dataset.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.12,
                    help="extra validation drawn from pseudo-labelled studies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pseudo", type=Path, default=None,
                    help="per-study 라벨 parquet (기본: pseudo_student_all)")
    ap.add_argument("--out", type=Path, default=None, help="출력 parquet")
    a = ap.parse_args()

    train = pd.read_csv(ROOT / "data" / "train.csv")
    pseudo = pd.read_parquet(a.pseudo or (ROOT / "artifacts" / "pseudo_student_all.parquet"))
    quality = pd.read_csv(ROOT / "artifacts" / "report_quality.csv")
    series = pd.read_csv(ROOT / "data" / "train_series.csv")

    is_gold = train[LABELS].notna().all(axis=1)
    print(f"{len(train)} studies, {int(is_gold.sum())} with expert labels")

    # per-study targets: expert where we have them, student pseudo-labels elsewhere
    tgt = train[["StudyInstanceUID"]].copy()
    p = pseudo.set_index("StudyInstanceUID").reindex(train.StudyInstanceUID)
    for c in LABELS:
        tgt[c] = np.where(is_gold, train[c].to_numpy(), p[c].to_numpy())
    tgt["is_gold"] = is_gold.to_numpy()
    if tgt[LABELS].isna().any().any():
        raise SystemExit("some studies have neither an expert nor a pseudo label")

    tgt = tgt.merge(quality[["StudyInstanceUID", "sample_weight", "lang"]], on="StudyInstanceUID")
    tgt["report_key"] = train.Report.fillna("").map(clean_report).to_numpy()

    # split: gold is validation; the rest is grouped by report text
    rng = np.random.default_rng(a.seed)
    # np.asarray: the column is an Arrow-backed string array, which rng.shuffle
    # warns about and does not promise to permute correctly
    groups = np.asarray(tgt.loc[~tgt.is_gold, "report_key"].unique(), dtype=object)
    rng.shuffle(groups)
    target_n = int((~tgt.is_gold).sum() * a.val_frac)
    val_groups, n = set(), 0
    for g in groups:
        if n >= target_n:
            break
        val_groups.add(g)
        n += int((tgt.report_key == g).sum())

    tgt["split"] = np.where(
        tgt.is_gold, "gold_val", np.where(tgt.report_key.isin(val_groups), "val", "train")
    )

    # expand to series; every series of a study inherits its study's row
    df = series.merge(tgt, on="StudyInstanceUID", how="inner")
    df = df.drop(columns=["report_key"])
    out = a.out or OUT
    df.to_parquet(out, index=False)

    print(f"\n{len(df)} series across {df.StudyInstanceUID.nunique()} studies")
    print(f"\n{'split':10}{'studies':>9}{'series':>9}{'mean w':>9}")
    for s, g in df.groupby("split"):
        print(f"{s:10}{g.StudyInstanceUID.nunique():9d}{len(g):9d}{g.sample_weight.mean():9.3f}")

    # the leak checks that actually matter
    per_split = df.groupby("StudyInstanceUID").split.nunique()
    print(f"\nstudies spanning >1 split: {int((per_split > 1).sum())}")
    keys = tgt.groupby("report_key").split.nunique()
    n_span = int((keys > 1).sum())
    spanning_gold = int(
        tgt[tgt.report_key.isin(keys[keys > 1].index)].is_gold.sum()
    )
    print(f"report texts spanning >1 split: {n_span} ({spanning_gold} involve a gold study)")
    if n_span and n_span == spanning_gold:
        print(
            "  ^ benign here: gold studies are forced into gold_val, so a gold\n"
            "    study sharing a report with a pseudo-labelled one must straddle.\n"
            "    The image model never reads reports, the two studies are\n"
            "    different knees, and the gold target came from an expert rather\n"
            "    than from the shared text - so nothing leaks through it."
        )
    tr_uid = set(df[df.split == "train"].StudyInstanceUID)
    print(f"gold studies leaking into train: {int(tgt[tgt.is_gold].StudyInstanceUID.isin(tr_uid).sum())}")

    print(f"\n{'finding':18}{'gold pos':>10}{'train mean':>12}")
    for c in LABELS:
        g = df[df.split == "gold_val"].drop_duplicates("StudyInstanceUID")[c]
        t = df[df.split == "train"][c]
        print(f"{c:18}{int(g.sum()):10d}{t.mean():12.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
