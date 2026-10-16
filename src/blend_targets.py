"""Blend the report-derived targets with the model's out-of-fold predictions.

The measurement this acts on, taken on the 58 gold studies: the teacher scores
0.8997, the image model 0.9094, and a blend of the two 0.9271. Text and pixels
fail differently, which is the condition under which averaging pays — the same
condition that made same-family model ensembling worthless here (+0.0035).

Two cautions carried over from the note that proposed this:

  - the 0.25 teacher weight was fitted on 58 studies, and 0.25 through 0.50 were
    all close. Trusting a decimal chosen on 58 points is exactly the error the
    instrument rule exists to prevent, so the default sits mid-range at 0.4 and
    the sweep is a flag, not a fitted value.
  - the model's predictions on studies it trained on are memorised. Only
    out-of-fold predictions can correct the teacher; in-fold ones would amplify
    whatever the model already got wrong. src/oof_predict.py produces them.

Gold studies keep their real labels untouched — they are the acceptance test and
must never be contaminated by a model's opinion of them.

    uv run python src/oof_predict.py --folds 5 --epochs 6
    uv run python src/blend_targets.py --teacher-weight 0.4
    uv run python -u src/train_image.py ... --dataset image_dataset_blend40.parquet

The output name is built from the weight as blend{round(w*100):02d}, so
--teacher-weight 0.4 writes image_dataset_blend40.parquet - NOT blend04, which
this line claimed until 24 August. It also silently overwrites an existing blend
at the same weight; pass --out to keep both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="oof_predictions.parquet")
    ap.add_argument("--teacher-weight", type=float, default=0.4,
                    help="weight on the report-derived label; the rest goes to "
                         "the out-of-fold image prediction")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    w = a.teacher_weight
    if not 0.0 <= w <= 1.0:
        raise SystemExit("--teacher-weight must be in [0, 1]")

    ds = pd.read_parquet(ART / "image_dataset.parquet")
    oof_path = ART / a.oof
    if not oof_path.exists():
        raise SystemExit(f"{oof_path} missing — run src/oof_predict.py first")
    oof = pd.read_parquet(oof_path)

    dup = oof.SeriesInstanceUID.duplicated().sum()
    if dup:
        raise SystemExit(f"{dup} series appear in more than one fold; folds overlap")

    merged = ds.merge(
        oof[["SeriesInstanceUID"] + LABELS],
        on="SeriesInstanceUID", how="left", suffixes=("", "_oof"),
    )
    have = merged[f"{LABELS[0]}_oof"].notna()
    print(f"{len(ds)} series, {int(have.sum())} carry an out-of-fold prediction "
          f"({have.mean() * 100:.1f}%)")

    # Only TRAINING rows are blended. Refining the validation targets too would
    # move the ruler and the measurement together: pv-sel and pv-hold would be
    # computed against different labels than every previous run, so the new
    # number could not be compared with the incumbent's and the whole experiment
    # would answer nothing. The val and gold_val rows keep the teacher's labels.
    is_train = (merged["split"] == "train").to_numpy()
    use = have.to_numpy() & is_train
    held = int((~is_train).sum())
    print(f"blending {int(use.sum())} training rows at teacher weight {w:.2f}; "
          f"{held} val/gold_val rows and {int((have.to_numpy() & ~is_train).sum())} "
          f"predicted-but-held-out rows keep the original targets")

    moved = []
    for c in LABELS:
        t = merged[c].to_numpy(float)
        p = merged[f"{c}_oof"].to_numpy(float)
        new = t.copy()
        new[use] = w * t[use] + (1.0 - w) * p[use]
        moved.append(np.abs(new[use] - t[use]).mean())
        merged[c] = new

    out = merged[list(ds.columns)]
    name = a.out or f"image_dataset_blend{int(round(w * 100)):02d}.parquet"
    out.to_parquet(ART / name, index=False)

    print(f"\n{'finding':18} {'mean |Δtarget|':>14}")
    for c, m in zip(LABELS, moved):
        print(f"{c:18} {m:14.4f}")
    print(f"\nwrote artifacts/{name}")
    print("Run it against the incumbent changing only --dataset. Validation "
          "targets are untouched, so pv-sel and pv-hold stay comparable with "
          "every previous run.")


if __name__ == "__main__":
    main()
