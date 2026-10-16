"""How good are the teacher's labels, finding by finding?

Every image model here is trained on pseudo-labels extracted from radiology
reports, and validated against more of the same. That makes one number
invisible: how much of the remaining error is the model failing to see the
image, and how much is the label being wrong in the first place.

The 58 gold studies have real radiologist labels, and the teacher labelled them
too. Scoring the teacher's output against that truth gives, per finding, a
ceiling: a model that fits the teacher perfectly cannot beat the teacher's own
agreement with reality. Where that ceiling is low, more training is wasted
effort and the work belongs in relabelling instead.

    uv run python src/label_quality.py

Read the output as follows. `teacher AUC` is the ceiling. `model AUC` is what
the current image model reaches against the same truth. When the model is at or
above the teacher, the labels are the binding constraint. When it is well
below, there is still image signal left to extract.

58 studies is a small sample and every number here carries roughly +/- 0.06 at
one standard error, so treat the ordering as the finding and the values as
indicative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    truth = pd.read_csv(ROOT / "artifacts" / "gold58_truth.csv").set_index("StudyInstanceUID")
    teacher = pd.read_parquet(
        ROOT / "artifacts" / "pseudo_student_all.parquet"
    ).set_index("StudyInstanceUID")

    common = truth.index.intersection(teacher.index)
    print(f"{len(common)} studies with both real and teacher labels\n")

    # the image model's own per-finding scores on the same gold set, if present
    model = {}
    best = ROOT / "artifacts" / "image_model_m18cos" / "image_model.json"
    if best.exists():
        model = json.load(open(best)).get("per_finding", {})

    rows = []
    for c in LABELS:
        y = truth.loc[common, c].to_numpy()
        p = teacher.loc[common, c].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, p)
        # how far the teacher's continuous score separates the two classes, in
        # units of its own spread - a thresholded label loses this, so it says
        # whether the information is there but miscalibrated
        sep = (p[y > 0.5].mean() - p[y < 0.5].mean()) / (p.std() + 1e-9)
        rows.append({
            "finding": c,
            "n_pos": int(y.sum()),
            "teacher_auc": auc,
            "separation": sep,
            "model_auc": model.get(c, float("nan")),
        })

    df = pd.DataFrame(rows).sort_values("teacher_auc")
    print(f"{'finding':18s} {'pos':>4s} {'teacher':>8s} {'model':>8s} "
          f"{'model-teacher':>14s} {'sep':>6s}")
    for _, r in df.iterrows():
        gap = r.model_auc - r.teacher_auc
        flag = ""
        if r.teacher_auc < 0.80:
            flag = "  <- labels are the ceiling"
        elif gap < -0.05:
            flag = "  <- image signal left"
        print(f"{r.finding:18s} {r.n_pos:4d} {r.teacher_auc:8.3f} {r.model_auc:8.3f} "
              f"{gap:+14.3f} {r.separation:6.2f}{flag}")

    print(f"\nmacro teacher AUC {df.teacher_auc.mean():.3f} "
          f"| macro model AUC {df.model_auc.mean():.3f}")
    print("58 studies: one standard error is about 0.06, so read the ordering, "
          "not the third decimal")


if __name__ == "__main__":
    main()
