"""Validate a teacher batch's label CSV against its input batch.

A teacher agent writing 100 x 12 numbers by hand can drop a row, invent an id,
or collapse onto a handful of values. All three are silent failures that only
show up much later as a bad student model, so they get checked on arrival.

    uv run python src/check_labels.py                 # every labels_*.csv
    uv run python src/check_labels.py 001 002         # just these
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "artifacts" / "batches"


def check(tag: str) -> bool:
    labels_path = BATCHES / f"labels_{tag}.csv"
    batch_path = BATCHES / f"batch_{tag}.json"
    if not labels_path.exists():
        print(f"{tag}: MISSING {labels_path.name}")
        return False

    df = pd.read_csv(labels_path)
    problems = []

    missing_cols = [c for c in LABELS if c not in df.columns]
    if missing_cols:
        print(f"{tag}: MISSING COLUMNS {missing_cols}")
        return False

    partial = False
    if batch_path.exists():
        want = {r["id"] for r in json.loads(batch_path.read_text())}
        got = set(df.id)
        if got - want:
            problems.append(f"{len(got - want)} invented ids")
        if want - got:
            # a file still being written is short but not wrong; say which
            partial = not (got - want)
            problems.append(f"{len(want - got)} missing ids")
    if df.id.duplicated().any():
        problems.append(f"{int(df.id.duplicated().sum())} duplicate ids")

    v = df[LABELS]
    if v.isna().any().any():
        problems.append(f"{int(v.isna().sum().sum())} NaN")
    if v.min().min() < 0 or v.max().max() > 1:
        problems.append("values outside [0,1]")

    # A column pinned to one value carries no ranking information at all, and
    # the metric is rank-based. Flag the worst offender - but only on a full
    # batch. In a 17-row set, "76% one value" is what a rare finding at its
    # prior legitimately looks like, not a collapsed column.
    mode_share = max(v[c].value_counts(normalize=True).iloc[0] for c in LABELS)
    worst = max(LABELS, key=lambda c: v[c].value_counts(normalize=True).iloc[0])
    if len(df) >= 50 and mode_share > 0.60:
        problems.append(f"{worst} is {mode_share:.0%} one value")

    if problems:
        status = ("still writing? " if partial else "FAIL ") + "; ".join(problems)
    else:
        status = "ok"
    print(
        f"{tag}: {len(df):3d} rows  uniq/col {v.nunique().min():2d}-{v.nunique().max():2d}"
        f"  mean {v.to_numpy().mean():.2f}  mode {mode_share:.0%}  {status}"
    )
    return not problems


def main():
    tags = sys.argv[1:] or sorted(
        p.stem.split("_", 1)[1] for p in BATCHES.glob("labels_*.csv")
    )
    ok = [check(t) for t in tags]
    print(f"\n{sum(ok)}/{len(ok)} clean")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
