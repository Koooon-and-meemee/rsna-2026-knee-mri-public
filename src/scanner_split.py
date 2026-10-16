"""Add a scanner fingerprint and a scanner-grouped split to the dataset table.

Our validation splits studies by report text, which says nothing about which
machine produced the images. Measured on the extracted DICOM headers, 516 of
the 522 validation studies were scanned on equipment that also appears in
training - 98.9%. Whatever a model learns about a particular scanner's
appearance is therefore available to it at validation time, and our held-out
number cannot see the difference.

A competitor measured the size of that effect for a metadata-only model: 0.6516
macro AUC under random folds against 0.5981 with folds grouped by scanner, so
0.053 of it does not transfer, concentrated on the osteoarthritis columns
(0.074-0.081) and fracture (0.086). That is a different model to ours, but the
same leak is available through scanner-specific image appearance.

Whether this matters depends on something we cannot check: if the hidden test
comes from the same institutions as training, a scanner-overlapping validation
is the representative one and grouping would make it pessimistic. If the test
holds out sites, our number is optimistic by an unknown amount. This script
does not resolve that - it builds the instrument that measures it, so one
training run can say how large the effect is for our model rather than for
someone else's.

**Fingerprint granularity is a choice, not a given.** The five raw header
fields used verbatim give 3,261 groups over 4,407 studies - ImagingFrequency
carries six decimal places, so 63.685259 and 63.68525 are different scanners.
Rounding the frequency, normalising the software-version list, and keeping
manufacturer, model and coil gives 151 groups with the largest at 8% of
studies, which is coarse enough to hold out and fine enough to separate
machines.

    uv run python src/scanner_split.py

Adds two columns to artifacts/image_dataset.parquet and leaves `split`
untouched, so every existing number stays comparable:

    scanner_fp      the fingerprint string
    split_scanner   train / val / gold_val, with val drawn from whole
                    scanner groups that training never sees
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def norm_software(s):
    """Software version lists arrive as stringified Python lists."""
    s = re.sub(r"[\[\]',]", " ", str(s))
    return " ".join(sorted(set(s.split())))[:40]


def round_freq(s):
    """Imaging frequency to the nearest MHz; the decimals are per-exam drift."""
    try:
        return f"{float(s):.0f}"
    except (TypeError, ValueError):
        return ""


def main():
    meta = ROOT / "artifacts" / "scanner_meta.parquet"
    if not meta.exists():
        raise SystemExit(f"{meta} missing - run src/push_scanner_meta.py first")
    sm = pd.read_parquet(meta)
    sm["fp"] = (sm.Manufacturer + "|" + sm.ManufacturerModelName + "|"
                + sm.SoftwareVersions.map(norm_software) + "|"
                + sm.ImagingFrequency.map(round_freq) + "|" + sm.ReceiveCoilName)
    # a study is one exam on one machine; the mode guards against a stray series
    per_study = sm.groupby("StudyInstanceUID").fp.agg(lambda s: s.mode().iloc[0])

    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet")
    ds["scanner_fp"] = ds.StudyInstanceUID.map(per_study).fillna("unknown")

    # Gold studies keep their role: they are the only real labels and are not
    # available to give up. Everything else is assigned by whole scanner group.
    gold = set(ds.loc[ds.split == "gold_val", "StudyInstanceUID"])
    pool = ds[~ds.StudyInstanceUID.isin(gold)]
    counts = (pool.groupby("scanner_fp").StudyInstanceUID.nunique()
              .sort_values(ascending=False))

    # Fill the validation side with whole groups, smallest first, until it is
    # about the size of the existing 522-study val set. Taking small groups
    # keeps a single large site from becoming the entire validation set - which
    # would measure that one site rather than scanner generalisation.
    target = ds[ds.split == "val"].StudyInstanceUID.nunique()
    val_groups, n = [], 0
    for fp, c in counts.sort_values().items():
        if n >= target:
            break
        val_groups.append(fp)
        n += c
    val_groups = set(val_groups)

    def assign(row):
        if row.StudyInstanceUID in gold:
            return "gold_val"
        return "val" if row.scanner_fp in val_groups else "train"

    ds["split_scanner"] = ds.apply(assign, axis=1)
    ds.to_parquet(ROOT / "artifacts" / "image_dataset.parquet", index=False)

    st = ds.groupby("split_scanner").StudyInstanceUID.nunique()
    tr_fp = set(ds.loc[ds.split_scanner == "train", "scanner_fp"])
    va_fp = set(ds.loc[ds.split_scanner == "val", "scanner_fp"])
    print(f"{per_study.nunique()} scanner groups over {len(per_study)} studies")
    print("studies per split:", st.to_dict())
    print(f"validation drawn from {len(va_fp)} groups, "
          f"{len(va_fp & tr_fp)} of which also appear in training")
    print("(the overlap should be zero - that is the point)")


if __name__ == "__main__":
    main()
