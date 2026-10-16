"""What does a series' plane and sequence type buy, once averaging count is held fixed?

`plane_probe.py` scores each plane on its own series and compares against the
mean over all of them. That comparison is not clean: a study's prediction is a
mean over whatever series it contributed, so a single-plane score is computed
over roughly a third as many views and carries the noise of the smaller average.
The probe therefore cannot tell "this plane carries less signal" from "this
study was averaged over fewer series", and it reported -0.0152 for best-plane
selection partly on the second effect.

The control here is a matched-count draw: for every study, take the same NUMBER
of series the bucket gave it, but drawn at random from all of that study's
series regardless of plane or sequence. Anything a bucket scores above its own
matched draw is what the bucket's identity is worth. Everything below is the
count.

Buckets scored: plane, the competition's own Fluid_Sensitive/Fat_Suppression
flag (present in test_series.csv, so usable at inference), their crossing, and
- inside the non-FS bucket, which is where forum thread 741310 says T1, PD and
T2 are silently merged - the sequence type derived from DICOM TE/TR.

    uv run python src/seq_probe_analysis.py artifacts/preds_plane_m38seed1.npz
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DRAWS = 40          # matched-count draws; the control's own sd is reported


def macro(P, T, U):
    dp = pd.DataFrame(P, columns=LABELS).assign(uid=U).groupby("uid").mean()
    dt = pd.DataFrame(T, columns=LABELS).assign(uid=U).groupby("uid").first()
    out = {}
    for c in LABELS:
        y = (dt[c].to_numpy() > 0.5).astype(int)
        if len(np.unique(y)) > 1:
            out[c] = float(roc_auc_score(y, dp[c].to_numpy()))
    return (float(np.mean(list(out.values()))) if out else float("nan")), out


def matched_control(P, T, U, mask, rng):
    """Same per-study series COUNT as `mask`, drawn from all of that study's series."""
    want = pd.Series(mask).groupby(U).sum()
    rows = []
    by_study = pd.Series(np.arange(len(U))).groupby(U).apply(list)
    for st, k in want.items():
        if k == 0:
            continue
        pool = by_study[st]
        rows.extend(rng.choice(pool, size=int(min(k, len(pool))), replace=False))
    rows = np.array(rows, dtype=int)
    return macro(P[rows], T[rows], U[rows])


def bucket_table(P, T, U, buckets, title, rng):
    print(f"\n=== {title} ===")
    print(f"{'bucket':22}{'series':>8}{'macro':>9}{'control':>9}{'+-':>7}{'bucket-control':>16}")
    rows = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            continue
        m, per = macro(P[mask], T[mask], U[mask])
        ctl = [matched_control(P, T, U, mask, rng)[0] for _ in range(DRAWS)]
        c, sd = float(np.mean(ctl)), float(np.std(ctl))
        rows[name] = (m, per, c, sd)
        print(f"{name:22}{int(mask.sum()):8d}{m:9.4f}{c:9.4f}{sd:7.4f}{m - c:+16.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    z = np.load(a.preds, allow_pickle=True)
    P, T, U, PL, SUID = z["P"], z["T"], z["U"], z["PL"], z["SUID"]

    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet").set_index("SeriesInstanceUID")
    FS = ds.Fat_Suppression.to_dict()
    FL = ds.Fluid_Sensitive.to_dict()
    fs = np.array([FS.get(s, -1) for s in SUID])
    fl = np.array([FL.get(s, -1) for s in SUID])
    print(f"{len(P)} series, {len(set(U))} studies, "
          f"mean {len(P)/len(set(U)):.2f} series/study")
    print(f"Fat_Suppression == Fluid_Sensitive on {(fs == fl).mean():.1%} of series")

    base, base_per = macro(P, T, U)
    print(f"\nuniform mean over every series: macro {base:.4f}")

    bucket_table(P, T, U, {p: PL == p for p in ("Sagittal", "Coronal", "Axial")},
                 "plane", rng)
    bucket_table(P, T, U, {"FS (fluid-sens)": fs == 1, "non-FS": fs == 0},
                 "competition flag (available in test_series.csv)", rng)
    bucket_table(P, T, U,
                 {f"{p[:4]}/{'FS' if f else 'nonFS'}": (PL == p) & (fs == f)
                  for p in ("Sagittal", "Coronal", "Axial") for f in (1, 0)},
                 "plane x flag", rng)

    # Thread 741310: inside non-FS, T1 / PD / T2 are merged. Derive them from
    # TE/TR, which every series has, rather than from the description, which
    # 6% of series do not.
    meta = pd.read_parquet(ROOT / "artifacts" / "scanner_meta.parquet").set_index("SeriesInstanceUID")
    te = pd.to_numeric(meta.EchoTime, errors="coerce").to_dict()
    tr = pd.to_numeric(meta.RepetitionTime, errors="coerce").to_dict()

    def seq_of(s):
        e, r = te.get(s, np.nan), tr.get(s, np.nan)
        if not np.isfinite(e) or not np.isfinite(r):
            return "unknown"
        if e < 40 and r < 900:
            return "T1"
        if e < 40:
            return "PD"
        return "T2"

    sq = np.array([seq_of(s) for s in SUID])
    print(f"\nnon-FS series by derived type: "
          f"{pd.Series(sq[fs == 0]).value_counts().to_dict()}")
    bucket_table(P, T, U,
                 {f"non-FS {t}": (fs == 0) & (sq == t) for t in ("T1", "PD", "T2", "unknown")},
                 "inside the non-FS bucket (forum 741310)", rng)

    print("\nper-finding, competition flag, against the uniform mean:")
    mfs, per_fs = macro(P[fs == 1], T[fs == 1], U[fs == 1])
    mnf, per_nf = macro(P[fs == 0], T[fs == 0], U[fs == 0])
    print(f"{'finding':18}{'FS':>8}{'non-FS':>9}{'mean':>9}{'FS-nonFS':>11}")
    for c in LABELS:
        print(f"{c:18}{per_fs.get(c, np.nan):8.3f}{per_nf.get(c, np.nan):9.3f}"
              f"{base_per.get(c, np.nan):9.3f}{per_fs.get(c, np.nan) - per_nf.get(c, np.nan):+11.3f}")


if __name__ == "__main__":
    main()
