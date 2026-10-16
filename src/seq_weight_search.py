"""Is the equal-weight average over a study's series leaving anything on the table?

`seq_probe_analysis.py` measured, with the averaging count held fixed, that a
fluid-sensitive/fat-suppressed series is worth +0.0118 macro more than a random
series of the same study and a non-FS one is worth -0.0186. The study
prediction is currently the unweighted mean over every series, so those two are
pooled as equals. This asks whether weighting them differently pays, and it
asks it honestly: the weights are fitted on one half of the studies and scored
on the other, both ways round, because a weight chosen on the same studies it
is scored on will always look like a gain.

Both inputs are columns of `test_series.csv`, so anything found here is usable
at inference with no DICOM tags, no new training and no shard rebuild.

    uv run python src/seq_weight_search.py artifacts/preds_plane_m38seed1.npz
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def weighted_macro(P, T, U, W, studies=None):
    """Study prediction = sum(w * p) / sum(w) over that study's series."""
    keep = np.isin(U, list(studies)) if studies is not None else np.ones(len(U), bool)
    p, t, u, w = P[keep], T[keep], U[keep], W[keep]
    df = pd.DataFrame(p * w[:, None], columns=LABELS).assign(uid=u, _w=w)
    num = df.groupby("uid")[LABELS].sum()
    den = df.groupby("uid")["_w"].sum()
    dp = num.div(den, axis=0)
    dt = pd.DataFrame(t, columns=LABELS).assign(uid=u).groupby("uid").first()
    out = {}
    for c in LABELS:
        y = (dt[c].to_numpy() > 0.5).astype(int)
        if len(np.unique(y)) > 1:
            out[c] = float(roc_auc_score(y, dp[c].to_numpy()))
    return (float(np.mean(list(out.values()))) if out else float("nan")), out


def halves(U):
    a, b = set(), set()
    for s in set(U):
        (a if int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 else b).add(s)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    a = ap.parse_args()

    z = np.load(a.preds, allow_pickle=True)
    P, T, U, PL, SUID = z["P"], z["T"], z["U"], z["PL"], z["SUID"]
    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet").set_index("SeriesInstanceUID")
    fs = np.array([ds.Fat_Suppression.get(s, 0) for s in SUID])

    A, B = halves(U)
    ones = np.ones(len(U))
    print(f"{len(P)} series, {len(set(U))} studies, halves {len(A)}/{len(B)}\n")

    uni_all, _ = weighted_macro(P, T, U, ones)
    print(f"uniform mean, all studies            {uni_all:.4f}   <- incumbent")
    fs_only, _ = weighted_macro(P, T, U, (fs == 1).astype(float) + 1e-9)
    print(f"FS series only                       {fs_only:.4f}")

    grid = np.round(np.arange(0.0, 2.01, 0.05), 2)

    def fit_global(train):
        best, bw = -1, 1.0
        for w in grid:
            W = np.where(fs == 1, 1.0, w)
            m, _ = weighted_macro(P, T, U, W, train)
            if m > best:
                best, bw = m, w
        return bw, best

    print("\n--- one global weight on non-FS, fitted on one half, scored on the other ---")
    gains = []
    for name, train, test in (("A->B", A, B), ("B->A", B, A)):
        w, fit = fit_global(train)
        W = np.where(fs == 1, 1.0, w)
        held, _ = weighted_macro(P, T, U, W, test)
        base, _ = weighted_macro(P, T, U, ones, test)
        gains.append(held - base)
        print(f"{name}  w(non-FS)={w:.2f}  fit {fit:.4f} | held-out {held:.4f} "
              f"vs uniform {base:.4f}   {held - base:+.4f}")
    print(f"mean held-out gain {np.mean(gains):+.4f}")

    print("\n--- per-finding weight on non-FS, same protocol ---")
    pf_gains = []
    for name, train, test in (("A->B", A, B), ("B->A", B, A)):
        ws = {}
        for c in LABELS:
            best, bw = -1, 1.0
            for w in grid:
                W = np.where(fs == 1, 1.0, w)
                _, per = weighted_macro(P, T, U, W, train)
                if per.get(c, -1) > best:
                    best, bw = per[c], w
            ws[c] = bw
        per_held = {}
        for c in LABELS:
            W = np.where(fs == 1, 1.0, ws[c])
            _, per = weighted_macro(P, T, U, W, test)
            per_held[c] = per.get(c, np.nan)
        _, base_per = weighted_macro(P, T, U, ones, test)
        held = float(np.nanmean(list(per_held.values())))
        base = float(np.nanmean([base_per.get(c, np.nan) for c in LABELS]))
        pf_gains.append(held - base)
        print(f"{name}  held-out {held:.4f} vs uniform {base:.4f}   {held - base:+.4f}")
        print("      weights " + " ".join(f"{c.split()[0][:4]}:{ws[c]:.2f}" for c in LABELS))
    print(f"mean held-out gain {np.mean(pf_gains):+.4f}")

    print("\n--- a weight per (plane x flag) bucket, coordinate ascent, same protocol ---")
    keys = [(p, f) for p in ("Sagittal", "Coronal", "Axial") for f in (1, 0)]
    bid = {k: i for i, k in enumerate(keys)}
    bucket = np.array([bid[(pl, int(f))] for pl, f in zip(PL, fs)])
    bg = []
    for name, train, test in (("A->B", A, B), ("B->A", B, A)):
        w = np.ones(len(keys))
        for _ in range(3):
            for i in range(len(keys)):
                best, bw = -1, w[i]
                for cand in grid:
                    w2 = w.copy(); w2[i] = cand
                    if w2.sum() == 0:
                        continue
                    m, _ = weighted_macro(P, T, U, w2[bucket], train)
                    if m > best:
                        best, bw = m, cand
                w[i] = bw
        held, _ = weighted_macro(P, T, U, w[bucket], test)
        base, _ = weighted_macro(P, T, U, ones, test)
        bg.append(held - base)
        print(f"{name}  held-out {held:.4f} vs uniform {base:.4f}   {held - base:+.4f}")
        print("      " + "  ".join(f"{k[0][:4]}/{'FS' if k[1] else 'non'}:{w[i]:.2f}"
                                   for i, k in enumerate(keys)))
    print(f"mean held-out gain {np.mean(bg):+.4f}")


if __name__ == "__main__":
    main()
