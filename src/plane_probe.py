"""Does the imaging plane carry finding-specific signal we are throwing away?

Every study contains all three planes, and the model is blind to which one it
is looking at: each series is scored independently and the study prediction is
their unweighted mean. Clinically that is wrong on its face - a meniscal tear
is read on sagittal images, the collateral ligaments on coronal - so an equal
average dilutes the plane that actually saw the finding with two that did not.

Before building plane conditioning, measure whether the effect is real and how
big it is. This scores an existing checkpoint per finding using only sagittal
series, only coronal, only axial, and all three averaged. If the per-plane
AUCs are within noise of each other, the model has already learned to ignore
irrelevant planes and there is nothing to recover. If they separate, the gap
between the best single plane and the mean is the size of the prize.

    uv run python src/plane_probe.py m3tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402
from ensemble import load_model  # noqa: E402
from train_image import build_cache, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VOLUMES = ROOT / "artifacts" / "volumes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-preds", default=None,
                    help="write per-series predictions so the aggregation "
                         "controls can be computed without a second GPU pass")
    a = ap.parse_args()

    dev = "cuda"
    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet")
    meta = ds.set_index("SeriesInstanceUID")
    SPLIT, STUDY = meta.split.to_dict(), meta.StudyInstanceUID.to_dict()
    PLANE = meta.Anatomical_Plane.to_dict()
    TARGET = {k: v.to_numpy(dtype=np.float32) for k, v in meta[LABELS].iterrows()}

    arr, ids = build_cache(sorted(VOLUMES.glob("vol_p*_s*.npz")), len(ds), 32, 128)
    idx = [i for i, s in enumerate(ids) if SPLIT.get(s) == a.split]
    model, info = load_model(a.tag, dev)
    print(f"{a.tag}: gold {info['gold_auc']:.3f}, {len(idx)} {a.split} series\n")

    P, T, U, PL = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(idx), a.batch):
            sel = idx[b : b + a.batch]
            x = torch.from_numpy(np.ascontiguousarray(arr[sel])).float().div_(255.0)[:, None]
            with torch.amp.autocast(dev, dtype=torch.bfloat16):
                P.append(torch.sigmoid(model(x.to(dev))).float().cpu().numpy())
            T.append(np.stack([TARGET[ids[i]] for i in sel]))
            U.extend(STUDY[ids[i]] for i in sel)
            PL.extend(PLANE[ids[i]] for i in sel)

    P, T = np.concatenate(P), np.concatenate(T)
    U, PL = np.array(U), np.array(PL)
    if a.save_preds:
        # Scoring one plane also scores a third of the series, and a study
        # prediction is a mean over whatever series it has. The per-plane
        # numbers below therefore confound "this plane carries less" with
        # "this study was averaged over fewer views", and the probe cannot
        # separate them. Keep the predictions so the matched-count control
        # can be run without paying for the forward passes again.
        np.savez_compressed(a.save_preds, P=P, T=T, U=U, PL=PL,
                            SUID=np.array([ids[i] for i in idx], dtype=object))
        print(f"wrote {a.save_preds}  {len(P)} series\n")
    planes = ["Sagittal", "Coronal", "Axial"]

    cols = {}
    for p in planes:
        m = PL == p
        s, per = score([P[m]], [T[m]], list(U[m]))
        cols[p] = per
        print(f"{p:9s} macro {s:.4f}  ({m.sum()} series)")
    s_all, per_all = score([P], [T], list(U))
    cols["mean"] = per_all
    print(f"{'ALL(mean)':9s} macro {s_all:.4f}\n")

    print(f"{'finding':18s} " + " ".join(f"{p[:4]:>7s}" for p in planes)
          + f" {'mean':>7s} {'best-mean':>10s}")
    gains = []
    for c in LABELS:
        vals = [cols[p].get(c, float("nan")) for p in planes]
        m = cols["mean"].get(c, float("nan"))
        gain = np.nanmax(vals) - m
        gains.append(gain)
        print(f"{c:18s} " + " ".join(f"{v:7.3f}" for v in vals)
              + f" {m:7.3f} {gain:+10.3f}")
    print(f"\nmean gain if each finding used its best single plane: {np.nanmean(gains):+.4f}")
    print("pseudo-val noise floor is 0.0034 - a gain under ~0.005 is not worth building for")


if __name__ == "__main__":
    main()
