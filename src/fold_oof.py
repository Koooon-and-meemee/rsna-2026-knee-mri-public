"""Out-of-fold predictions from the fold set, and the instrument they buy.

Every number this project has ranked models on comes from 261 studies. pv-hold
is half of a 522-study validation split, and the standing worry since 24 August
is that it has stopped moving because it hit the teacher's ceiling rather than
the model's - both halves carry the teacher's labels, so the split catches
epoch-selection inflation but is blind to a model fitting the teacher's
systematic errors.

The fold set fixes the first half of that complaint and nothing can fix the
second. Each of m63f0..m63f4 held out a different fifth of the corpus, so
predicting each fold with the model that never saw it gives unbiased
predictions over **all 4,349 pseudo-labelled studies** - sixteen times pv-hold's
sample. The labels are still the teacher's, so this is a better-powered
measurement of the same quantity, not a different one. Read it as such.

Two things come out of one pass over the shards:

  **OOF, 4,349 studies.** The wide instrument. Comparable across fold sets, not
  comparable to pv-hold - different studies, and no selection inflation to
  subtract.

  **gold-58, all five members averaged.** The comparison that decides whether
  the fold ensemble goes to the board, because the 58 gold studies are held out
  of every fold and out of the incumbent alike. The incumbent's gold is 0.9200
  (probability mean, measured 4 September).

Cost is one decode of the shards, not five: each series is shown only to the
one member that held it out. The gold studies are the exception - every member
sees those, since the point there is the ensemble.

    uv run python src/fold_oof.py --prefix m63f --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))

from labels import LABELS  # noqa: E402
from train_image import load_shard, score  # noqa: E402
from ensemble import load_model, resize2d, NATIVE, VOLUMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="m63f")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dataset", default="image_dataset.parquet")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--out", default="artifacts/oof_folds.parquet")
    a = ap.parse_args()

    dev = a.device
    ds = pd.read_parquet(ROOT / "artifacts" / a.dataset)
    meta = ds.set_index("SeriesInstanceUID")
    STUDY = meta.StudyInstanceUID.to_dict()
    TARGET = {k: v.to_numpy(dtype=np.float32) for k, v in meta[LABELS].iterrows()}

    cols = [f"fold{f}" for f in range(a.folds)]
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise SystemExit(f"{a.dataset} has no {missing} - run src/make_folds.py first")

    # Which member is out-of-fold for each series. gold_val is held out by every
    # fold, so it gets the whole ensemble rather than one member.
    OOF, GOLD = {}, set()
    for sid, row in meta[cols].iterrows():
        v = row.to_numpy()
        if (v == "gold_val").all():
            GOLD.add(sid)
        else:
            w = np.flatnonzero(v == "val")
            if len(w) != 1:
                raise SystemExit(f"{sid} is held out by {len(w)} folds, expected 1")
            OOF[sid] = int(w[0])

    models = []
    for f in range(a.folds):
        tag = f"{a.prefix}{f}"
        m, info = load_model(tag, dev)
        cfg = info["cfg"]
        models.append((tag, m, cfg.get("size", 128), cfg.get("upsample", 0)))
        print(f"{tag:10s} ep{info['epoch']:<3d} pv-sel {info['pseudo_val']:.4f}  "
              f"gold {info['gold_auc']:.4f}")

    shards = sorted(VOLUMES.glob("vol_p*_s*.npz"))
    if a.max_shards:
        shards = shards[: a.max_shards]

    P_oof, T_oof, U_oof, S_oof = [], [], [], []
    P_gold, T_gold, U_gold = [], [], []

    @torch.no_grad()
    def run(model, size, up, x0):
        x = x0 if size == NATIVE else resize2d(x0, size)
        if up:
            x = resize2d(x, up)
        with torch.amp.autocast(dev, dtype=torch.bfloat16):
            return torch.sigmoid(model(x)).float().cpu().numpy()

    for n, sh in enumerate(shards, 1):
        vols, sids = load_shard(sh, NATIVE)
        by_fold, gold_idx = defaultdict(list), []
        for i, s in enumerate(sids):
            if s in GOLD:
                gold_idx.append(i)
            elif s in OOF:
                by_fold[OOF[s]].append(i)

        for f, idx in by_fold.items():
            tag, model, size, up = models[f]
            for b in range(0, len(idx), a.batch):
                sel = idx[b : b + a.batch]
                x0 = torch.from_numpy(np.ascontiguousarray(vols[sel]))
                x0 = x0.float().div_(255.0)[:, None].to(dev, non_blocking=True)
                P_oof.append(run(model, size, up, x0))
                T_oof.append(np.stack([TARGET[sids[i]] for i in sel]))
                U_oof.extend(STUDY[sids[i]] for i in sel)
                S_oof.extend(sids[i] for i in sel)

        for b in range(0, len(gold_idx), a.batch):
            sel = gold_idx[b : b + a.batch]
            x0 = torch.from_numpy(np.ascontiguousarray(vols[sel]))
            x0 = x0.float().div_(255.0)[:, None].to(dev, non_blocking=True)
            # Probability mean, because that is what the submission kernel does.
            P_gold.append(np.mean([run(m, s, u, x0) for _, m, s, u in models], axis=0))
            T_gold.append(np.stack([TARGET[sids[i]] for i in sel]))
            U_gold.extend(STUDY[sids[i]] for i in sel)
        print(f"  shard {n}/{len(shards)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    oof, per_oof = score(P_oof, T_oof, U_oof)
    n_studies = len(set(U_oof))
    print(f"\n--- out-of-fold, {n_studies} studies / {sum(len(x) for x in P_oof)} series ---")
    print(f"  macro {oof:.4f}")
    for k, v in sorted(per_oof.items(), key=lambda kv: kv[1]):
        print(f"    {k:18s} {v:.4f}")

    if P_gold:
        g, per_g = score(P_gold, T_gold, U_gold)
        print(f"\n--- gold-58, {a.folds}-member probability mean ---")
        print(f"  macro {g:.4f}     incumbent 0.9200, and gold abstains below 0.008")
        for k, v in sorted(per_g.items(), key=lambda kv: kv[1]):
            print(f"    {k:18s} {v:.4f}")

    out = pd.DataFrame(np.concatenate(P_oof), columns=LABELS)
    out.insert(0, "SeriesInstanceUID", S_oof)
    out.insert(1, "StudyInstanceUID", U_oof)
    out.insert(2, "fold", [OOF[s] for s in S_oof])
    path = ROOT / a.out
    out.to_parquet(path, index=False)
    print(f"\n{len(out)} out-of-fold predictions -> {path}")


if __name__ == "__main__":
    main()
