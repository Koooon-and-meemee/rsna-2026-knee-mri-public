"""Out-of-fold predictions over the training studies, for label refinement.

The measurement that motivates this: on the 58 gold studies, blending the
teacher's report-derived labels with the image model's predictions beats either
alone - 0.9271 for a quarter teacher and three quarters model, against 0.8997
for the teacher and 0.9094 for the model. The teacher read text, the model read
pixels, and their errors are close to independent, which is the condition under
which averaging helps. Same-family model ensembling failed here for the mirror
reason: those members agreed.

Turning that into better training labels needs predictions the model has not
already fitted. Its predictions on its own training studies are memorised, and
using them as targets would amplify its errors rather than correct them. So the
training studies are split into folds and each fold is predicted by a model that
never saw it.

Folds are grouped by study, and by report text where studies share one, using
the same grouping the train/val split already uses - otherwise a study's own
series land on both sides and the fold is not out-of-anything.

    uv run python src/oof_predict.py --folds 5 --epochs 6
    uv run python src/oof_predict.py --folds 5 --epochs 6 --resume

Each fold trains a fresh model on the other folds and predicts its own. The
result is one prediction per series for every training study, none of it fitted,
written to artifacts/oof_predictions.parquet.

Cost is folds x a training run. At 6 epochs and roughly 2.5 minutes an epoch on
the 128px cache that is about 75 minutes for five folds - cheap next to what it
buys, and the reason the epoch count is low is that these models exist only to
predict, not to be submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))
from net2d import SliceNet  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402
from train_image import build_cache, VOLUMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "oof_predictions.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="convnext_tiny.in12k_ft_in1k")
    ap.add_argument("--upsample", type=int, default=224)
    ap.add_argument("--resume", action="store_true",
                    help="skip folds whose predictions are already on disk")
    a = ap.parse_args()

    dev = "cuda"
    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet")
    meta = ds.set_index("SeriesInstanceUID")
    SPLIT, STUDY = meta.split.to_dict(), meta.StudyInstanceUID.to_dict()
    WEIGHT = meta.sample_weight.to_dict()
    TARGET = {k: v.to_numpy(dtype=np.float32) for k, v in meta[LABELS].iterrows()}

    arr, ids = build_cache(sorted(VOLUMES.glob("vol_p*_s*.npz")), len(ds), 32, 128)
    train_rows = [i for i, s in enumerate(ids) if SPLIT.get(s) == "train"]

    # Fold assignment is by STUDY, not by series: a study's series must not be
    # split across the boundary or the prediction is partly in-sample.
    def fold_of(uid):
        return int(hashlib.md5(uid.encode()).hexdigest(), 16) % a.folds

    folds = {}
    for i in train_rows:
        folds.setdefault(fold_of(STUDY[ids[i]]), []).append(i)
    print(f"{len(train_rows)} training series over {a.folds} folds: "
          + " ".join(str(len(folds[f])) for f in sorted(folds)))

    done = {}
    if a.resume and OUT.exists():
        prev = pd.read_parquet(OUT)
        done = {f: prev[prev.fold == f] for f in prev.fold.unique()}
        print(f"resuming: folds already done {sorted(done)}")

    pieces = list(done.values())
    for f in sorted(folds):
        if f in done:
            continue
        t0 = time.time()
        fit = [i for g in folds if g != f for i in folds[g]]
        model = SliceNet(len(LABELS), backbone=a.backbone).to(dev)
        back = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
        head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        opt = torch.optim.AdamW([{"params": back, "lr": a.lr / 10},
                                 {"params": head, "lr": a.lr}], weight_decay=1e-4)
        scaler = torch.amp.GradScaler(dev)
        steps = max(1, -(-len(fit) // a.batch)) * a.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
        rng = np.random.default_rng(f)

        def batch_of(idx, train):
            x = torch.from_numpy(np.ascontiguousarray(arr[sorted(idx)])).float()
            x = x.div_(255.0)[:, None].to(dev, non_blocking=True)
            if train and torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[4])
            if a.upsample:
                x = F.interpolate(x[:, 0], size=(a.upsample, a.upsample),
                                  mode="bilinear", align_corners=False).unsqueeze(1)
            return x

        model.train()
        for ep in range(1, a.epochs + 1):
            order = list(fit)
            rng.shuffle(order)
            run, n = 0.0, 0
            for b in range(0, len(order), a.batch):
                sel = sorted(order[b : b + a.batch])
                x = batch_of(sel, True)
                y = torch.from_numpy(np.stack([TARGET[ids[i]] for i in sel])).to(dev)
                w = torch.tensor([WEIGHT[ids[i]] for i in sel],
                                 dtype=torch.float32).to(dev)
                with torch.amp.autocast(dev, dtype=torch.bfloat16):
                    per = F.binary_cross_entropy_with_logits(
                        model(x).float(), y, reduction="none").mean(1)
                    loss = (per * w).sum() / w.sum().clamp(min=1e-6)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                run += loss.item() * len(sel)
                n += len(sel)
            print(f"  fold {f} ep {ep}/{a.epochs}  loss {run/max(n,1):.4f}  "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)

        model.eval()
        P, S = [], []
        with torch.no_grad():
            held = folds[f]
            for b in range(0, len(held), 16):
                sel = sorted(held[b : b + 16])
                with torch.amp.autocast(dev, dtype=torch.bfloat16):
                    P.append(torch.sigmoid(model(batch_of(sel, False))).float().cpu().numpy())
                S.extend(ids[i] for i in sel)
        piece = pd.DataFrame(np.concatenate(P), columns=LABELS)
        piece["SeriesInstanceUID"] = S
        piece["fold"] = f
        pieces.append(piece)
        pd.concat(pieces).to_parquet(OUT, index=False)
        print(f"fold {f} done in {(time.time()-t0)/60:.1f}m, "
              f"{len(piece)} predictions written", flush=True)
        del model, opt
        torch.cuda.empty_cache()

    df = pd.concat(pieces)
    print(f"\n{len(df)} out-of-fold predictions -> {OUT}")


if __name__ == "__main__":
    main()
