"""Train at the study level, where the labels actually live.

src/train_image.py trains one series at a time and attaches the study's twelve
findings to every series in it. A knee study holds a median of 5 series across
all three planes, and a meniscal tear is legible on sagittal images and not on
axial ones - so that scheme tells the axial series a tear is present when the
tear is not visible in it. The model has no way to tell which series the label
came from, and the cheapest way to satisfy the loss is to learn whatever looks
the same in all of them: coarse study-level appearance instead of the lesion.

The plane probe is the evidence. Sagittal, coronal and axial each score
0.857 / 0.857 / 0.841 alone on pseudo-val, and medial meniscus reads 0.896
sagittal against 0.847 axial - close to plane-invariant, which a model that had
actually located a few millimetres of fissure could not be. It also explains
why capacity (0.868 / 0.874 / 0.872 for nano / tiny / small), pos_weight,
augmentation and ensembling all land on the same 0.872-0.874: they were all
tuning a model whose target was mislabelled by construction.

Here the loss is applied per study and attention runs twice - over slices
within a series, then over series within a study. The model gets to decide
which series evidences which finding rather than being forced to agree with
all of them.

    uv run python src/train_study.py --backbone convnext_tiny.in12k_ft_in1k --aug

Two details that are not cosmetic:

**Series are packed flat, not padded.** Studies hold 1 to 14 series. Padding to
the maximum would push up to 13x wasted slices through the backbone, which is
the whole cost of the model.

**Training samples at most --max-series per study.** It bounds the memory spike
from a 14-series study and doubles as augmentation - the study is seen with a
different subset each epoch. Evaluation always uses every series.
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))
from net2d import StudyNet  # noqa: E402

from labels import LABELS  # noqa: E402
from train_image import build_cache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VOLUMES = ROOT / "artifacts" / "volumes"


def study_auc(P, T):
    """Macro AUC over findings; P and T are already one row per study."""
    aucs = {}
    for j, c in enumerate(LABELS):
        y = (T[:, j] > 0.5).astype(int)
        if len(np.unique(y)) > 1:
            aucs[c] = float(roc_auc_score(y, P[:, j]))
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--studies", type=int, default=2, help="studies per batch")
    ap.add_argument("--max-series", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr", type=float, default=None)
    ap.add_argument("--backbone", default="convnext_tiny.in12k_ft_in1k")
    ap.add_argument("--upsample", type=int, default=0)
    ap.add_argument("--cosine", action="store_true",
                    help="cosine decay with 3%% warmup; see train_image.py")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--tag", default="study")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed)
    print(f"device: {torch.cuda.get_device_name(0)}")

    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet")
    meta = ds.set_index("SeriesInstanceUID")
    SPLIT = meta.split.to_dict()
    STUDY = meta.StudyInstanceUID.to_dict()
    WEIGHT = meta.sample_weight.to_dict()
    TARGET = {k: v.to_numpy(dtype=np.float32) for k, v in meta[LABELS].iterrows()}

    arr, ids = build_cache(sorted(VOLUMES.glob("vol_p*_s*.npz")), len(ds), a.depth, a.size)

    # study -> its cache rows, kept per split
    studies: dict[str, list[str]] = {}
    for i, s in enumerate(ids):
        studies.setdefault(STUDY[s], []).append(i)
    by_split: dict[str, list[str]] = {}
    for uid, rows in studies.items():
        sp = SPLIT.get(ids[rows[0]])
        by_split.setdefault(sp, []).append(uid)
    for k in ("train", "val", "gold_val"):
        n = len(by_split.get(k, []))
        print(f"{k}: {n} studies, {sum(len(studies[u]) for u in by_split.get(k, []))} series")

    model = StudyNet(len(LABELS), backbone=a.backbone).to(dev)
    blr = a.backbone_lr if a.backbone_lr is not None else a.lr / 10
    back = [p for n, p in model.named_parameters() if ".backbone." in n]
    head = [p for n, p in model.named_parameters() if ".backbone." not in n]
    opt = torch.optim.AdamW([{"params": back, "lr": blr}, {"params": head, "lr": a.lr}],
                            lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(dev)

    # Flat learning rates are why every run peaks early and then decays; see
    # train_image.py. LambdaLR scales both groups equally, preserving the
    # backbone-at-a-tenth ratio.
    sched = None
    if a.cosine:
        import math

        steps = max(1, -(-len(by_split.get("train", [])) // a.studies)) * a.epochs
        warm = max(1, int(0.03 * steps))

        def factor(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, steps - warm)
            return 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        print(f"cosine schedule: {steps} steps, {warm} warmup")

    rng = np.random.default_rng(a.seed)
    print(f"{a.backbone}  backbone lr {blr:g} / head lr {a.lr:g}  "
          f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    def augment(x):
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[4])
        if torch.rand(1).item() < 0.5:
            s = 0.85 + 0.15 * torch.rand(1).item()
            h = int(x.shape[3] * s)
            i = int(torch.randint(0, x.shape[3] - h + 1, (1,)).item())
            j = int(torch.randint(0, x.shape[4] - h + 1, (1,)).item())
            x = F.interpolate(x[:, :, :, i : i + h, j : j + h].flatten(0, 1),
                              size=x.shape[3:], mode="bilinear",
                              align_corners=False).unsqueeze(1)
        if torch.rand(1).item() < 0.5:
            x = (x * (0.9 + 0.2 * torch.rand(1).item())
                 + (torch.rand(1).item() - 0.5) * 0.1).clamp_(0, 1)
        return x

    def batches(uids, train):
        order = list(uids)
        if train:
            rng.shuffle(order)
        for b in range(0, len(order), a.studies):
            batch = order[b : b + a.studies]
            rows, counts = [], []
            for u in batch:
                r = studies[u]
                if train and len(r) > a.max_series:
                    r = list(rng.choice(r, a.max_series, replace=False))
                rows.extend(sorted(r))
                counts.append(len(r))
            # NOT sorted globally: torch.split downstream assumes rows arrive
            # grouped by study, and two studies' cache rows can interleave.
            x = torch.from_numpy(np.ascontiguousarray(arr[rows])).float()
            x = x.div_(255.0)[:, None].to(dev, non_blocking=True)
            if train and a.aug:
                x = augment(x)
            if a.upsample:
                x = F.interpolate(x[:, 0], size=(a.upsample, a.upsample),
                                  mode="bilinear", align_corners=False).unsqueeze(1)
            y = torch.from_numpy(np.stack([TARGET[ids[studies[u][0]]] for u in batch])).to(dev)
            w = torch.tensor([WEIGHT[ids[studies[u][0]]] for u in batch],
                             dtype=torch.float32).to(dev)
            yield x, counts, y, w

    # Half of pseudo-val selects the checkpoint, half never influences anything.
    # Taking the best of N epochs on a metric with sd 0.0034 inflates it, and
    # keeping the best of many experiments inflates it again; the gap between
    # the halves measures that rather than leaving it to be argued about.
    def half(uid):
        return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        P, T = [], []
        uids = list(by_split.get(split, []))
        for x, counts, y, _ in batches(uids, False):
            with torch.amp.autocast(dev, dtype=torch.bfloat16):
                P.append(torch.sigmoid(model(x, counts)).float().cpu().numpy())
            T.append(y.cpu().numpy())
        model.train()
        if not P:
            return float("nan"), {}, float("nan")
        P, T = np.concatenate(P), np.concatenate(T)
        s, per = study_auc(P, T)
        if split != "val":
            return s, per, float("nan")
        m = np.array([half(u) == 0 for u in uids])
        return study_auc(P[m], T[m])[0], per, study_auc(P[~m], T[~m])[0]

    out = ROOT / "artifacts" / f"image_model_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        run, n = 0.0, 0
        for x, counts, y, w in batches(by_split.get("train", []), True):
            with torch.amp.autocast(dev, dtype=torch.bfloat16):
                per = F.binary_cross_entropy_with_logits(
                    model(x, counts).float(), y, reduction="none").mean(1)
                loss = (per * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            run += loss.item() * len(counts)
            n += len(counts)
        g, per_finding, _ = evaluate("gold_val")
        v, _, h = evaluate("val")
        mark = "  <- best" if v > best else ""
        print(f"ep {ep}/{a.epochs}  loss {run/max(n,1):.4f}  pv-sel {v:.3f}"
              f"  pv-hold {h:.3f}  GOLD {g:.3f}  {(time.time()-t0)/60:.1f}m{mark}", flush=True)
        # selection on pseudo-val: 522 studies at sd 0.0034 against gold's 58 at
        # sd 0.0081, so gold cannot resolve which epoch is best
        if v > best:
            best = v
            torch.save(model.state_dict(), out / "image_model.pt")
            json.dump({"gold_auc": g, "pseudo_val": v, "pseudo_val_holdout": h,
                       "per_finding": per_finding,
                       "epoch": ep,
                       "cfg": {"arch": "studynet", "backbone": a.backbone,
                               "size": a.size, "depth": a.depth, "lr": a.lr,
                               "studies": a.studies, "max_series": a.max_series,
                               "upsample": a.upsample, "aug": a.aug,
                               "cosine": a.cosine}},
                      open(out / "image_model.json", "w"), indent=1)

    print(f"\nbest pseudo-val {best:.4f}  ({out})")
    print("reference: series-level tiny pseudo-val 0.874 / gold 0.898 / LB 0.900")


if __name__ == "__main__":
    main()
