"""Score an ensemble of trained checkpoints on both validation instruments.

Averaging models only helps when they make different mistakes, and there is
already evidence here about where that difference does and does not come from.
Two convnext_nano runs differing only in seed landed at pseudo-val 0.868 and
0.869 - a gap of 0.001 against a measured noise floor of 0.0034, so seeds
produce nearly identical models and seed-ensembling has little to average over.
Architectures are the axis with actual spread, which is why this takes a list
of checkpoints rather than a seed count.

Each checkpoint is read back with the architecture recorded in its own
image_model.json, so nano and tiny can be mixed in one call without flags.

    uv run python src/ensemble.py m2 m2s1 m3tiny

Reports each member alone and then the combination, on the 522-study
pseudo-val (the ranking instrument, sd 0.0034) and the 58-study gold set (the
calibration instrument, sd 0.0081). An ensemble that gains less than about
0.005 on pseudo-val is not worth the inference time it costs: the submission
kernel runs at 4.7 seconds per study per model, and code competitions cap
total runtime.

**Two combinations are reported, by probability and by rank** (2026-08-19).
The metric is AUC, which reads only the order of the predictions, so a member
that is confident in the wrong units drags a probability mean toward itself
without being any better ordered. Converting each member to per-label
percentile ranks first removes the units and averages only what the metric
actually reads. Every public ensemble on this competition does it this way.
Note that a member's own score can differ slightly between the two columns:
score() means over the series of a study before taking the AUC, and the mean
of ranks is not the rank of the mean. The earlier "same-family ensembling is
worth only +0.0035" verdict was measured on probabilities alone and should be
re-read here before it is trusted again - though not with its original members,
which predate the crop fix and would be scored against shards they never saw.

**Volumes are streamed from the shards at their native 256px** and resized per
member, rather than read from the 128px cache. Members are fed 256 -> their
own --size -> their own --upsample, in that order, which is the same path the
submission kernel takes. The cache version of this could not score any
checkpoint trained above 128px at all, and silently mis-scored the ones with
--upsample: m21real256 read as 0.785 against the 0.900 it actually trained to,
which looks like "256 is terrible" rather than "the harness is wrong".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from collections import defaultdict

PLANES = ("Sagittal", "Coronal", "Axial")
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))
from net import Net  # noqa: E402
from net2d import SliceNet  # noqa: E402

from labels import LABELS  # noqa: E402
from train_image import load_shard, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VOLUMES = ROOT / "artifacts" / "volumes"
NATIVE = 256  # the size the shards are stored at


def load_model(tag: str, dev: str):
    d = ROOT / "artifacts" / f"image_model_{tag}"
    meta = json.load(open(d / "image_model.json"))
    cfg = meta["cfg"]
    if cfg.get("arch") == "slicenet":
        model = SliceNet(len(LABELS), backbone=cfg["backbone"], pretrained=False,
                         neighbour_channels=cfg.get("neighbour_channels", False),
                         slice_attn=cfg.get("slice_attn", 0),
                         slice_gru=cfg.get("slice_gru", 0),
                         cond_classes=cfg.get("cond_classes", 0))
    else:
        model = Net(len(LABELS), cfg["width"])
    model.load_state_dict(torch.load(d / "image_model.pt", map_location="cpu"), strict=True)
    return model.to(dev).eval(), meta


def resize2d(x, n):
    """Resize the in-plane dimensions of a (B, 1, D, H, W) batch to n x n."""
    return F.interpolate(x[:, 0], size=(n, n), mode="bilinear",
                         align_corners=False).unsqueeze(1)


def to_ranks(P_list):
    """Per-label percentile ranks over the whole evaluation set.

    Takes and returns the same list-of-batches shape score() expects, so a
    ranked member can be scored and averaged exactly like a probability one.
    Ranks are taken across every series in the split at once - ranking within
    a batch would make the transform depend on batch size.
    """
    x = np.concatenate(P_list)
    r = pd.DataFrame(x).rank(pct=True).to_numpy(dtype=np.float32)
    out, i = [], 0
    for p in P_list:
        out.append(r[i : i + len(p)])
        i += len(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="artifacts/image_model_<tag> directories")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda",
                    help="cpu is for smoke-testing this script while the card is busy")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="stop after N shards; a smoke test, not a measurement")
    # Efficiency-track lever. Backbone cost is linear in slices per series, and
    # SliceNet pools over however many arrive, so dropping every other slice
    # halves inference for free structurally - the question is only what it
    # costs in accuracy, which is what this flag exists to measure. Added
    # 2026-08-25, when the efficiency leaderboard turned out to rank us 158th
    # of 2175 while nine teams above us score LOWER.
    # The other efficiency lever, and the larger one. Scoring cost is
    # 53 minutes of fixed DICOM decode plus 31 minutes per ensemble member, and
    # the fixed part is decoding every series of every study - 5.6 on average,
    # up to 13. Reading two instead would cut roughly 34 minutes, worth 0.023
    # AUC at the prize's 0.04-0.05 per hour, against the 0.0016 that halving
    # slices costs. Nobody has measured what it costs in accuracy.
    #
    # Deterministic "first N encountered", which is shard order. Not necessarily
    # the best N - the plane probe found series are not interchangeable - so a
    # small loss here is an upper bound on what a smarter choice would cost.
    ap.add_argument("--by-plane", action="store_true",
                    help="with --max-series, cover all three planes first")
    ap.add_argument("--max-series", type=int, default=0,
                    help="use at most N series per study (0 = all)")
    ap.add_argument("--slice-stride", type=int, default=1,
                    help="keep every Nth slice at inference (1 = all 32)")
    ap.add_argument("--flip-tta", action="store_true",
                    help="also average the in-plane mirrored volume; doubles inference cost")
    # 2026-09-04: every combination question so far has paid for a full GPU pass
    # over the shards, so only a handful of the ~25 submission-valid members
    # were ever tried together. The decode is the expensive part and it is
    # identical for every combination, so run it once and keep the per-member
    # predictions. Search is then arithmetic on a 40MB file.
    ap.add_argument("--save-preds", default=None,
                    help="write per-member predictions to this .npz and exit")
    ap.add_argument("--halves", action="store_true",
                    help="split pseudo-val the way the trainers do and report each half. "
                         "The selecting half is inflated by whatever was chosen on it; the "
                         "held-out half is not, so the gap is the size of that inflation")
    a = ap.parse_args()

    dev = a.device
    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet")
    meta = ds.set_index("SeriesInstanceUID")
    SPLIT, STUDY = meta.split.to_dict(), meta.StudyInstanceUID.to_dict()
    PLANE = (meta.Anatomical_Plane.to_dict()
             if "Anatomical_Plane" in meta.columns else {})
    TARGET = {k: v.to_numpy(dtype=np.float32) for k, v in meta[LABELS].iterrows()}

    PLANES = ["Sagittal", "Coronal", "Axial"]
    FLAG = meta.Fat_Suppression.to_dict() if "Fat_Suppression" in meta.columns else {}

    def cond_ids(mode, uids, dev):
        """The series-kind id a --cond member was trained against, or None."""
        if mode == "flag":
            v = [int(FLAG[u]) for u in uids]
        elif mode == "planeflag":
            v = [PLANES.index(PLANE[u]) * 2 + int(FLAG[u]) for u in uids]
        else:
            return None
        return torch.tensor(v, dtype=torch.long, device=dev)

    models = []
    for t in a.tags:
        m, info = load_model(t, dev)
        cfg = info["cfg"]
        size, up = cfg.get("size", 128), cfg.get("upsample", 0)
        # crop_px is read from the member's own cfg for the same reason size and
        # upsample are: feeding a member the full 130 mm frame when it trained on
        # a 100 mm crop of it returns the mismatch, not the model. This is the
        # failure that made every pre-crop-fix checkpoint unscoreable against the
        # current shards, and it is silent - a plausible number comes back.
        crop = int(cfg.get("crop_px", 0) or 0)
        # A conditioned member must be fed the same series-kind id it trained
        # on. strict=True above already refuses to load its weights into an
        # unconditioned model, but nothing would stop this loop from calling a
        # correctly built one with cond=None, which drops the embedding and
        # returns a number that looks fine.
        cond_mode = cfg.get("cond", "none")
        print(f"{t:16s} {cfg.get('arch','cnn3d'):9s} "
              f"{cfg.get('backbone','-').split('.')[0]:14s} "
              f"ep{info['epoch']:<3d} gold {info['gold_auc']:.3f}  "
              f"size {size}{'  up' + str(up) if up else ''}"
              f"{'  crop' + str(crop) + 'px' if crop else ''}")
        models.append((t, m, size, up, crop, cond_mode))

    shards = sorted(VOLUMES.glob("vol_p*_s*.npz"))
    if a.max_shards:
        shards = shards[: a.max_shards]
    splits = ("val", "gold_val")

    # One decode of each shard feeds every member: the volumes are the
    # expensive part, not the forward passes.
    @torch.no_grad()
    def collect():
        P = {t: {s: [] for s in splits} for t, *_ in models}
        T = {s: [] for s in splits}
        U = {s: [] for s in splits}
        seen_series = defaultdict(int)   # study -> series already used
        seen_planes = defaultdict(list)  # study -> planes already used
        for n, sh in enumerate(shards, 1):
            vols, sids = load_shard(sh, NATIVE)
            by = {}
            for i, s in enumerate(sids):
                sp = SPLIT.get(s)
                if sp in splits:
                    by.setdefault(sp, []).append(i)
            if a.max_series:
                for sp in by:
                    keep = []
                    for i in by[sp]:
                        st = STUDY[sids[i]]
                        if seen_series[st] >= a.max_series:
                            continue
                        if a.by_plane:
                            # Take the least-represented plane first, so three
                            # series cover all three. Every val and gold study
                            # carries all three planes, and the plane probe
                            # established they are not interchangeable - an
                            # arbitrary first-N can take three sagittals and
                            # see no coronal at all.
                            pl = PLANE.get(sids[i], "?")
                            used = seen_planes[st]
                            if used.count(pl) > min(used.count(z) for z in PLANES):
                                continue
                            used.append(pl)
                        seen_series[st] += 1
                        keep.append(i)
                    by[sp] = keep
            for sp, idx in by.items():
                for b in range(0, len(idx), a.batch):
                    sel = idx[b : b + a.batch]
                    x0 = torch.from_numpy(np.ascontiguousarray(vols[sel]))
                    x0 = x0.float().div_(255.0)[:, None].to(dev, non_blocking=True)
                    if a.slice_stride > 1:
                        x0 = x0[:, :, :: a.slice_stride].contiguous()
                    for t, m, size, up, crop, cond_mode in models:
                        c = cond_ids(cond_mode, [sids[i] for i in sel], dev)
                        xc = x0
                        if crop and crop < NATIVE:
                            o = (NATIVE - crop) // 2
                            xc = x0[:, :, :, o : o + crop, o : o + crop]
                        x = xc if xc.shape[-1] == size else resize2d(xc, size)
                        if up:
                            x = resize2d(x, up)
                        with torch.amp.autocast(dev, dtype=torch.bfloat16):
                            # Net (cnn3d) takes no cond argument at all, and
                            # an unconditioned SliceNet ignores one, so keep the
                            # one-argument call for every member that is not
                            # conditioned rather than widening Net's signature.
                            fwd = (lambda v: m(v)) if c is None else (lambda v: m(v, c))
                            p = torch.sigmoid(fwd(x)).float()
                            if a.flip_tta:
                                p = (p + torch.sigmoid(
                                    fwd(torch.flip(x, dims=[4]))).float()) / 2
                        P[t][sp].append(p.cpu().numpy())
                    T[sp].append(np.stack([TARGET[sids[i]] for i in sel]))
                    U[sp].extend(STUDY[sids[i]] for i in sel)
            print(f"  shard {n}/{len(shards)}", end="\r", flush=True)
        print(" " * 30, end="\r")
        return P, T, U

    def halves(P_list, T_list, U_list):
        """(selecting half, held-out half) - same md5 split the trainers use."""
        Pc, Tc, Uc = np.concatenate(P_list), np.concatenate(T_list), np.array(U_list)
        m = np.array([int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 for u in Uc])
        return (score([Pc[m]], [Tc[m]], list(Uc[m]))[0],
                score([Pc[~m]], [Tc[~m]], list(Uc[~m]))[0])

    P, T, U = collect()

    if a.save_preds:
        out = {}
        for split in ("val", "gold_val"):
            out[f"T__{split}"] = np.concatenate(T[split])
            out[f"U__{split}"] = np.array(U[split], dtype=object)
            for t, *_ in models:
                out[f"P__{split}__{t}"] = np.concatenate(P[t][split])
        path = Path(a.save_preds)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, tags=np.array([t for t, *_ in models], dtype=object),
                            **out)
        n = sum(len(x) for x in T["val"])
        print(f"wrote {path}  {len(models)} members x {n} val series")
        return

    print()
    for split, label in (("val", "pseudo-val"), ("gold_val", "gold-58")):
        t_, u_ = T[split], U[split]
        n = sum(len(x) for x in t_)
        print(f"--- {label} ({n} series) ---")
        ranked = {t: to_ranks(P[t][split]) for t, *_ in models}

        for t, *_ in models:
            s, _ = score(P[t][split], t_, u_)
            sr, _ = score(ranked[t], t_, u_)
            line = f"  {t:24s} prob {s:.4f}   rank {sr:.4f}"
            if a.halves and split == "val":
                sel, hold = halves(P[t][split], t_, u_)
                line += f"   sel {sel:.4f}  hold {hold:.4f}  ({hold - sel:+.4f})"
            print(line)

        if len(models) > 1:
            best = max(score(P[t][split], t_, u_)[0] for t, *_ in models)
            nb = len(t_)
            combos = {
                "prob mean": [np.mean([P[t][split][i] for t, *_ in models], axis=0)
                              for i in range(nb)],
                "rank mean": [np.mean([ranked[t][i] for t, *_ in models], axis=0)
                              for i in range(nb)],
            }
            name = "+".join(a.tags)
            for how, mean in combos.items():
                s, per = score(mean, t_, u_)
                line = (f"  {'ENSEMBLE ' + how:24s} {s:.4f}  "
                        f"({s - best:+.4f} vs best member)")
                if a.halves and split == "val":
                    sel, hold = halves(mean, t_, u_)
                    line += f"   sel {sel:.4f}  hold {hold:.4f}"
                print(line)
                if label == "gold-58" and how == "rank mean":
                    for k, v in sorted(per.items(), key=lambda kv: kv[1]):
                        print(f"      {k:18s} {v:.3f}")
            print(f"  ({name})")


if __name__ == "__main__":
    main()
