"""Where in the frame does a trained model actually look?

The crop is the one axis that has moved the leaderboard twice (+0.012 for the
physical-scale crop and laterality fix, +0.029 for matching the backbone's
operating point), and the next move on it - joint-centred cropping - has two
separable premises that have never been measured:

  1. the anatomy the model uses is off-centre, so re-centring the window would
     stop clipping it, and
  2. that anatomy occupies much less than the 130 mm the window spends on it,
     so a tighter window would put more pixels on the same structures.

Neither is answerable from pv-hold: a crop change is a *representation* change,
and both sides of that comparison move together (see HANDOFF, "The rule's
limit"). Both are answerable from the model we already have, for the cost of
one backward pass per series and no submission slot.

Saliency is |d(sum of the 12 logits)/dx| taken with respect to the 256 px
volume as it sits in the shard, so the map comes back in the shard's own
coordinates however the member resizes internally. It is averaged over slices
and over series. Two numbers come out of it:

    centroid offset   how far the saliency mass sits from the frame centre, in
                      mm at the shard's 130 mm / 256 px scale. Premise 1.
    mass inside a box  the share of saliency inside a centred box of a given
                      physical width, against the share a uniform map would put
                      there (the box's area ratio). Premise 2 - a tighter crop
                      is safe exactly to the degree this exceeds the uniform
                      baseline at the width being considered.

Gradient saliency is noisy per series and this reports distributions rather
than single maps for that reason. It says where the signal the model uses
lives, not where the joint is; those coincide only if the model has learned
the anatomy, which is itself worth seeing.

**A centred saliency map is not by itself evidence that the model found the
joint.** A convolutional stack has more paths from a central pixel to its
output than from a corner one - padding and the receptive field alone bias
gradients toward the middle, before any anatomy is involved. `--control` runs
the identical measurement through the same architecture with random weights,
which carries that bias and no learning. Read every number below as the
difference between the trained row and its control row; the uniform baseline
bounds the geometry, the control bounds the architecture.

    uv run python src/saliency_probe.py m31newshards --max-shards 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble import load_model, resize2d, NATIVE  # noqa: E402
from train_image import load_shard  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VOLUMES = ROOT / "artifacts" / "volumes"
CROP_MM = 130.0                      # what src/volume.py cropped these to
MM_PER_PX = CROP_MM / NATIVE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-shards", type=int, default=3)
    ap.add_argument("--max-series", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boxes", nargs="*", type=float,
                    default=[60.0, 80.0, 100.0, 110.0])
    ap.add_argument("--control", action="store_true",
                    help="also measure the same architecture with random "
                         "weights, which has the centre bias and none of the "
                         "learning")
    a = ap.parse_args()

    ds = pd.read_parquet(ROOT / "artifacts" / "image_dataset.parquet")
    meta = ds.set_index("SeriesInstanceUID")
    SPLIT = meta.split.to_dict()
    PLANE = (meta.Anatomical_Plane.to_dict()
             if "Anatomical_Plane" in meta.columns else {})

    model, info = load_model(a.tag, a.device)
    cfg = info["cfg"]
    size, up = cfg.get("size", 128), cfg.get("upsample", 0)
    print(f"{a.tag}  {cfg.get('backbone','-')}  size {size}"
          f"{'  up' + str(up) if up else ''}  ep{info['epoch']}  "
          f"gold {info['gold_auc']:.4f}")
    runs = [("trained", model)]
    if a.control:
        from net2d import SliceNet  # noqa: E402
        from labels import LABELS  # noqa: E402
        if cfg.get("arch") != "slicenet":
            raise SystemExit("--control only knows how to rebuild slicenet")
        ctrl = SliceNet(len(LABELS), backbone=cfg["backbone"], pretrained=False,
                        neighbour_channels=cfg.get("neighbour_channels", False),
                        slice_attn=cfg.get("slice_attn", 0),
                        slice_gru=cfg.get("slice_gru", 0)).to(a.device).eval()
        runs.append(("control", ctrl))
    for _, m in runs:
        for p in m.parameters():
            p.requires_grad_(False)

    maps: dict[str, dict[str, list[np.ndarray]]] = {w: {} for w, _ in runs}
    n_done = 0
    for sh in sorted(VOLUMES.glob("vol_p*_s*.npz")):
        if n_done >= a.max_series:
            break
        vols, sids = load_shard(sh, NATIVE)
        idx = [i for i, s in enumerate(sids) if SPLIT.get(s) == a.split]
        idx = idx[: max(0, a.max_series - n_done)]
        for b in range(0, len(idx), a.batch):
            sel = idx[b : b + a.batch]
            x0 = torch.from_numpy(np.ascontiguousarray(vols[sel]))
            x0 = x0.float().div_(255.0)[:, None].to(a.device)
            x0.requires_grad_(True)
            x = x0 if size == NATIVE else resize2d(x0, size)
            if up:
                x = resize2d(x, up)
            for which, m in runs:
                m(x).sum().backward(retain_graph=which != runs[-1][0])
                # (B, 1, D, 256, 256) -> (B, 256, 256), mean |grad| over slices
                g = x0.grad.detach().abs()[:, 0].mean(1).float().cpu().numpy()
                for k, i in enumerate(sel):
                    maps[which].setdefault(
                        PLANE.get(sids[i], "unknown"), []).append(g[k])
                x0.grad = None
            n_done += len(sel)
        print(f"  {n_done}/{a.max_series} series", end="\r", flush=True)
    print(" " * 40, end="\r")

    yy, xx = np.mgrid[0:NATIVE, 0:NATIVE]
    centre = (NATIVE - 1) / 2.0
    groups = []
    for which, _ in runs:
        by_plane = maps[which]
        all_maps = [m for v in by_plane.values() for m in v]
        groups += [(f"{p} {which}", v) for p, v in sorted(by_plane.items())]
        groups.append((f"ALL {which}", all_maps))
    n_series = len(maps[runs[0][0]] and
                   [m for v in maps[runs[0][0]].values() for m in v])

    print(f"\nsaliency over {n_series} {a.split} series, "
          f"{CROP_MM:.0f} mm / {NATIVE} px = {MM_PER_PX:.3f} mm/px\n")
    print(f"{'group':18s} {'n':>5s}  {'dy mm':>16s} {'dx mm':>16s}  {'|d| med':>7s}")
    for name, ms in groups:
        if not ms:
            continue
        dy, dx = [], []
        for m in ms:
            t = m.sum()
            if t <= 0:
                continue
            dy.append(((m * yy).sum() / t - centre) * MM_PER_PX)
            dx.append(((m * xx).sum() / t - centre) * MM_PER_PX)
        dy, dx = np.array(dy), np.array(dx)
        d = np.hypot(dy, dx)
        print(f"{name:18s} {len(ms):5d}  {dy.mean():+7.2f} sd {dy.std():5.2f} "
              f"{dx.mean():+7.2f} sd {dx.std():5.2f}  {np.median(d):7.2f}")

    print(f"\nshare of saliency inside a centred box, against the uniform "
          f"baseline (its area share)\n")
    head = "  ".join(f"{w:>10.0f}mm" for w in a.boxes)
    print(f"{'group':18s} {head}")
    for name, ms in groups:
        if not ms:
            continue
        row = []
        for w in a.boxes:
            half = w / 2.0 / MM_PER_PX
            box = (np.abs(yy - centre) <= half) & (np.abs(xx - centre) <= half)
            share = np.mean([m[box].sum() / m.sum() for m in ms if m.sum() > 0])
            row.append(f"{share:.3f}/{box.mean():.3f}")
        print(f"{name:18s} " + "  ".join(f"{c:>12s}" for c in row))
    print("\nread as inside/uniform: equal means the model spreads its "
          "attention evenly, higher means the box holds more than its area's "
          "worth and a crop to that width discards less than it looks like")


if __name__ == "__main__":
    main()
