"""Train the image model locally on the RTX 5090.

The Kaggle path (src/kernels/train_image_kernel.py) exists because the local
GPU was unavailable; this is the same training with the platform limits
removed. No 9-12 hour session ceiling, no 30-hour weekly quota, no two-session
cap, so epochs and parallel variants stop being the binding constraint.

Carried over from four Kaggle runs, with the reasoning:

**Volumes are cached to a memmap once.** Decompressing the 36GB of npz shards
per epoch cost more than the forward pass; caching cut per-epoch time 2.7x and,
more importantly, made a GLOBAL shuffle possible instead of shuffling within
one shard at a time.

**Resolution stays at 128.** v4 tried 192 and scored 0.812 against v2's 0.824
at the same 16 epochs and 2.6x the cost. Both gold and pseudo-val moved down
together, so that is a real negative, not noise. The volumes are stored at 256
if this is ever worth revisiting with a larger model.

**pos_weight is new here.** Positive rates run from 6.9% (Fracture) to 56.2%
(Effusion) - a 17x spread in class balance that the loss previously ignored,
and the per-finding AUCs correlate with positive rate at +0.53. Fracture at
0.660 has 313 positives in 4,407 studies. Whether reweighting helps a RANK
metric is not obvious, so --pos-weight defaults off and is a variable to test,
not an assumption to bake in.

    uv run python src/train_image.py --epochs 32
    uv run python src/train_image.py --epochs 32 --pos-weight --tag pw
"""

from __future__ import annotations

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
import hashlib
import glob
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))
from net import Net  # noqa: E402
from net2d import SliceNet  # noqa: E402

from labels import LABELS  # noqa: E402

# Column permutation applied to the target when --flip-swap mirrors a batch:
# Medial/Lateral Meniscus and Medial/Lateral OA trade places, everything else
# stays. Built from LABELS rather than hard-coded indices so a reordering of
# the label list cannot silently mislabel four columns.
SIDE_PAIRS = [("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")]
SIDE_SWAP = list(range(len(LABELS)))
for _a, _b in SIDE_PAIRS:
    _i, _j = LABELS.index(_a), LABELS.index(_b)
    SIDE_SWAP[_i], SIDE_SWAP[_j] = _j, _i

ROOT = Path(__file__).resolve().parent.parent
VOLUMES = ROOT / "artifacts" / "volumes"
CACHE = ROOT / "artifacts" / "cache"


SHARD_MM = 130.0     # the physical extent src/volume.py cropped the shards to


def load_shard(path: Path, size: int, chunk: int = 32, crop_px: int = 0):
    """Volumes from one npz shard, resized in chunks to bound peak memory.

    `crop_px` narrows the field of view before the resize: a centre crop to
    crop_px of the shard's own 256 px, which is 130 mm, and then the usual
    resize to `size`. It buys no detail - the pixels were already resampled
    once when the shards were written - so it measures the COST of a narrower
    window on its own, with the benefit (more real pixels on the same anatomy,
    which needs a rebuild from the DICOMs) held out. See the saliency probe:
    the model puts 0.797 of its attention inside the central 100 mm against a
    random-weight control's 0.678, so narrowing to 100 mm discards about a
    fifth of what it currently uses.
    """
    z = np.load(path)
    vols, ids = z["volumes"], [str(s) for s in z["series"]]
    if crop_px and crop_px < vols.shape[-1]:
        r0 = (vols.shape[-2] - crop_px) // 2
        c0 = (vols.shape[-1] - crop_px) // 2
        vols = vols[..., r0 : r0 + crop_px, c0 : c0 + crop_px]
    if vols.shape[-1] == size:
        return np.ascontiguousarray(vols), ids
    out = np.empty((len(vols), vols.shape[1], size, size), dtype=np.uint8)
    for i in range(0, len(vols), chunk):
        t = torch.from_numpy(vols[i : i + chunk]).float()
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        out[i : i + chunk] = t.round().clamp_(0, 255).byte().numpy()
        del t
    return out, ids


def build_cache(shards, n_max, depth, size, crop_px: int = 0):
    """Decompress every shard once into a memmap, reusing it across runs.

    Cache validity is checked against the source shards' mtimes, not just the
    (depth, size) key. Without this, replacing artifacts/volumes (e.g. after a
    preprocessing fix) leaves a stale cache silently in place - the key still
    matches, load succeeds, and training proceeds on the old data with no
    error at all. Caught 2026-08-13: a cache built 2026-08-12 21:51 (against
    the pre-crop/pre-laterality-fix shards) was reused by a run meant to
    re-baseline against the corrected shards swapped in at 2026-08-13 13:01 -
    the run would have silently measured nothing new.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    # The crop belongs in the key for the same reason the mtime check exists:
    # a cropped run that silently reused an uncropped cache would train on the
    # wrong pixels and report a perfectly plausible number.
    suffix = f"_c{crop_px}" if crop_px else ""
    path = CACHE / f"vols_{depth}x{size}{suffix}.npy"
    ids_path = CACHE / f"ids_{depth}x{size}{suffix}.json"
    if path.exists() and ids_path.exists():
        cache_mtime = path.stat().st_mtime
        stale = [p for p in shards if p.stat().st_mtime > cache_mtime]
        if stale:
            print(f"cache {path} is older than {len(stale)} source shard(s) - "
                  f"rebuilding rather than risk training on stale data")
        else:
            ids = json.loads(ids_path.read_text())
            arr = np.load(path, mmap_mode="r")
            print(f"reusing cache {path} ({len(ids)} volumes, {arr.nbytes / 1e9:.1f}GB)")
            return arr[: len(ids)], ids

    need = n_max * depth * size * size
    free = shutil.disk_usage(CACHE).free
    if free < need * 1.1:
        raise SystemExit(f"cache needs {need/1e9:.1f}GB, {free/1e9:.1f}GB free")

    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(n_max, depth, size, size)
    )
    ids, n, t0 = [], 0, time.time()
    for i, p in enumerate(shards, 1):
        vols, sids = load_shard(p, size, crop_px=crop_px)
        arr[n : n + len(vols)] = vols
        ids.extend(sids)
        n += len(vols)
        del vols
        gc.collect()
        if i % 6 == 0 or i == len(shards):
            print(f"  cache {i}/{len(shards)}, {n} volumes, {(time.time()-t0)/60:.1f}m",
                  flush=True)
    arr.flush()
    ids_path.write_text(json.dumps(ids))
    print(f"cached {n} volumes ({n*depth*size*size/1e9:.1f}GB) in {(time.time()-t0)/60:.1f}m")
    return arr[:n], ids


class ShardPool:
    """Volumes read from the npz shards each epoch instead of from a cache.

    The cache is faster and should stay the default, but it cannot go above
    192px here: a 192 cache is 28.7GB against 23GB free, and 256 - the size the
    volumes are actually stored at - would need 39GB, which deleting the 128
    cache still would not reach. RAM is 30GB total, so /dev/shm is out too.
    Streaming is the only path to the resolution the data already has.

    The cost is shuffling. Caching mattered mostly because it allowed a global
    shuffle; decompressing one shard at a time and shuffling inside it would
    put 677 consecutive series from the same sites in every batch. So shards
    are drawn into a POOL of several at a time, shuffled together, and the pool
    membership is re-randomised every epoch - each batch then mixes `pool`
    shards' worth of studies, and over epochs every pairing is reachable.
    """

    MAX_PER_SHARD = 700          # largest shard holds 677 series

    def __init__(self, shards, size, pool, rng, depth=32, crop_px: int = 0):
        self.shards, self.size, self.rng, self.depth = list(shards), size, rng, depth
        self.crop_px = crop_px
        per = self.MAX_PER_SHARD * depth * size * size
        # Buffers are filled in place, so pool size IS peak, not half of it -
        # an earlier version held the shards and their concatenated copy at
        # once, asked for 22.7GB of a 30GB machine at 256px, and was SIGKILLed
        # by the OOM killer with no traceback at 00:45 on 2026-08-09. Three
        # experiments went that way before anyone noticed the logs just ended.
        #
        # Read MemAvailable, not SC_AVPHYS_PAGES: the latter counts only wholly
        # free pages and ignores reclaimable page cache, which was 17GB of the
        # 19GB actually usable.
        avail = 0
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
        # Two caps, and the absolute one is the important one. This machine is
        # shared: root-owned jobs spiked to 17GB and 11GB at 03:17 and 03:24 on
        # 2026-08-09 and were themselves OOM-killed, and one of ours died at
        # 03:16 five epochs into a sixteen-epoch run. Sizing from MemAvailable
        # at startup is meaningless when another tenant can take 17GB an hour
        # later, so the buffers also get a fixed ceiling that leaves room for
        # someone else's spike.
        BUDGET = 6e9                      # bytes across BOTH prefetch buffers
        safe = max(1, int(min(avail / 4, BUDGET / 2) / per))
        self.pool = min(pool, safe)
        if self.pool < pool:
            print(f"  pool {pool} -> {self.pool} ({avail/1e9:.0f}GB available, "
                  f"{per/1e9:.1f}GB per shard at {size}px)")

    def _fill(self, buf, shard_idx, want):
        """Decompress a pool's worth of shards into buf; return (n, sids)."""
        n, sids = 0, []
        for j in shard_idx:
            v, s = load_shard(self.shards[j], self.size,
                               crop_px=self.crop_px)
            keep = [k for k, u in enumerate(s) if u in want]
            if keep:
                # copy straight into the buffer; never hold both the shard and
                # a concatenated copy of it
                buf[n : n + len(keep)] = v[keep]
                sids.extend(s[k] for k in keep)
                n += len(keep)
            del v
            gc.collect()
        return n, sids

    def epoch(self, want, shuffle):
        """Yield (volumes, series_ids, order) pools covering `want`.

        Two buffers, and the next pool is decompressed in a background thread
        while the current one trains. npz decompression is single-threaded zlib
        and was costing more than the forward pass: a 256px epoch ran 16.9 min
        against 6.2 for the same work from a cache, with the GPU idle for most
        of the difference. Overlapping recovers nearly all of it, and it is
        safe because the writer only ever touches the buffer the reader is not
        holding.
        """
        order = list(range(len(self.shards)))
        if shuffle:
            self.rng.shuffle(order)
        groups = [order[i : i + self.pool] for i in range(0, len(order), self.pool)]
        shape = (self.pool * self.MAX_PER_SHARD, self.depth, self.size, self.size)
        bufs = [np.empty(shape, dtype=np.uint8) for _ in range(2)]

        pending, cur = None, 0
        with ThreadPoolExecutor(max_workers=1) as ex:
            pending = ex.submit(self._fill, bufs[0], groups[0], want)
            for gi in range(len(groups)):
                n, sids = pending.result()
                nxt = (cur + 1) % 2
                pending = (ex.submit(self._fill, bufs[nxt], groups[gi + 1], want)
                           if gi + 1 < len(groups) else None)
                if sids:
                    idx = np.arange(n)
                    if shuffle:
                        self.rng.shuffle(idx)
                    yield bufs[cur][:n], sids, idx
                cur = nxt
        del bufs
        gc.collect()


def score(P, T, U):
    if not P:
        return float("nan"), {}
    dp = pd.DataFrame(np.concatenate(P), columns=LABELS)
    dt = pd.DataFrame(np.concatenate(T), columns=LABELS)
    dp["uid"], dt["uid"] = U, U
    dp, dt = dp.groupby("uid").mean(), dt.groupby("uid").first()  # study level
    aucs = {}
    for c in LABELS:
        y = (dt[c].to_numpy() > 0.5).astype(int)
        if len(np.unique(y)) > 1:
            aucs[c] = float(roc_auc_score(y, dp[c].to_numpy()))
    return float(np.mean(list(aucs.values()))) if aucs else float("nan"), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", choices=["cnn3d", "slicenet"], default="cnn3d")
    ap.add_argument("--backbone", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="slicenet only; defaults to lr/10 so pretrained weights are "
                         "fine-tuned rather than overwritten by an untrained head")
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth", type=int, default=32)
    # Which shard directory to train from. Default is artifacts/volumes; a bare
    # name is resolved under artifacts/. Added 2026-08-24 so depth-48 shards can
    # sit alongside the depth-32 ones instead of replacing them - every existing
    # member was trained on 32 and has to stay reproducible. The cache is keyed
    # on the source shards' mtimes, so pointing this elsewhere rebuilds rather
    # than silently reusing the wrong volumes.
    # Load a non-timm checkpoint into the backbone. Added 2026-08-25 for the
    # medically pretrained weights the public solutions use (RadImageNet,
    # MedicalNet) - a different thing from the closed backbone-family axis,
    # which was DINOv2 pretrained on natural images, twice.
    # 2.5D: put slices (i-1, i, i+1) in the backbone's three input channels
    # instead of the same slice three times. Free - identical shape and FLOPs -
    # and the first path this architecture has ever had to through-plane
    # information. Added 2026-08-25 after depth 48 bought nothing, which is what
    # you would expect if every slice is processed in isolation.
    # Self-attention over the slice sequence before pooling. AttentionPool
    # scores slices independently and sums them; the slices never see each
    # other. This lets a slice be read in the light of its neighbours, which is
    # how a stack is actually read. Costs ~9.5M parameters against a backbone
    # that runs 32 times per series, so the added compute is negligible.
    ap.add_argument("--slice-attn", type=int, default=0,
                    help="transformer encoder layers over slices (0 = off)")
    ap.add_argument("--slice-gru", type=int, default=0,
                    help="bidirectional GRU layers over the slice sequence before "
                         "pooling. Unlike --slice-attn this one can read slice ORDER")
    ap.add_argument("--neighbour-channels", action="store_true",
                    help="feed adjacent slices as RGB channels (2.5D)")
    ap.add_argument("--backbone-weights", default=None,
                    help="path to an external backbone state_dict")
    ap.add_argument("--volumes", default=None,
                    help="shard directory (default artifacts/volumes)")
    ap.add_argument("--finding-weights", default="",
                    help="comma-separated per-finding loss weights, in LABELS "
                         "order, or 'weak' for 2x on the four findings that are "
                         "weakest on every instrument measured. The metric is a "
                         "macro mean over 12 AUCs, so a point on Synovitis is "
                         "worth the same as a point on ACL and the loss spends "
                         "the same capacity on both regardless")
    ap.add_argument("--crop-mm", type=float, default=0.0,
                    help="narrow the field of view to this many mm before the "
                         "resize, centre-cropped out of the shards' 130 mm. "
                         "Buys no detail - the shards were already resampled - "
                         "so it measures what a narrower window COSTS, which is "
                         "the half of the crop trade that does not need a "
                         "rebuild from the DICOMs")
    ap.add_argument("--upsample", type=int, default=0,
                    help="resize slices to NxN on the GPU before the backbone. Adds no "
                         "detail - it separates 'the backbone wants its native 224' from "
                         "'the findings need more pixels', which a bigger cache would confound")
    ap.add_argument("--stream", action="store_true",
                    help="read volumes from the npz shards each epoch instead of a memmap "
                         "cache. Slower, but the only way past 192px: a 192 cache needs "
                         "28.7GB against 23GB free and 256 would need 39GB")
    ap.add_argument("--pool", type=int, default=8,
                    help="shards held in RAM together when streaming; larger mixes more "
                         "studies per batch at more memory")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine decay with 3%% warmup. Every run so far trained at a flat "
                         "rate and peaked around epoch 5; nothing was ever reducing the step "
                         "size near the minimum")
    ap.add_argument("--aug", action="store_true",
                    help="add zoom and intensity jitter to the in-plane flip")
    # 2026-09-29. The augmentation axis, second pass. 19 Sep measured that --aug
    # is worth 0.0106 gold (3 sd) and the flip none of it, so the value lives in
    # zoom and intensity - but nothing has ever been *added*. These flags split
    # the existing pair and add the MRI-competition staples one at a time. Every
    # default reproduces the current --aug exactly.
    ap.add_argument("--aug-only", choices=["zoom", "int"], default=None,
                    help="with --aug, keep only zoom or only intensity jitter")
    ap.add_argument("--zoom-min", type=float, default=0.85,
                    help="lower bound of the zoom crop scale (upper is 1.0)")
    ap.add_argument("--rot", type=float, default=0.0,
                    help="in-plane rotation, uniform +-deg, p=0.5, same for all slices")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="gamma jitter: x**exp(U(-g,g)), p=0.5")
    ap.add_argument("--depth-aug", action="store_true",
                    help="p=0.5 depth shift of up to 2 slices, and each slice "
                         "zeroed with p=0.1")
    # 2026-09-18. The shards are laterality-normalised: volume.py mirrors every
    # right knee onto a canonical left frame precisely so that "medial" is a
    # fixed direction in the array. The training flip below then mirrors that
    # frame again, at p=0.5, along a fixed array axis - and no label moves with
    # it. Four of the twelve findings name a side (Medial/Lateral Meniscus,
    # Medial/Lateral OA) and a fifth is one-sided by name (MCL), so half of
    # every epoch has shown the model a mirrored knee carrying unmirrored
    # side labels. Lateral OA and Lateral Meniscus are two of the weakest
    # columns on every instrument this project owns, which is the shape that
    # predicts. This flag is the ablation: same recipe, no mirror.
    ap.add_argument("--no-flip", action="store_true",
                    help="drop the random in-plane mirror. The shards are already "
                         "laterality-normalised and the side-specific labels do not "
                         "flip with the image, so the mirror teaches medial=lateral")
    # The other half of the same ablation. If the mirror is worth keeping as an
    # augmentation, it is worth keeping with the labels moved to match: a
    # mirrored knee's medial meniscus IS its lateral meniscus. Swaps the two
    # side-paired columns. MCL has no lateral counterpart in the label set, so
    # that column stays wrong under a mirror whatever we do here - which is
    # itself a reason to expect --no-flip to be the cleaner of the two.
    ap.add_argument("--flip-swap", action="store_true",
                    help="keep the mirror, but swap Medial/Lateral Meniscus and "
                         "Medial/Lateral OA in the target when it fires")
    ap.add_argument("--unfreeze-last", type=int, default=0,
                    help="train only the last N backbone blocks (0 = all of it). "
                         "The public 0.941 notebook runs its DINO arm at "
                         "UNFREEZE_LAST=6 with a 125:1 head/backbone lr ratio; "
                         "every DINO run here so far used 10:1 and all of it")
    ap.add_argument("--cond", choices=["none", "flag", "planeflag", "contrast"], default="none",
                    help="condition the head on what KIND of series this is: "
                         "'flag' the competition's Fluid_Sensitive/Fat_Suppression "
                         "bit (2 classes), 'planeflag' that crossed with the "
                         "imaging plane (6). Both columns ship in train_series.csv "
                         "AND test_series.csv, so this needs no DICOM tags at "
                         "inference. Zero-initialised: an untrained embedding "
                         "leaves the model bit-identical to --cond none")
    ap.add_argument("--pos-weight", action="store_true",
                    help="weight the BCE by (1-p)/p per finding; UNTESTED on a rank metric")
    ap.add_argument("--split-col", default="split",
                    help="which column defines train/val. `split_scanner` holds out whole "
                         "scanner groups: 98.9%% of the default val set was scanned on "
                         "equipment that also appears in training, so it cannot measure "
                         "whether the model generalises to an unseen machine")
    ap.add_argument("--dataset", default="image_dataset.parquet",
                    help="targets file under artifacts/. blend_targets.py writes "
                         "alternatives with the same schema, so a run against refined "
                         "labels differs from the incumbent in this flag alone")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    # augment() only runs under --aug; without it these would be silently ignored
    # and a run would be filed under an arm it never trained.
    if not a.aug and (a.aug_only or a.zoom_min != 0.85 or a.rot or a.gamma or a.depth_aug):
        raise SystemExit("--aug-only/--zoom-min/--rot/--gamma/--depth-aug need --aug")

    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(a.seed)

    ds = pd.read_parquet(ROOT / "artifacts" / a.dataset)
    volumes = Path(a.volumes) if a.volumes else VOLUMES
    if not volumes.is_absolute():
        volumes = ROOT / "artifacts" / volumes
    shards = sorted(volumes.glob("vol_p*_s*.npz"))
    if not shards:
        raise SystemExit(f"no volume shards in {volumes}")
    print(f"{len(ds)} series rows, {len(shards)} shards")

    meta = ds.set_index("SeriesInstanceUID")
    SPLIT = meta[a.split_col].to_dict()
    WEIGHT = meta.sample_weight.to_dict()
    STUDY = meta.StudyInstanceUID.to_dict()
    TARGET = {k: v.to_numpy(dtype=np.float32) for k, v in meta[LABELS].iterrows()}

    # Which bucket each series belongs to. Fat_Suppression and Fluid_Sensitive
    # are the same bit on 100% of the corpus, so only one of them is used.
    PLANES = ["Sagittal", "Coronal", "Axial"]
    if a.cond == "flag":
        n_cond = 2
        COND = {k: int(v) for k, v in meta.Fat_Suppression.items()}
    elif a.cond == "planeflag":
        n_cond = 6
        COND = {k: PLANES.index(p) * 2 + int(f) for k, (p, f)
                in meta[["Anatomical_Plane", "Fat_Suppression"]].iterrows()}
    elif a.cond == "contrast":
        # Forum thread 741310 names this as a data trap and the crosstab
        # confirms it on our corpus: keying a series on plane x fat-sat puts
        # 5,208 T1, 4,161 PD and 2,004 T2 series into one "non-FS" bucket, so
        # the head is told three different contrasts are the same kind of
        # picture. TR and TE separate them physically, so the key becomes
        # plane x contrast. Fat suppression drops out of the key: it is
        # largely redundant with contrast here (6,853 of 8,894 FS series are
        # PD) and keeping both would push the embedding to 30 classes.
        sm = pd.read_parquet(ROOT / "artifacts/scanner_meta.parquet")
        sm = sm.set_index("SeriesInstanceUID")
        tr = pd.to_numeric(sm.RepetitionTime, errors="coerce")
        te = pd.to_numeric(sm.EchoTime, errors="coerce")

        def _contrast(uid):
            a_, b_ = tr.get(uid), te.get(uid)
            if pd.isna(a_) or pd.isna(b_):
                return 4                      # unknown
            if a_ < 800:
                return 0                      # T1: short TR
            if b_ < 40:
                return 1                      # PD: long TR, short TE
            if b_ >= 60:
                return 2                      # T2: long TR, long TE
            return 3                          # intermediate TE

        n_cond = 15
        COND = {k: PLANES.index(p) * 5 + _contrast(k)
                for k, p in meta.Anatomical_Plane.items()}
    else:
        n_cond, COND = 0, {}
    if n_cond:
        print(f"conditioning on {a.cond}: {n_cond} classes, "
              f"{pd.Series(list(COND.values())).value_counts().sort_index().to_dict()}")

    crop_px = int(round(a.crop_mm / SHARD_MM * 256)) if a.crop_mm else 0
    if crop_px:
        print(f"crop {a.crop_mm:.0f}mm of the shards' {SHARD_MM:.0f}mm = "
              f"{crop_px}px of 256, then resize to {a.size}"
              f"{' and upsample to ' + str(a.upsample) if a.upsample else ''}")

    pool, arr, ids, rows, split_uids = None, None, None, {}, {}
    if a.stream:
        pool = ShardPool(shards, a.size, a.pool, np.random.default_rng(a.seed), a.depth,
                         crop_px=crop_px)
        for u, sp in SPLIT.items():
            split_uids.setdefault(sp, set()).add(u)
        rows = {k: sorted(v) for k, v in split_uids.items()}
        print(f"streaming {len(shards)} shards at {a.size}px, pool {a.pool}")
    else:
        arr, ids = build_cache(shards, len(ds), a.depth, a.size, crop_px=crop_px)
        for i, sid in enumerate(ids):
            rows.setdefault(SPLIT.get(sid), []).append(i)
    print(f"train {len(rows.get('train', []))} / val {len(rows.get('val', []))} "
          f"/ gold {len(rows.get('gold_val', []))}")

    WEAK = ("Synovitis", "Lateral OA", "Lateral Meniscus", "PF OA")
    fw = None
    if a.finding_weights:
        if a.finding_weights == "weak":
            v = [2.0 if c in WEAK else 1.0 for c in LABELS]
        else:
            v = [float(x) for x in a.finding_weights.split(",")]
            if len(v) != len(LABELS):
                raise SystemExit(f"--finding-weights needs {len(LABELS)} values "
                                 f"in LABELS order, got {len(v)}")
        fw = torch.tensor(v, dtype=torch.float32, device=dev)
        print("finding weights:", " ".join(f"{c}={x:g}" for c, x in zip(LABELS, v)))

    pw = None
    if a.pos_weight:
        tr = ds[ds[a.split_col] == "train"]
        rates = np.array([(tr[c] > 0.5).mean() for c in LABELS], dtype=np.float32)
        pw = torch.tensor((1 - rates) / np.clip(rates, 1e-6, None)).to(dev)
        print("pos_weight:", " ".join(f"{c}={w:.1f}" for c, w in zip(LABELS, pw.tolist())))

    if a.arch == "slicenet":
        model = SliceNet(len(LABELS), backbone=a.backbone,
                         backbone_weights=a.backbone_weights,
                         neighbour_channels=a.neighbour_channels,
                         slice_attn=a.slice_attn,
                         slice_gru=a.slice_gru,
                         cond_classes=n_cond,
                         unfreeze_last=a.unfreeze_last).to(dev)
        # The head starts random and the backbone starts informed; one shared
        # learning rate either wastes the pretraining or starves the head.
        blr = a.backbone_lr if a.backbone_lr is not None else a.lr / 10
        head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        back = [p for n, p in model.named_parameters()
                if n.startswith("backbone.") and p.requires_grad]
        groups = [{"params": back, "lr": blr}, {"params": head, "lr": a.lr}]
        print(f"backbone {a.backbone}  lr {blr:g} / head lr {a.lr:g}")
    else:
        model = Net(len(LABELS), a.width).to(dev)
        groups = [{"params": list(model.parameters()), "lr": a.lr}]
    print("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-4)

    # Every run so far has trained at a flat learning rate, which is why they
    # all peak around epoch 5 and then decay: nothing ever reduces the step
    # size, so the model keeps taking full-sized steps around a minimum it has
    # already reached. LambdaLR scales each group by the same factor, so the
    # backbone stays at a tenth of the head throughout.
    sched = None
    if a.cosine:
        import math

        steps = max(1, -(-len(rows.get("train", [])) // a.batch)) * a.epochs
        warm = max(1, int(0.03 * steps))

        def factor(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, steps - warm)
            return 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        print(f"cosine schedule: {steps} steps, {warm} warmup")
    scaler = torch.amp.GradScaler(dev)
    rng = np.random.default_rng(a.seed)

    def augment(x):
        """Zoom and intensity jitter, on top of the in-plane flip.

        Every pretrained run so far peaks at epoch 5 of 8 and then falls on both
        instruments, which is overfitting with only one augmentation (a flip) to
        push back. These two are chosen because they are true of the data rather
        than merely convenient: knees differ in size and in how the field of
        view was cropped, so a scale jitter is a real patient-to-patient
        variation; and MRI intensity has no absolute units, so a gain and offset
        change is a real scanner-to-scanner variation. Nothing here rotates or
        shears - joint geometry is what the findings are about, and a lateral
        meniscus does not appear at 15 degrees.
        """
        mirrored = False
        if not a.no_flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[4])            # in-plane only, never depth
            mirrored = True
        if a.aug_only != "int" and torch.rand(1).item() < 0.5:   # zoom, random crop
            s = a.zoom_min + (1.0 - a.zoom_min) * torch.rand(1).item()
            h = int(x.shape[3] * s)
            i = int(torch.randint(0, x.shape[3] - h + 1, (1,)).item())
            j = int(torch.randint(0, x.shape[4] - h + 1, (1,)).item())
            x = F.interpolate(x[:, :, :, i : i + h, j : j + h].flatten(0, 1),
                              size=x.shape[3:], mode="bilinear", align_corners=False)
            x = x.unsqueeze(1)
        if a.aug_only != "zoom" and torch.rand(1).item() < 0.5:  # gain and offset
            x = (x * (0.9 + 0.2 * torch.rand(1).item())
                 + (torch.rand(1).item() - 0.5) * 0.1).clamp_(0, 1)
        if a.rot > 0 and torch.rand(1).item() < 0.5:
            # One angle for the whole volume, so slices stay registered.
            t = math.radians(a.rot * (2 * torch.rand(1).item() - 1))
            B, C, D, H, W = x.shape
            theta = torch.tensor([[math.cos(t), -math.sin(t), 0.0],
                                  [math.sin(t), math.cos(t), 0.0]],
                                 dtype=x.dtype).expand(B * D, 2, 3)
            flat = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            grid = F.affine_grid(theta, flat.shape, align_corners=False)
            flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                                 align_corners=False)
            x = flat.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        if a.gamma > 0 and torch.rand(1).item() < 0.5:
            g = math.exp(a.gamma * (2 * torch.rand(1).item() - 1))
            x = x.clamp(min=0).pow(g)
        if a.depth_aug:
            if torch.rand(1).item() < 0.5:
                k = int(torch.randint(-2, 3, (1,)).item())
                if k:
                    x = torch.roll(x, shifts=k, dims=2)
                    if k > 0:
                        x[:, :, :k] = 0
                    else:
                        x[:, :, k:] = 0
            keep = (torch.rand(x.shape[2]) >= 0.1).to(x.dtype)
            x = x * keep.view(1, 1, -1, 1, 1)
        return x, mirrored

    def make_batch(source, idx, uids, shuffle):
        x = torch.from_numpy(np.ascontiguousarray(source[idx])).float().div_(255.0)[:, None]
        mirrored = False
        if shuffle:
            if a.aug:
                x, mirrored = augment(x)
            elif not a.no_flip and torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[4])
                mirrored = True
        y = torch.from_numpy(np.stack([TARGET[u] for u in uids]))
        if mirrored and a.flip_swap:
            y = y[:, SIDE_SWAP]
        w = torch.tensor([WEIGHT[u] for u in uids], dtype=torch.float32)
        c = (torch.tensor([COND[u] for u in uids], dtype=torch.long).to(dev)
             if n_cond else None)
        x = x.to(dev, non_blocking=True)
        if a.upsample:
            # on device, after the transfer - upsampling on the CPU would
            # move (up)^2/size^2 times more bytes over PCIe for no reason
            x = F.interpolate(x[:, 0], size=(a.upsample, a.upsample),
                              mode="bilinear", align_corners=False).unsqueeze(1)
        return x, y.to(dev), w.to(dev), list(uids), c

    def batches(split, shuffle):
        if pool is None:                                   # cached
            order = list(rows.get(split, []))
            if shuffle:
                rng.shuffle(order)
            for b in range(0, len(order), a.batch):
                idx = sorted(order[b : b + a.batch])
                yield make_batch(arr, idx, [ids[i] for i in idx], shuffle)
        else:                                              # streamed
            want = split_uids.get(split, set())
            for parr, psids, pidx in pool.epoch(want, shuffle):
                for b in range(0, len(pidx), a.batch):
                    idx = sorted(pidx[b : b + a.batch].tolist())
                    yield make_batch(parr, idx, [psids[i] for i in idx], shuffle)
                del parr
                gc.collect()

    # The 522 pseudo-val studies are split in half by a hash of the study UID:
    # one half selects the checkpoint, the other never influences any decision.
    # Selecting the best of ~12 epochs on a metric with sd 0.0034 inflates it,
    # and running many experiments and keeping the winner inflates it again.
    # The gap between the two halves measures that inflation instead of leaving
    # it as something to be reasoned about after the fact.
    def half(uid):
        return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        P, T, U = [], [], []
        for x, y, _, sids, c in batches(split, False):
            with torch.amp.autocast(dev, dtype=torch.bfloat16):
                P.append(torch.sigmoid(model(x, c)).float().cpu().numpy())
            T.append(y.cpu().numpy())
            U.extend(STUDY[s] for s in sids)
        model.train()
        if split != "val":
            return score(P, T, U)
        Pc, Tc, Uc = np.concatenate(P), np.concatenate(T), np.array(U)
        m = np.array([half(u) == 0 for u in Uc])
        sel = score([Pc[m]], [Tc[m]], list(Uc[m]))[0]
        hold = score([Pc[~m]], [Tc[~m]], list(Uc[~m]))[0]
        return sel, {"__holdout__": hold}

    out = ROOT / "artifacts" / f"image_model_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    best, t0, start_ep = -1.0, time.time(), 1

    # Resume state, written every epoch. This machine is shared and the OOM
    # killer has taken runs mid-flight twice: a 16-epoch run that dies at epoch
    # 5 otherwise throws away seventy minutes and restarts from nothing. With
    # this, the queue relaunching the same command picks up where it stopped.
    # The scheduler is restored by stepping it forward rather than saved, since
    # LambdaLR's state is just a step count.
    last, nsteps = out / "last.pt", 0
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck.get("cfg_key") == (a.arch, a.backbone, a.size, a.upsample, a.epochs, crop_px, a.cond):
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            best, start_ep, nsteps = ck["best"], ck["epoch"] + 1, ck["sched_steps"]
            if sched is not None:
                for _ in range(nsteps):
                    sched.step()
            print(f"resuming {a.tag} from epoch {ck['epoch']} (best pv-sel {best:.4f})")
        else:
            print(f"ignoring {last}: written by a different configuration")

    for ep in range(start_ep, a.epochs + 1):
        run, n = 0.0, 0
        for x, y, w, _, c in batches("train", True):
            with torch.amp.autocast(dev, dtype=torch.bfloat16):
                logits = model(x, c)
                per = F.binary_cross_entropy_with_logits(
                    logits.float(), y, pos_weight=pw, reduction="none"
                )
                per = (per * fw).sum(1) / fw.sum() if fw is not None else per.mean(1)
                loss = (per * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
                nsteps += 1
            run += loss.item() * len(x)
            n += len(x)
        g, per_finding = evaluate("gold_val")
        v, vextra = evaluate("val")
        h = vextra.get("__holdout__", float("nan"))
        mark = "  <- best" if v > best else ""
        print(f"ep {ep}/{a.epochs}  loss {run/max(n,1):.4f}  pv-sel {v:.3f}"
              f"  pv-hold {h:.3f}  GOLD {g:.3f}  {(time.time()-t0)/60:.1f}m{mark}", flush=True)
        # Selection is on pseudo-val, not gold. 522 studies at sd 0.0034 against
        # 58 at sd 0.0081: gold cannot tell one epoch from the next. Selecting on
        # it threw away a real epoch in m12up224 - gold preferred epoch 4 (0.912
        # over 0.907, well inside its own noise) while pseudo-val put epoch 5
        # ahead by 0.881 to 0.867, which is four standard deviations.
        if v > best:
            best = v
            torch.save(model.state_dict(), out / "image_model.pt")
            json.dump({"gold_auc": g, "pseudo_val": v, "pseudo_val_holdout": h,
                       "per_finding": per_finding,
                       "epoch": ep,
                       "cfg": {"arch": a.arch, "backbone": a.backbone,
                               "width": a.width, "size": a.size, "depth": a.depth,
                               "upsample": a.upsample,
                               "crop_mm": a.crop_mm, "crop_px": crop_px,
                               "finding_weights": a.finding_weights,
                               "lr": a.lr, "batch": a.batch, "aug": a.aug, "no_flip": a.no_flip, "stream": a.stream,
                               "aug_only": a.aug_only, "zoom_min": a.zoom_min, "rot": a.rot,
                               "gamma": a.gamma, "depth_aug": a.depth_aug,
                               "split_col": a.split_col,
                               "cosine": a.cosine,
                               "pos_weight": a.pos_weight,
                               # Architecture flags MUST live here. ensemble.py
                               # and the submission kernel rebuild the model from
                               # this cfg, and anything missing is silently
                               # rebuilt wrong: m53nbr scored 0.8709 instead of
                               # 0.8896 on 25 Aug because neighbour_channels was
                               # absent, so the weights loaded cleanly into a
                               # model whose forward path fed the wrong channels.
                               # m54attn at least failed loudly, because
                               # slice_attn adds parameters.
                               "neighbour_channels": a.neighbour_channels,
                               "slice_attn": a.slice_attn,
                               "slice_gru": a.slice_gru,
                               # Same rule, and this one is worse than a wrong
                               # forward path: a conditioned model scored with
                               # no cond argument silently loses the embedding
                               # and returns a plausible number. n_cond is kept
                               # beside the name so a reader of the json does
                               # not have to know the mapping to rebuild it.
                               "cond": a.cond,
                               "cond_classes": n_cond,
                               # Freezing changes no shape, so unlike the flags
                               # above this one cannot fail at load time. It is
                               # recorded because it is the difference between
                               # two runs that are otherwise the same command.
                               "unfreeze_last": a.unfreeze_last}},
                      open(out / "image_model.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best,
                    "sched_steps": nsteps,
                    "cfg_key": (a.arch, a.backbone, a.size, a.upsample, a.epochs, crop_px, a.cond)},
                   out / "last.pt")

    print(f"\nbest pseudo-val {best:.4f}  ({out})")
    print("reference: metadata 0.605 | v3ep32 0.860 (LB 0.868) | leaderboard #1 0.934")


if __name__ == "__main__":
    main()
