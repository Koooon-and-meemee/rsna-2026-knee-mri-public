"""Train the image model inside a Kaggle GPU kernel.

Runs here rather than locally because the local GPU belongs to another session,
and because the volumes already live on Kaggle as the preprocessing kernels'
output - training here means never moving 36GB anywhere.

The constraint that shapes everything is memory. The corpus is 24,371 volumes
at 32x256x256 uint8, which is 51GB; a Kaggle GPU kernel has on the order of
13-16GB of RAM. So volumes are never all resident: each epoch walks the shards,
holds one shard (~700 series, ~1.5GB) at a time, trains on it, and frees it.
Shuffling is over shard order and within a shard, which is weaker than a global
shuffle but is what fits.

Everything else mirrors src/train_image.py, and the reasoning for it is there:
the label belongs to the study so series predictions are averaged per study,
targets stay soft because the metric is rank-based, augmentation never flips
the depth axis because that would swap medial for lateral, and the 58 expert
studies are validation that is never trained on.
"""

import gc
import glob
import shutil
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

EPOCHS = int(os.environ.get("EPOCHS", 6))
BATCH = int(os.environ.get("BATCH", 16))
LR = float(os.environ.get("LR", 3e-4))
WIDTH = int(os.environ.get("WIDTH", 24))
SIZE = int(os.environ.get("SIZE", 128))   # downsampled from 256 on load
DEPTH = int(os.environ.get("DEPTH", 32))
MAX_SHARDS = int(os.environ.get("MAX_SHARDS", 0))
CACHE_DIR = os.environ.get("CACHE_DIR", "/kaggle/temp")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

# ---------------------------------------------------------------- inputs
hits = glob.glob("/kaggle/input/**/image_dataset.parquet", recursive=True)
if not hits:
    raise SystemExit("label dataset not attached: " + str(os.listdir("/kaggle/input")))
ds = pd.read_parquet(hits[0])
shards = sorted(glob.glob("/kaggle/input/**/vol_p*_s*.npz", recursive=True))
if not shards:
    raise SystemExit("no volume shards attached: " + str(os.listdir("/kaggle/input")))
# MAX_SHARDS exists so a smoke run can prove the kernel works end to end on a
# few shards before a full run spends GPU quota on it
if MAX_SHARDS:
    shards = shards[:MAX_SHARDS]
    print(f"MAX_SHARDS={MAX_SHARDS}: this is a smoke run, not a real score")
print(f"{len(ds)} series rows, {len(shards)} shards")

meta = ds.set_index("SeriesInstanceUID")
SPLIT = meta.split.to_dict()
WEIGHT = meta.sample_weight.to_dict()
STUDY = meta.StudyInstanceUID.to_dict()
TARGET = {k: v.to_numpy(dtype=np.float32) for k, v in meta[LABELS].iterrows()}


def load_shard(path, chunk=32):
    """Return (volumes uint8 (n,D,S,S), series ids) downsampled in plane.

    Resizing is chunked. Converting a whole shard to float32 first would
    allocate 700 x 32 x 256 x 256 x 4 = 5.9GB for the intermediate alone, which
    on a 13-16GB kernel is most of the machine for a buffer that is thrown away
    a moment later. In chunks of 32 volumes it is 270MB.
    """
    z = np.load(path)
    vols, ids = z["volumes"], [str(s) for s in z["series"]]
    if vols.shape[-1] == SIZE:
        return vols, ids
    out = np.empty((len(vols), vols.shape[1], SIZE, SIZE), dtype=np.uint8)
    for i in range(0, len(vols), chunk):
        t = torch.from_numpy(vols[i:i + chunk]).float()
        t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        out[i:i + chunk] = t.round().clamp_(0, 255).byte().numpy()
        del t
    del vols
    return out, ids


# ---------------------------------------------------------------- cache
def build_cache():
    """Decompress every shard once into a memmap, or return None if it won't fit.

    Without this, a run pays the shard cost 2x per epoch - once to train, once
    to evaluate - and each payment is a zlib decompress of ~950MB plus a resize
    from 256 to SIZE. Measured at ~27s per shard, which is where 3.2 of a 3.2
    hour run went; the GPU was mostly idle behind it.

    The cache also buys something the streaming path cannot give: a GLOBAL
    shuffle. Shard-resident training can only shuffle within a shard and across
    shard order, so every batch is drawn from one contiguous slice of the
    corpus. That correlates the gradient with whatever a shard happens to hold.

    Falls back to streaming rather than failing, because a kernel with too
    little scratch disk should still produce a model.
    """
    n_max = len(ds)
    nbytes = n_max * DEPTH * SIZE * SIZE
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        free = shutil.disk_usage(CACHE_DIR).free
    except Exception as e:
        print(f"cache: {CACHE_DIR} unusable ({e}); streaming instead", flush=True)
        return None, None
    if free < nbytes * 1.15:
        print(f"cache: needs {nbytes/1e9:.1f}GB, {free/1e9:.1f}GB free; streaming instead",
              flush=True)
        return None, None

    path = os.path.join(CACHE_DIR, f"vols_{DEPTH}x{SIZE}.npy")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                    shape=(n_max, DEPTH, SIZE, SIZE))
    ids, n, t0 = [], 0, time.time()
    for i, p in enumerate(shards, 1):
        vols, sids = load_shard(p)
        k = len(vols)
        arr[n:n + k] = vols
        ids.extend(sids)
        n += k
        del vols
        gc.collect()
        if i % 6 == 0 or i == len(shards):
            print(f"  cache {i}/{len(shards)} shards, {n} volumes, "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
    arr.flush()
    print(f"cached {n} volumes to {path} ({n*DEPTH*SIZE*SIZE/1e9:.1f}GB) "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)
    return arr[:n], ids


# ---------------------------------------------------------------- model
# Net comes from src/kernels/net.py, embedded above by the pusher, so the
# submission kernel rebuilds exactly what training saved.

model = Net(len(LABELS), WIDTH).to(dev)
print("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler(dev)

train_shards = [s for s in shards]
rng = np.random.default_rng(0)


def index_by_split(ids):
    """Row indices for each split, computed once against the cache order."""
    out = {}
    for i, s in enumerate(ids):
        out.setdefault(SPLIT.get(s), []).append(i)
    return out


def cached_train_batches(arr, ids, rows):
    """Globally shuffled batches straight out of the memmap."""
    order = rows[:]
    rng.shuffle(order)
    for b in range(0, len(order), BATCH):
        idx = sorted(order[b:b + BATCH])       # sorted: memmap reads sequentially
        x = torch.from_numpy(np.ascontiguousarray(arr[idx])).float().div_(255.0)[:, None]
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[4])        # in-plane only, never depth
        y = torch.from_numpy(np.stack([TARGET[ids[i]] for i in idx]))
        w = torch.tensor([WEIGHT[ids[i]] for i in idx], dtype=torch.float32)
        yield x.to(dev, non_blocking=True), y.to(dev), w.to(dev)


def train_batches(vols, ids):
    """Yield shuffled training batches from one resident shard (no cache)."""
    keep = [i for i, s in enumerate(ids) if SPLIT.get(s) == "train"]
    rng.shuffle(keep)
    for b in range(0, len(keep), BATCH):
        idx = keep[b:b + BATCH]
        x = torch.from_numpy(vols[idx]).float().div_(255.0)[:, None]
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[4])        # in-plane only, never depth
        y = torch.from_numpy(np.stack([TARGET[ids[i]] for i in idx]))
        w = torch.tensor([WEIGHT[ids[i]] for i in idx], dtype=torch.float32)
        yield x.to(dev, non_blocking=True), y.to(dev), w.to(dev)


def _score(P, T, U):
    if not P:
        return float("nan"), {}
    dp = pd.DataFrame(np.concatenate(P), columns=LABELS)
    dt = pd.DataFrame(np.concatenate(T), columns=LABELS)
    dp["uid"], dt["uid"] = U, U
    dp, dt = dp.groupby("uid").mean(), dt.groupby("uid").first()   # study level
    aucs = {}
    for c in LABELS:
        y = (dt[c].to_numpy() > 0.5).astype(int)
        if len(np.unique(y)) > 1:
            aucs[c] = float(roc_auc_score(y, dp[c].to_numpy()))
    return float(np.mean(list(aucs.values()))) if aucs else float("nan"), aucs


@torch.no_grad()
def _forward_rows(arr, ids, rows):
    P, T, U = [], [], []
    for b in range(0, len(rows), BATCH):
        idx = sorted(rows[b:b + BATCH])
        x = torch.from_numpy(np.ascontiguousarray(arr[idx])).float().div_(255.0)[:, None].to(dev)
        with torch.amp.autocast(dev, dtype=torch.float16):
            p = torch.sigmoid(model(x)).float().cpu().numpy()
        P.append(p)
        T.append(np.stack([TARGET[ids[i]] for i in idx]))
        U.extend(STUDY[ids[i]] for i in idx)
    return P, T, U


@torch.no_grad()
def evaluate(splits, arr=None, ids=None, rows_by_split=None):
    """Score several splits. From the cache when it exists, else one shard walk.

    The shard path collects every split in a single visit: loading a shard
    costs more than the forward pass, so scoring splits separately would spend
    an extra full pass over 34GB per epoch.
    """
    model.eval()
    try:
        if arr is not None:
            out = {}
            for sp in splits:
                out[sp] = _score(*_forward_rows(arr, ids, rows_by_split.get(sp, [])))
            return out
        acc = {sp: ([], [], []) for sp in splits}
        for path in shards:
            vols, sids = load_shard(path)
            for sp in splits:
                P, T, U = acc[sp]
                keep = [i for i, s in enumerate(sids) if SPLIT.get(s) == sp]
                for b in range(0, len(keep), BATCH):
                    idx = keep[b:b + BATCH]
                    x = torch.from_numpy(vols[idx]).float().div_(255.0)[:, None].to(dev)
                    with torch.amp.autocast(dev, dtype=torch.float16):
                        pr = torch.sigmoid(model(x)).float().cpu().numpy()
                    P.append(pr)
                    T.append(np.stack([TARGET[sids[i]] for i in idx]))
                    U.extend(STUDY[sids[i]] for i in idx)
            del vols
            gc.collect()
        return {sp: _score(*acc[sp]) for sp in splits}
    finally:
        model.train()


# ---------------------------------------------------------------- train
CACHE, CACHE_IDS = build_cache()
ROWS = index_by_split(CACHE_IDS) if CACHE is not None else None
if CACHE is not None:
    print(f"cache active: {len(ROWS.get('train', []))} train / "
          f"{len(ROWS.get('val', []))} val / {len(ROWS.get('gold_val', []))} gold rows",
          flush=True)

best, t0 = -1.0, time.time()
for ep in range(1, EPOCHS + 1):
    run, n = 0.0, 0
    if CACHE is not None:
        for x, y, w in cached_train_batches(CACHE, CACHE_IDS, ROWS.get("train", [])):
            with torch.amp.autocast(dev, dtype=torch.float16):
                logits = model(x)
                per = F.binary_cross_entropy_with_logits(
                    logits.float(), y, reduction="none").mean(1)
                loss = (per * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(x)
            n += len(x)
    else:
        order = list(range(len(train_shards)))
        rng.shuffle(order)
        for si in order:
            vols, sids = load_shard(train_shards[si])
            for x, y, w in train_batches(vols, sids):
                with torch.amp.autocast(dev, dtype=torch.float16):
                    logits = model(x)
                    per = F.binary_cross_entropy_with_logits(
                        logits.float(), y, reduction="none").mean(1)
                    loss = (per * w).sum() / w.sum().clamp(min=1e-6)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += loss.item() * len(x)
                n += len(x)
            del vols
            gc.collect()
    scores = evaluate(["gold_val", "val"], CACHE, CACHE_IDS, ROWS)
    g, per_finding = scores["gold_val"]
    v, _ = scores["val"]
    mark = "  <- best" if g > best else ""
    print(f"ep {ep}/{EPOCHS}  loss {run/max(n,1):.4f}  pseudo-val {v:.3f}"
          f"  GOLD {g:.3f}  {(time.time()-t0)/60:.1f}m{mark}", flush=True)
    if g > best:
        best = g
        torch.save(model.state_dict(), "/kaggle/working/image_model.pt")
        json.dump({"gold_auc": g, "per_finding": per_finding, "epoch": ep,
                   "cfg": {"width": WIDTH, "size": SIZE, "depth": DEPTH,
                           "lr": LR, "batch": BATCH}},
                  open("/kaggle/working/image_model.json", "w"), indent=1)

print(f"\nbest gold macro-AUC {best:.4f}")
print("reference: metadata 0.605 | student-on-reports 0.899 | leaderboard #1 0.932")
