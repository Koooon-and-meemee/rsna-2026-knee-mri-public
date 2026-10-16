"""Predict the test studies and write submission.csv.

This is the only code the competition actually runs against the hidden test
set, so it assumes nothing about what it will be given. test.csv is read at
run time and whatever studies it names are processed - the three visible ones
today, the full hidden set at rerun.

It also cannot take a shortcut the training path took. Training read volumes
that a preprocessing kernel had already built; here the DICOMs arrive raw, so
the same slice ordering and normalisation has to happen inline. That is why
volume.py is a dependency-free module with no I/O: it is embedded here and in
the preprocessing kernel, so a test volume is built exactly like a training
volume. If those two drifted, the model would be shown a distribution it never
saw and the failure would look like a bad model rather than a bad pipeline.

Net is embedded from net.py for the same reason: the checkpoint has to load
into the module it was saved from. SliceNet is embedded from net2d.py alongside
it, and which one gets built is read out of the checkpoint's own metadata
rather than passed in as a flag - a submission that silently builds the wrong
architecture would either crash on load or, worse, load partially.

SliceNet is constructed with pretrained=False. This kernel has no internet, but
it does not need any: timm's model registry is local, and the checkpoint we
load already contains the fine-tuned backbone.
"""

import glob
import json
import os
import time

T0 = time.time()
COMP = "rsna-knee-abnormality-detection"

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PRIORS = {"ACL": 0.41, "MCL": 0.16, "Medial Meniscus": 0.45,
          "Lateral Meniscus": 0.40, "Medial OA": 0.26, "Lateral OA": 0.19,
          "PF OA": 0.36, "Effusion": 0.60, "Synovitis": 0.47,
          "Baker's": 0.21, "Contusion": 0.33, "Fracture": 0.31}

SIZE = int(os.environ.get("SIZE", 128))
DEPTH = int(os.environ.get("DEPTH", 32))
# Keep every Nth slice at inference. The efficiency prize weights
# RuntimeSeconds/32400 against AUC/(Benchmark - maxAUC), which works out near
# 0.04-0.05 AUC per hour, and halving slices costs 0.0016 on the four-member
# ensemble - about two minutes' worth. Default 1 so the main-track kernel is
# unchanged unless this is set.
PLANES = ("sagittal", "coronal", "axial")
SLICE_STRIDE = int(os.environ.get("SLICE_STRIDE", 1))
# Use at most this many series per study (0 = all). See predict_study.
MAX_SERIES = int(os.environ.get("MAX_SERIES", 0))
# Spend that budget across imaging planes instead of on whatever sorts first.
BY_PLANE = int(os.environ.get("BY_PLANE", 0))
WIDTH = int(os.environ.get("WIDTH", 24))
PREP_SIZE = 256   # what the preprocessing kernel wrote; training downsampled

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# ------------------------------------------------------------------ inputs
# Look in the known mount points first and only fall back to a recursive walk.
# That fallback is not cheap: /kaggle/input holds the whole competition dataset,
# and in the first m2 run `glob(..., recursive=True)` spent 610 of the kernel's
# 648 seconds walking it - 94% of the runtime, against 18 seconds of actual
# inference. Correctness was never the issue; the efficiency prize track is
# scored on runtime, so a search that costs 30x the work it finds is the single
# most expensive line in the submission.
def find(name, hints):
    for h in hints:
        p = os.path.join(h, name)
        if os.path.exists(p):
            return [p]
    t0 = time.time()
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    print(f"  fell back to a recursive walk for {name}: {time.time()-t0:.0f}s", flush=True)
    return hits


def find_all(name, hints):
    """Every hint directory that holds `name`, not just the first.

    find() stops at the first hit because there is only ever one test.csv. An
    ensemble attaches one dataset per member, so the checkpoint search has to
    collect all of them - reading ckpts[0] and calling it the model is how a
    three-member submission silently scores as a one-member submission.
    """
    hits = [os.path.join(h, name) for h in hints
            if os.path.exists(os.path.join(h, name))]
    return sorted(hits) or find(name, hints)


COMP_HINTS = [f"/kaggle/input/competitions/{COMP}", f"/kaggle/input/{COMP}"]
cands = find("test.csv", COMP_HINTS)
if not cands:
    raise SystemExit("competition data not attached: " + str(os.listdir("/kaggle/input")))
ROOT = os.path.dirname(cands[0])
test = pd.read_csv(f"{ROOT}/test.csv")
print(f"ROOT = {ROOT}, {len(test)} test studies  ({time.time()-T0:.0f}s)", flush=True)

CKPT_HINTS = [d for d in glob.glob("/kaggle/input/datasets/*/*") + glob.glob("/kaggle/input/*")
              if os.path.isdir(d)]
# ------------------------------------------------------------- the members
# One checkpoint or several: every attached dataset holding an image_model.pt
# becomes a member, and each carries its own arch, size and upsample in its own
# image_model.json. Members may differ in input representation - m31 reads
# 128px stretched to 224, m33 reads 256 stretched to 336 - and that difference
# is the whole reason averaging them is worth anything, so nothing here may
# assume one global SIZE.
ckpts = find_all("image_model.pt", CKPT_HINTS)
if not ckpts:
    raise SystemExit("no checkpoint attached: " + str(os.listdir("/kaggle/input")))

MEMBERS = []
for cp in ckpts:
    meta_files = find("image_model.json", [os.path.dirname(cp)])
    arch, backbone, upsample = "cnn3d", "convnext_nano.in12k_ft_in1k", 0
    width, size, depth = WIDTH, SIZE, DEPTH
    nbr, sattn = False, 0
    if meta_files:
        cfg = json.load(open(meta_files[0]))
        print("checkpoint:", {k: cfg.get(k) for k in ("gold_auc", "epoch", "cfg")})
        c = cfg.get("cfg", {})
        width = c.get("width", width)
        size = c.get("size", size)
        depth = c.get("depth", depth)
        arch = c.get("arch", arch)
        backbone = c.get("backbone", backbone)
        upsample = c.get("upsample", 0)
        # Architecture flags, read from the checkpoint like every other one.
        # Omitting them does not fail: 2.5D adds no parameters, so the weights
        # load cleanly into a model that then feeds the backbone the wrong
        # channels and scores ~0.019 low without saying anything.
        nbr = c.get("neighbour_channels", False)
        sattn = c.get("slice_attn", 0)
        sgru = c.get("slice_gru", 0)

    if arch == "slicenet":
        model = SliceNet(len(LABELS), backbone=backbone, pretrained=False,
                         neighbour_channels=nbr, slice_attn=sattn,
                         slice_gru=sgru).to(dev)
    else:
        model = Net(len(LABELS), width).to(dev)
    # strict load: a partial match here means the architecture was reconstructed
    # wrong, and a half-loaded model still produces plausible-looking numbers
    model.load_state_dict(torch.load(cp, map_location=dev), strict=True)
    model.eval()
    # Depth is baked into the volume, which is built once and shared by every
    # member, so members that disagree about it cannot be averaged - fail here
    # rather than feed one of them 32 slices it never trained on.
    if MEMBERS and depth != DEPTH:
        raise SystemExit(f"{cp} wants depth {depth}, others use {DEPTH}")
    DEPTH = depth
    MEMBERS.append((cp, model, size, upsample))
    print(f"loaded {cp}  arch={arch} size={size} depth={depth} "
          f"upsample={upsample or 'off'}")

print(f"{len(MEMBERS)} member(s); predictions are averaged as probabilities")


def member_input(base, size, upsample):
    """Resize one shared volume to what a single member expects.

    Split out of predict_study so it can be exercised without DICOMs: the test
    set is only readable inside the competition kernel, and this is the part
    where members differ from each other.
    """
    x = base
    if size != PREP_SIZE:
        x = F.interpolate(x[0], size=(size, size), mode="bilinear",
                          align_corners=False)[None]
    if upsample:
        # Down to size first and only then up to upsample, even though going
        # straight from PREP_SIZE would be sharper. Training read volumes from
        # a 128px cache and stretched them to 224, so the model has only ever
        # seen 128px of detail spread over 224px of canvas. Feeding it
        # genuinely sharper images at test time is a distribution it was never
        # trained on, and the mismatch would look like a bad model rather than
        # a bad resize.
        x = F.interpolate(x[0], size=(upsample, upsample), mode="bilinear",
                          align_corners=False)[None]
    # div, not div_: `base` is shared with every member after this one, and a
    # member whose size and upsample are both no-ops gets `x is base` - scaling
    # in place would divide the volume by 255 once per such member.
    return x.div(255.0).to(dev)


def pick_series(d, names):
    """The MAX_SERIES series of a study to decode, spread across planes.

    Series within a study are not interchangeable - the plane probe measured
    that - so which ones the budget buys matters as much as how many. Taking
    the first N in UID order ignores that and lands wherever the sort falls:
    three arbitrary series score gold 0.9173, and the same count chosen one per
    plane scores 0.9200, which is the whole-study number to within a tenth of
    gold's own spread.

    The plane comes from ImageOrientationPatient via plane_of, so it costs one
    header read per series - pixels are never touched - and needs nothing that
    exists only in the training metadata.
    """
    if not MAX_SERIES:
        return names
    if not BY_PLANE or len(names) <= MAX_SERIES:
        return names[:MAX_SERIES]

    plane = {}
    for ser in names:
        try:
            f = sorted(glob.glob(f"{d}/{ser}/*.dcm"))[0]
            plane[ser] = plane_of(pydicom.dcmread(f, stop_before_pixels=True))
        except Exception:
            plane[ser] = "unknown"

    keep, used = [], []
    for ser in names:
        pl = plane[ser]
        # Skip anything from a plane already better represented than the
        # thinnest one; an unreadable plane never blocks, it just queues.
        if used.count(pl) > min(used.count(z) for z in PLANES):
            continue
        used.append(pl)
        keep.append(ser)
        if len(keep) == MAX_SERIES:
            return keep
    # A study that is all one plane still gets its full budget.
    return keep + [s for s in names if s not in keep][:MAX_SERIES - len(keep)]


@torch.no_grad()
def predict_study(uid):
    """Mean prediction over the study's series and members; None if unreadable.

    Averaged as probabilities, not as ranks. AUC reads only order, so ranking
    each member before averaging is the standard argument - but measured on
    2,885 pseudo-val series on 2026-08-20 the probability mean gained +0.0052
    over the best member and the rank mean only +0.0019. Ranks also cannot be
    formed here without holding every study's predictions to the end, so the
    measured-better combination is the cheaper one too.

    The DICOMs are read and the volume built once per series, then shown to
    every member at that member's own resolution. Decoding dominates this
    kernel - 4.7 s/study against a forward pass measured in milliseconds - so
    a second member costs far less than a second run.
    """
    d = f"{ROOT}/test_series/{uid}"
    if not os.path.isdir(d):
        return None
    preds = []
    # Efficiency lever, and the largest one: decoding is 53 of the 177 minutes
    # a four-member submission takes, and it decodes every series of every
    # study - 5.6 on average. Keeping four plane-balanced series costs gold
    # nothing measurable and saves about fifteen minutes; three saves
    # twenty-five for gold -0.0003.
    series = pick_series(d, sorted(os.listdir(d)))
    for ser in series:
        files = glob.glob(f"{d}/{ser}/*.dcm")
        if not files:
            continue
        try:
            ds = [pydicom.dcmread(f) for f in files]
            vol = to_volume(ds, DEPTH, PREP_SIZE)          # same as training prep
            base = torch.from_numpy(vol).float()[None, None]
            if SLICE_STRIDE > 1:
                # Efficiency lever. Backbone cost is linear in slices and
                # SliceNet pools over however many arrive, so this halves the
                # dominant term. Subsample the DEPTH-slice volume rather than
                # asking to_volume for fewer slices, because the accuracy cost
                # was measured that way locally (ensemble.py --slice-stride)
                # and a differently-resampled volume is a different experiment.
                base = base[:, :, ::SLICE_STRIDE].contiguous()
            for _, model, size, upsample in MEMBERS:
                x = member_input(base, size, upsample)
                preds.append(torch.sigmoid(model(x)).float().cpu().numpy()[0])
        except Exception as e:
            print(f"  series {ser[:16]} failed: {type(e).__name__}: {str(e)[:60]}", flush=True)
    return np.mean(preds, axis=0) if preds else None


rows, n_fallback = [], 0
for i, uid in enumerate(test.StudyInstanceUID, 1):
    p = predict_study(uid)
    if p is None:
        # a study we cannot read still needs a row; priors rank it neutrally
        # rather than dropping it and failing submission validation
        n_fallback += 1
        rows.append({"StudyInstanceUID": uid, **PRIORS})
    else:
        rows.append({"StudyInstanceUID": uid, **dict(zip(LABELS, p.tolist()))})
    if i % 100 == 0:
        print(f"  {i}/{len(test)}", flush=True)

sub = pd.DataFrame(rows)[["StudyInstanceUID"] + LABELS]
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"\nwrote submission.csv: {len(sub)} rows, {n_fallback} fell back to priors")
print(f"total kernel wall clock: {time.time() - T0:.0f}s "
      f"({(time.time() - T0) / max(len(sub), 1):.1f}s per study)")
if n_fallback:
    print("WARNING: fallbacks score at chance - check the test_series layout")
print(sub.head())
