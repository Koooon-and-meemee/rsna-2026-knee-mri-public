"""Blend our ensemble with the public CoAtNet Global96 package, by rank.

This is appended after submit_kernel.py, which has already written its own
submission.csv. It moves that aside, runs the Global96 package as a second
arm, and writes rank-mean blends at several weights.

Why rank and not probability. The two arms are not on the same scale and one
of them is not on a probability scale at all: Global96's own submission.csv
holds per-column percentile ranks, not probabilities. AUC reads order only, so
converting both arms to per-column percentile ranks before averaging is the
only combination that means anything here - a probability mean would let our
arm's calibration decide the result.

Why our arm runs first. The package pins timm to 1.0.22 and ships the wheel,
and pip install is global to the kernel, not scoped to the subprocess that
needs it. Installing it before our checkpoints are loaded would risk changing
the timm our backbones are built from. By the time this file runs, our arm has
finished and written its CSV, so the upgrade cannot reach it.

Why several weights in one run. Each arm costs its own pass over 955 studies,
and the weight is the one free parameter nobody can price locally: the local
blend gain is contaminated because the package's three checkpoints were chosen
out of 24 epochs on the same 58 gold studies we would score it against. Kaggle
scores whichever output file the submit call names, so one run can offer the
board the whole weight curve instead of one guess.

If the second arm fails this restores our arm's CSV and says so loudly rather
than crashing: a completed run with one arm is worth more than no run, and the
dev log is read before anything is submitted.
"""

import shutil as _shutil
import subprocess as _subprocess
import sys as _sys

BLEND_WEIGHTS = (0.20, 0.30, 0.40, 0.50)
PRIMARY_WEIGHT = 0.30

# Kaggle kills a run that passes the cap and keeps nothing, so a coat arm that
# overruns would cost the whole thing - including our arm, which has already
# finished by the time this runs. Give it the budget that is actually left and
# treat exhausting it as the blend failing, which lands in the except below and
# still writes a valid submission.csv. Sized off the 9h cap with the last half
# hour held back for the blend, the CSV writes and Kaggle's own notebook render.
DEADLINE = 8.5 * 3600
BLEND_RESERVE = 600

_W = "/kaggle/working"
print(f"\n{'='*60}\ncoat arm starting ({time.time()-T0:.0f}s elapsed)", flush=True)

_shutil.move(f"{_W}/submission.csv", f"{_W}/ours.csv")
print(f"our arm: {time.time()-T0:.0f}s total for {len(test)} studies", flush=True)

# The coat arm is a separate process and needs its own GPU memory. Ours is
# still holding ten checkpoints and whatever the allocator cached during 955
# studies, and the package runs CoAtNet at native 384 across both T4s. Drop the
# members and return the cache before handing over - the parent keeps its CUDA
# context either way, but that is a few hundred MB rather than several GB.
# Defensively: this sits above the try below, so anything raising here would
# end the run with ours.csv moved aside and no submission.csv at all. Failing
# to free memory is not a reason to lose the run.
try:
    import gc as _gc
    _f = lambda: [torch.cuda.mem_get_info(i)[0] / 2**30
                  for i in range(torch.cuda.device_count())]
    _free0 = _f()
    MEMBERS.clear()
    _gc.collect()
    torch.cuda.empty_cache()
    print("GPU free GiB before -> after releasing our members: "
          + ", ".join(f"{a:.1f}->{b:.1f}" for a, b in zip(_free0, _f())), flush=True)
except Exception as _e:
    print(f"could not release our GPU memory: {type(_e).__name__}: {_e}", flush=True)


MANIFEST = "coatnet_global96_top3_manifest.json"


def _find_package():
    """Where the package is mounted, without walking the competition data.

    The recursive glob this used to do cost 750 seconds on the dev run, and dev
    mounts three studies. At rerun the same tree holds 955 studies of DICOMs,
    so the walk grows with the test set and buys nothing - Kaggle mounts a
    dataset at a known depth. Try the shapes it actually uses, cheapest first,
    and keep the recursive walk as a fallback for the next time the layout is
    reshuffled (it already moved once: this kernel's first version looked for
    /kaggle/input/<slug>/ and found only ['competitions', 'datasets']).
    """
    for pat in (f"/kaggle/input/datasets/*/*/{MANIFEST}",
                f"/kaggle/input/*/{MANIFEST}",
                f"/kaggle/input/*/*/{MANIFEST}"):
        hits = sorted(glob.glob(pat))
        if hits:
            return os.path.dirname(hits[0])
    print("package not at any known depth; walking /kaggle/input", flush=True)
    hits = sorted(glob.glob(f"/kaggle/input/**/{MANIFEST}", recursive=True))
    if not hits:
        raise SystemExit("Global96 package not mounted under /kaggle/input")
    return os.path.dirname(hits[0])


def _run_coat():
    pkg = _find_package()
    print(f"package: {pkg}", flush=True)
    wheels = sorted(glob.glob(f"{pkg}/timm-*.whl"))
    if wheels:
        r = _subprocess.run([_sys.executable, "-m", "pip", "install", "--no-deps",
                             "--no-index", wheels[0]], capture_output=True, text=True)
        print(f"timm wheel: rc={r.returncode} {r.stdout.strip()[-120:]}", flush=True)
    run_dir = f"{_W}/coat"
    os.makedirs(run_dir, exist_ok=True)
    for f in glob.glob(f"{pkg}/*.py"):
        _shutil.copy(f, run_dir)
    if torch.cuda.device_count() != 2:
        print(f"WARNING: package wants 2 CUDA devices, sees "
              f"{torch.cuda.device_count()}", flush=True)
    budget = DEADLINE - (time.time() - T0) - BLEND_RESERVE
    if budget < 600:
        raise RuntimeError(f"only {budget:.0f}s left of the cap, not starting the arm")
    cmd = [_sys.executable, f"{run_dir}/coatnet_global96_baseline_top3_inference.py",
           "--artifact-root", pkg, "--output", f"{_W}/coat.csv"]
    print(f"running with a {budget/60:.0f} min budget:", " ".join(cmd), flush=True)
    try:
        r = _subprocess.run(cmd, cwd=run_dir, timeout=budget)
    except _subprocess.TimeoutExpired:
        raise RuntimeError(f"Global96 inference did not finish in {budget/60:.0f} min")
    if r.returncode != 0:
        raise RuntimeError(f"Global96 inference exited {r.returncode}")
    return pd.read_csv(f"{_W}/coat.csv")


def _pct_rank(col):
    # average ranks so ties do not get an arbitrary order, scaled to (0,1]
    return pd.Series(col).rank(method="average").to_numpy() / len(col)


try:
    coat = _run_coat()
    ours = pd.read_csv(f"{_W}/ours.csv")
    print(f"ours {ours.shape}  coat {coat.shape}", flush=True)

    # Align on UID. Row order is not guaranteed to match: the two arms iterate
    # the test set independently, and a silent misalignment would look like a
    # blend that simply does not help.
    coat = coat.set_index("StudyInstanceUID").reindex(ours.StudyInstanceUID)
    missing = int(coat[LABELS[0]].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} studies in ours.csv are absent from coat.csv")

    ro = {c: _pct_rank(ours[c].to_numpy()) for c in LABELS}
    rc = {c: _pct_rank(coat[c].to_numpy()) for c in LABELS}
    print("\nper-label rank correlation ours vs coat:", flush=True)
    for c in LABELS:
        print(f"  {c:<18} {np.corrcoef(ro[c], rc[c])[0,1]:.3f}", flush=True)

    for w in BLEND_WEIGHTS:
        out = pd.DataFrame({"StudyInstanceUID": ours.StudyInstanceUID.to_numpy()})
        for c in LABELS:
            out[c] = (1 - w) * ro[c] + w * rc[c]
        name = f"submission_w{int(round(w*100)):02d}.csv"
        out.to_csv(f"{_W}/{name}", index=False)
        if abs(w - PRIMARY_WEIGHT) < 1e-9:
            out.to_csv(f"{_W}/submission.csv", index=False)
        print(f"wrote {name}  {len(out)} rows", flush=True)

    print(f"\nBLEND OK - submission.csv is w={PRIMARY_WEIGHT}", flush=True)
except Exception as e:
    # Loud, and the dev log is read before any submit call.
    print(f"\nBLEND FAILED: {type(e).__name__}: {e}", flush=True)
    _shutil.copy(f"{_W}/ours.csv", f"{_W}/submission.csv")
    print("BLEND FAILED - submission.csv is OUR ARM ALONE, do not submit as a blend",
          flush=True)

print(f"total kernel wall clock: {time.time()-T0:.0f}s", flush=True)
