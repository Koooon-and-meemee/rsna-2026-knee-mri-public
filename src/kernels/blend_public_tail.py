"""Blend our ensemble with the strongest public pipeline, by rank.

Appended after submit_kernel.py, the same way blend_coat_tail.py is. Our arm has
already written submission.csv; this moves it aside, runs the public notebook as
a second arm in its own process, and writes rank-mean blends at several weights.

The public arm is the v9 "4-way consensus" notebook (DINOv2 x20, A5 attention
pooling, RadImageNet heads, Raptor + four CoAtNet readers), which the board
scores 0.943 on its own and which runs in under 30 minutes on 2xT4. Our arm is
0.948-0.949. Global96 alone was 0.930 and still bought +0.001 at w=0.2-0.3, so a
partner that is both stronger and built from four unrelated families is the
cheapest way onto the line the public notebooks have set.

It runs as a subprocess, after our arm, for the reason the coat tail gives: it
pip-installs wheels globally, and by now our checkpoints are loaded and done. Its
code cells are concatenated verbatim into a script at push time (it guards its
one get_ipython() call), zlib+base64 embedded below, so nothing it needs is
fetched at run time.

If the public arm fails, our arm's CSV is restored and the log says so loudly.
"""

import base64 as _b64
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import zlib as _zlib

BLEND_WEIGHTS = (0.35, 0.50, 0.65)
PRIMARY_WEIGHT = 0.50
STRICT = True

DEADLINE = 8.5 * 3600
BLEND_RESERVE = 600

PUBLIC_SCRIPT_B64 = "__PUBLIC_SCRIPT_B64__"

_W = "/kaggle/working"
print(f"\n{'='*60}\npublic arm starting ({time.time()-T0:.0f}s elapsed)", flush=True)

_shutil.move(f"{_W}/submission.csv", f"{_W}/ours.csv")
print(f"our arm: {time.time()-T0:.0f}s total for {len(test)} studies", flush=True)

try:
    import gc as _gc
    MEMBERS.clear()
    _gc.collect()
    torch.cuda.empty_cache()
except Exception as _e:
    print(f"could not release our GPU memory: {type(_e).__name__}: {_e}", flush=True)


def _run_public():
    script = f"{_W}/_public_arm.py"
    with open(script, "wb") as f:
        f.write(_zlib.decompress(_b64.b64decode(PUBLIC_SCRIPT_B64)))
    budget = DEADLINE - (time.time() - T0) - BLEND_RESERVE
    if budget < 900:
        raise RuntimeError(f"only {budget:.0f}s left of the cap, not starting the arm")
    print(f"running the public arm with a {budget/60:.0f} min budget", flush=True)
    t = time.time()
    try:
        r = _subprocess.run([_sys.executable, "-u", script], cwd=_W, timeout=budget)
    except _subprocess.TimeoutExpired:
        raise RuntimeError(f"public arm did not finish in {budget/60:.0f} min")
    print(f"public arm exited {r.returncode} after {time.time()-t:.0f}s", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"public arm exited {r.returncode}")
    if not os.path.isfile(f"{_W}/submission.csv"):
        raise RuntimeError("public arm wrote no submission.csv")
    _shutil.move(f"{_W}/submission.csv", f"{_W}/public.csv")
    return pd.read_csv(f"{_W}/public.csv", dtype={"StudyInstanceUID": str})


def _pct_rank(col):
    return pd.Series(col).rank(method="average").to_numpy() / len(col)


try:
    pub = _run_public()
    ours = pd.read_csv(f"{_W}/ours.csv", dtype={"StudyInstanceUID": str})
    print(f"ours {ours.shape}  public {pub.shape}", flush=True)

    pub = pub.set_index("StudyInstanceUID").reindex(ours.StudyInstanceUID)
    for c in LABELS:
        if pub[c].isna().any():
            raise RuntimeError(f"public.csv is missing studies or has nulls in {c}")
    if len(ours) > 10 and float(pub[LABELS].to_numpy(float).std(axis=0).min()) < 1e-9:
        raise RuntimeError("public arm has a constant column - its fallback fired")

    ro = {c: _pct_rank(ours[c].to_numpy()) for c in LABELS}
    rp = {c: _pct_rank(pub[c].to_numpy()) for c in LABELS}
    print("\nper-label rank correlation ours vs public:", flush=True)
    for c in LABELS:
        print(f"  {c:<18} {np.corrcoef(ro[c], rp[c])[0,1]:.3f}", flush=True)

    for w in BLEND_WEIGHTS:
        out = pd.DataFrame({"StudyInstanceUID": ours.StudyInstanceUID.to_numpy()})
        for c in LABELS:
            out[c] = (1 - w) * ro[c] + w * rp[c]
        name = f"submission_w{int(round(w*100)):02d}.csv"
        out.to_csv(f"{_W}/{name}", index=False)
        if abs(w - PRIMARY_WEIGHT) < 1e-9:
            out.to_csv(f"{_W}/submission.csv", index=False)
        print(f"wrote {name}  {len(out)} rows", flush=True)

    print(f"\nBLEND OK - submission.csv is w={PRIMARY_WEIGHT}", flush=True)
except Exception as e:
    print(f"\nBLEND FAILED: {type(e).__name__}: {e}", flush=True)
    if STRICT:
        # pub13 at w=0.50 and w=0.35 both scored 0.948, exactly our arm alone,
        # and the board shows no log - so a silent fallback cost two slots
        # without saying whether the public arm ever ran. Strict makes a failed
        # blend an ERROR on the board instead: any score it posts is a blend.
        os.remove(f"{_W}/submission.csv") if os.path.exists(f"{_W}/submission.csv") else None
        raise SystemExit("BLEND FAILED under STRICT - no submission.csv written")
    _shutil.copy(f"{_W}/ours.csv", f"{_W}/submission.csv")
    print("BLEND FAILED - submission.csv is OUR ARM ALONE, do not submit as a blend",
          flush=True)

print(f"total kernel wall clock: {time.time()-T0:.0f}s", flush=True)
