"""Run the public CoAtNet Global96 top-3 package unmodified, as a submission.

Why this exists. Four independent routes told this project the same thing: its
members all fail the same way, so a twelfth convnext buys nothing. On 23 Sep a
refreshed 51-member pool confirmed it once more - the best roster the search
could find gained +0.001. The deficit to the prize line is +0.005 and it is all
in ensemble diversity, not single-model quality (our best single is 0.943 on
the board, which matches what the 4th-place holder says a plain CNN reaches).

mattiaangeli's Global96 package is the first genuinely decorrelated partner
available: CoAtNet RMLP-2 at native 384 px over a 96-slice inferred stack with
finding-specific gated heads, against our convnext SliceNet at 128/224. On the
58 gold studies its rank correlation with our incumbent is 0.864, where our own
most dissimilar member (DINOv2) sits at 0.910.

What this run does NOT assume. Their published gold58 number (0.9305) is the
mean of the three checkpoints they selected out of 24 epochs *on those same 58
studies* - their training history has val_auc == gold_auc on every epoch. So
the +0.0153 a local blend appears to gain is contaminated by their selection
and cannot be believed. Their team's own board score is 0.946, below ours.

This kernel therefore asks the one question the local instruments cannot: what
does their package actually score on the board, run unmodified? That number is
the honest weight to give them in a blend, and it costs one slot to learn.
No blending here, nothing of ours in the mix - their submission.csv as written.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

t0 = time.time()
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def find_package():
    """The dataset directory holding the manifest, checkpoints and modules.

    Kaggle does not mount datasets flat under /kaggle/input any more - this
    kernel first looked for /kaggle/input/<slug>/ and found only
    ['competitions', 'datasets']. Recurse for the manifest instead of encoding
    a layout, which is also what makes this survive the next reshuffle. Their
    own find_artifact_root() makes the flat assumption, so --artifact-root is
    passed explicitly below rather than left to the default.
    """
    hits = sorted(INPUT.rglob("coatnet_global96_top3_manifest.json"))
    if not hits:
        seen = [str(q.relative_to(INPUT)) for q in INPUT.rglob("*") if q.is_dir()][:40]
        raise SystemExit(f"Global96 package not mounted; dirs under {INPUT}: {seen}")
    return hits[0].parent


pkg = find_package()
print(f"package: {pkg}", flush=True)
print(f"files: {sorted(f.name for f in pkg.glob('*'))}", flush=True)

# timm is pinned to 1.0.22 by their manifest and the wheel ships in the dataset
# because the kernel has no internet. Installing it before anything imports
# timm is the whole reason this runs as a subprocess below rather than inline.
wheels = sorted(pkg.glob("timm-*.whl"))
if wheels:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                        "--no-index", str(wheels[0])],
                       capture_output=True, text=True)
    print(f"timm wheel: {r.returncode} {r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]}",
          flush=True)

# Their modules import each other by bare name, so they need to sit on the
# path of the process that runs them. The dataset mount is read-only and the
# script writes beside its --output, so copy the code (not the 900MB of
# checkpoints) into the working directory and run it there.
run_dir = WORK / "coat"
run_dir.mkdir(exist_ok=True)
for f in pkg.glob("*.py"):
    shutil.copy(f, run_dir / f.name)
print(f"copied {len(list(run_dir.glob('*.py')))} modules", flush=True)

import torch  # noqa: E402  - after the wheel install
print(f"cuda devices: {torch.cuda.device_count()}", flush=True)
if torch.cuda.device_count() != 2:
    # Their run_submission raises on this; say so here with the actual count so
    # a machine_shape mistake is one line in the log rather than a traceback.
    print("WARNING: the package requires exactly 2 CUDA devices", flush=True)

cmd = [sys.executable, str(run_dir / "coatnet_global96_baseline_top3_inference.py"),
       "--artifact-root", str(pkg),
       "--output", str(WORK / "submission.csv")]
print("running:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd=run_dir)
print(f"exit {r.returncode} after {time.time()-t0:.0f}s", flush=True)
if r.returncode != 0:
    raise SystemExit(f"Global96 inference failed with {r.returncode}")

import pandas as pd  # noqa: E402
sub = pd.read_csv(WORK / "submission.csv")
print(f"submission.csv: {len(sub)} rows, {len(sub.columns)} columns", flush=True)
print(sub.head(3).to_string(), flush=True)
print(f"total kernel wall clock: {time.time()-t0:.0f}s", flush=True)
