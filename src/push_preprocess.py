"""Build and push the Kaggle kernel that preprocesses the DICOMs.

Why the work happens on Kaggle. Downloading the images is not viable: the API
rejects folder-prefix downloads, so every one of the ~730,000 slices needs its
own request, and it rate-limits around 150 requests/minute - roughly 84 hours
for the corpus, before any processing. Inside a kernel the same data is a
mounted filesystem: listing 4,407 studies takes 0.0s and a series reads and
decodes in 0.32s, so the whole corpus is ~2.1 hours single-threaded.

The output is written as a few large shards rather than one file per series.
Retrieval goes through the same rate-limited API, so 24,371 output files would
recreate the problem we just escaped; a dozen shards is a dozen requests.

Kernel output caps at 20GB, so the run is split into parts that each stay well
under it. Parts are independent and run concurrently.

src/volume.py is embedded verbatim into the pushed script - one copy of the
slice-ordering and normalisation logic, running identically here and there.

    uv run python src/push_preprocess.py --part 0 --parts 3
    uv run python src/push_preprocess.py --part 0 --parts 3 --limit 40  # smoke test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "qwrfsdr"
COMP = "rsna-knee-abnormality-detection"

DRIVER = '''
import os, glob, time, sys, gc
import numpy as np
import pandas as pd
import pydicom

PART, PARTS, LIMIT = {part}, {parts}, {limit}
DEPTH, SIZE, SHARD = {depth}, {size}, {shard}
ROOT = "/kaggle/input/competitions/{comp}"
OUT = "/kaggle/working"

series_meta = pd.read_csv(f"{{ROOT}}/train_series.csv")
# deterministic order, so parts are reproducible and disjoint
series_meta = series_meta.sort_values(["StudyInstanceUID", "SeriesInstanceUID"]).reset_index(drop=True)
mine = series_meta.iloc[PART::PARTS].reset_index(drop=True)
if LIMIT:
    mine = mine.head(LIMIT)
print(f"part {{PART}}/{{PARTS}}: {{len(mine)}} series", flush=True)

vols, ids, n_fail, t0, shard_i = [], [], 0, time.time(), 0

def flush():
    global vols, ids, shard_i
    if not vols:
        return
    path = f"{{OUT}}/vol_p{{PART}}_s{{shard_i:03d}}.npz"
    np.savez_compressed(path, volumes=np.stack(vols), series=np.array(ids))
    mb = os.path.getsize(path) / 1e6
    print(f"  wrote {{path}} ({{len(vols)}} series, {{mb:.0f}} MB, {{mb/len(vols):.2f}} MB/series)", flush=True)
    vols, ids = [], []
    shard_i += 1
    gc.collect()

for i, r in enumerate(mine.itertuples(), 1):
    d = f"{{ROOT}}/train_series/{{r.StudyInstanceUID}}/{{r.SeriesInstanceUID}}"
    try:
        fs = glob.glob(f"{{d}}/*.dcm")
        if not fs:
            raise RuntimeError("no dcm files")
        ds = [pydicom.dcmread(f) for f in fs]
        vols.append(to_volume(ds, DEPTH, SIZE))
        ids.append(r.SeriesInstanceUID)
    except Exception as e:
        n_fail += 1
        print(f"  FAIL {{r.SeriesInstanceUID[:20]}}: {{type(e).__name__}}: {{str(e)[:60]}}", flush=True)
    if len(vols) >= SHARD:
        flush()
    if i % 250 == 0:
        el = time.time() - t0
        print(f"  {{i}}/{{len(mine)}}  fail {{n_fail}}  {{el/60:.1f}}m elapsed, ~{{el/i*(len(mine)-i)/60:.0f}}m left", flush=True)
flush()

total = sum(os.path.getsize(f"{{OUT}}/{{f}}") for f in os.listdir(OUT) if f.endswith(".npz"))
print(f"\\ndone: {{len(mine)-n_fail}} series, {{n_fail}} failed, {{total/1e9:.2f}} GB in {{(time.time()-t0)/60:.0f}}m")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--shard", type=int, default=700, help="series per output file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    slug = f"rsna-knee-prep-{a.part}" + ("-smoke" if a.limit else "")
    body = (ROOT / "src" / "volume.py").read_text() + DRIVER.format(
        part=a.part, parts=a.parts, limit=a.limit,
        depth=a.depth, size=a.size, shard=a.shard, comp=COMP,
    )

    d = Path(tempfile.mkdtemp())
    (d / "prep.py").write_text(body)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "prep.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "competition_sources": [COMP],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=1))

    print(f"kernel {USER}/{slug}: part {a.part}/{a.parts}, {a.depth}x{a.size}x{a.size}")
    if a.dry_run:
        print(d / "prep.py")
        return
    r = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", str(d)],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
