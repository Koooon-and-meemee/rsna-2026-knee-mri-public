"""Push a Kaggle smoke-test kernel that checks the new laterality/crop code
against real DICOM before spending a full re-preprocessing pass on it.

The strongest test available without visual inspection: every series in one
study is the same physical knee, so knee_side() computed independently per
series - different planes, different orientation vectors, different geometry
math paths - must all agree. A sign error or an axis mix-up in the geometry
would show up as exactly this kind of disagreement, and a formula that is
merely imprecise near the midline would not, because the check only counts
resolved (non-dead-zone) sides.

    uv run python src/verify_laterality.py --studies 40
    uv run kaggle kernels status qwrfsdr/rsna-knee-lat-verify
    uv run kaggle kernels output qwrfsdr/rsna-knee-lat-verify -p /tmp/rsna/latverify
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
SLUG = "rsna-knee-lat-verify"

DRIVER = '''
import glob, time, random
import pandas as pd
import pydicom
from collections import Counter

N_STUDIES = {n_studies}
random.seed(0)

ROOT_DIR = "/kaggle/input/competitions/{comp}"
ts = pd.read_csv(f"{{ROOT_DIR}}/train_series.csv")
studies = sorted(ts.StudyInstanceUID.unique())
sample = random.sample(studies, min(N_STUDIES, len(studies)))

agree, disagree, unresolved_studies, plane_counts = 0, 0, 0, Counter()
crop_applied, crop_skipped = 0, 0
mismatch_examples = []

for si, study in enumerate(sample, 1):
    rows = ts[ts.StudyInstanceUID == study]
    sides, planes = {{}}, {{}}
    for _, r in rows.iterrows():
        d = f"{{ROOT_DIR}}/train_series/{{study}}/{{r.SeriesInstanceUID}}"
        fs = glob.glob(f"{{d}}/*.dcm")
        if not fs:
            continue
        try:
            ds = pydicom.dcmread(fs[len(fs) // 2])  # a middle slice
        except Exception:
            continue
        plane = plane_of(ds)
        side = knee_side(ds)
        plane_counts[plane] += 1
        if side is not None:
            sides[r.SeriesInstanceUID] = side
            planes[r.SeriesInstanceUID] = plane
        # crop stats on this same slice
        try:
            sp = [float(x) for x in ds.PixelSpacing]
            n_rows = round(CROP_MM / sp[0]); n_cols = round(CROP_MM / sp[1])
            if n_rows >= ds.Rows or n_cols >= ds.Columns:
                crop_skipped += 1
            else:
                crop_applied += 1
        except Exception:
            pass

    uniq = set(sides.values())
    if len(uniq) == 0:
        unresolved_studies += 1
    elif len(uniq) == 1:
        agree += 1
    else:
        disagree += 1
        if len(mismatch_examples) < 5:
            mismatch_examples.append((study[-16:], dict(sides), dict(planes)))

    if si % 10 == 0:
        print(f"  {{si}}/{{len(sample)}}  agree={{agree}} disagree={{disagree}} unresolved={{unresolved_studies}}", flush=True)

print()
print(f"studies checked        : {{len(sample)}}")
print(f"all-series-agree       : {{agree}}")
print(f"DISAGREE (bug signal)  : {{disagree}}")
print(f"fully unresolved       : {{unresolved_studies}}")
print(f"plane distribution     : {{dict(plane_counts)}}")
print(f"crop applied / skipped : {{crop_applied}} / {{crop_skipped}}")
if mismatch_examples:
    print("\\nmismatch examples:")
    for study, sides, planes in mismatch_examples:
        print(f"  {{study}}: sides={{sides}} planes={{planes}}")

# also render one small before/after volume pair for visual sanity, saved as npy
sample_study = sample[0]
rows = ts[ts.StudyInstanceUID == sample_study]
r = rows.iloc[0]
d = f"{{ROOT_DIR}}/train_series/{{sample_study}}/{{r.SeriesInstanceUID}}"
fs = glob.glob(f"{{d}}/*.dcm")
if fs:
    import numpy as np
    ds_list = [pydicom.dcmread(f) for f in fs]
    vol_new = to_volume(ds_list, depth=16, size=128, crop_mm=CROP_MM, normalize_laterality=True)
    vol_raw = to_volume(ds_list, depth=16, size=128, crop_mm=None, normalize_laterality=False)
    np.savez("/kaggle/working/sample_volumes.npz", new=vol_new, raw=vol_raw,
             side=knee_side(ds_list[len(ds_list)//2]) or "unresolved",
             plane=plane_of(ds_list[0]))
    print("\\nsaved sample_volumes.npz for visual inspection")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    body = (ROOT / "src" / "volume.py").read_text() + DRIVER.format(
        n_studies=a.studies, comp=COMP,
    )

    d = Path(tempfile.mkdtemp())
    (d / "verify.py").write_text(body)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{SLUG}",
        "title": SLUG,
        "code_file": "verify.py",
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

    print(f"kernel {USER}/{SLUG}: {a.studies} studies")
    if a.dry_run:
        print(d / "verify.py")
        return
    r = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", str(d)],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
