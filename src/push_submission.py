"""Build and push the submission kernel for this code competition.

Worth being explicit about what the submission can and cannot see: test.csv
carries StudyInstanceUID and nothing else. There are no reports at test time.
The entire text pipeline - teacher labelling, distillation, 4,407 pseudo-labels
- exists to manufacture training targets. Prediction is image-only.

The notebook is rerun against a hidden test set that replaces test.csv, so
nothing here may assume the 3 visible studies. Everything is driven off
whatever test.csv contains at run time.

The dummy variant writes the prior for every study. It scores 0.5 by
construction - a constant column has no ranking information - and that is the
point: it exercises the submission path end to end while the image model does
not exist yet, so a format or runtime problem surfaces now rather than in
October.

    uv run python src/push_submission.py --mode dummy
    uv run python src/push_submission.py --mode dummy --submit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from labels import GOLD_PRIORS, LABELS

ROOT = Path(__file__).resolve().parent.parent
USER = "qwrfsdr"
COMP = "rsna-knee-abnormality-detection"

DUMMY = '''
import os, glob
import pandas as pd

LABELS = {labels}
PRIORS = {priors}

ROOT = "/kaggle/input/competitions/{comp}"
if not os.path.exists(f"{{ROOT}}/test.csv"):
    cands = glob.glob("/kaggle/input/**/test.csv", recursive=True)
    ROOT = os.path.dirname(cands[0])
print("ROOT =", ROOT)

test = pd.read_csv(f"{{ROOT}}/test.csv")
print(f"{{len(test)}} test studies")

sub = test[["StudyInstanceUID"]].copy()
for c in LABELS:
    sub[c] = PRIORS[c]

sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head())
print("wrote submission.csv", os.path.getsize("/kaggle/working/submission.csv"), "bytes")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dummy"], default="dummy")
    ap.add_argument("--submit", action="store_true", help="submit after it completes")
    a = ap.parse_args()

    slug = f"rsna-knee-sub-{a.mode}"
    body = DUMMY.format(labels=repr(LABELS), priors=repr(GOLD_PRIORS), comp=COMP)

    d = Path(tempfile.mkdtemp())
    (d / "sub.py").write_text(body)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "sub.py",
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

    r = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", str(d)],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout.strip() or r.stderr.strip())
    print(f"\nkernel: {USER}/{slug}")
    print("when it completes, submit with:")
    print(
        f"  uv run kaggle competitions submit {COMP} "
        f"-k {USER}/{slug} -f submission.csv -m 'prior baseline'"
    )


if __name__ == "__main__":
    main()
