"""Upload one trained checkpoint to Kaggle as an ensemble-member dataset.

Step 1 of submitting has been a manual directory-assembly every time: copy
image_model.pt and image_model.json somewhere, hand-write a
dataset-metadata.json, run `kaggle datasets create`. It is four minutes of
fiddling on the critical path, and the critical path matters because the
submission kernel takes ~72 minutes to score after it is pushed - so every
minute spent before the push is a minute added to a wait that is already long
enough to lose an evening to.

It is also the step with the failure mode that costs the most. Kaggle finishes
indexing a dataset *after* `create` returns, and a kernel pushed against an
unindexed dataset mounts nothing, spends its run looking for weights that were
never attached, and dies. push_submit_model.py checks for this, but the check
can only tell you to wait - so this script waits, rather than handing back a
dataset that is not usable yet.

    uv run python src/push_member_dataset.py m37up288
    uv run python src/push_member_dataset.py m37up288 --wait 600

An existing dataset gets a new version rather than a create, because `create`
on a slug that exists fails and the obvious recovery - deleting it - would
break every submission kernel already attached to it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "qwrfsdr"


def kaggle(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", "run", "kaggle", *args],
                          capture_output=True, text=True, cwd=ROOT, check=check)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="model tag, i.e. artifacts/image_model_<tag>")
    ap.add_argument("--wait", type=int, default=900,
                    help="seconds to wait for Kaggle to index the upload (0 to skip)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = ROOT / "artifacts" / f"image_model_{a.tag}"
    pt, meta = src / "image_model.pt", src / "image_model.json"
    for f in (pt, meta):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    # The checkpoint is the pv-sel-best epoch, not the last one. Print which,
    # so a submission can never be traced back to a checkpoint nobody can name.
    j = json.loads(meta.read_text())
    print(f"{a.tag}: epoch {j['epoch']}  pv-sel {j['pseudo_val']:.4f}  "
          f"pv-hold {j['pseudo_val_holdout']:.4f}  gold {j['gold_auc']:.4f}")

    slug = f"rsna-knee-image-model-{a.tag}"
    ref = f"{USER}/{slug}"
    exists = "image_model.pt" in kaggle("datasets", "files", ref).stdout

    d = Path(tempfile.mkdtemp())
    shutil.copy2(pt, d / "image_model.pt")
    shutil.copy2(meta, d / "image_model.json")
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": slug, "id": ref,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=1))

    if a.dry_run:
        print(f"would {'version' if exists else 'create'} {ref} from {d}")
        return

    if exists:
        r = kaggle("datasets", "version", "-p", str(d), "-m", f"{a.tag} epoch {j['epoch']}")
    else:
        r = kaggle("datasets", "create", "-p", str(d))
    print(r.stdout.strip() or r.stderr.strip())

    if not a.wait:
        return
    # Poll rather than sleep a fixed guess: indexing has taken anywhere from
    # seconds to several minutes, and the cost of being wrong is a dead kernel
    # run, not a retry.
    deadline = time.time() + a.wait
    while time.time() < deadline:
        if "image_model.pt" in kaggle("datasets", "files", ref).stdout:
            print(f"{ref} is indexed and attachable")
            return
        time.sleep(15)
    raise SystemExit(f"{ref} still not listing image_model.pt after {a.wait}s - "
                     f"re-check before pushing a kernel against it")


if __name__ == "__main__":
    main()
