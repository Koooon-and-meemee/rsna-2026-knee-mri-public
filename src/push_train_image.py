"""Push the image-model training kernel to Kaggle GPU.

The volumes never leave Kaggle. The preprocessing kernels' outputs are
attached as kernel_sources, so a training run reads 36GB that was produced
there and is already there - no download, no dataset round-trip.

Hyperparameters are passed as environment variables baked into the script
header rather than as a config file, so each variant is a self-contained
kernel version and the slug records which variant it was.

    uv run python src/push_train_image.py                       # default
    uv run python src/push_train_image.py --tag w32 --width 32  # a variant
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
PREP = [f"{USER}/rsna-knee-prep-{i}" for i in range(3)]
LABELS_DS = f"{USER}/rsna-knee-pseudo-labels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=0,
                    help="smoke run: use only the first N shards")
    ap.add_argument("--prep", type=int, nargs="*", default=[0, 1, 2],
                    help="which preprocessing kernels to attach")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    slug = f"rsna-knee-train-{a.tag}"
    body = (
        "import os\n"
        f"os.environ.setdefault('EPOCHS', '{a.epochs}')\n"
        f"os.environ.setdefault('BATCH', '{a.batch}')\n"
        f"os.environ.setdefault('LR', '{a.lr}')\n"
        f"os.environ.setdefault('WIDTH', '{a.width}')\n"
        f"os.environ.setdefault('SIZE', '{a.size}')\n"
        f"os.environ.setdefault('DEPTH', '{a.depth}')\n"
        f"os.environ.setdefault('MAX_SHARDS', '{a.max_shards}')\n\n"
        + (ROOT / "src" / "kernels" / "net.py").read_text()
        + "\n"
        + (ROOT / "src" / "kernels" / "train_image_kernel.py").read_text()
    )

    d = Path(tempfile.mkdtemp())
    (d / "train.py").write_text(body)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        # Kaggle's own docs: "NvidiaTeslaP100 is not usable with the default
        # Kaggle image". enable_gpu alone assigns exactly that P100, whose
        # sm_60 is absent from the preinstalled torch's arch list, so conv3d
        # dies with "no kernel image is available". The API accepts a wrong or
        # misspelled value silently and falls back to the same P100 - which is
        # why the probe kernel prints the device it actually got.
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": "false",
        "competition_sources": [COMP],
        "dataset_sources": [LABELS_DS],
        "kernel_sources": [f"{USER}/rsna-knee-prep-{i}" for i in a.prep],
        "model_sources": [],
    }, indent=1))

    print(f"{USER}/{slug}: width {a.width}, size {a.size}, lr {a.lr}, {a.epochs} epochs")
    if a.dry_run:
        print(d / "train.py")
        return
    r = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", str(d)],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
