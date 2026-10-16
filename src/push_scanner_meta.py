"""Push the DICOM-header extraction kernel.

CPU only. Reading one header per series with stop_before_pixels is minutes of
work, and the weekly GPU quota is spent, so this runs alongside local training
without competing for anything.

    uv run python src/push_scanner_meta.py
    uv run python src/push_scanner_meta.py --dry-run
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
SLUG = "rsna-knee-scanner-meta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    d = Path(tempfile.mkdtemp())
    (d / "meta.py").write_text((ROOT / "src" / "kernels" / "scanner_meta_kernel.py").read_text())
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{SLUG}",
        "title": SLUG,
        "code_file": "meta.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",       # header reads are CPU work; GPU quota is spent
        "enable_internet": "false",
        "competition_sources": [COMP],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=1))

    if a.dry_run:
        print(d / "meta.py")
        return
    r = subprocess.run(["uv", "run", "kaggle", "kernels", "push", "-p", str(d)],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip() or r.stderr.strip())
    print(f"\nwhen COMPLETE:\n  uv run kaggle kernels output {USER}/{SLUG} -p artifacts/")


if __name__ == "__main__":
    main()
