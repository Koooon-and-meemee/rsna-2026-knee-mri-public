"""Batch the studies make_batches.py did not pick, without disturbing the ones it did.

make_batches.py samples a stratified subset and writes `artifacts/batches/`.
It is not incremental: it **deletes every batch_*.json and renumbers from zero**.
Re-running it with a larger --n therefore silently invalidates the labels
already produced for the current numbering - ids are positional into
`teacher_subset_index.csv`, so a renumbered index maps every existing
labels_NNN.csv onto the wrong studies. The 1,000 studies labelled between 5 and
7 August would be lost that way, and nothing would error.

This script extends instead. It leaves the existing batches and index rows
untouched, appends the remaining studies to `teacher_subset_index.csv`, and
writes new batch files continuing the numbering. collect_labels.py needs no
change: it globs labels_[0-9][0-9][0-9].csv and indexes the same array.

    uv run python src/make_batches_rest.py --dry-run
    uv run python src/make_batches_rest.py

Language interleaving is preserved for the same reason make_batches.py does it:
each batch is one teacher invocation, invocations anchor their probability scale
differently, and if batch boundaries lined up with language boundaries that
drift would land on the labels as language-correlated noise. Round-robin makes
every new batch a miniature of the remaining pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "batches"
INDEX = ROOT / "artifacts" / "teacher_subset_index.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402
from make_batches import detect_language  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only batch this many (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not INDEX.exists():
        raise SystemExit(f"{INDEX} missing - run make_batches.py first")
    index = pd.read_csv(INDEX)
    existing = set(index.StudyInstanceUID)

    tr = pd.read_csv(ROOT / "data" / "train.csv")
    is_gold = tr[LABELS].notna().all(axis=1)
    pool = tr[~is_gold]
    rest = pool[~pool.StudyInstanceUID.isin(existing)].reset_index(drop=True)
    if a.limit:
        rest = rest.head(a.limit)

    # The highest batch number already on disk; keep counting from there so no
    # existing file is overwritten even if the numbering has gaps.
    used = sorted(int(p.stem.split("_")[1]) for p in OUT.glob("batch_[0-9][0-9][0-9].json"))
    start_batch = (used[-1] + 1) if used else 0
    start_id = len(index)

    rest["lang"] = rest.Report.fillna("").map(detect_language)
    rest = rest.sample(frac=1, random_state=a.seed).sort_values("lang", kind="stable")
    rest["gid"] = np.arange(start_id, start_id + len(rest))

    n_out = int(np.ceil(len(rest) / a.batch_size))
    rest["_batch"] = np.tile(np.arange(n_out), int(np.ceil(len(rest) / n_out)))[: len(rest)]

    print(f"index has {len(index)} studies; {len(rest)} remain")
    print(f"writing batches {start_batch:03d}-{start_batch + n_out - 1:03d}, "
          f"ids {start_id}-{start_id + len(rest) - 1}")
    print("\nlanguage mix of the remainder:")
    for k, v in rest.lang.value_counts(normalize=True).items():
        print(f"  {k:4} {v * 100:5.1f}%")

    if a.dry_run:
        print("\ndry run - nothing written")
        return

    for bi, chunk in rest.groupby("_batch"):
        recs = [{"id": int(r.gid), "report": str(r.Report)} for r in chunk.itertuples()]
        path = OUT / f"batch_{start_batch + int(bi):03d}.json"
        if path.exists():
            raise SystemExit(f"{path} already exists - refusing to overwrite")
        path.write_text(json.dumps(recs, ensure_ascii=False, indent=1))

    # Append in gid order so row i of the index stays the study with id i.
    add = rest.sort_values("gid")[["StudyInstanceUID", "lang"]].copy()
    add["stratum"] = "rest"
    pd.concat([index, add], ignore_index=True).to_csv(INDEX, index=False)

    print(f"\nwrote {n_out} batches; index now {start_id + len(rest)} rows")
    print("ids stay positional into artifacts/teacher_subset_index.csv")


if __name__ == "__main__":
    main()
