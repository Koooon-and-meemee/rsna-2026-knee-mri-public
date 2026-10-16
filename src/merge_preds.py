#!/usr/bin/env python3
"""Merge ensemble.py --save-preds pools so a new member joins the search pool
without re-inferring the 25 that are already cached.

    uv run python src/merge_preds.py artifacts/preds_pool.npz \
        artifacts/preds_m64gru.npz -o artifacts/preds_pool2.npz

The pools must have been written against the same shards in the same order.
That is true when nothing between the two runs touched artifacts/volumes or
image_dataset.parquet, and false in a way no score would reveal when something
did - the merged pool would just quietly compare member A's series 7 against
member B's series 9. So the alignment columns (T__*, U__*) are compared exactly
and a mismatch is a hard failure, not a warning.
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pools", nargs="+", type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    a = ap.parse_args()

    merged: dict[str, np.ndarray] = {}
    tags: list[str] = []
    for p in a.pools:
        z = np.load(p, allow_pickle=True)
        for split in ("val", "gold_val"):
            for col in (f"T__{split}", f"U__{split}"):
                if col not in merged:
                    merged[col] = z[col]
                elif not np.array_equal(merged[col], z[col]):
                    raise SystemExit(
                        f"{p}: {col} does not match the first pool - these were "
                        "written against different shards or a different dataset "
                        "order, and merging them would misalign every member")
        for t in z["tags"]:
            t = str(t)
            if t in tags:
                print(f"  {t:24s} already present, keeping the first copy")
                continue
            tags.append(t)
            for split in ("val", "gold_val"):
                merged[f"P__{split}__{t}"] = z[f"P__{split}__{t}"]
        print(f"{p}  {len(z['tags'])} members")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, tags=np.array(tags, dtype=object), **merged)
    print(f"wrote {a.out}  {len(tags)} members x {len(merged['T__val'])} val series")


if __name__ == "__main__":
    main()
