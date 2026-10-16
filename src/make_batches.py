"""Pick a stratified subset of reports for the expensive teacher to label.

The subset has two jobs: it is the LoRA training set for distilling the big
teacher into the local 9B, and it is the pool the distilled model gets checked
against. Both jobs want the same thing - coverage, not a random draw.

Stratified on:
  - report language (extraction difficulty varies by language, and this is
    where the small model is weakest, so the LoRA needs to see all of them)
  - the 9B's predicted positive count, bucketed (samples the easy all-negative
    reports and the busy multi-finding ones in proportion)
  - report length quartile (short template reports behave very differently
    from long prose ones)

The 58 gold studies are excluded - they stay uncontaminated as the acceptance
test.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "batches"

LANG_PATTERNS = {
    "es": r"\b(hallazgos|menisco|rodilla|derrame|rotura|impresi[óo]n)\b",
    "tr": r"\b(mevcuttur|izlenmi|sinyal|d[üu]zey|y[ıi]rt[ıi]k|eklem)\b",
    "de": r"\b(kein|mit|nachweis|kniegelenk|meniskus|erguss)\b",
    "nl": r"\b(geen|meniscus|kniegewricht|vocht|rechts|links)\b",
    "fr": r"\b(genou|m[ée]nisque|[ée]panchement|sans|avec)\b",
    "it": r"\b(ginocchio|menisco|versamento|della|nella)\b",
    "pt": r"\b(joelho|menisco|derrame|sem|com)\b",
    "en": r"\b(the|there|is|no|and|with|meniscus|effusion)\b",
}


def detect_language(text: str) -> str:
    t = (text or "").lower()
    scores = {k: len(re.findall(p, t)) for k, p in LANG_PATTERNS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 2 else "unk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="how many to label")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr = pd.read_csv(ROOT / "data" / "train.csv")
    is_gold = tr[LABELS].notna().all(axis=1)
    pool = tr[~is_gold].reset_index(drop=True)

    # the 9B pass over everything gives us a cheap difficulty signal
    weak = pd.read_parquet(ROOT / "artifacts" / "pseudo_qwen9b_all.parquet")
    pool = pool.merge(weak, on="StudyInstanceUID", how="left", suffixes=("", "_p"))
    npos = pool[LABELS].gt(0.5).sum(axis=1)

    pool["lang"] = pool.Report.fillna("").map(detect_language)
    pool["npos_bin"] = pd.cut(npos, [-1, 0, 2, 4, 12], labels=["0", "1-2", "3-4", "5+"])
    pool["len_q"] = pd.qcut(pool.Report.fillna("").str.len(), 4, labels=list("abcd"))
    pool["stratum"] = (
        pool.lang.astype(str) + "|" + pool.npos_bin.astype(str) + "|" + pool.len_q.astype(str)
    )

    # proportional allocation, but guarantee at least one from every stratum so
    # rare languages are not dropped entirely
    rng = np.random.default_rng(a.seed)
    counts = pool.stratum.value_counts()
    take = np.maximum(1, np.round(counts / counts.sum() * a.n).astype(int))
    while take.sum() > a.n:  # trim the largest strata back down
        take[take.idxmax()] -= 1

    picked = pd.concat(
        [
            pool[pool.stratum == s].sample(min(k, (pool.stratum == s).sum()), random_state=rng.integers(1 << 30))
            for s, k in take.items()
        ]
    ).reset_index(drop=True)

    print(f"pool {len(pool)} -> picked {len(picked)} across {picked.stratum.nunique()} strata")
    print("\nlanguage mix (picked vs pool):")
    a_mix = picked.lang.value_counts(normalize=True)
    b_mix = pool.lang.value_counts(normalize=True)
    for k in b_mix.index:
        print(f"  {k:4} {a_mix.get(k, 0) * 100:5.1f}%  vs {b_mix[k] * 100:5.1f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("batch_*.json"):
        f.unlink()
    picked[["StudyInstanceUID", "lang", "stratum"]].to_csv(
        ROOT / "artifacts" / "teacher_subset_index.csv", index=False
    )

    # Interleave across languages before chunking. Each batch is labelled by a
    # separate teacher invocation, and separate invocations anchor their
    # probability scale slightly differently. If batch boundaries lined up with
    # language boundaries - which they do if you chunk the stratum-sorted frame
    # directly - that per-invocation drift would land on the pseudo-labels as
    # language-correlated noise, which is exactly the confound we are trying not
    # to introduce. Round-robin makes every batch look like the whole pool.
    picked = picked.sample(frac=1, random_state=a.seed).sort_values(
        "lang", kind="stable"
    )
    n_out = int(np.ceil(len(picked) / a.batch_size))
    picked["_batch"] = np.tile(np.arange(n_out), int(np.ceil(len(picked) / n_out)))[
        : len(picked)
    ]

    n_batches = 0
    for bi, chunk in picked.groupby("_batch"):
        i = bi * a.batch_size
        recs = [
            {"id": int(gid), "report": str(r.Report)}
            for gid, r in zip(chunk.index, chunk.itertuples())
        ]
        path = OUT / f"batch_{i // a.batch_size:03d}.json"
        path.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
        n_batches += 1
    print(f"\nwrote {n_batches} batches to {OUT}")
    print("ids are positional into artifacts/teacher_subset_index.csv")


if __name__ == "__main__":
    main()
