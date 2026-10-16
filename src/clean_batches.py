"""Rewrite the teacher batches from source text through the current cleaner.

Batches 000-004 were written before the de-identification repair existed and
005-009 before its word-boundary bug was fixed, so the corpus the teacher sees
is inconsistent across batches. Rather than patch the batch files in place -
which only works once, and silently no-ops afterwards - this rebuilds every
report from train.csv, so the script is idempotent and the batch files are
always exactly `clean_report(source)`.

A study inside an ALREADY-LABELLED batch needs re-labelling only if the
cleaning changed something that could flip a finding: a corrupted measurement
("3xintact5x1.8 cm") or a redacted span. Stripping a [DATE] or collapsing
whitespace cannot, and re-labelling those would cost teacher budget for nothing.

    uv run python src/clean_batches.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clean_report import clean_report, corruption_flags

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "artifacts" / "batches"


def main():
    index = pd.read_csv(ROOT / "artifacts" / "teacher_subset_index.csv")
    train = pd.read_csv(ROOT / "data" / "train.csv").set_index("StudyInstanceUID")
    # batch ids are positional into teacher_subset_index.csv
    source = {
        i: str(train.Report.get(uid, ""))
        for i, uid in enumerate(index.StudyInstanceUID)
    }

    labelled = {int(p.stem.split("_")[1]) for p in BATCHES.glob("labels_*.csv")}
    print(f"already-labelled batches: {sorted(labelled)}\n")

    relabel = []
    for path in sorted(BATCHES.glob("batch_[0-9][0-9][0-9].json")):
        bi = int(path.stem.split("_")[1])
        recs = json.loads(path.read_text())
        material = []
        for r in recs:
            raw = source[r["id"]]
            r["report"] = clean_report(raw)
            if any(corruption_flags(raw).values()):
                material.append(r)
        path.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
        mark = "  <- needs relabel" if material and bi in labelled else ""
        print(f"batch_{bi:03d}  {len(material):3d}/{len(recs)} materially damaged{mark}")
        if bi in labelled:
            relabel.extend(material)

    relabel.sort(key=lambda r: r["id"])
    (ROOT / "artifacts" / "relabel_ids.json").write_text(
        json.dumps([r["id"] for r in relabel])
    )
    (BATCHES / "batch_relabel.json").write_text(
        json.dumps(relabel, ensure_ascii=False, indent=1)
    )
    print(f"\n{len(relabel)} studies in labelled batches need re-labelling")
    print("wrote artifacts/batches/batch_relabel.json")


if __name__ == "__main__":
    main()
