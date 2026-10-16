"""Export the teacher labels as a LoRA fine-tuning set for the local 9B.

Why fine-tune rather than keep prompting. Prompting Qwen3.5-9B for a verbalised
probability failed in a specific way: it ignored the "grade your answers"
instruction and collapsed onto 0.0/1.0, leaving 93% of Fracture predictions
tied. A rank metric cannot see through ties. Fine-tuning on 1,000 graded
teacher targets is aimed exactly at that failure - the model is being taught
the output distribution, not just the task.

Two things here are not boilerplate.

**The split is grouped by report text.** 197 studies in the corpus share their
report verbatim with another study; 50 of them are in this labelled set. Split
those at random and the identical text lands on both sides, so validation loss
measures memorisation. Studies sharing a report always go to the same side.

**Defective studies are excluded from validation, not from training.** A
bilateral report yields a confidently wrong target, so scoring the model
against it punishes the model for being right. They stay in training at their
sample weight, where the noise is diluted; they are useless as a yardstick.

    uv run python src/make_sft.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labels import CRITERIA, LABELS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "sft"

SYSTEM = """You are a musculoskeletal radiologist reading a knee MRI report.

The report may be in any of the languages this dataset uses; read it in its own
language. For each of the 12 findings below, output the probability (0.00-1.00)
that the finding is PRESENT in this knee.

A structure being MENTIONED is not the same as it being ABNORMAL. Reports
routinely enumerate normal structures ("ACL is intact", "no effusion"). Read
the negation.

If the report never addresses a finding, output its prior rather than 0.

Findings:
{criteria}

Respond with ONLY a JSON object mapping each finding to its probability."""


def build_system() -> str:
    return SYSTEM.format(criteria="\n".join(f"- {k}: {v}" for k, v in CRITERIA.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    labels = pd.read_parquet(ROOT / "artifacts" / "teacher_labels.parquet")
    quality = pd.read_csv(ROOT / "artifacts" / "report_quality.csv")
    train = pd.read_csv(ROOT / "data" / "train.csv")[["StudyInstanceUID", "Report"]]

    df = labels.merge(quality, on="StudyInstanceUID").merge(train, on="StudyInstanceUID")
    from clean_report import clean_report

    df["report"] = df.Report.fillna("").map(clean_report)
    df = df[df.sample_weight > 0].reset_index(drop=True)
    print(f"{len(df)} studies with a usable report and a teacher label")

    # group id = the report text itself, so duplicates cannot straddle the split
    df["group"] = df.report.factorize()[0]
    groups = df.group.unique()
    rng = np.random.default_rng(a.seed)
    rng.shuffle(groups)

    # walk groups until the validation side is full, so the split lands on
    # group boundaries rather than cutting through one
    val_target = int(len(df) * a.val_frac)
    val_groups, n = set(), 0
    for g in groups:
        if n >= val_target:
            break
        val_groups.add(g)
        n += int((df.group == g).sum())

    is_val = df.group.isin(val_groups)
    defective = df.bilateral | df.prior_comparison | (df.duplicate_group > 1)
    # a defective study in the val split is moved to train, not dropped
    moved = int((is_val & defective).sum())
    is_val = is_val & ~defective

    print(f"split: {int((~is_val).sum())} train / {int(is_val.sum())} val")
    print(f"  {len(val_groups)} report groups held out")
    print(f"  {moved} defective studies moved out of val into train")
    print(f"  shared report text across the split: "
          f"{len(set(df.report[is_val]) & set(df.report[~is_val]))}")

    system = build_system()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", df[~is_val]), ("val", df[is_val])]:
        path = OUT / f"{name}.jsonl"
        with path.open("w") as fh:
            for _, r in rows.iterrows():
                target = {c: round(float(r[c]), 2) for c in LABELS}
                fh.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": r.report},
                        {"role": "assistant", "content": json.dumps(target)},
                    ],
                    "weight": round(float(r.sample_weight), 4),
                    "StudyInstanceUID": r.StudyInstanceUID,
                }, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
