"""Single source of truth for the prompt text sent to a Fable labelling agent.

Do not hand-write this prompt per batch. Import build_fable_prompt() and paste
its output into the Agent call, so every batch uses the same instructions.

History: all ten batches (000-009, the full 1,000-study set, completed
2026-08-07) were sent with a hand-written prompt that stated numeric priors
directly ("Priors: ACL 0.41, MCL 0.16, ..."). Those numbers came from the 58
gold studies, which are enriched for abnormality (mean 4.1 positive findings,
zero all-negative cases) and are not a usable estimate of corpus prevalence -
a Kaggle discussion thread independently estimated ACL at ~20% corpus-wide
against 41% in the 58. Feeding a model a specific prior and then citing its
output as evidence for that prior is circular, and it showed: the qwen9b
batch, whose prompt only says "output the prior" with no number, landed at
ACL=0.154; the fable batches, told 0.41, landed at 0.29-0.40 across all ten
(checked 000-004 and 005-009 separately, both contaminated to a similar
degree) - visibly pulled toward the injected value. That bias propagated into
the LoRA student's full-corpus output and from there into every image model's
training targets to date (student ACL mean 0.288).

This version removes the injected numbers and asks for qualitative judgement
instead, so the resulting corpus-wide average becomes usable evidence rather
than a self-fulfilling one. The existing 1,000 fable-labelled studies are not
being redone - re-labelling all of them to fix a second-order bias is not
worth the cost when blend_targets.py (out-of-fold image predictions replacing
the flat-prior fallback for training rows) fixes the sharper underlying
problem anyway: a constant target for every "not mentioned" study prevents the
image model from
learning whatever the pixels actually show for that subset, independent of
which constant was used.
"""

from __future__ import annotations

from labels import CRITERIA, LABELS

DEFINITIONS = "\n".join(f"- **{k}**: {v}" for k, v in CRITERIA.items())
COLUMNS = ", ".join(LABELS)
# The header line must not carry spaces after the commas: pandas keeps them in
# the column name, so `id,ACL, MCL` parses as " MCL" and every downstream lookup
# by LABELS misses. COLUMNS reads better in the prose list, so the two differ.
CSV_COLUMNS = ",".join(LABELS)

# Rule 2 exists in two versions. The original let a thorough negative report pull
# unaddressed findings toward zero, and GPT-5.6 Sol read that hard enough to put
# 91% of Fracture below 0.05 - which buries every present-but-unreported case at
# the bottom of a rank-based metric. The calibrated version forbids the floor.
RULE_TWO_ORIGINAL = """2. **If a report never addresses a finding, do not default to 0, 0.5, or any fixed number for every case.** Use your own clinical judgement of how common that finding actually is, unprompted, in a general clinical population referred for knee MRI — these reports are an ordinary clinical stream, not a curated teaching set, so treat prevalence as meaningfully lower than a case series would suggest. Let your estimate vary by finding (e.g. Effusion and Medial Meniscus tears are relatively common in this population; MCL tears and Lateral OA are less common) and by what the rest of the report implies (an otherwise-thorough negative report weakly implies normality for what it doesn't mention; a terse report implies much less)."""

RULE_TWO_CALIBRATED = """2. **If a report never addresses a finding, that is uncertainty, not absence.** Your probability there must sit in a genuinely intermediate band - roughly 0.10 to 0.35 - chosen from how common the finding is in an ordinary clinical stream referred for knee MRI and how thorough the rest of the report is. **Never put an unaddressed finding below 0.05.** Values under 0.05 are reserved for findings the report explicitly excludes ("no effusion", "ACL intact"). This matters more than it looks: the metric is rank-based, and reports routinely omit findings that are in fact present - fractures and bone contusions especially - so an unaddressed finding scored like an excluded one buries every such case at the bottom of the ranking with no way to separate it. Let the value vary by finding (Effusion and Medial Meniscus tears are relatively common here; MCL tears and Lateral OA are less so) and by the report: one that enumerated many normals implies more than a terse one does."""

TEMPLATE = """You are a musculoskeletal radiologist extracting structured findings from knee MRI reports for a Kaggle competition (RSNA Knee Abnormality Detection). Your output becomes training labels, so accuracy matters more than speed.

## Input
Read `/home/sangwon-park/kaggle/artifacts/batches/batch_{batch:03d}.json` — a JSON array of objects `{{"id": <int>, "report": "<free text>"}}`. The ids are non-contiguous; preserve them exactly as given.

{lang_note}Read each report in its own language; do not translate first.

## Task
For each report, output the probability (0.00–1.00) that each of these 12 findings is PRESENT in that knee:

{columns}

### Definitions
{definitions}

### Rules that decide the score
1. **Mentioned ≠ abnormal.** Reports routinely enumerate normal structures ("ACL is intact", "no effusion"). Read the negation — including postpositive negation (e.g. Turkish "efüzyon izlenmedi" negates *after* the term; do not assume negation words precede what they negate).
{rule_two}
3. **Grade your probabilities — do not answer only 0.0 and 1.0.** The metric is rank-based: a confidently stated finding (0.95) must outrank an equivocal one (0.6), which must outrank an unaddressed one (your own estimate), which must outrank an explicitly excluded one (0.03). Ties destroy AUC.
4. Use the **LIBERAL** reading: any abnormality of the named structure counts, not only a frank tear. Base rates are high and no study in the (enriched) reference set is all-negative — but do not use that enrichment to inflate your baseline for unaddressed findings; see rule 2.
5. Where a report contradicts itself between Findings and Impression, hedge to a mid-range value rather than picking a side.
6. Some reports are terse one-line templates. For these, lean toward your own low-end estimate per rule 2, but still differentiate: a template that explicitly states a normal finding IS evidence against it even though the report is short.

## Output
Write a CSV to `{out_dir}/labels_{batch:03d}.csv` with exactly these columns:

```
id,{csv_columns}
```

One row per input report, ids exactly as given in the input, probabilities to 2 decimals. Every input id must appear exactly once. Quote the `Baker's` header properly for CSV. Do not write any other file and do not modify anything else in the repo.

Reply with only: the number of rows written, and one sentence on anything anomalous you hit."""


def build_fable_prompt(batch: int, lang_hint: str | None = None,
                       out_dir: str = "/home/sangwon-park/kaggle/artifacts/batches",
                       calibrated: bool = False) -> str:
    """batch: the numeric suffix of artifacts/batches/batch_{batch:03d}.json.
    lang_hint: optional one-line language-mix note (see make_batches.py's
    per-batch language breakdown), inserted before the "read in its own
    language" instruction."""
    lang_note = f"{lang_hint}\n\n" if lang_hint else ""
    return TEMPLATE.format(batch=batch, lang_note=lang_note, columns=COLUMNS,
                           csv_columns=CSV_COLUMNS, definitions=DEFINITIONS,
                           out_dir=out_dir.rstrip('/'),
                           rule_two=RULE_TWO_CALIBRATED if calibrated else RULE_TWO_ORIGINAL)


if __name__ == "__main__":
    import sys

    b = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(build_fable_prompt(b))
