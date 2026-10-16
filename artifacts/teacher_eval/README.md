# Candidate teacher evaluation — 58 studies, real radiologist labels

Every image model in this project trains on labels a language model extracted
from radiology reports. That labeller agrees with real radiologists at **0.900
mean AUC**, and no image model can beat its own labels, so 0.900 is the ceiling
the whole pipeline has been sitting under since 13 August.

This bundle measures a candidate replacement labeller on the 58 studies where
real radiologist labels exist, **before** spending anything on the other 3,400.

## Two tracks

Run both. They fail differently, which is the point.

    fresh     PROMPT.md         label from the report alone, no anchor
    review    PROMPT_review.md  audit the existing labels and revise

**They are not symmetric, and the review track is the compromised one.** Showing
a model an existing number pulls its answer toward that number - measured in this
project, not assumed: the previous labeller was told ACL prevalence was 0.41 and
landed at 0.29-0.40, while one told nothing landed at 0.154. The review prompt
warns against this explicitly and tells the reviewer which findings are worth
overturning, but the bias cannot be removed by instruction alone.

So treat `fresh` as the measurement and `review` as the cheaper, safer edit. If
fresh wins outright, the review track was never needed. If review wins, check
that it won on the findings where the existing labels are weak (Synovitis,
Lateral OA, Effusion) rather than by copying the existing labels where they are
already strong - `score.py` prints the per-finding column for exactly this.

Scoring both also makes a third option available for free: averaging the two
outputs. Two labellers that fail differently usually beat either alone.

## Files

    PROMPT.md                fresh-labelling prompt (the current labeller's own)
    PROMPT_review.md         review-and-revise prompt for the second track
    existing_labels.csv      the current labels, input to the review track
    gold58_reports.json      the 58 reports, `id` 0-57, mixed languages
    gold58_truth.csv         the real radiologist labels, row i = report id i
    score.py                 scorer, standard library only
    example_fable_output.csv the current labeller's answers, as a format example

## Run

Feed each report in `gold58_reports.json` to the candidate model using
`PROMPT.md`, collect a CSV with `id` plus the twelve finding columns as
probabilities in [0, 1], then:

    python3 score.py candidate.csv

It prints per-finding AUC beside both existing labellers and the mean.

## What counts as a win

    fable    0.9004    produced the 1,000-study set
    student  0.8993    the LoRA distillation that labels the full corpus

**Read the per-finding column, not the mean.** The two existing labellers are
already near-perfect on findings a report states outright and weak on findings a
report grades or hedges:

    ACL              0.996      Effusion       0.847
    MCL              0.983      Lateral OA     0.829
    Medial Meniscus  0.963      Synovitis      0.751

Nothing is available on ACL — it is at 0.996. The entire headroom is in the
bottom five, and a candidate that ties on the mean while fixing Synovitis and
Effusion is worth far more than one that ties by being better at ligaments.
Lifting the worst three to 0.95 alone would move the labeller from 0.900 to
0.931.

There is a second reason to watch those findings specifically. The image model
**already beats the labeller** on Effusion (0.957 vs 0.847), Baker's, Medial OA
and Contusion — the things you can see but reports often omit. So for those, a
better labeller competes with simply trusting the image. Where a candidate can
win outright is the hedged-but-not-visible group, Synovitis above all.

## Caveats

58 studies puts about ±0.06 of standard error on any single finding. Treat the
ordering as the finding and the values as indicative, and do not tune anything
per-finding on this set — it is small enough to fit noise.

The reports are multilingual (Spanish, English and others). The current prompt
tells the model to read each in its own language rather than translating.

## How far to take it, if the candidate wins

Distillation is currently lossless: fable scores 0.9004 on these 58 studies and
the LoRA student trained on fable's 1,000 scores 0.8993 - a gap of 0.0011 on
studies neither of them was fitted to. That is why only 1,000 studies have ever
been labelled directly.

**That number is not transferable to a stronger teacher.** The student is a small
LoRA, and a student's ceiling is its own capacity: reproducing a 0.90 labeller
says nothing about whether it can reproduce a 0.93 one. If the candidate wins by
a wide margin, distillation may swallow the win.

So the 1,000-study stage is not a cheaper substitute for labelling everything.
It is the measurement that decides whether labelling everything is necessary:

1. **58 studies.** Does the candidate beat 0.9004 at all, and on the findings
   that matter? Nothing else is worth doing until this passes.
2. **1,000 studies**, then retrain the student and **score the student on these
   same 58**. If the student lands near the candidate, distillation held and the
   corpus is already as good as it can get. If it lands well short, the student
   is the bottleneck.
3. **All 4,407**, only if step 2 shows the student cannot carry it.

Skipping step 2 costs something either way: run everything without it and you may
spend the full corpus to fix a loss that was never there; stop at 1,000 without
it and you may hand back the label-quality advantage that is the only thing
separating this from every other team holding the same reports.

## A formatting bug that would have killed the scoring run

`fable_prompt.py` built one `COLUMNS` string, `", ".join(LABELS)`, and used it
both for the prose list of findings and for the CSV header line. The prose wants
those spaces; the header does not. `id,ACL, MCL, ...` parses back as a column
named `" MCL"`, and every lookup by `LABELS` then misses - so `compare.py` would
have died on a KeyError *after* the labelling was already paid for.

It had never fired because the existing labeller wrote sensible CSV instead of
following the spec literally. Split into `COLUMNS` (prose) and `CSV_COLUMNS`
(header) on 27 August, in `src/fable_prompt.py` and in both bundle prompts.
