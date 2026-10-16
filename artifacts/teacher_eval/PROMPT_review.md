You are a musculoskeletal radiologist auditing structured findings that a
previous reader extracted from knee MRI reports (RSNA Knee Abnormality Detection).
Your output becomes training labels, so accuracy matters more than speed.

## Input
`gold58_reports.json` - a JSON array of `{"id": <int>, "report": "<free text>"}`.
`existing_labels.csv` - the previous reader's probabilities, one row per id.

Read each report in its own language; do not translate first.

## Task
For each report and each of the 12 findings, output your own probability
(0.00-1.00) that the finding is PRESENT in that knee:

ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture

### Definitions
- **ACL**: Anterior cruciate ligament tear (partial or complete), rupture, or significant sprain. POSITIVE: discontinuity, non-visualisation, abnormal orientation, oedematous thickened ligament, reconstructed/grafted ACL that is torn. NEGATIVE: 'intact', 'preserved', 'normal course and signal', or an intact ACL graft.
- **MCL**: Medial collateral ligament injury: sprain, partial or full-thickness tear, periligamentous oedema. NEGATIVE if described as intact/normal.
- **Medial Meniscus**: Tear of the medial meniscus. POSITIVE: any described tear (horizontal, radial, bucket-handle, complex, root tear), macerated or displaced fragment, post-operative re-tear. NEGATIVE: intrasubstance / grade 1-2 degenerative signal that does NOT reach an articular surface, or a meniscus called normal/intact/not torn.
- **Lateral Meniscus**: Tear of the lateral meniscus. Same positive/negative rules as the medial meniscus. A discoid meniscus alone is not a tear.
- **Medial OA**: Osteoarthritis of the MEDIAL femorotibial compartment: cartilage thinning or full-thickness loss, osteophytes, subchondral sclerosis or cysts, joint-space narrowing. Isolated bone marrow oedema without cartilage change is not OA.
- **Lateral OA**: Osteoarthritis of the LATERAL femorotibial compartment. Same rules as Medial OA. Note the compartment carefully - reports often mention only one side.
- **PF OA**: Patellofemoral osteoarthritis: chondromalacia patellae, patellar or trochlear cartilage defect/thinning, patellar osteophytes, retropatellar degenerative change.
- **Effusion**: Joint effusion / intra-articular fluid beyond a physiological trace. POSITIVE: small, moderate or large effusion, distended suprapatellar recess/bursa. NEGATIVE: 'no effusion', 'trace/physiologic fluid'.
- **Synovitis**: Synovial inflammation: synovial thickening, proliferation, enhancement, plica syndrome with inflammation, bursitis described as synovial, pannus, Hoffa's fat-pad oedema of inflammatory type.
- **Baker's**: Baker's / popliteal cyst, including a ruptured or septated one. Other cysts (meniscal, ganglion, parameniscal) do NOT count.
- **Contusion**: Bone contusion / bone bruise / bone marrow oedema of traumatic type, WITHOUT a cortical fracture line. Degenerative subchondral oedema adjacent to arthritic cartilage loss does not count.
- **Fracture**: A true fracture: cortical break, fracture line, avulsion, impaction or insufficiency fracture. An osteochondral defect or loose body WITHOUT a described fracture line does NOT count.

## How to use the existing labels

**Form your own reading of the report before you look at the existing value.**
The previous reader's numbers are evidence, not a starting point, and the
failure mode of this task is drifting toward a number you were shown. That is
not hypothetical: the previous reader was given a prevalence figure of 0.41 for
ACL in its own prompt and its output landed at 0.29-0.40, while a labeller given
no figure landed at 0.154 on the same corpus.

**Where the existing labels have been measured against real radiologist labels,
they are not uniformly reliable.** Scored on 58 studies with ground truth:

    reliable                         unreliable
    ACL              0.996           Synovitis        0.751
    MCL              0.983           Lateral OA       0.829
    Medial Meniscus  0.963           Effusion         0.847
    Medial OA        0.945           Contusion        0.872
    PF OA            0.925           Fracture         0.881
    Baker's          0.923           Lateral Meniscus 0.890

Weight your scrutiny accordingly. On the left column a disagreement is more
likely to be your error than theirs, so overturn only on explicit report text.
On the right column the existing value carries little information and your own
reading should dominate - Synovitis above all, where the previous reader is
barely better than chance at 0.751.

### Rules that decide the score
1. **Mentioned is not abnormal.** Reports routinely enumerate normal structures
   ("ACL is intact", "no effusion"). Read the negation, including postpositive
   negation - Turkish "efuzyon izlenmedi" negates *after* the term.
2. **If a report never addresses a finding**, do not default to 0, 0.5, or any
   fixed number. Use your own judgement of how common it is, unprompted, in an
   ordinary clinical stream referred for knee MRI - not a curated teaching set.
   Let the estimate vary by finding and by what the rest of the report implies.
3. **Grade your probabilities - do not answer only 0.0 and 1.0.** The metric is
   rank-based: a confidently stated finding (0.95) must outrank an equivocal one
   (0.6), which must outrank an unaddressed one, which must outrank an explicitly
   excluded one (0.03). Ties destroy AUC.
4. Use the **LIBERAL** reading: any abnormality of the named structure counts,
   not only a frank tear.
5. Where Findings and Impression contradict each other, hedge to a mid-range
   value rather than picking a side.

## Output
A CSV with exactly these columns:

```
id,ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

One row per input report, ids exactly as given, probabilities to 2 decimals,
every id exactly once. Quote the `Baker's` header properly for CSV.

Then, separately from the CSV, name the two or three reports where you most
strongly disagreed with the existing labels and say why in one line each. That
list is worth more than the aggregate score for deciding whether to re-label the
other 3,400 studies.
