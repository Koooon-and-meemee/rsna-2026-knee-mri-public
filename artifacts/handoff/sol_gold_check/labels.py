"""12 target findings for RSNA Knee Abnormality Detection.

CRITERIA are written to be read by an LLM extracting findings from free-text
radiology reports. They encode two things the naive keyword approach misses:
  1. negation ("ACL is intact" mentions ACL but is a negative finding)
  2. degree thresholds (intrasubstance signal is not a tear; trace fluid is
     not an effusion)
"""

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

CRITERIA = {
    "ACL": (
        "Anterior cruciate ligament tear (partial or complete), rupture, or "
        "significant sprain. POSITIVE: discontinuity, non-visualisation, "
        "abnormal orientation, oedematous thickened ligament, reconstructed/"
        "grafted ACL that is torn. NEGATIVE: 'intact', 'preserved', 'normal "
        "course and signal', or an intact ACL graft."
    ),
    "MCL": (
        "Medial collateral ligament injury: sprain, partial or full-thickness "
        "tear, periligamentous oedema. NEGATIVE if described as intact/normal."
    ),
    "Medial Meniscus": (
        "Tear of the medial meniscus. POSITIVE: any described tear "
        "(horizontal, radial, bucket-handle, complex, root tear), macerated or "
        "displaced fragment, post-operative re-tear. NEGATIVE: intrasubstance "
        "/ grade 1-2 degenerative signal that does NOT reach an articular "
        "surface, or a meniscus called normal/intact/not torn."
    ),
    "Lateral Meniscus": (
        "Tear of the lateral meniscus. Same positive/negative rules as the "
        "medial meniscus. A discoid meniscus alone is not a tear."
    ),
    "Medial OA": (
        "Osteoarthritis of the MEDIAL femorotibial compartment: cartilage "
        "thinning or full-thickness loss, osteophytes, subchondral sclerosis "
        "or cysts, joint-space narrowing. Isolated bone marrow oedema without "
        "cartilage change is not OA."
    ),
    "Lateral OA": (
        "Osteoarthritis of the LATERAL femorotibial compartment. Same rules "
        "as Medial OA. Note the compartment carefully - reports often mention "
        "only one side."
    ),
    "PF OA": (
        "Patellofemoral osteoarthritis: chondromalacia patellae, patellar or "
        "trochlear cartilage defect/thinning, patellar osteophytes, "
        "retropatellar degenerative change."
    ),
    "Effusion": (
        "Joint effusion / intra-articular fluid beyond a physiological trace. "
        "POSITIVE: small, moderate or large effusion, distended suprapatellar "
        "recess/bursa. NEGATIVE: 'no effusion', 'trace/physiologic fluid'."
    ),
    "Synovitis": (
        "Synovial inflammation: synovial thickening, proliferation, "
        "enhancement, plica syndrome with inflammation, bursitis described as "
        "synovial, pannus, Hoffa's fat-pad oedema of inflammatory type."
    ),
    "Baker's": (
        "Baker's / popliteal cyst, including a ruptured or septated one. "
        "Other cysts (meniscal, ganglion, parameniscal) do NOT count."
    ),
    "Contusion": (
        "Bone contusion / bone bruise / bone marrow oedema of traumatic type, "
        "WITHOUT a cortical fracture line. Degenerative subchondral oedema "
        "adjacent to arthritic cartilage loss does not count."
    ),
    "Fracture": (
        "A true fracture: cortical break, fracture line, avulsion, impaction "
        "or insufficiency fracture. An osteochondral defect or loose body "
        "WITHOUT a described fracture line does NOT count."
    ),
}

SYSTEM_PROMPT = """You are a musculoskeletal radiologist reading knee MRI reports.

The reports come from 16 sites worldwide and may be in English, Spanish,
Turkish, German, Dutch, French, Italian, Portuguese or another language. Read
them in whatever language they are written; do not translate first.

For each of the 12 findings below, output the probability (0.00-1.00) that the
finding is PRESENT in this knee.

Rules that matter:
- A structure being MENTIONED is not the same as it being ABNORMAL. Reports
  routinely list normal structures ("ACL is intact", "no effusion"). Read the
  negation.
- If the report genuinely does not address a finding, output the prior
  probability for that finding rather than 0. Absence of evidence is weak
  evidence of absence, not proof.
- Respect the degree thresholds in the criteria (e.g. intrasubstance signal is
  not a meniscal tear).
- Distinguish medial from lateral. Reports often describe only one compartment.
- GRADE your probabilities. Do not answer only 0.0 and 1.0. The scoring metric
  ranks studies against each other, so a confidently-stated finding (0.95) must
  outrank an equivocal one (0.6), which must outrank an unmentioned one (the
  prior), which must outrank an explicitly excluded one (0.03). Reserve values
  below 0.05 and above 0.95 for reports that state the finding outright.

Findings:
{criteria}

Respond with ONLY a JSON object, no prose:
{{{schema}}}"""


def build_system_prompt() -> str:
    criteria = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    schema = ", ".join(f'"{k}": 0.0' for k in LABELS)
    return SYSTEM_PROMPT.format(criteria=criteria, schema=schema)


# Prior positive rates measured on the 58 expert-labelled studies. NOTE: those
# 58 are enriched for abnormality (mean 4.1 positive findings, zero all-negative
# cases) and are NOT a usable estimate of corpus- or test-set prevalence - a
# Kaggle discussion thread independently estimated ACL at ~20% corpus-wide
# against 41% here. Do not feed these numbers to an extractor as a prior: the
# qwen9b batch (whose prompt never states a number, only "output the prior")
# landed at 0.154 for ACL, while an earlier fable batch that WAS told "prior:
# 0.41" in the prompt text landed at 0.296 - visibly pulled toward the injected
# number. That is circularity, not measurement: telling a model what prevalence
# to expect and then citing its output as evidence of that prevalence is not
# an independent estimate. Kept here only as a historical/diagnostic reference;
# nothing downstream should read from this dict as a probability. See
# build_system_prompt(), which does not inject these numbers.
GOLD_PRIORS = {
    "ACL": 0.41,
    "MCL": 0.16,
    "Medial Meniscus": 0.45,
    "Lateral Meniscus": 0.40,
    "Medial OA": 0.26,
    "Lateral OA": 0.19,
    "PF OA": 0.36,
    "Effusion": 0.60,
    "Synovitis": 0.47,
    "Baker's": 0.21,
    "Contusion": 0.33,
    "Fracture": 0.31,
}
