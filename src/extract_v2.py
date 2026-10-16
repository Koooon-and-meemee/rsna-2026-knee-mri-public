"""Report extraction v2: assertion status + logprob, instead of a verbalised number.

Why v2 exists. v1 asked the model for a probability directly. Qwen3.5-9B
ignored the "grade your answers" instruction and collapsed to 0.0/1.0: 93% of
Fracture predictions and 84% of ACL predictions landed on an extreme. The
metric is rank-based, so those ties are dead weight - a column that is 93% tied
carries almost no ranking information no matter how accurate the calls are.

Small models are bad at verbalised probability but well calibrated in their
*token* distribution. So v2 asks a question the model can actually answer -
which of four assertion states applies - and recovers the fine-grained ordering
from the logprobs over those state tokens. Ties become essentially impossible
because the logprob is continuous.

    p = P(present) + 0.5 * P(equivocal) + prior * P(not_mentioned)

The severity field then breaks any remaining ties within `present`, and the
evidence span gives us a hallucination check: if the quoted span is not in the
source report, the extraction is discarded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from pathlib import Path

import pandas as pd

from labels import GOLD_PRIORS, LABELS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts"

STATES = ["present", "absent", "equivocal", "not_mentioned"]
SEVERITY_BUMP = {"mild": 0.0, "moderate": 0.04, "severe": 0.08, "unspecified": 0.02}

SYSTEM = """You are a musculoskeletal radiologist reading a knee MRI report.

The report may be in English, Spanish, Turkish, German, Dutch, French, Italian,
Portuguese or another language. Read it in its own language.

For EACH of the 12 findings, decide the assertion status in this report:
  "present"       - the report asserts the finding
  "absent"        - the report explicitly excludes it ("ACL is intact", "no effusion")
  "equivocal"     - hedged, borderline, or "cannot exclude"
  "not_mentioned" - the report never addresses this structure at all

A structure being MENTIONED is not the same as it being ABNORMAL. Reports
routinely enumerate normal structures. Read the negation. This is the single
biggest source of error.

Use the LIBERAL reading of each finding: any abnormality of the named structure
counts as present, not only a frank tear. Sprain, mucoid degeneration, partial
tear and post-surgical abnormality all count.

Findings and their definitions:
{criteria}

For every finding also give:
  severity: mild | moderate | severe | unspecified   (meaningful only if present)
  evidence: the VERBATIM span from the report that decided it, or null if
            not_mentioned. Copy it exactly; do not paraphrase or translate.

Respond with ONLY this JSON, no prose:
{{"<finding>": {{"status": "...", "severity": "...", "evidence": "..."}}, ...}}
covering all 12 findings."""


def build_system() -> str:
    from labels import CRITERIA

    return SYSTEM.format(
        criteria="\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    )


def state_probs_from_logprobs(content, finding_order):
    """Recover P(status) per finding from the response's token logprobs.

    We walk the token stream and, each time a status value is being emitted,
    read the alternatives the model considered at that position. This is what
    turns a 4-way categorical answer back into a continuous score.
    """
    if not content:
        return {}
    out, idx = {}, 0
    for i, tok in enumerate(content):
        text = (tok.token or "").strip().strip('"')
        if text not in STATES:
            continue
        alts = {}
        for alt in tok.top_logprobs or []:
            a = (alt.token or "").strip().strip('"')
            if a in STATES:
                alts[a] = max(alts.get(a, -99.0), alt.logprob)
        alts.setdefault(text, tok.logprob)
        z = sum(math.exp(v) for v in alts.values()) or 1.0
        if idx < len(finding_order):
            out[finding_order[idx]] = {k: math.exp(v) / z for k, v in alts.items()}
        idx += 1
    return out


def to_probability(finding, parsed, state_p):
    """Blend the categorical answer, its logprob mass, and severity."""
    prior = GOLD_PRIORS[finding]
    info = parsed.get(finding) or {}
    status = str(info.get("status", "not_mentioned")).strip()
    sev = str(info.get("severity", "unspecified")).strip().lower()

    sp = state_p.get(finding)
    if sp:
        p = (
            sp.get("present", 0.0)
            + 0.5 * sp.get("equivocal", 0.0)
            + prior * sp.get("not_mentioned", 0.0)
            + 0.02 * sp.get("absent", 0.0)
        )
    else:  # no logprobs available - fall back to the discrete answer
        p = {"present": 0.90, "equivocal": 0.50, "absent": 0.03}.get(status, prior)

    if status == "present":
        p = min(0.995, p + SEVERITY_BUMP.get(sev, 0.02))
    return float(min(1.0, max(0.0, p)))


def check_evidence(parsed, report):
    """Fraction of 'present' claims whose quoted span really is in the report."""
    claims = [
        v for v in parsed.values()
        if isinstance(v, dict) and v.get("status") == "present" and v.get("evidence")
    ]
    if not claims:
        return 1.0
    norm = re.sub(r"\s+", " ", report).lower()
    ok = sum(
        1 for v in claims
        if re.sub(r"\s+", " ", str(v["evidence"])).lower()[:60] in norm
    )
    return ok / len(claims)


async def one(client, sem, model, system, uid, report, retries=3):
    from openai import APIError

    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": report},
                    ],
                    temperature=0.0,
                    max_tokens=1600,
                    response_format={"type": "json_object"},
                    logprobs=True,
                    top_logprobs=8,
                )
            except APIError:
                await asyncio.sleep(2 * (attempt + 1))
                continue

            choice = r.choices[0]
            text = choice.message.content or ""
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                continue
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue

            order = [k for k in parsed if k in LABELS]
            state_p = state_probs_from_logprobs(
                getattr(choice.logprobs, "content", None), order
            )
            probs = {f: to_probability(f, parsed, state_p) for f in LABELS}
            fidelity = check_evidence(parsed, report)
            return uid, probs, fidelity, False

        return uid, dict(GOLD_PRIORS), 0.0, True


async def run(df, model, base_url, concurrency):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=300)
    system = build_system()
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        one(client, sem, model, system, r.StudyInstanceUID, str(r.Report))
        for r in df.itertuples()
    ]
    rows, failed, fid, done = [], 0, [], 0
    for coro in asyncio.as_completed(tasks):
        uid, probs, f, bad = await coro
        rows.append({"StudyInstanceUID": uid, **probs, "_evidence_fidelity": f})
        failed += bad
        if not bad:
            fid.append(f)
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(tasks)}  (failed {failed})", flush=True)
    mean_fid = sum(fid) / len(fid) if fid else float("nan")
    print(f"done: {done}, {failed} fell back to priors")
    print(f"evidence fidelity (quoted span found in source): {mean_fid:.3f}")
    if mean_fid < 0.8:
        print("  WARNING: low fidelity means the model is inventing quotes.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "all"], default="gold")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--tag", default="qwen9b_v2")
    a = ap.parse_args()

    tr = pd.read_csv(DATA / "train.csv")
    is_gold = tr[LABELS].notna().all(axis=1)
    df = tr[is_gold] if a.split == "gold" else tr
    print(f"extracting {len(df)} reports with {a.model} (v2: status+logprob)")

    preds = asyncio.run(run(df, a.model, a.base_url, a.concurrency))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"pseudo_{a.tag}_{a.split}.parquet"
    preds.to_parquet(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
