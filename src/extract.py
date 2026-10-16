"""Extract 12 finding probabilities from free-text knee MRI reports.

Talks to an OpenAI-compatible endpoint (vLLM), so the same script drives the
local Qwen3.5-9B and any hosted model we want to compare against.

    # serve (reuses the vllm image already on this machine)
    docker run --rm --gpus all -p 8000:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      vllm/vllm-openai:latest --model Qwen/Qwen3.5-9B --max-model-len 8192

    uv run python src/extract.py --split gold      # the 58, for scoring
    uv run python src/extract.py --split all       # all 4,407, for training
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import pandas as pd

from labels import GOLD_PRIORS, LABELS, build_system_prompt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts"

JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_reply(text: str) -> dict[str, float] | None:
    """Pull the JSON object out of a model reply. Returns None if unusable."""
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in LABELS:
        v = raw.get(k)
        if not isinstance(v, (int, float)):
            return None
        out[k] = min(1.0, max(0.0, float(v)))
    return out


def probability_schema() -> dict:
    """JSON schema forcing exactly the 12 findings as bounded floats.

    Without this the model spends its whole budget on chain-of-thought and
    gets truncated before it ever emits the JSON.
    """
    return {
        "type": "object",
        "properties": {
            k: {"type": "number", "minimum": 0.0, "maximum": 1.0} for k in LABELS
        },
        "required": list(LABELS),
        "additionalProperties": False,
    }


async def one(client, sem, model, system, uid, report, max_tokens, guided, retries=3):
    from openai import APIError

    # NOTE: vLLM 0.20.1 silently ignores extra_body={"guided_json": ...} - the
    # model just free-runs and gets truncated mid chain-of-thought. The plain
    # OpenAI json_object mode does constrain it, so use that.
    kwargs = {"response_format": {"type": "json_object"}} if guided else {}

    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": report},
                    ],
                    # deterministic: we compare prompt variants, so sampling
                    # noise would masquerade as a prompt difference
                    temperature=0.0,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except APIError:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            parsed = parse_reply(r.choices[0].message.content or "")
            if parsed is not None:
                return uid, parsed, False
        # fall back to priors rather than dropping the study
        return uid, dict(GOLD_PRIORS), True


async def run(df, model, base_url, concurrency, max_tokens, guided):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=180)
    system = build_system_prompt()
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        one(
            client, sem, model, system,
            r.StudyInstanceUID, str(r.Report), max_tokens, guided,
        )
        for r in df.itertuples()
    ]
    rows, n_failed, done = [], 0, 0
    for coro in asyncio.as_completed(tasks):
        uid, probs, failed = await coro
        rows.append({"StudyInstanceUID": uid, **probs})
        n_failed += failed
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(tasks)}  (unparseable so far: {n_failed})", flush=True)
    print(f"done: {done} studies, {n_failed} fell back to priors")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "all"], default="gold")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", default="qwen9b", help="names the output file")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument(
        "--free-form",
        action="store_true",
        help="let the model reason in prose before the JSON instead of "
        "constraining decoding; needs a much larger --max-tokens",
    )
    a = ap.parse_args()

    tr = pd.read_csv(DATA / "train.csv")
    is_gold = tr[LABELS].notna().all(axis=1)
    df = tr[is_gold] if a.split == "gold" else tr
    print(f"extracting {len(df)} reports with {a.model}")

    preds = asyncio.run(
        run(df, a.model, a.base_url, a.concurrency, a.max_tokens, not a.free_form)
    )
    OUT.mkdir(exist_ok=True)
    path = OUT / f"pseudo_{a.tag}_{a.split}.parquet"
    preds.to_parquet(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
