"""Run the LoRA-distilled 9B over the reports.

Not a flag on extract.py, because three things must match training exactly or
the adapter is being asked a question it was never trained on:

  1. **The system prompt** is the one in make_sft.py, not the longer prompting
     prompt in labels.py. The adapter learned to answer that specific string.
  2. **The report is cleaned.** Training saw clean_report() output, so raw
     de-identification damage at inference is off-distribution.
  3. **Decoding is unconstrained.** extract.py needs response_format=json_object
     because the base model free-runs into chain-of-thought and gets truncated.
     The fine-tuned model emits "\\n</think>\\n\\n{...}" by construction, and
     forcing json_object mode would fight the chat template's think block.

Serve the adapter first. --max-lora-rank matters: it defaults to 16 and this
adapter is rank 32, so vLLM rejects the adapter at load time without it.

    vllm serve Qwen/Qwen3.5-9B --enable-lora --max-lora-rank 32 \\
      --lora-modules knee=artifacts/lora/best --max-model-len 4096

    uv run python src/extract_student.py --split gold   # score it
    uv run python src/extract_student.py --split all    # 4,407 pseudo-labels
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from clean_report import clean_report
from extract import parse_reply
from labels import GOLD_PRIORS, LABELS
from make_sft import build_system

ROOT = Path(__file__).resolve().parent.parent


async def one(client, sem, model, system, uid, report, max_tokens, retries=3):
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
                    max_tokens=max_tokens,
                )
            except APIError:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            parsed = parse_reply(r.choices[0].message.content or "")
            if parsed is not None:
                return uid, parsed, False
        return uid, dict(GOLD_PRIORS), True


async def run(df, model, base_url, concurrency, max_tokens):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=180)
    system = build_system()
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        one(client, sem, model, system, r.StudyInstanceUID, r.report, max_tokens)
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
    print(f"done: {done} studies, {n_failed} fell back to priors", flush=True)
    if n_failed > 0.02 * done:
        print(
            f"WARNING: {n_failed / done:.1%} unparseable. The adapter should emit "
            f"well-formed JSON by construction - check that --lora-modules is "
            f"actually loaded and that --model names the adapter, not the base."
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "all"], default="gold")
    ap.add_argument("--model", default="knee", help="the --lora-modules name")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", default="student")
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()

    tr = pd.read_csv(ROOT / "data" / "train.csv")
    is_gold = tr[LABELS].notna().all(axis=1)
    df = (tr[is_gold] if a.split == "gold" else tr).copy()
    df["report"] = df.Report.fillna("").map(clean_report)
    print(f"extracting {len(df)} reports with {a.model}", flush=True)

    preds = asyncio.run(run(df, a.model, a.base_url, a.concurrency, a.max_tokens))
    path = ROOT / "artifacts" / f"pseudo_{a.tag}_{a.split}.parquet"
    preds.to_parquet(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
