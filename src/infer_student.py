"""Run the distilled student over the reports with transformers, not vLLM.

vLLM cannot serve this model. Qwen3.5-9B is multimodal, and the thing we
fine-tuned is its text tower: AutoModelForCausalLM returns that tower, PEFT
wrapped it, and the merged checkpoint carries a Qwen3_5TextConfig. vLLM routes
the name through its multimodal path and rejects the config outright. The same
mismatch is why serving the adapter silently no-opped - vLLM addresses the
language model's layers by a path the adapter's keys do not use.

Rather than translate key names or perform surgery on the config - both of
which can appear to work while quietly loading the wrong weights - this drives
the exact model object that was trained. Slower than vLLM, and correct by
construction.

Batched greedy generation with left padding. Prompts are built by the same
chat template call training used, so the tokens the model sees at position 0
are the tokens it saw in training.

    uv run python src/infer_student.py --split gold --limit 8   # smoke test
    uv run python src/infer_student.py --split gold
    uv run python src/infer_student.py --split all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from clean_report import clean_report
from extract import parse_reply
from labels import GOLD_PRIORS, LABELS
from make_sft import build_system

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "artifacts" / "merged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "all"], default="gold")
    ap.add_argument("--model", type=Path, default=MERGED)
    ap.add_argument("--tag", default="student")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0, help="only the first N, for smoke tests")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(a.model))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tr = pd.read_csv(ROOT / "data" / "train.csv")
    is_gold = tr[LABELS].notna().all(axis=1)
    df = (tr[is_gold] if a.split == "gold" else tr).copy()
    if a.limit:
        df = df.head(a.limit)
    df["report"] = df.Report.fillna("").map(clean_report)
    print(f"{len(df)} reports, batch {a.batch}", flush=True)

    system = build_system()
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": r}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in df.report
    ]

    model = AutoModelForCausalLM.from_pretrained(
        str(a.model), dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()

    rows, n_failed, t0 = [], 0, time.time()
    for i in range(0, len(prompts), a.batch):
        chunk = prompts[i : i + a.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=a.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for j, seq in enumerate(out):
            text = tok.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = parse_reply(text)
            if parsed is None:
                n_failed += 1
                parsed = dict(GOLD_PRIORS)
            rows.append({"StudyInstanceUID": df.StudyInstanceUID.iloc[i + j], **parsed})
        done = len(rows)
        el = time.time() - t0
        print(
            f"  {done}/{len(prompts)}  unparseable {n_failed}"
            f"  {el / 60:.1f}m elapsed, ~{el / done * (len(prompts) - done) / 60:.0f}m left",
            flush=True,
        )

    print(f"done: {len(rows)} studies, {n_failed} fell back to priors")
    if n_failed > 0.02 * len(rows):
        print(
            f"WARNING: {n_failed / len(rows):.1%} unparseable. The fine-tuned model "
            f"should emit well-formed JSON by construction - check that --model "
            f"points at the merged checkpoint and not the base."
        )
    path = ROOT / "artifacts" / f"pseudo_{a.tag}_{a.split}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
