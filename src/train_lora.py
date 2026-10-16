"""Distil the fable teacher into Qwen3.5-9B with a LoRA adapter.

The target is calibration, not accuracy. Prompted, this model already knows
roughly which findings a report asserts - it scored 0.861 macro-AUC on the 58
gold studies against the teacher's 0.900. What it cannot do is grade its
answers: told to output a probability it collapses onto 0.0/1.0, leaving 93% of
Fracture predictions tied, and a rank metric cannot see through ties. So it is
being trained on 860 graded teacher targets to learn the output distribution.

Three details that are not defaults.

**Prompt and completion are tokenised separately, then concatenated.** The chat
template's generation prompt ends mid-whitespace ("<think>\\n"), so tokenising
the full string yields different tokens at that boundary than tokenising the
prompt alone. Training on the joint tokenisation would teach the model a prefix
it will never be given at inference. Splitting keeps the two identical.

**Loss is masked to the completion.** Only the JSON answer is supervised; the
system prompt is 800 of the ~1,230 tokens and learning to reproduce it is
wasted capacity.

**Per-example weights are applied.** Studies whose report is bilateral,
compares to a prior, or is shared verbatim with other studies carry noisier
targets; report_quality.py assigns each a weight and the loss honours it.
Without this a 37-study template group would pull 37 times as hard as one
individually-reported knee on an identical, possibly wrong, target.

    uv run python src/train_lora.py
    uv run python src/train_lora.py --epochs 2 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
SFT = ROOT / "artifacts" / "sft"
OUT = ROOT / "artifacts" / "lora"
MODEL = "Qwen/Qwen3.5-9B"


class SFTSet(Dataset):
    def __init__(self, path: Path, tok, max_len: int):
        self.rows, self.tok, self.max_len = [], tok, max_len
        n_trunc = 0
        for line in path.open():
            r = json.loads(line)
            full = tok.apply_chat_template(r["messages"], tokenize=False)
            gen = tok.apply_chat_template(
                r["messages"][:2], tokenize=False, add_generation_prompt=True
            )
            assert full.startswith(gen), "generation prompt is not a text prefix"
            p = tok(gen, add_special_tokens=False)["input_ids"]
            c = tok(full[len(gen) :], add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_len:
                # trim the PROMPT from the left, never the answer: a truncated
                # target would teach the model to emit malformed JSON
                p = p[: max_len - len(c)]
                n_trunc += 1
            self.rows.append((p, c, float(r.get("weight", 1.0))))
        if n_trunc:
            print(f"  {n_trunc} prompts truncated to fit {max_len}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c, w = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + c
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "weight": w,
        }


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k], lab[i, :k], att[i, :k] = b["input_ids"], b["labels"], 1
    return ids, lab, att, torch.tensor([b["weight"] for b in batch], dtype=torch.float)


def weighted_loss(logits, labels, weights) -> torch.Tensor:
    """Mean token loss per example, then a weighted mean across the batch.

    Not a plain token-mean over the whole batch: that would let a long answer
    outvote a short one, and the per-example weight would have nothing to
    attach to.
    """
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    per_tok = torch.nn.functional.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, reduction="none", ignore_index=-100
    )
    mask = (shift_labels != -100).float()
    per_ex = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)
    w = weights.to(per_ex.device)
    return (per_ex * w).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def evaluate(model, loader, pad_id) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        ids, lab, att, w = collate(batch, pad_id)
        out = model(input_ids=ids.cuda(), attention_mask=att.cuda())
        tot += weighted_loss(out.logits, lab.cuda(), w).item() * len(batch)
        n += len(batch)
    model.train()
    return tot / max(n, 1)


def batches(ds, size, shuffle, seed=0):
    idx = list(range(len(ds)))
    if shuffle:
        import random

        random.Random(seed).shuffle(idx)
    # length-sorted buckets keep padding down without breaking the shuffle
    for i in range(0, len(idx), size):
        yield [ds[j] for j in idx[i : i + size]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2304)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print("loading data")
    train = SFTSet(SFT / "train.jsonl", tok, a.max_len)
    val = SFTSet(SFT / "val.jsonl", tok, a.max_len)
    print(f"  {len(train)} train / {len(val)} val")

    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank,
            lora_alpha=a.rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / a.accum)
    total_steps = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0
    )
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nbaseline val loss: {evaluate(model, batches(val, 1, False), pad_id):.4f}")
    print(f"training {total_steps} optimiser steps\n")

    best, t0, step = float("inf"), time.time(), 0
    for ep in range(a.epochs):
        run, seen = 0.0, 0
        for i, batch in enumerate(batches(train, 1, True, a.seed + ep)):
            ids, lab, att, w = collate(batch, pad_id)
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda())
            loss = weighted_loss(out.logits, lab.cuda(), w)
            (loss / a.accum).backward()
            run += loss.item()
            seen += 1
            if (i + 1) % a.accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    el = time.time() - t0
                    print(
                        f"  ep{ep + 1} step {step}/{total_steps}  loss {run / seen:.4f}"
                        f"  lr {sched.get_last_lr()[0]:.2e}"
                        f"  {el / 60:.1f}m elapsed, ~{el / step * (total_steps - step) / 60:.0f}m left"
                    )
                    run, seen = 0.0, 0
        vl = evaluate(model, batches(val, 1, False), pad_id)
        print(f"epoch {ep + 1}: val loss {vl:.4f}" + ("  <- best" if vl < best else ""))
        if vl < best:
            best = vl
            model.save_pretrained(OUT / "best")
            (OUT / "best" / "train_meta.json").write_text(
                json.dumps({"epoch": ep + 1, "val_loss": vl, "args": vars(a)}, indent=1)
            )
    model.save_pretrained(OUT / "last")
    print(f"\ndone in {(time.time() - t0) / 60:.1f}m; best val loss {best:.4f}")
    print(f"adapter: {OUT / 'best'}")


if __name__ == "__main__":
    main()
