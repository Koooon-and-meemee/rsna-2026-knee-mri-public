"""Merge the LoRA adapter into the base weights and save a standalone model.

Why merge instead of serving the adapter. vLLM loads the adapter without
complaint - "Loaded new LoRA adapter: name 'knee'" - and then produces output
byte-identical to the base model. Qwen3.5-9B is a Qwen3_5ForConditionalGeneration,
and PEFT saved the adapter under `base_model.model.model.layers.N...` while
vLLM addresses the language model's layers by a different path, so every
weight silently fails to bind. A silent no-op is the worst failure mode
available: the pipeline runs, the numbers look plausible, and the model is the
one we did not train.

Merging sidesteps the naming question entirely - the result is an ordinary
checkpoint with no adapter to bind - and costs ~19GB of disk.

The script refuses to write unless the merged model actually differs from the
base, because that is the exact bug being worked around.

    uv run python src/merge_lora.py

The adapter itself is backed up as the Kaggle dataset
qwrfsdr/rsna-knee-lora-adapter. artifacts/lora/ is gitignored - the ignore
rule that keeps the 17.9GB merged model out of the repository also excludes
the 223MB adapter - so without that copy, 62 minutes of GPU training would
exist in exactly one place on one disk. The merged model needs no such care:
this script rebuilds it from the adapter on CPU in about ten minutes, which
matters because the local GPU is not currently ours to use.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen3.5-9B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=ROOT / "artifacts" / "lora" / "best")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "merged")
    a = ap.parse_args()

    print(f"loading {BASE}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cpu"
    )

    # snapshot one weight the adapter targets, to prove the merge did something
    probe_name = next(
        n for n, _ in model.named_parameters() if n.endswith("layers.0.mlp.down_proj.weight")
    )
    before = dict(model.named_parameters())[probe_name].detach().clone()

    print(f"applying adapter {a.adapter}")
    model = PeftModel.from_pretrained(model, str(a.adapter))
    model = model.merge_and_unload()

    after = dict(model.named_parameters())[probe_name].detach()
    delta = (after.float() - before.float()).abs().max().item()
    print(f"max weight change on {probe_name}: {delta:.6f}")
    if delta == 0.0:
        raise SystemExit(
            "MERGE WAS A NO-OP - the adapter did not bind to the base model.\n"
            "Saving would produce a copy of the base model wearing the student's\n"
            "name, which is worse than failing. Check the adapter key prefixes\n"
            "against the model's module names."
        )

    if a.out.exists():
        shutil.rmtree(a.out)
    print(f"saving to {a.out}")
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(a.out)
    size = sum(f.stat().st_size for f in a.out.rglob("*")) / 1e9
    print(f"done: {size:.1f} GB")
    print(
        f"\nserve it as an ordinary model:\n"
        f"  docker run --rm --gpus all -p 8000:8000 \\\n"
        f"    -v {a.out}:/merged vllm/vllm-openai:latest \\\n"
        f"    --model /merged --served-model-name knee --max-model-len 4096"
    )


if __name__ == "__main__":
    main()
