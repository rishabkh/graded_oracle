"""Bake a trained adapter into its base model and save the result.

Training saves the adapter (the trained add-on, about 1 GB) and then a
merged copy (base plus adapter, about 65 GB for the 32B model) so vLLM
can serve it exactly as it served the first fine-tune. On 1 Oct 2026
that merged save ran into the 100 GB home quota, so the run was stopped
once the adapter had been written. This does the merge on its own.

No GPU is needed: merging adds each low-rank update into its weight
matrix, which is arithmetic, not learning. Shards are kept small so a
CPU node never holds a 50 GB shard twice while writing it. The output
must not be in the home folder; scratch is the place for 65 GB.

  python training/merge_adapter.py \
      --adapter runs/v2/adapter \
      --model Qwen/Qwen2.5-Coder-32B-Instruct \
      --out /n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/runs/v2/merged
"""
import argparse
import os
import sys
from pathlib import Path


def check_paths(adapter, out):
    """Refuse, before any large load, an adapter folder that is not one
    and an output inside the home folder."""
    adapter, out = Path(adapter), Path(out)
    if not (adapter / "adapter_config.json").is_file():
        sys.exit(f"{adapter} has no adapter_config.json - point --adapter "
                 "at the folder training saved the adapter into")
    home = Path(os.path.expanduser("~")).resolve()
    target = out.resolve()
    if target == home or home in target.parents:
        sys.exit(f"{out} is inside the home folder ({home}); a merged 32B "
                 "model is about 65 GB against a 100 GB quota. Write it to "
                 "scratch instead.")


def merge(adapter, model, out):
    """Load the base model on CPU in bf16, apply the adapter, merge, save."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    check_paths(args.adapter, args.out)
    out = merge(args.adapter, args.model, args.out)
    print(f"saved merged model -> {out}")


if __name__ == "__main__":
    main()
