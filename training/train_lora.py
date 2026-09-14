"""LoRA fine-tune on the corpus pairs. Smoke test and real run are the
same script: the smoke test just points at a small model and stops after
a few steps, which is what proves the plumbing before GPU hours go in.

  # smoke: small model, 20 steps, one GPU, minutes
  python training/train_lora.py --pairs extender/sft_smoke.jsonl \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --out runs/smoke --steps 20

  # real: the model Formal Disco fine-tunes, the whole file
  python training/train_lora.py --pairs extender/sft_train.jsonl \
      --model Qwen/Qwen2.5-Coder-32B-Instruct --out runs/v1 --epochs 3

Answers only, per Nada: the loss counts the invariant list and not the
design that was handed to the model.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sft_data import IGNORE, load_pairs, mask_labels, split, to_messages


def build_rows(pairs, tok, max_len):
    """One tokenised row per pair, with the question masked out. Rows
    over the length budget are dropped and counted, never truncated: a
    half design with a full answer is a lie to train on."""
    rows, dropped = [], 0
    for pair in pairs:
        msgs = to_messages(pair)
        question = tok.apply_chat_template(msgs[:1], tokenize=False,
                                           add_generation_prompt=True)
        full = question + pair["completion"] + tok.eos_token
        ids = tok(full, add_special_tokens=False)["input_ids"]
        if len(ids) > max_len:
            dropped += 1
            continue
        q_len = len(tok(question, add_special_tokens=False)["input_ids"])
        rows.append({"input_ids": ids,
                     "labels": mask_labels(ids, q_len),
                     "attention_mask": [1] * len(ids)})
    return rows, dropped


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    import torch

    def pad(key, fill):
        return torch.tensor([b[key] + [fill] * (width - len(b[key]))
                             for b in batch])
    return {"input_ids": pad("input_ids", pad_id),
            "labels": pad("labels", IGNORE),
            "attention_mask": pad("attention_mask", 0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=-1,
                   help="stop after N steps; -1 runs --epochs instead")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    # defaults copied from formal-disco/config/distill.yaml so our run and
    # the software side differ in the data, not in the training setup
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--merge", action="store_true",
                   help="also save base+adapter merged into one model: "
                        "vLLM serves that, and the scoring step needs it")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-4bit", action="store_true",
                   help="quantised base weights; needed for 32B on one "
                        "80GB card, not needed for the smoke test")
    args = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pairs = load_pairs(args.pairs)
    train_pairs, eval_pairs = split(pairs, args.holdout, args.seed)
    train_rows, dropped = build_rows(train_pairs, tok, args.max_len)
    eval_rows, dropped_eval = build_rows(eval_pairs, tok, args.max_len)
    print(f"{len(pairs)} pairs -> {len(train_rows)} train, "
          f"{len(eval_rows)} eval, {dropped + dropped_eval} over "
          f"{args.max_len} tokens and dropped")
    if not train_rows:
        sys.exit("no training rows survived the length budget")

    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # transformers 5 dropped warmup_ratio, so work out the same 2% of the
    # run that formal-disco uses and pass it as a step count
    per_epoch = max(1, len(train_rows) // (args.batch * args.grad_accum))
    total = args.steps if args.steps > 0 else int(per_epoch * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="linear",
        warmup_steps=max(1, int(total * 0.02)), max_grad_norm=1.0,
        num_train_epochs=args.epochs, max_steps=args.steps,
        logging_steps=1, save_strategy="no", bf16=True, seed=args.seed,
        report_to=[], eval_strategy="no")
    trainer = Trainer(model=model, args=targs, train_dataset=train_rows,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    result = trainer.train()

    out = Path(args.out)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged")
        tok.save_pretrained(out / "merged")
        print(f"saved merged model -> {out / 'merged'}")
    (out / "run.json").write_text(json.dumps(
        {"pairs": args.pairs, "model": args.model, "steps": args.steps,
         "epochs": args.epochs, "rank": args.rank, "lr": args.lr,
         "train_rows": len(train_rows), "dropped": dropped + dropped_eval,
         "train_loss": result.training_loss}, indent=2))
    print(f"\nsaved adapter -> {out / 'adapter'}  "
          f"(final loss {result.training_loss:.4f})")


if __name__ == "__main__":
    main()
