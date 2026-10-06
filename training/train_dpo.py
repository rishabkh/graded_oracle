"""Preference training on top of a fine-tuned model: prefer the full
answer over the same answer missing one needed fact.

Run 4b. The pairs (training/build_sft.py --preference) come from answers
that pass the benchmark's 1-step rule: `chosen` is the full answer,
`rejected` drops one fact the 1-step proof needs, so it is still true and
too weak, the exact failure v1 and v2 show.

Loss per pair: DPO (Rafailov et al., 2023), -log sigmoid(beta * margin),
margin = (log p(chosen) - log p_ref(chosen)) - (log p(rejected) -
log p_ref(rejected)); plus the ordinary loss on the chosen answer
(sft_weight * -log p(chosen) per token). The extra term guards against
DPO lowering the chosen answer's probability along with the rejected
one, which is most likely when the two differ by one fact (likelihood
displacement, Razin et al., 2024).

The reference is the starting model itself: LoRA is added on top and the
reference log-probabilities are read with the adapter switched off, once,
before training. Same LoRA shape and input building as train_lora.py, so
4b differs from 4a only in the objective. Written directly rather than
through a preference library, so the cluster environment that trained
v1 to v3 is unchanged.

  python training/train_dpo.py --pairs extender/pref_v4b.jsonl \\
      --model <4a merged> --out <scratch>/runs/v4b --merge
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sft_data import IGNORE                                  # noqa: E402
from train_lora import build_rows                            # noqa: E402


def load_pairs(path):
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        p = json.loads(line)
        if not (p.get("prompt") and p.get("chosen") and p.get("rejected")):
            raise ValueError(f"{path} line {n}: prompt, chosen and rejected "
                             "must all be non-empty")
        out.append(p)
    return out


def split_pairs(pairs):
    """v2's own held-back rows (each pair carries the mark) are never
    trained on, as in runs 3 and 4a."""
    return ([p for p in pairs if not p.get("holdout")],
            [p for p in pairs if p.get("holdout")])


def encode(pair, tok, max_len):
    """Both answers built exactly as train_lora builds a row. None when
    either side is over the length budget."""
    rows, _ = build_rows([{"prompt": pair["prompt"],
                           "completion": pair["chosen"]},
                          {"prompt": pair["prompt"],
                           "completion": pair["rejected"]}], tok, max_len)
    if len(rows) != 2:
        return None
    return {"chosen": rows[0], "rejected": rows[1]}


def sequence_logprob(logits, labels):
    """Sum of log-probabilities of the answer tokens (labels not IGNORE),
    each predicted from the position before it. Returns (sum, count)."""
    import torch
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    mask = labels != IGNORE
    safe = labels.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits, dim=-1).gather(
        -1, safe.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(), int(mask.sum())


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    import torch
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return -torch.nn.functional.logsigmoid(beta * margin)


def _logp(model, row, device):
    import torch
    ids = torch.tensor([row["input_ids"]], device=device)
    labels = torch.tensor([row["labels"]], device=device)
    logits = model(input_ids=ids,
                   attention_mask=torch.ones_like(ids)).logits
    return sequence_logprob(logits, labels)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--model", required=True,
                   help="the fine-tuned starting model (also the reference)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--cpu", action="store_true",
                   help="float32 on the CPU: the tiny local test only")
    args = p.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pairs = load_pairs(args.pairs)
    train, held = split_pairs(pairs)
    enc = [e for e in (encode(q, tok, args.max_len) for q in train) if e]
    print(f"{len(pairs)} pairs -> {len(train)} train, {len(held)} held back, "
          f"{len(train) - len(enc)} over {args.max_len} tokens and dropped",
          flush=True)
    if not enc:
        sys.exit("no training pairs survived the length budget")

    kw = ({"dtype": torch.float32} if args.cpu
          else {"dtype": torch.bfloat16, "device_map": "auto"})
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
    device = next(model.parameters()).device

    # the reference: the starting model, read once with the adapter off
    model.eval()
    ref = []
    with torch.no_grad(), model.disable_adapter():
        for e in enc:
            ref.append((_logp(model, e["chosen"], device)[0].item(),
                        _logp(model, e["rejected"], device)[0].item()))
    model.train()

    total = args.steps if args.steps > 0 else \
        max(1, math.ceil(len(enc) * args.epochs / args.grad_accum))
    warmup = max(1, int(total * 0.02))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup,
                           max(0.0, (total - s) / max(1, total - warmup))))

    log, step, micro, t0 = [], 0, 0, time.monotonic()
    acc = {"loss": 0.0, "dpo": 0.0, "nll": 0.0, "margin": 0.0, "right": 0}
    while step < total:
        e, (rc, rr) = enc[micro % len(enc)], ref[micro % len(enc)]
        pc, nc = _logp(model, e["chosen"], device)
        pr, _ = _logp(model, e["rejected"], device)
        d = dpo_loss(pc, pr, torch.tensor(rc, device=device),
                     torch.tensor(rr, device=device), args.beta)
        nll = -pc / max(nc, 1)
        loss = d + args.sft_weight * nll
        (loss / args.grad_accum).backward()
        margin = (pc.item() - rc) - (pr.item() - rr)
        acc["loss"] += loss.item(); acc["dpo"] += d.item()
        acc["nll"] += nll.item(); acc["margin"] += margin
        acc["right"] += margin > 0
        micro += 1
        if micro % args.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(
            [q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        rec = {k: v / args.grad_accum for k, v in acc.items()}
        rec["step"] = step
        if step <= 2:              # the reference must not move with training
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                now = _logp(model, enc[0]["chosen"], device)[0].item()
            model.train()
            rec["ref_unchanged"] = abs(now - ref[0][0]) < 1e-3 * max(1, abs(now))
        log.append(rec)
        print(f"step {step}/{total} loss {rec['loss']:.4f} dpo {rec['dpo']:.4f} "
              f"nll {rec['nll']:.4f} margin {rec['margin']:.3f} "
              f"chosen-preferred {rec['right']:.2f} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        acc = {k: 0.0 for k in acc}

    out = Path(args.out)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged")
        tok.save_pretrained(out / "merged")
        print(f"saved merged model -> {out / 'merged'}")
    (out / "run.json").write_text(json.dumps(
        {"pairs": args.pairs, "model": args.model, "steps": total,
         "epochs": args.epochs, "beta": args.beta,
         "sft_weight": args.sft_weight, "lr": args.lr, "rank": args.rank,
         "train_pairs": len(enc), "held_back_pairs": len(held),
         "final": log[-1] if log else None}, indent=2))
    print(f"saved adapter -> {out / 'adapter'}")
    return log


if __name__ == "__main__":
    main()
