"""Load the saved adapter back and ask it one question. This is the
half of the smoke test that matters: training can finish and still leave
you with weights nothing can load.

  python training/check_adapter.py --adapter runs/smoke/adapter \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --pairs extender/sft_smoke.jsonl
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sft_data import load_pairs, to_messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--max-new", type=int, default=256)
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    pair = load_pairs(args.pairs)[args.index]
    text = tok.apply_chat_template(to_messages(pair)[:1], tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new,
                             do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    answer = tok.decode(out[0][ids["input_ids"].shape[1]:],
                        skip_special_tokens=True)
    print("=== model answer ===")
    print(answer.strip())
    print("\n=== the checked answer for this design ===")
    print(pair["completion"].strip())


if __name__ == "__main__":
    main()
