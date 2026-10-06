"""Preference training (DPO plus the ordinary loss on the chosen answer).
Needs torch; skips without it. The tiny end-to-end run also needs a local
Qwen tokenizer folder named by DPO_TOKENIZER."""
import json
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import train_dpo as td                                           # noqa: E402

TOKENIZER = os.environ.get("DPO_TOKENIZER")


def test_dpo_loss_on_numbers_worked_by_hand():
    # policy prefers chosen by 1 nat more than the reference does, and
    # dislikes rejected by 1 nat more: margin 2, beta 0.1 -> -log sigmoid(0.2)
    loss = td.dpo_loss(torch.tensor(-10.0), torch.tensor(-12.0),
                       torch.tensor(-11.0), torch.tensor(-11.0), beta=0.1)
    assert math.isclose(loss.item(), math.log(1 + math.exp(-0.2)), rel_tol=1e-6)


def test_dpo_loss_is_log2_when_the_policy_is_the_reference():
    x, y = torch.tensor(-5.0), torch.tensor(-7.0)
    assert math.isclose(td.dpo_loss(x, y, x, y, beta=0.1).item(), math.log(2),
                        rel_tol=1e-6)


def test_sequence_logprob_sums_only_the_answer_tokens():
    # vocab 3; logits favour token 2 then token 1; labels hide position 0
    logits = torch.log(torch.tensor([[[0.1, 0.1, 0.8],
                                      [0.1, 0.7, 0.2],
                                      [0.3, 0.3, 0.4]]]))
    labels = torch.tensor([[-100, 2, 1]])
    total, n = td.sequence_logprob(logits, labels)
    # next-token: logits[0] predicts labels[1]=2 (0.8), logits[1] predicts labels[2]=1 (0.7)
    assert math.isclose(total.item(), math.log(0.8) + math.log(0.7), rel_tol=1e-5)
    assert n == 2


def test_pairs_split_by_v2s_held_back_marks():
    pairs = [{"prompt": "q", "chosen": "a", "rejected": "b", "holdout": h}
             for h in (False, True, False)]
    train, held = td.split_pairs(pairs)
    assert len(train) == 2 and len(held) == 1


@pytest.mark.skipif(not TOKENIZER, reason="set DPO_TOKENIZER to a Qwen tokenizer folder")
def test_inputs_are_built_like_v2s_training_rows():
    from transformers import AutoTokenizer
    import train_lora
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    pair = {"prompt": "Design here", "chosen": '{"invariants": ["a == b"]}',
            "rejected": '{"invariants": []}', "holdout": False}
    enc = td.encode(pair, tok, max_len=8192)
    ref, _ = train_lora.build_rows([{"prompt": "Design here",
                                     "completion": pair["chosen"]}], tok, 8192)
    assert enc["chosen"]["input_ids"] == ref[0]["input_ids"]
    assert enc["chosen"]["labels"] == ref[0]["labels"]
    # same question, so the hidden part is the same length in both
    q = sum(1 for x in enc["chosen"]["labels"] if x == -100)
    assert q == sum(1 for x in enc["rejected"]["labels"] if x == -100)


@pytest.mark.skipif(not TOKENIZER, reason="set DPO_TOKENIZER to a Qwen tokenizer folder")
def test_a_tiny_model_trains_and_the_reference_never_moves(tmp_path):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    base = tmp_path / "base"
    Qwen2ForCausalLM(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(p) + "\n" for p in [
        {"prompt": "x == 1?", "chosen": '{"invariants": ["x <= 1"]}',
         "rejected": '{"invariants": []}', "holdout": False},
        {"prompt": "y == 2?", "chosen": '{"invariants": ["y <= 2", "y >= 0"]}',
         "rejected": '{"invariants": ["y <= 2"]}', "holdout": False},
        {"prompt": "z?", "chosen": '{"invariants": ["z"]}',
         "rejected": '{"invariants": []}', "holdout": True}]))
    out = tmp_path / "out"
    log = td.main(["--pairs", str(pairs), "--model", str(base), "--out",
                   str(out), "--steps", "2", "--grad-accum", "1",
                   "--cpu"])
    assert (out / "adapter" / "adapter_config.json").exists()
    assert len(log) == 2 and all(math.isfinite(r["loss"]) for r in log)
    assert all(r["ref_unchanged"] for r in log)
    run = json.loads((out / "run.json").read_text())
    assert run["train_pairs"] == 2 and run["held_back_pairs"] == 1
