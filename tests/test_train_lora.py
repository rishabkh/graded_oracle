"""The trainer's data side (8 Oct 2026): chat records keep their system
message in the question, and dropped long examples stop the run instead of
passing as a printed count. No torch: a fake tokenizer counts characters."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import train_lora as tl                                           # noqa: E402


class FakeTok:
    eos_token = "|"

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_a_chat_record_keeps_its_system_message_in_the_question():
    pair = {"prompt": [{"role": "system", "content": "SYS"},
                       {"role": "user", "content": "Q"}],
            "completion": "ANS", "kind": "generate"}
    [row], dropped = tl.build_rows([pair], FakeTok(), 1000)
    text = "".join(map(chr, row["input_ids"]))
    assert text == "<system>SYS<user>Q<assistant>ANS|"
    q_len = len("<system>SYS<user>Q<assistant>")
    assert row["labels"][:q_len] == [tl.IGNORE] * q_len
    assert dropped == {}


def test_a_solving_pair_is_asked_as_before():
    [row], _ = tl.build_rows([{"prompt": "Q", "completion": "A"}],
                             FakeTok(), 1000)
    assert "".join(map(chr, row["input_ids"])) == "<user>Q<assistant>A|"


def test_drops_are_counted_by_kind():
    pairs = [{"prompt": "Q", "completion": "A" * 50, "kind": "repair"},
             {"prompt": "Q", "completion": "A" * 50},
             {"prompt": "Q", "completion": "A"}]
    rows, dropped = tl.build_rows(pairs, FakeTok(), 30)
    assert len(rows) == 1 and dropped == {"repair": 1, "solve": 1}


def test_too_many_drops_stop_the_run():
    with pytest.raises(SystemExit) as e:
        tl.check_drops({"repair": 3}, kept=97, max_share=0.01)
    assert "repair" in str(e.value)
    tl.check_drops({"repair": 1}, kept=199, max_share=0.01)   # 0.5%: fine
    tl.check_drops({}, kept=10, max_share=0.0)


def test_the_length_limit_is_raised_for_bigger_designs():
    import inspect
    src = inspect.getsource(tl.main)
    assert '"--max-len", type=int, default=16384' in src
