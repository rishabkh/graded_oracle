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
    src = inspect.getsource(tl.parser)
    assert '"--max-len", type=int, default=16384' in src


# --- counting each pair equally (8 Nov 2026) ---------------------------------
# The storage set's answers were 24% of its pairs but 79% of its answer text,
# and the library averages the loss over every answer word in a step (8 pairs
# added up), so those answers took most of what the model learned from; the
# easy score fell from 13.56 to 8.17 per try.

TRAINING = Path(__file__).resolve().parent.parent / "training"


class FakeTrainer:
    model_accepts_loss_kwargs = True


def test_counting_each_pair_equally_turns_off_the_word_count():
    trainer = FakeTrainer()
    tl.count_each_pair_equally(trainer)
    assert trainer.model_accepts_loss_kwargs is False


def test_a_library_without_that_switch_stops_before_training():
    class OldTrainer:
        pass
    with pytest.raises(SystemExit) as e:
        tl.count_each_pair_equally(OldTrainer())
    assert "nothing was trained" in str(e.value)


def test_every_word_counts_unless_asked():
    need = ["--pairs", "p", "--model", "m", "--out", "o"]
    assert tl.parser().parse_args(need).count_each == "word"
    assert tl.parser().parse_args(need + ["--count-each", "pair"]).count_each == "pair"
    with pytest.raises(SystemExit):
        tl.parser().parse_args(need + ["--count-each", "token"])


def test_the_choice_is_used_and_saved_with_the_model():
    import inspect
    src = inspect.getsource(tl.main)
    assert 'if args.count_each == "pair":' in src
    assert "count_each_pair_equally(trainer)" in src
    assert '"count_each": args.count_each' in src


def test_the_training_job_passes_the_choice_on():
    text = (TRAINING / "train32b.sbatch").read_text()
    assert '--count-each "${COUNT_EACH:-word}"' in text
