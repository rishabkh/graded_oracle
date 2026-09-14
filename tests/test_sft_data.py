"""Data side of the fine-tune: reading the pairs, turning them into
chat turns, splitting, and masking the question so the loss only counts
the answer. Pure python on purpose - it runs here, without a GPU."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from sft_data import load_pairs, mask_labels, split, to_messages


def write(tmp_path, rows):
    p = tmp_path / "pairs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_load_pairs_reads_prompt_and_completion(tmp_path):
    p = write(tmp_path, [{"prompt": "q", "completion": "a"}])
    assert load_pairs(p) == [{"prompt": "q", "completion": "a"}]


def test_load_pairs_rejects_a_row_missing_the_answer(tmp_path):
    p = write(tmp_path, [{"prompt": "q", "completion": ""}])
    with pytest.raises(ValueError):
        load_pairs(p)


def test_to_messages_is_one_user_turn_and_one_answer():
    assert to_messages({"prompt": "q", "completion": "a"}) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_split_is_deterministic_and_keeps_every_row():
    rows = [{"prompt": f"q{i}", "completion": "a"} for i in range(10)]
    tr, ev = split(rows, holdout=0.2, seed=0)
    assert len(ev) == 2 and len(tr) == 8
    assert sorted(r["prompt"] for r in tr + ev) == sorted(
        r["prompt"] for r in rows)
    assert split(rows, holdout=0.2, seed=0) == (tr, ev)


def test_mask_labels_hides_the_question_from_the_loss():
    assert mask_labels([5, 6, 7, 8], prompt_len=2) == [-100, -100, 7, 8]


def test_mask_labels_refuses_a_prompt_longer_than_the_row():
    with pytest.raises(ValueError):
        mask_labels([5, 6], prompt_len=3)
