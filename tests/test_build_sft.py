"""Turning the corpus into training pairs.

The rule that matters: the training question must be byte-identical in
shape to the evaluation question, or we train the model on one task and
score it on another."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from build_sft import build_pair, build_pairs, repair_pairs

ROW = {
    "id": "g0_000",
    "generation": 0,
    "top_module": "m",
    "verilog": "module m (input wire clk);\n  reg [3:0] c;\nendmodule\n",
    "property": ["c <= 4'd9"],
    "invariants": ["c <= 4'd9", "c != 4'd15"],
}


def test_the_training_question_is_the_evaluation_question():
    from solver_baseline import SOLVER_PROMPT
    pair = build_pair(ROW)
    assert pair["prompt"] == SOLVER_PROMPT.format(verilog=ROW["verilog"])


def test_the_answer_is_the_json_the_grader_parses():
    pair = build_pair(ROW)
    assert json.loads(pair["completion"]) == {"invariants": ROW["invariants"]}


def test_a_row_with_no_invariants_is_refused():
    with pytest.raises(ValueError):
        build_pair({**ROW, "invariants": []})


def test_identical_designs_are_not_trained_on_twice():
    pairs = build_pairs([ROW, dict(ROW, id="g0_001")])
    assert len(pairs) == 1


def test_pairs_carry_the_row_they_came_from():
    pairs = build_pairs([ROW])
    assert pairs[0]["id"] == "g0_000" and pairs[0]["generation"] == 0


def test_repairs_pair_the_failed_attempt_with_the_fix():
    """Run 2 data: what the model first said, why it was wrong, and what
    finally closed the proof."""
    attempts = [
        {"task_id": 7, "attempt": 1, "verdict": "NOT_PROVEN",
         "invariants": ["c <= 4'd15"], "diagnosis": "too weak",
         "cti_leg": "c = 4'd12"},
        {"task_id": 7, "attempt": 2, "verdict": "NECESSARY",
         "invariants": ["c <= 4'd9"]},
    ]
    out = repair_pairs(attempts, {7: ROW})
    assert len(out) == 1
    assert "c <= 4'd15" in out[0]["prompt"]        # the failed attempt
    assert "c = 4'd12" in out[0]["prompt"]         # the counterexample
    assert json.loads(out[0]["completion"]) == {"invariants": ["c <= 4'd9"]}


def test_a_task_that_never_got_fixed_yields_nothing():
    attempts = [{"task_id": 9, "attempt": 1, "verdict": "NOT_PROVEN",
                 "invariants": ["x"], "diagnosis": "d", "cti_leg": "c"}]
    assert repair_pairs(attempts, {9: ROW}) == []
