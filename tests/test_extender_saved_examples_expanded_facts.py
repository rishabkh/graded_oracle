"""Saved extension answers carry the facts the checker got, expanded
(22 Oct 2026, review finding F2, extender part).

An extension answer is saved exactly as the model wrote it, and it may
hold compact facts (`forall k in 0..3: hist[k] <= 4'd9`), which the
benchmark's checker, EBMC 6.0, cannot read. The example's metadata now
also carries "invariants_expanded": the facts as our checker got them,
every compact fact expanded, so the training builder can put the plain
list in the answer. The model and the checker are faked: no calls, no
proofs."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import distill_log                                              # noqa: E402
import extend                                                   # noqa: E402
from fact_format import all_plain                               # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict, Tier,  # noqa: E402
                          TripleResult)

USAGE = {"input": 1, "output": 2}

CTR = """module ctr (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg  [3:0] cnt
);
    reg [3:0] lim;
    initial cnt = 4'd0;
    initial lim = 4'd9;
    always @(posedge clk) begin
        if (rst)
            cnt <= 4'd0;
        else if (en)
            cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1;
    end
    always @(posedge clk) assert (cnt <= 4'd9);
endmodule
"""
PARENT = {"id": "c0_7", "generation": 0, "parent": None, "parent2": None,
          "top_module": "ctr", "clock": "clk", "antecedents": [],
          "sanity_covers": [], "verilog": CTR, "property": ["cnt <= 4'd9"],
          "invariants": ["lim == 4'd9"], "metrics": {}}

HISTORY = """\
@@ reg [3:0] lim; @@
+    reg [3:0] hist [0:3];
+    integer j;
+    initial for (j = 0; j <= 3; j = j + 1) hist[j] = 4'd0;
@@ cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1; @@
= end
+    always @(posedge clk)
+        if (en) begin
+            for (j = 3; j > 0; j = j - 1) hist[j] <= hist[j - 1];
+            hist[0] <= cnt;
+        end"""

COMPACT = ["lim == 4'd9", "forall k in 0..3: hist[k] <= 4'd9"]
EXPANDED = ["lim == 4'd9", "hist[0] <= 4'd9", "hist[1] <= 4'd9",
            "hist[2] <= 4'd9", "hist[3] <= 4'd9"]
ANSWER = {"new_state": "a history of the last four counts",
          "coupling": "c", "induction_gap": "g",
          "why_parent_insufficient": "w", "patch": HISTORY,
          "invariants": COMPACT,
          "dispositions": [{"clause": "lim == 4'd9", "status": "kept",
                            "replaced_by": [], "reason": ""}]}


class Model:
    def __init__(self, *answers):
        self.answers = list(answers)

    def __call__(self, prompt, schema=None):
        answer = self.answers.pop(0)
        if isinstance(answer, tuple):
            return answer[0], USAGE, answer[1]
        return answer, USAGE, "ok"


@pytest.fixture
def saved(monkeypatch, tmp_path):
    path = tmp_path / "distill_extensions.jsonl"
    monkeypatch.setattr(batch, "DISTILL_LOG", path)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    return lambda: distill_log.read(path) if path.exists() else []


@pytest.fixture
def checker(monkeypatch):
    """The real step check, with the proofs faked: NECESSARY, and the new
    state needs facts."""
    monkeypatch.setattr(extend, "state_bits", lambda v, top: len(v))
    monkeypatch.setattr(
        extend, "grade_triple_generated",
        lambda payload, **kw: TripleResult(
            verdict=NecessityVerdict.NECESSARY,
            reason="strengthening is load-bearing"))
    monkeypatch.setattr(extend, "grade",
                        lambda *a, **k: GradeResult(tier=Tier.NOT_INDUCTIVE,
                                                    reason="alone"))


def task(kind="structural", **kw):
    t = {"task_id": 1, "attempt": 1, "ext_type": kind, "parent_id": "c0_7",
         "parent2_id": None, "move": "STORAGE" if kind == "structural"
         else None, "k": 1}
    t.update(kw)
    return t


def test_a_proved_answer_with_compact_facts_carries_them_expanded(
        saved, checker, monkeypatch):
    monkeypatch.setattr(batch, "call_model", Model(ANSWER))
    rec = batch._real_executor(task(), [PARENT])
    assert rec["verdict"] == "NECESSARY", rec.get("reason")
    [ex] = saved()
    assert ex["outcome"] == "success"
    # the answer as the model wrote it, compact fact and all ...
    assert json.loads(ex["response"])["invariants"] == COMPACT
    assert not all_plain(COMPACT)
    # ... and the facts the checker proved, which EBMC can read
    assert ex["metadata"]["invariants_expanded"] == EXPANDED
    assert all_plain(ex["metadata"]["invariants_expanded"])


def test_a_plain_answer_carries_its_own_facts(saved, checker, monkeypatch):
    plain = dict(ANSWER, invariants=EXPANDED)
    monkeypatch.setattr(batch, "call_model", Model(plain))
    batch._real_executor(task(), [PARENT])
    [ex] = saved()
    assert ex["metadata"]["invariants_expanded"] == EXPANDED


def test_a_refused_answer_has_no_facts(saved, monkeypatch):
    monkeypatch.setattr(batch, "call_model", Model((None, "refusal")))
    batch._real_executor(task(), [PARENT])
    [ex] = saved()
    assert ex["outcome"] == "error"
    assert ex["metadata"]["invariants_expanded"] is None


def test_a_debug_blocks_facts_are_its_parents(saved, monkeypatch):
    """A debug block keeps its parent's facts verbatim; they are what the
    checker proved for the grown design."""
    monkeypatch.setattr(batch, "distractor_call",
                        Model({"note": "a status counter",
                               "patch": "@@ endmodule @@\n+ // debug"}))

    def check(parent, patch, record):
        record.update(verdict="NOT_PROVEN", reason="r")
        return record
    monkeypatch.setattr(batch, "grade_extension", check)
    batch._real_executor(task("distractor"), [PARENT])
    [ex] = saved()
    assert ex["metadata"]["invariants_expanded"] == ["lim == 4'd9"]


@pytest.mark.parametrize("record, expected", [
    ({"invariants": EXPANDED}, EXPANDED),
    ({"invariants": []}, []),
    ({}, None),                                   # no answer, no facts
    ({"invariants": None}, None),
    # a fact that could not be expanded: the record holds them as written
    ({"invariants": ["forall k in 0..DEPTH-1: v[k]"]}, None),
    ({"invariants": COMPACT}, None),              # not expanded
    ({"invariants": ["a", 3]}, None),             # not a fact list
])
def test_the_expanded_facts_of_a_record(record, expected):
    assert batch.expanded_facts(record) == expected


def test_a_fact_turned_away_after_expansion_is_still_expanded(saved,
                                                              monkeypatch):
    """A fact with a system function is turned away (FACT_FORMAT) after
    the compact facts are expanded: the metadata holds the expanded list
    the check read, and the example is a failure."""
    answer = dict(ANSWER, invariants=["lim == 4'd9",
                                      "forall k in 0..1: hist[k] == "
                                      "$past(cnt)"])
    monkeypatch.setattr(batch, "call_model", Model(answer))
    rec = batch._real_executor(task(), [PARENT])
    assert rec["verdict"] == "FACT_FORMAT"
    [ex] = saved()
    assert ex["outcome"] == "fail"
    assert ex["metadata"]["invariants_expanded"] == [
        "lim == 4'd9", "hist[0] == $past(cnt)", "hist[1] == $past(cnt)"]
