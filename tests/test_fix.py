"""Tests for the dormant Fixer harness (no API, no sby): CTI extraction
from failure evidence, context resolution, and the cheap pre-gates.
The harness is NOT plugged into any loop — see fix.py's docstring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from fix import (failing_cti, fixer_context, pdr_says_true,   # noqa: E402
                 pending_tasks, pre_gate, repaired_record)

CTI = "Trace summary (trace_induct.vcd):\n  At start state (step 0):\n    x = 1"

RECORD = {
    "extension_id": "e9", "ext_type": "structural", "parent_id": "g0_014",
    "verdict": "NOT_PROVEN",
    "child_verilog": "module token_bucket; always @(*) assert (a); endmodule",
    "invariants": ["a == b"],
    "result": {
        "with_invariants": {"runs": [
            {"mode": "prove", "rc": 4, "engine": "smtbmc yices",
             "notes": [], "trace_text": CTI}]},
        "without_invariants": None,
    },
}

PARENT = {"id": "g0_014", "top_module": "token_bucket", "clock": "clk",
          "antecedents": [], "sanity_covers": ["a"],
          "verilog": "module token_bucket; endmodule",
          "invariants": ["a == b"]}


def test_failing_cti_prefers_with_run():
    text, leg = failing_cti(RECORD)
    assert text == CTI
    assert leg == "with_invariants"


def test_failing_cti_none_when_no_trace():
    rec = {"result": {"with_invariants": {"runs": [
        {"mode": "prove", "rc": 1, "notes": [], "trace_text": None}]}}}
    text, leg = failing_cti(rec)
    assert text is None


def test_fixer_context_resolves_parent_fields():
    ctx = fixer_context(RECORD, [PARENT])
    assert ctx["top_module"] == "token_bucket"
    assert ctx["clock"] == "clk"
    assert ctx["sanity_covers"] == ["a"]
    assert ctx["verilog"] == RECORD["child_verilog"]
    assert ctx["property"] == ["a"]          # extracted from the child
    assert ctx["invariants"] == ["a == b"]   # the failed list


def test_pre_gate_rejects_unchanged_list():
    # resubmitting the failed list would waste a grading run
    assert pre_gate(["a == b"], ["a  ==  b"], ["p"]) is not None


def test_pre_gate_rejects_property_copy():
    assert pre_gate(["p == 1"], ["a == b"], ["p == 1"]) is not None


def test_pre_gate_accepts_new_list():
    assert pre_gate(["a == b", "c <= 2"], ["a == b"], ["p"]) is None


def test_failing_cti_ignores_cover_traces():
    # the induction CTI lives on a prove run; a cover run's trace is a
    # reachability witness and must never be presented as the CTI
    rec = {"result": {"with_invariants": {"runs": [
        {"mode": "cover", "trace_text": "cover witness"},
        {"mode": "prove", "trace_text": CTI}]}}}
    text, leg = failing_cti(rec)
    assert text == CTI


def test_pdr_says_true_reads_second_opinion_note():
    rec = {"result": {"with_invariants": {"runs": [
        {"mode": "prove", "engine": "abc pdr",
         "notes": ["pdr_second_opinion: property proven true (unbounded) "
                   "by abc pdr — induction failure is a strengthening "
                   "problem"]}]}}}
    assert pdr_says_true(rec)
    assert not pdr_says_true(RECORD)


def test_repaired_record_promotes_as_the_childs_generation():
    from promote import promote
    win_invs = ["a == b", "c <= 2"]
    fixed = repaired_record(RECORD, win_invs, {"fake": "result"})
    assert fixed["verdict"] == "NECESSARY"
    assert fixed["invariants"] == win_invs
    assert fixed["extension_id"] == "e9_fix"
    corpus = [dict(PARENT, generation=0, metrics={})]
    buckets, rows = promote(corpus, [fixed], compute_metrics=False)
    assert len(rows) == 1 and rows[0]["generation"] == 1
    assert rows[0]["invariants"] == win_invs


def test_pending_tasks_skips_already_attempted():
    queue = [{"extension_id": "a"}, {"extension_id": "b"},
             {"extension_id": "a"}]                      # duplicate line
    out = pending_tasks(queue, attempted={"b"})
    assert [t["extension_id"] for t in out] == ["a"]
