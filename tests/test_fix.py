"""Tests for the dormant Fixer harness (no API, no sby): CTI extraction
from failure evidence, context resolution, and the cheap pre-gates.
The harness is NOT plugged into any loop — see fix.py's docstring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from fix import failing_cti, fixer_context, pre_gate   # noqa: E402

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
