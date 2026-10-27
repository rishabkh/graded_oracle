"""Real open-source modules as starting designs (28 Oct 2026, Rishab chose
option 1: real designs only as starting points; our pipeline writes every
property and helper fact).

Tested without tools: the seeds carry none of the module's own formal
checks, the question shows the module, the edits go where they should and a
parameter is set without touching a localparam, and an answer runs through
check, repair and the corpus with a stand-in for the model. The logic check
and the grading run yosys and sby and skip without hwtools."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))
sys.path.insert(0, str(ROOT / "extender"))

import open_source_seeds                                       # noqa: E402
import seed_designs as sd                                      # noqa: E402

MOD = """module top #(
    parameter W = 8,
    parameter DEPTH = (1 << 4), // sixteen
    parameter [0:0] OPT = 1'b1
) (input wire clk, input wire push, output reg [W-1:0] cnt);
    localparam LAST = DEPTH - 1;
    initial cnt = 0;
    always @(posedge clk) if (push && cnt != LAST) cnt <= cnt + 1;
endmodule
module helper #(parameter N = 32) (input wire a, output wire b);
    assign b = a;
endmodule
"""
SEED = {"name": "repo/top", "repo": "repo", "licence": "MIT", "commit": "abc",
        "files": ["rtl/top.v"], "top_module": "top", "verilog": MOD}
ANSWER = {"parameters": [{"module": "top", "name": "DEPTH", "value": "4"},
                         {"module": "helper", "name": "N", "value": "2"}],
          "initial_values": [], "property_code": "always @(posedge clk) assert (cnt <= 8'd3);",
          "clock": "clk", "invariants": [], "antecedents": [], "sanity_covers": [],
          "cti_state": [{"signal": "cnt", "value": "8'd3"}], "cti_explanation": "x"}


def test_formal_sections_are_cut_and_an_else_branch_kept():
    text = """module m;
`ifdef FORMAL
    always @(*) assert(x);
`else
    wire kept_else;
`endif
`ifdef SOMETHING
    wire kept_other;
`endif
`ifndef FORMAL
    wire kept_ifndef;
`endif
endmodule
"""
    out = open_source_seeds.strip_formal(text)
    assert "assert" not in out
    assert "kept_else" in out and "kept_other" in out and "kept_ifndef" in out
    assert "`ifdef SOMETHING" in out and out.count("`endif") == 1


def test_a_parameter_default_is_set_and_a_localparam_is_refused():
    out = sd.set_param(MOD, "top", "DEPTH", "4")
    assert "parameter DEPTH = 4, // sixteen" in out
    assert "parameter W = 8," in out                     # the others untouched
    assert sd.set_param(MOD, "helper", "N", "2").count("parameter N = 2)") == 1
    with pytest.raises(sd.EditError):
        sd.set_param(MOD, "top", "LAST", "1")
    with pytest.raises(sd.EditError):
        sd.set_param(MOD, "top", "NOPE", "1")


def test_edits_go_at_the_end_of_their_module():
    base, full = sd.apply_edits(SEED, dict(ANSWER, initial_values=[
        {"module": "helper", "code": "initial begin end"}]))
    assert "assert" not in base
    top_part, helper_part = full.split("module helper")
    assert "assert (cnt <= 8'd3);" in top_part
    assert top_part.index("assert") < top_part.index("endmodule")
    assert "initial begin end" in helper_part
    with pytest.raises(sd.EditError):
        sd.apply_edits(SEED, dict(ANSWER, property_code="  "))


def test_question_shows_the_module_and_the_reply_asks_for_edits_only():
    q = sd.question(SEED)
    assert MOD.strip() in q and "Top module: `top`" in q
    assert "stays as it is" in sd.SEED_SYSTEM
    assert "verilog" not in sd.SEED_SCHEMA["properties"]  # never the file back
    # the trial's one failure: a name inside a generate block
    assert "never a\n  dotted name" in sd.SEED_SYSTEM


def test_a_failed_answer_is_repaired_once_and_logged(monkeypatch, tmp_path):
    verdicts = iter([{"verdict": "NOT_PROVEN", "reason": "weak facts", "result": None},
                     {"verdict": "NECESSARY", "reason": "ok", "result": None,
                      "triple": {"verilog": MOD, "top_module": "top", "clock": "clk",
                                 "invariants": ["cnt <= 8'd3"], "antecedents": [],
                                 "sanity_covers": [], "cti_state": [], "cti_explanation": ""}}])
    monkeypatch.setattr(sd, "check", lambda seed, answer, root, grade=True: next(verdicts))
    asked = []
    def call(user, effort):
        asked.append(user)
        return ANSWER, {"input": 10, "output": 20}, "ok"
    rec = sd.run_seed(SEED, tmp_path, repairs=1, call=call)
    assert rec["verdict"] == "NECESSARY" and rec["repair_attempts"] == 1
    assert len(asked) == 2 and "weak facts" in asked[1] and MOD.strip() in asked[1]
    assert rec["usage"] == {"input": 20, "output": 40}
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps(rec) + "\n")
    n = sd.write_corpus(log, tmp_path / "corpus.jsonl")
    row = json.loads((tmp_path / "corpus.jsonl").read_text())
    assert n == 1 and row["id"] == "os0_000" and row["open_source"]["repo"] == "repo"


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not on PATH (run hwtools)")
def test_logic_check_passes_assertions_and_catches_a_changed_module():
    base, full = sd.apply_edits(SEED, ANSWER)
    assert sd.same_logic(base, full, "top")[0] is True
    changed = full.replace("cnt != LAST", "cnt != LAST - 1")
    assert sd.same_logic(base, changed, "top")[0] is False
