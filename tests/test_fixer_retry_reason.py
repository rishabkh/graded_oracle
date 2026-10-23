"""The fixer's next question says why the last list was turned away (23 Oct
2026, review finding).

After a list was turned away before any check (FACT_FORMAT, READS_RESET,
PRE_GATE), the fixer's next paid attempt got exactly the same question,
with no reason, so it could repeat the mistake: the reviewer's probe had
attempts 1 and 2 both turned away as FACT_FORMAT, with identical
questions. The batch runner already ends a retry's question with the
reason (batch.retry_section, review finding B1 of 22 Oct); the fixer now
does the same. A checked attempt needs no reason: the next question
carries its list and its new counterexample, as before.
The model and the checker are faked: no calls, no proofs."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import fix                                                      # noqa: E402
from oracle.types import NecessityVerdict, TripleResult         # noqa: E402

DESIGN = """module m(input clk, input en, output reg [3:0] c, output reg [3:0] d);
initial c = 0; initial d = 0;
always @(posedge clk) if (en) begin c <= c + 1; d <= c; end
always @(posedge clk) assert (d <= 4'd15);
endmodule"""
ROWS = [{"id": "p0", "verilog": DESIGN, "top_module": "m", "clock": "clk",
         "antecedents": [], "sanity_covers": [],
         "invariants": ["c <= 4'd15"]}]
RECORD = {"extension_id": "2026-10-23_t001", "parent_id": "p0",
          "ext_type": "structural", "child_verilog": DESIGN,
          "invariants": ["c <= 4'd15"], "verdict": "NOT_PROVEN",
          "result": {"with_invariants": {"runs": [
              {"mode": "prove", "rc": 4, "notes": [],
               "trace_text": "c = 3, d = 9"}]}}}
NOT_YET = TripleResult(
    verdict=NecessityVerdict.NOT_PROVEN, reason="still fails",
    with_invariants=None)


@pytest.fixture
def fixer(monkeypatch, tmp_path):
    """The attempt log in a temporary folder, the checker faked (it says
    NOT_PROVEN), and the questions the model was asked."""
    monkeypatch.setattr(fix, "ATTEMPT_LOG", tmp_path / "fixer_attempts.jsonl")
    monkeypatch.setattr(fix, "grade_triple_generated",
                        lambda payload, **kw: NOT_YET)
    asked = []

    def run(*lists):
        queue = [{"diagnosis": f"try {n}", "invariants": list(x)}
                 for n, x in enumerate(lists, 1)]

        def call(prompt, attempt):
            asked.append(prompt)
            return queue.pop(0)
        fix.run_task(RECORD, ROWS, call=call, max_attempts=len(lists))
        log = tmp_path / "fixer_attempts.jsonl"
        return [json.loads(l) for l in log.read_text().splitlines()]
    return {"asked": asked, "run": run}


def base_question(invariants=("c <= 4'd15",), cti="c = 3, d = 9"):
    return fix.FIXER_PROMPT.format(verilog=DESIGN, property="d <= 4'd15",
                                   invariants="\n".join(invariants), pdr="",
                                   cti=cti)


def test_the_reviewers_probe_no_longer_asks_the_same_question_twice(fixer):
    lines = fixer["run"](["$past(c) + 1 == c"], ["$signed(d) >= 0",
                                                 "c <= 4'd15"],
                         ["d <= c", "c <= 4'd15"])
    assert [l["verdict"] for l in lines] == ["FACT_FORMAT", "FACT_FORMAT",
                                            "NOT_PROVEN"]
    q1, q2, q3 = fixer["asked"]
    assert q1 == base_question()                    # the first is as before
    assert q2 != q1 and q3 != q2
    first = fix.rejection_lesson(lines[0]["error"])
    assert q2 == base_question() + batch.retry_section(first)
    assert "$past(c) + 1 == c (system function $past)" in q2
    assert q2.endswith("Fix that in a new, complete answer.")
    # each question answers only the attempt before it
    second = fix.rejection_lesson(lines[1]["error"])
    assert q3 == base_question() + batch.retry_section(second)
    assert "$past" not in q3.split("sent back:")[1]
    # the log keeps the reason each question carried
    assert "retry_lesson" not in lines[0]
    assert [l.get("retry_lesson") for l in lines[1:]] == [first, second]


def test_an_unchanged_list_is_named_too(fixer):
    lines = fixer["run"](["c <= 4'd15"], ["d <= c", "c <= 4'd15"])
    assert lines[0]["verdict"] == "PRE_GATE"
    assert fixer["asked"][1].endswith(
        "sent back: candidate list is identical to the failed list.\n"
        "Fix that in a new, complete answer.")


def test_a_fact_that_reads_the_reset_is_named_too(fixer, monkeypatch):
    """A design with an asynchronous reset: the first list reads it."""
    reset = DESIGN.replace("input clk,", "input clk, input rst_n,").replace(
        "always @(posedge clk) if (en)",
        "always @(posedge clk or negedge rst_n) if (!rst_n) begin c <= 0; "
        "d <= 0; end else if (en)")
    monkeypatch.setitem(RECORD, "child_verilog", reset)
    lines = fixer["run"](["rst_n || c == 4'd0"], ["d <= c", "c <= 4'd15"])
    assert lines[0]["verdict"] == "READS_RESET"
    assert "fact `rst_n || c == 4'd0`" in fixer["asked"][1].split(
        "sent back:")[1]
    assert lines[1]["retry_lesson"] == fix.rejection_lesson(
        lines[0]["error"])


def test_a_checked_attempt_is_followed_by_its_own_counterexample_only(
        fixer, monkeypatch):
    trace = {"with_invariants": {"runs": [
        {"mode": "prove", "rc": 4, "notes": [], "trace_text": "c = 7, d = 8"}]}}
    monkeypatch.setattr(fix, "asdict", lambda result: trace)
    lines = fixer["run"](["$past(c) == c"], ["d <= c", "c <= 4'd15"],
                         ["d <= c + 1", "c <= 4'd15"])
    assert [l["verdict"] for l in lines] == ["FACT_FORMAT", "NOT_PROVEN",
                                            "NOT_PROVEN"]
    q3 = fixer["asked"][2]
    assert "sent back" not in q3 and "retry_lesson" not in lines[2]
    assert q3 == base_question(["d <= c", "c <= 4'd15"], "c = 7, d = 8")


def test_a_long_reason_is_cut_like_the_batch_runners():
    long = "x" * (batch.LESSON_MAX + 50)
    assert fix.rejection_lesson(long) == "x" * batch.LESSON_MAX + " ..."
    assert fix.rejection_lesson("two  spaces\nand a line") == \
        "two spaces and a line."
    assert fix.rejection_lesson("already ends.") == "already ends."
