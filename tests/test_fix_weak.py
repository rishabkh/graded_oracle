"""Task 4: Opus strengthens the corpus answers that are true but too weak
under the benchmark's 1-step rule. Opus and the proof tool are both
faked here: no calls, no sby, no money."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import fix_weak as fw                                            # noqa: E402

DESIGN = """module toy (input wire clk, input wire i_ce);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    always @(posedge clk) if (i_ce) begin a <= a + 3'd1; b <= b + 3'd1; end
    always @(posedge clk) assert (!(a == 3'd3) || (b == 3'd3));
endmodule
"""
CEX = "Trace summary (trace_induct.vcd):\n  At start state (step 0):\n    a = 3'h2\n    b = 3'h0"


def row(i, verilog=DESIGN, facts=("a <= 3'd7",)):
    return {"id": i, "top_module": "toy", "clock": "clk", "verilog": verilog,
            "invariants": list(facts)}


def one_step_record(i, dropped, tier, trace=None):
    return {"id": i, "dropped": dropped, "tier": tier, "trace": trace}


# --- which rows ----------------------------------------------------------

def test_targets_are_the_too_weak_rows_that_do_not_look_back():
    looks_back = DESIGN.replace("assert (!(a == 3'd3) || (b == 3'd3))",
                                "assert ($stable(a))")
    past = DESIGN.replace("assert (!(a == 3'd3) || (b == 3'd3))",
                          "assert (a == $past(b))")
    corpus = [row("weak"), row("fine"), row("stable", looks_back),
              row("past", past)]
    log = [one_step_record(i, -1, "INDUCTIVE") for i in
           ("weak", "fine", "stable", "past")]
    log += [one_step_record("weak", -2, "NOT_INDUCTIVE", CEX),
            one_step_record("fine", -2, "INDUCTIVE"),
            one_step_record("stable", -2, "NOT_INDUCTIVE", CEX),
            one_step_record("past", -2, "NOT_INDUCTIVE", CEX)]
    picked = fw.targets(corpus, log)
    assert [(r["id"], cex) for r, cex in picked] == [("weak", CEX)]


# --- what Opus is asked --------------------------------------------------

def test_the_question_carries_design_property_facts_and_counterexample():
    q = fw.build_prompt(row("weak"), CEX)
    assert DESIGN.strip() in q
    assert "!(a == 3'd3) || (b == 3'd3)" in q          # the property
    assert "- a <= 3'd7" in q                          # the current facts
    assert CEX in q
    # the same fact rules the generator was given
    assert "$past" in q and "!a || b" in q and "always @(*)" in q


def test_a_rejection_says_which_fact_and_why():
    false = fw.feedback(["a == b", "a == 3'd0"],
                        {"tier": "FALSE", "fact": "a == 3'd0",
                         "trace": "RUN FROM RESET"})
    assert "- a == 3'd0" in false and "`a == 3'd0` is false" in false
    assert "RUN FROM RESET" in false
    weak = fw.feedback(["a <= 3'd7"], {"tier": "NOT_INDUCTIVE", "fact": None,
                                       "trace": CEX})
    assert "the property" in weak and CEX in weak
    bad = fw.feedback(["a == q"], {"tier": "BAD_FACT", "fact": "a == q",
                                   "detail": "unknown name `q`"})
    assert "`a == q`" in bad and "unknown name `q`" in bad


@pytest.mark.parametrize("text, facts", [
    ('{"why": "x", "invariants": ["a == b"]}', ["a == b"]),
    ('{"why": "x", "invariants": []}', None),
    ('{"why": "x", "invariants": ["a == b", 3]}', None),
    ("not json", None),
    (None, None),
])
def test_only_a_non_empty_list_of_strings_is_an_answer(text, facts):
    assert fw.parse_answer(text) == facts


# --- one row: ask, check, retry -----------------------------------------

class Fakes:
    """Opus replies in order; the proof tool's verdict per fact list."""
    def __init__(self, monkeypatch, tmp_path, replies, verdicts):
        self.prompts, self.replies, self.verdicts = [], list(replies), verdicts
        monkeypatch.setattr(fw, "LOG", tmp_path / "fix.jsonl")
        monkeypatch.setattr(fw.llm_client, "call_claude", self.call)
        monkeypatch.setattr(fw, "check", self.check)

    def call(self, *, user, **kw):
        self.prompts.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"input": 1000, "output": 100}, "ok"

    def check(self, row, facts):
        return dict(self.verdicts.get(tuple(facts), {"tier": "NOT_INDUCTIVE",
                                                     "fact": None,
                                                     "trace": "NEW CEX"}))


def answer(*facts):
    return json.dumps({"why": "because", "invariants": list(facts)})


def test_a_proven_list_is_accepted_and_logged(monkeypatch, tmp_path):
    f = Fakes(monkeypatch, tmp_path, [answer("a == b")],
              {("a == b",): {"tier": "PROVEN"}})
    out = fw.fix_row(row("weak"), CEX, effort="high", max_tokens=100,
                     max_attempts=3)
    assert out["accepted"] and out["attempts"] == 1
    [rec] = fw.read_log(fw.LOG)
    assert rec["accepted"] and rec["invariants"] == ["a == b"]
    assert rec["original"] == ["a <= 3'd7"] and rec["check"]["tier"] == "PROVEN"


def test_a_rejection_goes_back_with_the_tools_answer(monkeypatch, tmp_path):
    f = Fakes(monkeypatch, tmp_path,
              [answer("a == 3'd0"), answer("a == b")],
              {("a == 3'd0",): {"tier": "FALSE", "fact": "a == 3'd0",
                                "trace": "RUN FROM RESET"},
               ("a == b",): {"tier": "PROVEN"}})
    out = fw.fix_row(row("weak"), CEX, effort="high", max_tokens=100,
                     max_attempts=3)
    assert out["accepted"] and out["attempts"] == 2
    assert "RUN FROM RESET" not in f.prompts[0]
    assert "RUN FROM RESET" in f.prompts[1] and CEX in f.prompts[1]
    assert [r["accepted"] for r in fw.read_log(fw.LOG)] == [False, True]


def test_attempts_run_out(monkeypatch, tmp_path):
    f = Fakes(monkeypatch, tmp_path, [answer("a <= 3'd6")] * 3, {})
    out = fw.fix_row(row("weak"), CEX, effort="high", max_tokens=100,
                     max_attempts=3)
    assert not out["accepted"] and out["attempts"] == 3
    assert out["verdict"] == "NOT_INDUCTIVE"


@pytest.mark.parametrize("tier", ["TIMEOUT", "ERROR"])
def test_a_tool_that_cannot_judge_stops_the_row_without_paying_more(
        monkeypatch, tmp_path, tier):
    f = Fakes(monkeypatch, tmp_path, [answer("a == b")] * 3,
              {("a == b",): {"tier": tier}})
    out = fw.fix_row(row("weak"), CEX, effort="high", max_tokens=100,
                     max_attempts=3)
    assert out["attempts"] == 1 and out["verdict"] == tier
    assert len(f.prompts) == 1


def test_an_api_error_stops_the_row_for_a_rerun(monkeypatch, tmp_path):
    f = Fakes(monkeypatch, tmp_path, [RuntimeError("overloaded")], {})
    out = fw.fix_row(row("weak"), CEX, effort="high", max_tokens=100,
                     max_attempts=3)
    assert out["verdict"] == "error" and not out["accepted"]
    [rec] = fw.read_log(fw.LOG)
    assert "overloaded" in rec["error"]


def test_an_unreadable_reply_is_asked_again_unchanged(monkeypatch, tmp_path):
    f = Fakes(monkeypatch, tmp_path, ["no json", answer("a == b")],
              {("a == b",): {"tier": "PROVEN"}})
    out = fw.fix_row(row("weak"), CEX, effort="high", max_tokens=100,
                     max_attempts=3)
    assert out["accepted"] and f.prompts[0] == f.prompts[1]


# --- the run ---------------------------------------------------------------

def test_a_rerun_skips_accepted_rows_and_numbers_attempts_on(monkeypatch,
                                                            tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row(i)) + "\n" for i in ("r1", "r2")))
    steps = tmp_path / "one_step.jsonl"
    steps.write_text("".join(json.dumps(r) + "\n" for i in ("r1", "r2") for r in (
        one_step_record(i, -1, "INDUCTIVE"),
        one_step_record(i, -2, "NOT_INDUCTIVE", CEX))))
    monkeypatch.setattr(fw, "CORPUS", corpus)
    monkeypatch.setattr(fw, "ONE_STEP", steps)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(fw, "preflight", lambda: None)
    f = Fakes(monkeypatch, tmp_path, [answer("a <= 3'd6"), answer("a == b")],
              {("a == b",): {"tier": "PROVEN"}})
    earlier = {"original": ["a <= 3'd7"], "invariants": ["a == b"],
               "usage": {"input": 0, "output": 0}}
    fw.LOG.write_text(
        json.dumps(dict(earlier, id="r1", attempt=1, accepted=True,
                        check={"tier": "PROVEN"})) + "\n"
        + json.dumps(dict(earlier, id="r2", attempt=1, accepted=False,
                          check={"tier": "NOT_INDUCTIVE"})) + "\n")
    fw.main(["--workers", "1", "--max-attempts", "2"])
    new = fw.read_log(fw.LOG)[2:]
    assert {r["id"] for r in new} == {"r2"}
    assert [r["attempt"] for r in new] == [2, 3]


def test_dry_and_report_make_no_calls(monkeypatch, tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(row("r1")) + "\n")
    steps = tmp_path / "one_step.jsonl"
    steps.write_text("".join(json.dumps(r) + "\n" for r in (
        one_step_record("r1", -1, "INDUCTIVE"),
        one_step_record("r1", -2, "NOT_INDUCTIVE", CEX))))
    monkeypatch.setattr(fw, "CORPUS", corpus)
    monkeypatch.setattr(fw, "ONE_STEP", steps)
    f = Fakes(monkeypatch, tmp_path, [], {})
    fw.main(["--dry"])
    assert "1 rows would be called" in capsys.readouterr().out
    fw.main(["--report"])
    assert f.prompts == []
