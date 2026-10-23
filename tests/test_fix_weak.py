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


def one_step_record(i, dropped, tier, trace=None, fixed=True):
    """A 1-step log line; `fixed` lines carry the mark one_step.py puts on
    every line of its fixed check (21 Oct 2026)."""
    line = {"id": i, "dropped": dropped, "tier": tier, "trace": trace}
    return dict(line, clock_edge_fix=True) if fixed else line


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


def test_its_proofs_are_kept_next_to_its_log(tmp_path, monkeypatch):
    """8 Oct 2026: they ran in a temporary folder and were lost."""
    import one_step
    seen = {}

    def fake_prove_true(row, facts, work, depth=1, name=None):
        seen.update(work=Path(work), name=name)
        return {"tier": "PROVEN", "proof_dir": f"{name}_abc"}
    monkeypatch.setattr(one_step, "prove_true", fake_prove_true)
    monkeypatch.setattr(fw, "LOG", tmp_path / "logs" / "fix_weak.jsonl")
    monkeypatch.setattr(fw, "_work", {})
    r = fw.check({"id": "g0_007"}, ["a == b"])
    assert seen["work"] == tmp_path / "logs" / "fix_weak_proofs"
    assert seen["name"] == "g0_007"
    assert r["proof_dir"] == "fix_weak_proofs/g0_007_abc"


# --- only the fixed 1-step check picks rows (22 Oct 2026, review F6) -----
# The 6 Oct log (extender/logs/one_step.jsonl) came from the check that
# read clocked assertions one step late; all 207 answers it called too weak
# pass with the fix, so a run on it would pay to "fix" 103 good answers.

OLD_LOG = ROOT / "extender" / "logs" / "one_step.jsonl"


def _weak_log(fixed=True):
    return [one_step_record("weak", -1, "INDUCTIVE", fixed=fixed),
            one_step_record("weak", -2, "NOT_INDUCTIVE", CEX, fixed=fixed)]


def test_the_fixer_reads_the_fixed_checks_log():
    import one_step
    assert fw.ONE_STEP == one_step.OUT
    assert fw.ONE_STEP.name == "one_step_clock_edge_fix_2026-10-21.jsonl"


def test_lines_made_before_the_fix_pick_no_row():
    with pytest.raises(ValueError,
                       match="2 of 2 .* before the clock-edge fix"):
        fw.targets([row("weak")], _weak_log(fixed=False))
    # one old line among fixed ones is refused too
    mixed = _weak_log() + [one_step_record("other", -1, "INDUCTIVE",
                                           fixed=False)]
    with pytest.raises(ValueError, match="1 of 3"):
        fw.targets([row("weak")], mixed)
    assert [r["id"] for r, _ in fw.targets([row("weak")], _weak_log())] \
        == ["weak"]


def test_a_line_written_by_the_fixed_check_is_accepted(tmp_path,
                                                       monkeypatch):
    """The mark is the one one_step.run really writes: its proof faked,
    the full answer too weak at one step."""
    import one_step

    def fake_prove(r, facts, work, depth=1, name=None):
        weak = depth == 1 and len(facts) == len(r["invariants"])
        return {"tier": "NOT_INDUCTIVE" if weak else "INDUCTIVE",
                "trace": CEX if weak else None, "broke": None, "secs": 0}
    monkeypatch.setattr(one_step, "prove", fake_prove)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(row("weak")) + "\n")
    out = tmp_path / "logs" / "one_step_fixed.jsonl"
    one_step.run(corpus, out, workers=1)
    lines = fw.read_log(out)
    assert lines and all(l["clock_edge_fix"] is True for l in lines)
    assert [(r["id"], c) for r, c in fw.targets([row("weak")], lines)] == \
        [("weak", CEX)]


def _main_setup(monkeypatch, tmp_path, log_lines=None):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(row("weak")) + "\n")
    steps = tmp_path / "one_step_fixed.jsonl"
    if log_lines is not None:
        steps.write_text("".join(json.dumps(r) + "\n" for r in log_lines))
    monkeypatch.setattr(fw, "CORPUS", corpus)
    monkeypatch.setattr(fw, "ONE_STEP", steps)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(fw, "preflight", lambda: None)
    return Fakes(monkeypatch, tmp_path, [answer("a == b")] * 3,
                 {("a == b",): {"tier": "PROVEN"}})


@pytest.mark.parametrize("flags", [[], ["--dry"], ["--report"]])
def test_a_log_from_before_the_fix_stops_every_mode_before_any_call(
        monkeypatch, tmp_path, flags):
    f = _main_setup(monkeypatch, tmp_path, _weak_log(fixed=False))
    with pytest.raises(SystemExit) as e:
        fw.main(["--workers", "1", *flags])
    said = str(e.value.code)
    assert "before the clock-edge fix" in said
    assert "nothing was called" in said and str(fw.ONE_STEP) in said
    assert f.prompts == [] and not fw.LOG.exists()


@pytest.mark.parametrize("flags", [[], ["--dry"], ["--report"]])
def test_no_fixed_check_log_stops_every_mode_before_any_call(
        monkeypatch, tmp_path, flags):
    """A missing log must not read as "nothing to fix"."""
    f = _main_setup(monkeypatch, tmp_path, log_lines=None)
    with pytest.raises(SystemExit) as e:
        fw.main(["--workers", "1", *flags])
    said = str(e.value.code)
    assert "run training/one_step.py first" in said
    assert "nothing was called" in said
    assert f.prompts == [] and not fw.LOG.exists()


def test_a_fixed_log_still_runs(monkeypatch, tmp_path):
    f = _main_setup(monkeypatch, tmp_path, _weak_log())
    fw.main(["--workers", "1", "--max-attempts", "1"])
    assert len(f.prompts) == 1
    assert [r["accepted"] for r in fw.read_log(fw.LOG)] == [True]


@pytest.mark.skipif(not (OLD_LOG.exists() and fw.CORPUS.exists()),
                    reason="no 6 Oct 1-step log here")
def test_the_real_6_oct_log_picks_no_row():
    """It picked 103 rows before (22 Oct 2026, read only)."""
    rows = [json.loads(l) for l in fw.CORPUS.read_text().splitlines()
            if l.strip()]
    with pytest.raises(ValueError, match="before the clock-edge fix"):
        fw.targets(rows, fw.read_log(OLD_LOG))
