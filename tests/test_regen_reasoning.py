"""Regenerating the fake-state reasoning for every corpus row.

No network and no prover here: call_claude and the checker are both
replaced, so these tests pin the bookkeeping (prompt wording, retries,
log lines, resume, cost) that decides what we pay for and what we keep."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "initiator"))

import llm_client                                              # noqa: E402
import regen_reasoning as rr                                   # noqa: E402
from prompts import SYSTEM_PROMPT                              # noqa: E402

VERILOG = """\
module m (input wire clk, input wire step);
    reg [3:0] a = 4'd0;
    reg [3:0] b = 4'd0;
    always @(posedge clk) if (step) begin a <= a + 4'd1; b <= b + 4'd1; end
    always @(posedge clk) if (a == 4'd3) assert (b == 4'd3);
endmodule
"""

ROW_A = {"id": "g0_000", "generation": 0, "verilog": VERILOG,
         "top_module": "m", "clock": "clk", "property": ["b == 4'd3"],
         "antecedents": ["a == 4'd3"], "invariants": ["a == b", "a <= 4'd9"],
         "source_run_id": "r1", "source_attempt": 2}
ROW_B = dict(ROW_A, id="g1_000", generation=1, source_run_id=None,
             source_attempt=None)
ROW_C = dict(ROW_A, id="g1_001", generation=1, source_run_id=None,
             source_attempt=None)

GOOD = {"cti_reasoning": "a=2, b=0 holds P since a != 3, breaks a == b; "
                         "one step gives a=3, b=1.",
        "cti_state": [{"signal": "a", "value": "4'd2"},
                      {"signal": "b", "value": "4'd0"}]}
OK = {"ok": True, "verdict": "REAL", "detail": "all three hold.",
      "style": "x", "wall_s": 0.1}
NO = {"ok": False, "verdict": "R_TRUE", "detail": "R is true at this state.",
      "style": "x", "wall_s": 0.1}
USAGE = {"input": 1000, "output": 200}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tiny corpus, empty logs, a key, and a recording fake model."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n"
                              for r in (ROW_A, ROW_B, ROW_C)))
    monkeypatch.setattr(rr, "CORPUS", corpus)
    monkeypatch.setattr(rr, "LOG", tmp_path / "logs" / "regen.jsonl")
    monkeypatch.setattr(rr, "ORIG_LOG", tmp_path / "logs" / "orig.jsonl")
    monkeypatch.setattr(rr, "ATTEMPTS", tmp_path / "attempts.jsonl")
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(rr, "preflight", lambda: None)

    state = {"prompts": [], "replies": [], "checks": []}

    def fake_call(**kw):
        state["prompts"].append(kw["user"])
        state["kwargs"] = kw
        reply = state["replies"].pop(0) if state["replies"] else GOOD
        if reply == "refusal":
            return None, dict(USAGE), "refusal"
        return json.dumps(reply), dict(USAGE), "ok"

    def fake_check(row, cti_state):
        verdicts = state["checks"]
        return dict(verdicts.pop(0) if verdicts else OK)

    monkeypatch.setattr(llm_client, "call_claude", fake_call)
    monkeypatch.setattr(rr, "check", fake_check)
    state["tmp"] = tmp_path
    return state


def log_lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


# --- prompt -------------------------------------------------------------

def _between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


def test_prompt_carries_the_original_field_notes_verbatim():
    prompt = rr.build_prompt(ROW_A)
    reasoning_note = _between(SYSTEM_PROMPT, "- `cti_reasoning`:", "breaks P.")
    vague = _between(SYSTEM_PROMPT, "Every signal needed", "{\"signal\": \"rptr\", \"value\": \"2'd0\"}] is.")
    self_check = _between(SYSTEM_PROMPT, "Self-check before you answer",
                          "It MUST come out false.")
    for passage in (reasoning_note, vague, self_check):
        assert passage in prompt


def test_prompt_restores_the_original_idle_paragraph_verbatim():
    """Dropped by the first draft; Rishab asked for the original wording
    back (4 Oct 2026). Compared with whitespace squeezed, because the
    original is indented inside its numbered condition."""
    squeeze = lambda s: " ".join(s.split())                    # noqa: E731
    idle = _between(SYSTEM_PROMPT, "S must also be a state the design can IDLE IN",
                    "and the triple is rejected.")
    assert squeeze(idle) in squeeze(rr.build_prompt(ROW_A))


def test_prompt_keeps_idling_and_the_breaking_step_apart():
    """The checker fixes a named input on the breaking step. 11 of the 17
    original fake states that failed named the enable at its idle value,
    so the step did nothing. The rule must name both moments separately."""
    prompt = " ".join(rr.build_prompt(ROW_A).split())
    assert "Two moments" in prompt
    assert "with its enable held low" in prompt
    assert "taken with the enable ON" in prompt
    assert "give its value on that breaking step" in prompt
    assert "never at its idle value" in prompt


def test_prompt_shows_design_assertions_and_each_invariant_on_its_own_line():
    prompt = rr.build_prompt(ROW_A)
    assert VERILOG in prompt
    assert "b == 4'd3" in prompt
    lines = prompt.splitlines()
    for clause in ROW_A["invariants"]:
        assert any(line.strip().endswith(clause) for line in lines)


def test_prompt_states_the_three_conditions_and_the_naming_rules():
    prompt = rr.build_prompt(ROW_A).lower()
    assert "every assertion holds" in prompt
    assert "at least one clause" in prompt
    assert "one clock step" in prompt
    assert "top-level" in prompt and "sized" in prompt
    assert "do not modify the design" in prompt


def test_assert_property_form_is_listed_when_property_field_is_empty():
    src = VERILOG.replace("if (a == 4'd3) assert (b == 4'd3);",
                          "assert property (a == b);")
    row = dict(ROW_A, verilog=src, property=[])
    assert rr.assertions(row) == ["a == b"]


def test_schema_mirrors_the_generator_fields_and_is_strict():
    from schema import TRIPLE_SCHEMA
    assert rr.SCHEMA["required"] == ["cti_reasoning", "cti_state"]
    assert rr.SCHEMA["additionalProperties"] is False
    for k in ("cti_reasoning", "cti_state"):
        assert rr.SCHEMA["properties"][k] == TRIPLE_SCHEMA["properties"][k]


# --- calling, checking, retrying, logging --------------------------------

def test_dry_makes_no_calls_and_needs_no_key(env, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rr.main(["--dry", "--n", "2"])
    out = capsys.readouterr().out
    assert env["prompts"] == []
    assert "`cti_reasoning`" in out
    assert "2 rows" in out
    assert not rr.LOG.exists()


def test_accepted_first_try_logs_one_line(env):
    rr.main(["--ids", "g0_000", "--workers", "1"])
    lines = log_lines(rr.LOG)
    assert len(lines) == 1
    line = lines[0]
    assert line["id"] == "g0_000" and line["attempt"] == 1
    assert line["prompt_kind"] == "regen" and line["accepted"] is True
    assert line["cti_state"] == GOOD["cti_state"]
    assert line["cti_reasoning"] == GOOD["cti_reasoning"]
    assert line["check"]["verdict"] == "REAL"
    assert line["usage"] == USAGE and line["stop"] == "ok"
    assert line["model"] == "claude-opus-5" and line["effort"] == "medium"
    assert "timestamp" in line
    assert env["kwargs"]["model"] == "claude-opus-5"
    assert env["kwargs"]["schema"] == rr.SCHEMA
    assert env["kwargs"]["max_tokens"] == 16000


def test_reject_then_accept_logs_two_lines_and_feeds_back(env):
    env["checks"] = [NO, OK]
    rr.main(["--ids", "g0_000", "--workers", "1"])
    lines = log_lines(rr.LOG)
    assert [l["accepted"] for l in lines] == [False, True]
    assert [l["attempt"] for l in lines] == [1, 2]
    first, second = env["prompts"]
    assert second.startswith(first)
    extra = second[len(first):]
    assert "4'd2" in extra and "R_TRUE" in extra
    assert "R is true at this state." in extra
    assert "Give a corrected fake state and reasoning." in extra


def test_max_attempts_is_respected(env):
    env["checks"] = [NO, NO, NO, NO]
    rr.main(["--ids", "g0_000", "--workers", "1", "--max-attempts", "2"])
    lines = log_lines(rr.LOG)
    assert len(env["prompts"]) == 2
    assert [l["accepted"] for l in lines] == [False, False]


def test_refusal_counts_as_an_attempt_and_is_not_checked(env):
    env["replies"] = ["refusal", GOOD]
    rr.main(["--ids", "g0_000", "--workers", "1", "--max-attempts", "2"])
    lines = log_lines(rr.LOG)
    assert [l["stop"] for l in lines] == ["refusal", "ok"]
    assert lines[0]["check"] is None and lines[0]["accepted"] is False
    assert lines[1]["accepted"] is True


def test_unparseable_reply_counts_as_an_attempt(env):
    env["replies"] = [{"cti_reasoning": "no state given"}, GOOD]
    rr.main(["--ids", "g0_000", "--workers", "1"])
    lines = log_lines(rr.LOG)
    assert len(lines) == 2
    assert lines[0]["check"] is None and lines[0]["accepted"] is False


def test_checker_crash_stops_the_row_without_paying_again(env, monkeypatch):
    def boom(row, state):
        raise RuntimeError("sby missing")
    monkeypatch.setattr(rr, "check", boom)
    rr.main(["--ids", "g0_000", "--workers", "1"])
    lines = log_lines(rr.LOG)
    assert len(env["prompts"]) == 1 and len(lines) == 1
    assert lines[0]["check"]["verdict"] == "check_error"
    assert "sby missing" in lines[0]["check"]["detail"]


@pytest.mark.parametrize("verdict", ["ERROR", "TIMEOUT"])
def test_checker_tool_failure_is_not_sent_back_to_the_model(env, verdict):
    # the checker could not judge the state; a new paid state would not help
    env["checks"] = [{"ok": False, "verdict": verdict, "detail": "sby died"}]
    rr.main(["--ids", "g0_000", "--workers", "1"])
    assert len(env["prompts"]) == 1
    assert len(log_lines(rr.LOG)) == 1


def test_prompt_warns_against_bit_slices():
    assert "bit slice" in rr.build_prompt(ROW_A).lower()


def test_resume_skips_rows_already_accepted(env):
    rr.LOG.parent.mkdir(parents=True)
    rr.LOG.write_text(json.dumps({"id": "g0_000", "attempt": 1,
                                  "accepted": True, "usage": USAGE}) + "\n"
                      + json.dumps({"id": "g1_000", "attempt": 1,
                                    "accepted": False, "usage": USAGE}) + "\n")
    rr.main(["--n", "2", "--workers", "1"])
    lines = log_lines(rr.LOG)
    new = lines[2:]
    assert [l["id"] for l in new] == ["g1_000"]
    # numbering continues after the earlier attempt
    assert new[0]["attempt"] == 2


def test_many_workers_write_whole_lines(env):
    rr.main(["--n", "3", "--workers", "3"])
    lines = log_lines(rr.LOG)
    assert sorted(l["id"] for l in lines) == ["g0_000", "g1_000", "g1_001"]


def test_a_line_cut_short_by_a_killed_run_does_not_swallow_the_next(tmp_path):
    # a kill mid-write leaves a line with no newline; the next record must
    # start on its own line, or it is glued on and lost with the cut one
    log = tmp_path / "regen.jsonl"
    log.write_text(json.dumps({"id": "a", "attempt": 1, "accepted": True})
                   + "\n" + '{"id": "b", "attempt": 1, "cti_reas')
    rr.append(log, {"id": "c", "attempt": 1, "accepted": True})
    assert [r["id"] for r in rr.read_log(log)] == ["a", "c"]


def test_missing_anthropic_key_exits_before_any_call(env, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        rr.main(["--n", "1"])
    assert e.value.code != 0
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert env["prompts"] == []


def test_missing_openrouter_key_exits_before_any_call(env, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        rr.main(["--n", "1"])
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert env["prompts"] == []


def test_checker_not_ready_exits_before_any_call(env, monkeypatch, capsys):
    monkeypatch.setattr(rr, "preflight", lambda: "sby is not on PATH")
    with pytest.raises(SystemExit):
        rr.main(["--n", "1"])
    assert "sby is not on PATH" in capsys.readouterr().err
    assert env["prompts"] == []


def test_preflight_names_the_missing_prover(monkeypatch):
    monkeypatch.setattr(rr.shutil, "which", lambda name: None)
    assert "oss-cad-suite" in rr.preflight()


# --- report -------------------------------------------------------------

def test_report_turns_tokens_into_dollars(tmp_path):
    log = tmp_path / "regen.jsonl"
    rows = [
        {"id": "a", "attempt": 1, "accepted": True, "stop": "ok",
         "check": {"verdict": "REAL"}, "usage": {"input": 1_000_000, "output": 0}},
        {"id": "b", "attempt": 1, "accepted": False, "stop": "ok",
         "check": {"verdict": "R_TRUE"}, "usage": {"input": 0, "output": 1_000_000}},
        {"id": "b", "attempt": 2, "accepted": True, "stop": "ok",
         "check": {"verdict": "REAL"}, "usage": {"input": 0, "output": 0}},
        {"id": "c", "attempt": 1, "accepted": False, "stop": "refusal",
         "check": None, "usage": {"input": 0, "output": 0}},
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = rr.report(log, ["a", "b", "c", "d"])
    assert rep["verdicts"] == {"REAL": 2, "R_TRUE": 1, "refusal": 1}
    assert rep["first_try"] == 1 and rep["after_retries"] == 1
    assert rep["missing"] == 2          # c tried and failed, d never tried
    assert rep["input"] == 1_000_000 and rep["output"] == 1_000_000
    assert rep["cost"] == pytest.approx(30.0)
    assert rep["cost_per_accepted"] == pytest.approx(15.0)


def test_report_flag_makes_no_calls(env, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rr.LOG.parent.mkdir(parents=True)
    rr.LOG.write_text(json.dumps({"id": "g0_000", "attempt": 1,
                                  "accepted": True, "stop": "ok",
                                  "check": {"verdict": "REAL"},
                                  "usage": USAGE}) + "\n")
    rr.main(["--report"])
    out = capsys.readouterr().out
    assert env["prompts"] == []
    assert "REAL" in out and "$" in out


# --- the original 228 ---------------------------------------------------

def _attempt(run_id, attempt, reasoning, state):
    raw = json.dumps({"cti_reasoning": reasoning, "cti_state": state,
                      "verilog": "x"})
    return {"run_id": run_id, "attempt": attempt, "verdict": "NECESSARY",
            "raw_json": raw}


def test_originals_link_gen0_rows_by_run_and_attempt(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    # the right line comes first, so a loose match would be overwritten
    lines = [_attempt("r1", 2, "the right one", GOOD["cti_state"]),
             _attempt("r1", 1, "wrong attempt", []),
             _attempt("r2", 2, "wrong run", [])]
    attempts.write_text("".join(json.dumps(l) + "\n" for l in lines))
    found = rr.load_originals([ROW_A, ROW_B], attempts)
    assert len(found) == 1
    row, reasoning, state = found[0]
    assert row["id"] == "g0_000"
    assert reasoning == "the right one"
    assert state == GOOD["cti_state"]


def test_originals_flag_checks_and_logs_without_calls(env, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rr.ATTEMPTS.write_text(json.dumps(
        _attempt("r1", 2, "the right one", GOOD["cti_state"])) + "\n")
    env["checks"] = [NO]
    rr.main(["--originals", "--workers", "1"])
    lines = log_lines(rr.ORIG_LOG)
    assert len(lines) == 1
    assert lines[0]["id"] == "g0_000"
    assert lines[0]["prompt_kind"] == "original"
    assert lines[0]["accepted"] is False
    assert lines[0]["cti_reasoning"] == "the right one"
    assert env["prompts"] == []
    assert "R_TRUE" in capsys.readouterr().out


# --- wiring to the real checker, when it and the prover are present ------

@pytest.mark.skipif(shutil.which("sby") is None,
                    reason="sby not on PATH - source oss-cad-suite first")
def test_check_reaches_the_real_checker_on_an_original_row():
    pytest.importorskip("cti_check")
    rows = [json.loads(l) for l in rr.CORPUS.read_text().splitlines()]
    found = rr.load_originals(rows[:1], rr.ATTEMPTS)
    row, _, state = found[0]
    result = rr.check(row, state)
    assert {"ok", "verdict", "detail"} <= set(result)


def test_progress_label_shows_rows_done_in_progress_and_cost():
    label = rr.progress_label(done=2, total=5, active=3, dollars=0.123)
    assert "2/5 rows done" in label
    assert "3 in progress" in label
    assert "$0.12" in label


def test_a_run_drives_one_spinner_to_the_end(env, monkeypatch):
    """One spinner for the whole run (several workers share it), its
    label updated as rows start and finish."""
    seen = []

    class FakeSpinner:
        def __init__(self, label, always=False):
            self._label = label
            seen.append(label)

        @property
        def label(self):
            return self._label

        @label.setter
        def label(self, value):
            self._label = value
            seen.append(value)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(rr, "Spinner", FakeSpinner)
    rr.main(["--n", "2", "--workers", "1"])
    assert "0/2 rows done" in seen[0]
    assert "1 in progress" in " ".join(seen)
    assert "2/2 rows done" in seen[-1] and "0 in progress" in seen[-1]
