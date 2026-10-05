"""The untrained model writing its own fake state, reasoning and answer.

No network and no prover here: the model endpoint, the proof grader and
the fake-state checker are all replaced, so these tests pin what we ask,
what we keep, and what lands in the log. One test at the end uses the
real grader and checker and skips when sby is not on PATH."""
import json
import re
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "initiator"))

import self_reason as sr                                       # noqa: E402
from solver_baseline import SOLVER_PROMPT                      # noqa: E402

VERILOG = """\
module m (input wire clk, input wire step);
    // narration that gives the answer away: a and b always agree
    reg [3:0] a = 4'd0;
    reg [3:0] b = 4'd0;
    always @(posedge clk) if (step) begin a <= a + 4'd1; b <= b + 4'd1; end
    always @(posedge clk) if (a == 4'd3) assert (b == 4'd3); /* a == b */
endmodule
"""

ROW_A = {"id": "g0_000", "generation": 0, "ext_type": None,
         "verilog": VERILOG, "top_module": "m", "clock": "clk",
         "antecedents": ["a == 4'd3"], "sanity_covers": [],
         "property": ["b == 4'd3"], "invariants": ["a == b", "a <= 4'd9"]}
ROW_B = dict(ROW_A, id="g1_000", generation=1, ext_type="structural")
ROW_C = dict(ROW_A, id="g2_000", generation=2, ext_type="distractor")

STATE = [{"signal": "a", "value": "4'd2"}, {"signal": "b", "value": "4'd0"}]
GOOD = {"fake_state": STATE,
        "reasoning": "a=2, b=0 holds the assertion since a != 3; real runs "
                     "keep a == b; one step gives a=3, b=1.",
        "invariants": ["a == b"]}
SERVED = "Qwen/Qwen2.5-Coder-32B-Instruct"
TRAINED = "/n/home/rk/graded_oracle/runs/v3/merged"


def reply(**changes):
    return json.dumps(dict(GOOD, **changes))


class FakeClient:
    """Stands in for the OpenAI client: one listing, and a chat call that
    hands back n choices from a queue of reply texts (GOOD when empty).
    An entry may be (text, finish_reason)."""

    def __init__(self, served=SERVED):
        self.served = served
        self.requests = []
        self.replies = []
        self.fail = 0           # this many next requests raise
        self.choices_cap = None  # return at most this many choices
        self.models = SimpleNamespace(list=self._list)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _list(self):
        return SimpleNamespace(data=[SimpleNamespace(id="llm",
                                                     root=self.served)])

    def _create(self, **kw):
        self.requests.append(kw)
        if self.fail:
            self.fail -= 1
            raise ConnectionError("tunnel closed")
        n = kw.get("n", 1)
        if self.choices_cap is not None:
            n = min(n, self.choices_cap)
        choices = []
        for _ in range(n):
            item = self.replies.pop(0) if self.replies else json.dumps(GOOD)
            text, finish = item if isinstance(item, tuple) else (item, "stop")
            choices.append(SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason=finish))
        return SimpleNamespace(choices=choices, usage=SimpleNamespace(
            prompt_tokens=900, completion_tokens=150 * n))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tiny corpus, an empty log, a fake endpoint, grader and checker."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n"
                              for r in (ROW_A, ROW_B, ROW_C)))
    monkeypatch.setattr(sr, "CORPUS", corpus)
    monkeypatch.setattr(sr, "LOG", tmp_path / "logs" / "self_reason.jsonl")
    monkeypatch.setattr(sr, "preflight", lambda: None)
    monkeypatch.setenv("QWEN_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    client = FakeClient()
    state = {"client": client, "grades": [], "checks": [],
             "proof": "NECESSARY", "fake": "REAL"}
    lock = threading.Lock()

    def fake_grade(row, invariants):
        with lock:
            state["grades"].append((row["id"], list(invariants)))
        verdict = state["proof"]
        if callable(verdict):
            verdict = verdict(invariants)
        return {"verdict": verdict, "reason": "fake grader"}

    def fake_check(row, fake_state):
        with lock:
            state["checks"].append((dict(row), fake_state))
        verdict = state["fake"]
        if callable(verdict):
            verdict = verdict(row["invariants"])
        return {"ok": verdict == "REAL", "verdict": verdict,
                "detail": "fake checker", "style": "clocked", "wall_s": 0.0}

    monkeypatch.setattr(sr, "make_client", lambda: client)
    monkeypatch.setattr(sr, "grade", fake_grade)
    monkeypatch.setattr(sr, "check", fake_check)
    return state


def log_lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


# --- prompt -------------------------------------------------------------

def _names_the_answer(text):
    return (re.search(r"\bR\b", text) or " R " in text or "R's" in text
            or "clause" in text.lower())


def test_prompt_never_calls_the_answer_R_or_a_clause():
    """v3 learned to talk about an R the question never mentions."""
    assert not _names_the_answer(sr.PROMPT)
    assert not _names_the_answer(sr.build_prompt(ROW_A))


def test_no_corpus_design_gets_a_prompt_that_names_R_or_a_clause():
    if not sr.CORPUS.exists():
        pytest.skip("no corpus here")
    rows = [json.loads(l) for l in sr.CORPUS.read_text().splitlines()
            if l.strip()]
    bad = [r["id"] for r in rows if _names_the_answer(sr.build_prompt(r))]
    assert bad == []


def test_prompt_shows_the_design_with_comments_stripped():
    prompt = sr.build_prompt(ROW_A)
    assert "narration" not in prompt and "/*" not in prompt
    assert "always @(posedge clk) if (a == 4'd3) assert (b == 4'd3);" \
        in prompt


def test_prompt_is_the_solver_question_with_only_the_reply_format_changed():
    prompt = sr.build_prompt(ROW_A)
    head = SOLVER_PROMPT[:SOLVER_PROMPT.index("{verilog}")]
    assert prompt.startswith(head)
    find = SOLVER_PROMPT[SOLVER_PROMPT.index("Find strengthening"):
                         SOLVER_PROMPT.index("Reply with JSON")]
    assert find in prompt
    rules = SOLVER_PROMPT[SOLVER_PROMPT.index("Format rules"):]
    assert rules in prompt
    assert '{"invariants": ["<expr>", ...]}' not in prompt


def test_prompt_asks_for_fake_state_then_reasoning_then_invariants():
    prompt = sr.build_prompt(ROW_A)
    fmt = prompt[prompt.index("Reply with JSON"):]
    first = fmt.index('"fake_state"')
    assert first < fmt.index('"reasoning"') < fmt.index('"invariants"')
    flat = " ".join(prompt.split())
    assert "every assertion holds" in flat
    assert "can never reach in real operation" in flat
    assert "ONE clock step breaks an assertion" in flat
    assert "one or two sentences" in flat
    assert "rule that state out" in flat


def test_prompt_states_the_fake_state_rules():
    flat = " ".join(sr.build_prompt(ROW_A).split())
    assert "top-level signal names only" in flat.lower()
    assert "sized constant" in flat
    assert ("every signal needed to evaluate the assertions and your "
            "invariants must appear") in flat.lower()


def test_prompt_keeps_idling_and_the_breaking_step_apart():
    flat = " ".join(sr.build_prompt(ROW_A).split())
    assert "Two moments, kept apart" in flat
    assert "with its enable held low" in flat
    assert "taken with the enable ON" in flat
    assert "give its value on that breaking step" in flat
    assert "never its idle value" in flat
    assert "You may leave inputs out" in flat


# --- reading a reply -----------------------------------------------------

def test_parse_is_lenient_about_fences_prose_and_the_quote_escape():
    text = ("Here is my answer.\n```json\n"
            + json.dumps(GOOD).replace("a=2", "a\\'s value 2") + "\n```\n")
    fields, problem = sr.parse_reply(text)
    assert problem is None
    assert fields["fake_state"] == STATE
    assert fields["invariants"] == ["a == b"]
    assert "a's value 2" in fields["reasoning"]


@pytest.mark.parametrize("bad,field", [
    ({"fake_state": []}, "fake_state"),
    ({"fake_state": [{"signal": "a", "value": 2}]}, "fake_state"),
    ({"reasoning": "  "}, "reasoning"),
    ({"invariants": []}, "invariants"),
    ({"invariants": ["a == b", ""]}, "invariants"),
])
def test_parse_names_the_field_that_is_wrong(bad, field):
    fields, problem = sr.parse_reply(json.dumps(dict(GOOD, **bad)))
    assert fields[field] is None
    assert field in problem


def test_parse_of_no_json_at_all():
    fields, problem = sr.parse_reply("the answer is a == b")
    assert fields == {"fake_state": None, "reasoning": None,
                      "invariants": None}
    assert problem


# --- the guard and the checks before any request -------------------------

def test_a_fine_tuned_model_is_refused_before_any_request(env, capsys):
    env["client"].served = TRAINED
    with pytest.raises(SystemExit) as e:
        sr.main(["--ids", "g0_000"])
    assert e.value.code != 0
    err = capsys.readouterr().err
    assert "runs/" in err and "--allow-trained" in err
    assert env["client"].requests == []
    assert not sr.LOG.exists()


def test_allow_trained_lets_a_fine_tuned_model_run(env):
    env["client"].served = TRAINED
    sr.main(["--ids", "g0_000", "--tries", "2", "--allow-trained"])
    lines = log_lines(sr.LOG)
    assert len(lines) == 2
    assert {l["served_model"] for l in lines} == {TRAINED}


def test_an_endpoint_that_will_not_say_what_it_serves_is_refused(env):
    env["client"].served = None
    env["client"].models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[]))
    with pytest.raises(SystemExit):
        sr.main(["--ids", "g0_000"])
    assert env["client"].requests == []


def _listing(*entries):
    return lambda: SimpleNamespace(data=[SimpleNamespace(**e)
                                         for e in entries])


def test_a_fine_tune_served_as_an_adapter_under_our_name_is_refused(
        env, monkeypatch):
    """vLLM lists the base and each adapter; the guard must look at the
    entry we will ask, not trust the name we ask for."""
    monkeypatch.setenv("QWEN_MODEL", "llm")
    env["client"].models = SimpleNamespace(list=_listing(
        dict(id="base", root=SERVED),
        dict(id="llm", root=TRAINED.replace("merged", "adapter"),
             parent="base")))
    with pytest.raises(SystemExit):
        sr.main(["--ids", "g0_000"])
    assert env["client"].requests == []


def test_a_fine_tune_served_under_two_names_is_refused(env, monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "llm")
    env["client"].models = SimpleNamespace(list=_listing(
        dict(id="llm", root=TRAINED), dict(id="v3", root=TRAINED)))
    with pytest.raises(SystemExit):
        sr.main(["--ids", "g0_000"])
    assert env["client"].requests == []


def test_the_base_under_our_name_runs_beside_an_adapter(env, monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "llm")
    env["client"].models = SimpleNamespace(list=_listing(
        dict(id="llm", root=SERVED),
        dict(id="v3", root=TRAINED.replace("merged", "adapter"),
             parent="llm")))
    sr.main(["--ids", "g0_000", "--tries", "1"])
    assert log_lines(sr.LOG)[0]["served_model"] == SERVED


def test_a_name_the_endpoint_does_not_list_is_refused(env, monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "something-else")
    with pytest.raises(SystemExit):
        sr.main(["--ids", "g0_000"])
    assert env["client"].requests == []


def test_endpoint_not_answering_exits_before_any_request(env, monkeypatch,
                                                         capsys):
    def down():
        raise ConnectionError("connection refused")
    env["client"].models = SimpleNamespace(list=down)
    with pytest.raises(SystemExit):
        sr.main(["--ids", "g0_000"])
    assert "not answering" in capsys.readouterr().err
    assert env["client"].requests == []


def test_tools_not_ready_exits_before_any_request(env, monkeypatch, capsys):
    monkeypatch.setattr(sr, "preflight", lambda: "sby is not on PATH")
    with pytest.raises(SystemExit):
        sr.main(["--ids", "g0_000"])
    assert "sby is not on PATH" in capsys.readouterr().err
    assert env["client"].requests == []


def test_missing_base_url_exits_before_any_request(env, monkeypatch, capsys):
    monkeypatch.delenv("QWEN_BASE_URL")
    with pytest.raises(SystemExit):
        sr.main(["--ids", "g0_000"])
    assert "QWEN_BASE_URL" in capsys.readouterr().err
    assert env["client"].requests == []


def test_preflight_names_the_missing_prover(monkeypatch):
    monkeypatch.setattr(sr.shutil, "which", lambda name: None)
    assert "oss-cad-suite" in sr.preflight()


def test_dry_makes_no_request_and_needs_no_endpoint(env, monkeypatch,
                                                    capsys):
    monkeypatch.delenv("QWEN_BASE_URL")

    def no_client():
        raise AssertionError("dry must not build a client")
    monkeypatch.setattr(sr, "make_client", no_client)
    sr.main(["--dry", "--per-gen", "1", "--gens", "0,1,2"])
    out = capsys.readouterr().out
    for row in (ROW_A, ROW_B, ROW_C):
        assert row["id"] in out
    assert "structural" in out and "lines=" in out
    assert '"fake_state"' in out          # the first design's prompt
    assert env["client"].requests == []
    assert not sr.LOG.exists()


# --- one request per design, n tries -------------------------------------

def test_one_request_per_design_asks_for_all_tries_at_once(env):
    sr.main(["--per-gen", "1", "--gens", "0,1"])
    reqs = env["client"].requests
    assert len(reqs) == 2
    for kw in reqs:
        assert kw["n"] == 8 and kw["max_tokens"] == 8000
        assert kw["model"] == "llm"
        assert "temperature" not in kw
        assert kw["messages"][0]["content"].startswith("Here is a")
    assert len(log_lines(sr.LOG)) == 16


def test_temperature_is_sent_and_logged_only_when_given(env):
    sr.main(["--ids", "g0_000", "--tries", "2"])
    assert {l["temperature"] for l in log_lines(sr.LOG)} == {None}
    sr.main(["--ids", "g1_000", "--tries", "2", "--temperature", "0.7",
             "--max-tokens", "3000"])
    kw = env["client"].requests[-1]
    assert kw["temperature"] == 0.7 and kw["max_tokens"] == 3000
    last = log_lines(sr.LOG)[-1]
    assert last["temperature"] == 0.7 and last["max_tokens"] == 3000


def test_every_try_is_one_full_log_line(env):
    env["client"].replies = [reply(), ("garbled", "length")]
    sr.main(["--ids", "g1_000", "--tries", "2"])
    lines = log_lines(sr.LOG)
    assert sorted(l["try"] for l in lines) == [1, 2]
    good = next(l for l in lines if l["try"] == 1)
    for key in ("id", "generation", "ext_type", "try", "served_model",
                "temperature", "max_tokens", "finish", "raw", "fake_state",
                "reasoning", "invariants", "proof", "fake_check", "kept",
                "timestamp", "usage"):
        assert key in good
    assert good["id"] == "g1_000" and good["generation"] == 1
    assert good["ext_type"] == "structural"
    assert good["served_model"] == SERVED
    assert good["finish"] == "stop" and good["raw"] == reply()
    assert good["fake_state"] == STATE and good["invariants"] == ["a == b"]
    assert good["proof"] == {"verdict": "NECESSARY",
                             "reason": "fake grader"}
    assert good["fake_check"]["verdict"] == "REAL"
    assert good["kept"] is True
    assert good["usage"]["prompt_tokens"] == 900
    cut = next(l for l in lines if l["try"] == 2)
    assert cut["finish"] == "length"
    assert cut["proof"]["verdict"] == "TRUNCATED"


@pytest.mark.parametrize("proof,fake,kept", [
    ("NECESSARY", "REAL", True),
    ("NECESSARY", "R_TRUE", False),
    ("NOT_PROVEN", "REAL", False),
    ("DECORATIVE", "NO_BREAK", False),
])
def test_a_try_is_kept_only_when_both_checks_pass(env, proof, fake, kept):
    env["proof"], env["fake"] = proof, fake
    sr.main(["--ids", "g0_000", "--tries", "1"])
    line = log_lines(sr.LOG)[0]
    assert line["proof"]["verdict"] == proof
    assert line["fake_check"]["verdict"] == fake
    assert line["kept"] is kept


def test_the_checker_gets_each_try_s_own_invariants(env):
    env["client"].replies = [reply(invariants=["a == b"]),
                             reply(invariants=["b <= a", "a <= 4'd3"])]
    sr.main(["--ids", "g0_000", "--tries", "2", "--workers", "1"])
    seen = sorted(tuple(row["invariants"]) for row, _ in env["checks"])
    assert seen == [("a == b",), ("b <= a", "a <= 4'd3")]
    for row, state in env["checks"]:
        assert row["invariants"] != ROW_A["invariants"]
        assert row["verilog"] == VERILOG and row["top_module"] == "m"
        assert state == STATE


def test_the_fake_state_is_checked_even_when_the_answer_is_wrong(env):
    env["proof"] = "NOT_PROVEN"
    sr.main(["--ids", "g0_000", "--tries", "1"])
    assert len(env["checks"]) == 1
    assert log_lines(sr.LOG)[0]["fake_check"]["verdict"] == "REAL"


def test_an_out_of_scope_answer_is_not_graded(env):
    env["client"].replies = [reply(invariants=["ghost == 1'b0"])]
    sr.main(["--ids", "g0_000", "--tries", "1"])
    line = log_lines(sr.LOG)[0]
    assert env["grades"] == []
    assert line["proof"]["verdict"] == "OUT_OF_SCOPE"
    assert "ghost" in line["proof"]["reason"]
    # an unknown name is a free wire to the checker, so it is not asked
    assert env["checks"] == []
    assert line["fake_check"]["verdict"] == "OUT_OF_SCOPE"
    assert line["kept"] is False


def test_an_unparseable_reply_is_logged_as_such_and_not_kept(env):
    env["client"].replies = ["I believe a == b is the invariant."]
    sr.main(["--ids", "g0_000", "--tries", "1"])
    line = log_lines(sr.LOG)[0]
    assert line["raw"] == "I believe a == b is the invariant."
    assert line["proof"]["verdict"] == "UNPARSEABLE"
    assert line["fake_check"] is None and line["kept"] is False
    assert line["invariants"] is None and line["fake_state"] is None
    assert env["grades"] == [] and env["checks"] == []


def test_a_reply_with_no_reasoning_is_graded_but_not_kept(env):
    env["client"].replies = [reply(reasoning="")]
    sr.main(["--ids", "g0_000", "--tries", "1"])
    line = log_lines(sr.LOG)[0]
    assert line["proof"]["verdict"] == "NECESSARY"
    assert line["fake_check"]["verdict"] == "REAL"
    assert "reasoning" in line["parse_problem"]
    assert line["kept"] is False


def test_a_grader_or_checker_crash_is_a_tool_failure_not_a_verdict(env,
                                                                   monkeypatch):
    def boom(*a):
        raise RuntimeError("yosys died")
    monkeypatch.setattr(sr, "grade", boom)
    monkeypatch.setattr(sr, "check", boom)
    sr.main(["--ids", "g0_000", "--tries", "1"])
    line = log_lines(sr.LOG)[0]
    assert line["proof"]["verdict"] == "grade_error"
    assert "yosys died" in line["proof"]["reason"]
    assert line["fake_check"]["verdict"] == "check_error"
    assert line["kept"] is False


def test_fewer_replies_than_asked_logs_only_those(env, capsys):
    env["client"].choices_cap = 3
    sr.main(["--ids", "g0_000"])
    assert len(log_lines(sr.LOG)) == 3
    assert "3 of 8" in capsys.readouterr().out


def test_a_failed_request_logs_nothing_and_the_run_goes_on(env, capsys):
    env["client"].fail = 1
    sr.main(["--ids", "g0_000,g1_000", "--tries", "2"])
    lines = log_lines(sr.LOG)
    assert {l["id"] for l in lines} == {"g1_000"}
    assert "tunnel closed" in capsys.readouterr().out


def test_three_failed_requests_in_a_row_stop_the_run(env):
    rows = [dict(ROW_A, id=f"g0_{i:03d}") for i in range(5)]
    sr.CORPUS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    env["client"].fail = 5
    sr.main(["--ids", ",".join(r["id"] for r in rows), "--tries", "1"])
    assert len(env["client"].requests) == 3
    assert not sr.LOG.exists() or log_lines(sr.LOG) == []


# --- resume and the log --------------------------------------------------

def test_resume_skips_fully_tried_designs_and_tops_up_the_rest(env):
    sr.LOG.parent.mkdir(parents=True)
    old = [{"id": "g0_000", "try": t, "kept": False} for t in range(1, 9)]
    old += [{"id": "g1_000", "try": t, "kept": False} for t in range(1, 4)]
    sr.LOG.write_text("".join(json.dumps(r) + "\n" for r in old))
    sr.main(["--ids", "g0_000,g1_000"])
    reqs = env["client"].requests
    assert len(reqs) == 1 and reqs[0]["n"] == 5
    new = log_lines(sr.LOG)[len(old):]
    assert {l["id"] for l in new} == {"g1_000"}
    assert sorted(l["try"] for l in new) == [4, 5, 6, 7, 8]


def test_many_workers_write_whole_lines(env):
    rows = [dict(ROW_A, id=f"g0_{i:03d}") for i in range(6)]
    sr.CORPUS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sr.main(["--ids", ",".join(r["id"] for r in rows), "--workers", "8"])
    text = sr.LOG.read_text()
    lines = [json.loads(l) for l in text.splitlines()]
    assert len(lines) == 48 and text.endswith("\n")
    for r in rows:
        tries = sorted(l["try"] for l in lines if l["id"] == r["id"])
        assert tries == list(range(1, 9))


def test_a_line_cut_short_by_a_killed_run_does_not_swallow_the_next(tmp_path):
    log = tmp_path / "self_reason.jsonl"
    log.write_text(json.dumps({"id": "a", "try": 1}) + "\n"
                   + '{"id": "b", "try": 1, "raw": "cut')
    sr.append(log, {"id": "c", "try": 1})
    assert [r["id"] for r in sr.read_log(log)] == ["a", "c"]


def test_a_run_drives_one_spinner_to_the_end(env, monkeypatch):
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

    monkeypatch.setattr(sr, "Spinner", FakeSpinner)
    sr.main(["--ids", "g0_000,g1_000", "--tries", "2", "--workers", "1"])
    assert "0/2 designs asked" in seen[0]
    assert "2/2 designs asked" in seen[-1]
    assert "4/4 tries graded" in seen[-1] and "4 kept" in seen[-1]


# --- report ---------------------------------------------------------------

def _line(id_, gen, proof, fake, kept=None):
    nec, real = proof == "NECESSARY", fake == "REAL"
    return {"id": id_, "generation": gen, "try": 1,
            "proof": {"verdict": proof, "reason": ""},
            "fake_check": None if fake is None else {"verdict": fake},
            "kept": (nec and real) if kept is None else kept,
            "served_model": SERVED}


def test_report_counts_and_the_two_by_two(tmp_path):
    log = tmp_path / "self_reason.jsonl"
    lines = [
        _line("a", 0, "NECESSARY", "REAL"),
        _line("a", 0, "NECESSARY", "R_TRUE"),          # right answer only
        _line("a", 0, "NOT_PROVEN", "REAL"),           # real state only
        _line("b", 0, "NOT_PROVEN", "NO_BREAK"),
        _line("c", 1, "NECESSARY", "REAL", kept=False),  # no reasoning
        _line("c", 1, "UNPARSEABLE", None),
        _line("d", 1, "NECESSARY", "REAL"),
        _line("d", 1, "OUT_OF_SCOPE", "OUT_OF_SCOPE"),
    ]
    log.write_text("".join(json.dumps(l) + "\n" for l in lines))
    rep = sr.report(log)
    assert rep["by_gen"][0] == {"designs": 2, "kept_designs": 1,
                                "kept": 1, "tries": 4}
    assert rep["by_gen"][1] == {"designs": 2, "kept_designs": 1,
                                "kept": 1, "tries": 4}
    assert rep["grid"] == {"necessary_real": 3, "necessary_not_real": 1,
                           "not_necessary_real": 1, "neither": 3}
    assert sum(rep["grid"].values()) == len(lines)
    assert rep["proof"] == {"NECESSARY": 4, "NOT_PROVEN": 2,
                            "UNPARSEABLE": 1, "OUT_OF_SCOPE": 1}
    assert rep["fake"]["REAL"] == 4 and rep["fake"]["not run"] == 1
    assert rep["kept"] == 2 and rep["tries"] == 8
    only_a = sr.report(log, ids=["a"])
    assert only_a["tries"] == 3


def test_report_counts_timeouts_and_crashes_apart_from_verdicts(tmp_path,
                                                                capsys):
    """A timeout is not a wrong answer and a checker crash is not wrong
    reasoning; the grid cannot tell them apart, so they are counted."""
    log = tmp_path / "self_reason.jsonl"
    timed_out = _line("a", 0, "NOT_PROVEN", "REAL")
    timed_out["proof"]["reason"] = ("with-invariants grade is TIMEOUT, not "
                                    "PROVEN")
    malformed = _line("a", 0, "NOT_PROVEN", "REAL")
    malformed["proof"]["reason"] = "with-invariants grade is ERROR, not PROVEN"
    # induction already failed: the answer is too weak, a real "no"
    too_weak = _line("a", 0, "NOT_PROVEN", "REAL")
    too_weak["proof"].update(reason=timed_out["proof"]["reason"],
                             induction_failed=True)
    lines = [timed_out, malformed, too_weak,
             _line("b", 0, "INCONCLUSIVE", "REAL"),
             _line("b", 0, "grade_error", "check_error"),
             _line("c", 0, "NECESSARY", "TIMEOUT"),
             _line("c", 0, "NECESSARY", "R_TRUE"),
             _line("c", 0, "UNPARSEABLE", None)]
    log.write_text("".join(json.dumps(l) + "\n" for l in lines))
    rep = sr.report(log)
    assert rep["tool_failures"] == {"proof": 3, "fake": 2, "tries": 4}
    out = capsys.readouterr().out
    assert "timeouts and crashes" in out


def test_grade_records_that_induction_failed_before_the_timeout(
        monkeypatch):
    """On a big design a too-weak answer times out in the base case after
    induction has already failed; that is a verdict, not a tool failure."""
    import oracle
    from oracle.types import (GradeResult, NecessityVerdict, RunEvidence,
                              Tier, TripleResult)

    def fake(text, **kw):
        ev = RunEvidence(mode="prove", rc=8, depth=20, engine="smtbmc",
                         duration_s=120.0, workdir=Path("."), log_excerpt="",
                         notes=notes)
        return TripleResult(NecessityVerdict.NOT_PROVEN,
                            "with-invariants grade is TIMEOUT, not PROVEN",
                            with_invariants=GradeResult(Tier.TIMEOUT, "",
                                                        [ev]))
    monkeypatch.setattr(oracle, "grade_triple_generated", fake)
    notes = ["induction_failed_before_timeout: induction returned FAIL"]
    assert sr.grade(ROW_A, ["a == b"])["induction_failed"] is True
    notes = []
    assert "induction_failed" not in sr.grade(ROW_A, ["a == b"])


def test_report_flag_makes_no_request(env, monkeypatch, capsys):
    monkeypatch.delenv("QWEN_BASE_URL")
    sr.LOG.parent.mkdir(parents=True)
    sr.LOG.write_text(json.dumps(_line("g0_000", 0, "NECESSARY", "R_TRUE"))
                      + "\n")
    sr.main(["--report"])
    out = capsys.readouterr().out
    assert env["client"].requests == []
    assert "right answer, wrong reasoning" in out
    assert "NECESSARY" in out and "R_TRUE" in out


def test_a_run_ends_with_the_report(env, capsys):
    sr.main(["--ids", "g0_000", "--tries", "2"])
    assert "right answer, wrong reasoning" in capsys.readouterr().out


# --- selection ------------------------------------------------------------

def _corpus():
    rows = []
    for gen in range(7):
        for k in range(10):
            ext = None if gen == 0 else ("structural", "distractor",
                                         "second")[k % 3]
            rows.append(dict(ROW_A, id=f"g{gen}_{k:03d}", generation=gen,
                             ext_type=ext))
    return rows


def test_stratified_selection_is_fixed_by_the_seed_and_covers_each_gen():
    rows = _corpus()
    gens = (0, 1, 2, 3, 4, 5, 6)
    one = sr.select(rows, per_gen=7, gens=gens, seed=0)
    two = sr.select(_corpus(), per_gen=7, gens=gens, seed=0)
    assert [r["id"] for r in one] == [r["id"] for r in two]
    assert len(one) == 49
    for g in gens:
        assert sum(r["generation"] == g for r in one) == 7
    picked = sr.select(rows, per_gen=2, gens=(1, 4), seed=3)
    assert sorted({r["generation"] for r in picked}) == [1, 4]


def test_ids_pick_exactly_those_designs_in_order():
    rows = _corpus()
    picked = sr.select(rows, ids="g3_001,g0_002")
    assert [r["id"] for r in picked] == ["g3_001", "g0_002"]
    with pytest.raises(SystemExit):
        sr.select(rows, ids="g3_001,nope")


# --- the real grader and checker, when the prover is here -----------------

TOY = """\
module toy (input wire clk, input wire i_ce);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    always @(posedge clk) if (i_ce) begin
        a <= a + 3'd1;
        b <= b + 3'd1;
    end
    always @(posedge clk) assert (!(a == 3'd3) || (b == 3'd3));
endmodule
"""


@pytest.mark.skipif(shutil.which("sby") is None,
                    reason="sby not on PATH - source oss-cad-suite first")
def test_real_tools_keep_a_right_try_and_judge_the_state_by_its_own_answer(
        tmp_path, monkeypatch):
    pytest.importorskip("cti_check")
    # the grader keeps its work folders; keep them out of runs/
    monkeypatch.setattr(sr, "GRADE_KWARGS",
                        dict(sr.GRADE_KWARGS, workdir_root=tmp_path))
    row = {"id": "toy", "generation": 0, "verilog": TOY, "top_module": "toy",
           "clock": "clk", "antecedents": [], "sanity_covers": [],
           "invariants": ["a <= 3'd7"]}
    state = [{"signal": "a", "value": "3'd2"},
             {"signal": "b", "value": "3'd0"}]
    right = json.dumps({"fake_state": state, "reasoning": "a=2, b=0.",
                        "invariants": ["a == b"]})
    out = sr.judge(row, right, "stop")
    assert out["proof"]["verdict"] == "NECESSARY", out["proof"]
    assert out["fake_check"]["verdict"] == "REAL", out["fake_check"]
    assert out["kept"] is True
    # same state, an answer that is true there: judged by that answer
    true_there = json.dumps({"fake_state": state, "reasoning": "x",
                             "invariants": ["a <= 3'd5"]})
    out = sr.judge(row, true_there, "stop")
    assert out["fake_check"]["verdict"] == "R_TRUE", out["fake_check"]
    assert out["kept"] is False
