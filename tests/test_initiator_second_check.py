"""A starting design that ran out of time is checked once more with a
longer limit (16 Nov 2026, Rishab: "alright let's do it", change 5 of 7).

The extender has done this since 21 Oct 2026 (distractor.necessity_check,
900 s): in the 15 Nov pilot it checked 27 steps again and 11 of them then
proved. The initiator never did: 46 of the pilot's 300 starting designs
were lost to the 300 s limit, and a design that ran out of time is not
sent for repair either. So:
  - the rule for "this verdict came from the time limit" moves into the
    checker's own package, one copy for both (oracle.ran_out_of_time);
  - with --recheck-timeout-s S the initiator checks such a design once
    more with S seconds, its proofs in a "recheck" folder beside the
    first ones; the second verdict stands, the first is kept whole;
  - without the option nothing changes (0, as before).
The model and the checker are faked: no calls, no sby, no money."""
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))
sys.path.insert(0, str(ROOT / "extender"))

import oracle                                                    # noqa: E402
import run as R                                                  # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict,        # noqa: E402
                          RunEvidence, Tier, TripleResult)

# armed and conditional, so the scope gate passes whichever scope is drawn
TRIPLE = {"top_module": "m", "antecedents": ["armed"], "sanity_covers": [],
          "verilog": "module m(input clk);\n reg armed = 0;\n"
                     " always @(posedge clk) if (armed) assert (1);\nendmodule\n",
          "invariants": ["1"]}
RAW = json.dumps(TRIPLE)


def leg(tier):
    run = RunEvidence(mode="prove", rc=8 if tier is Tier.TIMEOUT else 0,
                      depth=20, engine="smtbmc yices", duration_s=1.0,
                      workdir=Path("job"), log_excerpt="", notes=[])
    return GradeResult(tier=tier, reason=tier.name.lower(), runs=[run])


def ran_out():
    return TripleResult(NecessityVerdict.NOT_PROVEN,
                        "with-invariants grade is TIMEOUT, not PROVEN",
                        with_invariants=leg(Tier.TIMEOUT))


def proved():
    return TripleResult(NecessityVerdict.NECESSARY,
                        "strengthening is load-bearing: PROVEN with "
                        "invariants, NOT_INDUCTIVE without",
                        with_invariants=leg(Tier.PROVEN),
                        without_invariants=leg(Tier.NOT_INDUCTIVE))


def false_fact():
    return TripleResult(NecessityVerdict.NOT_PROVEN,
                        "with-invariants grade is FALSE, not PROVEN",
                        with_invariants=leg(Tier.FALSE))


class Checker:
    """Gives its answers in turn and keeps the settings of every check."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []
        self.lock = threading.Lock()

    def __call__(self, payload, **kw):
        with self.lock:
            self.calls.append(kw)
            return self.answers.pop(0)


@pytest.fixture
def checker(monkeypatch):
    def use(*answers):
        c = Checker(*answers)
        monkeypatch.setattr(R, "grade_triple_generated", c)
        return c
    return use


@pytest.fixture
def second_check(monkeypatch):
    monkeypatch.setattr(R, "RECHECK_TIMEOUT_S", 900)


# --- one rule for both ------------------------------------------------------------

def test_the_checker_package_says_which_verdicts_came_from_the_time_limit():
    assert oracle.ran_out_of_time(ran_out())
    assert not oracle.ran_out_of_time(proved())
    assert not oracle.ran_out_of_time(false_fact())


def test_the_extender_uses_the_same_rule():
    import distractor
    assert distractor.ran_out_of_time is oracle.ran_out_of_time


# --- the initiator's check --------------------------------------------------------

def test_the_second_check_is_off_unless_the_run_asks():
    assert R.RECHECK_TIMEOUT_S == 0


def test_without_it_a_design_that_ran_out_of_time_is_checked_once(checker):
    c = checker(ran_out())
    j = R.judge(RAW, "ok", "globally", spin=False)
    assert j["verdict"] == "NOT_PROVEN" and len(c.calls) == 1
    assert "rechecked" not in j


def test_a_design_that_ran_out_of_time_is_checked_again_with_the_longer_limit(
        checker, second_check, tmp_path):
    c = checker(ran_out(), proved())
    j = R.judge(RAW, "ok", "globally", spin=False, proof_root=tmp_path / "r0")
    assert j["verdict"] == "NECESSARY"
    assert [k["timeout_s"] for k in c.calls] == [300, 900]
    # the second check's proofs sit beside the first ones
    assert c.calls[0]["workdir_root"] == tmp_path / "r0"
    assert c.calls[1]["workdir_root"] == tmp_path / "r0" / "recheck"
    # only the main limit grows, as in the extender
    assert c.calls[1]["pdr_timeout_s"] == c.calls[0]["pdr_timeout_s"]


def test_the_first_check_is_kept_whole(checker, second_check):
    checker(ran_out(), proved())
    j = R.judge(RAW, "ok", "globally", spin=False)
    assert j["rechecked"] is True and j["recheck_timeout_s"] == 900
    assert j["first_result"]["verdict"] == "NOT_PROVEN"
    assert "TIMEOUT" in j["first_result"]["reason"]
    assert j["first_result"]["result"]["with_invariants"]["tier"] is \
        Tier.TIMEOUT
    assert j["recheck_wall_s"] == j["grade_wall_s"]


def test_it_is_checked_again_once_never_twice(checker, second_check):
    c = checker(ran_out(), ran_out())
    j = R.judge(RAW, "ok", "globally", spin=False)
    assert j["verdict"] == "NOT_PROVEN" and len(c.calls) == 2
    assert j["rechecked"] is True


def test_a_verdict_that_did_not_come_from_the_time_limit_is_not_checked_again(
        checker, second_check):
    c = checker(false_fact())
    j = R.judge(RAW, "ok", "globally", spin=False)
    assert j["verdict"] == "NOT_PROVEN" and len(c.calls) == 1
    assert "rechecked" not in j


# --- through a run ----------------------------------------------------------------

def run_one(monkeypatch, tmp_path, repairs=1):
    def call_model(user_msg, model=None, system=None, schema=None):
        return RAW, {"input": 2, "output": 3}, "ok"
    monkeypatch.setattr(R, "call_model", call_model)
    monkeypatch.setattr(R, "assert_exemplar_pool", lambda ex, **kw: None)
    log = tmp_path / "log.jsonl"
    R.run_pipeline(1, cmd="pilot", log_path=log, workers=1, repairs=repairs,
                   distill_path=tmp_path / "distill.jsonl")
    return [json.loads(l) for l in log.read_text().splitlines()]


def test_a_design_saved_by_the_second_check_is_kept_and_the_log_says_so(
        checker, second_check, monkeypatch, tmp_path, capsys):
    checker(ran_out(), proved())
    [row] = run_one(monkeypatch, tmp_path)
    assert row["verdict"] == "NECESSARY" and row["repair_attempts"] == 0
    assert row["rechecked"] is True and row["recheck_timeout_s"] == 900
    assert row["first_result"]["verdict"] == "NOT_PROVEN"
    assert row["history"][0]["rechecked"] is True
    assert "checked again with a 900 s limit; the first check said " \
           "NOT_PROVEN" in capsys.readouterr().out


def test_a_design_that_runs_out_of_time_twice_is_still_not_repaired(
        checker, second_check, monkeypatch, tmp_path):
    c = checker(ran_out(), ran_out())
    [row] = run_one(monkeypatch, tmp_path, repairs=2)
    assert row["verdict"] == "NOT_PROVEN" and row["repair_attempts"] == 0
    assert "ran out of time" in row["repair_skipped"]
    assert len(c.calls) == 2


def test_a_run_without_repairs_keeps_what_the_second_check_saw(
        checker, second_check, monkeypatch, tmp_path):
    checker(ran_out(), proved())
    monkeypatch.setattr(R, "call_model", lambda *a, **kw: (
        RAW, {"input": 2, "output": 3}, "ok"))
    monkeypatch.setattr(R, "assert_exemplar_pool", lambda ex, **kw: None)
    log = tmp_path / "log.jsonl"
    R.run_attempts(1, cmd="pilot", log_path=log,
                   distill_path=tmp_path / "distill.jsonl")
    [row] = [json.loads(l) for l in log.read_text().splitlines()]
    assert row["verdict"] == "NECESSARY" and row["rechecked"] is True
    assert row["first_result"]["verdict"] == "NOT_PROVEN"


# --- the command line -------------------------------------------------------------

def test_the_pilot_can_switch_it_on():
    import inspect
    src = inspect.getsource(R.main)
    assert 'pilot.add_argument("--recheck-timeout-s"' in src
    assert "RECHECK_TIMEOUT_S = args.recheck_timeout_s" in src


@pytest.mark.parametrize("limit", ["300", "120", "-5"])
def test_a_second_limit_no_longer_than_the_first_is_refused(monkeypatch,
                                                            limit, capsys):
    monkeypatch.setattr(sys, "argv", ["run.py", "pilot", "--n", "1",
                                      "--recheck-timeout-s", limit])
    monkeypatch.setattr(R, "run_attempts", lambda *a, **kw: pytest.fail(
        "nothing may run"))
    with pytest.raises(SystemExit):
        R.main()
    assert "longer than the first check's 300 s" in capsys.readouterr().err
