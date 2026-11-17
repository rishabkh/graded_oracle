"""The design checker on another machine must judge like the laptop's
(16 Nov 2026, Rishab: "alright let's do it", change 7 of 7).

Making designs moves to the cluster: 150 growth steps took 4.5 hours of
the laptop on 15 Nov, so the 5,000 of the big run would take about six
days. The cluster runs another build of the same tools on another kind of
processor, so its verdicts count only after designs the laptop judged,
replayed there, all come back with the same verdict (as
initiator/ebmc_calibrate.py did for EBMC on 3 Nov):
  - the laptop chooses the designs from a run's log: passing designs
    judged well under the time limit, spread from quick to slow, plus
    designs the freeze check turned away and designs whose facts were
    false; never one that ran out of time, which a faster or slower
    machine could flip;
  - it judges them again with today's code and saves what it got;
  - any machine replays them and stops with an error on any difference;
  - on request it also asks the model one tiny question, to show that
    the machine reaches it.
The checker and the model are faked here: no sby, no calls, no money."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import checker_calibrate as cc                                   # noqa: E402
import freeze_check                                              # noqa: E402


def design(name):
    return json.dumps({"top_module": name, "verilog": f"module {name}; "
                       "endmodule", "invariants": ["1"]})


def round_(verdict, name, seconds=1.0, reason="because"):
    return {"verdict": verdict, "reason": reason, "grade_wall_s": seconds,
            "design": design(name)}


def row(attempt, verdict="NECESSARY", seconds=5.0, history=None,
        reason="proved"):
    name = f"m{attempt}"
    return {"attempt": attempt, "scope": "globally", "verdict": verdict,
            "reason": reason, "grade_wall_s": seconds,
            "raw_json": design(name),
            "history": history if history is not None
            else [round_(verdict, name, seconds, reason)]}


TIMED_OUT = "with-invariants grade is TIMEOUT, not PROVEN"
FALSE = "with-invariants grade is FALSE, not PROVEN"


# --- which designs can be replayed ------------------------------------------------

def test_a_passing_design_judged_well_under_the_limit_can_be_replayed():
    [c] = cc.candidates([row(3, seconds=12.5)])
    assert c == {"key": "attempt_3_final", "kind": "passing",
                 "raw_json": design("m3"), "scope": "globally",
                 "log_verdict": "NECESSARY", "log_seconds": 12.5}


def test_a_slow_design_or_one_that_ran_out_of_time_is_never_replayed():
    rows = [row(1, seconds=61.0),
            row(2, "NOT_PROVEN", 305.0, reason=TIMED_OUT)]
    assert cc.candidates(rows, max_time=60) == []


def test_designs_turned_away_in_earlier_rounds_can_be_replayed_too():
    history = [round_("FREEZES", "froze", 0.0, "froze: ..."),
               round_("NOT_PROVEN", "false_fact", 4.0, FALSE),
               round_("NOT_PROVEN", "slow", 301.0, TIMED_OUT),
               round_("NECESSARY", "m7", 9.0)]
    cands = cc.candidates([row(7, seconds=9.0, history=history)])
    assert [(c["key"], c["kind"], c["log_verdict"]) for c in cands] == [
        ("attempt_7_final", "passing", "NECESSARY"),
        ("attempt_7_round_0", "frozen", "FREEZES"),
        ("attempt_7_round_1", "failing", "NOT_PROVEN")]
    assert json.loads(cands[1]["raw_json"])["top_module"] == "froze"


def test_the_choice_is_spread_from_quick_to_slow_and_fixed():
    rows = [row(i, seconds=float(i)) for i in range(1, 41)]
    chosen = cc.select(cc.candidates(rows), passing=5, frozen=2, failing=2)
    assert [c["log_seconds"] for c in chosen] == [1.0, 10.0, 20.0, 30.0, 40.0]
    again = cc.select(cc.candidates(list(reversed(rows))), passing=5,
                      frozen=2, failing=2)
    assert again == chosen                    # whatever order the log is in


def test_each_kind_is_taken_up_to_its_count():
    history = [round_("FREEZES", "f", 0.0), round_("NECESSARY", "m", 2.0)]
    rows = [row(i, seconds=2.0, history=history) for i in range(6)]
    chosen = cc.select(cc.candidates(rows), passing=3, frozen=2, failing=4)
    kinds = [c["kind"] for c in chosen]
    assert kinds.count("passing") == 3 and kinds.count("frozen") == 2
    assert kinds.count("failing") == 0


# --- judging them again -----------------------------------------------------------

def fake_judge(verdicts):
    """Answers by the design's module name; notes what it was asked."""
    seen = []

    def judge(raw_json, stop, scope, grade=True, label="", proof_root=None,
              spin=True):
        name = json.loads(raw_json)["top_module"]
        seen.append({"name": name, "scope": scope, "spin": spin,
                     "freeze_check_on": freeze_check.ON,
                     "proof_root": proof_root})
        return {"verdict": verdicts.get(name, "NECESSARY"),
                "grade_wall_s": 1.0}
    judge.seen = seen
    return judge


def test_a_replay_judges_every_design_with_the_freeze_check_on(tmp_path):
    items = cc.candidates([row(1), row(2)])
    judge = fake_judge({})
    assert freeze_check.ON is False
    got = cc.replay(items, slots=2, judge=judge, work=tmp_path)
    assert [g["verdict"] for g in got] == ["NECESSARY", "NECESSARY"]
    assert all(s["freeze_check_on"] and not s["spin"] for s in judge.seen)
    assert {s["proof_root"] for s in judge.seen} == {
        tmp_path / "attempt_1_final", tmp_path / "attempt_2_final"}
    assert freeze_check.ON is False            # put back afterwards


def test_making_the_file_saves_what_this_machine_judged_today(tmp_path):
    log = tmp_path / "run.jsonl"
    log.write_text("".join(json.dumps(r) + "\n"
                           for r in [row(1), row(2, seconds=3.0)]))
    out = tmp_path / "calibration.jsonl"
    # today's code turns design m2 away, though the log says it passed
    cc.make(log, out, judge=fake_judge({"m2": "FREEZES"}), slots=1)
    saved = [json.loads(l) for l in out.read_text().splitlines()]
    assert [(s["key"], s["verdict"], s["log_verdict"]) for s in saved] == [
        ("attempt_1_final", "NECESSARY", "NECESSARY"),
        ("attempt_2_final", "FREEZES", "NECESSARY")]
    assert all("seconds" in s and "raw_json" in s for s in saved)


def test_a_file_that_is_already_there_is_not_written_over(tmp_path):
    log = tmp_path / "run.jsonl"
    log.write_text(json.dumps(row(1)) + "\n")
    out = tmp_path / "calibration.jsonl"
    out.write_text("kept\n")
    with pytest.raises(SystemExit) as stop:
        cc.make(log, out, judge=fake_judge({}), slots=1)
    assert "already there" in str(stop.value)
    assert out.read_text() == "kept\n"


def saved(tmp_path, *pairs):
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(
        {"key": f"attempt_{i}_final", "kind": "passing",
         "raw_json": design(name), "scope": "globally", "verdict": verdict,
         "seconds": 2.0, "log_verdict": verdict, "log_seconds": 2.0}) + "\n"
        for i, (name, verdict) in enumerate(pairs)))
    return path


def test_a_machine_that_judges_the_same_passes(tmp_path, capsys):
    path = saved(tmp_path, ("a", "NECESSARY"), ("b", "FREEZES"))
    code = cc.main(["--file", str(path), "--slots", "2"],
                   judge=fake_judge({"b": "FREEZES"}))
    assert code == 0
    out = capsys.readouterr().out
    assert "2 of 2 designs judged the same" in out


def test_any_difference_is_named_and_fails(tmp_path, capsys):
    path = saved(tmp_path, ("a", "NECESSARY"), ("b", "NECESSARY"))
    code = cc.main(["--file", str(path)], judge=fake_judge(
        {"b": "NOT_PROVEN"}))
    assert code == 1
    out = capsys.readouterr().out
    assert "1 of 2 designs judged the same" in out
    assert "attempt_1_final: NECESSARY on the laptop, NOT_PROVEN here" in out


def test_with_no_file_to_replay_nothing_runs(tmp_path):
    with pytest.raises(SystemExit) as stop:
        cc.main(["--file", str(tmp_path / "missing.jsonl")],
                judge=fake_judge({}))
    assert "--make" in str(stop.value)


# --- reaching the model -----------------------------------------------------------

def test_the_model_is_asked_only_on_request(tmp_path, capsys):
    path = saved(tmp_path, ("a", "NECESSARY"))
    asked = []

    def ask():
        asked.append(1)
        return "ready"
    assert cc.main(["--file", str(path)], judge=fake_judge({}), ask=ask) == 0
    assert asked == []
    assert cc.main(["--file", str(path), "--ask-model"],
                   judge=fake_judge({}), ask=ask) == 0
    assert asked == [1]
    assert "the model answered" in capsys.readouterr().out


def test_a_machine_that_cannot_reach_the_model_fails_and_says_why(tmp_path,
                                                                 capsys):
    path = saved(tmp_path, ("a", "NECESSARY"))

    def ask():
        raise ConnectionError("Connection error.")
    code = cc.main(["--file", str(path), "--ask-model"],
                   judge=fake_judge({}), ask=ask)
    assert code == 1
    out = capsys.readouterr().out
    assert "the model was not reached: ConnectionError: Connection error." \
        in out
    # the checker's result is still reported
    assert "1 of 1 designs judged the same" in out
