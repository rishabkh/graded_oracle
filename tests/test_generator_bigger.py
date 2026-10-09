"""The generator, made ready for bigger designs and kept-everything runs
(9 Oct 2026). The model and the checker are faked: no calls, no sby."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import distill_log as dl                                         # noqa: E402
import prompts as P                                              # noqa: E402
from tests.test_initiator_pipeline import TRIPLE, Fake, _Graded  # noqa: E402


# --- the prompts no longer hold designs small -------------------------------

def test_the_prompts_no_longer_ask_for_small_designs():
    for text in (P.SYSTEM_PROMPT, P.USER_TEMPLATE, P.IDEA_SYSTEM,
                 P.IDEA_TEMPLATE, P.IMPLEMENT_TEMPLATE):
        assert "small" not in text.lower()
    assert "Keep it simple" not in P.IDEA_TEMPLATE
    assert "250 words" not in P.IDEA_TEMPLATE


def test_the_line_cap_is_gone_and_rows_say_so():
    assert "under 200 lines" not in P.SYSTEM_PROMPT
    assert P.prompt_version() == "nocap"


def test_the_design_step_of_two_step_sees_the_size(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"])
    scales = tmp_path / "scales.txt"
    scales.write_text("Make it 64-bit with 128 entries.\n")
    f.run(mode="two-step", scales_path=scales)
    implement_question = f.prompts[1][1]
    assert "Make it 64-bit with 128 entries." in implement_question


# --- answer length and checker settings -------------------------------------

def test_answers_may_run_to_the_models_limit():
    import run as R
    assert R.MAX_TOKENS == 128000


def test_designs_are_checked_with_the_new_limits():
    import run as R
    assert R.GRADE_KWARGS == {"timeout_s": 300, "pdr_timeout_s": 120,
                              "cover_depth": "auto"}


# --- every check keeps its proof folder next to the run log -----------------

def _capture_grades(f, verdicts):
    from oracle import NecessityVerdict
    roots, left = [], list(verdicts)

    def grade(raw, **kw):
        roots.append(kw.get("workdir_root"))
        v = left.pop(0) if left else "NECESSARY"
        return _Graded(NecessityVerdict[v], f"because {v}")
    f.R.grade_triple_generated = grade
    return roots


def test_each_check_gets_its_own_proof_folder(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, [])
    roots = _capture_grades(f, ["NOT_PROVEN", "NECESSARY"])
    monkeypatch.setattr(f.R, "grade_triple_generated",
                        f.R.grade_triple_generated)
    [row], ex = f.run(repairs=2)
    base = f"log_proofs/{row['run_id']}_a000"
    assert row["proof_dir"] == base
    assert [h["proof_dir"] for h in row["history"]] == [f"{base}/r0",
                                                        f"{base}/r1"]
    assert [Path(r) for r in roots] == [tmp_path / base / "r0",
                                        tmp_path / base / "r1"]
    assert all(e["metadata"]["proof_dir"].startswith(base) for e in ex)


def test_the_worked_examples_are_checked_in_the_runs_own_folder(
        monkeypatch, tmp_path):
    import run as R
    seen = []
    from oracle import NecessityVerdict

    def grade(raw, **kw):
        seen.append(kw)
        return _Graded(NecessityVerdict.NECESSARY, "ok")
    monkeypatch.setattr(R, "grade_triple_generated", grade)
    R.assert_exemplar_pool({"A": {"x": 1}}, proof_root=tmp_path / "p")
    assert Path(seen[0]["workdir_root"]) == tmp_path / "p" / "A"
    assert seen[0]["timeout_s"] == 300


# --- a design that only ran out of time ------------------------------------

@pytest.mark.parametrize("verdict, reason", [
    ("NOT_PROVEN", "with-invariants grade is TIMEOUT, not PROVEN"),
    ("INCONCLUSIVE", "without-invariants grade is TIMEOUT - necessity not "
                     "established")])
def test_a_timed_out_check_is_not_repaired_and_saved_as_an_error(
        monkeypatch, tmp_path, verdict, reason):
    from oracle import NecessityVerdict
    f = Fake(monkeypatch, tmp_path, [])
    monkeypatch.setattr(f.R, "grade_triple_generated",
                        lambda raw, **kw: _Graded(NecessityVerdict[verdict],
                                                  reason))
    [row], ex = f.run(repairs=2)
    assert len(f.prompts) == 1                     # no paid repair call
    assert row["repair_attempts"] == 0
    assert "out of time" in row["repair_skipped"]
    [gen] = ex
    assert gen["outcome"] == "error"


def test_timeouts_and_inconclusive_count_as_errors_not_failures():
    assert dl.outcome_of("NOT_PROVEN",
                         "with-invariants grade is TIMEOUT, not PROVEN") == "error"
    assert dl.outcome_of("INCONCLUSIVE") == "error"
    assert dl.outcome_of("NOT_PROVEN", "with-invariants grade is "
                         "NOT_INDUCTIVE, not PROVEN") == "fail"


# --- what a repair question is told -----------------------------------------

def test_repair_notes_leave_out_cover_witnesses(tmp_path):
    import run as R
    result = {"with_invariants": {"reason": "r", "runs": [
        {"mode": "prove", "rc": 4, "trace_text": "CTI TRACE"},
        {"mode": "cover", "rc": 0, "trace_text": "COVER WITNESS"}]}}
    notes = R.checker_notes("NOT_PROVEN", "x", result)
    assert "CTI TRACE" in notes and "COVER WITNESS" not in notes


def test_repair_notes_read_errors_from_the_full_log(tmp_path):
    import run as R
    log = tmp_path / "logfile.txt"
    log.write_text("design.sv:3: ERROR: syntax error\n" + "ok\n" * 100)
    result = {"with_invariants": {"runs": [
        {"mode": "prove", "rc": 1, "log_path": str(log),
         "log_excerpt": "ok\n" * 40}]}}
    assert "syntax error" in R.checker_notes("NOT_PROVEN", "x", result)


def test_repair_notes_have_room_for_bigger_counterexamples():
    import run as R
    assert R.NOTES_CAP == 12000


# --- the scope gate on bigger designs ----------------------------------------

def test_a_guard_far_above_the_assert_still_counts():
    import run as R
    v = ("module m (input clk);\n"
         "  always @(posedge clk) begin\n"
         "    if (armed) begin\n"
         "      a <= 1;\n      b <= 2;\n      c <= 3;\n      d <= 4;\n"
         "      assert (x == y);\n"
         "    end\n  end\nendmodule\n")
    assert R._property_is_conditional(v)


def test_an_or_on_the_next_line_of_the_assert_counts():
    import run as R
    v = ("module m (input clk);\n"
         "  always @(*)\n"
         "    assert (!armed\n"
         "            || x == y);\nendmodule\n")
    assert R._property_is_conditional(v)


def test_a_plain_assert_is_still_unconditional():
    import run as R
    v = ("module m (input clk);\n"
         "  always @(posedge clk)\n"
         "    assert (x == y);\nendmodule\n")
    assert not R._property_is_conditional(v)


# --- a crash keeps what was already done --------------------------------------

def test_a_crash_in_a_repair_round_keeps_the_earlier_rounds(monkeypatch,
                                                            tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN"])
    calls = {"n": 0}
    real = f.R.call_model

    def flaky(user_msg, model=None, system=None, schema=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("network went away")
        return real(user_msg, model=model, system=system, schema=schema)
    monkeypatch.setattr(f.R, "call_model", flaky)
    f.R.run_pipeline(1, cmd="pilot", log_path=f.log, distill_path=f.distill,
                     workers=1, repairs=2)
    [row] = [json.loads(l) for l in f.log.read_text().splitlines()]
    assert "ConnectionError" in row["error"]
    assert [h["verdict"] for h in row["history"]] == ["NOT_PROVEN"]
    assert "Traceback" in row["traceback"]


# --- Formal Disco's rules for the saved examples ----------------------------

def test_implement_is_the_first_answer_with_its_own_result(monkeypatch,
                                                           tmp_path):
    """formal-disco: an implement example is the design written from the
    idea, judged on its own; fixes are separate repair examples."""
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN", "NECESSARY"])
    [row], ex = f.run(mode="two-step", repairs=2)
    by = {e["prompt"]: e for e in ex}
    assert json.loads(by["implement"]["response"])["invariants"] == ["v1"]
    assert by["implement"]["outcome"] == "fail"
    assert by["idea"]["outcome"] == "fail"
    assert by["repair"]["outcome"] == "success"
    assert row["verdict"] == "NECESSARY"              # the run still kept it
    assert json.loads(row["raw_json"])["invariants"] == ["v2"]


def test_an_idea_is_saved_exactly_as_the_model_sent_it(monkeypatch,
                                                       tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"])
    _, ex = f.run(mode="two-step")
    idea = next(e for e in ex if e["prompt"] == "idea")
    assert json.loads(idea["response"]) == {"idea": "IDEA TEXT"}
