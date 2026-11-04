"""Re-grading saved one-try answers with the benchmark authors' rule.

Our one-try score checks all of an answer's facts together, so one false
fact sinks an answer that also holds the right one. The authors' scorer
(evaluate_non_agentic -> evaluate_subsets) checks each fact alone, drops
the ones it can refute, and tries the rest together. Every one-try number
we have is on our rule; the paper's are on theirs (checked 31 Oct 2026).
EBMC is faked here, so these pin which answers get re-checked and how
the results are counted."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import regrade_authors_rule as rg                                # noqa: E402

MODEL = ("/n/netscratch/amin_lab/Lab/x/hardware-formal-disco/runs/"
         "good_weak_open_source_unlabeled_2026-10-29_seed2/merged")


def row(bench="a.sv", lemmas=("good",), verdict="NOT_SOLVED", run_id="r1",
        set_="hard", served=MODEL, **kw):
    r = {"run_id": run_id, "solver": "qwen", "bench": bench, "set": set_,
         "served_model": served, "file_judgeable": True, "verdict": verdict}
    if lemmas is not None:
        r["lemmas"] = list(lemmas)
    r.update(kw)
    return r


def fake_run(table, calls=None):
    """table: frozenset(lemmas) -> mode -> verdict; anything else is
    INCONCLUSIVE, which is what EBMC says of a true fact that needs
    others to be proved."""
    def run(bench, lemmas, mode, workdir, buechi=False):
        if calls is not None:
            calls.append((Path(bench).name, tuple(lemmas), mode))
        return {"verdict": table.get(frozenset(lemmas), {}).get(
            mode, "INCONCLUSIVE")}
    return run


GOOD_ALONE = {"correctness_bounded": "PROVEN",
              "one_inductive_with_prop": "PROVEN"}


def test_the_right_fact_beside_a_false_one_solves_on_their_rule_only(tmp_path):
    table = {frozenset(["good"]): GOOD_ALONE,
             frozenset(["bad"]): {"correctness_bounded": "CEX"}}
    g = rg.regrade_row(row(lemmas=["good", "bad"]), fake_run(table),
                       tmp_path, tmp_path)
    assert g["authors_solved"] is True
    assert g["ours_solved"] is False


def test_facts_that_only_close_the_proof_together_are_tried_together(tmp_path):
    # each fact is true but not inductive alone (INCONCLUSIVE); the
    # authors keep both and try them as a set
    table = {frozenset(["a"]): {"correctness_bounded": "PROVEN"},
             frozenset(["b"]): {"correctness_bounded": "PROVEN"},
             frozenset(["a", "b"]): {"one_inductive_with_prop": "PROVEN"}}
    g = rg.regrade_row(row(lemmas=["a", "b"], verdict="PROVEN"),
                       fake_run(table), tmp_path, tmp_path)
    assert g["authors_solved"] is True and g["ours_solved"] is True


def test_a_fact_whose_30_step_check_runs_out_of_time_is_dropped(tmp_path):
    # the commonest per-fact result on 17 Oct (2,003 rows): their rule
    # drops the fact, ours never ran that check
    table = {frozenset(["slow"]): {"correctness_bounded": "TIMEOUT"}}
    g = rg.regrade_row(row(lemmas=["slow"], verdict="PROVEN"),
                       fake_run(table), tmp_path, tmp_path)
    assert g["authors_solved"] is False and g["ours_solved"] is True


def test_each_fact_keeps_its_three_verdicts(tmp_path):
    table = {frozenset(["bad"]): {"correctness_bounded": "CEX"}}
    g = rg.regrade_row(row(lemmas=["bad"]), fake_run(table), tmp_path,
                       tmp_path)
    assert g["facts"] == [{"lemma": "bad", "correct": False,
                           "correctness": "CEX",
                           "one_induction": "INCONCLUSIVE",
                           "with_prop": "INCONCLUSIVE"}]


def test_no_answer_or_an_unjudgeable_file_costs_no_checks(tmp_path):
    calls = []
    run = fake_run({}, calls)
    empty = rg.regrade_row(row(lemmas=None, verdict="NO_ANSWER"), run,
                           tmp_path, tmp_path)
    unjudgeable = rg.regrade_row(row(file_judgeable=False), run, tmp_path,
                                 tmp_path)
    assert calls == []
    assert empty["authors_solved"] is False and empty["judgeable"] is True
    assert unjudgeable["authors_solved"] is False
    assert unjudgeable["judgeable"] is False


def test_only_one_try_answers_are_regraded():
    # the feedback-loop jobs write their rounds to the same log
    rows = [row(), row(round=0, feedback_mode="per_lemma"),
            row(feedback_mode="none"), row(served=None)]
    assert rg.one_try_rows(rows, match="_seed") == [row()]


def test_only_the_models_asked_for_are_regraded():
    older = row(served="/n/x/runs/v2/merged")
    assert rg.one_try_rows([row(), older], match="_seed") == [row()]


def test_a_model_is_named_by_its_folder():
    assert rg.model_name(MODEL) == \
        "good_weak_open_source_unlabeled_2026-10-29_seed2"
    assert rg.model_name(None) is None


def test_the_last_three_complete_runs_of_each_set_are_kept():
    # a job bumped mid-pass leaves a run with files missing; the job's
    # restart then writes complete runs of its own
    sizes = {"hard": 2}
    rows = [row("a.sv", run_id="r0")]                          # cut short
    for rid in ("r1", "r2", "r3", "r4"):
        rows += [row("a.sv", run_id=rid), row("b.sv", run_id=rid)]
    kept = rg.pick_runs(rows, sizes, keep=3)
    assert sorted({r["run_id"] for r in kept}) == ["r2", "r3", "r4"]
    assert len(kept) == 6


def test_runs_are_kept_per_model_and_per_set():
    sizes = {"hard": 1, "main_experiment": 1}
    other = MODEL.replace("seed2", "seed0")
    rows = [row(run_id="h1"), row(run_id="e1", set_="main_experiment"),
            row(run_id="h2", served=other)]
    assert len(rg.pick_runs(rows, sizes, keep=3)) == 3


def test_a_restarted_regrade_carries_on_where_it_stopped(tmp_path):
    out = tmp_path / "regraded.jsonl"
    done = row("a.sv")
    out.write_text(json.dumps(dict(rg.key_fields(done), authors_solved=True,
                                   ours_solved=False, judgeable=True,
                                   facts=[])) + "\n")
    calls = []
    graded = rg.grade_all([done, row("b.sv")], out, fake_run({}, calls),
                          tmp_path, tmp_path)
    assert {c[0] for c in calls} == {"b.sv"}
    assert len(graded) == 2
    assert len(out.read_text().splitlines()) == 2


def test_a_check_already_made_on_a_design_is_not_run_again(tmp_path):
    calls = []
    rows = [row("a.sv", run_id="r1"), row("a.sv", run_id="r2")]
    rg.grade_all(rows, tmp_path / "o.jsonl", fake_run({}, calls), tmp_path,
                 tmp_path)
    assert len(calls) == len(set(calls))
    assert calls                                    # the first one did run


def test_the_same_fact_on_two_designs_is_checked_on_each(tmp_path):
    calls = []
    rows = [row("a.sv"), row("b.sv")]
    rg.grade_all(rows, tmp_path / "o.jsonl", fake_run({}, calls), tmp_path,
                 tmp_path)
    assert {c[0] for c in calls} == {"a.sv", "b.sv"}


def graded(model_folder, set_, run_id, ours, authors, judgeable=True):
    return {"model": model_folder, "set": set_, "run_id": run_id,
            "ours_solved": ours, "authors_solved": authors,
            "judgeable": judgeable}


def test_the_summary_counts_each_pass_under_both_rules():
    g = [graded("s_seed0", "hard", "r1", True, True),
         graded("s_seed0", "hard", "r1", False, True),
         graded("s_seed0", "hard", "r2", False, False),
         graded("s_seed0", "hard", "r2", False, False, judgeable=False)]
    s = rg.summarize(g)
    assert s["s_seed0"]["hard"] == {"ours": [1, 0], "authors": [2, 0],
                                    "judgeable": [2, 1]}


def test_the_table_averages_each_set_over_its_models():
    g = [graded("s_seed0", "hard", "r1", True, True),
         graded("s_seed1", "hard", "r2", False, True)]
    text = rg.table(rg.summarize(g))
    line = next(l for l in text.splitlines() if l.startswith("s "))
    assert "0.50" in line and "1.00" in line


TRAINING = ROOT / "training"


def test_the_cluster_job_runs_ebmc_with_the_compiler_it_was_built_with():
    job = (TRAINING / "regrade_authors_rule.sbatch").read_text()
    score = (TRAINING / "serve_and_score.sbatch").read_text()
    pin = "gcc/13.2.0-fasrc01"
    assert pin in job and pin in score
    assert 'EBMC_PATH="$HOME/bin/ebmc"' in job
    assert "--gres" not in job                       # no GPU needed
    assert "initiator/regrade_authors_rule.py" in job
