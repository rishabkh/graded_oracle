"""Designs the automatic engine cannot prove alone are grown first (21 Oct
2026, Rishab: "make all 1-8 changes", item 4).

The checker already runs one automatic engine, abc pdr, on the proof
without helper facts whenever k-induction does not close there (the
"second opinion"). Measured 21 Oct 2026 from the logs: it proved the
property with no facts for 198 of the 252 starting designs of 9 Oct (79%)
and ran out of its 120 s for 36 (14%); for 15 of the 19 designs kept in
the 21 Oct trial it proved it alone. A real tool would need no
hand-written facts for those, and Gary (formal verification expert) said
engineers write facts by hand where the engines get stuck. So:
  - every row the corpus builder or promote makes says what the engine
    found alone (metrics.engine_alone: proved, timed_out, error or
    not_tried);
  - the batch run grows first the rows it ran out of time on, then those
    where it failed or was not tried (or that say nothing), then those it
    proved; within one kind the run's own order holds, an explicit start
    list keeps its order, a chased child still goes next, and
    --no-prefer-engine-hard turns the preference off;
  - the corpus build and every run report the share of each.
No model and no checker run here; the checker's results are written out."""
import json
import random
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import build_corpus                                             # noqa: E402
import distractor                                               # noqa: E402
import promote                                                  # noqa: E402
from frontier import order_frontier                             # noqa: E402
from oracle.grading import INDUCTION_FAIL_NOTE                  # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict,        # noqa: E402
                          RunEvidence, Tier, TripleResult)

CATALOG = json.loads((ROOT / "initiator" / "catalog" /
                      "kind_categories_2026-10-09.json").read_text())
_GROUP = {a["id"]: a["group"] for a in CATALOG["areas"]}


def kind_in(group):
    return next(k["name"] for k in CATALOG["kinds"]
                if _GROUP.get(k["category"]) == group)


# --- the checker's results, as a record holds them ---------------------------

def run(engine, rc, mode="prove", notes=()):
    return {"mode": mode, "engine": engine, "rc": rc, "depth": 20,
            "duration_s": 1.0, "notes": list(notes)}


INDUCTION = run("smtbmc yices", 4)   # k-induction did not close without facts


def graded(*without_runs, with_runs=(run("smtbmc yices", 0),)):
    """A record's "result" (asdict of the checker's TripleResult)."""
    return {"result": {"verdict": "NECESSARY",
                       "with_invariants": {"runs": list(with_runs)},
                       "without_invariants": {"runs": list(without_runs)}}}


def pdr(rc):
    """The second opinion on the proof without facts."""
    return run("abc pdr", rc,
               notes=[f"pdr_second_opinion: rc={rc}"])


# --- what the engine found alone ---------------------------------------------

@pytest.mark.parametrize("rc, found", [
    (0, "proved"),            # proved the property with no facts
    (8, "timed_out"),         # ran out of its time
    (None, "timed_out"),      # stopped by the outer guard
    (16, "error"),            # the tool failed
    (2, "error"),             # refuted: any other rc is an error
    (1, "error")])
def test_what_the_engine_found_alone(rc, found):
    assert build_corpus.engine_alone(graded(INDUCTION, pdr(rc))) == found


@pytest.mark.parametrize("record", [
    # k-induction ran out of time first (already failing): no pdr run
    graded(run("smtbmc yices", 8, notes=[INDUCTION_FAIL_NOTE])),
    {},                                               # no result at all
    {"result": None},
    {"result": {"verdict": "NOT_PROVEN", "with_invariants": {"runs": []},
                "without_invariants": None}},
])
def test_an_engine_that_never_ran_was_not_tried(record):
    assert build_corpus.engine_alone(record) == "not_tried"


def test_only_the_proof_without_facts_counts():
    record = graded(INDUCTION, with_runs=[run("smtbmc yices", 4), pdr(0)])
    assert build_corpus.engine_alone(record) == "not_tried"


def test_a_run_that_does_not_say_its_rc_is_an_error_not_a_timeout():
    """A timeout is the most wanted kind, so a run that does not say how
    it ended must never pass for one."""
    run_without_rc = {"mode": "prove", "engine": "abc pdr", "notes": []}
    assert build_corpus.engine_alone(graded(INDUCTION, run_without_rc)) == \
        "error"


def test_it_reads_the_checkers_own_result_as_the_log_keeps_it():
    def evidence(engine, rc):
        return RunEvidence(mode="prove", rc=rc, depth=20, engine=engine,
                           duration_s=2.0, workdir=Path("job"),
                           log_excerpt="")
    result = TripleResult(
        NecessityVerdict.NECESSARY, "load-bearing",
        with_invariants=GradeResult(Tier.PROVEN, "proven",
                                    [evidence("smtbmc yices", 0)]),
        without_invariants=GradeResult(
            Tier.NOT_INDUCTIVE, "not inductive",
            [evidence("smtbmc yices", 4), evidence("abc pdr", 8)]))
    record = {"verdict": "NECESSARY", "result": asdict(result)}
    assert build_corpus.engine_alone(record) == "timed_out"
    # and from the log line, read back
    logged = json.loads(json.dumps(record, default=str))
    assert build_corpus.engine_alone(logged) == "timed_out"


# --- on the rows ---------------------------------------------------------------

ATTEMPT = dict({
    "run_id": "2026-10-09_10h00m00s", "attempt": 1, "verdict": "NECESSARY",
    "raw_json": json.dumps({
        "verilog": "module m; always @(*) assert (a == b); endmodule",
        "top_module": "m", "clock": "clk", "antecedents": [],
        "sanity_covers": [], "invariants": ["a == b"]})},
    **graded(INDUCTION, pdr(8)))


def test_a_starting_row_says_what_the_engine_found_alone():
    row = build_corpus.flatten_record(ATTEMPT, 0)
    assert row["metrics"]["engine_alone"] == "timed_out"


PARENT = {"id": "c0_000", "generation": 0, "top_module": "m",
          "clock": "clk", "antecedents": [], "sanity_covers": [],
          "verilog": "module m; endmodule", "property": ["a"],
          "invariants": ["a == b"]}


def step(**kw):
    rec = {"extension_id": "e1", "ext_type": "structural", "move": "STAGE",
           "parent_id": "c0_000", "verdict": "NECESSARY",
           "child_verilog": "module m; reg c; endmodule",
           "invariants": ["a == b", "c <= 4'd3"]}
    rec.update(kw)
    return rec


def test_a_promoted_row_says_what_the_engine_found_alone():
    _, rows = promote.promote([PARENT], [step(**graded(INDUCTION, pdr(8)))],
                              compute_metrics=False)
    assert rows[0]["metrics"]["engine_alone"] == "timed_out"
    _, rows = promote.promote([PARENT], [step()], compute_metrics=False)
    assert rows[0]["metrics"]["engine_alone"] == "not_tried"


def test_a_step_checked_again_is_read_from_its_second_check():
    rec = step(**graded(INDUCTION, pdr(0)))
    rec["first_result"] = {"verdict": "INCONCLUSIVE",
                           **graded(run("smtbmc yices", 8))}
    rec["rechecked"] = True
    _, rows = promote.promote([PARENT], [rec], compute_metrics=False)
    assert rows[0]["metrics"]["engine_alone"] == "proved"


def rows_found(*found):
    return [{"id": f"c0_{i}", "metrics": {"engine_alone": f} if f else {}}
            for i, f in enumerate(found)]


def test_the_share_of_each_kind_is_reported():
    rows = rows_found("proved", "timed_out", None, "error")
    assert build_corpus.engine_alone_share(rows) == (
        "timed_out 1 (25%), error 1 (25%), proved 1 (25%), unknown 1 (25%)")
    assert build_corpus.engine_alone_share([]) == "no rows"


def test_the_corpus_build_reports_the_share(capsys):
    rows = [build_corpus.flatten_record(ATTEMPT, 0),
            build_corpus.flatten_record(
                dict(ATTEMPT, **graded(INDUCTION, pdr(0))), 1)]
    build_corpus.print_distributions(rows)
    assert ("engine alone (abc pdr, no facts): timed_out 1 (50%), "
            "proved 1 (50%)") in capsys.readouterr().out


# --- which rows the run grows first --------------------------------------------

def design(id, found=None, lines=40, group=None, gen=0, parent=None):
    """A design of `lines` lines, no two alike; `found` is what the engine
    found alone (None: a row made before 21 Oct 2026, which says nothing)."""
    top = f"m_{id}"
    body = [f"module {top} (input wire clk, input wire en, "
            "output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(lines - 2)] \
        + ["endmodule"]
    row = {"id": id, "generation": gen, "parent": parent, "top_module": top,
           "clock": "clk", "antecedents": [], "sanity_covers": [],
           "verilog": "\n".join(body), "property": ["a <= 4'd9"],
           "invariants": ["a <= 4'd9"], "metrics": {}}
    if found:
        row["metrics"]["engine_alone"] = found
    if group:
        row["construct"] = f"{kind_in(group)}: does its job"
    return row


def test_rows_rank_by_what_the_engine_found_alone():
    ranks = {found: batch.engine_rank(design("c0_1", found))
             for found in ("timed_out", "error", "not_tried", "proved")}
    assert ranks == {"timed_out": 0, "error": 1, "not_tried": 1,
                     "proved": 2}
    # a row that says nothing is not pushed back
    assert batch.engine_rank(design("c0_1")) == 1
    assert batch.engine_rank({"id": "x"}) == 1


def failing(seen):
    """An executor whose every step fails; it notes each parent drawn."""
    def exe(task, corpus):
        seen.append(task["parent_id"])
        return {"verdict": "DECORATIVE", "parent_id": task["parent_id"],
                "extension_id": f"e{len(seen)}"}
    return exe


MIXED = [design("c0_1", "proved"), design("c0_2", "timed_out"),
         design("c0_3"), design("c0_4", "error"), design("c0_5", "not_tried"),
         design("c0_6", "proved"), design("c0_7", "timed_out"),
         design("c0_8")]


def shuffled(rows, seed=0):
    """The run's own order: run_loop shuffles the rows with its rng first."""
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def test_the_rows_the_engine_could_not_prove_alone_start_first(monkeypatch):
    # one failure ends a branch, so every row is drawn once, in order
    monkeypatch.setattr(batch, "BRANCH_FAILS", 1)
    seen = []
    batch.run_loop(MIXED, failing(seen), max_calls=len(MIXED),
                   rng=random.Random(0))
    expected = sorted(shuffled(MIXED), key=batch.engine_rank)
    assert seen == [r["id"] for r in expected]
    assert seen[:2] == ["c0_7", "c0_2"] or seen[:2] == ["c0_2", "c0_7"]
    assert set(seen[-2:]) == {"c0_1", "c0_6"}
    # within one kind, the run's own (shuffled) order holds
    plain = [r["id"] for r in shuffled(MIXED)]
    middle = [i for i in plain if i in {"c0_3", "c0_4", "c0_5", "c0_8"}]
    assert seen[2:6] == middle


def test_without_the_preference_the_run_keeps_its_own_order(monkeypatch):
    monkeypatch.setattr(batch, "BRANCH_FAILS", 1)
    seen = []
    batch.run_loop(MIXED, failing(seen), max_calls=len(MIXED),
                   rng=random.Random(0), prefer_engine_hard=False)
    assert seen == [r["id"] for r in shuffled(MIXED)]


def test_an_explicit_start_list_keeps_its_order(monkeypatch):
    monkeypatch.setattr(batch, "BRANCH_FAILS", 1)
    seen = []
    batch.run_loop(MIXED, failing(seen), max_calls=3, rng=random.Random(0),
                   start_ids=["c0_1", "c0_3", "c0_2"])
    assert seen == ["c0_1", "c0_3", "c0_2"]


def test_the_depth_order_keeps_its_order_within_each_kind(monkeypatch):
    monkeypatch.setattr(batch, "BRANCH_FAILS", 1)
    rows = [design("c0_1", "proved"), design("c0_2"),
            design("c1_000", "proved", gen=1, parent="c0_1"),
            design("c1_001", "timed_out", gen=1, parent="c0_2"),
            design("c0_3", "timed_out")]
    seen = []
    batch.run_loop(rows, failing(seen), max_calls=len(rows),
                   rng=random.Random(0), order="depth")
    by_id = {r["id"]: r for r in rows}
    expected = sorted(order_frontier(rows, by_id, per_family=3),
                      key=batch.engine_rank)
    assert seen == [r["id"] for r in expected]
    assert seen[:2] == ["c1_001", "c0_3"]           # deepest first, as before


def test_a_row_the_engine_could_not_prove_alone_is_tried_again_first():
    rows = [design("c0_1", "proved"), design("c0_2", "timed_out")]
    seen = []
    batch.run_loop(rows, failing(seen), max_calls=6, rng=random.Random(0))
    assert seen == ["c0_2"] * 3 + ["c0_1"] * 3       # three failures each
    # without the preference the two take turns, as before
    seen = []
    batch.run_loop(rows, failing(seen), max_calls=6, rng=random.Random(0),
                   prefer_engine_hard=False)
    assert seen[0] != seen[1] and seen == seen[:2] * 3


def grown(parent, rc):
    """An accepted step whose child the engine found `rc` for, alone."""
    return {"verdict": "NECESSARY", "extension_id": f"e_{parent['id']}",
            "ext_type": "structural", "move": "SUBUNIT",
            "parent_id": parent["id"],
            "child_verilog": parent["verilog"] + "\n// grew",
            "invariants": parent["invariants"] + ["v + w == 4'd3"],
            **graded(INDUCTION, pdr(rc))}


def first_accepted(rc, seen):
    """Accepts the first step (its child: `rc` for the engine alone) and
    fails every later one."""
    def exe(task, corpus):
        seen.append(task["parent_id"])
        if len(seen) == 1:
            parent = next(r for r in corpus if r["id"] == task["parent_id"])
            return grown(parent, rc)
        return {"verdict": "DECORATIVE", "parent_id": task["parent_id"],
                "extension_id": f"e{len(seen)}"}
    return exe


def test_a_child_the_engine_could_not_prove_alone_goes_next():
    rows = [design("c0_1"), design("c0_2")]
    seen = []
    out = batch.run_loop(rows, first_accepted(8, seen), max_calls=2,
                         rng=random.Random(0))
    assert out[0]["metrics"]["engine_alone"] == "timed_out"
    assert seen[1] == out[0]["id"]
    # without the preference it waits behind the other starting row
    seen = []
    out = batch.run_loop(rows, first_accepted(8, seen), max_calls=2,
                         rng=random.Random(0), prefer_engine_hard=False)
    assert seen[1] != out[0]["id"] and seen[1] != seen[0]


def test_a_chased_child_goes_next_even_when_the_engine_proved_it_alone():
    rows = [design("c0_1", "timed_out"), design("c0_2", "timed_out")]
    seen = []
    out = batch.run_loop(rows, first_accepted(0, seen), max_calls=2,
                         rng=random.Random(0), order="chase",
                         start_ids=["c0_1", "c0_2"])
    assert out[0]["metrics"]["engine_alone"] == "proved"
    assert seen == ["c0_1", out[0]["id"]]


def ranks_of(frontier):
    return [batch.engine_rank(r) for r in frontier]


def test_a_row_rejoins_after_the_last_row_ranked_no_better():
    frontier = deque([design("a", "timed_out"), design("b"),
                      design("c", "error"), design("d", "proved")])
    batch.join_frontier(frontier, design("e", "not_tried"))
    assert [r["id"] for r in frontier] == ["a", "b", "c", "e", "d"]
    batch.join_frontier(frontier, design("f", "timed_out"))
    assert [r["id"] for r in frontier][:2] == ["a", "f"]
    batch.join_frontier(frontier, design("g", "proved"))
    assert frontier[-1]["id"] == "g"
    assert ranks_of(frontier) == sorted(ranks_of(frontier))


def test_a_row_ranked_better_than_all_goes_to_the_front():
    frontier = deque([design("a", "proved"), design("b")])
    batch.join_frontier(frontier, design("c", "timed_out"))
    assert [r["id"] for r in frontier] == ["c", "a", "b"]
    empty = deque()
    batch.join_frontier(empty, design("d", "proved"))
    assert [r["id"] for r in empty] == ["d"]


def test_without_the_preference_a_row_rejoins_at_the_back():
    frontier = deque([design("a", "proved")])
    batch.join_frontier(frontier, design("b", "timed_out"), prefer=False)
    assert [r["id"] for r in frontier] == ["a", "b"]


# --- starting from the biggest designs -----------------------------------------

STARTS = [design("c0_1", "proved", 180, "memory"),
          design("c0_2", "timed_out", 120, "memory"),
          design("c0_3", None, 150, "crypto"),
          design("c0_4", "timed_out", 90, "DSP"),
          design("c0_5", "error", 200, "control")]


def test_the_biggest_starts_are_taken_from_the_engine_hard_designs_first():
    # one per hardware group still; biggest first within one kind
    assert batch.largest_starts(STARTS, 2) == ["c0_2", "c0_4"]
    assert batch.largest_starts(STARTS, 4) == ["c0_2", "c0_4", "c0_5",
                                               "c0_3"]
    assert batch.largest_starts(STARTS, 2, prefer_engine_hard=False) == \
        ["c0_5", "c0_1"]


# --- the command line -----------------------------------------------------------

@pytest.fixture
def command(monkeypatch, tmp_path):
    """main() with the loop replaced: it notes what it was given and
    returns the rows the test hands it. Logs go to a temporary folder."""
    monkeypatch.setattr(distractor, "OUT_LOG", tmp_path / "extensions.jsonl")
    monkeypatch.setattr(batch, "QUEUE_LOG", tmp_path / "batch_queue.jsonl")
    monkeypatch.setattr(batch, "FIXER_QUEUE", tmp_path / "fixer_queue.jsonl")
    seen = {}

    def run(args, rows=STARTS, kept=()):
        corpus = tmp_path / "corpus_trial.jsonl"
        corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))

        def fake_loop(corpus_rows, executor, **kw):
            seen.update(kw)
            return list(kept)
        monkeypatch.setattr(batch, "run_loop", fake_loop)
        batch.main(["--max-calls", "1", "--corpus", str(corpus),
                    "--log-dir", str(tmp_path / "logs")] + args)
        return seen
    return run


def test_the_run_prefers_engine_hard_rows_unless_told_not_to(command):
    assert command([])["prefer_engine_hard"] is True
    assert command(["--no-prefer-engine-hard"])["prefer_engine_hard"] is False


def test_the_biggest_starts_follow_the_switch(command, capsys):
    assert command(["--start-largest", "2"])["start_ids"] == ["c0_2", "c0_4"]
    out = capsys.readouterr().out
    assert "c0_2 (120 lines, engine alone: timed_out)" in out
    assert command(["--start-largest", "2", "--no-prefer-engine-hard"]
                   )["start_ids"] == ["c0_5", "c0_1"]


def test_a_run_reports_the_share_of_its_kept_designs(command, capsys):
    kept = rows_found("timed_out", "proved")
    command([], kept=kept)
    assert ("could the engine prove the kept designs without facts? "
            "timed_out 1 (50%), proved 1 (50%)") in capsys.readouterr().out


def test_the_plan_puts_engine_hard_rows_first(monkeypatch, tmp_path, capsys):
    rows = [design("c0_1", "proved"), design("c0_2", "proved"),
            design("c0_3", "timed_out")]
    corpus = tmp_path / "corpus_trial.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
    batch.main(["--dry", "--max-calls", "3", "--corpus", str(corpus)])
    planned = [l for l in capsys.readouterr().out.splitlines()
               if " on c0_" in l]
    assert " on c0_3 " in planned[0]
