"""Tests for the batch loop (no API, no sby): extendability stops,
task sampling policy, and the while-loop mechanics with an injected
executor — call cap, branch death, reformat retries, recursive
enqueue of promoted children.
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from batch import (DISTRACTOR_RATE, extendable, finalize_distractor,   # noqa: E402
                   run_loop, sample_task, without_wall)


def row(id="g0_001", gen=0, lines=40, top="m", invs=("a <= 2",)):
    return {"id": id, "generation": gen, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": "\n".join(["module %s (input wire clk, output reg [1:0] a);" % top]
                                 + ["// filler"] * (lines - 2) + ["endmodule"]),
            "invariants": list(invs), "metrics": {}}


# --- stopping conditions ---

def test_extendable_blocks_deep_generations():
    assert extendable(row(gen=2), 1.0, max_gen=3)
    assert not extendable(row(gen=3), 1.0, max_gen=3)


def test_extendable_blocks_thousand_lines():
    assert not extendable(row(lines=1001), 1.0)
    assert extendable(row(lines=999), 1.0)


def test_extendable_blocks_slow_without_run():
    assert not extendable(row(), 61.0)
    assert extendable(row(), 59.0)
    assert extendable(row(), None)      # no measurement -> allowed


def test_without_wall_reads_prove_durations():
    rec = {"result": {"without_invariants": {"runs": [
        {"mode": "prove", "duration_s": 3.0},
        {"mode": "prove", "duration_s": 121.0}]}}}
    assert without_wall(rec) == 121.0
    assert without_wall({"result": {}}) is None


# --- sampling policy ---

def test_sample_task_never_picks_guard_and_respects_eligibility():
    rng = random.Random(7)
    hidden = row(id="g0_002", top="w",
                 invs=("timer <= 8",))   # timer not a port -> ineligible
    pool = [row(), hidden]
    for _ in range(200):
        t = sample_task(rng, row(), pool)
        assert t["move"] != "GUARD"
        if t["ext_type"] == "compose":
            assert t["parent2_id"] != "g0_002"


def test_sample_task_draws_distractor_about_one_in_six():
    rng = random.Random(11)
    draws = [sample_task(rng, row(), [row()]) for _ in range(3000)]
    share = sum(t["ext_type"] == "distractor" for t in draws) / len(draws)
    assert abs(DISTRACTOR_RATE - 1 / 6) < 1e-9
    assert 0.12 < share < 0.21, share
    d = next(t for t in draws if t["ext_type"] == "distractor")
    assert d["move"] is None and d["parent2_id"] is None


def test_distractor_dead_logic_is_a_reformat_not_a_promotion():
    # yosys swept the added logic: the child is the parent with dead text.
    # NECESSARY is true but meaningless; it must not enter the corpus.
    dead = finalize_distractor({"verdict": "NECESSARY", "live": False})
    assert dead["verdict"] == "DEAD_LOGIC"
    live = finalize_distractor({"verdict": "NECESSARY", "live": True})
    assert live["verdict"] == "NECESSARY"
    other = finalize_distractor({"verdict": "PATCH_ERROR"})
    assert other["verdict"] == "PATCH_ERROR"


def test_sample_task_is_reproducible_with_seed():
    a = [sample_task(random.Random(3), row(), [row()]) for _ in range(5)]
    b = [sample_task(random.Random(3), row(), [row()]) for _ in range(5)]
    assert a == b


# --- the loop, with an injected executor ---

def necessary(task, corpus):
    # each child unique, or promote's content-hash dedup (correctly)
    # collapses them and the test would measure dedup, not the loop
    return {"verdict": "NECESSARY", "extension_id": f"e{task['task_id']}",
            "ext_type": task["ext_type"], "move": task.get("move"),
            "parent_id": task["parent_id"],
            "child_verilog": row()["verilog"] + f"\n// grew {task['task_id']}",
            "invariants": ["a <= 2", "b <= 1"], "result": {}}


def test_loop_respects_hard_call_cap():
    calls = []
    def executor(task, corpus):
        calls.append(task)
        return necessary(task, corpus)
    run_loop([row()], executor, max_calls=5, max_gen=9, rng=random.Random(1))
    assert len(calls) == 5


def test_loop_branch_dies_after_three_consecutive_failures():
    calls = []
    def executor(task, corpus):
        calls.append(task)
        return {"verdict": "NOT_PROVEN", "parent_id": task["parent_id"],
                "extension_id": f"e{task['task_id']}", "result": {}}
    run_loop([row()], executor, max_calls=50, rng=random.Random(1))
    # only one branch existed; it must die after exactly 3 failures
    assert len(calls) == 3


def test_loop_reformat_retries_same_task_up_to_three_attempts():
    seen = []
    def executor(task, corpus):
        seen.append((task["task_id"], task["attempt"]))
        return {"verdict": "TRUNCATED", "parent_id": task["parent_id"],
                "extension_id": f"e{task['task_id']}.{task['attempt']}",
                "result": {}}
    run_loop([row()], executor, max_calls=50, rng=random.Random(1))
    # same task id retried with attempt 1,2,3 before the branch dies
    assert seen[0][0] == seen[1][0] == seen[2][0]
    assert [a for _, a in seen[:3]] == [1, 2, 3]


def test_loop_promotes_and_extends_children_recursively():
    generations = []
    def executor(task, corpus):
        parent = next(r for r in corpus if r["id"] == task["parent_id"])
        generations.append(parent.get("generation", 0))
        return necessary(task, corpus)
    new_rows = run_loop([row()], executor, max_calls=6, max_gen=3,
                        rng=random.Random(2))
    assert len(new_rows) == 6
    assert max(generations) >= 1      # some call extended a promoted child
    assert all(r["generation"] >= 1 for r in new_rows)


def test_loop_hands_fixer_verdicts_to_the_sink():
    # NOT_PROVEN routes to the fixer queue, not silently to branch death
    got = []
    def failing(task, corpus):
        return {"verdict": "NOT_PROVEN", "extension_id": f"e{task['task_id']}",
                "parent_id": task["parent_id"], "result": {}}
    run_loop([row()], failing, max_calls=3, rng=random.Random(0),
             on_fixer=got.append)
    assert len(got) == 3
    assert all(g["verdict"] == "NOT_PROVEN" for g in got)
