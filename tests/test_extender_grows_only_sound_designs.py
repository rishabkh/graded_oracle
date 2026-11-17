"""A design a reviewer labeled wrong is never grown from (16 Nov 2026,
Rishab: "alright let's do it", change 1 of 7).

In the pilot of 15 Nov 2026 the extender grew designs before anyone had
labeled them. 19 of its 86 kept designs came from a parent later found
wrong or borderline, and 16 of those 19 were wrong themselves; from a
sound parent it was 14 of 67. So the starting designs are labeled first
and the run is told the labels (--labels, the file label_designs.py
writes):
  - a design labeled wrong is never a parent and never a partner;
  - a starting design the labels do not mention was not reviewed, so it
    is left out as well;
  - a grown design the labels do not mention is grown as before (the run
    makes these itself, after the labels were written);
  - a key labeled twice counts by its last label;
  - a labels file that is not there stops the run before any call.
No model and no checker run here."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import distractor                                               # noqa: E402
import label_designs                                            # noqa: E402

CATALOG = json.loads((ROOT / "initiator" / "catalog" /
                      "kind_categories_2026-10-09.json").read_text())
_GROUP = {a["id"]: a["group"] for a in CATALOG["areas"]}


def kind_in(group):
    return next(k["name"] for k in CATALOG["kinds"]
                if _GROUP.get(k["category"]) == group)


def design(id, top=None, gen=0, n=60, group=None):
    """A design of n lines, every line different (no copies)."""
    top = top or f"m_{id}"
    body = [f"module {top} (input wire clk, input wire en, "
            "output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(n - 2)] + \
        ["endmodule"]
    row = {"id": id, "generation": gen, "top_module": top, "clock": "clk",
           "antecedents": [], "sanity_covers": [],
           "verilog": "\n".join(body), "property": ["a <= 4'd9"],
           "invariants": ["a <= 4'd9"], "metrics": {}}
    if group:
        row["construct"] = f"{kind_in(group)}: does its job"
    return row


def labels_file(tmp_path, *pairs):
    path = tmp_path / "labels_trial.jsonl"
    path.write_text("".join(json.dumps({"key": k, "label": l}) + "\n"
                            for k, l in pairs))
    return path


def kept(task, corpus):
    """A checked step that adds a claim of a new shape (see test_batch)."""
    parent = next(r for r in corpus if r["id"] == task["parent_id"])
    fresh = " + ".join(f"v{i}" for i in range(task["task_id"] + 2))
    return {"verdict": "NECESSARY", "extension_id": f"e{task['task_id']}",
            "ext_type": task["ext_type"], "move": task.get("move"),
            "parent_id": task["parent_id"],
            "child_verilog": parent["verilog"] + f"\n// grew {task['task_id']}",
            "invariants": list(parent["invariants"]) + [f"{fresh} == total"],
            "result": {}}


# --- reading the labels -----------------------------------------------------------

def test_the_last_label_of_a_key_counts(tmp_path):
    path = labels_file(tmp_path, ("a", "wrong"), ("b", "good"),
                       ("a", "good"), ("b", "wrong"), ("c", "weak"))
    assert label_designs.labels_by_key(path) == {
        "a": "good", "b": "wrong", "c": "weak"}


def test_a_row_that_holds_no_label_is_not_a_label(tmp_path):
    # a call that failed leaves a row with no label; a rerun asks again
    path = labels_file(tmp_path, ("a", None), ("b", "good"))
    assert label_designs.labels_by_key(path) == {"b": "good"}


# --- which designs the labels leave out -------------------------------------------

def test_wrong_designs_and_unreviewed_starts_are_left_out():
    rows = [design("s_good"), design("s_weak"), design("s_wrong"),
            design("s_unseen"), design("g_good", gen=1),
            design("g_wrong", gen=2), design("g_unseen", gen=1)]
    labels = {"s_good": "good", "s_weak": "weak", "s_wrong": "wrong",
              "g_good": "good", "g_wrong": "wrong"}
    assert batch.left_out_by_labels(rows, labels) == {
        "s_wrong", "s_unseen", "g_wrong"}


def test_no_design_is_left_out_unless_the_run_is_told_labels():
    assert batch.NOT_GROWN == frozenset()
    assert batch.extendable(design("c0_1"), None)


def test_a_left_out_design_cannot_be_grown(monkeypatch):
    monkeypatch.setattr(batch, "NOT_GROWN", frozenset({"c0_1"}))
    assert not batch.extendable(design("c0_1"), None)
    assert batch.extendable(design("c0_2"), None)


def test_the_run_never_grows_a_left_out_design(monkeypatch):
    monkeypatch.setattr(batch, "NOT_GROWN", frozenset({"c0_1"}))
    asked = []

    def executor(task, corpus):
        asked.append(task["parent_id"])
        return kept(task, corpus)
    batch.run_loop([design("c0_1"), design("c0_2")], executor, max_calls=6,
                   max_gen=1, rng=random.Random(0),
                   prefer_engine_hard=False)
    assert asked and set(asked) == {"c0_2"}


def partners_offered(rows, seed=1):
    rng = random.Random(seed)
    tasks = (batch.sample_task(rng, rows[0], rows) for _ in range(400))
    return {c for t in tasks for c in t.get("candidates", [])}


def pool():
    """A memory design and three starting designs that fit it."""
    return [design("c0_1", top="mem_a", n=80, group="memory")] + \
        [design(f"c0_{10 + i}", top=f"if_{i}", group="interface")
         for i in range(3)]


def test_a_left_out_design_is_never_offered_as_a_partner(monkeypatch):
    assert partners_offered(pool()) == {"c0_10", "c0_11", "c0_12"}
    monkeypatch.setattr(batch, "NOT_GROWN", frozenset({"c0_11"}))
    assert partners_offered(pool()) == {"c0_10", "c0_12"}


# --- the command line -------------------------------------------------------------

ROWS = [design("c0_1"), design("c0_2"), design("c0_3"),
        design("c1_000", gen=1)]


@pytest.fixture
def command(monkeypatch, tmp_path):
    """main() with the loop replaced: it notes which designs were left
    out while the run was going. Logs go to a temporary folder."""
    monkeypatch.setattr(distractor, "OUT_LOG", tmp_path / "extensions.jsonl")
    monkeypatch.setattr(batch, "QUEUE_LOG", tmp_path / "batch_queue.jsonl")
    monkeypatch.setattr(batch, "FIXER_QUEUE", tmp_path / "fixer_queue.jsonl")
    corpus = tmp_path / "corpus_trial.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    seen = {}

    def run(args):
        def fake_loop(corpus_rows, executor, **kw):
            seen["left_out"] = batch.NOT_GROWN
            return []
        monkeypatch.setattr(batch, "run_loop", fake_loop)
        batch.main(["--max-calls", "1", "--corpus", str(corpus),
                    "--log-dir", str(tmp_path / "logs")] + args)
        return seen
    return run


def test_the_run_is_told_the_labels(command, tmp_path, capsys):
    labels = labels_file(tmp_path, ("c0_1", "good"), ("c0_2", "wrong"))
    seen = command(["--labels", str(labels)])
    # c0_2 is wrong, c0_3 is a starting design nobody reviewed, and the
    # grown c1_000 has no label yet
    assert seen["left_out"] == {"c0_2", "c0_3"}
    out = capsys.readouterr().out
    assert "labels: 2 of 4 designs are left out" in out
    assert "1 labeled wrong" in out and "1 starting design with no label" in out


def test_the_labels_last_only_for_that_run(command, tmp_path):
    labels = labels_file(tmp_path, ("c0_1", "good"), ("c0_2", "wrong"))
    command(["--labels", str(labels)])
    assert batch.NOT_GROWN == frozenset()


def test_a_run_without_labels_leaves_nothing_out(command):
    assert command([])["left_out"] == frozenset()


def test_a_missing_labels_file_stops_the_run_before_any_call(command,
                                                            tmp_path):
    with pytest.raises(SystemExit) as stop:
        command(["--labels", str(tmp_path / "no_such_labels.jsonl")])
    assert "no_such_labels.jsonl" in str(stop.value)
    assert "nothing was run" in str(stop.value)


def test_the_plan_leaves_them_out_too(tmp_path, capsys):
    corpus = tmp_path / "corpus_trial.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    labels = labels_file(tmp_path, ("c0_1", "good"), ("c0_2", "wrong"),
                         ("c0_3", "weak"))
    batch.main(["--dry", "--max-calls", "4", "--corpus", str(corpus),
                "--labels", str(labels), "--max-gen", "1"])
    out = capsys.readouterr().out
    assert "frontier: 2 extendable rows" in out
    assert " on c0_2 " not in out
