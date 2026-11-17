"""What a grown part has taken must get out, and added storage is not
always four entries (16 Nov 2026, Rishab: "alright let's do it", change 3
of 7).

From the 15 Nov pilot (86 kept designs, GPT's labels, each complaint
traced in the code):
  - 8 grown designs starved an added lane or queue: the new queue emptied
    only while the old input was idle (`start_peer = in_ready && !in_valid
    && ...`), or a new requester was served only when the old one was
    quiet. The PEER step's own text offered "round robin or fixed
    priority". 2 more froze because an added side queue emptied only on
    another output's ready.
  - 78 of the stores the kept designs declare hold exactly 4 entries, the
    number in the instructions' example (`#(parameter DEPTH = 4)`).
So every big step is told that streams sharing a part take turns and that
a queue empties on its own receiver's ready, and each big step is handed
a size drawn for it (4 to 8, the range the proofs are kept to).
No model and no checker run here."""
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import extend                                                   # noqa: E402


def squash(text):
    return " ".join(text.split())


def design(id="c0_1", top="m", n=60):
    body = [f"module {top} (input wire clk, input wire en, "
            "output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(n - 2)] + \
        ["endmodule"]
    return {"id": id, "generation": 0, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": "\n".join(body), "property": ["a <= 4'd9"],
            "invariants": ["a <= 4'd9"], "metrics": {}}


def question(move, **kw):
    return squash(extend.build_prompt(design(), "structural", move, **kw))


# --- what was taken must get out ---------------------------------------------------

def test_the_waiting_rule_says_what_was_taken_must_get_out():
    rule = squash(extend.WAITING_RULE)
    assert "What a part has taken must also get out" in rule
    assert "take turns" in rule
    assert "the one that waited goes next" in rule
    # the two ways the pilot's grown designs starved or froze
    assert "never only while another stream is idle" in rule
    assert "never on another output's ready" in rule


def test_every_big_step_is_asked_it():
    for move in extend.BIG_MOVES:
        assert "What a part has taken must also get out" in question(move), \
            move


def test_a_small_step_reads_as_before():
    assert "What a part has taken must also get out" not in question("SPLIT")


def test_a_second_lane_takes_turns_instead_of_fixed_priority():
    peer = squash(extend.MOVES["PEER"])
    assert "round robin or fixed priority" not in peer
    assert "round robin" in peer
    assert "the lane that waited goes next" in peer
    assert "never fixed priority" in peer


# --- the size of what a big step adds ---------------------------------------------

def test_sizes_stay_in_the_range_the_proofs_are_kept_to():
    assert extend.NEW_DEPTHS == (4, 5, 6, 7, 8)
    assert max(extend.NEW_DEPTHS) * 2 <= extend.MAX_NEW_DEPTH   # two lanes


def drawn(n=3000, seed=11):
    rng = random.Random(seed)
    rows = [design()]
    return [batch.sample_task(rng, rows[0], rows) for _ in range(n)]


def test_every_big_step_is_handed_a_size():
    tasks = drawn()
    big = [t for t in tasks if t.get("move") in extend.BIG_MOVES]
    assert big
    assert all(t["depth"] in extend.NEW_DEPTHS for t in big)
    for t in tasks:
        if t.get("move") not in extend.BIG_MOVES:
            assert "depth" not in t, t


def test_four_entries_is_one_size_among_five():
    sizes = Counter(t["depth"] for t in drawn()
                    if t.get("move") in extend.BIG_MOVES)
    assert set(sizes) == set(extend.NEW_DEPTHS)
    share = sizes[4] / sum(sizes.values())
    assert 0.14 < share < 0.26          # one in five, was about 3 in 5


def test_the_same_seed_draws_the_same_sizes():
    assert [t.get("depth") for t in drawn(200, seed=5)] == \
        [t.get("depth") for t in drawn(200, seed=5)]


def test_the_question_names_the_size_drawn_for_the_step():
    for move in extend.BIG_MOVES:
        text = question(move, depth=7)
        assert "Size of what you add: 7 entries" in text, move
        assert "The 4 in the examples is only an example" in text, move


def test_a_question_without_a_drawn_size_reads_as_before():
    for move in extend.BIG_MOVES:
        assert "Size of what you add" not in question(move), move
    # and a small step never gets one
    assert "Size of what you add" not in question("SPLIT", depth=7)


def test_an_array_that_matches_an_existing_one_keeps_its_size():
    text = question("STORAGE", depth=6)
    assert "unless it holds one entry for each of the same things" in text


def test_the_run_puts_the_tasks_size_in_the_question(monkeypatch):
    parent = design(id="c0_000")
    prompts = []
    out = {"new_state": "A queue in front of the input.", "coupling": "c",
           "induction_gap": "g", "why_parent_insufficient": "w",
           "patch": "", "invariants": [], "dispositions": []}

    def model(prompt, schema):
        prompts.append(prompt)
        return out, {"input": 1, "output": 1}, "ok"

    def grade(p, ext_type, o, record):
        record.update(verdict="NECESSARY", child_verilog="module m; endmodule",
                      invariants=["v + a == 4'd9"], patch="")
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "grade_step4", grade)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    task = {"task_id": 1, "attempt": 1, "ext_type": "structural",
            "move": "STORAGE", "parent_id": "c0_000", "k": 1, "depth": 5}
    batch._real_executor(dict(task), [parent])
    assert "Size of what you add: 5 entries" in squash(prompts[0])
