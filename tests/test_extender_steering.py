"""What the extender asks for, after the hand review of the 17 Oct trial
(18 Oct 2026).

Big moves (a sub-unit, storage, a second unit) gave 5 real designs out of 5;
small tweaks gave 4 contrived ones out of 4; combinations of designs drawn
at random gave 0 good ones out of 7 (a power sequencer feeding a DCT, a
status word reseeding a random-number generator); copies restated their
facts per copy up to 202. So: big moves drawn most, partners from hardware
groups that are actually connected, copies and combinations only while the
fact list stays within 64, and questions that state the rules the checks
in gates.py now enforce."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import extend                                                   # noqa: E402

CATALOG = json.loads((ROOT / "initiator" / "catalog" /
                      "kind_categories_2026-10-09.json").read_text())
_GROUP = {a["id"]: a["group"] for a in CATALOG["areas"]}


def kind_in(group):
    return next(k["name"] for k in CATALOG["kinds"]
                if _GROUP.get(k["category"]) == group)


def row(id, group=None, facts=2, top=None):
    top = top or f"m_{id}"
    return {"id": id, "generation": 0, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "construct": f"{kind_in(group)}: x" if group else None,
            "verilog": f"module {top} (input wire clk, output reg [3:0] a);"
                       "\nendmodule",
            "invariants": [f"a <= 4'd{i % 9}" for i in range(facts)],
            "metrics": {}}


# --- partners that fit ------------------------------------------------------

def test_a_design_knows_its_hardware_group_and_children_inherit_it():
    rows = [row("c0_1", "memory"), row("c0_2", "crypto")]
    child = dict(row("c1_0"), construct=None, parent="c0_1", parent2="c0_2")
    by_id = {r["id"]: r for r in rows + [child]}
    assert batch.design_groups(rows[0], by_id) == {"memory"}
    assert batch.design_groups(child, by_id) == {"memory", "crypto"}


def test_connected_groups_fit_and_unrelated_ones_do_not():
    by_id = {}
    mem, iface = row("c0_1", "memory"), row("c0_2", "interface")
    power, dsp = row("c0_3", "clocking/reset/power"), row("c0_4", "DSP")
    for r in (mem, iface, power, dsp):
        by_id[r["id"]] = r
    assert batch.fits(mem, iface, by_id) and batch.fits(iface, mem, by_id)
    assert batch.fits(mem, dict(mem, id="c0_5", top_module="m2"), by_id)
    assert not batch.fits(power, dsp, by_id)    # the trial's ramp-into-DCT


def test_a_design_with_no_known_group_is_never_combined():
    by_id = {}
    assert not batch.fits(row("c0_1"), row("c0_2", "memory"), by_id)


def test_the_fit_table_reads_the_same_both_ways():
    for g, near in batch.GROUP_FIT.items():
        for h in near:
            assert g in batch.GROUP_FIT[h], (g, h)


def test_only_fitting_partners_are_offered():
    rng = random.Random(4)
    parent = row("c0_1", "clocking/reset/power")
    pool = [parent, row("c0_2", "DSP"), row("c0_3", "control")]
    # since 19 Oct the model picks from a short list (candidates)
    partners = {c for t in
                (batch.sample_task(rng, parent, pool) for _ in range(3000))
                if t["ext_type"] == "compose" for c in t["candidates"]}
    assert partners == {"c0_3"}


# --- the fact budget ---------------------------------------------------------

def test_copies_are_offered_only_while_the_facts_fit():
    assert batch.copies_that_fit(10) == [4, 6]      # 61 and 41 facts
    assert batch.copies_that_fit(15) == [4]         # 61; six would be 91
    assert batch.copies_that_fit(20) == []          # four would be 81


def test_a_design_with_too_many_facts_is_not_copied_or_combined():
    rng = random.Random(9)
    big = row("c0_1", "memory", facts=40)
    pool = [big, row("c0_2", "memory", facts=30)]
    kinds = {batch.sample_task(rng, big, pool)["ext_type"]
             for _ in range(2000)}
    assert not kinds & {"compose", "replicate"}


def test_combining_and_copying_take_a_quarter_of_draws():
    assert batch.MULTIPLIER_RATE == 0.25


# --- big moves drawn most -----------------------------------------------------

def test_big_moves_take_most_of_the_draws_and_small_tweaks_little():
    w = extend.MOVE_WEIGHTS
    big = w["SUBUNIT"] + w["STORAGE"] + w["PEER"]
    assert big * 3 >= 2 * sum(w.values())
    for small in ("STAGE", "SPLIT", "BOUND", "GUARD"):
        assert w[small] <= 1
    # kept on from 6 Sep; dropped 18 Oct after 0 good designs in 4
    # (Rishab: "do all 6 changes")
    assert w["LEDGER"] == 0


# --- the questions state the rules the checks enforce -------------------------

def test_no_question_asks_for_a_copy_of_an_existing_register():
    for text in (extend.SYSTEM_PROMPT, extend.COMPOSE_TEMPLATE):
        assert "only copies" in text


def test_counters_are_sized_to_their_range_not_the_widest_word():
    assert "sized to the range" in extend.SYSTEM_PROMPT
    assert "match the widths" in extend.SYSTEM_PROMPT.lower()
    for text in (extend.COMPOSE_TEMPLATE, extend.REPLICATE_TEMPLATE):
        assert "sized to their range" in text


def test_copies_must_use_their_results_and_return_credits_when_work_ends():
    t = " ".join(extend.REPLICATE_TEMPLATE.split())
    assert "Every copy's results must be used" in t
    assert "never XOR" in t
    assert "when the work it paid for finishes" in t


def test_a_combination_must_connect_what_means_something():
    t = " ".join(extend.COMPOSE_TEMPLATE.split())
    # 26 Oct 2026: "to B" became "to the receiver", because the plan may
    # send from B to A (9 of 21 plans; change 7)
    assert "must mean something to the receiver" in t


def test_new_output_ports_are_allowed_so_ledger_no_longer_contradicts():
    """LEDGER asked for every new register as an output port while rule 3
    said the port list was fixed (the other session's plan, 18 Oct)."""
    assert "may add OUTPUT ports" in extend.SYSTEM_PROMPT
    assert "The port list is fixed." not in extend.SYSTEM_PROMPT


def test_ledger_counters_are_not_forced_to_gate_anything():
    """The trial's ledger child added a gate that was always true, only to
    satisfy 'the new state must write or gate' (hand review, 18 Oct)."""
    p = {"verilog": "module m;\nendmodule", "property": ["a"],
         "invariants": ["a"]}
    ledger = extend.build_prompt(p, "structural", "LEDGER")
    other = extend.build_prompt(p, "structural", "SUBUNIT")
    assert "must write, or gate the write of" not in ledger
    assert "do not need to write or gate" in ledger
    assert "must write, or gate the write of" in other
