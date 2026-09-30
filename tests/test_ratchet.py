"""A generation step must add a kind of claim the parent did not need.

Measured 26 Sep 2026 on 142 parent-to-child pairs: the only acceptance
rule was NECESSARY, so 56% of children left the most-signals-per-clause
unchanged and 27 proved faster than their parents. Generation number
counted edits, not progress. The rule: a proved child is kept only if
at least one of its clauses has a shape (identifiers and constants
blanked) that no parent clause has. Distractors are exempt: they copy
the invariants on purpose, as the control for irrelevant logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from promote import new_shapes, promote, ratchet, route        # noqa: E402

PARENT = {"id": "g1_000", "generation": 1, "top_module": "m",
          "clock": "clk", "antecedents": [], "sanity_covers": [],
          "verilog": "module m; endmodule",
          "invariants": ["cnt <= 4'd9", "gp1[1] == (waddr == 2'b11)"]}
OTHER = {"id": "g0_005", "generation": 0, "top_module": "n",
         "clock": "clk", "antecedents": [], "sanity_covers": [],
         "verilog": "module n; endmodule",
         "invariants": ["(a + b) == total"]}
CHILD_V = "module m; always @(*) assert (a); endmodule"


def rec(invariants, ext_type="structural", verdict="NECESSARY", **kw):
    r = {"extension_id": "e9", "ext_type": ext_type, "move": "STAGE",
         "parent_id": "g1_000", "verdict": verdict,
         "child_verilog": CHILD_V, "invariants": invariants}
    r.update(kw)
    return r


def test_renamed_copies_are_not_a_new_kind_of_claim():
    """gp2[1] against gp1[1], a different constant: the same idea again."""
    assert new_shapes(["gp2[1] == (waddr == 2'b00)", "cnt <= 4'd7"],
                      PARENT["invariants"]) == []


def test_a_clause_of_a_new_shape_is_found():
    assert new_shapes(PARENT["invariants"] + ["cnt == (a + b)"],
                      PARENT["invariants"]) != []


def test_a_proved_child_with_nothing_new_is_turned_away():
    r = ratchet(rec(PARENT["invariants"] + ["gp3[0] == (raddr == 2'b01)"]),
                [PARENT])
    assert r["verdict"] == "NO_NEW_SHAPE"
    assert "reason" in r


def test_a_proved_child_with_a_new_kind_of_claim_is_kept():
    r = ratchet(rec(PARENT["invariants"] + ["cnt == (a + b)"]), [PARENT])
    assert r["verdict"] == "NECESSARY"
    assert r["new_shapes"]


def test_a_child_that_replaced_a_clause_with_a_new_kind_is_kept():
    """Same number of clauses, but one idea swapped for another: that
    is new reasoning, which a count-must-rise rule would miss."""
    r = ratchet(rec(["cnt <= 4'd9", "wptr == (rptr + fill)"]), [PARENT])
    assert r["verdict"] == "NECESSARY"


def test_the_distractor_is_exempt():
    r = ratchet(rec(PARENT["invariants"], ext_type="distractor"), [PARENT])
    assert r["verdict"] == "NECESSARY"


def test_both_parents_count_as_old_for_a_composition():
    r = ratchet(rec(PARENT["invariants"] + ["(x + y) == sum"],
                    ext_type="compose", parent2_id="g0_005"),
                [PARENT, OTHER])
    assert r["verdict"] == "NO_NEW_SHAPE"


def test_only_a_proved_child_is_judged():
    r = ratchet(rec([], verdict="NOT_PROVEN"), [PARENT])
    assert r["verdict"] == "NOT_PROVEN"


def test_nothing_new_is_a_rejection_not_a_repair():
    """The child is correct, just not harder. Nothing for the Fixer to
    fix, and it counts against the parent like any failed attempt."""
    assert route("NO_NEW_SHAPE") == "reject"


def test_promotion_applies_the_rule_on_the_manual_path_too():
    same = rec(PARENT["invariants"] + ["gp3[0] == (raddr == 2'b01)"])
    buckets, rows = promote([PARENT], [same], compute_metrics=False)
    assert rows == []
    assert buckets["reject"] and buckets["reject"][0]["verdict"] == "NO_NEW_SHAPE"


def test_rows_may_now_grow_to_generation_six():
    """The cap was 3, two generations short of answers the size a real
    core check needs (about 20 clauses by the measured x1.45 a step)."""
    from batch import MAX_GEN, extendable
    assert MAX_GEN == 6
    row = {"id": "g5_000", "generation": 5, "top_module": "m",
           "verilog": "module m;\nendmodule", "invariants": ["a"]}
    assert extendable(row, None)
    row["generation"] = 6
    assert not extendable(row, None)
