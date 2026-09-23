"""Which parent gets extended next.

Today the frontier is shuffled, so depth is an accident of how many
generation-zero rows there are. The free rounds need depth on purpose,
without letting one lineage eat a whole run: the corpus's deepest rows
already trace back to a single REPLICATE, and a naive depth-first order
would keep feeding it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from frontier import family_of, order_frontier


def row(rid, gen, parent=None, invariants=("count <= 4'd9",), bits=8):
    return {"id": rid, "generation": gen, "parent": parent,
            "verilog": f"module {rid} (input clk);\n  reg [3:0] c;\nendmodule",
            "property": [f"c <= 4'd{gen}"], "invariants": list(invariants),
            "metrics": {"state_bits": bits}}


CORPUS = [
    row("g0_000", 0),
    row("g1_000", 1, "g0_000"), row("g2_000", 2, "g1_000"),
    row("g3_000", 3, "g2_000"),
    row("g0_001", 0), row("g1_001", 1, "g0_001"),
    row("g0_002", 0, invariants=["(a - b) == {1'b0, c}"]),
]
BY_ID = {r["id"]: r for r in CORPUS}


def test_a_rows_family_is_its_generation_zero_ancestor():
    assert family_of(BY_ID["g3_000"], BY_ID) == "g0_000"
    assert family_of(BY_ID["g0_002"], BY_ID) == "g0_002"


def test_deeper_rows_come_first():
    order = [r["id"] for r in order_frontier(CORPUS, BY_ID, per_family=99)]
    assert order.index("g3_000") < order.index("g0_001")


def test_no_family_takes_more_than_its_share():
    order = order_frontier(CORPUS, BY_ID, per_family=2)
    picked = [family_of(r, BY_ID) for r in order]
    assert picked.count("g0_000") <= 2


def test_every_family_gets_a_turn_before_any_repeats():
    order = order_frontier(CORPUS, BY_ID, per_family=2)
    first_three = {family_of(r, BY_ID) for r in order[:3]}
    assert len(first_three) == 3


def test_rarity_breaks_ties_between_equal_depths():
    """g0_002 carries a shape nothing else has; among generation-zero
    rows it should be preferred."""
    order = [r["id"] for r in order_frontier(CORPUS, BY_ID, per_family=99)]
    assert order.index("g0_002") < order.index("g0_001")
