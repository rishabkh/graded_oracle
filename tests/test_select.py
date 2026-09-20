"""Entropy-style selection, ported from formal-disco: train on the third
of the corpus that is least like the rest, so each self-improvement round
is pulled towards shapes the pile does not already have."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from select_rows import (feature_sets, pooled_stats, select_top, surprisals)

PLAIN = ("module m (input wire clk, input wire i_ce);\n"
         "  reg [3:0] count = 0;\n"
         "  always @(*) assert (count <= 4'd9);\nendmodule\n")
PAST = ("module m (input wire clk, input wire i_ce);\n"
        "  reg pv = 0, out = 0, in = 0;\n"
        "  always @(posedge clk) if (pv) assert (out == $past(in));\n"
        "endmodule\n")


def row(rid, invariants, verilog=PLAIN, prop=None):
    return {"id": rid, "verilog": verilog,
            "property": prop or ["count <= 4'd9"],
            "invariants": invariants,
            "metrics": {"state_bits": 8}}


def test_feature_sets_names_the_assertion_form_it_finds():
    assert "plain" in feature_sets(row("a", ["x == 1"]))["assert_form"]
    assert "past" in feature_sets(row("b", ["x == 1"], PAST))["assert_form"]


def test_a_rare_shape_is_more_surprising_than_a_common_one():
    rows = [row(f"r{i}", ["count <= 4'd9"]) for i in range(5)]
    rows.append(row("odd", ["(a - b) == {1'b0, c}"]))
    pooled = pooled_stats(rows)
    common = surprisals(rows[0], pooled)["invariant_template"]
    rare = surprisals(rows[-1], pooled)["invariant_template"]
    assert rare > common


def test_select_top_keeps_the_unusual_row_and_drops_the_copies():
    rows = [row(f"r{i}", ["count <= 4'd9"]) for i in range(5)]
    rows.append(row("odd", ["(a - b) == {1'b0, c}"], PAST))
    kept = select_top(rows, fraction=1 / 3)
    assert len(kept) == 2
    assert "odd" in [r["id"] for r in kept]


def test_select_top_keeps_corpus_order():
    rows = [row(f"r{i}", [f"x{i} == {i}"]) for i in range(6)]
    kept = select_top(rows, fraction=0.5)
    assert [r["id"] for r in kept] == sorted(r["id"] for r in kept)


def test_fraction_of_one_keeps_everything_and_a_bad_fraction_raises():
    rows = [row(f"r{i}", ["count <= 4'd9"]) for i in range(4)]
    assert len(select_top(rows, fraction=1.0)) == 4
    with pytest.raises(ValueError):
        select_top(rows, fraction=0)
