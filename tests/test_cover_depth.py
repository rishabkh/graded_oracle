"""How far the cover check looks for a property's condition (8 Oct 2026).

The proof stays at 20 steps. Only the "can this condition happen at all"
check looks further, by the design's storage size or the largest count a
condition names: a 64-slot buffer needs 64 pushes before "full"."""
from oracle.depth import COVER_DEPTH_CAP, suggest_cover_depth


def sd(src, exprs=(), depth=20):
    return suggest_cover_depth(src, list(exprs), depth)


def test_plain_storage():
    assert sd("reg [31:0] mem [0:15];") == 36


def test_reversed_bounds():
    assert sd("reg [7:0] mem [31:0];") == 52


def test_storage_sized_by_a_named_constant():
    src = "localparam int DEPTH = 64;\nlogic [W-1:0] q [DEPTH];"
    assert sd(src) == 84


def test_storage_sized_by_a_parameter_expression():
    src = "module m #(parameter N = 2**5) ();\n reg [7:0] buf_q [0:N-1];"
    assert sd(src) == 52


def test_two_dimensional_storage_counts_every_slot():
    assert sd("reg [7:0] m [0:3][0:7];") == 52


def test_a_count_named_in_a_condition():
    assert sd("reg [6:0] count;", ["count == 7'd48"]) == 68
    assert sd("reg [5:0] p;", ["p == 6'h3F"]) == 83


def test_the_bigger_of_storage_and_count_wins():
    assert sd("reg [7:0] m [0:15];", ["n == 40"]) == 60


def test_small_designs_get_no_deeper_look():
    assert sd("reg [3:0] count;", ["count == 4'd5"]) == 25
    assert sd("reg [3:0] count;") == 20


def test_huge_counts_are_ignored_and_the_depth_is_capped():
    assert sd("reg [63:0] c;", ["c == 64'hFFFFFFFFFFFFFFFF"]) == 20
    assert sd("reg [7:0] m [0:4095];") == COVER_DEPTH_CAP


def test_unreadable_sizes_are_skipped_not_fatal():
    assert sd("reg [7:0] m [0:SIZE_FROM_A_PACKAGE-1];") == 20
    assert sd("") == 20


def test_a_named_constant_in_a_condition():
    src = "localparam DEPTH = 48;\nreg [5:0] count;"
    assert sd(src, ["count == DEPTH"]) == 68


def test_a_named_constant_in_the_design_counts_too():
    # a timer that starts at PERIOD and counts down: "remaining == 0" names
    # no count, but needs PERIOD steps
    src = "localparam [63:0] PERIOD = 64'd48;\nreg [63:0] remaining;"
    assert sd(src, ["remaining == 64'd0"]) == 68
