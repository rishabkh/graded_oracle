"""Storage written in other ways counts for the depth rule (23 Oct 2026,
review finding).

The rule that a step's new array holds at most 16 entries (or as many as
the design's own biggest array) was escaped, each at 256 or 1024 entries,
by: an array of int, byte, shortint or longint; an array whose entries are
a type the design declares (typedef, struct), or a typedef that is itself
an array; a depth set by a `define; registers declared inside a generate
loop (c0_102 of 9 Oct builds 256 slots that way); and a new copy of a
module the parent already has, or an old instance, given a deeper size,
never judged because its array was "old". All are now counted, and every
array is judged by the size the step builds it at.
Reads design text only: no model, no checker."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import extend                                                   # noqa: E402

PARENT = """module top(input clk, input rst, input [7:0] d, output reg [7:0] q);
always @(posedge clk) if (rst) q <= 0; else q <= d;
endmodule
"""


def with_decl(decl, parent=PARENT):
    return parent.replace("endmodule\n", decl + "\nendmodule\n", 1)


def judged(decl, parent=PARENT, top="top"):
    rec = {}
    deep = extend.storage_too_deep(rec, parent, with_decl(decl, parent), top)
    return deep, rec


# --- the ways of writing storage that escaped ----------------------------------

@pytest.mark.parametrize("word", ["int", "byte", "shortint", "longint",
                                  "int unsigned"])
def test_arrays_of_the_other_integer_types_count(word):
    deep, rec = judged(f"{word} m [0:255];")
    assert deep and rec["too_deep"] == [{"module": "top", "array": "m",
                                        "entries": 256}]
    assert not judged(f"{word} m [0:15];")[0]


def test_a_loop_variable_or_an_argument_is_not_storage():
    assert extend.module_arrays(
        "always @(*) for (int i = 0; i < 300; i = i + 1) x[i] = 0;\n"
        "function automatic int f(int a [0:255]); f = a[0]; endfunction") == []


@pytest.mark.parametrize("decl", [
    "typedef logic [7:0] byte_t;\nbyte_t tm [0:255];",
    "typedef struct packed { logic [7:0] a; logic v; } ent_t;\n"
    "ent_t sm [0:255];",
    "typedef enum logic [1:0] {IDLE, RUN} st_t;\nst_t st_q [0:255];",
])
def test_an_array_of_a_type_the_design_declares_counts(decl):
    deep, rec = judged(decl)
    assert deep and rec["too_deep"][0]["entries"] == 256


def test_a_type_that_is_itself_an_array_counts_its_entries():
    deep, rec = judged("typedef logic [7:0] mem_t [0:255];\nmem_t tm2;")
    assert rec["too_deep"] == [{"module": "top", "array": "tm2",
                                "entries": 256}]
    # a packed array type counts its outer dimension, as a variable does
    assert extend.declared_types("typedef logic [3:0][7:0] w4_t;") == \
        {"w4_t": ["3:0"]}
    assert extend.storage_depths(with_decl(
        "typedef logic [3:0][7:0] w4_t;\nw4_t w [0:7];"), "top") == \
        {("top", "w"): 32}


def test_a_type_declared_outside_the_module_counts_too():
    child = "typedef logic [7:0] byte_t;\n" + with_decl("byte_t tm [0:31];")
    rec = {}
    assert extend.storage_too_deep(rec, PARENT, child, "top")
    assert rec["too_deep"][0]["entries"] == 32


def test_the_members_of_a_struct_are_not_arrays():
    text = with_decl("typedef struct packed { logic [3:0][7:0] w; } s_t;\n"
                     "s_t one;")
    assert extend.storage_depths(text, "top") == {}


def test_a_depth_set_by_a_define_is_read():
    deep, rec = judged("`define DEPTH 256\nreg [7:0] mm [0:`DEPTH-1];")
    assert rec["too_deep"] == [{"module": "top", "array": "mm",
                                "entries": 256}]
    nested = "`define HALF 8\n`define DEPTH (`HALF*2)\nreg [7:0] m [0:`DEPTH-1];"
    assert not judged(nested)[0]                       # 16: allowed
    assert extend.storage_depths(with_decl(nested), "top") == \
        {("top", "m"): 16}


# --- registers made by a generate loop --------------------------------------

LOOP = """genvar g;
generate for (g = 0; g < N; g = g + 1) begin : slot
    reg [7:0] r;
    always @(posedge clk) r <= d;
end endgenerate"""


@pytest.mark.parametrize("n, deep", [(16, False), (17, True), (256, True)])
def test_a_register_inside_a_generate_loop_counts_once_per_pass(n, deep):
    got, rec = judged(LOOP.replace("N", str(n)))
    assert got is deep
    if deep:
        assert rec["too_deep"] == [{"module": "top", "array": "r",
                                    "entries": n}]
        assert "built once for each pass of the loop" in rec["error"]


@pytest.mark.parametrize("header, passes", [
    ("genvar i = 0; i < 20; i = i + 1", 20),
    ("genvar i = 0; i <= 20; i++", 21),
    ("genvar i = 2; i < 20; i += 2", 9),
    ("genvar i = 19; i >= 0; i = i - 1", 20),
    ("genvar i = 0; i != 20; ++i", 20),
])
def test_the_passes_of_every_usual_loop_header(header, passes):
    text = with_decl(f"for ({header}) begin : s\n    logic [3:0] x;\nend")
    assert extend.storage_depths(text, "top") == {("top", "x"): passes}


def test_nested_loops_multiply_and_an_array_inside_multiplies_too():
    text = with_decl("genvar a, b;\n"
                     "for (a = 0; a < 4; a = a + 1) begin : ch\n"
                     "    reg [7:0] head;\n"
                     "    for (b = 0; b < 4; b = b + 1) begin : lane\n"
                     "        reg [7:0] buf_q [0:3];\n"
                     "    end\n"
                     "end")
    assert extend.storage_depths(text, "top") == {("top", "head"): 4,
                                                  ("top", "buf_q"): 64}


def test_a_loop_inside_an_always_block_makes_no_registers():
    text = with_decl("reg [7:0] acc;\ninteger k;\n"
                     "always @(posedge clk) for (k = 0; k < 300; k = k + 1)"
                     " acc <= acc + 8'd1;")
    assert extend.storage_depths(text, "top") == {}
    # nor does the loop variable of such a loop inside a generate loop
    text = with_decl("genvar g;\n"
                     "for (g = 0; g < 32; g = g + 1) begin : s\n"
                     "    reg [7:0] r;\n"
                     "    always @(*) for (int i = 0; i < 4; i++) r[i] = d[i];\n"
                     "end")
    assert extend.storage_depths(text, "top") == {("top", "r"): 32}


def test_the_9_oct_design_built_by_a_generate_loop_is_measured():
    """c0_102 builds 256 slots in a generate loop; before, the rule saw no
    storage in it at all."""
    import json
    corpus = ROOT / "extender" / "corpus_2026-10-09.jsonl"
    if not corpus.exists():
        pytest.skip("no 9 Oct corpus here")
    row = next(json.loads(l) for l in corpus.read_text().splitlines()
               if '"id": "c0_102"' in l)
    assert extend.storage_depths(row["verilog"], row["top_module"]) == \
        {("brmask_alloc", "m"): 256}
    assert extend.generated_registers(row["verilog"]) == \
        {("brmask_alloc", "m")}


# --- registers built by a generate loop do not raise the limit ---------------
# 23 Oct 2026, Rishab chose: c0_102's 256 slots would have let any new array
# of a step on it hold 256 entries. They may keep their own size.

def test_registers_built_by_a_generate_loop_do_not_raise_the_limit():
    parent = with_decl(LOOP.replace("N", "64"))
    assert extend.too_deep(PARENT, parent, "top") == [("top", "r", 64)]
    assert extend.new_array_limit(parent, ["top"]) == (16, None)
    # a new array of 64 entries is too deep, as on a design without the loop
    rec = {}
    assert extend.storage_too_deep(rec, parent,
                                   with_decl("reg [7:0] t [0:63];", parent),
                                   "top")
    assert rec["too_deep"] == [{"module": "top", "array": "t",
                                "entries": 64}]
    # the parent's own 64 slots are never blamed, and may not grow
    assert extend.too_deep(parent, with_decl("reg [7:0] t [0:7];", parent),
                           "top") == []
    deeper = with_decl(LOOP.replace("N", "128"))
    rec = {}
    assert extend.storage_too_deep(rec, parent, deeper, "top")
    error = " ".join(rec["error"].split())
    assert "r in module top (128 entries), which held 64 entries before " \
        "this step" in error
    assert "or than it already was when it was bigger" in error


def test_a_declared_array_still_raises_the_limit_beside_a_generate_loop():
    parent = with_decl("reg [7:0] tbl [0:23];\n" + LOOP.replace("N", "64"))
    assert extend.new_array_limit(parent, ["top"]) == (24, ("top", "tbl", 24))
    assert extend.too_deep(parent, with_decl("reg [7:0] t [0:23];", parent),
                           "top") == []
    assert extend.too_deep(parent, with_decl("reg [7:0] t [0:24];", parent),
                           "top") == [("top", "t", 25)]


def test_the_9_oct_design_built_by_a_generate_loop_keeps_the_usual_limit():
    """c0_102 allowed 256 entries for any new array before; now 16, and a
    step that leaves its 256 slots alone is not blamed for them."""
    import json
    corpus = ROOT / "extender" / "corpus_2026-10-09.jsonl"
    if not corpus.exists():
        pytest.skip("no 9 Oct corpus here")
    row = next(json.loads(l) for l in corpus.read_text().splitlines()
               if '"id": "c0_102"' in l)
    v, top = row["verilog"], row["top_module"]
    assert extend.new_array_limit(v, [top]) == (16, None)
    assert extend.too_deep(v, v, top) == []
    grown = v.replace("endmodule", "    reg [7:0] t [0:255];\nendmodule", 1)
    assert [n for *_, n in extend.too_deep(v, grown, top)] == [256]


# --- an old module or array given a deeper size --------------------------------

FIFO = """module fifo #(parameter DEPTH = 4) (input clk, input [7:0] d, output [7:0] q);
reg [7:0] mem [0:DEPTH-1];
assign q = mem[0];
always @(posedge clk) mem[0] <= d;
endmodule
module top(input clk, input rst, input [7:0] d, output [7:0] q);
fifo #(.DEPTH(4)) u0 (.clk(clk), .d(d), .q(q));
endmodule
"""


def test_a_new_copy_of_an_old_module_with_a_deeper_size_is_judged():
    child = FIFO.replace(
        "fifo #(.DEPTH(4)) u0 (.clk(clk), .d(d), .q(q));",
        "fifo #(.DEPTH(4)) u0 (.clk(clk), .d(d), .q(q));\nwire [7:0] q2;\n"
        "fifo #(.DEPTH(1024)) u1 (.clk(clk), .d(d), .q(q2));")
    rec = {}
    assert extend.storage_too_deep(rec, FIFO, child, "top")
    assert rec["too_deep"] == [{"module": "fifo", "array": "mem",
                                "entries": 1024}]
    error = " ".join(rec["error"].split())
    assert "mem in module fifo (1024 entries), which held 4 entries before " \
        "this step" in error
    assert "An array the design already has may not be made deeper than " \
        "that either" in error


def test_an_old_instance_made_deeper_is_judged_and_a_small_copy_is_not():
    deeper = FIFO.replace("#(.DEPTH(4)) u0", "#(.DEPTH(1024)) u0")
    assert extend.too_deep(FIFO, deeper, "top") == [("fifo", "mem", 1024)]
    small = FIFO.replace(
        "fifo #(.DEPTH(4)) u0 (.clk(clk), .d(d), .q(q));",
        "fifo #(.DEPTH(4)) u0 (.clk(clk), .d(d), .q(q));\nwire [7:0] q2;\n"
        "fifo #(.DEPTH(8)) u1 (.clk(clk), .d(d), .q(q2));")
    assert extend.too_deep(FIFO, small, "top") == []


def test_what_the_parent_already_built_is_never_blamed():
    """The limit is never below the parent's own size of an array, so a
    step that leaves a big old array alone passes."""
    big = FIFO.replace("#(.DEPTH(4)) u0", "#(.DEPTH(64)) u0")
    grown = big.replace("endmodule\nmodule top",
                        "endmodule\nmodule spare(input clk);\nendmodule\n"
                        "module top")
    assert extend.too_deep(big, grown, "top") == []


def test_rule_11_says_old_arrays_and_generate_loops_count():
    text = " ".join(extend.SYSTEM_PROMPT.split())
    assert ("unless the module already has a bigger array, not counting "
            "registers built by a generate loop:") in text
    assert ("An existing array may not be made deeper than that either, or "
            "than it already is when it is bigger, and a register declared "
            "inside a generate loop counts once for each pass of the loop."
            ) in text
