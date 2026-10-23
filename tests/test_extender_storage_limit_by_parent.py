"""How big an array a step may add (22 Oct 2026, review findings F1, F5, F8).

F1  The 16-entry limit sent back storage sized by the parent's own design:
    of the 6 steps of the 21 Oct real-systems trial it sent back, 5 held
    one entry for each of the 24 slaves of a design that already had a
    24-entry queue (t008 and t026 were kept as good designs); only t028's
    32-byte replay memory was a free choice. A new array may now hold as
    many entries as the parent's own biggest array, and the error and the
    question name that array.
F5  A packed array of words (reg [255:0][7:0] dly) escaped the limit; its
    outer dimensions now count as entries. A flat vector (reg [191:0] dly)
    cannot be told from a wide word and is left alone.
F8  A 2-D array counts every entry (channels x depth), which neither the
    rules nor the error said; the error told the model to use a depth of 4
    to 8, which it already had. Both now say that the sizes multiply.
The model and the checker are faked: no calls, no proofs."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import extend                                                   # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict, Tier,  # noqa: E402
                          TripleResult)


def squash(text):
    return " ".join(text.split())


def module(body, name="top"):
    return f"module {name} (input wire clk);\n{body}endmodule\n"


SLAVES = module("    reg [4:0] q_slv [0:23];\n    reg [7:0] cfg [0:3];\n")


# --- F1: the parent's own biggest array --------------------------------------

def test_a_new_array_may_hold_as_many_entries_as_the_parents_biggest():
    assert extend.new_array_limit(SLAVES, ["top"]) == \
        (24, ("top", "q_slv", 24))
    per_slave = SLAVES.replace("endmodule",
                               "    reg [7:0] dly [0:23];\nendmodule")
    assert extend.too_deep(SLAVES, per_slave, "top") == []
    one_more = SLAVES.replace("endmodule",
                              "    reg [7:0] dly [0:24];\nendmodule")
    assert extend.too_deep(SLAVES, one_more, "top") == [("top", "dly", 25)]


def test_with_only_small_arrays_in_the_parent_the_limit_stays_16():
    small = module("    reg [7:0] cfg [0:7];\n")
    assert extend.new_array_limit(small, ["top"]) == (16, ("top", "cfg", 8))
    assert extend.new_array_limit(module(""), ["top"]) == (16, None)
    child = small.replace("endmodule", "    reg [7:0] t [0:16];\nendmodule")
    assert extend.too_deep(small, child, "top") == [("top", "t", 17)]


def test_the_limit_is_the_parents_size_not_what_the_step_made_it():
    """A step cannot raise its own limit by making an old array deeper;
    since 23 Oct 2026 (review finding) the old array made deeper is sent
    back too, where before it was never judged."""
    child = SLAVES.replace("[0:23]", "[0:63]").replace(
        "endmodule", "    reg [7:0] big [0:39];\nendmodule")
    assert extend.too_deep(SLAVES, child, "top") == [("top", "big", 40),
                                                      ("top", "q_slv", 64)]


def test_the_error_names_the_array_whose_size_a_new_one_may_match():
    child = SLAVES.replace("endmodule",
                           "    reg [7:0] replay [0:31];\nendmodule")
    rec = {}
    assert extend.storage_too_deep(rec, SLAVES, child, "top")
    error = squash(rec["error"])
    assert error.startswith("the step adds storage of more than 24 entries")
    assert "replay in module top (32 entries)" in error
    assert "design's own biggest array (q_slv in module top, 24 entries)" \
        in error
    assert "one entry for each of the same things" in error
    assert "never more than 24" in error
    assert rec["too_deep"] == [{"module": "top", "array": "replay",
                                "entries": 32}]


def test_a_small_parents_error_does_not_offer_its_size():
    small = module("    reg [7:0] cfg [0:7];\n")
    child = small.replace("endmodule", "    reg [7:0] t [0:31];\nendmodule")
    rec = {}
    assert extend.storage_too_deep(rec, small, child, "top")
    assert "biggest array" not in rec["error"]
    assert "never more than 16" in rec["error"]


A_PART = """\
module part_a (input wire clk, input wire push, output wire [7:0] head);
    reg [7:0] slot_q [0:23];
    assign head = slot_q[0];
endmodule
"""
B_PART = """\
module part_b (input wire clk, input wire [7:0] din, output wire [7:0] dout);
    reg [7:0] r [0:3];
    assign dout = r[0];
endmodule
"""


def glue(last):
    return f"""\
module pair (input wire clk, input wire push, output wire [7:0] out);
    wire [7:0] h;
    reg [7:0] hold_q [0:{last}];
    part_a a (.clk(clk), .push(push), .head(h));
    part_b b (.clk(clk), .din(hold_q[0]), .dout(out));
endmodule
"""


def test_a_combination_may_match_either_parts_biggest_array():
    both = A_PART + B_PART
    tops = ["part_a", "part_b"]
    assert extend.new_array_limit(both, tops) == \
        (24, ("part_a", "slot_q", 24))
    assert extend.too_deep(both, both + glue(23), "pair", tops) == []
    assert extend.too_deep(both, both + glue(24), "pair", tops) == \
        [("pair", "hold_q", 25)]
    # judged from the wrapper's top alone, the parts' sizes would not count:
    # that is why the graders pass the parts' own tops (since 23 Oct 2026,
    # when every array is judged, even the part's own queue would be sent
    # back)
    assert extend.new_array_limit(both, ["pair"]) == (16, None)
    assert extend.too_deep(both, both + glue(23), "pair") == \
        [("pair", "hold_q", 24), ("part_a", "slot_q", 24)]


@pytest.fixture
def checker(monkeypatch):
    """A checker that says NECESSARY; counts the checks."""
    seen = {"graded": 0}

    def triple(payload, **kw):
        seen["graded"] += 1
        return TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")
    monkeypatch.setattr(extend, "state_bits", lambda v, top: len(v))
    monkeypatch.setattr(extend, "grade_triple_generated", triple)
    monkeypatch.setattr(extend, "grade",
                        lambda *a, **k: GradeResult(tier=Tier.NOT_INDUCTIVE,
                                                    reason="alone"))
    return seen


def part(row_id, verilog):
    from build_corpus import extract_asserts
    top = re.search(r"module\s+(\w+)", verilog).group(1)
    return {"id": row_id, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [], "verilog": verilog,
            "property": extract_asserts(verilog), "invariants": []}


def test_the_combination_grader_judges_against_both_parts(checker):
    """grade_compose passes both parts' tops, so its glue may be as big as
    either part's biggest array, and no bigger."""
    pa, pb = part("x_a", A_PART), part("x_b", B_PART)

    def graded(last):
        out = {"wrapper": glue(last), "top_module": "pair",
               "antecedents": [], "sanity_covers": [], "invariants": []}
        return extend.grade_compose(pa, pb, out, {})
    assert graded(23)["verdict"] != "STORAGE_TOO_DEEP"
    rec = graded(24)
    assert rec["verdict"] == "STORAGE_TOO_DEEP"
    assert "slot_q in module part_a, 24 entries" in squash(rec["error"])


def test_the_copies_grader_judges_against_the_copied_design(checker):
    pa = part("x_a", A_PART)

    def pool(last):
        wrapper = f"""\
module pool (input wire clk, input wire push, output wire [7:0] h0);
    wire [7:0] h1;
    reg [7:0] seen_q [0:{last}];
    part_a u0 (.clk(clk), .push(push), .head(h0));
    part_a u1 (.clk(clk), .push(push), .head(h1));
endmodule
"""
        out = {"wrapper": wrapper, "top_module": "pool", "antecedents": [],
               "sanity_covers": [], "invariants": []}
        return extend.grade_replicate(pa, out, {}, 2)
    assert pool(23)["verdict"] != "STORAGE_TOO_DEEP"
    assert pool(24)["verdict"] == "STORAGE_TOO_DEEP"


def ctr_with_slaves():
    verilog = """module ctr (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg  [3:0] cnt
);
    reg [3:0] lim;
    reg [4:0] q_slv [0:23];
    initial cnt = 4'd0;
    initial lim = 4'd9;
    always @(posedge clk) begin
        if (rst)
            cnt <= 4'd0;
        else if (en)
            cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1;
    end
    always @(posedge clk) assert (cnt <= 4'd9);
endmodule
"""
    return {"id": "c0_7", "top_module": "ctr", "clock": "clk",
            "antecedents": [], "sanity_covers": [], "verilog": verilog,
            "property": ["cnt <= 4'd9"], "invariants": ["lim == 4'd9"]}


@pytest.mark.parametrize("move", ["STORAGE", "SUBUNIT", "PEER", "STAGE"])
def test_the_question_says_how_big_a_new_array_may_be(move):
    text = squash(extend.build_prompt(ctr_with_slaves(), "structural", move,
                                      k=4))
    assert "This design already has an array of 24 entries (q_slv in " \
        "module ctr)" in text
    assert "may hold up to 24 entries" in text
    assert "one entry for each of the same things" in text
    assert "never more than 16" in text


def test_a_question_on_a_design_with_small_arrays_says_nothing_more():
    row = ctr_with_slaves()
    row["verilog"] = row["verilog"].replace("[0:23]", "[0:7]")
    text = squash(extend.build_prompt(row, "structural", "STORAGE"))
    assert "already has an array" not in text


def test_the_rules_say_a_new_array_may_match_the_modules_biggest():
    text = squash(extend.SYSTEM_PROMPT)
    assert "unless the module already has a bigger array" in text
    assert "as many entries as the module's biggest array" in text


# --- F1 replayed on the 21 Oct trial -----------------------------------------

TRIAL_LOG = ROOT / "extender/logs/real_systems_trial_2026-10-21/" \
    "extensions.jsonl"
TRIAL_CORPUS = ROOT / "extender/corpus_real_systems_trial_2026-10-21.jsonl"


@pytest.mark.skipif(not (TRIAL_LOG.exists() and TRIAL_CORPUS.exists()),
                    reason="the 21 Oct trial's files are not on this machine")
def test_on_the_trial_only_the_free_choice_is_still_sent_back():
    """Every grown design of the trial, judged against its own parent as
    the grader judges it: the 16-entry rule alone sends back 6 steps, the
    rule with the parent's own size only t028."""
    rows = {}
    for line in TRIAL_CORPUS.read_text().splitlines():
        row = json.loads(line)
        rows[row["id"]] = row
    by_16, by_parent = set(), set()
    for line in TRIAL_LOG.read_text().splitlines():
        rec = json.loads(line)
        child = rec.get("child_verilog")
        if not child:
            continue
        parents = [rows[rec["parent_id"]]] + (
            [rows[rec["parent2_id"]]] if rec.get("parent2_id") else [])
        pv = "\n".join(p["verilog"] for p in parents)
        tops = [p["top_module"] for p in parents]
        top = (re.findall(r"\bmodule\s+(\w+)", child)[-1]
               if rec["ext_type"] in ("compose", "replicate") else tops[0])
        step = rec["extension_id"][-4:]
        old = extend.declared_arrays(pv)
        if any(n > extend.MAX_NEW_DEPTH for key, n in
               extend.storage_depths(child, top).items() if key not in old):
            by_16.add(step)
        if extend.too_deep(pv, child, top, tops):
            by_parent.add(step)
    assert by_16 == {"t008", "t014", "t026", "t028", "t029", "t035"}
    assert by_parent == {"t028"}


# --- F5: a packed array of words ---------------------------------------------

def test_a_packed_array_counts_its_outer_dimensions():
    text = module("    reg [255:0][7:0] dly;\n"
                  "    logic [3:0][7:0] w [0:7];\n"
                  "    logic [1:0][3:0][7:0] x, y;\n"
                  "    reg [24*8-1:0] flat;\n"           # a wide word
                  "    reg [7:0] word;\n")
    shapes = extend.array_shapes(text, "top")
    assert shapes[("top", "dly")] == (256, [256])
    assert shapes[("top", "w")] == (32, [4, 8])
    assert shapes[("top", "x")] == shapes[("top", "y")] == (8, [2, 4])
    assert ("top", "flat") not in shapes and ("top", "word") not in shapes
    assert extend.too_deep(module(""), text, "top") == [
        ("top", "dly", 256), ("top", "w", 32)]


def test_declarations_that_hold_no_state_are_not_arrays():
    text = """\
module top (
    input  logic [31:0][7:0] din,
    output logic [3:0][7:0] q
);
    typedef logic [31:0][7:0] block_t;
    localparam logic [31:0][7:0] ZERO = '0;
    function automatic logic [31:0][7:0] swap(input logic [31:0][7:0] v);
        swap = v;
    endfunction
endmodule
"""
    assert extend.storage_depths(text, "top") == {("top", "q"): 4}


PACKED_DELAY = """\
@@ reg [3:0] lim; @@
+    reg [{last}:0][3:0] dly;
+    initial dly = '0;
@@ cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1; @@
= end
+    always @(posedge clk)
+        if (en) dly <= {{dly[{last} - 1:0], cnt}};"""


def small_counter():
    row = ctr_with_slaves()
    row["verilog"] = row["verilog"].replace(
        "    reg [4:0] q_slv [0:23];\n", "")
    return row


@pytest.mark.parametrize("last, verdict", [(31, "STORAGE_TOO_DEEP"),
                                           (7, "NECESSARY")])
def test_a_step_adding_a_packed_delay_line_is_judged(checker, last, verdict):
    out = {"patch": PACKED_DELAY.format(last=last),
           "invariants": ["lim == 4'd9", "dly[0] <= 4'd9"],
           "dispositions": [{"clause": "lim == 4'd9", "status": "kept",
                             "replaced_by": [], "reason": ""}]}
    rec = extend.grade_step4(small_counter(), "structural", out,
                             {"ext_type": "structural", "move": "STORAGE"})
    assert rec["verdict"] == verdict, rec.get("error")
    if verdict == "STORAGE_TOO_DEEP":
        assert "dly in module ctr (32 entries)" in rec["error"]
        assert checker["graded"] == 0


# --- F8: the entries of one array count together -----------------------------

def test_rule_11_says_the_entries_of_one_array_count_together():
    text = squash(extend.SYSTEM_PROMPT)
    assert "All the entries of one array count together" in text
    assert "4 channels of 8 entries are 32 entries" in text
    assert "reg [3:0][7:0] w" in text


def test_the_peer_move_says_channels_times_depth_count():
    text = squash(extend.MOVES["PEER"])
    assert "All the entries of one array count together" in text
    assert "holds CHANNELS times DEPTH entries" in text
    assert "never more than 16" in text


CHANNELS = module("    localparam CHANNELS = 4;\n    localparam DEPTH = 8;\n"
                  "    reg [7:0] buf_q [0:CHANNELS-1][0:DEPTH-1];\n")


def test_the_error_names_the_total_and_its_dimensions():
    rec = {}
    assert extend.storage_too_deep(rec, module(""), CHANNELS, "top")
    error = squash(rec["error"])
    assert "buf_q in module top (32 entries: 4 x 8, every dimension " \
        "counted)" in error
    assert "the sizes of its dimensions multiply" in error
    assert "Use fewer channels or a smaller depth" in error
    assert "never more than 16 in the whole array" in error


def test_a_one_dimension_error_has_no_channel_advice():
    rec = {}
    child = module("    reg [7:0] t [0:31];\n")
    assert extend.storage_too_deep(rec, module(""), child, "top")
    assert "t in module top (32 entries)." in rec["error"]
    assert "channels" not in rec["error"] and " x " not in rec["error"]
