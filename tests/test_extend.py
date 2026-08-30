"""Tests for the step-4 extension machinery (no API, no sby):
disposition auditing (silent clause deletion is rejected before grading),
second-property assert-set discipline, selective assert removal for the
P2-proves-unaided check, and coupling-clause detection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from extend import (                     # noqa: E402
    check_dispositions, split_second_property_asserts, remove_asserts,
    has_coupling_clause)

PARENT_INVS = ["tokens_hi == (tokens >= 3'd2)", "tokens <= 3'd4"]


def d(clause, status, replaced_by=(), reason=""):
    return {"clause": clause, "status": status,
            "replaced_by": list(replaced_by), "reason": reason}


# --- dispositions: every parent clause accounted for, or rejected ---

def test_dispositions_all_kept_ok():
    new = PARENT_INVS + ["pending <= 2'd2"]
    err = check_dispositions(PARENT_INVS, new,
                             [d(PARENT_INVS[0], "kept"),
                              d(PARENT_INVS[1], "kept")])
    assert err is None


def test_dispositions_missing_entry_is_silent_deletion():
    err = check_dispositions(PARENT_INVS, PARENT_INVS[:1],
                             [d(PARENT_INVS[0], "kept")])
    assert err is not None and "tokens <= 3'd4" in err


def test_dispositions_kept_but_absent_from_list():
    err = check_dispositions(PARENT_INVS, PARENT_INVS[:1],
                             [d(PARENT_INVS[0], "kept"),
                              d(PARENT_INVS[1], "kept")])
    assert err is not None and "kept" in err


def test_dispositions_superseded_ok():
    new = ["tokens_hi == (tokens >= 3'd2 || boost)", "tokens <= 3'd4"]
    err = check_dispositions(
        PARENT_INVS, new,
        [d(PARENT_INVS[0], "superseded",
           replaced_by=["tokens_hi == (tokens >= 3'd2 || boost)"],
           reason="tokens_hi has a second source now"),
         d(PARENT_INVS[1], "kept")])
    assert err is None


def test_dispositions_superseded_but_replacement_missing():
    err = check_dispositions(
        PARENT_INVS, ["tokens <= 3'd4"],
        [d(PARENT_INVS[0], "superseded", replaced_by=["something_else == 1"],
           reason="r"),
         d(PARENT_INVS[1], "kept")])
    assert err is not None


def test_dispositions_superseded_needs_reason():
    new = ["stronger", "tokens <= 3'd4"]
    err = check_dispositions(
        PARENT_INVS, new,
        [d(PARENT_INVS[0], "superseded", replaced_by=["stronger"], reason=""),
         d(PARENT_INVS[1], "kept")])
    assert err is not None and "reason" in err


def test_dispositions_unknown_clause_rejected():
    err = check_dispositions(
        PARENT_INVS, PARENT_INVS,
        [d(PARENT_INVS[0], "kept"), d(PARENT_INVS[1], "kept"),
         d("invented == 1", "kept")])
    assert err is not None and "invented" in err


# --- second property: parent asserts verbatim + exactly one new ---

def test_second_property_split_ok():
    parent = ["tokens != 3'd0"]
    child = ["tokens != 3'd0", "free <= 3'd4"]
    new, err = split_second_property_asserts(parent, child)
    assert err is None
    assert new == "free <= 3'd4"


def test_second_property_missing_new_assert():
    new, err = split_second_property_asserts(["a"], ["a"])
    assert err is not None


def test_second_property_parent_modified():
    new, err = split_second_property_asserts(["a == 1"], ["a == 2", "b"])
    assert err is not None


def test_second_property_two_new_asserts():
    new, err = split_second_property_asserts(["a"], ["a", "b", "c"])
    assert err is not None


# --- second property must read state the first does not ---

MODULE_REGS = ("module m (input wire clk);\n"
               "reg [3:0] par; reg [7:0] bin; reg [7:0] gray;\n"
               "localparam TOTAL = 9;\n"
               "endmodule")


def test_p2_new_ids_empty_when_support_is_a_subset():
    from extend import p2_new_ids
    assert p2_new_ids(["bin != 8'd0"], "bin <= 8'd4", MODULE_REGS) == set()


def test_p2_new_ids_finds_the_new_register():
    from extend import p2_new_ids
    # the g1_026 shape: par is state the first property never reads
    assert p2_new_ids(["gray == (bin >> 1) ^ bin"], "par == bin[0]",
                      MODULE_REGS) == {"par"}


def test_p2_new_ids_ignores_constants():
    from extend import p2_new_ids
    # TOTAL is a localparam: a new NAME but not new STATE
    assert p2_new_ids(["bin != 8'd0"], "bin <= TOTAL", MODULE_REGS) == set()


def test_grade_step4_rejects_second_property_with_same_support():
    from extend import grade_step4
    parent = {"verilog": ("module m (input wire clk, output reg a);\n"
                          "always @(posedge clk)\n"
                          "    assert (a == 1'b0);\n"
                          "endmodule"),
              "property": ["a == 1'b0"], "invariants": []}
    out = {"patch": ("@@     assert (a == 1'b0); @@\n"
                     "+ always @(posedge clk)\n"
                     "+     assert (!a || !a);"),
           "invariants": [], "dispositions": []}
    record = {}
    grade_step4(parent, "second", out, record)
    assert record["verdict"] == "P2_SAME_SUPPORT", record


# --- remove_asserts: blank a specific assertion, keep the file legal ---

def test_remove_asserts_removes_only_the_named_one():
    v = ("module m;\n"
         "always @(posedge clk) if (g) assert (a == b);\n"
         "always @(posedge clk) assert (c);\n"
         "endmodule\n")
    out = remove_asserts(v, ["a == b"])
    assert "a == b" not in out
    assert "assert (c);" in out
    assert "endmodule" in out


def test_remove_asserts_guarded_position_stays_legal():
    v = "module m;\nalways @(posedge clk) if (g) assert (x);\nendmodule\n"
    out = remove_asserts(v, ["x"])
    # the guarded statement still has a body (empty statement)
    assert "if (g) ;" in " ".join(out.split())


# --- coupling clause: mentions both new and existing identifiers ---

PARENT_IDS = {"tokens", "tokens_hi", "drip", "spend", "clk", "rst"}
NEW_IDS = {"pending"}


def test_coupling_clause_detected():
    assert has_coupling_clause(
        ["pending <= 2'd2", "tokens + pending <= 3'd4"], PARENT_IDS, NEW_IDS)


def test_no_coupling_when_new_state_isolated():
    assert not has_coupling_clause(
        ["pending <= 2'd2"], PARENT_IDS, NEW_IDS)


def test_dispositions_superseded_by_two_clauses_jointly():
    # the real g0_035 STAGE case: one parent clause replaced by two new
    # clauses together — must be accepted, not forced into one string
    new = ["(cnt != 8'd0) || (high_cnt == 8'd0)",
           "(cnt == 8'd0) || (high_cnt == ((cnt - 8'd1 < d) ? cnt - 8'd1 : d))"]
    err = check_dispositions(
        ["high_cnt == ((cnt < d) ? cnt : d)"], new,
        [d("high_cnt == ((cnt < d) ? cnt : d)", "superseded",
           replaced_by=new, reason="accumulator now trails the ramp by one")])
    assert err is None


def test_dispositions_superseded_empty_replacement_list():
    err = check_dispositions(
        PARENT_INVS, ["tokens <= 3'd4"],
        [d(PARENT_INVS[0], "superseded", replaced_by=[], reason="r"),
         d(PARENT_INVS[1], "kept")])
    assert err is not None


def test_identifiers_ignore_comment_words():
    from extend import identifiers
    v = ("reg deep;  // registered level flag, re-evaluated when it moves\n"
         "wire bulk = dec && deep;\n")
    ids = identifiers(v)
    assert "deep" in ids and "bulk" in ids and "dec" in ids
    assert "registered" not in ids and "when" not in ids and "moves" not in ids


# --- property-copy: an invariant identical to a property is a cheat ---

def test_property_copy_detected_after_whitespace_normalisation():
    from extend import property_copy
    props = ["{1'b0, hwm} + {1'b0, hdrm} == 5'd8"]
    invs = ["hwm <= 4'd8", "{1'b0,hwm}   + {1'b0,hdrm} ==  5'd8"]
    assert property_copy(invs, props) is not None


def test_property_copy_none_when_lists_disjoint():
    from extend import property_copy
    assert property_copy(["hwm >= sp"], ["tokens != 3'd0"]) is None


def test_identifiers_ignore_verilog_keywords():
    from extend import identifiers
    v = ("localparam [7:0] HI_LIMIT = 8'd192;\n"
         "parameter W = 4;\n"
         "case (sel) default: q <= 0; endcase\n")
    ids = identifiers(v)
    assert "HI_LIMIT" in ids and "sel" in ids and "q" in ids
    for kw in ("localparam", "parameter", "case", "endcase", "default"):
        assert kw not in ids, kw


# --- REPLICATE checks: per-instance coverage + the aggregate requirement.
# N per-instance copies of the parent family are EXPECTED (instance-internal
# asserts need them); what must ALSO exist is >=1 clause that is a genuine
# sum/reduction across instances — the shape the corpus has zero of. ---

def test_aggregate_clause_detects_sum_over_instances():
    from extend import aggregate_clause
    invs = ["t0 <= 3'd4", "t1 <= 3'd4",
            "({1'b0, free} + {2'b00, t0} + {2'b00, t1} + {2'b00, t2} + "
            "{2'b00, t3}) == 5'd8"]
    assert aggregate_clause(invs, 4) is not None


def test_aggregate_clause_rejects_pairwise_peer_shape():
    # the real PEER clause: relates TWO things, no reduction over N
    from extend import aggregate_clause
    invs = ["pri_b == (4'b0001 << ((idx + off) & 2'b11))",
            "pri == (4'b0001 << idx)"]
    assert aggregate_clause(invs, 4) is None


def test_aggregate_clause_bit_selects_count_as_indices():
    from extend import aggregate_clause
    assert aggregate_clause(
        ["(gnt[0] + gnt[1] + gnt[2] + gnt[3]) <= 2'd1"], 4) is not None


def test_instance_coverage_gaps_satisfied():
    from extend import instance_coverage_gaps
    parent = ["tokens_hi == (tokens >= 3'd2)", "tokens <= 3'd4"]
    child = (["h0 == (t0 >= 3'd2)", "h1 == (t1 >= 3'd2)"]
             + ["t0 <= 3'd4", "t1 <= 3'd4"]
             + ["({1'b0, free} + {2'b00, t0} + {2'b00, t1}) == 5'd8"])
    assert instance_coverage_gaps(parent, child, 2) == []


def test_instance_coverage_gaps_reports_missing_family():
    from extend import instance_coverage_gaps
    parent = ["tokens_hi == (tokens >= 3'd2)", "tokens <= 3'd4"]
    child = ["t0 <= 3'd4", "t1 <= 3'd4"]     # hi-relation family absent
    gaps = instance_coverage_gaps(parent, child, 2)
    assert len(gaps) == 1 and "tokens_hi" in gaps[0]


# --- COMPOSE gates ---

HOT_SRC = """\
module hot_src (
    input wire clk, input wire rst, input wire step,
    output wire [3:0] hot
);
    reg [1:0] sel;
    initial sel = 2'd0;
    always @(posedge clk)
        if (rst) sel <= 2'd0;
        else if (step) sel <= sel + 2'd1;
    assign hot = 4'b0001 << sel;
endmodule
"""

HOT_LATCH = """\
module hot_latch (
    input wire clk, input wire rst, input wire in_valid,
    input wire [3:0] in_data,
    output reg [3:0] held
);
    initial held = 4'b0001;
    always @(posedge clk)
        if (rst) held <= 4'b0001;
        else if (in_valid) held <= in_data;
    always @(posedge clk)
        assert (held == 4'b0001 || held == 4'b0010 ||
                held == 4'b0100 || held == 4'b1000);
endmodule
"""


def make_compose_parents():
    from build_corpus import extract_asserts as ea
    pa = {"id": "x_src", "top_module": "hot_src", "clock": "clk",
          "antecedents": [], "sanity_covers": [], "verilog": HOT_SRC,
          "property": ea(HOT_SRC), "invariants": []}
    pb = {"id": "x_latch", "top_module": "hot_latch", "clock": "clk",
          "antecedents": [], "sanity_covers": [], "verilog": HOT_LATCH,
          "property": ea(HOT_LATCH), "invariants": []}
    return pa, pb


WIRE_WRAPPER = """\
module hot_link (
    input wire clk, input wire rst, input wire step, input wire in_valid,
    output wire [3:0] held_o
);
    wire [3:0] hotw;
    hot_src  a (.clk(clk), .rst(rst), .step(step), .hot(hotw));
    hot_latch b (.clk(clk), .rst(rst), .in_valid(in_valid),
                 .in_data(hotw), .held(held_o));
    always @(posedge clk)
        if (in_valid) assert (held_o != 4'b1111);
endmodule
"""


def test_compose_plain_wire_caught_before_grading():
    # no invariant clause about glue state (there IS no glue state):
    # rejected structurally, before any sby run or API money
    from extend import grade_compose
    pa, pb = make_compose_parents()
    out = {"wrapper": WIRE_WRAPPER, "top_module": "hot_link",
           "antecedents": ["in_valid"], "sanity_covers": [],
           "invariants": ["held_o != 4'b0000"]}
    rec = grade_compose(pa, pb, out, {})
    assert rec["verdict"] == "NO_GLUE_CLAUSE"
    assert "result" not in rec     # oracle never ran


def test_compose_wrapper_must_be_new_module():
    from extend import grade_compose
    pa, pb = make_compose_parents()
    out = {"wrapper": WIRE_WRAPPER, "top_module": "hot_src",
           "antecedents": [], "sanity_covers": [], "invariants": []}
    rec = grade_compose(pa, pb, out, {})
    assert rec["verdict"] == "WRAPPER_ERROR"


# --- COMPOSE eligibility: a parent whose invariants mention non-port
# signals cannot be composed (wrapper invariants cannot reach inside an
# instance — hierarchical refs are fabricated wires). Check BEFORE paying. ---

def test_ports_of_token_bucket():
    from extend import ports_of
    from tests.test_patch import TOKEN_BUCKET
    ports = ports_of(TOKEN_BUCKET, "token_bucket")
    assert {"clk", "rst", "drip", "spend", "tokens", "tokens_hi"} <= ports


def test_compose_eligibility_token_bucket_ok():
    from extend import compose_hidden_signals
    from tests.test_patch import TOKEN_BUCKET
    row = {"top_module": "token_bucket", "verilog": TOKEN_BUCKET,
           "invariants": ["tokens_hi == (tokens >= 3'd2)", "tokens <= 3'd4"]}
    assert compose_hidden_signals(row) == set()


def test_compose_eligibility_hidden_state_flagged():
    from extend import compose_hidden_signals
    v = ("module wd (input wire clk, input wire rst, output reg fault);\n"
         "reg [3:0] timer;\ninitial timer = 4'd8;\ninitial fault = 0;\n"
         "always @(posedge clk) begin end\nendmodule\n")
    row = {"top_module": "wd", "verilog": v,
           "invariants": ["timer <= 4'd8", "fault == (timer == 4'd0)"]}
    assert compose_hidden_signals(row) == {"timer"}


def test_compose_dotted_invariants_get_their_own_verdict():
    # dotted clauses must be named as the problem — not surface as a
    # confusing "coverage missing" for clauses that are visibly present
    from extend import grade_compose
    pa, pb = make_compose_parents()
    out = {"wrapper": WIRE_WRAPPER.replace("hot_link", "hot_link2"),
           "top_module": "hot_link2",
           "antecedents": [], "sanity_covers": [],
           "invariants": ["u_a.sel <= 2'd3"]}
    rec = grade_compose(pa, pb, out, {})
    assert rec["verdict"] == "HIERARCHICAL_REF"
    assert "u_a.sel" in rec["error"]


# --- diff-decoration stripping: the model sometimes wraps a plain-module
# answer in unified-diff syntax (---/+++/@@ headers, + prefixes) out of
# patch-format habit. Same philosophy as fence-stripping: clean it,
# don't waste the call. ---

DIFF_DECORATED = """\
--- a/design.sv
+++ b/design.sv
@@
+// a comment
+module wrapped_top (
+    input wire clk
+);
+    reg r;
+    initial r = 1'b0;
+endmodule
"""


def test_strip_diff_decoration_recovers_module():
    from extend import strip_diff_decoration
    out = strip_diff_decoration(DIFF_DECORATED)
    assert out.splitlines()[0] == "// a comment"
    assert "module wrapped_top (" in out
    assert "+" not in out and "@@" not in out


def test_strip_diff_decoration_leaves_plain_verilog_alone():
    from extend import strip_diff_decoration
    plain = "module m (\n    input wire clk\n);\n    reg r;\nendmodule\n"
    assert strip_diff_decoration(plain) == plain
