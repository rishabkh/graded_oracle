"""The reset check both sides now share (reset_reads.py, 23 Oct 2026).

Review findings of 23 Oct 2026:
- the generator's check missed a reset read through a wire that copies it
  (`wire in_reset = !rst_n`): our checker said NECESSARY and EBMC, which
  scores the trained model, said REFUTED;
- it skipped an assertion with a label (`p0: assert (...)`) whole, so not
  even a direct read of the reset in it was seen;
- the extender had no check at all, so the check moved next to
  fact_format.py, where the extender and the fixer can ask it too.
No model, no checker: these read the design text only. The last test runs
our checker and EBMC on the wire copy and skips when they are missing."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import reset_reads as RR                                        # noqa: E402
import run as R                                                 # noqa: E402
from oracle.sby import sby_available                            # noqa: E402

# The reviewer's design: an asynchronous reset, written the way the
# asynchronous-reset trait asks. BODY is replaced by each test.
PAIR = """module rst_pair (input wire clk, input wire rst_n, input wire en);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a <= 3'd0;
            b <= 3'd0;
        end else if (en) begin
            a <= (a == 3'd4) ? 3'd0 : a + 3'd1;
            b <= (a == 3'd4) ? 3'd0 : a + 3'd1;
        end
    end
BODY
endmodule
"""


def pair(body, **fields):
    triple = {"verilog": PAIR.replace("BODY", body), "top_module": "rst_pair",
              "clock": "clk", "antecedents": [], "sanity_covers": [],
              "invariants": ["a == b", "a <= 3'd4"],
              "cti_explanation": "a=5, b=4", "cti_state": []}
    return dict(triple, **fields)


# --- the generator's check now uses the shared module ----------------------

def test_the_generator_asks_the_shared_module():
    assert R.RESET_NAMES is RR.RESET_NAMES
    assert R.reset_names is RR.reset_names
    assert R.reset_reads is RR.reset_reads


# --- a reset carried by another signal ---------------------------------------

def test_a_wire_that_copies_the_reset_is_a_reset():
    t = pair("    wire in_reset = !rst_n;\n"
             "    always @(posedge clk) assert "
             "((!in_reset || a == 3'd0) && b <= 3'd4);")
    reason = R.reset_gate(t)
    assert reason.startswith(
        "the reset input rst_n is read by the property `assert ((!in_reset "
        "|| a == 3'd0) && b <= 3'd4);` (through in_reset, which carries the "
        "reset)")
    assert "or a signal that carries one" in reason


def test_a_copy_of_a_copy_is_followed():
    t = pair("    wire live = rst_n;\n"
             "    wire calm;\n"
             "    assign calm = live & en;\n"
             "    always @(posedge clk) assert (b <= 3'd4);",
             invariants=["a == b", "!calm || a <= 3'd4"])
    assert "fact `!calm || a <= 3'd4` (through calm, which carries the " \
        "reset)" in R.reset_gate(t)
    names = RR.reset_names(t["verilog"], "clk")
    assert names == {"rst_n": {"rst_n"}, "live": {"rst_n"},
                     "calm": {"rst_n"}}


@pytest.mark.parametrize("block", [
    "    reg live;\n    always @(*) live = rst_n && en;",
    "    reg live;\n    always @* begin\n        live = 1'b0;\n"
    "        if (rst_n) live = en;\n    end",
    "    logic live;\n    always_comb live = rst_n;",
    "    reg live;\n    always @(rst_n or en) live = rst_n & en;",
])
def test_a_signal_set_in_a_combinational_block_that_reads_it(block):
    t = pair(block + "\n    always @(posedge clk) if (live) "
                     "assert (b <= 3'd4);")
    assert "the property's guard `(live)` (through live, which carries " \
        "the reset)" in R.reset_gate(t)


def test_a_port_connected_to_the_reset_and_what_it_drives():
    """rst_n goes into a sub-module's port by name; the sub-module drives
    an output from it; the top's wire on that output carries the reset."""
    sub = """module watch (input wire clk_i, input wire rst_ni,
              input wire en_i, output wire quiet_o);
    reg seen = 1'b0;
    always @(posedge clk_i or negedge rst_ni)
        if (!rst_ni) seen <= 1'b0;
        else if (en_i) seen <= 1'b1;
    assign quiet_o = !rst_ni || !seen;
endmodule
"""
    t = pair("    wire quiet;\n"
             "    watch u_w (.clk_i(clk), .rst_ni(rst_n), .en_i(en),"
             " .quiet_o(quiet));\n"
             "    always @(posedge clk) assert (b <= 3'd4);",
             invariants=["a == b", "quiet || a <= 3'd4"])
    t["verilog"] = sub + t["verilog"]
    names = RR.reset_names(t["verilog"], "clk")
    assert {"rst_n", "rst_ni", "quiet_o", "quiet"} <= set(names)
    # the sub-module's clock port is a clock, not a reset
    assert "clk_i" not in names and "clk" not in names
    assert "fact `quiet || a <= 3'd4` (through quiet, which carries the " \
        "reset)" in R.reset_gate(t)


def test_a_define_that_names_the_reset():
    t = pair("    `define LIVE rst_n\n"
             "    always @(posedge clk) assert "
             "((`LIVE || a == 3'd0) && b <= 3'd4);")
    assert "(through LIVE, which carries the reset)" in R.reset_gate(t)


def test_a_register_loaded_from_the_reset_is_not_followed():
    t = pair("    reg rst_q = 1'b1;\n"
             "    always @(posedge clk) rst_q <= rst_n;\n"
             "    always @(posedge clk) assert (b <= 3'd4);",
             invariants=["a == b", "a <= 3'd4", "rst_q || a == 3'd0"])
    assert R.reset_gate(t) is None
    assert "rst_q" not in RR.reset_names(t["verilog"], "clk")


def test_a_synchronous_reset_and_its_copies_are_not_resets():
    """A reset with no edge in a sensitivity list is read the same way by
    both checkers. Since 23 Oct 2026 (Rishab chose) no side counts it:
    the generator turned away a direct read until then, never a wire made
    from it."""
    sync = """module s (input wire clk, input wire rst, input wire en);
    reg [2:0] x = 3'd0;
    wire live = !rst && en;
    always @(posedge clk) if (rst) x <= 3'd0; else if (live) x <= x + 3'd1;
    always @(posedge clk) assert (x <= 3'd7);
endmodule
"""
    t = {"verilog": sync, "clock": "clk", "antecedents": [],
         "invariants": ["!live || x <= 3'd7"]}
    assert R.reset_gate(t) is None
    assert R.reset_gate(dict(t, invariants=["rst || x <= 3'd7"])) is None
    assert RR.reset_names(sync, "clk") == {}
    # with the usual names, the old rule, a copy is still not followed
    assert "rst" in RR.reset_names(sync, "clk", usual_names=True)
    assert "live" not in RR.reset_names(sync, "clk", usual_names=True)


# --- which edge is the clock ---------------------------------------------------

def test_the_edge_the_first_if_tests_is_the_reset_the_other_the_clock():
    v = """module m (input wire ck, input wire clr);
    reg [1:0] q = 2'd0;
    always @(posedge ck or posedge clr)
        if (clr) q <= 2'd0; else q <= q + 2'd1;
endmodule
"""
    # the row says its clock is "clk", but the design's is ck: ck is still
    # read as the clock, from the block itself
    assert RR.reset_names(v, "clk") == {"clr": {"clr"}}


def test_a_signal_alone_in_a_sensitivity_list_is_a_clock():
    v = """module m (input wire ck, input wire arst_l);
    reg q = 1'b0;
    reg r = 1'b0;
    always @(posedge ck) r <= !r;
    always @(posedge ck or negedge arst_l)
        q <= arst_l ? !q : 1'b0;
endmodule
"""
    # no leading if here: every edge but the known clocks is a reset
    assert RR.reset_names(v, "clk") == {"arst_l": {"arst_l"}}


# --- an assertion with a label -------------------------------------------------

def test_an_assertion_with_a_label_is_read():
    reason = R.reset_gate(pair("    always @(posedge clk) "
                               "p0: assert ((rst_n || a == 3'd0) && "
                               "b <= 3'd4);"))
    assert reason.startswith("the reset input rst_n is read by the property "
                             "`assert ((rst_n || a == 3'd0) && b <= 3'd4);`")


def test_a_labelled_assertion_inside_a_guard_keeps_its_guard():
    reason = R.reset_gate(pair("    always @(posedge clk) begin\n"
                               "        if (rst_n) begin\n"
                               "            p_bound: assert (b <= 3'd4);\n"
                               "        end\n"
                               "    end"))
    assert "the property's guard `(rst_n)`" in reason


def test_a_labelled_assertion_that_leaves_the_reset_alone_passes():
    assert R.reset_gate(pair("    always @(posedge clk) "
                             "p0: assert (b <= 3'd4);")) is None
    [((text, _), under)] = RR.assertions(
        PAIR.replace("BODY", "    always @(posedge clk) "
                             "p0: assert (b <= 3'd4);"))
    assert text == "assert (b <= 3'd4);"
    assert [kind for kind, *_ in under] == ["block"]


def test_odd_text_never_crashes_and_finds_no_reset():
    for odd in ("", "module m; wire a = ; assign = rst_n; endmodule",
                "`define\nmodule m; always @(posedge clk or", None, 3):
        assert RR.reset_names(odd, "clk") == {}
        assert RR.reset_reads(odd, "clk", ["rst_n"]) == []
        assert RR.step_reset_error(odd, "clk", ["rst_n"]) is None


# --- the two checkers on the wire copy (sby and EBMC) --------------------------

EBMC_BIN = os.environ.get("EBMC_PATH")


@pytest.mark.skipif(not sby_available(),
                    reason="sby not on PATH - activate hwtools first")
@pytest.mark.skipif(not (EBMC_BIN and Path(EBMC_BIN).is_file()),
                    reason="needs EBMC_PATH pointing at the ebmc binary")
def test_the_wire_copy_splits_the_checkers_and_never_reaches_ours(tmp_path):
    from tests.test_generator_reset_and_fact_gates import ebmc, ours
    copy = pair("    wire in_reset = !rst_n;\n"
                "    always @(posedge clk) assert "
                "((!in_reset || a == 3'd0) && b <= 3'd4);")
    assert ours(copy, tmp_path / "ours") == "NECESSARY"
    verdict, out = ebmc(copy, [], "timing_without_lemmas", tmp_path)
    assert verdict == "CEX" and "REFUTED" in out
    j = R.judge(json.dumps(copy), "end_turn",
                "globally: the property applies in every state", grade=True,
                proof_root=tmp_path / "judge", spin=False)
    assert j["verdict"] == "READS_RESET"
    assert not (tmp_path / "judge").exists()


def test_a_block_with_two_resets_finds_both():
    v = """module m (input wire clk, input wire rst_n, input wire clr);
    reg [1:0] q = 2'd0;
    always @(posedge clk or negedge rst_n or posedge clr)
        if (!rst_n) q <= 2'd0;
        else if (clr) q <= 2'd1;
        else q <= q + 2'd1;
endmodule
"""
    assert RR.reset_names(v, "clk") == {"rst_n": {"rst_n"}, "clr": {"clr"}}
