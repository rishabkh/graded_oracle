"""The fake-state checker: is a claimed counterexample to induction real?

A toy with two 3-bit counters a and b that move together. The property
says a == 3 implies b == 3, and the invariant that closes it is a == b.
Each named case below has a known answer, worked out by hand, and runs
for both assertion styles plus a design that mixes them.

The sby cases skip when sby is not on PATH (source the oss-cad-suite
environment first). The input checks run without it.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from cti_check import (VERDICTS, assert_styles, check_state,   # noqa: E402
                       design_style, failed_lines, pin_block)

requires_sby = pytest.mark.skipif(
    shutil.which("sby") is None,
    reason="sby not on PATH - source oss-cad-suite/environment first")

PROP = "!(a == 3'd3) || (b == 3'd3)"

ASSERTS = {
    "clocked": f"    always @(posedge clk) assert ({PROP});\n",
    "comb": f"    always @(*) assert ({PROP});\n",
    # both kinds of assertion; the side one is always true
    "mixed_clocked_prop": (f"    always @(posedge clk) assert ({PROP});\n"
                           "    always @(*) assert (a <= 3'd7);\n"),
    "mixed_comb_prop": (f"    always @(*) assert ({PROP});\n"
                        "    always @(posedge clk) assert (a <= 3'd7);\n"),
}


def toy(asserts, clock="clk"):
    return (f"module toy (input wire {clock}, input wire i_ce);\n"
            "    reg [2:0] a = 3'd0;\n"
            "    reg [2:0] b = 3'd0;\n"
            f"    always @(posedge {clock}) if (i_ce) begin\n"
            "        a <= a + 3'd1;\n"
            "        b <= b + 3'd1;\n"
            "    end\n"
            + asserts.replace("posedge clk", f"posedge {clock}") +
            "endmodule\n")


def row(asserts, clock="clk"):
    return {"verilog": toy(asserts, clock), "top_module": "toy",
            "invariants": ["a == b"]}


def st(**pins):
    return [{"signal": k, "value": v} for k, v in pins.items()]


CASES = [
    ("real", st(a="3'd2", b="3'd0"), "REAL"),
    ("pfalse", st(a="3'd3", b="3'd0"), "P_FALSE"),
    ("rtrue", st(a="3'd2", b="3'd2"), "R_TRUE"),
    ("nobreak", st(a="3'd0", b="3'd5"), "NO_BREAK"),
    ("impossible", [{"signal": "a", "value": "3'd2"},
                    {"signal": "a", "value": "3'd4"}], "STATE_IMPOSSIBLE"),
]


@requires_sby
@pytest.mark.parametrize("design", list(ASSERTS))
@pytest.mark.parametrize("name,state,expected", CASES,
                         ids=[c[0] for c in CASES])
def test_toy_case(design, name, state, expected):
    out = check_state(row(ASSERTS[design]), state, timeout_s=60)
    assert out["verdict"] == expected, out["detail"]
    assert out["ok"] is (expected == "REAL")
    assert out["style"] == design.split("_")[0]
    assert out["detail"] and "\n" not in out["detail"]
    assert out["wall_s"] >= 0


@requires_sby
def test_no_break_feedback_points_at_inputs_held_idle():
    # the most common reason an original fake state failed: the enable was
    # named at its idle value, so the one step changed nothing
    out = check_state(row(ASSERTS["clocked"]),
                      st(a="3'd2", b="3'd0", i_ce="1'b0"), timeout_s=60)
    assert out["verdict"] == "NO_BREAK", out["detail"]
    assert "input" in out["detail"] and "idle value" in out["detail"]


@requires_sby
def test_mixed_design_with_a_two_step_break_is_not_one_step():
    # one step from a=1, b=0 gives 2, 1: fine. Only the second step
    # reaches a=3, b=2, which breaks the same-cycle property
    out = check_state(row(ASSERTS["mixed_comb_prop"]), st(a="3'd1", b="3'd0"),
                      timeout_s=60)
    assert out["style"] == "mixed"
    assert out["verdict"] == "NOT_ONE_STEP", out["detail"]
    assert out["ok"] is False


@requires_sby
def test_mixed_design_where_every_step_breaks_the_clocked_property():
    # no enable, so from a=2, b=0 every step reaches a=3, b=1. sby makes
    # clocked assumes act at once by default, so the same-cycle run must
    # not demand the clocked asserts of the state after the step too
    src = toy(ASSERTS["mixed_clocked_prop"]).replace(" if (i_ce)", "")
    r = {"verilog": src, "top_module": "toy", "invariants": ["a == b"]}
    out = check_state(r, st(a="3'd2", b="3'd0"), timeout_s=60)
    assert out["style"] == "mixed"
    assert out["verdict"] == "REAL", out["detail"]
    out = check_state(r, st(a="3'd3", b="3'd0"), timeout_s=60)
    assert out["verdict"] == "P_FALSE", out["detail"]


@requires_sby
def test_clock_name_does_not_matter():
    out = check_state(row(ASSERTS["clocked"], clock="i_clk"),
                      st(a="3'd2", b="3'd0"), timeout_s=60)
    assert out["verdict"] == "REAL", out["detail"]


@requires_sby
def test_module_level_assert_property_after_a_clocked_block_is_comb():
    # sits after an always @(posedge) block but is outside it
    asserts = f"    assert property ({PROP});\n"
    r = row(asserts)
    assert design_style(r["verilog"]) == "comb"
    out = check_state(r, st(a="3'd2", b="3'd0"), timeout_s=60)
    assert out["verdict"] == "REAL", out["detail"]


@requires_sby
def test_assert_inside_a_sub_module():
    src = ("module chk (input wire clk, input wire [2:0] x, "
           "input wire [2:0] y);\n"
           "    always @(posedge clk) assert (!(x == 3'd3) || (y == 3'd3));\n"
           "endmodule\n"
           "module toy (input wire clk, input wire i_ce);\n"
           "    reg [2:0] a = 3'd0;\n"
           "    reg [2:0] b = 3'd0;\n"
           "    always @(posedge clk) if (i_ce) begin\n"
           "        a <= a + 3'd1;\n"
           "        b <= b + 3'd1;\n"
           "    end\n"
           "    chk u_chk (.clk(clk), .x(a), .y(b));\n"
           "endmodule\n")
    r = {"verilog": src, "top_module": "toy", "invariants": ["a == b"]}
    out = check_state(r, st(a="3'd2", b="3'd0"), timeout_s=60)
    assert out["style"] == "clocked"
    assert out["verdict"] == "REAL", out["detail"]
    out = check_state(r, st(a="3'd2", b="3'd2"), timeout_s=60)
    assert out["verdict"] == "R_TRUE", out["detail"]


@requires_sby
def test_memory_word_can_be_pinned_even_with_an_initial_loop():
    # without memory_map the start value from the initial loop sticks,
    # and every pin on a memory word reads as an impossible state
    src = ("module toy (input wire clk, input wire i_ce, input wire i_sel);\n"
           "    reg [2:0] mem [0:1];\n"
           "    integer k;\n"
           "    initial for (k = 0; k < 2; k = k + 1) mem[k] = 3'd0;\n"
           "    always @(posedge clk) if (i_ce) mem[i_sel] <= mem[i_sel] + 3'd1;\n"
           "    always @(posedge clk) assert (mem[1] != 3'd3);\n"
           "endmodule\n")
    r = {"verilog": src, "top_module": "toy", "invariants": ["mem[1] <= 3'd1"]}
    out = check_state(r, st(**{"mem[1]": "3'd2", "i_ce": "1'b1",
                               "i_sel": "1'b1"}), timeout_s=60)
    assert out["verdict"] == "REAL", out["detail"]


@requires_sby
def test_calls_from_several_threads_do_not_mix():
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(c[1], c[2]) for c in CASES] * 2
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda j: check_state(row(ASSERTS["clocked"]),
                                                 j[0], timeout_s=60), jobs))
    assert [o["verdict"] for o in outs] == [j[1] for j in jobs]


@requires_sby
def test_keep_dir_keeps_the_pinned_design(tmp_path):
    check_state(row(ASSERTS["clocked"]), st(a="3'd2", b="3'd0"),
                timeout_s=60, keep_dir=tmp_path)
    designs = list(tmp_path.rglob("design.sv"))
    assert designs
    text = designs[0].read_text()
    assert "assume (a == 3'd2);" in text
    assert "assume (!((a == b)));" in text


# --- input checks, no sby needed ---

@pytest.mark.parametrize("state", [
    [], "a=2", None, [{"signal": "a"}], [{"signal": "", "value": "3'd1"}],
    [{"signal": "a", "value": 2}], ["a"],
], ids=["empty", "string", "none", "no_value", "empty_signal", "int_value",
        "not_dict"])
def test_malformed_state_is_bad_state(state):
    out = check_state(row(ASSERTS["clocked"]), state)
    assert out["verdict"] == "BAD_STATE"
    assert out["ok"] is False
    assert out["detail"] and "\n" not in out["detail"]


@pytest.mark.parametrize("pins", [
    {"$past(a)": "3'd1"},
    {"a (held)": "3'd1"},
    {"a": "3'd1 (on the breaking step)"},
    {"a": "b"},
    {"a": "3'bx01"},
])
def test_annotated_or_non_constant_pins_are_bad_state(pins):
    out = check_state(row(ASSERTS["clocked"]), st(**pins))
    assert out["verdict"] == "BAD_STATE", out["detail"]


@pytest.mark.parametrize("value", ["3'd8", "4'd20", "2'b111", "4'h1F",
                                   "1'b10", "3'o17"])
def test_a_value_too_big_for_its_size_is_bad_state(value):
    # Verilog quietly cuts 4'd20 down to 4'd4, so the claim would be
    # checked as a different number from the one written in the reasoning
    out = check_state(row(ASSERTS["clocked"]), st(a=value, b="3'd0"))
    assert out["verdict"] == "BAD_STATE", out["detail"]
    assert value in out["detail"]


@requires_sby
@pytest.mark.parametrize("value", ["3'd2", "8'd2", "3'b010", "3'b0_10",
                                   "2", "3'sd2", "3 'd 2"])
def test_value_forms_that_fit_are_accepted(value):
    out = check_state(row(ASSERTS["clocked"]), st(a=value, b="3'd0"),
                      timeout_s=60)
    assert out["verdict"] == "REAL", out["detail"]


def test_unknown_signal_is_caught_before_running():
    out = check_state(row(ASSERTS["clocked"]), st(a="3'd2", c="3'd0"))
    assert out["verdict"] == "UNKNOWN_SIGNAL"
    assert "c" in out["detail"]


def test_select_checks_the_base_name():
    out = check_state(row(ASSERTS["clocked"]), st(**{"c[1]": "1'b0"}))
    assert out["verdict"] == "UNKNOWN_SIGNAL"


def test_sub_module_path_is_unknown_signal():
    out = check_state(row(ASSERTS["clocked"]), st(**{"u_chk.x": "3'd0"}))
    assert out["verdict"] == "UNKNOWN_SIGNAL"


def test_pin_block_keeps_the_full_select():
    text = pin_block(st(**{"mem[1]": "3'd2", "x[3:0]": "4'h5"}),
                     ["a == b", "b <= 3'd4"])
    assert "assume (mem[1] == 3'd2);" in text
    assert "assume (x[3:0] == 4'h5);" in text
    assert "assume (!((a == b) && (b <= 3'd4)));" in text
    assert "if ($initstate) begin" in text


def test_styles_of_each_assert():
    src = ("module m (input clk, input i_clk);\n"
           "    reg a; // always @(posedge clk) assert (a);\n"
           "    always @(posedge i_clk) begin\n"
           "        if (a) begin\n"
           "            assert (a);\n"
           "        end\n"
           "    end\n"
           "    always @(*)\n"
           "        if (a) assert (a);\n"
           "        else assert (!a);\n"
           "    assign b = a;\n"
           "    always @(posedge clk) a <= 1'b1;\n"
           "    assert property (a);\n"
           "endmodule\n")
    assert assert_styles(src) == {5: "clocked", 9: "comb", 10: "comb",
                                  13: "comb"}
    assert design_style(src) == "mixed"
    assert design_style("module m; endmodule\n") == "none"


def test_failed_lines_reads_sub_module_and_span_forms():
    log = ("SBY 1 [job] engine_0: ##   0:00:00  Assert failed in toy: "
           "design.sv:5.27-5.63 (_witness_.check_assert_design_sv_5_6)\n"
           "SBY 1 [job] engine_0: ##   0:00:00  Assert failed in toy.u_chk: "
           "design.sv:4.76-6.50 (_witness_.x)\n")
    assert failed_lines(log) == [(5, 5), (4, 6)]


# --- real data ---

CORPUS = ROOT / "extender" / "corpus.jsonl"
ATTEMPTS = ROOT / "initiator" / "logs" / "attempts.jsonl"


@requires_sby
@pytest.mark.skipif(not (CORPUS.exists() and ATTEMPTS.exists()),
                    reason="corpus or attempts log missing")
def test_original_state_of_g0_000_gets_a_known_verdict():
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines()
            if line.strip()]
    r = next(x for x in rows if x["id"] == "g0_000")
    state = None
    for line in ATTEMPTS.read_text().splitlines():
        a = json.loads(line)
        if (a.get("run_id"), a.get("attempt")) == (r["source_run_id"],
                                                   r["source_attempt"]):
            state = json.loads(a["raw_json"])["cti_state"]
    assert state is not None
    out = check_state(r, state, timeout_s=120)
    assert out["verdict"] in VERDICTS
    assert out["verdict"] not in ("ERROR", "TIMEOUT"), out["detail"]


# --- keep everything (8 Oct 2026) ------------------------------------------

@requires_sby
def test_a_kept_check_says_where_its_proof_is(tmp_path):
    r = dict(row(ASSERTS["clocked"]), id="g0_009")
    out = check_state(r, st(a="3'd2", b="3'd0"), timeout_s=60,
                      keep_dir=tmp_path)
    assert out["verdict"] == "REAL"
    folder = tmp_path / out["proof_dir"]
    assert out["proof_dir"].startswith("g0_009_cti_")
    assert list(folder.rglob("logfile.txt"))
    assert list(folder.rglob("sby_stderr.txt"))
