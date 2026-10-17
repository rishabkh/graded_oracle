"""Stricter checks on a proved child (18 Oct 2026).

Reading all 28 children of the 17 Oct growth trial by hand found weak and
wrong designs the checker cannot see, because it proves what is written,
not what is meant: new registers that only copy existing ones, copies whose
results go nowhere or are XOR-ed together to keep them alive, glue that
controls nothing, fact lists restated per copy until they reach 202. Each
check below rejects one of those, and none of them rejected any of the 10
good children (notes_2026-10-18/strict_checks)."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import gates                                                    # noqa: E402
import promote                                                  # noqa: E402

needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None,
                                 reason="yosys not on PATH (run hwtools)")

LEAF = """module leaf (
    input  wire        clk,
    input  wire        en,
    output reg  [31:0] data,
    output reg         flag
);
    always @(posedge clk) if (en) begin
        data <= data + 32'd1;
        flag <= ~flag;
    end
endmodule
"""


def wrapper_row(wrapper, invariants=("pool <= 4'd3",), top="top"):
    return {"verilog": LEAF + "\n" + wrapper, "top_module": top,
            "invariants": list(invariants)}


def record(wrapper, ext_type="replicate", invariants=("pool <= 4'd3",),
           **kw):
    r = {"verdict": "NECESSARY", "ext_type": ext_type, "wrapper": wrapper,
         "child_verilog": LEAF + "\n" + wrapper,
         "invariants": list(invariants), "new_shapes": ["N <= N"]}
    r.update(kw)
    return r


# a pool that grants the copies' enables, uses every result, and keeps
# its own count: the shape of the one good copy in the trial
GOOD = """module top (
    input  wire        clk,
    input  wire [1:0]  req,
    output wire [31:0] d0, d1,
    output wire [1:0]  f,
    output reg  [3:0]  pool
);
    wire g0 = req[0] && (pool != 4'd0);
    wire g1 = req[1] && !req[0] && (pool != 4'd0);
    leaf u0 (.clk(clk), .en(g0), .data(d0), .flag(f[0]));
    leaf u1 (.clk(clk), .en(g1), .data(d1), .flag(f[1]));
    always @(posedge clk) pool <= pool - {3'd0, g0 | g1};
endmodule
"""


# --- 1. a new register that only copies an existing one ---------------------

def test_a_new_register_equal_to_an_existing_one_is_a_shadow():
    """c1_008 kept xfer_cnt == b_wr_cnt; c2_001 job_wd == m_deadline."""
    new = {"xfer_cnt", "job_wd"}
    assert gates.shadow_facts(["xfer_cnt == b_wr_cnt",
                               "{8'd0, job_wd} == m_deadline"], new) == \
        ["xfer_cnt == b_wr_cnt", "{8'd0, job_wd} == m_deadline"]


def test_sums_differences_and_old_facts_are_not_shadows():
    new = {"pool", "iss", "ret"}
    assert gates.shadow_facts(["pool + h0 + h1 == 8'd3",
                               "(iss - ret) == busy",
                               "cnt == {1'b0, rd_ptr}",
                               "pool <= 4'd3"], new) == []


def test_the_new_registers_of_a_patch_are_those_the_parent_lacks():
    parent = {"verilog": "module m; reg [3:0] cnt; logic a; endmodule"}
    rec = {"ext_type": "structural",
           "child_verilog": "module m; reg [3:0] cnt, half; logic a, b;"
                            " endmodule"}
    assert gates.new_registers(rec, [parent]) == {"half", "b"}


# --- 2. copies' results XOR-ed together to keep them alive ------------------

def test_copies_xored_into_one_output_are_caught():
    w = GOOD.replace("output wire [31:0] d0, d1,",
                     "output wire [31:0] mix,") \
        .replace(".data(d0)", ".data(q0)").replace(".data(d1)", ".data(q1)") \
        .replace("    wire g0", "    wire [31:0] q0, q1;\n"
                                  "    assign mix = q0 ^ q1;\n    wire g0")
    assert gates.xor_of_copies(w, LEAF + w, "top")


def test_a_parity_of_ordinary_signals_is_not_flagged():
    w = GOOD.replace("output reg  [3:0]  pool",
                     "output reg  [3:0]  pool,\n    output wire par") \
        .replace("    always", "    assign par = req[0] ^ req[1] ^ g0;\n"
                               "    always")
    assert gates.xor_of_copies(w, LEAF + w, "top") == []


# --- 3. glue that controls nothing -----------------------------------------

def test_glue_that_drives_a_copy_controls_it():
    assert gates.glue_controls(GOOD, LEAF + GOOD, "top")


def test_a_pool_that_only_watches_the_copies_controls_nothing():
    """c3_002: credits counted, but every copy runs on the raw inputs."""
    idle = GOOD.replace(".en(g0)", ".en(req[0])").replace(".en(g1)",
                                                          ".en(req[1])")
    assert not gates.glue_controls(idle, LEAF + idle, "top")


# --- 4. copies whose state drives no output ---------------------------------

@needs_yosys
def test_copies_whose_results_are_used_keep_their_state():
    assert gates.copied_state_used(LEAF + GOOD, "top", GOOD) > 0.9


@needs_yosys
def test_copies_whose_data_goes_nowhere_lose_most_of_their_state():
    """c4_001 kept 4% of its copies' flip-flops, c3_001 31%."""
    dropped = GOOD.replace(".data(d0)", ".data()").replace(".data(d1)",
                                                           ".data()")
    assert gates.copied_state_used(LEAF + dropped, "top", dropped) < 0.2


# --- 5. the fact list -------------------------------------------------------

def test_more_than_64_facts_is_too_many():
    assert gates.MAX_FACTS == 64


# --- applying the checks ----------------------------------------------------

def test_each_failed_check_names_its_verdict():
    shadow = record(GOOD, ext_type="compose",
                    invariants=["pool == f_count"],
                    parent_state_bits=[1, 1])
    gates.apply(shadow, [], use_yosys=False)
    assert shadow["verdict"] == "SHADOW_REGISTER"
    many = record(GOOD, invariants=[f"x{i} <= 4'd3" for i in range(65)])
    gates.apply(many, [], use_yosys=False)
    assert many["verdict"] == "TOO_MANY_FACTS"
    idle = record(GOOD.replace(".en(g0)", ".en(req[0])")
                  .replace(".en(g1)", ".en(req[1])"))
    gates.apply(idle, [], use_yosys=False)
    assert idle["verdict"] == "IDLE_GLUE"


def test_a_good_child_passes_and_the_result_is_kept():
    ok = record(GOOD)
    gates.apply(ok, [], use_yosys=False)
    assert ok["verdict"] == "NECESSARY"
    assert ok["gates"]["failed"] is None


def test_only_a_proved_child_is_judged_and_only_once(monkeypatch):
    failed = record(GOOD, verdict="NOT_PROVEN")
    gates.apply(failed, [], use_yosys=False)
    assert failed["verdict"] == "NOT_PROVEN" and "gates" not in failed
    calls = []
    monkeypatch.setattr(gates, "glue_controls",
                        lambda *a: calls.append(1) or True)
    ok = record(GOOD)
    gates.apply(ok, [], use_yosys=False)
    gates.apply(ok, [], use_yosys=False)
    assert len(calls) == 1


def test_the_new_verdicts_are_rejections_not_repairs():
    for v in gates.VERDICTS:
        assert promote.route(v) == "reject", v


def test_the_rule_now_runs_the_checks_after_the_new_kind_of_fact_rule():
    parent = {"id": "c0_000", "verilog": "module m; endmodule",
              "invariants": ["cnt <= 4'd9"]}
    rec = record(GOOD, ext_type="compose", invariants=["pool == f_count",
                                                       "a + b == c"])
    promote.ratchet(rec, [parent])
    assert rec["verdict"] == "SHADOW_REGISTER"
