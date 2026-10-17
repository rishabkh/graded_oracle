"""The extender, made to grow designs fast (17 Oct 2026).

Measured 15 Oct (notes_2026-10-15/extender_size): a typical real block is
about 1,900 lines over 5 modules; ours are about 70 lines in one module,
and an accepted step added a median of 12 lines. The changes tested here:
  1. a port step: internal registers a design's facts mention get an
     output port, so 229 of the 252 new designs can be combined or copied
     (was 9);
  2. the size count merges the modules first (three 8-bit counters and a
     4-bit pool read as 40 flip-flop bits, not 28);
  3. the branch stop reads the main check only, at the checker's own
     300 s (it read the second-opinion run, capped at 120 s, against 60 s);
  4. combining and copying are drawn more often, and lineages go 12 deep;
  5. bigger moves, sized so one step grows a design by half to three
     quarters;
  6. a child that grew by half or more is kept without a new kind of fact.
Plus a run that chases lineages (the newest child is extended next) from
named starting designs, which is what a trial of deep growth needs."""
import random
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
import promote                                                  # noqa: E402
from build_corpus import state_bits, without_main_s             # noqa: E402

needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None,
                                 reason="yosys not on PATH (run hwtools)")

FIFO = """module fifo_ctl #(parameter W = 8) (
    input  wire clk,
    input  wire rst,
    input  wire push,   // a comment with a ) in it
    input  wire pop,
    input  wire [W-1:0] din,
    output wire [W-1:0] dout,
    output wire full
);
    reg [$clog2(16):0] count;
    reg [3:0] wptr, rptr;
    reg signed [7:0] acc;
    reg [W-1:0] mem [0:15];
    assign full = (count == 5'd16);
    assign dout = mem[rptr];
    always @(posedge clk) begin
        if (rst) begin
            count <= 5'd0; wptr <= 4'd0; rptr <= 4'd0; acc <= 8'sd0;
        end else begin
            if (push && !full) begin
                mem[wptr] <= din;
                count <= count + 5'd1; wptr <= wptr + 4'd1;
                acc <= acc + 8'sd1;
            end else if (pop && count != 5'd0) begin
                count <= count - 5'd1; rptr <= rptr + 4'd1;
            end
        end
    end
    always @(posedge clk) if (!rst) assert (count <= 5'd16);
endmodule
"""


def fifo_row(invariants=("count[3:0] == wptr - rptr", "count <= 5'd16"),
             verilog=FIFO, id="c0_900"):
    return {"id": id, "generation": 0, "top_module": "fifo_ctl",
            "clock": "clk", "antecedents": [], "sanity_covers": [],
            "verilog": verilog, "property": ["count <= 5'd16"],
            "invariants": list(invariants), "metrics": {}}


# --- 1. the port step ---------------------------------------------------

def test_internal_registers_the_facts_need_get_an_output_port():
    row = fifo_row()
    assert extend.compose_hidden_signals(row) == {"count", "wptr", "rptr"}
    out = extend.expose_for_wrapper(row)
    assert extend.compose_hidden_signals(out) == set()
    assert out["exposed"] == {"count": "count_o", "rptr": "rptr_o",
                              "wptr": "wptr_o"}
    ports = extend.ports_of(out["verilog"], "fifo_ctl")
    assert {"count_o", "wptr_o", "rptr_o", "full", "clk"} <= ports
    assert "W" not in extend.compose_hidden_signals(
        fifo_row(invariants=("full == (count == 5'd16)",)))
    assert "assign count_o = count;" in out["verilog"]
    # the design's own property and facts are untouched, and so is the row
    assert out["invariants"] == row["invariants"]
    assert out["property"] == row["property"]
    assert row["verilog"] == FIFO and "exposed" not in row


def test_the_port_keeps_the_width_and_the_sign():
    out = extend.expose_for_wrapper(fifo_row(invariants=("acc >= -8'sd1",
                                                        "count <= 5'd16")))
    assert "output wire signed [7:0] acc_o" in out["verilog"]
    assert "output wire [$clog2(16):0] count_o" in out["verilog"]


def test_a_design_that_needs_nothing_comes_back_as_it_is():
    row = fifo_row(invariants=("full == 1'b0 || !push",))
    assert extend.expose_for_wrapper(row) is row


def test_a_memory_in_the_facts_cannot_be_exposed():
    row = fifo_row(invariants=("mem[0] == 8'd0", "count <= 5'd16"))
    assert extend.expose_for_wrapper(row) is None
    rng = random.Random(1)
    draws = [batch.sample_task(rng, row, [row, fifo_row(id="c0_901")])
             for _ in range(300)]
    assert not any(t["ext_type"] in ("compose", "replicate") for t in draws)


def test_a_name_already_taken_gets_another_port_name():
    taken = FIFO.replace("output wire full", "output wire full,\n"
                         "    output wire [3:0] count_o")
    taken = taken.replace("assign full", "assign count_o = 4'd0;\n"
                          "    assign full")
    out = extend.expose_for_wrapper(fifo_row(verilog=taken))
    assert out["exposed"]["count"] == "count_obs"
    assert "assign count_obs = count;" in out["verilog"]


@needs_yosys
def test_an_exposed_design_reads_and_counts_the_same():
    """The mirrored registers already drive outputs here, so the live
    state is the same. (A register nothing reads would become live once
    mirrored: yosys counts it then, and the behaviour is still the same.)"""
    out = extend.expose_for_wrapper(fifo_row())
    assert state_bits(out["verilog"], "fifo_ctl") == \
        state_bits(FIFO, "fifo_ctl")


def test_the_wrapper_questions_explain_the_mirror_ports():
    for template in (extend.COMPOSE_TEMPLATE, extend.REPLICATE_TEMPLATE):
        assert "assign count_o = count;" in template


# --- 2. the size count --------------------------------------------------

LEAF = """module leaf (input wire clk, input wire rst, input wire en,
             output reg [7:0] q);
  always @(posedge clk) if (rst) q <= 8'd0; else if (en) q <= q + 8'd1;
endmodule
"""
POOL = LEAF + """module top (input wire clk, input wire rst, input wire en,
            output wire [7:0] a, b, c, output reg [3:0] pool);
  leaf u0 (.clk(clk), .rst(rst), .en(en), .q(a));
  leaf u1 (.clk(clk), .rst(rst), .en(en), .q(b));
  leaf u2 (.clk(clk), .rst(rst), .en(en), .q(c));
  always @(posedge clk) if (rst) pool <= 4'd0; else pool <= pool + 4'd1;
endmodule
"""


@needs_yosys
def test_flip_flops_in_sub_modules_count_once_per_copy():
    assert state_bits(LEAF, "leaf") == 8
    assert state_bits(POOL, "top") == 3 * 8 + 4


# --- 3. the branch stop -------------------------------------------------

def checked(main_s, pdr_s):
    return {"result": {"without_invariants": {"runs": [
        {"mode": "prove", "engine": "smtbmc yices", "duration_s": main_s},
        {"mode": "prove", "engine": "abc pdr", "duration_s": pdr_s,
         "notes": ["pdr_second_opinion: inconclusive (rc=8)"]}]}}}


def test_the_branch_stop_reads_the_main_check_only():
    """c1 child of c0_066 on 15 Oct: main check 0.9 s, second opinion
    121 s; the old stop read 121 against 60 and ended the lineage."""
    assert without_main_s(checked(0.94, 121.2)) == 0.94
    assert batch.without_wall(checked(0.94, 121.2)) == 0.94
    assert without_main_s({"result": {}}) is None


def test_the_branch_stop_is_the_checkers_own_limit():
    assert batch.MAX_WITHOUT_S == distractor.GRADE_KWARGS["timeout_s"] == 300
    row = fifo_row()
    assert batch.extendable(row, 299.0)
    assert not batch.extendable(row, 301.0)


def test_a_new_run_reads_the_stop_off_the_row():
    row = fifo_row()
    row["metrics"] = {"without_main_s": 400.0}
    assert not batch.extendable(row, None)
    row["metrics"] = {"without_main_s": 3.0}
    assert batch.extendable(row, None)


def test_promotion_keeps_the_main_check_time_on_the_row():
    parent = fifo_row(id="c0_000")
    rec = {"extension_id": "e1", "ext_type": "structural", "move": "STAGE",
           "parent_id": "c0_000", "verdict": "NECESSARY",
           "child_verilog": FIFO + "\n// grew\n",
           "invariants": parent["invariants"] + ["acc == acc"],
           **checked(2.5, 120.0)}
    _, rows = promote.promote([parent], [rec], compute_metrics=False)
    assert rows[0]["metrics"]["without_main_s"] == 2.5


# --- 4. combining, copying and depth ------------------------------------

def test_combining_and_copying_are_drawn_far_more_often():
    assert batch.MULTIPLIER_RATE >= 0.4
    rng = random.Random(5)
    pool = [fifo_row(id="c0_901"), fifo_row(id="c0_902")]
    pool[1] = dict(pool[1], top_module="fifo_b",
                   verilog=FIFO.replace("fifo_ctl", "fifo_b"))
    draws = [batch.sample_task(rng, pool[0], pool) for _ in range(2000)]
    share = sum(t["ext_type"] in ("compose", "replicate")
                for t in draws) / len(draws)
    assert 0.33 < share < 0.47, share


def test_lineages_may_go_twelve_rounds():
    assert batch.MAX_GEN == 12
    row = fifo_row()
    row["generation"] = 11
    assert batch.extendable(row, None)
    row["generation"] = 12
    assert not batch.extendable(row, None)


# --- 5. bigger moves ----------------------------------------------------

def lines_row(n):
    body = ["module m (input wire clk, output reg [3:0] a);"] + \
        ["    // line"] * (n - 2) + ["endmodule"]
    return {"id": "c0_1", "top_module": "m", "verilog": "\n".join(body),
            "property": ["a <= 4'd9"], "invariants": ["a <= 4'd9"]}


def test_the_big_moves_exist_and_are_drawn_most():
    for move in ("SUBUNIT", "STORAGE"):
        assert move in extend.MOVES
    w = extend.MOVE_WEIGHTS
    big = w["SUBUNIT"] + w["STORAGE"] + w["PEER"]
    assert big * 2 >= sum(w.values())
    assert w["LEDGER"] >= 2                 # Rishab, 6 Sep: keep it on


def test_a_big_move_asks_for_half_to_three_quarters_more_lines():
    for move in extend.BIG_MOVES:
        text = extend.build_prompt(lines_row(80), "structural", move)
        assert "80 lines" in text and "40 to 60 new lines" in text, move
    small = extend.build_prompt(lines_row(80), "structural", "SPLIT")
    assert "new lines" not in small


def test_the_irrelevant_logic_step_asks_for_a_real_block():
    assert "small piece" not in distractor.DISTRACTOR_PROMPT
    text = distractor.build_prompt(lines_row(80))
    assert "80 lines" in text and "40 to 60 new lines" in text


def test_pipeline_stages_come_in_bigger_runs():
    rng = random.Random(2)
    ks = {t["k"] for t in (batch.sample_task(rng, lines_row(40),
                                             [lines_row(40)])
                           for _ in range(3000))
          if t.get("move") == "STAGE"}
    assert ks == {1, 4, 8}


# --- 6. the new-kind-of-fact rule lets big growth through ----------------

PARENT = {"id": "c0_000", "top_module": "m", "verilog": "\n" * 39 + "x",
          "invariants": ["cnt <= 4'd9"]}


def grown(lines, **kw):
    r = {"extension_id": "e9", "ext_type": "structural", "move": "SUBUNIT",
         "parent_id": "c0_000", "verdict": "NECESSARY",
         "child_verilog": "\n" * (lines - 1) + "x",
         "invariants": ["cnt <= 4'd7"]}         # the same kind of fact
    r.update(kw)
    return r


def test_a_child_that_grew_by_half_is_kept_without_a_new_kind_of_fact():
    assert promote.GROWTH_EXEMPT == 0.5
    r = promote.ratchet(grown(60), [PARENT])
    assert r["verdict"] == "NECESSARY"
    assert r["grew_by"] == 0.5


def test_a_child_that_grew_less_still_needs_a_new_kind_of_fact():
    r = promote.ratchet(grown(50), [PARENT])
    assert r["verdict"] == "NO_NEW_SHAPE"


def test_growth_in_flip_flops_counts_too():
    r = promote.ratchet(grown(41, parent_state_bits=100,
                              child_state_bits=160), [PARENT])
    assert r["verdict"] == "NECESSARY" and r["grew_by"] == 0.6


# --- chasing lineages from named designs --------------------------------

def accept_all(task, corpus):
    parent = next(r for r in corpus if r["id"] == task["parent_id"])
    fresh = " + ".join(f"v{i}" for i in range(task["task_id"] + 2))
    return {"verdict": "NECESSARY", "extension_id": f"e{task['task_id']}",
            "ext_type": "structural", "move": "SUBUNIT",
            "parent_id": task["parent_id"],
            "child_verilog": parent["verilog"] + f"\n// {task['task_id']}",
            "invariants": list(parent["invariants"]) + [f"{fresh} == t"],
            "result": {}}


def test_a_chase_run_extends_the_newest_child_next():
    seen = []
    rows = [lines_row(40), dict(lines_row(40), id="c0_2")]
    out = batch.run_loop(rows, lambda t, c: (seen.append(t["parent_id"]),
                                             accept_all(t, c))[1],
                         max_calls=4, rng=random.Random(0), order="chase",
                         start_ids=["c0_1"])
    assert seen == ["c0_1", "c1_000", "c2_000", "c3_000"]
    assert [r["generation"] for r in out] == [1, 2, 3, 4]


def test_a_run_can_start_from_named_designs_only():
    seen = []
    rows = [lines_row(40), dict(lines_row(40), id="c0_2")]
    batch.run_loop(rows, lambda t, c: (seen.append(t["parent_id"]),
                                       {"verdict": "PATCH_ERROR"})[1],
                   max_calls=6, rng=random.Random(0), start_ids=["c0_2"])
    assert set(seen) == {"c0_2"}
    with pytest.raises(ValueError, match="c0_9"):
        batch.run_loop(rows, accept_all, max_calls=1, start_ids=["c0_9"])


def test_the_command_line_offers_the_chase_and_the_start():
    import inspect
    src = inspect.getsource(batch.main)
    assert '"chase"' in src and "--start" in src
