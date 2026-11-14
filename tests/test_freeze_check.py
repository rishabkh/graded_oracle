"""The checker's freeze check (8 Nov 2026, Rishab approved).

On 27 Oct 2026, 6 of the 7 storage steps the checker kept were labeled
wrong, 3 of them the same way: a new part drained only on another
output's ready, so a receiver that says ready only after it sees valid
(allowed, and fine for the parent) froze the design for good. The proof
and the nothing-lost search both passed them: nothing is lost, the items
just never move again. The check simulates the design with random inputs,
every sender holding its item until it is taken and every receiver saying
ready one cycle after it sees valid, and turns a design away when, after
items have moved, nothing moves again under every seed, or when nothing
ever moves until the receivers are always ready (8 Nov 2026).

The pairing and the testbench are tested without tools; the simulations
need iverilog (activate hwtools first) and take about a second each."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import freeze_check as fc                                        # noqa: E402

requires_iverilog = pytest.mark.skipif(
    shutil.which("iverilog") is None or shutil.which("yosys") is None,
    reason="needs iverilog and yosys - activate hwtools first")

# a result with a trace store beside it: each item taken adds 4 trace
# words, and the store empties one word a cycle while the RESULT's receiver
# says ready; with 5 or more words held the design takes nothing. A
# receiver that says ready only when a result waits empties one word per
# result, so after two results the design stops for good.
TRACER = """
module tracer (
    input  wire       clk,
    input  wire       in_valid,
    output wire       in_ready,
    input  wire [7:0] in_data,
    output reg        out_valid = 1'b0,
    input  wire       out_ready,
    output reg  [7:0] out_data = 8'd0,
    output wire       trace_valid,
    {trace_ready_port}
    output wire [3:0] trace_words
);
    reg [3:0] held = 4'd0;
    assign trace_valid = held != 4'd0;
    assign trace_words = held;
    wire take  = in_valid && in_ready;
    wire give  = out_valid && out_ready;
    wire drain = {drain} && held != 4'd0;
    assign in_ready = !out_valid && held <= 4'd4;
    always @(posedge clk) begin
        if (take) begin
            out_valid <= 1'b1;
            out_data <= in_data;
        end else if (give)
            out_valid <= 1'b0;
        held <= held + (take ? 4'd4 : 4'd0) - (drain ? 4'd1 : 4'd0);
    end
    always @(posedge clk) assert (held <= 4'd8);
endmodule
"""
FROZEN = TRACER.format(trace_ready_port="", drain="out_ready")
MOVING = TRACER.format(trace_ready_port="input  wire       trace_ready,",
                       drain="trace_ready")


def test_each_ready_is_paired_with_its_own_valid():
    ports = {"clk": ("input", 1), "in_valid": ("input", 1),
             "in_ready": ("output", 1), "in_data": ("input", 8),
             "out_valid": ("output", 1), "out_ready": ("input", 1),
             "trace_valid": ("output", 1), "s_axis_tvalid": ("input", 1),
             "s_axis_tready": ("output", 1), "rdy": ("input", 1)}
    assert fc.channels(ports) == [
        {"name": "in", "valid": "in_valid", "ready": "in_ready", "side": "in"},
        {"name": "out", "valid": "out_valid", "ready": "out_ready", "side": "out"},
        {"name": "s_axis_t", "valid": "s_axis_tvalid", "ready": "s_axis_tready",
         "side": "in"}]


def test_i_and_o_in_front_can_say_which_way_a_port_points():
    """8 Nov 2026 trial: 7 of the 40 designs named their ports by the way
    they point (i_req_valid with o_req_ready), and none of their channels
    was paired, so none was checked. Where i_ and o_ name the channel
    (i_valid with i_ready, in the same trial), they pair as before."""
    ports = {"clk": ("input", 1), "i_ce": ("input", 1),
             "i_req_valid": ("input", 1), "o_req_ready": ("output", 1),
             "i_req_data": ("input", 8),
             "o_word_valid": ("output", 1), "i_word_ready": ("input", 1)}
    assert fc.channels(ports) == [
        {"name": "i_req", "valid": "i_req_valid", "ready": "o_req_ready",
         "side": "in"},
        {"name": "o_word", "valid": "o_word_valid", "ready": "i_word_ready",
         "side": "out"}]
    by_way = {"i_valid": ("input", 1), "o_ready": ("output", 1),
              "o_valid": ("output", 1), "i_ready": ("input", 1)}
    assert fc.channels(by_way) == [
        {"name": "i", "valid": "i_valid", "ready": "o_ready", "side": "in"},
        {"name": "o", "valid": "o_valid", "ready": "i_ready", "side": "out"}]
    by_channel = {"i_valid": ("input", 1), "i_ready": ("output", 1),
                  "o_valid": ("output", 1), "o_ready": ("input", 1)}
    assert fc.channels(by_channel) == [
        {"name": "i", "valid": "i_valid", "ready": "i_ready", "side": "in"},
        {"name": "o", "valid": "o_valid", "ready": "o_ready", "side": "out"}]
    # names that differ are not guessed at (design 0: o_valid, i_out_ready)
    assert fc.channels({"o_valid": ("output", 1),
                        "i_out_ready": ("input", 1)}) == []


def test_i_and_o_at_the_end_can_say_which_way_a_port_points():
    """As Ibex names them: req_valid_i with req_ready_o."""
    ports = {"clk_i": ("input", 1), "req_valid_i": ("input", 1),
             "req_ready_o": ("output", 1), "req_data_i": ("input", 8),
             "rsp_valid_o": ("output", 1), "rsp_ready_i": ("input", 1)}
    assert fc.channels(ports) == [
        {"name": "req", "valid": "req_valid_i", "ready": "req_ready_o",
         "side": "in"},
        {"name": "rsp", "valid": "rsp_valid_o", "ready": "rsp_ready_i",
         "side": "out"}]


def test_a_reset_is_held_at_the_start_whichever_way_it_points():
    assert fc.reset_level("rst") == 1
    assert fc.reset_level("rst_n") == 0
    assert fc.reset_level("aresetn") == 0
    assert fc.reset_level("reset") == 1
    assert fc.reset_level("restart") is None
    assert fc.reset_level("burst_n") is None


def test_a_reset_with_i_in_front_is_held_too():
    """Of our designs, 102 name the reset i_rst and 73 i_reset; the
    testbench took them for settings and reset the design at random."""
    assert fc.reset_level("i_rst") == 1
    assert fc.reset_level("i_reset") == 1
    assert fc.reset_level("i_rst_n") == 0
    assert fc.reset_level("i_first") is None


def test_the_testbench_receivers_say_ready_only_after_they_see_valid():
    ports = {"clk": ("input", 1), "out_valid": ("output", 1),
             "out_ready": ("input", 1), "out_data": ("output", 8),
             "mode": ("input", 2)}
    tb = fc.testbench("m", ports, "clk", seed=3, cycles=100)
    assert "out_ready <= out_valid;" in tb
    # every other input is random, from the testbench's own seed, never a port
    assert "mode <= {$random(oracle_fz_seed)} % 16;" in tb


def test_every_word_wide_input_takes_an_ordinary_small_value():
    """8 Nov 2026 calibration: random 32-bit values made good designs wait
    billions of cycles, which looked like a freeze: a latency setting
    (cfg_lat), and fields carried with an item, a timer's new period and a
    rotator's shift amount. Every input wider than a bit gets 0 to 15; a
    freeze does not hang on the values."""
    ports = {"clk": ("input", 1), "in_valid": ("input", 1),
             "in_ready": ("output", 1), "in_amt": ("input", 32),
             "cfg_lat": ("input", 32), "load_cfg": ("input", 1),
             "out_valid": ("output", 1), "out_ready": ("input", 1)}
    tb = fc.testbench("m", ports, "clk", seed=3, cycles=100)
    assert "cfg_lat <= {$random(oracle_fz_seed)} % 16;" in tb
    assert "in_amt <= {$random(oracle_fz_seed)} % 16;" in tb
    assert "load_cfg <= $random(oracle_fz_seed);" in tb


def test_the_testbench_senders_hold_their_item_until_it_is_taken():
    ports = {"clk": ("input", 1), "in_valid": ("input", 1),
             "in_ready": ("output", 1), "in_data": ("input", 8)}
    tb = fc.testbench("m", ports, "clk", seed=3, cycles=100)
    # a new item only after the last one was taken or when none was offered
    assert "if (!in_valid || oracle_fz_took_in) begin" in tb
    assert "in_data <= {$random(oracle_fz_seed)} % 16;" in tb


def test_the_simulation_copy_drops_what_only_a_proof_tool_reads():
    """27 Oct 2026 designs keep $fell(busy) in the guard of an assertion,
    which iverilog cannot read; with the assertion gone the guard does
    nothing, so it becomes 0 (as do $past, $rose, $stable, $changed), and
    a cover or an assume becomes an empty statement."""
    src = ("always @(posedge clk) begin\n"
           "    if (pv && $fell(busy)) assert (q == $past(q, 2));\n"
           "    if ($rose(go) || $stable(x[1:0])) cover (done);\n"
           "    assume (!(a && b));\n"
           "end\n")
    out = fc.simulation_copy(src)
    assert "$" not in out and "assert" not in out
    assert "cover" not in out and "assume" not in out
    assert "if (pv && 1'b0) ;" in out


def test_a_design_with_no_handshake_is_not_checked():
    out = fc.check("module m (input wire clk, output reg q = 1'b0);\n"
                   "always @(posedge clk) q <= !q;\nendmodule\n", "m", "clk")
    assert out["result"] == "not_checked"


@requires_iverilog
def test_a_store_that_drains_on_another_outputs_ready_freezes(tmp_path):
    out = fc.check(FROZEN, "tracer", "clk", work=tmp_path)
    assert out["result"] == "frozen", out
    assert out["moved"] >= 1 and out["still"] >= fc.STILL


@requires_iverilog
def test_the_same_store_with_its_own_ready_keeps_moving(tmp_path):
    out = fc.check(MOVING, "tracer", "clk", work=tmp_path)
    assert out["result"] == "moves", out


@requires_iverilog
def test_the_lesson_says_where_it_stopped(tmp_path):
    out = fc.check(FROZEN, "tracer", "clk", work=tmp_path)
    text = fc.lesson(out)
    assert "says ready only after it sees valid" in text
    assert f"cycle {out['last_move']}" in text


@requires_iverilog
def test_a_sub_module_sized_by_a_body_parameter_still_runs(tmp_path):
    """27 Oct 2026, t001: iverilog cannot size a port from a localparam of
    the module body, which yosys reads; the simulation runs on the copy
    yosys writes out, flattened, with every size worked out."""
    design = MOVING + """
module wrap (
    input  wire       clk,
    input  wire       in_valid,
    output wire       in_ready,
    input  wire [7:0] in_data,
    output wire       out_valid,
    input  wire       out_ready,
    output wire [W-1:0] out_data,
    output wire       trace_valid,
    input  wire       trace_ready
);
    localparam W = 8;
    wire [3:0] words;
    tracer t (.clk(clk), .in_valid(in_valid), .in_ready(in_ready),
              .in_data(in_data), .out_valid(out_valid), .out_ready(out_ready),
              .out_data(out_data), .trace_valid(trace_valid),
              .trace_ready(trace_ready), .trace_words(words));
endmodule
"""
    out = fc.check(design, "wrap", "clk", work=tmp_path)
    assert out["result"] == "moves", out


@requires_iverilog
def test_a_design_the_simulator_cannot_read_changes_nothing(tmp_path):
    out = fc.check("module m (input wire clk, input wire a_valid,\n"
                   "output wire a_ready); assign a_ready = ;\nendmodule\n",
                   "m", "clk", work=tmp_path)
    assert out["result"] == "error"


def _renamed(verilog, names):
    return re.sub(r"\b(" + "|".join(names) + r")\b",
                  lambda m: names[m.group(1)], verilog)


# the tracer's ports named by the way they point, in front or at the end
IN_FRONT = {"in_valid": "i_valid", "in_ready": "o_ready", "in_data": "i_data",
            "out_valid": "o_valid", "out_ready": "i_ready",
            "out_data": "o_data", "trace_valid": "o_trace_valid",
            "trace_ready": "i_trace_ready", "trace_words": "o_trace_words"}
AT_THE_END = {"in_valid": "in_valid_i", "in_ready": "in_ready_o",
              "in_data": "in_data_i", "out_valid": "out_valid_o",
              "out_ready": "out_ready_i", "out_data": "out_data_o",
              "trace_valid": "trace_valid_o", "trace_ready": "trace_ready_i",
              "trace_words": "trace_words_o"}


@requires_iverilog
@pytest.mark.parametrize("names", [IN_FRONT, AT_THE_END],
                         ids=["in front", "at the end"])
def test_a_design_named_by_the_way_its_ports_point_is_checked(names,
                                                              tmp_path):
    frozen = fc.check(_renamed(FROZEN, names), "tracer", "clk",
                      work=tmp_path / "frozen")
    moving = fc.check(_renamed(MOVING, names), "tracer", "clk",
                      work=tmp_path / "moving")
    assert (frozen["result"], moving["result"]) == ("frozen", "moves"), \
        (frozen, moving)


# design 1 of the 8 Nov 2026 trial, cut down: it takes an item only while
# its clock enable is high
CE_QUEUE = """
module ce_queue (
    input  wire       clk,
    input  wire       i_ce,
    input  wire       i_valid,
    output wire       i_ready,
    input  wire [7:0] i_data,
    output reg  [7:0] o_sum = 8'd0
);
    assign i_ready = i_ce;
    always @(posedge clk)
        if (i_valid && i_ready) o_sum <= o_sum + i_data;
endmodule
"""


# works 1,200 cycles of its clock enable on each item, about 2,400 cycles:
# one item moves in the 2,000-cycle run, the next only after it ends
SLOW_WORKER = """
module slow_worker (
    input  wire       clk,
    input  wire       i_ce,
    input  wire       i_req_valid,
    output wire       o_req_ready,
    input  wire [7:0] i_req_data,
    output reg  [7:0] o_sum = 8'd0
);
    reg [10:0] left = 11'd0;
    assign o_req_ready = i_ce && left == 11'd0;
    always @(posedge clk)
        if (i_req_valid && o_req_ready) begin
            left <= 11'd1200;
            o_sum <= o_sum + i_req_data;
        end else if (i_ce && left != 11'd0)
            left <= left - 11'd1;
endmodule
"""


@requires_iverilog
def test_no_item_moves_before_the_count_starts(tmp_path):
    """8 Nov 2026: the count starts after the 3 cycles a reset is held, but
    the senders offered items from the first cycle: under seed 2 the slow
    design took its only item of the run at cycle 2, uncounted, and looked
    as if it never moved. Senders now wait for the count to start."""
    out = fc.check(SLOW_WORKER, "slow_worker", "clk", work=tmp_path)
    assert [r["first"] >= 0 for r in out["runs"]] == [True] * len(fc.SEEDS)


@requires_iverilog
def test_a_design_slower_than_the_run_is_not_frozen(tmp_path):
    """8 Nov 2026 trial, design 33: a 768-bit ladder works one bit per
    cycle of its clock enable, about 1,550 cycles an item, so after an item
    nothing moved for the last 440 cycles of the run and it looked frozen.
    A design that looks frozen runs again 10 times as long, and is frozen
    only when nothing moves in the last 4,000 cycles of that run too."""
    out = fc.check(SLOW_WORKER, "slow_worker", "clk", work=tmp_path)
    assert out["still"] >= fc.STILL             # it looked frozen at first
    assert out["result"] == "moves", out
    assert out.get("long_run_still", fc.STILL * 10) < fc.STILL * 10


@requires_iverilog
def test_a_clock_enable_is_not_held_with_an_item(tmp_path):
    """8 Nov 2026 trial, design 1: with i_valid/i_ready every input starts
    with i_, so the testbench held the clock enable i_ce with each item as
    if it were the item's data; an item offered with i_ce low was never
    taken, and a good design was turned away as frozen. A channel named
    only i or o, like a plain valid/ready, holds none of the inputs."""
    out = fc.check(CE_QUEUE, "ce_queue", "clk", work=tmp_path)
    assert out["result"] == "moves", out
    assert out["moved"] > 100 and out["still"] < fc.STILL


# designs 12 and 28 of the 8 Nov 2026 trial, cut down: an item is taken
# only while the output's receiver already says ready, and the output's
# valid comes only after an item was taken; a receiver that waits to see
# valid first never says ready, so nothing ever moves
ASK_FIRST = """
module ask_first (
    input  wire       clk,
    input  wire       i_valid,
    output wire       i_ready,
    input  wire [7:0] i_word,
    output wire       o_valid,
    input  wire       o_ready,
    output reg  [7:0] o_word = 8'd0
);
    reg busy = 1'b0;
    assign i_ready = {take};
    assign o_valid = busy;
    always @(posedge clk)
        if (i_valid && i_ready) begin
            busy <= 1'b1;
            o_word <= i_word;
        end else if (o_valid && o_ready)
            busy <= 1'b0;
endmodule
"""
WAITS = ASK_FIRST.format(take="!busy && o_ready")


@requires_iverilog
def test_a_design_that_moves_only_with_receivers_always_ready_is_turned_away(
        tmp_path):
    """8 Nov 2026 trial: designs 12 and 28 never moved, which passed. With
    receivers that are always ready they move: they wait for a ready
    before anything moves, and a receiver may wait to see valid first."""
    out = fc.check(WAITS, "ask_first", "clk", work=tmp_path)
    assert out["result"] == "waits_for_ready", out
    assert out["moved"] == 0 and out.get("always_ready_moved", 0) > 0


@requires_iverilog
def test_the_same_design_taking_items_while_it_has_room_moves(tmp_path):
    out = fc.check(ASK_FIRST.format(take="!busy || o_ready"), "ask_first",
                   "clk", work=tmp_path)
    assert out["result"] == "moves", out


# hands over its first item after 5,000 cycles, long after the run ends
SLOW_START = """
module slow_start (
    input  wire        clk,
    output wire        o_valid,
    input  wire        o_ready,
    output reg  [12:0] count = 13'd0
);
    assign o_valid = count == 13'd5000;
    always @(posedge clk)
        if (!o_valid) count <= count + 13'd1;
endmodule
"""


@requires_iverilog
def test_a_design_that_moves_with_neither_receiver_is_not_turned_away(
        tmp_path):
    out = fc.check(SLOW_START, "slow_start", "clk", work=tmp_path)
    assert out["result"] == "never_moved", out
    assert out.get("always_ready_moved") == 0        # it was run again


# --- in the pipeline: off unless a run switches it on -------------------------

@pytest.fixture
def switched_on(monkeypatch):
    monkeypatch.setattr(fc, "ON", True)


def test_the_gate_does_nothing_unless_the_run_switches_it_on():
    record = {}
    assert fc.ON is False
    assert fc.gate(record, FROZEN, "tracer", "clk") is False
    assert record == {}


@requires_iverilog
def test_the_gate_sends_a_frozen_step_back_with_its_lesson(switched_on):
    record = {}
    assert fc.gate(record, FROZEN, "tracer", "clk") is True
    assert record["verdict"] == "FREEZES"
    assert "says ready only after it sees valid" in record["error"]
    assert record["reason"].startswith("froze: ")
    assert record["freeze_check"]["result"] == "frozen"


@requires_iverilog
def test_the_gate_keeps_what_it_saw_on_a_design_that_moves(switched_on):
    record = {}
    assert fc.gate(record, MOVING, "tracer", "clk") is False
    assert record == {"freeze_check": record["freeze_check"]}
    assert record["freeze_check"]["result"] == "moves"


@requires_iverilog
def test_the_gate_sends_a_design_that_waits_for_ready_back(switched_on):
    record = {}
    assert fc.gate(record, WAITS, "ask_first", "clk") is True
    assert record["verdict"] == "FREEZES"
    assert "says ready only after it sees valid" in record["error"]
    assert "always ready" in record["error"]
    assert record["reason"].startswith("waits for ready: ")
    assert record["freeze_check"]["result"] == "waits_for_ready"
    assert record["freeze_check"]["always_ready_moved"] > 0


def test_the_extender_checks_every_big_step_and_combination_after_nothing_lost():
    import inspect
    import extend
    for f in (extend.grade_step4, extend.grade_compose):
        src = inspect.getsource(f)
        assert "freeze_check.gate(" in src, f.__name__
        assert src.index("nothing_lost.check(") < src.index("freeze_check.gate(")


def test_a_batch_run_can_switch_it_on_and_remembers_a_freeze():
    import inspect
    import batch
    assert "FREEZES" in batch.REMEMBERED
    src = inspect.getsource(batch.main)
    assert '"--freeze-check"' in src
    assert "freeze_check.ON = args.freeze_check" in src


def test_the_initiators_judge_checks_for_a_freeze_after_its_gates():
    import inspect
    sys.path.insert(0, str(ROOT / "initiator"))
    import run
    src = inspect.getsource(run.judge)
    assert src.index('"FACTS_NOT_PLAIN"') < src.index("freeze_check.gate(") \
        < src.index("grade_triple_generated(")
    assert "FREEZES" in run.GATE_VERDICTS


@requires_iverilog
def test_a_frozen_answer_is_turned_away_before_the_proof(switched_on,
                                                          monkeypatch):
    import json
    sys.path.insert(0, str(ROOT / "initiator"))
    import run

    def no_proof(*a, **kw):
        raise AssertionError("a frozen design never reaches the proof")
    monkeypatch.setattr(run, "grade_triple_generated", no_proof)
    triple = {"verilog": FROZEN, "top_module": "tracer", "clock": "clk",
              "property": ["held <= 4'd8"], "invariants": ["held <= 4'd8"],
              "antecedents": [], "sanity_covers": []}
    j = run.judge(json.dumps(triple), "ok", "globally", spin=False)
    assert j["verdict"] == "FREEZES"
    assert "says ready only after it sees valid" in j["reason"]


def test_the_pilot_can_switch_it_on():
    import inspect
    sys.path.insert(0, str(ROOT / "initiator"))
    import run
    src = inspect.getsource(run.main)
    assert 'pilot.add_argument("--freeze-check"' in src
    assert "freeze_check.ON = args.freeze_check" in src


# --- the initiator's log keeps what the check saw (8 Nov 2026) ----------------
# The 2-design check run of 8 Nov logged nothing about this check: the
# initiator kept what it saw only for a frozen design, so its log could not
# tell "nothing froze" from "the check never ran". The simulator's answers
# are given here; the gate's own tests above run the real one.

SENDS = [{"valid": "out_valid", "ready": "out_ready", "side": "out"}]
SAW_FROZEN = {"result": "frozen", "moved": 2, "still": 1991, "last_move": 9,
              "channels": SENDS}
SAW_MOVES = {"result": "moves", "moved": 640, "still": 4, "last_move": 1996,
             "channels": SENDS}
SAW_ERROR = {"result": "error", "why": "iverilog or yosys is not on PATH"}


def _simulator_says(monkeypatch, *answers):
    left = list(answers)
    monkeypatch.setattr(fc, "check",
                        lambda verilog, top, clock="clk": left.pop(0))


def _initiator(monkeypatch, tmp_path, verdicts):
    """The initiator with its model and proof faked (the proof's verdicts in
    turn), as in test_initiator_pipeline."""
    sys.path.insert(0, str(ROOT / "initiator"))
    from tests.test_initiator_pipeline import Fake
    return Fake(monkeypatch, tmp_path, verdicts)


def test_the_initiator_logs_what_the_check_saw_on_every_round(
        switched_on, monkeypatch, tmp_path):
    _simulator_says(monkeypatch, SAW_FROZEN, SAW_MOVES)
    f = _initiator(monkeypatch, tmp_path, ["NECESSARY"])
    [row], _ = f.run(repairs=1)
    assert [h["verdict"] for h in row["history"]] == ["FREEZES", "NECESSARY"]
    assert [h["freeze_check"]["result"] for h in row["history"]] == [
        "frozen", "moves"]
    assert row["freeze_check"] == SAW_MOVES
    assert "Verdict: FREEZES" in f.prompts[1][1]
    assert "says ready only after it sees valid" in f.prompts[1][1]


def test_a_run_without_repairs_logs_it_too(switched_on, monkeypatch,
                                           tmp_path):
    _simulator_says(monkeypatch, SAW_FROZEN, SAW_MOVES)
    f = _initiator(monkeypatch, tmp_path, ["NECESSARY"])
    f.R.run_attempts(2, cmd="pilot", log_path=f.log, distill_path=f.distill)
    rows = [json.loads(l) for l in f.log.read_text().splitlines()]
    assert [r["verdict"] for r in rows] == ["FREEZES", "NECESSARY"]
    assert [r["freeze_check"]["result"] for r in rows] == ["frozen", "moves"]


def test_a_design_the_check_could_not_run_on_passes_and_the_log_says_so(
        switched_on, monkeypatch, tmp_path):
    _simulator_says(monkeypatch, SAW_ERROR)
    f = _initiator(monkeypatch, tmp_path, ["NECESSARY"])
    [row], _ = f.run(repairs=1)
    assert row["verdict"] == "NECESSARY" and row["repair_attempts"] == 0
    assert row["freeze_check"] == SAW_ERROR
    assert row["history"][0]["freeze_check"] == SAW_ERROR


def test_with_the_check_off_the_initiators_log_is_as_before(monkeypatch,
                                                            tmp_path):
    assert fc.ON is False
    f = _initiator(monkeypatch, tmp_path, ["NOT_PROVEN", "NECESSARY"])
    [row], _ = f.run(repairs=1)
    assert "freeze_check" not in row
    assert all("freeze_check" not in h for h in row["history"])
