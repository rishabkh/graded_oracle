"""The freeze check also tries a sender that goes quiet (16 Nov 2026,
Rishab: "alright let's do it", change 4 of 7).

Four starting designs of the 15 Nov pilot (p0_107, p0_122, p0_160,
p0_184) passed the freeze check and were then labeled wrong for one
pattern: a pipeline that moves only when a new input arrives or the
receiver is already ready (`advance = in_ready && (in_valid ||
out_ready)`). A lone item is taken and never reaches the output of a
receiver that waits to see valid first. The check missed it because its
senders never stop: the next input pushes the item along.

So a design that moves is run twice more for each short burst of 1, 2
and 4 items, after which every sender stays quiet: once with receivers
that say ready only after they see valid, once with receivers that are
always ready, on the same inputs. It is "stuck_when_quiet", and turned
away, when under every seed the always-ready receivers got more items
out than the waiting ones, both runs having long gone still. Comparing
the two keeps a design that rightly holds part of an item (two words
make one packet) from being blamed, and a slow one too.
"""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import freeze_check as fc                                        # noqa: E402

requires_iverilog = pytest.mark.skipif(
    shutil.which("iverilog") is None or shutil.which("yosys") is None,
    reason="needs iverilog and yosys - activate hwtools first")

# three stages; an item is never lost: when the last stage is full the
# pipeline moves only while the receiver takes it
PIPE = """
module pipe (
    input  wire       clk,
    input  wire       in_valid,
    output wire       in_ready,
    input  wire [7:0] in_data,
    output wire       out_valid,
    input  wire       out_ready,
    output wire [7:0] out_data
);
    reg [2:0] full = 3'd0;
    reg [7:0] d0 = 8'd0, d1 = 8'd0, d2 = 8'd0;
    assign in_ready = !full[2] || out_ready;
    assign out_valid = full[2];
    assign out_data = d2;
    wire advance = {advance};
    always @(posedge clk)
        if (advance) begin
            full <= {{full[1:0], in_valid}};
            d2 <= d1;
            d1 <= d0;
            d0 <= in_data;
        end
endmodule
"""
PUSHED = PIPE.format(advance="in_ready && (in_valid || out_ready)")
FLOWS = PIPE.format(advance="in_ready")

# two words make one packet: a lone word rightly stays inside
PAIRS = """
module pairs (
    input  wire        clk,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [7:0]  in_data,
    output wire        out_valid,
    input  wire        out_ready,
    output reg  [15:0] out_data = 16'd0
);
    reg have = 1'b0, full = 1'b0;
    reg [7:0] first = 8'd0;
    assign in_ready = !full;
    assign out_valid = full;
    always @(posedge clk) begin
        if (out_valid && out_ready)
            full <= 1'b0;
        if (in_valid && in_ready) begin
            if (!have) begin
                first <= in_data;
                have <= 1'b1;
            end else begin
                out_data <= {first, in_data};
                full <= 1'b1;
                have <= 1'b0;
            end
        end
    end
endmodule
"""

# the answer comes 300 cycles after the item, whoever waits for it
SLOW = """
module slow (
    input  wire       clk,
    input  wire       in_valid,
    output wire       in_ready,
    input  wire [7:0] in_data,
    output wire       out_valid,
    input  wire       out_ready,
    output reg  [7:0] out_data = 8'd0
);
    reg busy = 1'b0, done = 1'b0;
    reg [8:0] left = 9'd0;
    assign in_ready = !busy && !done;
    assign out_valid = done;
    always @(posedge clk) begin
        if (in_valid && in_ready) begin
            busy <= 1'b1;
            left <= 9'd300;
            out_data <= in_data;
        end else if (busy && left != 9'd0)
            left <= left - 9'd1;
        else if (busy) begin
            busy <= 1'b0;
            done <= 1'b1;
        end
        if (out_valid && out_ready)
            done <= 1'b0;
    end
endmodule
"""

IN_ONLY = {"clk": ("input", 1), "in_valid": ("input", 1),
           "in_ready": ("output", 1), "in_data": ("input", 8)}
BOTH = dict(IN_ONLY, out_valid=("output", 1), out_ready=("input", 1),
            out_data=("output", 8))


# --- the testbench ----------------------------------------------------------------

def test_a_quiet_sender_stops_after_its_burst():
    tb = fc.testbench("m", IN_ONLY, "clk", seed=3, cycles=100, quiet_after=2)
    assert "integer oracle_fz_sent_in = 0;" in tb
    assert "oracle_fz_sent_in = oracle_fz_sent_in + 1;" in tb
    assert "if (oracle_fz_sent_in >= 2) in_valid <= 1'b0;" in tb


def test_a_sender_that_never_goes_quiet_reads_as_before():
    tb = fc.testbench("m", IN_ONLY, "clk", seed=3, cycles=100)
    assert "oracle_fz_sent_in >=" not in tb
    assert "if (!in_valid || oracle_fz_took_in) begin" in tb


def test_the_testbench_counts_the_items_that_come_out():
    tb = fc.testbench("m", BOTH, "clk", seed=3, cycles=100)
    assert "if (out_valid && out_ready) begin" in tb
    assert "oracle_fz_outs = oracle_fz_outs + 1;" in tb
    assert "outs=%0d last_out=%0d" in tb
    # items taken in are not items that came out
    assert "if (in_valid && in_ready) begin\n                oracle_fz_outs" \
        not in tb


# --- the check --------------------------------------------------------------------

@requires_iverilog
def test_a_lone_item_that_needs_the_next_input_to_push_it_is_turned_away(
        tmp_path):
    out = fc.check(PUSHED, "pipe", "clk", work=tmp_path)
    assert out["result"] == "stuck_when_quiet", out
    # with steady senders it moves, which is why it used to pass
    assert out["moved"] > 100
    assert out["quiet"] == {"burst": 1, "delivered": 0,
                            "always_ready_delivered": 1}


@requires_iverilog
def test_the_same_pipeline_moving_whenever_there_is_room_passes(tmp_path):
    out = fc.check(FLOWS, "pipe", "clk", work=tmp_path)
    assert out["result"] == "moves", out
    assert out["quiet"] == {"bursts": list(fc.QUIET_BURSTS), "stuck": False}


@requires_iverilog
def test_a_design_that_rightly_holds_half_an_item_passes(tmp_path):
    out = fc.check(PAIRS, "pairs", "clk", work=tmp_path)
    assert out["result"] == "moves", out


@requires_iverilog
def test_a_slow_design_passes(tmp_path):
    out = fc.check(SLOW, "slow", "clk", work=tmp_path)
    assert out["result"] == "moves", out


@requires_iverilog
def test_a_design_with_no_receiver_is_not_tried(tmp_path):
    design = """
module sink (
    input  wire       clk,
    input  wire       in_valid,
    output wire       in_ready,
    input  wire [7:0] in_data,
    output reg  [7:0] sum = 8'd0
);
    assign in_ready = 1'b1;
    always @(posedge clk) if (in_valid) sum <= sum + in_data;
endmodule
"""
    out = fc.check(design, "sink", "clk", work=tmp_path)
    assert out["result"] == "moves", out
    assert "quiet" not in out


# --- what the retry and the log are told ------------------------------------------

STUCK = {"result": "stuck_when_quiet", "moved": 640, "still": 3,
         "last_move": 1996,
         "channels": [{"name": "in", "valid": "in_valid", "ready": "in_ready",
                       "side": "in"},
                      {"name": "out", "valid": "out_valid",
                       "ready": "out_ready", "side": "out"}],
         "quiet": {"burst": 1, "delivered": 0, "always_ready_delivered": 1}}


def test_the_lesson_says_what_was_seen_and_what_to_do():
    text = fc.lesson(STUCK)
    assert "says ready only after it sees valid" in text
    assert "out_valid/out_ready" in text
    assert "1 item was sent and then nothing more" in text
    assert "0 came out" in text and "1 came out" in text
    assert "without waiting for another input or for the receiver's ready" \
        in text


def test_the_reason_is_one_line():
    line = fc.reason(STUCK)
    assert line.startswith("stuck when the sender goes quiet: ")
    assert "\n" not in line


def test_it_is_turned_away_and_what_was_seen_is_kept():
    assert "stuck_when_quiet" in fc.TURNED_AWAY
    assert "quiet" in fc.KEPT


@requires_iverilog
def test_the_gate_sends_it_back_with_its_lesson(monkeypatch):
    monkeypatch.setattr(fc, "ON", True)
    record = {}
    assert fc.gate(record, PUSHED, "pipe", "clk") is True
    assert record["verdict"] == "FREEZES"
    assert record["reason"].startswith("stuck when the sender goes quiet: ")
    assert record["freeze_check"]["result"] == "stuck_when_quiet"
    assert record["freeze_check"]["quiet"]["burst"] == 1
