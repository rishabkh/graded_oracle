"""Combinations whose parts fit at the wires, and one rule about new ports
(21 Oct 2026, Rishab: "make all 1-8 changes", items 6 and 8c).

In the 21 Oct trial 2 of 10 combinations were kept. The reviewer turned
down a font bitmap row (8 one-bit pixels) fed in as a 32-bit brightness
sample, and a source with no way to wait feeding a queue that can refuse.
Two more failed UNUSED_COPIES at 69%, one point under the 70% line: both
padded a 64-bit word with zeros into a 128-bit input, so 1,536 of the
receiver's flip-flops held zero for ever (fed from a real source instead,
both measure 101%). So:
  - the partner question and the combine question ask that the parts fit
    at the wires: widths that match, or a conversion that keeps the
    meaning, and a sender that waits instead of losing data;
  - the 70% line stays: the check saw real waste, and its reason now says
    what was found (flip-flops that never change);
  - no question says any more that new state is seen only through
    existing outputs: new OUTPUT ports have been allowed since 18 Oct.
No model is called; the yosys tests build designs in temporary folders."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import extend                                                   # noqa: E402
import gates                                                    # noqa: E402

needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None,
                                 reason="yosys not on PATH (run hwtools)")


def squash(text):
    return " ".join(text.split())


def row(id, top, inputs, outputs):
    ports = ", ".join([f"input wire {p}" for p in inputs]
                      + [f"output wire {p}" for p in outputs])
    return {"id": id, "top_module": top, "construct": f"{top}: a part",
            "verilog": f"module {top} ({ports});\nendmodule\n",
            "property": ["1'b1"], "invariants": ["1'b1"]}


# --- 6a. the partner question ------------------------------------------------

def test_the_partner_question_asks_for_widths_that_fit():
    text = squash(extend.PARTNER_TEMPLATE)
    assert "fit at the wires" in text
    assert "an output drives an input of the same width" in text
    assert "converted in a way that keeps its meaning, and you say how" in text
    # a number may be widened; a padded data word or packed pixels may not
    assert "zero- or sign-extended into a wider input" in text
    assert "it is still the same number" in text
    assert ("Eight 1-bit font pixels packed in a byte are not a 32-bit "
            "brightness sample") in text
    assert "padded with zeros to fill a wider data input does not fit" in text


def test_the_partner_question_asks_for_a_sender_that_can_wait():
    text = squash(extend.PARTNER_TEMPLATE)
    assert "can refuse or stall" in text
    assert ("a ready or busy input of its own or an enable that holds "
            "it") in text
    assert "no data is ever dropped" in text
    assert ("A sender that cannot be made to wait cannot feed a receiver "
            "that can refuse") in text


def test_the_partner_answer_names_widths_and_waiting():
    text = squash(extend.PARTNER_TEMPLATE)
    assert "the width on each side and how a difference is converted" in text
    assert "how the sender waits when the receiver cannot take more" in text
    assert '"none"' in text                 # still a good answer


def test_the_partner_question_as_sent_holds_the_rules_and_the_ports():
    a = row("c0_1", "ring_stop", ["clk", "en"], ["ej_vld", "ej_data[63:0]"])
    b = row("c0_2", "spi_seq", ["clk", "push", "push_data[127:0]"],
            ["full"])
    text = squash(extend.partner_prompt(a, [b], {"c0_1": a, "c0_2": b}))
    assert "push_data[127:0]" in text and "ej_data[63:0]" in text
    assert "fit at the wires" in text
    assert "no data is ever dropped" in text


# --- 6a. the combine question ------------------------------------------------

def test_the_combine_question_asks_for_widths_that_fit():
    text = squash(extend.COMPOSE_TEMPLATE)
    assert "The two must also fit at the wires" in text
    assert "connect signals of the same width" in text
    assert "keeps its meaning and say how in glue_scheme" in text
    assert "zero- or sign-extended into a wider input" in text
    assert "Do not pad a data word with zeros to fill a wider data input" \
        in text
    assert "never changes and does no work" in text


def test_the_combine_question_asks_the_glue_to_hold_and_make_a_wait():
    text = squash(extend.COMPOSE_TEMPLATE)
    assert "if B can refuse or stall" in text
    assert "the glue holds the data and tells A to wait" in text
    assert "by holding A's enable low until B can take more" in text
    assert "Never drop data" in text
    # the trial's glue counted what it dropped (gb_drop, q_ovf)
    assert "counting dropped data does not make dropping it acceptable" in text


def test_the_combine_question_keeps_its_earlier_rules():
    text = squash(extend.COMPOSE_TEMPLATE)
    assert "must mean something to B" in text
    assert "parents' widths" in text
    assert "NEVER through a plain wire" in text
    assert "narrow" not in text.lower()


def test_the_combine_question_as_sent_holds_the_rules_and_the_plan():
    a = row("c0_1", "fifo_a", ["clk"], ["rdata[63:0]"])
    b = row("c0_2", "seq_b", ["clk", "push_data[127:0]"], ["full"])
    plan = {"system": "a FIFO in front of a sequencer",
            "connection": "rdata[63:0] into push_data[63:0]"}
    text = squash(extend.compose_prompt(a, b, plan))
    assert "Planned connection: rdata[63:0] into push_data[63:0]" in text
    assert "fit at the wires" in text
    assert "tells A to wait" in text


# --- 6b. the 70% line, and what its reason says ------------------------------

def test_the_line_for_unused_copies_stays_at_70_percent():
    """The two 69% combinations of the 21 Oct trial were not good designs
    wrongly turned away: both padded half of a queue with zeros, and one
    lost data whenever its buffer was full."""
    assert gates.MIN_STATE_USED == 0.7


PARTS = """module src8 (
    input  wire       clk,
    input  wire       en,
    output reg  [7:0] data
);
    initial data = 8'd0;
    always @(posedge clk) if (en) data <= data + 8'd3;
endmodule

module q16 (
    input  wire        clk,
    input  wire        push,
    input  wire [15:0] din,
    output wire [15:0] dout
);
    reg [15:0] mem [0:3];
    reg [1:0]  wp = 2'd0;
    reg [1:0]  rp = 2'd0;
    integer i;
    initial for (i = 0; i < 4; i = i + 1) mem[i] = 16'd0;
    always @(posedge clk) begin
        if (push) begin
            mem[wp] <= din;
            wp <= wp + 2'd1;
        end
        rp <= rp + 2'd1;
    end
    assign dout = mem[rp];
endmodule
"""


def pair(upper):
    """An 8-bit source feeding a 16-bit queue through a holding register;
    `upper` fills the top half of each entry."""
    return f"""module pair (
    input  wire        clk,
    input  wire        en,
    input  wire [7:0]  hi,
    output wire [15:0] q
);
    wire [7:0] a;
    reg  [7:0] hold = 8'd0;
    src8 u_a (.clk(clk), .en(en), .data(a));
    always @(posedge clk) if (en) hold <= a;
    q16 u_b (.clk(clk), .push(en), .din({{{upper}, hold}}), .dout(q));
endmodule
"""


def combined(upper):
    wrapper = pair(upper)
    return {"verdict": "NECESSARY", "ext_type": "compose",
            "wrapper": wrapper, "child_verilog": PARTS + "\n" + wrapper,
            "invariants": ["hold <= 8'd255"], "new_shapes": ["id <= N"]}


@needs_yosys
def test_a_word_padded_with_zeros_leaves_part_of_the_receiver_idle():
    """The 21 Oct trial's case in small: half of every queue entry is
    always zero, so its flip-flops do no work."""
    used = gates.copied_state_used(PARTS + pair("8'd0"), "pair",
                                   pair("8'd0"))
    assert used < gates.MIN_STATE_USED
    rec = gates.apply(combined("8'd0"), [])
    assert rec["verdict"] == "UNUSED_COPIES"
    assert "never change" in rec["reason"]
    assert "padded into a wider input" in rec["reason"]


@needs_yosys
def test_the_same_pair_fed_real_data_uses_all_of_it():
    used = gates.copied_state_used(PARTS + pair("hi"), "pair", pair("hi"))
    assert used >= 0.99
    rec = gates.apply(combined("hi"), [])
    assert rec["verdict"] == "NECESSARY"
    assert rec["gates"]["failed"] is None


# --- 8c. new output ports, said the same way everywhere ----------------------

def every_question_text():
    return ([extend.SYSTEM_PROMPT, extend.STRUCTURAL_TEMPLATE,
             extend.SECOND_TEMPLATE, extend.COMPOSE_TEMPLATE,
             extend.REPLICATE_TEMPLATE, extend.PARTNER_TEMPLATE,
             extend.EFFECT_RULE, extend.LEDGER_EFFECT_RULE,
             extend.SIDEWAYS_EFFECT_RULE, extend.INVARIANT_RULES]
            + list(extend.MOVES.values()))


def test_the_rule_about_ports_allows_new_outputs_and_no_new_inputs():
    text = squash(extend.SYSTEM_PROMPT)
    assert "add no inputs" in text
    assert "You may add OUTPUT ports for new state" in text


def test_no_question_still_says_the_ports_are_fixed():
    for text in every_question_text():
        t = squash(text).lower()
        assert "port list is fixed" not in t
        assert "through existing outputs" not in t


def test_the_second_property_question_allows_a_new_output_port():
    text = squash(extend.SECOND_TEMPLATE)
    assert "observable through an output (an existing one or a new OUTPUT " \
        "port)" in text
    p = {"verilog": "module m;\nendmodule", "property": ["a"],
         "invariants": ["a"], "top_module": "m"}
    assert "a new OUTPUT port" in squash(extend.build_prompt(p, "second"))


def test_ledger_asking_for_output_ports_agrees_with_the_rule():
    """LEDGER asks every new register to be an output port; rule 3 allows
    new OUTPUT ports, so the two agree (they did not while the port list
    was fixed)."""
    ledger = squash(extend.MOVES["LEDGER"])
    assert "must be an output port" in ledger
    assert "OUTPUT ports" in extend.SYSTEM_PROMPT
