"""A second lane with an input of its own (26 Oct 2026, Rishab approved,
change 5).

SYSTEM_PROMPT rule 3 allowed no new inputs, so every PEER step of the 21 and
23 Oct trials fed its new lane by splitting an existing input (by a data
bit, an ID range or a command type), and "shares a resource" was met with a
counter over two separate queues; nothing in code checked the rule. Now:
  - rule 3 names its one exception: a PEER step's new lane gets an input
    stream of its own;
  - the PEER move asks for that input, ONE real thing the two lanes share
    through an arbiter, and a losing lane that waits;
  - a PEER question reads PEER_EFFECT_RULE: the new lane affects the old
    logic only through what the lanes share;
  - extend.new_input_error sends back any other step that adds an input
    port to the top module (NEW_INPUT), before any tool runs;
  - a PEER step declares the stream through the shared thing with both
    lanes' inputs (nothing_lost.declaration_error).
No model is called and no proof runs: the checker and the nothing-lost
search are faked where a step is graded."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
import nothing_lost                                             # noqa: E402
from build_corpus import extract_asserts                        # noqa: E402
from oracle.types import NecessityVerdict, TripleResult         # noqa: E402


def squash(text):
    return " ".join(text.split())


# a register that takes a word from the one lane the design has
LANE = """module lane (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       a_valid,
    input  wire [7:0] a_data,
    output reg  [7:0] q
);
    initial q = 8'd0;
    always @(posedge clk) begin
        if (rst) q <= 8'd0;
        else if (en && a_valid) q <= a_data;
    end
    always @(posedge clk) assert (q <= 8'd255);
endmodule
"""
NEW_PORT = ("@@ input  wire [7:0] a_data, @@\n"
            "+    input  wire       b_valid,\n")
# a second lane with its own valid, data and ready: it shares the register
# q with the old lane, which wins every conflict, and it waits while it loses
SECOND_LANE = ("@@ input  wire [7:0] a_data, @@\n"
               "+    input  wire       b_valid,\n"
               "+    input  wire [7:0] b_data,\n"
               "+    output wire       b_ready,\n"
               "@@ initial q = 8'd0; @@\n"
               "+    wire grant_b = b_valid && !a_valid;\n"
               "+    assign b_ready = en && !a_valid;\n"
               "@@ else if (en && a_valid) q <= a_data; @@\n"
               "+        else if (en && grant_b) q <= b_data;\n")
NEW_LANE = {"offer": "en && b_valid", "ready": "b_ready"}
OLD_LANE = {"offer": "en && a_valid", "ready": ""}


def into_q(*inputs):
    return [{"name": "words into q", "inputs": list(inputs),
             "delivered": "en && (a_valid || grant_b)", "held": "0",
             "flush": ""}]


def parent():
    return {"id": "c0_lane", "top_module": "lane", "clock": "clk",
            "verilog": LANE, "property": extract_asserts(LANE),
            "invariants": [], "antecedents": [], "sanity_covers": []}


@pytest.fixture
def checker(monkeypatch):
    """The checker faked (NECESSARY), sizes taken from the text, and the
    nothing-lost search faked; each is counted."""
    seen = {"graded": 0, "searched": 0}

    def triple(payload, **kw):
        seen["graded"] += 1
        return TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")

    def search(*a, **k):
        seen["searched"] += 1
        return False
    for module in (extend, distractor):
        monkeypatch.setattr(module, "state_bits", lambda v, top: len(v))
        monkeypatch.setattr(module, "grade_triple_generated", triple)
    monkeypatch.setattr(nothing_lost, "check", search)
    return seen


def step(patch, move="STORAGE", ext_type="structural", **out):
    answer = dict({"patch": patch, "invariants": [], "dispositions": []},
                  **out)
    return extend.grade_step4(parent(), ext_type, answer,
                              {"extension_id": "t1", "parent_id": "c0_lane",
                               "ext_type": ext_type, "move": move})


# --- the questions -----------------------------------------------------------

def test_rule_3_names_its_one_exception():
    text = squash(extend.SYSTEM_PROMPT)
    assert "add no inputs" in text
    assert "The one exception is a PEER step" in text
    assert "a ready or busy output that tells a sender to wait" in text


def test_the_second_lane_has_its_own_input_shares_one_thing_and_waits():
    text = squash(extend.MOVES["PEER"])
    for asked in ("input stream of its OWN",
                  "Never feed the new lane by splitting",
                  "an arbiter written as real logic",
                  "never dropped or overwritten, counted or not",
                  "Two separate queues joined only by a counter"):
        assert asked in text, asked
    assert "shares a resource with the existing logic" not in text


def test_the_peer_question_uses_the_peer_effect_rule():
    row = parent()
    peer = squash(extend.build_prompt(row, "structural", "PEER"))
    assert "through the one thing the lanes share" in peer
    assert squash(extend.PEER_EFFECT_RULE) in peer
    assert squash(extend.SIDEWAYS_EFFECT_RULE) not in peer
    storage = squash(extend.build_prompt(row, "structural", "STORAGE"))
    assert squash(extend.SIDEWAYS_EFFECT_RULE) in storage
    assert "through the one thing the lanes share" not in storage


# --- only a second lane may add an input (NEW_INPUT) -------------------------

def test_only_a_second_lane_may_add_an_input(checker):
    rec = step(NEW_PORT, move="STORAGE")
    assert rec["verdict"] == "NEW_INPUT"
    assert rec["error"] == (
        "the step adds the input port(s) b_valid to module lane. Only a PEER "
        "step may add inputs, for its new lane's own input stream. Keep the "
        "module's inputs as they are; to make a sender wait, add a ready or "
        "busy OUTPUT instead.")
    assert checker["graded"] == 0 and checker["searched"] == 0
    assert "child_state_bits" not in rec          # before yosys
    # the same patch as a second lane passes this check
    rec = step(NEW_PORT, move="PEER")
    assert rec["verdict"] == "NECESSARY" and rec["new_inputs"] == ["b_valid"]
    # the inputs of a new sub-module are its own, not the design's
    helper = ("+module seen_once (input wire clk, input wire go, "
              "output reg seen);\n"
              "+    initial seen = 1'b0;\n"
              "+    always @(posedge clk) if (go) seen <= 1'b1;\n"
              "+endmodule\n"
              "@@ initial q = 8'd0; @@\n"
              "+    wire seen_w;\n"
              "+    seen_once u_seen (.clk(clk), .go(a_valid), "
              ".seen(seen_w));\n")
    rec = step(helper, move="SUBUNIT")
    assert rec["verdict"] == "NECESSARY" and rec["new_inputs"] == []
    # rule 3 covers a second property too
    second = NEW_PORT + ("@@ always @(posedge clk) assert (q <= 8'd255); @@\n"
                         "+    always @(posedge clk) assert (!b_valid || "
                         "q <= 8'd255);\n")
    rec = step(second, move=None, ext_type="second")
    assert rec["verdict"] == "NEW_INPUT"
    assert checker["graded"] == 2                 # the two that passed


# --- a second lane declares both lanes ---------------------------------------

def test_a_second_lane_declares_both_lanes_into_the_shared_thing(checker):
    from patch import apply_patch
    child = apply_patch(LANE, SECOND_LANE)

    def declared(streams):
        return nothing_lost.declaration_error(streams, "PEER", child, "lane",
                                              "clk", parent=parent())
    rule = ("A PEER step must declare the stream through the thing its two "
            "lanes share, with at least two inputs: the new lane's, whose "
            "offer reads its own new input (b_data, b_valid), and the "
            "existing lane's, whose offer reads the inputs it had before.")
    assert declared(into_q(NEW_LANE)) == rule               # one input
    # two inputs that both read the old lane's inputs: an existing input
    # split by a data bit
    split = {"offer": "en && a_valid && a_data[0]", "ready": ""}
    assert declared(into_q(OLD_LANE, split)) == rule
    assert declared(into_q(NEW_LANE, OLD_LANE)) is None
    # in the grader the rule runs before any tool ...
    rec = step(SECOND_LANE, move="PEER", streams=into_q(NEW_LANE))
    assert rec["verdict"] == "STREAM_INVALID" and rec["error"] == rule
    assert checker["searched"] == 0 and checker["graded"] == 0
    # ... and an honest declaration goes on to the search and the proof
    rec = step(SECOND_LANE, move="PEER", streams=into_q(NEW_LANE, OLD_LANE))
    assert rec["verdict"] == "NECESSARY"
    assert rec["new_inputs"] == ["b_data", "b_valid"]
    assert checker["searched"] == 1 and checker["graded"] == 1
