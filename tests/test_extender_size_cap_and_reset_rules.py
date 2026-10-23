"""Two fixes to what the extender's questions ask (22 Oct 2026).

F7  The cap on a step's size (200 to 300 new lines, so the answer fits in
    one reply) started only above 300 lines, so a sub-unit or storage step
    on a design of 241 to 300 lines was still asked for up to 375 new
    lines. The cap now holds whenever the usual ask goes over 300, and the
    words of every question that states the size agree with the numbers.
F12 The rules asked for "a reset in the existing if (rst) branch" and
    "always @(posedge clk) or always @(*) logic", but since 21 Oct one new
    design in five resets at once (always @(posedge clk or negedge rst_n),
    if (!rst_n)). Every question now asks for the module's own reset.
No calls, no proofs."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
from build_corpus import extract_asserts, honest_lines          # noqa: E402


def squash(text):
    return " ".join(text.split())


def lines_row(n, top="m"):
    """A design of n - 1 counted lines, no two alike."""
    body = [f"module {top} (input wire clk, output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(n - 2)] + \
        ["endmodule"]
    return {"id": "c0_1", "generation": 0, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": "\n".join(body), "property": ["a <= 4'd9"],
            "invariants": ["a <= 4'd9"], "metrics": {}}


# --- F7: the cap holds whenever the usual ask passes 300 ---------------------

@pytest.mark.parametrize("lines", [241, 250, 280, 300])
@pytest.mark.parametrize("move", ["SUBUNIT", "STORAGE"])
def test_a_doubling_step_is_capped_once_its_ask_passes_300(lines, move):
    v = lines_row(lines + 1)["verilog"]
    assert honest_lines(v) == lines
    assert extend.size_target(v, move) == (lines, 200, 300)


@pytest.mark.parametrize("move", ["SUBUNIT", "STORAGE"])
def test_a_doubling_step_that_fits_still_doubles(move):
    v = lines_row(241)["verilog"]                     # 240 lines
    assert extend.size_target(v, move) == (240, 240, 300)


@pytest.mark.parametrize("move", ["PEER", None])
def test_other_steps_up_to_300_lines_are_unchanged(move):
    v = lines_row(301)["verilog"]                     # 300 lines
    assert extend.size_target(v, move) == (300, 150, 225)


def test_the_question_states_the_capped_size():
    text = squash(extend.build_prompt(lines_row(251), "structural",
                                      "STORAGE"))
    assert "The module now has 250 lines" in text
    assert "200 to 300 new lines" in text
    assert "at most 300 in one step, so that the whole answer fits in one " \
        "reply" in text
    assert "doubles" not in text
    # over 300 lines the words are as before
    big = squash(extend.build_prompt(lines_row(401), "structural", "PEER"))
    assert "at most 300 for a design over 300 lines" in big


def test_the_words_always_agree_with_the_numbers():
    for move in ("SUBUNIT", "STORAGE", "PEER", "STAGE", None):
        for n in list(range(1, 420, 7)) + [240, 241, 300, 301]:
            capped = extend.size_capped(move, n)
            share = extend.size_share(move, n)
            assert ("at most 300" in share) == capped, (move, n)
            lo, hi = (200, 300) if capped else extend._usual_ask(move, n)
            assert hi <= extend.BIG_STEP_CAP, (move, n)
            assert lo <= hi, (move, n)


def test_the_debug_block_question_never_passes_the_cap():
    for n in (100, 241, 300, 301, 500):
        text = squash(distractor.build_prompt(lines_row(n + 1)))
        _, lo, hi = extend.size_target(lines_row(n + 1)["verilog"])
        assert f"add roughly {lo} to {hi} new lines" in text
        assert hi <= 300


# --- F12: the module's own reset ---------------------------------------------

ASYNC = """module tally (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       en,
    output reg  [3:0] cnt
);
    initial cnt = 4'd0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt <= 4'd0;
        else if (en && cnt != 4'd9)
            cnt <= cnt + 4'd1;
    end
    always @(posedge clk) assert (cnt <= 4'd9);
endmodule
"""
ASYNC_ROW = {"id": "c0_9", "top_module": "tally", "clock": "clk",
             "antecedents": [], "sanity_covers": [], "verilog": ASYNC,
             "property": extract_asserts(ASYNC), "invariants": []}

# what only a design that resets in an if (rst) branch can do
SYNCHRONOUS_ONLY = ("if (rst) branch", "always @(posedge clk) or always @(*)",
                    "synthesizable always @(posedge clk) logic",
                    "always @(posedge clk) or always")


def questions_for(row):
    """Every question the extender can ask about `row`, with its rules."""
    out = {"rules": extend.SYSTEM_PROMPT,
           "second": extend.build_prompt(row, "second"),
           "replicate": extend.build_prompt(row, "replicate", n=4),
           "compose": extend.compose_prompt(row, row),
           "debug block": distractor.build_prompt(row)}
    for move in extend.MOVES:
        out[move] = extend.build_prompt(row, "structural", move, k=4)
    return out


def test_no_question_about_an_asynchronous_reset_design_asks_for_another():
    for name, text in questions_for(ASYNC_ROW).items():
        flat = squash(text)
        for phrase in SYNCHRONOUS_ONLY:
            assert phrase not in flat, (name, phrase)


def test_the_rules_ask_for_the_modules_own_reset():
    rules = squash(extend.SYSTEM_PROMPT)
    assert "Match the module's own reset: every new register resets the " \
        "same way, in the same kind of block, as the module's existing " \
        "registers" in rules
    assert "written like the module's own (the same clock edge and the " \
        "same reset)" in rules
    copies = squash(extend.REPLICATE_TEMPLATE)
    assert "the parent's own kind of reset" in copies
    assert "in the same kind of block, as the parent's registers" in copies
    pair = squash(extend.COMPOSE_TEMPLATE)
    assert "resets the way the parents' registers do, in the same kind of " \
        "block" in pair
    debug = squash(distractor.DISTRACTOR_PROMPT)
    assert "Match the module's own reset" in debug
    assert "in the same kind of block, as the module's existing registers" \
        in debug


def test_the_questions_built_for_the_design_carry_the_rule():
    """The templates are filled with str.format, so building them also
    shows the new words broke none of them."""
    questions = questions_for(ASYNC_ROW)
    assert "Match the module's own reset" in squash(questions["debug block"])
    assert "the parent's own kind of reset" in squash(questions["replicate"])
    assert "resets the way the parents' registers do" in squash(
        questions["compose"])
