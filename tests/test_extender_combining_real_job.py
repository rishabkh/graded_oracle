"""Combining two designs through a part that does a real job (26 Oct 2026,
Rishab approved, changes 2, 4 and 7).

The root-cause study of 25 Oct 2026 found that the forced register between
two combined designs only made copies of a value one cycle late (proven in
5 kept designs), while the rule it was for, that a combination must not be
a plain wire, never fired (DECORATIVE for 0 of 36 combinations). The advice
to hold the sender's enable low froze whole designs: t033 and c2_004 lost
the route-table writes sent during a pause. And 8 of 12 partner answers said
the receiver never refuses, at least twice falsely. So:
  - the combine question asks for a part that does a real job (a buffer
    that makes the sender wait, a conversion that keeps the meaning, or an
    arbiter), and a wrapper with no register is graded like any other;
  - gates.IDLE_GLUE accepts an input of one design computed from the
    other's outputs, and still turns away a plain wire;
  - a combination that drives a design's enable with logic while that
    design's other control inputs come straight from outside is sent back
    before any tool runs (PAUSES_WHOLE_DESIGN, nothing_lost.py);
  - the partner question and the plan decide refusal from what the
    receiver does, in the plan's own direction.
No model is called and no proof runs: the checker and the nothing-lost
search are faked where a step is graded."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
import gates                                                    # noqa: E402
import nothing_lost                                             # noqa: E402
from build_corpus import extract_asserts                        # noqa: E402
from oracle.types import NecessityVerdict, TripleResult         # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / \
    "nothing_lost_root_cause_2026-10-25"


def squash(text):
    return " ".join(text.split())


# --- two small designs and the wrappers that combine them --------------------

# a counter that offers each new count for one cycle (it cannot wait)
SRC = """module src4 (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg        o_valid,
    output reg  [3:0] o_count
);
    initial o_valid = 1'b0;
    initial o_count = 4'd0;
    always @(posedge clk) begin
        if (rst) begin o_valid <= 1'b0; o_count <= 4'd0; end
        else if (en) begin
            o_valid <= !o_valid;
            if (o_valid) o_count <= o_count + 4'd1;
        end
    end
    always @(posedge clk) assert (o_count <= 4'd15);
endmodule
"""
# an accumulator that takes a value in every cycle it is offered one
ACC = """module acc8 (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       i_valid,
    input  wire [7:0] i_data,
    output reg  [7:0] o_sum
);
    initial o_sum = 8'd0;
    always @(posedge clk) begin
        if (rst) o_sum <= 8'd0;
        else if (en && i_valid) o_sum <= o_sum + i_data;
    end
    always @(posedge clk) assert (o_sum <= 8'd255);
endmodule
"""
# a route table with its own enable (i_ce), written through i_wr_en, as in
# the study's t033 (flow_match_table)
TABLE = """module route_tbl (
    input  wire       clk,
    input  wire       rst,
    input  wire       i_ce,
    input  wire       i_wr_en,
    input  wire [1:0] i_wr_addr,
    input  wire [3:0] i_wr_port,
    input  wire       i_req,
    input  wire [1:0] i_req_addr,
    output reg        o_valid,
    output reg  [3:0] o_port
);
    reg [3:0] tbl [0:3];
    initial o_valid = 1'b0;
    initial o_port = 4'd0;
    always @(posedge clk) begin
        if (rst) begin o_valid <= 1'b0; o_port <= 4'd0; end
        else if (i_ce) begin
            if (i_wr_en) tbl[i_wr_addr] <= i_wr_port;
            o_valid <= i_req;
            if (i_req) o_port <= tbl[i_req_addr];
        end
    end
    always @(posedge clk) assert (!o_valid || o_port <= 4'd15);
endmodule
"""
# the count, zero-extended into the accumulator's data input: a conversion
# that keeps the meaning, and no register of its own
WIDEN = """module link (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output wire [7:0] sum
);
    wire       a_valid;
    wire [3:0] a_count;
    src4 u_a (.clk(clk), .rst(rst), .en(en), .o_valid(a_valid), .o_count(a_count));
    acc8 u_b (.clk(clk), .rst(rst), .en(en), .i_valid(a_valid), .i_data({4'd0, a_count}), .o_sum(sum));
    always @(posedge clk) if (a_valid) assert (sum <= 8'd255);
endmodule
"""
COUNTS = [{"name": "counts into the accumulator",
           "inputs": [{"offer": "a_valid", "ready": ""}],
           "delivered": "en && a_valid", "held": "0", "flush": ""}]
# the table's enable held low while its last answer is out, as t033 held
# u_fmt's i_ce; its writes still come straight from the wrapper's inputs
FROZEN = """module frozen (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       wr_en,
    input  wire [1:0] wr_addr,
    input  wire [3:0] wr_port,
    output wire [3:0] port
);
    wire       a_valid;
    wire [3:0] a_count;
    wire       b_valid;
    reg        busy = 1'b0;
    src4 u_a (.clk(clk), .rst(rst), .en(en), .o_valid(a_valid), .o_count(a_count));
    route_tbl u_b (.clk(clk), .rst(rst), .i_ce(TABLE_CE), .i_wr_en(TABLE_WE), .i_wr_addr(wr_addr), .i_wr_port(wr_port), .i_req(a_valid), .i_req_addr(a_count[1:0]), .o_valid(b_valid), .o_port(port));
    always @(posedge clk) begin
        if (rst) busy <= 1'b0;
        else if (en) busy <= b_valid && !busy;
    end
    always @(posedge clk) if (b_valid) assert (port <= 4'd15);
endmodule
"""
# the same table as the sending design: its answers go to the accumulator
SENDS = """module sends (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       wr_en,
    input  wire [1:0] wr_addr,
    input  wire [3:0] wr_port,
    input  wire       req,
    input  wire [1:0] req_addr,
    output wire [7:0] sum
);
    wire       t_valid;
    wire [3:0] t_port;
    reg        stall = 1'b0;
    route_tbl u_a (.clk(clk), .rst(rst), .i_ce(en && !stall), .i_wr_en(wr_en), .i_wr_addr(wr_addr), .i_wr_port(wr_port), .i_req(req), .i_req_addr(req_addr), .o_valid(t_valid), .o_port(t_port));
    acc8 u_b (.clk(clk), .rst(rst), .en(en), .i_valid(t_valid), .i_data({4'd0, t_port}), .o_sum(sum));
    always @(posedge clk) begin
        if (rst) stall <= 1'b0;
        else if (en) stall <= t_valid && !stall;
    end
    always @(posedge clk) if (t_valid) assert (sum <= 8'd255);
endmodule
"""


def frozen(ce="en && !busy", we="wr_en"):
    return FROZEN.replace("TABLE_CE", ce).replace("TABLE_WE", we)


def design(verilog, top):
    return {"id": f"c0_{top}", "top_module": top, "clock": "clk",
            "verilog": verilog, "property": extract_asserts(verilog),
            "invariants": [], "antecedents": [], "sanity_covers": []}


def top_of(verilog):
    return re.findall(r"\bmodule\s+(\w+)", verilog)[-1]


def rule(child):
    """The frozen-enable rule on the whole design, its wrapper last."""
    top = top_of(child)
    return nothing_lost.frozen_enable_error(nothing_lost.top_text(child, top),
                                            child, top, "clk")


@pytest.fixture
def checker(monkeypatch):
    """The checker faked (NECESSARY unless the test says otherwise), sizes
    taken from the text, and the nothing-lost search faked; each is
    counted."""
    seen = {"graded": 0, "searched": [], "verdict": NecessityVerdict.NECESSARY}

    def triple(payload, **kw):
        seen["graded"] += 1
        return TripleResult(verdict=seen["verdict"], reason="faked")

    def search(record, child, top, clock, parents, wrapper=None):
        seen["searched"].append({"top": top, "clock": clock,
                                 "parents": [p["id"] for p in parents],
                                 "wrapper": wrapper})
        return False
    for module in (extend, distractor):
        monkeypatch.setattr(module, "state_bits", lambda v, top: len(v))
        monkeypatch.setattr(module, "grade_triple_generated", triple)
    monkeypatch.setattr(nothing_lost, "check", search)
    return seen


def combine(a, b, wrapper, streams=COUNTS, **fields):
    out = dict({"wrapper": wrapper, "top_module": top_of(wrapper),
                "antecedents": [], "sanity_covers": [], "invariants": [],
                "streams": streams}, **fields)
    return extend.grade_compose(a, b, out, {"extension_id": "t1",
                                            "ext_type": "compose"})


# --- the combine question ----------------------------------------------------

def test_the_combine_question_asks_for_a_real_job_not_a_register():
    text = squash(extend.COMPOSE_TEMPLATE)
    for removed in ("THROUGH STATEFUL GLUE", "NEVER through a plain wire",
                    "Put a register stage, a valid/data handshake pair, or a "
                    "small buffer", "HOLD its contents", "refills itself",
                    "by holding A's enable low until B can take more",
                    "PLUS at least one clause about the glue state itself",
                    "The glue clause is the point",
                    "a state with garbage in the glue",
                    "must mean something to B"):
        assert removed not in text, removed
    for asked in ("a buffer that holds items and makes the sender wait",
                  "a conversion that keeps the meaning", "an arbiter",
                  "never add a register just to have one",
                  "A plain wire does no job"):
        assert asked in text, asked


# --- grading a combination with no register ----------------------------------

def test_a_combination_without_registers_is_graded(checker):
    a, b = design(SRC, "src4"), design(ACC, "acc8")
    rec = combine(a, b, WIDEN)
    assert rec["verdict"] == "NECESSARY", rec.get("error")
    assert rec["glue_identifiers"] == []
    assert "glue_clause" not in rec and checker["graded"] == 1
    # the stream it declared is searched once, before the proof, as a
    # combination: the wrapper and both parents
    assert rec["streams"] == COUNTS
    assert checker["searched"] == [{"top": "link", "clock": "clk",
                                    "parents": ["c0_src4", "c0_acc8"],
                                    "wrapper": rec["wrapper"]}]


def test_the_glue_fact_rule_is_gone():
    assert not hasattr(extend, "glue_clause")


def test_a_combination_that_needs_no_fact_is_told_so(checker):
    checker["verdict"] = NecessityVerdict.DECORATIVE
    rec = combine(design(SRC, "src4"), design(ACC, "acc8"), WIDEN)
    assert rec["verdict"] == "DECORATIVE"
    assert "proves without any facts" in rec["error"]
    assert "glue not stateful enough" not in rec["error"]


# --- the glue check after the proof (gates.IDLE_GLUE) ------------------------

def glued(wrapper, ext_type="compose", parts=SRC + "\n" + ACC):
    return {"verdict": "NECESSARY", "ext_type": ext_type, "wrapper": wrapper,
            "child_verilog": parts + "\n" + wrapper,
            "invariants": ["sum <= 8'd255"], "new_shapes": ["N <= N"]}


def link(valid, data, extra=""):
    """WIDEN with the accumulator's inputs given by `valid` and `data`."""
    inputs = ".i_valid(a_valid), .i_data({4'd0, a_count})"
    assert inputs in WIDEN
    return WIDEN.replace(inputs, f".i_valid({valid}), .i_data({data})") \
        .replace("    src4 u_a", extra + "    src4 u_a")


def test_a_plain_wire_between_two_designs_is_idle_glue():
    renamed = ("    wire       v = a_valid;\n"
               "    wire [3:0] d;\n"
               "    assign d = a_count;\n")
    for wrapper in (link("a_valid", "a_count"), link("v", "d", renamed)):
        rec = gates.apply(glued(wrapper), [], use_yosys=False)
        assert rec["verdict"] == "IDLE_GLUE"
        assert rec["reason"] == (
            "the wrapper's registers drive no input of the designs it "
            "wraps, and no input of one design is computed from the other's "
            "outputs: the parts sit side by side, or are joined by a plain "
            "wire")


def test_a_conversion_or_an_arbiter_between_two_designs_passes_the_glue_check():
    widened = "    wire [7:0] wide = {4'd0, a_count};\n"
    for wrapper in (WIDEN,                                   # an extension
                    link("a_valid", "wide", widened),        # through a wire
                    link("a_count == 4'd15", "8'd1"),        # a comparison
                    link("a_valid", "{4'd0, a_count[3:1]}")):  # a slice
        assert gates.parts_linked(wrapper, SRC + ACC + wrapper, "link")
        rec = gates.apply(glued(wrapper), [], use_yosys=False)
        assert rec["verdict"] == "NECESSARY", (wrapper, rec.get("reason"))
        assert rec["gates"]["failed"] is None
    # an arbiter that shares the accumulator's input between the counter
    # and an outside source
    shared = (link("a_valid || ext_valid",
                   "a_valid ? {4'd0, a_count} : ext_data")
              .replace("input  wire       en,",
                       "input  wire       en,\n    input  wire       "
                       "ext_valid,\n    input  wire [7:0] ext_data,"))
    rec = gates.apply(glued(shared), [], use_yosys=False)
    assert rec["verdict"] == "NECESSARY" and rec["gates"]["failed"] is None


def test_copies_keep_the_old_glue_rule():
    # two copies of the accumulator: the second sums what the first holds,
    # a computed link, but the pool register drives nothing
    copies = """module two (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire [7:0] d,
    output wire [7:0] s1
);
    wire [7:0] s0;
    reg  [3:0] pool = 4'd0;
    acc8 u0 (.clk(clk), .rst(rst), .en(en), .i_valid(1'b1), .i_data(d), .o_sum(s0));
    acc8 u1 (.clk(clk), .rst(rst), .en(en), .i_valid(1'b1), .i_data(s0 + 8'd1), .o_sum(s1));
    always @(posedge clk) pool <= pool + 4'd1;
endmodule
"""
    assert gates.parts_linked(copies, ACC + copies, "two")
    rec = gates.apply(glued(copies, ext_type="replicate", parts=ACC), [],
                      use_yosys=False)
    assert rec["verdict"] == "IDLE_GLUE"
    assert rec["reason"] == ("the wrapper's registers drive no input of the "
                             "designs it wraps")


# --- a design paused whole by its enable (PAUSES_WHOLE_DESIGN) ---------------

def test_a_frozen_enable_with_other_inputs_from_outside_is_sent_back(checker):
    a, b = design(SRC, "src4"), design(TABLE, "route_tbl")
    rec = combine(a, b, frozen(), streams=[
        {"name": "counts into the table", "inputs": [
            {"offer": "a_valid", "ready": ""}],
         "delivered": "en && !busy && a_valid", "held": "0", "flush": ""}])
    assert rec["verdict"] == "PAUSES_WHOLE_DESIGN"
    assert rec["error"] == (
        "u_b (route_tbl)'s input i_ce looks like its enable, the input that "
        "leaves all of its state unchanged while it is low. The wrapper "
        "drives it with en && !busy, so while that is low route_tbl also "
        "ignores i_wr_en, which come straight from the wrapper's inputs: "
        "settings, table writes or requests sent then are lost. Connect i_ce "
        "straight to the wrapper's own enable input. Make the sender wait on "
        "the stream's own path instead (its ready or busy input, or the "
        "request, start or valid input that begins its next item), show the "
        "wait on a ready output of the wrapper, and feed route_tbl its items "
        "through its own valid, start or data inputs, never through i_ce.")
    assert rec["reason"] == rec["error"].split(".")[0]
    # sent back before any tool: no search, no proof
    assert checker["searched"] == [] and checker["graded"] == 0
    assert "child_state_bits" not in rec


def test_an_enable_wired_straight_or_a_part_fed_wholly_by_the_stream_passes():
    parts = SRC + "\n" + TABLE + "\n"
    assert rule(parts + frozen(ce="en")) is None
    assert rule(parts + frozen(ce="1'b1")) is None
    # a square-root unit whose enable and data both come from the sending
    # design: nothing of it comes straight from outside
    sqrt = """module sqrt8 (
    input  wire       clk,
    input  wire       rst,
    input  wire       ce,
    input  wire [7:0] din,
    output reg  [3:0] root
);
    initial root = 4'd0;
    always @(posedge clk) if (rst) root <= 4'd0; else if (ce) root <= din[7:4];
    always @(posedge clk) assert (root <= 4'd15);
endmodule
"""
    wrapper = """module roots (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output wire [3:0] root
);
    wire       a_valid;
    wire [3:0] a_count;
    src4 u_a (.clk(clk), .rst(rst), .en(en), .o_valid(a_valid), .o_count(a_count));
    sqrt8 u_b (.clk(clk), .rst(rst), .ce(en && a_valid), .din({a_count, 4'd0}), .root(root));
    always @(posedge clk) assert (root <= 4'd15);
endmodule
"""
    assert rule(SRC + sqrt + wrapper) is None
    # a table whose only other inputs straight from outside are buses: the
    # rule reads 1-bit control inputs only (it reads port names, a guard
    # for code, not a judgement of behaviour; the reviewer asks the rest)
    assert rule(parts + frozen(we="wr_en && !busy")) is None


def test_the_frozen_enable_rule_looks_at_both_designs_whichever_sends():
    # the paused table receives the counts ...
    for order in (SRC + TABLE, TABLE + SRC):
        assert rule(order + frozen()).startswith(
            "u_b (route_tbl)'s input i_ce looks like its enable")
    # ... or sends its answers to the accumulator
    for order in (TABLE + ACC, ACC + TABLE):
        err = rule(order + SENDS)
        assert err.startswith(
            "u_a (route_tbl)'s input i_ce looks like its enable, the input "
            "that leaves all of its state unchanged while it is low. The "
            "wrapper drives it with en && !stall, so while that is low "
            "route_tbl also ignores i_req, i_wr_en, which come straight from "
            "the wrapper's inputs")


def test_the_frozen_enable_rule_on_the_studys_combinations():
    """The study's t033 and c1_007 froze u_fmt's i_ce while its requests
    and writes came straight from outside; the skeptic's rebuild of
    sys_c1_003 froze the whole shuffler while its settings input cfg_we
    came from outside (spec Decision 3); c1_001 wires both enables
    straight."""
    def fires(name):
        return rule((FIXTURES / name).read_text())
    for name in ("t033_child.sv", "c1_007_grown.sv"):
        assert fires(name).startswith(
            "u_fmt (flow_match_table)'s input i_ce looks like its enable"), \
            name
    rebuild = fires("sys_c1_003_skeptic_rebuild.sv")
    assert rebuild.startswith("u_shuf (benes_lane_shuffler)'s input en looks "
                              "like its enable")
    assert "cfg_we" in rebuild
    assert fires("c1_001_grown.sv") is None


# --- refusal judged from what the receiver does (change 7) -------------------

def test_the_partner_question_decides_refusal_from_behaviour():
    text = squash(extend.PARTNER_TEMPLATE)
    for asked in ("not from its signal names",
                  "takes a request only when it is idle",
                  "A design with no full, busy or ready signal can still "
                  "refuse",
                  "Either may send to the other",
                  "Holding a design's enable low does not count as waiting",
                  "in which states the receiver ignores or loses what "
                  "arrives"):
        assert asked in text, asked
    assert "(it has a full, busy or ready signal)" not in text
    assert "(or why the receiver never refuses)" not in text


def test_the_plan_is_checked_against_the_code_in_its_own_direction():
    plan = {"system": "a counter feeding an accumulator",
            "connection": "acc8 never refuses"}
    text = squash(extend.compose_prompt(design(SRC, "src4"),
                                        design(ACC, "acc8"), plan))
    assert "Planned connection: acc8 never refuses" in text
    assert "check it against the code above" in text
    assert "in the plan's direction" in text
    assert "whatever the plan says" in text
    # without a plan there is nothing to check
    text = squash(extend.compose_prompt(design(SRC, "src4"),
                                        design(ACC, "acc8")))
    assert "check it against the code above" not in text
