"""Never drop work, counted or not, and the checker's own nothing-is-lost
search (26 Oct 2026, Rishab approved, changes 1, 2 and 3).

The root-cause study of 25 Oct 2026 found 8 of 14 combined and second-lane
designs losing data, 5 of them kept because the loss was counted or
flagged. Counting what the sender hands over caught all 8, with no false
alarm on the 5 good designs; counting only the items the design chose to
take caught none. So a big step or a combination now declares the streams
of items it carries, the checker reads the declaration by rules
(nothing_lost.declaration_error), and then counts every item handed over in
a copy of the design and searches 48 to 80 cycles from reset for one that is
neither delivered nor still held (nothing_lost.check), before the proof.

The questions and their schemas, the rules, the copy, the reading of sby's
logs and where the search sits in the grading are tested without tools.
The searches run sby and skip when it is missing (activate hwtools first);
each takes about a second. The study's own designs are in
fixtures/nothing_lost_root_cause_2026-10-25 (SOURCES.txt says what each
shows)."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
import nothing_lost                                             # noqa: E402
import promote                                                  # noqa: E402
import sense_check                                              # noqa: E402
from build_corpus import extract_asserts                        # noqa: E402
from oracle.sby import SbyOutcome, sby_available                # noqa: E402
from oracle.types import NecessityVerdict, TripleResult         # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / \
    "nothing_lost_root_cause_2026-10-25"
requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH - activate hwtools first")


@pytest.fixture(autouse=True)
def search_folder(monkeypatch, tmp_path):
    """Every search writes under the test's own folder."""
    monkeypatch.setattr(nothing_lost, "NOTHING_LOST_ROOT",
                        tmp_path / "nothing_lost")
    monkeypatch.setattr(distractor, "PROOF_ROOT", None)


# --- small designs -----------------------------------------------------------

def queue(ready="1'b1", clear="1'b0", counted=False, reset="rst",
          async_low=False):
    """A 4-entry queue in front of an output (module q4), with a clear
    input. in_ready is `ready`: with 1'b1 an item that arrives while the
    queue is full is dropped; with !full its sender waits. `counted` counts
    the drops on an output; `async_low` makes the reset asynchronous and
    active low."""
    ports = ["input  wire       clk", f"input  wire       {reset}",
             "input  wire       en", "input  wire       clr",
             "input  wire       in_valid", "input  wire [7:0] in_data",
             "output wire       in_ready", "input  wire       out_ready",
             "output wire       out_valid", "output wire [7:0] out_data"]
    if counted:
        ports.append("output reg  [3:0] drops")
    test = f"!{reset}" if async_low else reset
    edge = f" or negedge {reset}" if async_low else ""
    return (
        "module q4 (\n    " + ",\n    ".join(ports) + "\n);\n"
        "    reg [7:0] mem [0:3];\n"
        "    reg [2:0] wp = 3'd0;\n"
        "    reg [2:0] rp = 3'd0;\n"
        "    wire [2:0] used = wp - rp;\n"
        "    wire full = (used == 3'd4);\n"
        f"    assign in_ready = {ready};\n"
        "    wire push = en && in_valid && !full;\n"
        "    wire pop  = en && out_valid && out_ready;\n"
        "    assign out_valid = (used != 3'd0);\n"
        "    assign out_data = mem[rp[1:0]];\n"
        + ("    initial drops = 4'd0;\n" if counted else "")
        + f"    always @(posedge clk{edge}) begin\n"
        f"        if ({test}) begin wp <= 3'd0; rp <= 3'd0;"
        + (" drops <= 4'd0;" if counted else "") + " end\n"
        f"        else if ({clear}) begin wp <= 3'd0; rp <= 3'd0; end\n"
        "        else begin\n"
        "            if (push) begin mem[wp[1:0]] <= in_data; "
        "wp <= wp + 3'd1; end\n"
        "            if (pop) rp <= rp + 3'd1;\n"
        + ("            if (en && in_valid && full && drops != 4'd15) "
           "drops <= drops + 4'd1;\n" if counted else "")
        + "        end\n"
        "    end\n"
        "    always @(posedge clk) assert (used <= 3'd4);\n"
        "endmodule\n")


def parent_of(verilog):
    """The design's ports with nothing inside: the parent of a step that
    added everything the design holds."""
    return verilog[:verilog.index(");") + 2] + "\nendmodule\n"


def row(verilog, top="q4"):
    return {"id": "p0", "verilog": verilog, "top_module": top,
            "clock": "clk", "property": extract_asserts(verilog),
            "invariants": [], "antecedents": [], "sanity_covers": []}


def stream(offer="en && in_valid", ready="", delivered="pop", held="used",
           flush="", name="requests into the queue"):
    return [{"name": name, "inputs": [{"offer": offer, "ready": ready}],
             "delivered": delivered, "held": held, "flush": flush}]


def storage_error(streams, child=None, kind="STORAGE", parent=None,
                  top="q4"):
    child = child or queue(ready="!full", clear="clr", counted=True)
    parent = parent or parent_of(child)
    return nothing_lost.declaration_error(streams, kind, child, top, "clk",
                                          parent=row(parent, top))


def top_of(verilog):
    return re.findall(r"\bmodule\s+(\w+)", verilog)[-1]


def compose_error(streams, child):
    top = top_of(child)
    return nothing_lost.declaration_error(
        streams, "compose", child, top, "clk",
        wrapper=nothing_lost.top_text(child, top))


# a sender inside the design: a stage that offers its item to a 2-entry
# queue; HOLDS keeps the item until it is taken, GIVES drops it the next
# cycle whatever happened
PRODUCER = """module prodq (
    input  wire clk, input wire rst, input wire en, input wire start,
    input  wire out_ready, output wire out_valid, output wire [7:0] out_data
);
    reg       st_v = 1'b0;
    reg [7:0] st_d = 8'd0;
    reg [7:0] mem [0:1];
    reg [1:0] wp = 2'd0, rp = 2'd0;
    wire [1:0] used = wp - rp;
    wire full = (used == 2'd2);
    wire take = en && st_v && !full;
    wire pop = en && out_valid && out_ready;
    assign out_valid = used != 2'd0;
    assign out_data = mem[rp[0]];
    always @(posedge clk) begin
        if (rst) begin st_v <= 1'b0; wp <= 2'd0; rp <= 2'd0; end
        else if (en) begin
            if (take) begin mem[wp[0]] <= st_d; wp <= wp + 2'd1; end
            if (pop) rp <= rp + 2'd1;
            STAGE
        end
    end
endmodule
"""
HOLDS = ("if (start && (!st_v || take)) begin st_v <= 1'b1; "
         "st_d <= st_d + 8'd1; end else if (take) st_v <= 1'b0;")
GIVES = ("if (start) begin st_v <= 1'b1; st_d <= st_d + 8'd1; end "
         "else st_v <= 1'b0;")
STAGE_STREAM = stream("st_v", "en && !full", "pop", "used",
                      name="stage into the queue")

# a PEER step: a second lane (b_valid, b_data, b_ready) shares one output
# register with the existing lane, and wins every conflict
TWO_LANES = """module two_lane (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       in_valid,
    input  wire [7:0] in_data,
    output wire       in_ready,
    input  wire       b_valid,
    input  wire [7:0] b_data,
    output wire       b_ready,
    input  wire       out_ready,
    output reg        out_valid,
    output reg  [7:0] out_data
);
    initial out_valid = 1'b0;
    initial out_data = 8'd0;
    wire slot_free = !out_valid || out_ready;
    wire grant_b = b_valid && slot_free;
    wire grant_a = in_valid && slot_free && !b_valid;
    assign b_ready  = en && slot_free;
    assign in_ready = OLD_READY;
    always @(posedge clk) begin
        if (rst) begin out_valid <= 1'b0; out_data <= 8'd0; end
        else if (en) begin
            if (grant_b) begin out_valid <= 1'b1; out_data <= b_data; end
            else if (grant_a) begin out_valid <= 1'b1; out_data <= in_data; end
            else if (out_ready) out_valid <= 1'b0;
        end
    end
    always @(posedge clk) assert (!out_valid || 1'b1);
endmodule
"""
ONE_LANE = """module two_lane (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       in_valid,
    input  wire [7:0] in_data,
    input  wire       out_ready,
    output reg        out_valid,
    output reg  [7:0] out_data
);
endmodule
"""


def lanes_stream(old_ready):
    return [{"name": "the shared output",
             "inputs": [{"offer": "en && b_valid", "ready": "b_ready"},
                        {"offer": "en && in_valid", "ready": old_ready}],
             "delivered": "en && out_valid && out_ready",
             "held": "out_valid", "flush": ""}]


# a combination: prod offers one result every two cycles, cons is busy for
# three cycles after each start, and a one-entry slot (g_valid) sits
# between them; A_CE is what drives prod's enable
PROD = """module prod (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg        o_valid,
    output reg  [7:0] o_data
);
    reg ph = 1'b0;
    initial o_valid = 1'b0;
    initial o_data = 8'd0;
    always @(posedge clk) begin
        if (rst) begin ph <= 1'b0; o_valid <= 1'b0; o_data <= 8'd0; end
        else if (en) begin
            ph <= !ph;
            o_valid <= ph;
            if (ph) o_data <= o_data + 8'd1;
        end
    end
    always @(posedge clk) assert (o_data <= 8'd255);
endmodule
"""
CONS = """module cons (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       i_start,
    input  wire [7:0] i_data,
    output wire       o_busy,
    output reg  [7:0] o_sum
);
    reg [1:0] left = 2'd0;
    initial o_sum = 8'd0;
    assign o_busy = (left != 2'd0);
    always @(posedge clk) begin
        if (rst) begin left <= 2'd0; o_sum <= 8'd0; end
        else if (en) begin
            if (i_start && !o_busy) begin left <= 2'd3; o_sum <= o_sum + i_data; end
            else if (o_busy) left <= left - 2'd1;
        end
    end
    always @(posedge clk) assert (left <= 2'd3);
endmodule
"""
PAIR = """module pair (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output wire       o_ready,
    output wire [7:0] o_sum
);
    wire a_valid; wire [7:0] a_data; wire b_busy;
    reg g_valid = 1'b0; reg [7:0] g_data = 8'd0;
    wire g_take = en && g_valid && !b_busy;
    wire a_ce = A_CE;
    wire a_valid_en = en && a_valid;
    assign o_ready = a_ce;
    prod u_a (.clk(clk), .rst(rst), .en(a_ce), .o_valid(a_valid), .o_data(a_data));
    cons u_b (.clk(clk), .rst(rst), .en(en), .i_start(g_take), .i_data(g_data), .o_busy(b_busy), .o_sum(o_sum));
    always @(posedge clk) begin
        if (rst) begin g_valid <= 1'b0; g_data <= 8'd0; end
        else begin
            if (a_ce && a_valid && (!g_valid || g_take)) begin g_valid <= 1'b1; g_data <= a_data; end
            else if (g_take) g_valid <= 1'b0;
        end
    end
endmodule
"""
CANNOT_WAIT = "en"
WAITS = "en && !(g_valid && !g_take)"


def pair(a_ce=WAITS, b_to_a=False):
    """The combination; with b_to_a the sending design is the second
    parent (B), as when a plan sends from B to A."""
    parts = CONS + "\n" + PROD if b_to_a else PROD + "\n" + CONS
    return parts + "\n" + PAIR.replace("A_CE", a_ce)


def pair_stream(offer="a_valid", ready="a_ce", delivered="g_take",
                held="g_valid"):
    return stream(offer, ready, delivered, held,
                  name="results into the consumer")


# a step's prefetch unit pulls words from a counter while it is active and
# more remain, into a 4-entry buffer, and waits while the buffer is full
PREFETCH = """module pf (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       start,
    input  wire [3:0] count,
    input  wire       out_ready,
    output wire       out_valid,
    output wire [7:0] out_data
);
    reg [3:0] left = 4'd0;
    reg [7:0] word = 8'd0;
    reg       pf_active = 1'b0;
    wire      pf_more = (left != 4'd0);
    reg [7:0] pf_buf [0:3];
    reg [2:0] pf_count = 3'd0;
    reg [1:0] pf_wr = 2'd0;
    reg [1:0] pf_rd = 2'd0;
    wire      pf_full = (pf_count == 3'd4);
    wire      pf_take = en && pf_active && pf_more && !pf_full;
    wire      pop = en && out_valid && out_ready;
    assign out_valid = (pf_count != 3'd0);
    assign out_data = pf_buf[pf_rd];
    always @(posedge clk) begin
        if (rst) begin
            left <= 4'd0; pf_active <= 1'b0; pf_count <= 3'd0;
            pf_wr <= 2'd0; pf_rd <= 2'd0;
        end else if (en) begin
            if (start && !pf_active) begin
                pf_active <= 1'b1;
                left <= count;
            end else if (pf_active && !pf_more) begin
                pf_active <= 1'b0;
            end
            if (pf_take) begin
                pf_buf[pf_wr] <= word;
                word <= word + 8'd1;
                left <= left - 4'd1;
                pf_wr <= pf_wr + 2'd1;
            end
            if (pop) pf_rd <= pf_rd + 2'd1;
            if (pf_take && !pop) pf_count <= pf_count + 3'd1;
            else if (pop && !pf_take) pf_count <= pf_count - 3'd1;
        end
    end
endmodule
"""
PREFETCH_PARENT = parent_of(PREFETCH).replace(
    "\nendmodule", "\n    reg [3:0] left = 4'd0;\n    reg [7:0] word = 8'd0;\n"
    "    assign out_valid = 1'b0;\n    assign out_data = word;\nendmodule")

# a one-entry stage whose state is an empty flag
ONE_ENTRY = """module stage1 (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    input  wire       in_valid,
    input  wire [7:0] in_data,
    output wire       in_ready,
    input  wire       out_ready,
    output wire       out_valid,
    output wire [7:0] out_data
);
    reg       empty = 1'b1;
    reg [7:0] d = 8'd0;
    assign in_ready = en && (empty || out_ready);
    assign out_valid = !empty;
    assign out_data = d;
    always @(posedge clk) begin
        if (rst) empty <= 1'b1;
        else if (en) begin
            if (in_valid && (empty || out_ready)) begin
                d <= in_data;
                empty <= 1'b0;
            end else if (out_ready) begin
                empty <= 1'b1;
            end
        end
    end
endmodule
"""


def fixture(name):
    return (FIXTURES / name).read_text()


def search(child, streams, parents=(), wrapper=None, **fields):
    """nothing_lost.check on a fresh record: (sent back, record)."""
    record = dict(fields, streams=streams)
    top = top_of(child)
    sent = nothing_lost.check(record, child, top, "clk", list(parents),
                              wrapper=wrapper)
    return sent, record


def search_combination(child, streams, **fields):
    top = top_of(child)
    return search(child, streams, wrapper=nothing_lost.top_text(child, top),
                  ext_type="compose", **fields)


# --- the questions and their schemas (no tools) ------------------------------

def squash(text):
    return " ".join(text.split())


def every_question_text():
    """Every text a builder or the reviewer is shown."""
    return ([extend.SYSTEM_PROMPT, extend.STRUCTURAL_TEMPLATE,
             extend.SECOND_TEMPLATE, extend.COMPOSE_TEMPLATE,
             extend.REPLICATE_TEMPLATE, extend.PARTNER_TEMPLATE,
             extend.PLAN_TEXT, extend.WAITING_RULE, extend.STREAMS_TEXT,
             extend.EFFECT_RULE, extend.LEDGER_EFFECT_RULE,
             extend.SIDEWAYS_EFFECT_RULE, extend.PEER_EFFECT_RULE,
             sense_check.PROMPT]
            + list(extend.MOVES.values())
            + list(extend.STREAMS_RULES.values()))


def test_no_question_accepts_a_counted_or_flagged_drop():
    """Five kept designs of the 25 Oct study dropped work and counted or
    flagged it (drop_cnt, q_ovf)."""
    moves = extend.MOVES
    for text in (extend.WAITING_RULE, moves["SUBUNIT"], moves["STORAGE"],
                 moves["PEER"], extend.COMPOSE_TEMPLATE,
                 extend.PARTNER_TEMPLATE):
        assert "counted or not" in squash(text)
    for text in (extend.WAITING_RULE, extend.COMPOSE_TEMPLATE):
        assert "flagging" in squash(text)
    # "never drop data silently" let a counted drop through; the combine
    # question's "the toolchain silently turns them into dangling wires" is
    # about instance.signal, not about losing an item
    for text in list(moves.values()) + [extend.COMPOSE_TEMPLATE,
                                        extend.PARTNER_TEMPLATE,
                                        sense_check.PROMPT]:
        for sentence in squash(text).split("."):
            if "silently" in sentence:
                assert not re.search(r"\b(drop|lose|lost|loss|overwrit)",
                                     sentence), sentence


def test_no_question_advises_freezing_the_whole_sender():
    """An enable held low froze whole designs: t033 and c2_004 lost the
    route-table writes sent during a pause."""
    for text in every_question_text():
        t = squash(text)
        assert "holding A's enable low" not in t
        assert "an enable that holds it" not in t
    waiting = squash(extend.WAITING_RULE)
    assert "never by gating the module's enable for everything" in waiting
    assert "shows on an output" in waiting
    assert "Never hold it low to make the design wait" in \
        squash(extend.COMPOSE_TEMPLATE)


def test_big_moves_carry_the_waiting_and_streams_blocks_and_small_ones_do_not():
    parent = row(queue())
    for move in ("SUBUNIT", "STORAGE", "PEER"):
        text = squash(extend.build_prompt(parent, "structural", move))
        assert "WAITING. Never drop work" in text, move
        assert "STREAMS." in text, move
        assert '"streams": [...]' in text, move
        assert ("Then write the patch, then streams (see STREAMS above), "
                "then the full new invariant list") in text, move
    for move, k in (("STAGE", 4), ("SPLIT", 1)):
        text = squash(extend.build_prompt(parent, "structural", move, k=k))
        for words in ("WAITING. Never drop work", "STREAMS.",
                      '"streams": [...]', "then streams"):
            assert words not in text, (move, words)
        # the question of a small move reads as it did before
        assert "Then write the patch, then the full new invariant list" in \
            text, move


def test_each_big_move_says_what_it_must_declare():
    parent = row(queue())
    rules = {"STORAGE": "must declare at least one stream through its storage",
             "PEER": "at least two inputs",
             "SUBUNIT": "a unit that adds an array must"}
    for move, words in rules.items():
        text = squash(extend.build_prompt(parent, "structural", move))
        assert words in text, move
        for other in set(rules) - {move}:
            assert rules[other] not in text, (move, other)


def test_the_combine_question_asks_for_the_stream_between_the_designs():
    text = squash(extend.compose_prompt(row(PROD, "prod"), row(CONS, "cons")))
    assert "STREAMS." in text
    assert squash(extend.STREAMS_RULES["compose"]) in text
    assert "delivered includes that condition" in text
    assert '"wrapper": "...", "streams": [...], "antecedents": [...]' in text
    assert ("Then the wrapper module, then streams (see STREAMS above), "
            "then the complete invariant list") in text


def _schema_keys(node):
    """Every property name and every key anywhere in a schema."""
    if isinstance(node, dict):
        out = set(node)
        for k, v in node.items():
            out |= _schema_keys(v)
            if k == "properties":
                out |= set(v)
        return out
    if isinstance(node, list):
        return set().union(*(_schema_keys(x) for x in node)) if node else set()
    return set()


def test_the_stream_schemas_have_fixed_keys_and_order():
    storage = extend.structural_schema("STORAGE")
    props = list(storage["properties"])
    assert props == ["new_state", "coupling", "induction_gap",
                     "why_parent_insufficient", "patch", "streams",
                     "invariants", "dispositions"]
    assert storage["required"] == props
    assert storage["additionalProperties"] is False
    stream_item = storage["properties"]["streams"]["items"]
    assert list(stream_item["properties"]) == ["name", "inputs", "delivered",
                                               "held", "flush"]
    assert stream_item["required"] == ["name", "inputs", "delivered", "held",
                                       "flush"]
    assert stream_item["additionalProperties"] is False
    entry = stream_item["properties"]["inputs"]["items"]
    assert list(entry["properties"]) == ["offer", "ready"]
    assert entry["required"] == ["offer", "ready"]
    assert entry["additionalProperties"] is False
    for move in extend.BIG_MOVES:
        assert extend.structural_schema(move) is storage
    for move in ("STAGE", "SPLIT", "GUARD", "BOUND", "LEDGER"):
        assert extend.structural_schema(move) is extend.STRUCTURAL_SCHEMA
    assert "streams" not in extend.STRUCTURAL_SCHEMA["properties"]
    assert list(extend.COMPOSE_SCHEMA["properties"]) == [
        "glue_scheme", "induction_gap", "why_parents_insufficient",
        "top_module", "wrapper", "streams", "antecedents", "sanity_covers",
        "invariants"]
    assert "streams" in extend.COMPOSE_SCHEMA["required"]
    assert extend.COMPOSE_SCHEMA["properties"]["streams"] == \
        storage["properties"]["streams"]
    for schema in (extend.SECOND_SCHEMA, extend.REPLICATE_SCHEMA):
        assert "streams" not in schema["properties"]
    # counts are checked in code, and a field named "reasoning" is blocked
    # before it reaches the model (test_distractor)
    for schema in (storage, extend.STRUCTURAL_SCHEMA, extend.COMPOSE_SCHEMA,
                   extend.SECOND_SCHEMA, extend.REPLICATE_SCHEMA):
        keys = _schema_keys(schema)
        assert not keys & {"minItems", "maxItems", "reasoning"}


def test_the_runner_asks_for_the_schema_of_the_move(monkeypatch):
    sent = []

    def model(prompt, schema):
        sent.append(schema)
        return None, {"input": 1, "output": 1}, "refusal"
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    parent = row(queue())
    for move in ("STORAGE", "STAGE"):
        rec = batch._real_executor({"task_id": 1, "attempt": 1,
                                    "ext_type": "structural", "move": move,
                                    "parent_id": "p0", "k": 1}, [parent])
        assert rec["verdict"] == "REFUSED"
    storage, stage = sent
    assert storage is extend.structural_schema("STORAGE")
    assert "streams" in storage["properties"]
    assert stage is extend.STRUCTURAL_SCHEMA
    assert "streams" not in stage["properties"]


# --- what each kind of step must declare (pure) ------------------------------

def test_storage_and_combinations_must_declare_a_stream():
    assert storage_error([]) == (
        "A STORAGE step must declare at least one stream through its "
        "storage (for a register file or lookup table, its writes); this "
        "answer declares none.")
    assert compose_error([], pair()).startswith(
        "A combination must declare at least one stream from one design "
        "into the other")
    # a sub-unit that carries no items and adds no array (a timer)
    timer = parent_of(queue()).replace(
        "\nendmodule", "\n    reg [3:0] t = 4'd0;\n"
        "    always @(posedge clk) t <= t + 4'd1;\nendmodule")
    assert storage_error([], child=timer, kind="SUBUNIT",
                         parent=parent_of(queue())) is None
    # one that adds an array must declare a stream through it
    assert storage_error([], child=queue(), kind="SUBUNIT") == (
        "This step adds the array mem, so it must declare at least one "
        "stream through the new unit (for a register bank, its writes).")


@pytest.mark.parametrize("field, value, says", [
    ("held", "the queue count", "(the queue count) is not a plain same-cycle "
     "Verilog expression over the top module's signals: it is prose."),
    ("held", "q_count", "held names q_count, which module q4 does not "
     "declare."),
    ("held", "wp - rp", "held (wp - rp) subtracts. The checker reads counts "
     "at 16 bits, where a difference of wrapping pointers comes out wrong"),
    ("delivered", "pop - 1'b0", "delivered (pop - 1'b0) subtracts."),
])
def test_a_value_that_is_prose_names_a_missing_signal_or_subtracts_is_sent_back(
        field, value, says):
    honest = stream(ready="in_ready")[0]
    err = storage_error([dict(honest, **{field: value})])
    assert err.startswith("stream 'requests into the queue': ")
    assert says in err
    assert storage_error([honest]) is None
    # a minus inside an index bracket, even a nested one, is no subtraction
    for delivered in ("pop && wp[2-1]", "pop && mem[rp[1:0]-1] != 8'd0"):
        assert storage_error([dict(honest, delivered=delivered)]) is None


@pytest.mark.parametrize("streams, says", [
    ("none", '"streams" must be a list of the streams your step adds or '
     "changes, as the question explains."),
    (["not a stream"], '"streams" must be a list'),
    ([{"name": "q", "inputs": [{"offer": 1, "ready": ""}],
       "delivered": "pop", "held": "used", "flush": ""}],
     '"streams" must be a list'),
    (stream(ready="in_ready") * 4, "Declare at most 3 streams; this answer "
     "declares 4."),
    ([dict(stream()[0], inputs=[])], "stream 'requests into the queue' has "
     "no inputs: give one entry for each place items enter it, with offer "
     "and ready."),
    ([dict(stream()[0], inputs=stream()[0]["inputs"] * 5)],
     "stream 'requests into the queue' has 5 inputs; at most 4."),
    (stream(offer=""), "offer is empty. offer, delivered and held must each "
     'be a Verilog expression ("0" for held when the stream holds nothing).'),
    (stream(held=" "), "held is empty."),
    (stream(offer="en && in_valid;"), "it has a ;"),
    (stream(held="$countones(mem)"), "it uses a $ function"),
    (stream(held="u_q.count"), "it names an instance (instance.signal)"),
    (stream(held="(used"), "its brackets do not match"),
    (stream(offer="en = in_valid"), "it assigns with ="),
    (stream(held="used + " * 60 + "used"), "it is longer than 300 "
     "characters; name a wire of the design instead"),
    (stream(offer="1'b1"), "stream 'requests into the queue': offer (1'b1) "
     "reads no signal."),
    (stream(ready="in_ready && clk"), "ready reads the clock clk."),
])
def test_a_declaration_of_the_wrong_shape_or_size_is_sent_back(streams,
                                                               says):
    err = storage_error(streams)
    assert err is not None and says in err


def test_an_offer_that_counts_only_items_taken_is_sent_back():
    for offer in ("push", "en && in_valid && !full"):
        err = storage_error(stream(offer=offer))
        assert err.startswith(f"stream 'requests into the queue': the offer "
                              f"{offer} reads rp, used, wp, which delivered "
                              "or held also reads, so it folds in whether "
                              "the item will be taken")
    assert storage_error(stream(offer="en && in_valid")) is None
    assert storage_error(stream(offer="en && in_valid",
                                ready="in_ready")) is None


def test_an_edge_detected_offer_and_a_pulling_source_are_allowed():
    # sys_c1_007's honest offer: one pulse per result of a done level that
    # stays high, through done_q, a register of the wrapper
    child = fixture("sys_c1_007_kept.sv")
    assert compose_error(stream("a_done && !done_q", "", "q_pop", "q_cnt",
                                name="results into the fold"), child) is None
    assert compose_error(stream("a_fin", "", "q_pop", "q_cnt"), child) is None
    # a prefetch unit that pulls words while it is active and more remain
    assert storage_error(stream("pf_active && pf_more", "en && !pf_full",
                                "pop", "pf_count"),
                         child=PREFETCH, parent=PREFETCH_PARENT,
                         top="pf") is None


def test_an_outside_sender_waits_on_an_output_port():
    err = storage_error(stream(ready="!full"))
    assert err == (
        "stream 'requests into the queue': the offer en && in_valid comes "
        "from outside the design (it reads only input ports), and such a "
        "sender sees only the design's ports: ready must read an output "
        'port it can see ("in_ready", "!busy_o") and nothing but ports, or '
        'be "" when the sender cannot be told to wait; !full does not.')
    assert storage_error(stream(ready="in_ready")) is None
    assert storage_error(stream(ready="")) is None


def test_a_sender_inside_keeps_the_enable_out_of_its_offer():
    child = fixture("sys_c1_003_skeptic_rebuild.sv")
    beats = dict(delivered="en && g_valid && ready", held="g_valid",
                 name="beats to the port")
    err = compose_error(stream("en && a_out_valid", "en && slot_free_w",
                               **beats), child)
    assert err == (
        "stream 'beats to the port': the offer en && a_out_valid reads the "
        "input port(s) en. For a sender inside the design, offer is the "
        "sender's own valid, which stays true while it waits; put the "
        "design's enable in ready instead.")
    assert compose_error(stream("a_out_valid", "en && slot_free", **beats),
                         child) is None
    assert compose_error(stream("a_out_valid", "en && slot_free_w", **beats),
                         child) is None
    # the kept design's sender cannot wait: its enable goes in the offer,
    # with no ready
    assert compose_error(stream("en && a_out_valid", "", **beats),
                         fixture("sys_c1_003_kept.sv")) is None


def test_an_offer_read_through_a_wire_is_judged_by_what_the_wire_reads():
    """26 Oct 2026: the rules follow the design's wires, so an inside
    offer that reads the enable through a wire is still sent back, and an
    offer made of input ports through a wire is still from outside (no
    check that it keeps its item, which the outside may take back)."""
    err = compose_error(pair_stream(offer="a_valid_en"), pair())
    assert "the offer a_valid_en reads the input port(s) en" in err
    lane = TWO_LANES.replace("OLD_READY", "en && slot_free && !b_valid") \
        .replace("    wire slot_free", "    wire in_offer = en && in_valid;\n"
                 "    wire slot_free")
    declared = stream("in_offer", "slot_free", "en && out_valid && out_ready",
                      "out_valid")
    err = storage_error(declared, child=lane, kind="STORAGE",
                        parent=parent_of(lane), top="two_lane")
    assert "the offer in_offer comes from outside the design" in err
    waits = [dict(declared[0],
                  inputs=[{"offer": "in_offer", "ready": "in_ready"}])]
    assert storage_error(waits, child=lane, kind="STORAGE",
                         parent=parent_of(lane), top="two_lane") is None
    copy, line_map, _ = nothing_lost.checked_copy(
        lane, "two_lane", "clk", stream("in_offer", "in_ready",
                                        "en && out_valid && out_ready",
                                        "out_valid"), "(rst)")
    assert "gave_up" not in {kind for _, kind, _ in line_map.values()}


def test_a_combinations_offer_reads_one_design_and_its_ready_reaches_it():
    for b_to_a in (False, True):
        child = pair(b_to_a=b_to_a)
        # the direction comes from the stream, never from A and B
        assert compose_error(pair_stream(), child) is None
        err = compose_error(pair_stream(offer="a_valid && !b_busy", ready=""),
                            child)
        assert err == (
            "stream 'results into the consumer': the offer a_valid && "
            "!b_busy reads outputs of u_a and u_b; it must read only the "
            "sending design's own outputs: reading the receiver folds in "
            "whether it can take the item.")
        err = compose_error(pair_stream(ready="en"), child)
        assert err == (
            "stream 'results into the consumer': ready en reaches no input "
            "of u_a, the design that sends the items, so it cannot make it "
            "wait. Name what drives its ready, busy, request or valid input, "
            'or write "" if it cannot wait.')
        # a delivered that never reaches the receiving design
        err = compose_error(pair_stream(delivered="a_valid", held="0"), child)
        assert err.startswith("A combination must declare at least one "
                              "stream from one design into the other")
    # the wrapper's enable reaches every part, so it never makes a
    # delivered reach the receiver (26 Oct 2026)
    child = fixture("sys_c1_007_kept.sv")
    for delivered in ("a_fin", "fold_en && a_fin"):
        assert compose_error(stream("a_fin", "", delivered, "0"),
                             child).startswith("A combination must declare")


def test_held_may_not_read_an_input_and_no_value_may_name_a_drop_counter():
    err = storage_error(stream(ready="in_ready", held="used + in_valid"))
    assert err == (
        "stream 'requests into the queue': held used + in_valid reads the "
        "input port(s) in_valid; held counts the items inside the stream, "
        "so it reads only the design's own state.")
    err = storage_error(stream(ready="in_ready", delivered="pop + drops"))
    assert err == (
        "stream 'requests into the queue': delivered names drops, a signal "
        "that counts or flags lost items. A lost item is neither delivered "
        "nor held: never count it there, and make the sender wait instead "
        "of losing it.")
    # words are split on _ and digits: q_ovf flags an overflow
    err = compose_error(stream("a_fin", "", "q_pop", "q_cnt + q_ovf"),
                        fixture("sys_c1_007_kept.sv"))
    assert "held names q_ovf, a signal that counts or flags lost items" in err


def test_a_flush_must_be_an_outside_command():
    for flush, names in (("in_valid", "in_valid"), ("en", "en"),
                         ("wp", "wp"), ("rst || clr", "rst")):
        err = storage_error(stream(ready="in_ready", flush=flush))
        assert err == (
            f"stream 'requests into the queue': flush {flush} must read only "
            "input ports that empty the stream on purpose (a clear, abort or "
            "flush command), never the enable, the reset or the stream's own "
            f"offer or ready signals; it reads {names}.")
    assert storage_error(stream(ready="in_ready", flush="clr")) is None


def test_a_second_lane_declares_both_lanes_through_the_shared_thing():
    """The rule's own test is the second-lane file's; this one pins that
    the two-lane design the searches below use is declared honestly."""
    for old_ready in ("", "in_ready"):
        assert storage_error(lanes_stream(old_ready), kind="PEER",
                             child=TWO_LANES.replace("OLD_READY", "1'b1"),
                             parent=ONE_LANE, top="two_lane") is None
    old_lane_only = [dict(lanes_stream("")[0],
                          inputs=lanes_stream("")[0]["inputs"][1:])]
    err = storage_error(old_lane_only, kind="PEER",
                        child=TWO_LANES.replace("OLD_READY", "1'b1"),
                        parent=ONE_LANE, top="two_lane")
    assert err.startswith("A PEER step must declare the stream through the "
                          "thing its two lanes share")
    assert "(b_data, b_valid)" in err


@pytest.fixture
def checker(monkeypatch):
    """The checker faked: NECESSARY, the checks counted, and the search
    counted too."""
    seen = {"graded": 0, "searched": 0}

    def triple(payload, **kw):
        seen["graded"] += 1
        return TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")

    def check(*a, **k):
        seen["searched"] += 1
        return False
    for mod in (extend, distractor):
        monkeypatch.setattr(mod, "state_bits", lambda v, top: len(v))
        monkeypatch.setattr(mod, "grade_triple_generated", triple)
    monkeypatch.setattr(nothing_lost, "check", check)
    return seen


SMALL = """module sp (input wire clk, input wire rst, input wire en, output wire [2:0] a_o);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    always @(posedge clk) begin
        if (rst) begin a <= 3'd0; b <= 3'd0; end
        else if (en) begin
            a <= (a == 3'd4) ? 3'd0 : a + 3'd1;
            b <= (a == 3'd4) ? 3'd0 : a + 3'd1;
        end
    end
    assign a_o = a;
    always @(posedge clk) assert (b <= 3'd4);
endmodule
"""


def test_answers_written_before_the_rule_are_not_judged(checker):
    parent = dict(row(SMALL, "sp"), invariants=["a == b", "a <= 3'd4"],
                  antecedents=[], sanity_covers=["a == 3'd4"])
    out = {"patch": ("@@ assign a_o = a; @@\n"
                     "+    reg [7:0] hist [0:3];\n"
                     "+    always @(posedge clk) if (en) hist[a[1:0]] <= "
                     "{5'd0, a};"),
           "invariants": ["a == b", "a <= 3'd4"],
           "dispositions": [{"clause": c, "status": "kept", "replaced_by": [],
                             "reason": ""} for c in ("a == b", "a <= 3'd4")]}
    record = {"extension_id": "t_old", "parent_id": "p0",
              "ext_type": "structural", "move": "STORAGE"}
    rec = extend.grade_step4(parent, "structural", out, record)
    assert rec["verdict"] == "NECESSARY" and checker["graded"] == 1
    assert checker["searched"] == 0
    assert "nothing_lost" not in rec and "streams" not in rec


# --- where the search sits in the grading (no tools) -------------------------

HISTORY = ("@@ assign a_o = a; @@\n"
           "+    reg [7:0] hist [0:DEPTH];\n"
           "+    always @(posedge clk) if (en) hist[a[1:0]] <= {5'd0, a};")
# a part with an assertion of its own and a count of its writes
COUNTED = ("@@ assign a_o = a; @@\n"
           "+    reg [7:0] hist [0:3];\n"
           "+    reg [2:0] n_w = 3'd0;\n"
           "+    always @(posedge clk) if (en) begin\n"
           "+        hist[a[1:0]] <= {5'd0, a};\n"
           "+        n_w <= (n_w == 3'd4) ? n_w : n_w + 3'd1;\n"
           "+    end\n"
           "+    always @(posedge clk) assert (n_w <= 3'd4);")
WRITES = [{"name": "writes into the history",
           "inputs": [{"offer": "en", "ready": ""}],
           "delivered": "en", "held": "0", "flush": ""}]


def test_the_search_runs_after_the_early_checks_and_before_the_proof(
        checker, monkeypatch):
    parent = dict(row(SMALL, "sp"), invariants=["a == b", "a <= 3'd4"],
                  sanity_covers=["a == 3'd4"])
    calls = []

    def search(record, child, top, clock, parents, wrapper=None):
        calls.append({"top": top, "clock": clock, "parents": parents,
                      "wrapper": wrapper,
                      "after_yosys": "child_state_bits" in record})
        return False
    monkeypatch.setattr(nothing_lost, "check", search)

    def grade(patch):
        out = {"patch": patch, "invariants": list(parent["invariants"]),
               "dispositions": [{"clause": c, "status": "kept",
                                 "replaced_by": [], "reason": ""}
                                for c in parent["invariants"]],
               "streams": WRITES}
        return extend.grade_step4(parent, "structural", out,
                                  {"extension_id": "t35", "parent_id": "p0",
                                   "ext_type": "structural",
                                   "move": "STORAGE"})
    # storage too deep for the proof is sent back before the search
    rec = grade(HISTORY.replace("DEPTH", "31"))
    assert rec["verdict"] == "STORAGE_TOO_DEEP" and calls == []
    # and so is a part with an assertion of its own that no fact links to
    # the old state
    rec = grade(COUNTED)
    assert rec["verdict"] == "NO_COUPLING" and calls == []
    # an honest step is searched once, after yosys read it, then proved
    rec = grade(HISTORY.replace("DEPTH", "3"))
    assert rec["verdict"] == "NECESSARY" and checker["graded"] == 1
    assert calls == [{"top": "sp", "clock": "clk", "parents": [parent],
                      "wrapper": None, "after_yosys": True}]
    assert rec["streams"] == WRITES

    # a step the search sends back never reaches the proof
    def loses(record, *a, **k):
        record["nothing_lost"] = nothing_lost._result("lost", step=6,
                                                      stream=0, depth=48)
        record["verdict"] = "ITEMS_LOST"
        record["error"] = nothing_lost.lesson(record["nothing_lost"],
                                              record["streams"])
        return True
    monkeypatch.setattr(nothing_lost, "check", loses)
    rec = grade(HISTORY.replace("DEPTH", "3"))
    assert rec["verdict"] == "ITEMS_LOST" and checker["graded"] == 1


def test_the_lesson_reaches_the_retry_question():
    found = nothing_lost._result("lost", step=6, stream=0, depth=48)
    record = {"usage": {"input": 1, "output": 1}, "ext_type": "structural",
              "move": "STORAGE", "new_state": "a 4-entry queue in front of "
              "the output.", "verdict": "ITEMS_LOST",
              "error": nothing_lost.lesson(found, stream()),
              "reason": nothing_lost.reason(found, stream())}
    lesson = batch.retry_lesson(record)
    assert "stream 'requests into the queue' losing an item at cycle 6" in \
        lesson
    assert lesson in batch.retry_section(lesson)
    for verdict in ("NEW_INPUT", "STREAM_INVALID", "ITEMS_LOST",
                    "PAUSES_WHOLE_DESIGN"):
        assert promote.route(verdict) == "reformat", verdict
    # later questions on the design list a lossy step as tried and rejected,
    # by its one-line reason; a declaration or a new input is how the answer
    # was written, and its retry is told why instead
    assert "ITEMS_LOST" in batch.REMEMBERED
    assert "STREAM_INVALID" not in batch.REMEMBERED
    assert "NEW_INPUT" not in batch.REMEMBERED
    assert batch.rejection_note(record) == (
        "STORAGE: a 4-entry queue in front of the output. Rejected: the "
        "checker's nothing-is-lost search found stream 'requests into the "
        "queue' losing an item at cycle 6")


# --- the copy the search runs on (pure) --------------------------------------

def test_the_copy_drops_the_designs_assertions_and_adds_only_the_checkers_counts():
    child = queue(ready="!full")
    copy, line_map, (first, last) = nothing_lost.checked_copy(
        child, "q4", "clk", stream(ready="in_ready"), "(rst)")
    lines = copy.splitlines()
    assert "assert (used" not in copy
    assert lines[first - 1] == nothing_lost.HEADER
    assert lines[last] == "endmodule"            # before the top's endmodule
    block = lines[first - 1:last]
    asserts = {first + k for k, line in enumerate(block) if "assert (" in line}
    assert asserts == set(line_map)
    kinds = {"nothing lost": "lost", "counts add up": "count_off",
             "kept while told to wait": "gave_up"}
    for n, (i, kind, j) in line_map.items():
        assert i == 0
        assert any(kinds[w] == kind for w in kinds if w in lines[n - 1])
    assert "wire [15:0] oracle_nl_give_0 = pop;" in copy
    assert "wire [15:0] oracle_nl_held_0 = used;" in copy
    assert "always @(*) assume (((rst)) == !oracle_nl_started);" in copy
    # a sender from outside the design may take back an item never taken
    assert "gave_up" not in {k for _, k, _ in line_map.values()}
    # a sender inside, with a ready, must keep its item while it waits
    copy, line_map, _ = nothing_lost.checked_copy(
        PRODUCER.replace("STAGE", HOLDS), "prodq", "clk", STAGE_STREAM,
        "(rst)")
    waits = [n for n, (_, kind, j) in line_map.items() if kind == "gave_up"]
    assert len(waits) == 1 and line_map[waits[0]] == (0, "gave_up", 0)
    assert copy.splitlines()[waits[0] - 1].strip().startswith(
        "assert (|(st_v));")
    # and one with no ready counts each cycle it offers, with no such check
    _, line_map, _ = nothing_lost.checked_copy(
        PRODUCER.replace("STAGE", HOLDS), "prodq", "clk",
        stream("st_v && en", "", "pop", "used"), "(rst)")
    assert "gave_up" not in {k for _, k, _ in line_map.values()}


def test_the_copy_keeps_its_own_names_and_lines_whatever_the_design_says():
    child = queue(ready="!full").replace(
        "    reg [2:0] wp", "    wire oracle_nl_x = 1'b0;\n    reg [2:0] wp")
    copy, line_map, _ = nothing_lost.checked_copy(
        child, "q4", "clk", stream(ready="in_ready"), "(rst)")
    assert "oracle_nl2_in_0" in copy and "oracle_nl_in_0" not in copy
    # a value written over several lines, with a comment, stays on one line
    copy, line_map, (first, last) = nothing_lost.checked_copy(
        queue(ready="!full"), "q4", "clk",
        stream(ready="in_ready", delivered="en &&\n out_valid // taken\n"
                                           " && out_ready"), "(rst)")
    assert "oracle_nl_give_0 = en && out_valid && out_ready;" in copy
    lines = copy.splitlines()
    assert all("assert (" in lines[n - 1] for n in line_map)
    assert lines[last] == "endmodule"


def test_a_not_in_a_count_is_sized_by_itself():
    """26 Oct 2026: counts are read at 16 bits so that sums of 1-bit flags
    add up, and at 16 bits ~empty would come out as 65534 or 65535."""
    copy, _, _ = nothing_lost.checked_copy(
        ONE_ENTRY, "stage1", "clk",
        stream("en && in_valid", "in_ready", "en & ~(empty) & out_ready",
               "~empty + ~&d[1:0]"), "(rst)")
    assert "oracle_nl_give_0 = en & {~(empty)} & out_ready;" in copy
    assert "oracle_nl_held_0 = {~empty} + ~&d[1:0];" in copy


def test_the_reset_is_found_with_its_polarity():
    assert nothing_lost.reset_condition(queue(), "q4", "clk") == "(rst)"
    assert nothing_lost.reset_condition(
        queue(reset="rst_n", async_low=True), "q4", "clk") == "(!rst_n)"
    low = queue(reset="i_rst_n").replace("if (i_rst_n)", "if (~i_rst_n)")
    assert nothing_lost.reset_condition(low, "q4", "clk") == "(!i_rst_n)"
    # what the design does decides over the name
    low = queue(reset="reset").replace("if (reset)", "if (reset == 1'b0)")
    assert nothing_lost.reset_condition(low, "q4", "clk") == "(!reset)"
    high = queue(reset="rst_n")
    assert nothing_lost.reset_condition(high, "q4", "clk") == "(rst_n)"
    # with no evidence, the name decides
    none = queue(reset="rstn").replace("if (rstn)", "if (1'b0)")
    assert nothing_lost.reset_condition(none, "q4", "clk") == "(!rstn)"
    # a reset only a part reads is read in the part
    wrapped = (CONS.replace("if (rst)", "if (!rst)") + "\n"
               "module top (input wire clk, input wire rst, input wire en,\n"
               "            input wire go, input wire burst_len,\n"
               "            output wire busy);\n"
               "    wire [7:0] sum;\n"
               "    cons u (.clk(clk), .rst(rst), .en(en), .i_start(go), "
               ".i_data(8'd1), .o_busy(busy), .o_sum(sum));\n"
               "endmodule\n")
    assert nothing_lost.reset_condition(wrapped, "top", "clk") == "(!rst)"
    assert not nothing_lost.is_reset_name("burst_len")
    assert nothing_lost.is_reset_name("areset_n")
    assert nothing_lost.reset_condition(
        wrapped.replace("input wire rst,", "").replace(".rst(rst)",
                                                       ".rst(1'b0)"),
        "top", "clk") is None


def with_array(entries):
    return ("module m (input wire clk, input wire en, output wire o);\n"
            + (f"    reg [7:0] buf_q [0:{entries - 1}];\n" if entries else "")
            + "    reg r = 1'b0;\n"
            "    always @(posedge clk) if (en) r <= !r;\n"
            "    assign o = r;\nendmodule\n")


def test_the_search_depth_grows_with_the_new_storage():
    parent = [{"verilog": with_array(0)}]
    depth = nothing_lost.search_depth
    assert depth(with_array(0), "m", parent) == 48
    assert depth(with_array(4), "m", parent) == 48
    assert depth(with_array(16), "m", parent) == 72
    assert depth(with_array(32), "m", parent) == 80
    # an array the parent already had is not new
    assert depth(with_array(16), "m", [{"verilog": with_array(16)}]) == 48
    # a combination counts the arrays of its wrapper
    child = pair().replace("reg g_valid = 1'b0;",
                           "reg [7:0] slots [0:15]; reg g_valid = 1'b0;")
    assert depth(child, "pair", [], wrapper="module pair") == 72
    assert depth(pair(), "pair", [], wrapper="module pair") == 48


def test_the_frozen_enable_rule_follows_renames_and_leaves_resets_out():
    """The rule's tests through the grader are the combining file's; these
    pin what the rule reads."""
    ctrl = """module part (input wire clk, input wire rst, input wire ce,
             input wire cfg_we, input wire [7:0] cfg_d, output reg [7:0] q);
    always @(posedge clk) if (rst) q <= 8'd0; else if (ce && cfg_we) q <= cfg_d;
endmodule
"""

    def wrap(ce, we="cfg_we", extra=""):
        return ctrl + f"""module w (input wire clk, input wire rst, input wire en,
          input wire cfg_we, input wire [7:0] cfg_d, input wire hold,
          output wire [7:0] q);
    wire en_w = en;
{extra}    part u_p (.clk(clk), .rst(rst), .ce({ce}), .cfg_we({we}), .cfg_d(cfg_d), .q(q));
endmodule
"""
    def rule(child):
        return nothing_lost.frozen_enable_error(
            nothing_lost.top_text(child, "w"), child, "w", "clk")
    frozen = rule(wrap("en && !hold"))
    assert frozen.startswith(
        "u_p (part)'s input ce looks like its enable, the input that leaves "
        "all of its state unchanged while it is low. The wrapper drives it "
        "with en && !hold, so while that is low part also ignores cfg_we, "
        "which come straight from the wrapper's inputs")
    assert frozen.endswith("never through ce.")
    # straight, through a wire that only renames it, or a constant one
    for ce in ("en", "en_w", "1'b1"):
        assert rule(wrap(ce)) is None
    # the settings input fed by the wrapper's logic too: wholly the stream's
    assert rule(wrap("en && !hold", we="cfg_we && !hold")) is None
    # an input wired to the wrapper's reset is a reset, not a setting
    assert rule(wrap("en && !hold", we="rst")) is None


# --- the search (sby) ---------------------------------------------------------

@requires_sby
def test_a_queue_that_drops_when_full_loses_items():
    child = queue()
    sent, rec = search(child, stream(), parents=[row(parent_of(child))])
    assert sent is True and rec["verdict"] == "ITEMS_LOST"
    found = rec["nothing_lost"]
    assert found["result"] == "lost" and found["stream"] == 0
    assert isinstance(found["step"], int) and found["depth"] == 48
    assert found["reset"] == "(rst)"
    assert set(found) == set(nothing_lost.RESULT_KEYS)
    err = rec["error"]
    assert err.startswith(
        "the checker's own nothing-is-lost search (written by the checker, "
        "not you) found stream 'requests into the queue' losing an item at "
        f"cycle {found['step']} after the start: more items were handed over "
        "at its inputs (counted in a cycle when `en && in_valid`) than were "
        "delivered (pop) plus still held (used).")
    assert "Never drop work, counted or not" in err
    assert len(err) <= batch.LESSON_MAX
    assert rec["reason"] == ("the checker's nothing-is-lost search found "
                             "stream 'requests into the queue' losing an "
                             f"item at cycle {found['step']}")
    assert promote.route("ITEMS_LOST") == "reformat"


@requires_sby
def test_counting_the_drops_does_not_save_it():
    child = queue(counted=True)
    sent, rec = search(child, stream(), parents=[row(parent_of(child))])
    assert sent and rec["verdict"] == "ITEMS_LOST"
    assert rec["nothing_lost"]["result"] == "lost"


@requires_sby
def test_a_queue_that_makes_the_sender_wait_loses_nothing():
    child = queue(ready="!full", counted=True)
    sent, rec = search(child, stream(ready="in_ready"),
                       parents=[row(parent_of(child))])
    assert sent is False and "verdict" not in rec and "error" not in rec
    found = rec["nothing_lost"]
    assert found["result"] == "nothing_lost" and found["depth"] == 48
    assert found["workdir"].startswith(str(nothing_lost.NOTHING_LOST_ROOT))
    # once per record
    assert nothing_lost.check(rec, child, "q4", "clk", []) is False
    assert rec["nothing_lost"] is found


@requires_sby
def test_two_lanes_into_one_output_keep_the_old_lanes_items():
    sent, rec = search(TWO_LANES.replace("OLD_READY", "1'b1"),
                       lanes_stream(""), parents=[row(ONE_LANE, "two_lane")])
    assert sent and rec["verdict"] == "ITEMS_LOST"
    sent, rec = search(
        TWO_LANES.replace("OLD_READY", "en && slot_free && !b_valid"),
        lanes_stream("in_ready"), parents=[row(ONE_LANE, "two_lane")])
    assert not sent and rec["nothing_lost"]["result"] == "nothing_lost"


@requires_sby
def test_a_combination_whose_sender_cannot_wait_loses_items_and_one_that_waits_does_not():
    child = pair(a_ce=CANNOT_WAIT)
    declared = pair_stream(offer="en && a_valid", ready="")
    assert compose_error(declared, child) is None
    assert nothing_lost.frozen_enable_error(
        nothing_lost.top_text(child, "pair"), child, "pair", "clk") is None
    sent, rec = search_combination(child, declared)
    assert sent and rec["verdict"] == "ITEMS_LOST"
    # the slot pauses the sender on its own path, which here is its enable:
    # nothing else of it comes from outside, so the rule lets it pass
    child = pair(a_ce=WAITS)
    assert compose_error(pair_stream(), child) is None
    assert nothing_lost.frozen_enable_error(
        nothing_lost.top_text(child, "pair"), child, "pair", "clk") is None
    sent, rec = search_combination(child, pair_stream())
    assert not sent and rec["nothing_lost"]["result"] == "nothing_lost"


@requires_sby
def test_a_sender_that_gives_up_an_item_it_was_told_to_hold_loses_it():
    sent, rec = search(PRODUCER.replace("STAGE", HOLDS), STAGE_STREAM)
    assert not sent and rec["nothing_lost"]["result"] == "nothing_lost"
    sent, rec = search(PRODUCER.replace("STAGE", GIVES), STAGE_STREAM)
    assert sent and rec["verdict"] == "ITEMS_LOST"
    found = rec["nothing_lost"]
    assert found["result"] == "gave_up" and found["input"] == 0
    assert rec["error"] == (
        "the checker's own nothing-is-lost search found that in stream "
        f"'stage into the queue' the sender (st_v) gave up an item at cycle "
        f"{found['step']} while the stream told it to wait (its ready "
        "en && !full was false the cycle before), so the item is lost. A "
        "sender that cannot hold its item cannot be made to wait: write its "
        'ready as "" and make the part take every item it offers, or make '
        "the sender hold its item until ready is true.")
    assert rec["reason"].endswith(
        "giving up an item it was told to hold, at cycle "
        f"{found['step']}")


@requires_sby
def test_counts_that_add_up_the_other_way_are_a_declaration_error():
    sent, rec = search(queue(ready="!full"),
                       stream(ready="in_ready", delivered="out_valid"))
    assert sent and rec["verdict"] == "STREAM_INVALID"
    assert rec["nothing_lost"]["result"] == "count_off"
    assert "adding up the other way" in rec["error"]
    assert rec["reason"].startswith("stream 'requests into the queue' is "
                                    "declared so that its counts add up the "
                                    "other way (cycle ")


@requires_sby
def test_an_active_low_asynchronous_reset_is_held_only_in_the_first_cycle():
    sent, rec = search(queue(reset="rst_n", async_low=True), stream())
    assert sent and rec["nothing_lost"]["result"] == "lost"
    assert rec["nothing_lost"]["reset"] == "(!rst_n)"
    # held for ever, nothing would ever be lost; free later, a reset would
    # empty the queue and look like a loss
    sent, rec = search(queue(ready="!full", reset="rst_n", async_low=True),
                       stream(ready="in_ready"))
    assert not sent and rec["nothing_lost"]["result"] == "nothing_lost"


@requires_sby
def test_a_declared_flush_forgives_only_what_it_empties():
    child = queue(ready="!full", clear="clr")
    sent, rec = search(child, stream(ready="in_ready"))
    assert sent and rec["verdict"] == "ITEMS_LOST"
    assert "an input that empties the stream on purpose goes in flush" in \
        rec["error"]
    sent, rec = search(child, stream(ready="in_ready", flush="clr"))
    assert not sent and rec["nothing_lost"]["result"] == "nothing_lost"
    # the clear forgives only what it empties: a queue that also drops
    # when full is still sent back
    sent, rec = search(queue(clear="clr"), stream(flush="clr"))
    assert sent and rec["nothing_lost"]["result"] == "lost"


@requires_sby
def test_a_value_that_does_not_build_is_a_declaration_error():
    sent, rec = search(queue(ready="!full"),
                       stream(ready="in_ready", held="used +"))
    assert sent and rec["verdict"] == "STREAM_INVALID"
    assert rec["nothing_lost"]["result"] == "build_error"
    assert rec["error"].startswith("the checker could not build its count "
                                   "from the declared streams: ")
    assert rec["error"].endswith("Write each value as a plain Verilog "
                                 "expression over the top module's signals.")
    assert rec["reason"] == ("the checker could not build its count from "
                             "the declared streams")


@requires_sby
def test_a_one_entry_stage_counted_by_its_empty_flag_loses_nothing():
    sent, rec = search(ONE_ENTRY, stream("en && in_valid", "in_ready",
                                         "en && out_valid && out_ready",
                                         "~empty"))
    assert not sent and rec["nothing_lost"]["result"] == "nothing_lost"


# --- reading sby's logs (no tools) -------------------------------------------

@pytest.fixture
def fake_sby(monkeypatch, tmp_path):
    """run_sby faked with a saved log; the AIGER map abc's failing output
    is read from names the line `line_of` picks in the copy (the first
    "nothing lost" assertion by default)."""
    calls = []

    def install(log, rc, line_of=None):
        def run(name, sv_path, top, mode, depth, timeout_s, root,
                engine="", options=()):
            text = Path(sv_path).read_text()
            calls.append(dict(name=name, top=top, mode=mode, depth=depth,
                              timeout_s=timeout_s, root=root, engine=engine,
                              options=options, text=text))
            lines = text.splitlines()
            line = (line_of or (lambda ls: next(
                n for n, x in enumerate(ls, 1) if "nothing lost" in x)))(lines)
            workdir = tmp_path / f"job_{len(calls)}"
            (workdir / "model").mkdir(parents=True)
            (workdir / "model" / "design_aiger.ywa").write_text(json.dumps(
                {"asserts": [["\\_witness_", "\\check_assert_"
                              f"{top}_nothing_lost_sv_{line}_5"]]}))
            return SbyOutcome(rc=rc, duration_s=1.26, workdir=workdir,
                              log_text=log)
        monkeypatch.setattr(nothing_lost, "sby_available", lambda: True)
        monkeypatch.setattr(nothing_lost, "run_sby", run)
        return calls
    return install


ABC_FAIL_TIMEOUT = """\
SBY  3:32:35 [job] engine_1: Output {output} of miter "model/design_aiger" was asserted in frame {frame}. Time =     0.06 sec
SBY  3:32:35 [job] engine_1: Status returned by engine: FAIL
SBY  3:34:33 [job] summary: engine_0 (smtbmc yices) did not return a status
SBY  3:34:33 [job] summary: engine_0 did not produce any traces
SBY  3:34:33 [job] summary: engine_1 (abc bmc3) returned FAIL
SBY  3:34:33 [job] summary: engine_1 did not produce any traces
SBY  3:34:33 [job] DONE (TIMEOUT, rc=8)
"""


def test_an_engine_fail_counts_even_when_sby_says_time_out(fake_sby):
    calls = fake_sby(ABC_FAIL_TIMEOUT.format(output=0, frame=39), rc=8)
    child = queue()
    sent, rec = search(child, stream(), parents=[row(parent_of(child))])
    assert sent and rec["verdict"] == "ITEMS_LOST"
    found = rec["nothing_lost"]
    assert (found["result"], found["step"], found["stream"]) == ("lost", 39, 0)
    assert found["seconds"] == 1.3 and found["depth"] == 48
    # the search as it was asked for: two engines, no traces, in its folder
    (call,) = calls
    assert call["mode"] == "bmc" and call["depth"] == 48
    assert call["timeout_s"] == nothing_lost.NOTHING_LOST_TIMEOUT_S == 120
    assert call["engine"] == "smtbmc yices\nabc bmc3"
    assert call["options"] == ("vcd off",)
    assert call["root"] == nothing_lost.NOTHING_LOST_ROOT
    assert "assert (used" not in call["text"]


def test_the_search_runs_in_the_records_own_proof_folder(fake_sby,
                                                         monkeypatch,
                                                         tmp_path):
    calls = fake_sby(ABC_FAIL_TIMEOUT.format(output=0, frame=7), rc=2)
    monkeypatch.setattr(distractor, "PROOF_ROOT", tmp_path / "proofs")
    sent, rec = search(queue(), stream(), extension_id="t9", parent_id="p0")
    assert sent and rec["proof_dir"].startswith("proofs/t9_p0_")
    assert calls[0]["root"] == (tmp_path / "proofs" /
                                Path(rec["proof_dir"]).name / "nothing_lost")


def test_an_smtbmc_failure_is_read_from_its_assertion_lines():
    log = ("SBY  4:02:59 [job] engine_0: ##   0:00:00  Assert failed in q4: "
           "q4_nothing_lost.sv:{a}.5-{a}.66 (_witness_.check_assert_q4_"
           "nothing_lost_sv_{a}_50)\n"
           "SBY  4:02:59 [job] engine_0: ##   0:00:00  Assert failed in q4: "
           "q4_nothing_lost.sv:{b}.5-{b}.66 (_witness_.check_assert_q4_"
           "nothing_lost_sv_{b}_50)\n"
           "SBY  4:02:59 [job] engine_0: Status returned by engine: FAIL\n"
           "SBY  4:02:59 [job] summary:   failed assertion q4._witness_."
           "check_assert_q4_nothing_lost_sv_{a}_50 at q4_nothing_lost.sv:"
           "{a}.5-{a}.66 step 6\n"
           "SBY  4:02:59 [job] DONE (FAIL, rc=2)\n")
    line_map = {53: (0, "lost", None), 54: (0, "count_off", None),
                60: (1, "gave_up", 0)}
    outcome = SbyOutcome(rc=2, duration_s=0.21, workdir=Path("/nowhere"),
                         log_text=log.format(a=60, b=53))
    found = nothing_lost.read_result(outcome, line_map, (40, 61))
    assert (found["result"], found["stream"], found["step"]) == ("lost", 0, 6)
    outcome.log_text = log.format(a=54, b=54)
    found = nothing_lost.read_result(outcome, line_map, (40, 61))
    assert (found["result"], found["stream"]) == ("count_off", 0)
    outcome.log_text = log.format(a=60, b=60)
    found = nothing_lost.read_result(outcome, line_map, (40, 61))
    assert (found["result"], found["stream"], found["input"]) == \
        ("gave_up", 1, 0)


def test_a_failure_with_no_line_is_items_lost_with_the_combined_lesson(
        fake_sby):
    # abc names an output the map does not have
    fake_sby(ABC_FAIL_TIMEOUT.format(output=3, frame=12), rc=8)
    sent, rec = search(queue(), stream())
    assert sent and rec["verdict"] == "ITEMS_LOST"
    assert rec["nothing_lost"]["result"] == "unattributed"
    assert rec["error"] == (
        "the checker's own nothing-is-lost search found the counts of the "
        "declared stream(s) 'requests into the queue' not adding up within "
        "12 cycles (it could not tell which way in time): either an item "
        "handed over is lost (dropped, overwritten or ignored when a part is "
        "full or busy: make the sender wait instead), or the declaration "
        "does not match the design (offer once per item handed over, "
        "delivered once per item that leaves, held every item inside).")
    assert rec["reason"] == ("the checker's nothing-is-lost search found the "
                             "counts of a declared stream not adding up "
                             "within 12 cycles")


def test_a_search_that_runs_out_of_time_changes_nothing(fake_sby):
    fake_sby("SBY 17:06:23 [job] engine_0: ##   0:02:20  Checking assertions "
             "in step 16..\n"
             "SBY 17:06:36 [job] engine_0: ##   0:02:33  Checking assertions "
             "in step 17..\n"
             "SBY 17:06:36 [job] engine_1: 9 + : Var =    7175. Cla = 2.\n"
             "SBY 17:08:53 [job] summary: engine_0 (smtbmc yices) did not "
             "return a status\n"
             "SBY 17:08:53 [job] DONE (TIMEOUT, rc=8)\n", rc=8)
    sent, rec = search(queue(), stream(), verdict="KEPT")
    assert sent is False and rec["verdict"] == "KEPT" and "error" not in rec
    found = rec["nothing_lost"]
    assert found["result"] == "unfinished" and found["checked_to"] == 17
    (note,) = nothing_lost.reviewer_notes(rec)
    assert note.endswith("the search ran out of time at cycle 17 of 48, so "
                         "nothing is known about it")


def test_a_tool_failure_outside_the_block_changes_nothing(fake_sby):
    fake_sby("SBY  4:03:01 [job] base: q4_nothing_lost.sv:3: ERROR: "
             "something about the design\n"
             "SBY  4:03:01 [job] DONE (ERROR, rc=16)\n", rc=16)
    sent, rec = search(queue(), stream())
    assert sent is False and "verdict" not in rec
    found = rec["nothing_lost"]
    assert found["result"] == "error"
    assert "something about the design" in found["why"]


def test_no_sby_means_not_run(monkeypatch):
    monkeypatch.setattr(nothing_lost, "sby_available", lambda: False)
    monkeypatch.setattr(nothing_lost, "run_sby", lambda *a, **k: 1 / 0)
    sent, rec = search(queue(), stream())
    assert sent is False and "verdict" not in rec
    assert rec["nothing_lost"]["result"] == "not_run"
    (note,) = nothing_lost.reviewer_notes(rec)
    assert note.endswith(": the search could not run")
    # no streams, or none declared yet: no search at all
    for streams in ([], None):
        record = {"streams": streams}
        assert nothing_lost.check(record, queue(), "q4", "clk", []) is False
        assert "nothing_lost" not in record


def test_every_lesson_fits_the_retry_question():
    long = "a" * 290 + " && b"
    worst = [{"name": "n" * 500,
              "inputs": [{"offer": long, "ready": long}] * 4,
              "delivered": long, "held": long, "flush": long}] * 3
    for result in ("lost", "gave_up", "count_off", "unattributed",
                   "build_error"):
        found = nothing_lost._result(result, step=79, stream=2, input=3,
                                     depth=80, why="x" * 2000)
        text = nothing_lost.lesson(found, worst)
        assert text and len(text) <= batch.LESSON_MAX
        assert nothing_lost.reason(found, worst)
    for result in ("nothing_lost", "unfinished", "error", "not_run"):
        assert nothing_lost.lesson(nothing_lost._result(result), worst) is None
        assert nothing_lost.reason(nothing_lost._result(result), worst) is None


# --- what the reviewer is told -----------------------------------------------

def test_the_reviewer_notes_say_what_the_search_found():
    rec = {"streams": pair_stream(), "ext_type": "compose",
           "nothing_lost": nothing_lost._result("nothing_lost", depth=48),
           "glue_identifiers": [], "new_inputs": ["b_valid", "b_data"]}
    first, *rest = nothing_lost.reviewer_notes(rec)
    assert first == (
        "the checker's own nothing-is-lost search, stream 'results into the "
        "consumer' (handed over when `a_valid` and `a_ce`; delivered g_take; "
        "held g_valid): no item lost in any input sequence of 48 cycles from "
        "the start")
    assert rest == ["the step adds the input ports b_valid, b_data",
                    "the part between the two designs has no registers"]
    # a combination that assumes the receiver takes every item is named
    rec["streams"] = stream("en && a_valid", "", "en&&a_valid", "1'b0",
                            name="results")
    rec["streams"][0]["flush"] = "clr"
    notes = nothing_lost.reviewer_notes(rec)
    assert "; emptied by clr): no item lost" in notes[0]
    assert notes[1] == (
        "stream 'results' counts each item delivered in the cycle it is "
        "offered and holds none, so it assumes the receiving design takes "
        "every item in every state")
    rec["streams"] = stream("a_valid", "a_ce", "a_valid && a_ce", "0")
    assert "assumes the receiving design" in \
        nothing_lost.reviewer_notes(rec)[1]
    # a patch is not told about receivers, and a sent-back step has no note
    rec["ext_type"] = "structural"
    rec["nothing_lost"] = nothing_lost._result("lost", depth=48, step=3)
    assert nothing_lost.reviewer_notes(rec) == [
        "the step adds the input ports b_valid, b_data"]


# --- the study's own designs (sby) --------------------------------------------

@requires_sby
def test_the_search_finds_the_studys_real_losses_and_passes_the_rebuild(
        monkeypatch):
    beats = dict(delivered="en && g_valid && ready", held="g_valid",
                 name="beats to the port")
    sent, rec = search_combination(fixture("sys_c1_003_kept.sv"),
                                   stream("en && a_out_valid", "", **beats))
    assert sent and rec["verdict"] == "ITEMS_LOST"
    assert (rec["nothing_lost"]["result"], rec["nothing_lost"]["step"]) == \
        ("lost", 8)
    sent, rec = search_combination(fixture("sys_c1_003_skeptic_rebuild.sv"),
                                   stream("a_out_valid", "en && slot_free_w",
                                          **beats))
    assert not sent and rec["nothing_lost"]["result"] == "nothing_lost"
    folds = stream("a_done && !done_q", "", "q_pop", "q_cnt",
                   name="results into the fold")
    sent, rec = search_combination(fixture("sys_c1_007_kept.sv"), folds)
    assert sent and rec["nothing_lost"]["depth"] == 48
    assert (rec["nothing_lost"]["result"], rec["nothing_lost"]["step"]) == \
        ("lost", 39)
    # why the search is at least 48 cycles deep: at 30 the loss is missed
    monkeypatch.setattr(nothing_lost, "NOTHING_LOST_MIN_DEPTH", 30)
    sent, rec = search_combination(fixture("sys_c1_007_kept.sv"), folds)
    assert not sent and rec["nothing_lost"]["depth"] == 30
    assert rec["nothing_lost"]["result"] == "nothing_lost"
