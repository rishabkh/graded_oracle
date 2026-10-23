"""Steps the checkers can finish (21 Oct 2026).

Three of the changes Rishab approved on 21 Oct ("make all 1-8 changes"),
each from the 21 Oct real-systems trial:
  1. sizes are parameters, proved at small values. Real designs declare
     sizes as parameters and engineers prove them at small sizes (ZipCPU
     proves his FIFOs at small depths). 5 of 10 storage steps ran out of
     the checker's 300 s, and of the 6 steps that added an array of 24 or
     32 entries, 3 ran out of time. The questions ask for parameters with
     4 to 8 entries, and a step that adds an array of more than 16 entries
     is sent back before anything is checked (STORAGE_TOO_DEEP);
  5. answers no longer cut off. 2 storage answers stopped at the
     64,000-token limit after designs of 381 and 397 lines were asked for
     about as many new lines again; once a design is over 300 lines every
     step asks for 200 to 300;
  3. facts the benchmark's checker can read. EBMC 6.0 answers "unknown
     identifier $past" for a fact with $past, and the same for $countones,
     $onehot, $stable, $signed and $clog2; the benchmark's question forbids
     them in facts. A fact with a system function is a format failure
     (FACT_FORMAT); the design's own assertions may still use them.
The model and the checker are faked: no calls, no proofs."""
import json
import random
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
from oracle.types import (GradeResult, NecessityVerdict, Tier,  # noqa: E402
                          TripleResult)


def squash(text):
    return " ".join(text.split())


@pytest.fixture
def checker(monkeypatch):
    """A checker that says NECESSARY and keeps the facts it was given; the
    new assertion alone needs facts. Counts the size reads too, so a test
    can see that nothing was checked."""
    seen = {"graded": 0, "sized": 0}

    def bits(verilog, top):
        seen["sized"] += 1
        return len(verilog)

    def triple(payload, **kw):
        seen["graded"] += 1
        seen["facts"] = json.loads(payload)["invariants"]
        return TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")
    monkeypatch.setattr(extend, "state_bits", bits)
    monkeypatch.setattr(extend, "grade_triple_generated", triple)
    monkeypatch.setattr(extend, "grade",
                        lambda *a, **k: GradeResult(tier=Tier.NOT_INDUCTIVE,
                                                    reason="alone"))
    return seen


CTR = """module ctr (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg  [3:0] cnt
);
    reg [3:0] lim;
    initial cnt = 4'd0;
    initial lim = 4'd9;
    always @(posedge clk) begin
        if (rst)
            cnt <= 4'd0;
        else if (en)
            cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1;
    end
    always @(posedge clk) assert (cnt <= 4'd9);
endmodule
"""


def ctr_row(verilog=CTR):
    return {"id": "c0_7", "top_module": "ctr", "clock": "clk",
            "antecedents": [], "sanity_covers": [], "verilog": verilog,
            "property": ["cnt <= 4'd9"], "invariants": ["lim == 4'd9"]}


KEPT = [{"clause": "lim == 4'd9", "status": "kept", "replaced_by": [],
         "reason": ""}]


def history_patch(dims="[0:3]", last=3):
    """A shift register of past counts, `dims` deep."""
    return f"""\
@@ reg [3:0] lim; @@
+    reg [3:0] hist {dims};
+    integer j;
+    initial for (j = 0; j <= {last}; j = j + 1) hist[j] = 4'd0;
@@ cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1; @@
= end
+    always @(posedge clk)
+        if (en) begin
+            for (j = {last}; j > 0; j = j - 1) hist[j] <= hist[j - 1];
+            hist[0] <= cnt;
+        end"""


def storage_step(patch, facts=("lim == 4'd9",
                                "forall k in 0..3: hist[k] <= 4'd9"),
                 move="STORAGE", parent=None):
    out = {"patch": patch, "invariants": list(facts), "dispositions": KEPT}
    return extend.grade_step4(parent or ctr_row(), "structural", out,
                              {"ext_type": "structural", "move": move})


# --- 1. sizes are parameters, proved at small values ------------------------

def test_the_rules_ask_for_sizes_as_parameters_with_small_values():
    text = squash(extend.SYSTEM_PROMPT)
    assert "#(parameter DEPTH = 4)" in text and "localparam" in text
    assert "4 to 8 entries" in text
    assert f"never more than {extend.MAX_NEW_DEPTH}" in text
    assert f"more than {extend.MAX_NEW_DEPTH} entries is rejected" in text
    # only counts are kept small: data keeps the module's own width
    assert "module's own data width" in text


@pytest.mark.parametrize("move", ["STORAGE", "SUBUNIT", "PEER", "STAGE"])
def test_each_move_that_adds_entries_asks_for_a_parameter(move):
    text = squash(extend.MOVES[move])
    assert "parameter" in text
    assert f"never more than {extend.MAX_NEW_DEPTH}" in text


def test_a_compact_facts_range_is_asked_for_in_numbers():
    """With sizes now parameters, `forall k in 0..DEPTH-1` would come
    naturally, and only number ranges can be expanded."""
    assert "(0..3 when DEPTH = 4), never its name" in squash(
        extend.INVARIANT_RULES)
    facts, err = extend.expand_facts(["forall k in 0..DEPTH-1: v[k]"])
    assert facts is None and "LO..HI" in err


def test_the_storage_question_still_asks_for_real_storage():
    text = squash(extend.MOVES["STORAGE"])
    assert "#(parameter DEPTH = 4)" in text and "4 to 8 entries" in text
    assert "wrap around" in text and "oldest valid entry" in text


@pytest.mark.parametrize("expr, value", [
    ("16", 16), ("1_024", 1024), ("6'd48", 48), ("'h10", 16),
    ("4'b1010", 10), ("8'sd1", 1), ("DEPTH - 1", 3), ("(1 << AW) - 1", 7),
    ("2**AW", 8), ("$clog2(64)", 6), ("$clog2(DEPTH)", 2), ("DEPTH / 2", 2),
])
def test_a_size_is_worked_out_from_numbers_and_parameters(expr, value):
    assert extend.const_value(expr, {"DEPTH": 4, "AW": 3}) == value


@pytest.mark.parametrize("expr", [
    "N - 1",            # a parameter whose value is not known
    "4'bx01",           # an unknown digit
    "f(3)",             # a function
    "DEPTH / 0", "2**200", "a ? 4 : 8", "", "$bits(x)",
])
def test_a_size_that_cannot_be_worked_out_is_none(expr):
    assert extend.const_value(expr, {"DEPTH": 4}) is None


SIZES = """\
module q #(parameter DEPTH = 4, parameter W = 8) (input wire clk);
    localparam integer AW = $clog2(DEPTH);
    reg [W-1:0] mem [0:DEPTH-1];
    reg [AW:0] wp;
    reg [7:0] grid [0:3][0:7];
endmodule
module top (input wire clk, output reg [3:0] a);
    localparam QD = 16;
    localparam [3:0] SLOTS = 4'd8, LINES = SLOTS * 2;
    reg [7:0] t1 [0:31], t2 [15:0];   // two arrays in one line
    reg [7:0] t3 [8];
    reg [7:0] t4 [0:(1 << 3) - 1];
    reg [7:0] t5 [0:SIZE-1];
    reg signed [7:0] t6 [0:LINES-1];
    q #(.DEPTH(QD * 2)) u_big (.clk(clk));
    q u_small (.clk(clk));
endmodule
module unused (input wire clk);
    reg [7:0] huge [0:1023];
endmodule
"""


def test_array_sizes_are_read_as_the_checker_builds_the_design():
    depths = extend.storage_depths(SIZES, "top")
    assert depths[("top", "t1")] == 32 and depths[("top", "t2")] == 16
    assert depths[("top", "t3")] == 8 and depths[("top", "t4")] == 8
    assert depths[("top", "t6")] == 16       # a localparam from another
    assert depths[("q", "grid")] == 32       # every entry of both dimensions
    # one instance asks for 32 entries, the other keeps the default 4
    assert depths[("q", "mem")] == 32
    assert ("top", "t5") not in depths       # SIZE is not known: let through
    assert ("unused", "huge") not in depths  # never built from top
    assert ("q", "wp") not in depths         # a plain register


def test_a_constant_table_is_not_storage():
    text = ("module top (input wire clk);\n"
            "    localparam logic [7:0] SQUARES [0:31] = '{default: 8'd0};\n"
            "    reg [7:0] q [0:3];\nendmodule\n")
    assert extend.storage_depths(text, "top") == {("top", "q"): 4}
    assert extend.declared_arrays(text) == {("top", "q")}


def test_an_instance_may_set_a_size_in_order():
    text = SIZES.replace("q #(.DEPTH(QD * 2)) u_big", "q #(64, 8) u_big")
    assert extend.storage_depths(text, "top")[("q", "mem")] == 64


def test_new_arrays_and_old_ones_made_deeper_are_judged():
    # 22 Oct 2026 (review finding F1): the parent's own biggest array now
    # sets the limit when it is over 16, so this parent's t1 holds 16
    parent = ("module top (input wire clk);\n"
              "    reg [7:0] t1 [0:15];\nendmodule\n")
    child = parent.replace("[0:15];", "[0:127];\n    reg [7:0] t2 [0:16];\n"
                           "    reg [7:0] t3 [0:15];")
    # t1's size in the parent, not the 128 the step gave it, sets the
    # limit; 16 entries are allowed, 17 not. Until 23 Oct 2026 t1, the
    # parent's own array, was never judged even made deeper (review
    # finding): now it is
    assert extend.too_deep(parent, child, "top") == [("top", "t1", 128),
                                                      ("top", "t2", 17)]
    assert extend.too_deep(parent, parent.replace(
        "endmodule", "    reg [7:0] t3 [0:15];\nendmodule"), "top") == []


def test_a_step_with_deep_storage_is_sent_back_before_any_check(checker):
    rec = storage_step(history_patch("[0:31]", 31))
    assert rec["verdict"] == "STORAGE_TOO_DEEP"
    assert rec["too_deep"] == [{"module": "ctr", "array": "hist",
                                "entries": 32}]
    error = squash(rec["error"])
    assert "hist in module ctr (32 entries)" in error
    assert "#(parameter DEPTH = 4)" in error and "4 to 8 entries" in error
    assert f"never more than {extend.MAX_NEW_DEPTH}" in error
    assert checker["graded"] == 0 and checker["sized"] == 0


def test_a_step_with_sixteen_entries_is_checked(checker):
    rec = storage_step(history_patch("[0:15]", 15))
    assert rec["verdict"] == "NECESSARY", rec.get("error")
    assert checker["graded"] == 1


def test_deep_storage_is_a_format_failure_the_run_retries():
    assert promote.route("STORAGE_TOO_DEEP") == "reformat"
    row = dict(ctr_row(), generation=0, metrics={})
    calls = []

    def executor(task, corpus):
        calls.append((task["task_id"], task["attempt"]))
        return {"verdict": "STORAGE_TOO_DEEP", "parent_id": task["parent_id"],
                "extension_id": f"e{len(calls)}"}
    batch.run_loop([row], executor, max_calls=3, rng=random.Random(0),
                   start_ids=[row["id"]])
    # the same task, three attempts: a format failure, not a failed design
    assert calls == [(1, 1), (1, 2), (1, 3)]


SUB_FIFO = """\
+module hist_fifo #(parameter DEPTH = 4) (
+    input  wire       clk,
+    input  wire       push,
+    input  wire [3:0] din,
+    output wire [3:0] oldest
+);
+    reg [3:0] mem [0:DEPTH-1];
+    integer i;
+    initial for (i = 0; i < DEPTH; i = i + 1) mem[i] = 4'd0;
+    always @(posedge clk)
+        if (push) begin
+            for (i = DEPTH - 1; i > 0; i = i - 1) mem[i] <= mem[i - 1];
+            mem[0] <= din;
+        end
+    assign oldest = mem[DEPTH - 1];
+endmodule
@@ reg [3:0] lim; @@
+    wire [3:0] oldest_w;
+    hist_fifo {params}u_hist (.clk(clk), .push(en), .din(cnt),
+                             .oldest(oldest_w));"""


@pytest.mark.parametrize("params, verdict", [
    ("", "NECESSARY"),                         # the default, 4
    ("#(.DEPTH(8)) ", "NECESSARY"),
    ("#(.DEPTH(64)) ", "STORAGE_TOO_DEEP"),    # the size the checker sees
    ("#(32) ", "STORAGE_TOO_DEEP"),
])
def test_a_new_module_is_judged_at_the_size_its_instance_gives(
        checker, params, verdict):
    rec = storage_step(SUB_FIFO.format(params=params),
                       facts=("lim == 4'd9", "oldest_w <= 4'd9"),
                       move="SUBUNIT")
    assert rec["verdict"] == verdict, rec.get("error")


HOT_SRC = """\
module hot_src (
    input wire clk, input wire rst, input wire step,
    output wire [3:0] hot
);
    reg [1:0] sel;
    initial sel = 2'd0;
    always @(posedge clk)
        if (rst) sel <= 2'd0;
        else if (step) sel <= sel + 2'd1;
    assign hot = 4'b0001 << sel;
endmodule
"""

HOT_LATCH = """\
module hot_latch (
    input wire clk, input wire rst, input wire in_valid,
    input wire [3:0] in_data,
    output reg [3:0] held
);
    initial held = 4'b0001;
    always @(posedge clk)
        if (rst) held <= 4'b0001;
        else if (in_valid) held <= in_data;
    always @(posedge clk)
        assert (held == 4'b0001 || held == 4'b0010 ||
                held == 4'b0100 || held == 4'b1000);
endmodule
"""


def wrapped(name, verilog, facts=()):
    from build_corpus import extract_asserts
    return {"id": f"x_{name}", "top_module": name, "clock": "clk",
            "antecedents": [], "sanity_covers": [], "verilog": verilog,
            "property": extract_asserts(verilog), "invariants": list(facts)}


DEEP_GLUE = """\
module hot_link (
    input wire clk, input wire rst, input wire step, input wire in_valid,
    output wire [3:0] held_o
);
    wire [3:0] hotw;
    reg [3:0] buf_q [0:{last}];
    integer i;
    initial for (i = 0; i <= {last}; i = i + 1) buf_q[i] = 4'b0001;
    always @(posedge clk)
        if (step) begin
            for (i = {last}; i > 0; i = i - 1) buf_q[i] <= buf_q[i - 1];
            buf_q[0] <= hotw;
        end
    hot_src a (.clk(clk), .rst(rst), .step(step), .hot(hotw));
    hot_latch b (.clk(clk), .rst(rst), .in_valid(in_valid),
                 .in_data(buf_q[{last}]), .held(held_o));
    always @(posedge clk)
        if (in_valid) assert (held_o != 4'b1111);
endmodule
"""


def test_a_combinations_deep_glue_is_sent_back(checker):
    out = {"wrapper": DEEP_GLUE.format(last=31), "top_module": "hot_link",
           "antecedents": [], "sanity_covers": [],
           "invariants": ["buf_q[0] != 4'b0000"]}
    rec = extend.grade_compose(wrapped("hot_src", HOT_SRC),
                               wrapped("hot_latch", HOT_LATCH), out, {})
    assert rec["verdict"] == "STORAGE_TOO_DEEP"
    assert "buf_q in module hot_link (32 entries)" in rec["error"]
    assert checker["graded"] == 0 and checker["sized"] == 0
    small = dict(out, wrapper=DEEP_GLUE.format(last=7))
    assert extend.grade_compose(wrapped("hot_src", HOT_SRC),
                                wrapped("hot_latch", HOT_LATCH), small,
                                {})["verdict"] != "STORAGE_TOO_DEEP"


POOL = """\
module hot_pool (input wire clk, input wire rst, input wire step,
                 output wire [3:0] h0, output wire [3:0] h1);
    reg [3:0] seen [0:{last}];
    hot_src u0 (.clk(clk), .rst(rst), .step(step), .hot(h0));
    hot_src u1 (.clk(clk), .rst(rst), .step(step), .hot(h1));
    always @(posedge clk) seen[0] <= h0 | h1;
    always @(posedge clk) assert (h0 != 4'b1111);
endmodule
"""


def test_a_copys_deep_pool_is_sent_back(checker):
    out = {"wrapper": POOL.format(last=63), "top_module": "hot_pool",
           "antecedents": [], "sanity_covers": [], "invariants": []}
    rec = extend.grade_replicate(wrapped("hot_src", HOT_SRC), out, {}, 2)
    assert rec["verdict"] == "STORAGE_TOO_DEEP"
    assert checker["graded"] == 0


# --- 5. a big design is asked for 200 to 300 new lines -----------------------

def lines_row(n, id="c0_1", top="m"):
    # n - 1 lines of real content, no two alike (endmodule is not counted)
    body = [f"module {top} (input wire clk, output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(n - 2)] + \
        ["endmodule"]
    return {"id": id, "generation": 0, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": "\n".join(body), "property": ["a <= 4'd9"],
            "invariants": ["a <= 4'd9"], "metrics": {}}


@pytest.mark.parametrize("move", ["SUBUNIT", "STORAGE", "PEER", None])
def test_a_design_over_300_lines_is_asked_for_200_to_300(move):
    assert extend.BIG_STEP_CAP == 300 and extend.BIG_STEP_FEWEST == 200
    assert extend.size_target(lines_row(701)["verilog"], move) == \
        (700, 200, 300)
    assert extend.size_target(lines_row(302)["verilog"], move) == \
        (301, 200, 300)


def test_up_to_300_lines_the_sizes_are_as_before():
    # 22 Oct 2026 (review finding F7): as before only while the usual ask
    # stays within 300 new lines; a 300-line design is no longer asked to
    # double (tests/test_extender_size_cap_and_reset_rules.py)
    v = lines_row(301)["verilog"]                     # 300 lines
    assert extend.size_target(v, "PEER") == (300, 150, 225)
    v = lines_row(241)["verilog"]                     # 240 lines
    assert extend.size_target(v, "SUBUNIT") == (240, 240, 300)


def test_the_question_says_the_capped_size_in_words():
    for move in ("SUBUNIT", "STORAGE", "PEER"):
        text = squash(extend.build_prompt(lines_row(701), "structural", move))
        assert "700 lines" in text and "200 to 300 new lines" in text, move
        assert "at most 300 for a design over 300 lines" in text, move
        assert "doubles" not in text and "three quarters" not in text, move
    small = squash(extend.build_prompt(lines_row(81), "structural", "STORAGE"))
    assert "80 to 100 new lines" in small and "doubles" in small


def test_the_irrelevant_logic_step_follows_the_same_rule():
    text = squash(distractor.build_prompt(lines_row(701)))
    assert "700 lines" in text and "200 to 300 new lines" in text
    assert "three quarters" not in text
    small = squash(distractor.build_prompt(lines_row(81)))
    assert "40 to 60 new lines" in small
    assert "half to three quarters of its current size" in small


# --- 3. facts the benchmark's checker can read ------------------------------

@pytest.mark.parametrize("fact", [
    "cnt == $past(cnt)", "$stable(lim)", "$rose(en) || (cnt == 4'd0)",
    "!$fell(en)", "$changed(cnt) == 1'b0", "$countones(cnt) <= 3'd2",
    "$onehot(cnt)", "$onehot0(cnt)", "!$isunknown(cnt)", "lim == $clog2(16)",
    "$signed(cnt) >= 0",
])
def test_a_fact_with_a_system_function_is_found(fact):
    assert extend.system_function_facts(["lim == 4'd9", fact]) == [fact]


def test_plain_facts_pass():
    facts = ["lim == 4'd9", "hist[0] + hist[1] <= 5'd18",
             "a$b == 1'b0"]                  # a $ inside a name is the name
    assert extend.system_function_facts(facts) == []
    assert extend.system_function_error(facts) is None


def test_a_step_whose_fact_uses_past_is_a_format_failure(checker):
    rec = storage_step(history_patch(),
                       facts=("lim == 4'd9", "hist[0] == $past(cnt)"))
    assert rec["verdict"] == "FACT_FORMAT"
    assert promote.route("FACT_FORMAT") == "reformat"
    error = squash(rec["error"])
    assert "($past)" in error and "hist[0] == $past(cnt)" in error
    # the fix: registers that hold the earlier value
    assert "a pipeline stage and the stage before it" in error
    assert checker["graded"] == 0


def test_a_compact_fact_is_checked_once_expanded(checker):
    compact = ["lim == 4'd9", "forall k in 0..3: $stable(hist[k])"]
    rec = storage_step(history_patch(), facts=compact)
    assert rec["verdict"] == "FACT_FORMAT"
    assert "$stable" in rec["error"] and "(and 1 more)" in rec["error"]
    assert rec["invariants_compact"] == compact
    assert len(rec["invariants"]) == 5
    assert checker["graded"] == 0


WRAPS = """\
@@ reg [3:0] lim; @@
+    reg [2:0] wraps;
+    initial wraps = 3'd0;
@@ cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1; @@
= end
+    always @(posedge clk)
+        if (!rst && en && cnt == lim && wraps != 3'd5)
+            wraps <= wraps + 3'd1;
@@ always @(posedge clk) assert (cnt <= 4'd9); @@
+    always @(posedge clk) assert (wraps >= $past(wraps) || $past(rst));"""


def test_the_designs_own_assertion_may_still_use_past(checker):
    out = {"patch": WRAPS, "dispositions": KEPT,
           "invariants": ["lim == 4'd9", "(wraps == 3'd0) || (lim == 4'd9)"]}
    rec = extend.grade_step4(ctr_row(), "structural", out,
                             {"ext_type": "structural", "move": "SUBUNIT"})
    assert rec["verdict"] == "NECESSARY", rec.get("error")
    assert "$past" in rec["own_property"]


def test_a_combination_whose_fact_uses_a_system_function_is_sent_back(
        checker):
    out = {"wrapper": DEEP_GLUE.format(last=3), "top_module": "hot_link",
           "antecedents": [], "sanity_covers": [],
           "invariants": ["$onehot(buf_q[0])"]}
    rec = extend.grade_compose(wrapped("hot_src", HOT_SRC),
                               wrapped("hot_latch", HOT_LATCH), out, {})
    assert rec["verdict"] == "FACT_FORMAT" and "$onehot" in rec["error"]
    assert checker["graded"] == 0


def test_the_questions_say_facts_are_same_cycle_without_system_functions():
    rules = squash(extend.INVARIANT_RULES)
    assert "same-cycle boolean expression" in rules
    for name in ("$past", "$stable", "$rose", "$fell", "$changed",
                 "$countones", "$onehot", "$onehot0", "$isunknown",
                 "$clog2"):
        assert name in rules, name
    for carrier in ("pipeline stage and the stage before it", "delay line",
                    "stored entry", "captured copy"):
        assert carrier in rules, carrier
    assert "own assertions are not facts and may still use $past" in rules
    for template in (extend.COMPOSE_TEMPLATE, extend.REPLICATE_TEMPLATE):
        text = squash(template)
        assert "same-cycle expression with no system function" in text
    # the questions are built with them
    assert "same-cycle boolean expression" in squash(
        extend.build_prompt(ctr_row(), "structural", "STORAGE"))


def test_a_design_whose_facts_use_a_system_function_is_never_wrapped():
    """A wrapper restates every fact of what it wraps, so such a design
    could never pass a combination now. The name after the $ already reads
    as a signal no port can mirror, so the sampler never offers it."""
    plain = wrapped("hot_src", HOT_SRC, ["hot != 4'd0"])
    counted = wrapped("hot_src", HOT_SRC, ["$countones(hot) == 3'd1"])
    assert extend.wrappable(plain)
    assert not extend.wrappable(counted)
