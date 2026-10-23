"""A step of the extender must not read an asynchronous reset (23 Oct 2026,
review finding F4, the extender's side).

The generator turns away a design whose property, guard or facts read the
reset, because our checker and EBMC, which scores the trained model,
disagree about what an asynchronous reset does within a cycle. The
extender never asked: a STORAGE step that added `assert ((rst_n || a ==
3'd0) && d <= 4'd8)` to a design with an asynchronous reset was NECESSARY
for our checker and REFUTED by EBMC. Every grader (grade_step4,
grade_compose, grade_replicate, the distractor's grade_extension) and the
fixer now ask reset_reads.py about what a step adds, before any check, and
the rules say so (rule 12). Only asynchronous resets count, so the older
designs whose property's guard reads a synchronous reset are untouched.

The model and the checker are faked, except in the last test, which runs
sby and EBMC and skips when they are missing."""
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))
sys.path.insert(0, str(ROOT / "initiator"))

import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
import fix                                                      # noqa: E402
from build_corpus import extract_asserts                        # noqa: E402
from oracle.sby import sby_available                            # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict, Tier,  # noqa: E402
                          TripleResult)


def squash(text):
    return " ".join(text.split())


# The reviewer's parent: an asynchronous reset, a property that leaves it
# alone (it passes the generator's check).
ASYNC = """module arp (input wire clk, input wire rst_n, input wire en, output wire [2:0] a_o);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a <= 3'd0;
            b <= 3'd0;
        end else if (en) begin
            a <= (a == 3'd4) ? 3'd0 : a + 3'd1;
            b <= (a == 3'd4) ? 3'd0 : a + 3'd1;
        end
    end
    assign a_o = a;
    always @(posedge clk) assert (b <= 3'd4);
endmodule
"""
# One of the older designs: a synchronous reset its property's guard reads.
SYNC = """module sp (input wire clk, input wire rst, input wire en, output wire [2:0] a_o);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    reg pv = 1'b0;
    always @(posedge clk) begin
        if (rst) begin
            a <= 3'd0;
            b <= 3'd0;
            pv <= 1'b0;
        end else if (en) begin
            pv <= 1'b1;
            a <= (a == 3'd4) ? 3'd0 : a + 3'd1;
            b <= (a == 3'd4) ? 3'd0 : a + 3'd1;
        end
    end
    assign a_o = a;
    always @(posedge clk) if (pv && !rst) assert (b <= 3'd4);
endmodule
"""
FACTS = ["a == b", "a <= 3'd4"]


def row(verilog, row_id="p0", facts=FACTS):
    top = re.search(r"module\s+(\w+)", verilog).group(1)
    return {"id": row_id, "verilog": verilog, "top_module": top,
            "clock": "clk", "property": extract_asserts(verilog),
            "invariants": list(facts), "antecedents": [],
            "sanity_covers": ["a == 3'd4"], "generation": 0}


# A STORAGE step's new register d; the reviewer's assertion reads the reset.
NEW_REG = """+    reg [3:0] d = 4'd0;
+    always @(posedge clk or negedge rst_n) begin
+        if (!rst_n) d <= 4'd0;
+        else if (en) d <= (a == 3'd4) ? 4'd0 : {a + 3'd1, 1'b0};
+    end"""
READS = "    always @(posedge clk) assert ((rst_n || a == 3'd0) && d <= 4'd8);"
CLEAN = "    always @(posedge clk) assert (d <= 4'd8);"


def step(lines, facts=None, anchor="assign a_o = a;"):
    """A step's answer: `lines` added after the anchor."""
    facts = facts or FACTS + ["d == {a, 1'b0}"]
    added = "\n".join("+" + l if not l.startswith("+") else l
                      for l in lines.splitlines())
    return {"patch": f"@@ {anchor} @@\n{added}", "invariants": facts,
            "dispositions": [{"clause": c, "status": "kept",
                              "replaced_by": [], "reason": ""}
                             for c in FACTS]}


@pytest.fixture
def checker(monkeypatch):
    """A checker that says NECESSARY and counts the checks."""
    seen = {"graded": 0}

    def triple(payload, **kw):
        seen["graded"] += 1
        return TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")
    for mod in (extend, distractor):
        monkeypatch.setattr(mod, "state_bits", lambda v, top: len(v))
        monkeypatch.setattr(mod, "grade_triple_generated", triple)
    monkeypatch.setattr(extend, "grade",
                        lambda *a, **k: GradeResult(tier=Tier.NOT_INDUCTIVE,
                                                    reason="alone"))
    return seen


def storage(parent, out):
    record = {"extension_id": "t_reset", "parent_id": parent["id"],
              "ext_type": "structural", "move": "STORAGE"}
    return extend.grade_step4(parent, "structural", out, record)


# --- grade_step4 -------------------------------------------------------------

def test_the_reviewers_step_is_turned_away_before_any_check(checker):
    rec = storage(row(ASYNC), step(NEW_REG + "\n" + READS))
    assert rec["verdict"] == "READS_RESET" and checker["graded"] == 0
    assert rec["error"].startswith(
        "the asynchronous reset rst_n is read by the assertion `assert "
        "((rst_n || a == 3'd0) && d <= 4'd8);` - a new assertion, its guard "
        "and the facts describe the design in normal operation")
    assert "guard an assertion that must skip the cycles of a reset with a " \
        "register the reset clears" in squash(rec["error"])
    assert rec["child_verilog"]          # kept, as for any format failure


def test_the_same_step_leaving_the_reset_alone_reaches_the_checker(checker):
    rec = storage(row(ASYNC), step(NEW_REG + "\n" + CLEAN))
    assert rec["verdict"] == "NECESSARY" and checker["graded"] == 1


def test_a_fact_that_reads_the_reset_is_turned_away(checker):
    rec = storage(row(ASYNC), step(NEW_REG + "\n" + CLEAN, FACTS + [
        "d == {a, 1'b0}", "rst_n || d == 4'd0"]))
    assert rec["verdict"] == "READS_RESET" and checker["graded"] == 0
    assert "is read by fact `rst_n || d == 4'd0`" in rec["error"]


def test_a_wire_that_copies_the_reset_is_followed(checker):
    rec = storage(row(ASYNC), step(
        NEW_REG + "\n    wire held = !rst_n;\n"
        "    always @(posedge clk) assert ((!held || a == 3'd0) "
        "&& d <= 4'd8);"))
    assert rec["verdict"] == "READS_RESET"
    assert "(through held, which carries the reset)" in rec["error"]


def test_a_guard_put_on_the_existing_assertion_is_judged(checker):
    """The assertion's text is unchanged, so the property check passes;
    its guard is new."""
    out = step("+    reg [2:0] c = 3'd0;", FACTS)
    out["patch"] = ("- always @(posedge clk) assert (b <= 3'd4);\n"
                    "+    always @(posedge clk) if (!rst_n) "
                    "assert (b <= 3'd4);")
    rec = extend.grade_step4(row(ASYNC), "structural", out,
                             {"extension_id": "t_guard", "move": "STAGE"})
    assert "if (!rst_n) assert (b <= 3'd4);" in rec["child_verilog"]
    assert rec["verdict"] == "READS_RESET" and checker["graded"] == 0
    assert "the assertion's guard `(!rst_n)`" in rec["error"]


def test_what_the_parent_already_has_is_not_judged_again(checker):
    """A parent whose assertion sits in the block the reset triggers (the
    generator turns such a design away; older rows could have one): a
    step that keeps it as it is, beside a clean new part, is not blamed
    for it."""
    inside = ASYNC.replace(
        "        end\n    end\n    assign a_o = a;\n"
        "    always @(posedge clk) assert (b <= 3'd4);",
        "        end\n        assert (b <= 3'd4);\n    end\n"
        "    assign a_o = a;")
    assert "        assert (b <= 3'd4);\n    end" in inside
    rec = storage(row(inside), step(NEW_REG + "\n" + CLEAN))
    assert rec["verdict"] == "NECESSARY"


def test_a_synchronous_reset_in_a_guard_is_left_alone(checker):
    """One of the 30 older designs: its own property's guard reads the
    synchronous reset, and a new part written the same way is not turned
    away (both checkers read a synchronous reset alike)."""
    new = ("    reg [3:0] d = 4'd0;\n"
           "    always @(posedge clk)\n"
           "        if (rst) d <= 4'd0;\n"
           "        else if (en) d <= (a == 3'd4) ? 4'd0 : {a + 3'd1, 1'b0};\n"
           "    always @(posedge clk) if (pv && !rst) assert (d <= 4'd8);")
    rec = storage(row(SYNC), step(new))
    assert rec["verdict"] == "NECESSARY" and checker["graded"] == 1


def test_rule_12_says_so():
    text = squash(extend.SYSTEM_PROMPT)
    assert ("12. Never read the reset in a new assertion, its guard or a "
            "fact, directly or through a wire that copies it") in text
    assert "rejected before it is checked" in text


# --- grade_compose and grade_replicate -----------------------------------------

# a part whose own guard reads a synchronous input that happens to share
# the asynchronous part's reset name: harmless, and kept verbatim
SYNC_PART = """module tick (input wire clk, input wire rst_n, input wire go, output reg [2:0] t);
    initial t = 3'd0;
    always @(posedge clk) if (!rst_n) t <= 3'd0; else if (go) t <= (t == 3'd5) ? 3'd0 : t + 3'd1;
    always @(posedge clk) if (rst_n) assert (t <= 3'd5);
endmodule
"""


def wrapper(assertion, name="pair"):
    return f"""module {name} (input wire clk, input wire rst_n, input wire en, output wire [2:0] a_o);
    wire [2:0] t;
    reg [2:0] seen = 3'd0;
    arp u_a (.clk(clk), .rst_n(rst_n), .en(en), .a_o(a_o));
    tick u_t (.clk(clk), .rst_n(rst_n), .go(en), .t(t));
    always @(posedge clk) if (en) seen <= t;
{assertion}
endmodule
"""


def compose(assertion, facts=("seen <= 3'd5",)):
    pa, pb = row(ASYNC, "p_a", []), row(SYNC_PART, "p_b", [])
    out = {"wrapper": wrapper(assertion), "top_module": "pair",
           "antecedents": [], "sanity_covers": [], "invariants": list(facts)}
    return extend.grade_compose(pa, pb, out, {"extension_id": "t_pair"})


def test_a_combination_whose_assertion_reads_the_reset_is_turned_away(
        checker):
    rec = compose("    always @(posedge clk) assert (rst_n || seen == 3'd0);")
    assert rec["verdict"] == "READS_RESET" and checker["graded"] == 0
    assert "the assertion `assert (rst_n || seen == 3'd0);`" in rec["error"]


def test_a_combination_that_leaves_the_reset_alone_is_checked(checker):
    """The parts' own assertions are not judged again, even the one whose
    guard reads `rst_n`, a reset only in the other part."""
    rec = compose("    always @(posedge clk) assert (seen != 3'd7);")
    assert rec["verdict"] == "NECESSARY" and checker["graded"] == 1


def test_a_combination_fact_that_reads_the_reset_is_turned_away(checker):
    rec = compose("    always @(posedge clk) assert (seen != 3'd7);",
                  facts=["seen <= 3'd5", "rst_n || seen == 3'd0"])
    assert rec["verdict"] == "READS_RESET"
    assert "fact `rst_n || seen == 3'd0`" in rec["error"]


def pool(assertion):
    parent = row(ASYNC, "p_a", [])
    wrap = f"""module pool (input wire clk, input wire rst_n, input wire en, output wire [2:0] a0);
    wire [2:0] a1;
    reg [3:0] both = 4'd0;
    arp u0 (.clk(clk), .rst_n(rst_n), .en(en), .a_o(a0));
    arp u1 (.clk(clk), .rst_n(rst_n), .en(en), .a_o(a1));
    always @(posedge clk) both <= a0 + a1;
{assertion}
endmodule
"""
    out = {"wrapper": wrap, "top_module": "pool", "antecedents": [],
           "sanity_covers": [], "invariants": ["both <= a0 + a1"]}
    return extend.grade_replicate(parent, out, {"extension_id": "t_pool"}, 2)


def test_a_pool_whose_assertion_guard_reads_the_reset_is_turned_away(
        checker):
    rec = pool("    always @(posedge clk) if (rst_n) assert (both <= 4'd8);")
    assert rec["verdict"] == "READS_RESET"
    assert "the assertion's guard `(rst_n)`" in rec["error"]
    assert pool("    always @(posedge clk) assert (both <= 4'd8);")[
        "verdict"] != "READS_RESET"


# --- the distractor ------------------------------------------------------------

def test_a_distractor_may_not_guard_the_property_with_the_reset(checker):
    patch = ("- always @(posedge clk) assert (b <= 3'd4);\n"
             "+    always @(posedge clk) if (!rst_n) assert (b <= 3'd4);")
    rec = distractor.grade_extension(row(ASYNC), patch, {})
    assert rec["verdict"] == "READS_RESET" and checker["graded"] == 0
    assert "the assertion's guard `(!rst_n)`" in rec["error"]
    plain = distractor.grade_extension(
        row(ASYNC), "@@ assign a_o = a; @@\n+    reg [3:0] spare = 4'd0;", {})
    assert plain["verdict"] == "NECESSARY"


# --- the fixer ---------------------------------------------------------------------

def test_a_repaired_fact_that_reads_the_reset_is_turned_away(monkeypatch,
                                                             tmp_path):
    monkeypatch.setattr(fix, "ATTEMPT_LOG", tmp_path / "fixer_attempts.jsonl")
    checks = []
    monkeypatch.setattr(fix, "grade_triple_generated",
                        lambda payload, **kw: checks.append(payload))
    record = {"extension_id": "t_fix", "ext_type": "structural",
              "parent_id": "p0", "verdict": "NOT_PROVEN",
              "child_verilog": ASYNC, "invariants": ["a <= 3'd4"],
              "result": {"with_invariants": {"runs": [
                  {"mode": "prove", "rc": 4, "notes": [],
                   "trace_text": "a = 3'd2, b = 3'd3"}]}}}
    won = fix.run_task(record, [row(ASYNC)], max_attempts=1,
                       call=lambda prompt, n: {
                           "diagnosis": "d",
                           "invariants": ["a == b", "rst_n || a == 3'd0"]})
    assert won is None and checks == []
    [line] = [json.loads(l) for l in
              (tmp_path / "fixer_attempts.jsonl").read_text().splitlines()]
    assert line["verdict"] == "READS_RESET"
    assert "fact `rst_n || a == 3'd0`" in line["error"]
    # the design's own facts and assertion are the fixer's to keep: a
    # plain list goes on to be checked
    assert fix.reset_error(["a == b", "a <= 3'd4"],
                           fix.fixer_context(record, [row(ASYNC)])) is None


def test_the_fixer_question_says_so():
    assert "No clause reads the reset, directly or through a wire that " \
        "copies it" in squash(fix.FIXER_PROMPT)


# --- why: the two checkers on the reviewer's step (sby and EBMC) -----------------

EBMC_BIN = os.environ.get("EBMC_PATH")


@pytest.mark.skipif(not sby_available(),
                    reason="sby not on PATH - activate hwtools first")
@pytest.mark.skipif(not (EBMC_BIN and Path(EBMC_BIN).is_file()),
                    reason="needs EBMC_PATH pointing at the ebmc binary")
def test_the_reviewers_step_splits_the_checkers_and_is_never_checked(
        tmp_path, monkeypatch):
    from tests.test_generator_reset_and_fact_gates import ebmc, ours
    from patch import apply_patch
    out = step(NEW_REG + "\n" + READS)
    child = apply_patch(ASYNC, out["patch"])
    grown = {"verilog": child, "top_module": "arp", "clock": "clk",
             "antecedents": [], "sanity_covers": ["a == 3'd4"],
             "invariants": out["invariants"]}
    assert ours(grown, tmp_path / "ours") == "NECESSARY"
    # the new assertion alone, in the benchmark's form
    alone = dict(grown, verilog=child.replace(
        "    always @(posedge clk) assert (b <= 3'd4);\n", ""))
    assert ebmc(alone, [], "timing_without_lemmas", tmp_path)[0] == "CEX"
    assert ebmc(alone, out["invariants"], "one_inductive_with_prop",
                tmp_path)[0] == "CEX"
    # the extender turns it away before our checker ever sees it
    monkeypatch.setattr(distractor, "PROOF_ROOT", tmp_path / "proofs")
    rec = storage(row(ASYNC), out)
    assert rec["verdict"] == "READS_RESET"
    assert not (tmp_path / "proofs").exists()


def test_the_batch_run_retries_it_with_the_reason(checker):
    """A format failure: the run asks again, and the next question says
    why (batch.retry_lesson), like any other step sent back."""
    import batch
    import promote
    rec = storage(row(ASYNC), step(NEW_REG + "\n" + READS))
    assert promote.route(rec["verdict"]) == "reformat"
    lesson = batch.retry_lesson(dict(rec, usage={"input": 1, "output": 1}))
    assert lesson.startswith("the asynchronous reset rst_n is read by the "
                             "assertion")
    assert batch.retry_section(lesson).startswith(
        "\n\nYour previous answer to this question was sent back: the "
        "asynchronous reset rst_n")
