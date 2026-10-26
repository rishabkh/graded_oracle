"""Storage built the way real storage is built, and facts about what it
holds (21 Oct 2026).

Suggestions from Gary (hardware expert, Oct 2026), taken as general rules
and not copied from any benchmark:
  - queues with pointers that wrap around, full and empty worked out from
    the pointers, back-pressure instead of dropped data, an output stage;
  - facts about the stored values themselves: an entry does not change
    while it waits, the read pointer points at the oldest entry;
  - one compact fact for every entry ("forall k in 0..7: ..."), pairs of
    entries included, instead of the same fact written once per entry;
  - facts relating whole words, and values moved from one cycle to the
    next, which provers find hard bit by bit.
Measured first on the 1,277 facts of the three growth trials: 6% are about
stored entries, 5% relate words by arithmetic. Our own trials also showed
the back-pressure gap: the reviewer rejected three credit pools on one
design for dropping input data with no ready signal to the sender.
Rishab: "make gary's suggestion fixes now". The checker is faked."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import extend                                                   # noqa: E402
import gates                                                    # noqa: E402
import promote                                                  # noqa: E402
import sense_check                                              # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict, Tier,  # noqa: E402
                          TripleResult)


def squash(text):
    return " ".join(text.split())


def design(n=40):
    body = ["module m (input wire clk, input wire en, output reg [3:0] a);"] \
        + [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(n - 2)] \
        + ["endmodule"]
    return {"id": "c0_1", "top_module": "m", "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": "\n".join(body), "property": ["a <= 4'd9"],
            "invariants": ["a <= 4'd9"], "metrics": {}}


# --- the questions --------------------------------------------------------------

def test_storage_is_built_like_real_storage():
    text = squash(extend.MOVES["STORAGE"])
    assert "wrap around" in text
    # 26 Oct 2026: "never drops data silently" became "never drops or
    # overwrites data, counted or not", because five kept designs of the 25
    # Oct study dropped data and counted or flagged it (change 1)
    assert "ready output" in text and \
        "never drops or overwrites data, counted or not" in text
    assert "output register stage" in text
    # at least one fact about the stored values themselves
    assert "stored values themselves" in text
    assert "does not change while it waits" in text
    assert "oldest valid entry" in text


def test_a_unit_that_can_refuse_work_tells_the_sender_to_wait():
    text = squash(extend.MOVES["SUBUNIT"])
    assert "tell the sender to wait" in text


def test_the_questions_ask_for_word_and_cycle_to_cycle_facts():
    text = squash(extend.INVARIANT_RULES)
    assert "whole words" in text
    assert "from one cycle to the next" in text
    assert "forall k in 0..7:" in text
    assert "where k != m" in text


# 26 Oct 2026: renamed from test_the_reviewer_asks_about_data_dropped_silently
# and now pins "counted, flagged or raised as an error bit", because change 6
# asks about any lost item, counted or not: a flagged drop was kept 5 times
# in the 25 Oct study
def test_the_reviewer_asks_about_any_lost_item_counted_or_not():
    text = squash(sense_check.PROMPT)
    assert "counted, flagged or raised as an error bit" in text
    assert "drop data silently" not in text


# --- compact facts: one fact for every entry ------------------------------------

def test_a_fact_for_every_entry_is_expanded_once_per_index():
    facts, err = extend.expand_facts(
        ["a <= 4'd9", "forall k in 0..3: !v[k] || (tag[k] != 4'd15)"])
    assert err is None
    assert facts == ["a <= 4'd9"] + [
        f"!v[{k}] || (tag[{k}] != 4'd15)" for k in range(4)]


def test_pairs_of_entries_are_expanded_with_their_condition():
    facts, err = extend.expand_facts(
        ["forall k in 0..2, m in 0..2 where k < m: "
         "!(v[k] && v[m]) || (age[k] != age[m])"])
    assert err is None
    assert facts == ["!(v[0] && v[1]) || (age[0] != age[1])",
                     "!(v[0] && v[2]) || (age[0] != age[2])",
                     "!(v[1] && v[2]) || (age[1] != age[2])"]


def test_only_whole_index_names_are_replaced():
    facts, _ = extend.expand_facts(["forall k in 0..1: kq[k] == k"])
    assert facts == ["kq[0] == 0", "kq[1] == 1"]


def test_index_arithmetic_stays_legal_verilog():
    facts, _ = extend.expand_facts(
        ["forall k in 6..7: !v[k] || (seq[(k+1)%8] == seq[k] + 3'd1)"])
    assert facts[1] == "!v[7] || (seq[(7+1)%8] == seq[7] + 3'd1)"


@pytest.mark.parametrize("bad", [
    "forall k in 0..: v[k]",                       # no upper bound
    "forall k in 3..1: v[k]",                      # empty range
    "forall k in 0..999: v[k]",                    # too many
    "forall k in 0..3 where open(k): v[k]",        # not an index condition
    "forall k in 0..3: ",                          # no fact
])
def test_a_malformed_compact_fact_is_named(bad):
    facts, err = extend.expand_facts([bad])
    assert facts is None and err


CTR = """module ctr (
    input  wire       clk,
    input  wire       en,
    output reg  [3:0] cnt
);
    reg [3:0] lim;
    initial cnt = 4'd0;
    initial lim = 4'd9;
    always @(posedge clk)
        if (en)
            cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1;
    always @(posedge clk) assert (cnt <= 4'd9);
endmodule
"""

ARRAY_PATCH = """\
@@ reg [3:0] lim; @@
+    reg [3:0] hist [0:3];
+    integer j;
+    initial for (j = 0; j < 4; j = j + 1) hist[j] = 4'd0;
@@ cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1; @@
+    always @(posedge clk)
+        if (en) begin
+            for (j = 3; j > 0; j = j - 1) hist[j] <= hist[j - 1];
+            hist[0] <= cnt;
+        end"""


@pytest.fixture
def checker(monkeypatch):
    seen = {}
    monkeypatch.setattr(extend, "state_bits", lambda v, top: len(v))

    def triple(payload, **kw):
        import json
        seen["facts"] = json.loads(payload)["invariants"]
        return TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")
    monkeypatch.setattr(extend, "grade_triple_generated", triple)
    monkeypatch.setattr(extend, "grade",
                        lambda *a, **k: GradeResult(tier=Tier.NOT_INDUCTIVE,
                                                    reason="alone"))
    return seen


def parent_row(compact=None):
    row = {"id": "c0_7", "top_module": "ctr", "clock": "clk",
           "antecedents": [], "sanity_covers": [], "verilog": CTR,
           "property": ["cnt <= 4'd9"], "invariants": ["lim == 4'd9"]}
    if compact:
        row["invariants_compact"] = compact
    return row


def test_a_step_with_compact_facts_is_graded_on_the_expanded_ones(checker):
    compact = ["lim == 4'd9", "forall k in 0..3: hist[k] <= 4'd9"]
    out = {"patch": ARRAY_PATCH, "invariants": compact,
           "dispositions": [{"clause": "lim == 4'd9", "status": "kept",
                             "replaced_by": [], "reason": ""}]}
    rec = extend.grade_step4(parent_row(), "structural", out,
                             {"ext_type": "structural", "move": "STORAGE"})
    assert rec["verdict"] == "NECESSARY"
    expanded = ["lim == 4'd9"] + [f"hist[{k}] <= 4'd9" for k in range(4)]
    assert checker["facts"] == expanded == rec["invariants"]
    assert rec["invariants_compact"] == compact


def test_dispositions_are_checked_against_the_parents_compact_facts(checker):
    compact = ["lim == 4'd9", "forall k in 0..1: (k == 0) || (lim != 4'd0)"]
    parent = parent_row(compact)
    parent["invariants"] = ["lim == 4'd9", "(0 == 0) || (lim != 4'd0)",
                            "(1 == 0) || (lim != 4'd0)"]
    out = {"patch": ARRAY_PATCH, "invariants": compact +
           ["forall k in 0..3: hist[k] <= 4'd9"],
           "dispositions": [{"clause": c, "status": "kept", "replaced_by": [],
                             "reason": ""} for c in compact]}
    rec = extend.grade_step4(parent, "structural", out,
                             {"ext_type": "structural", "move": "STORAGE"})
    assert rec["verdict"] == "NECESSARY", rec.get("error")


def test_a_malformed_compact_fact_is_a_format_failure(checker):
    out = {"patch": ARRAY_PATCH, "invariants": ["forall k in 0..: hist[k]"],
           "dispositions": []}
    rec = extend.grade_step4(parent_row(), "structural", out,
                             {"ext_type": "structural", "move": "STORAGE"})
    assert rec["verdict"] == "FACT_FORMAT"
    assert promote.route("FACT_FORMAT") == "reformat"
    assert "facts" not in checker


def test_the_question_shows_the_parents_facts_compactly():
    row = dict(design(), invariants=["v[0]", "v[1]", "v[2]", "v[3]"],
               invariants_compact=["forall k in 0..3: v[k]"])
    text = extend.build_prompt(row, "structural", "STORAGE")
    assert "forall k in 0..3: v[k]" in text and "\nv[2]\n" not in text


def test_the_fact_limit_counts_compact_facts():
    rec = {"verdict": "NECESSARY", "ext_type": "structural",
           "child_verilog": "module m; endmodule", "patch": "",
           "invariants": [f"v[{k}] == 1'b0" for k in range(80)],
           "invariants_compact": ["forall k in 0..79: v[k] == 1'b0"]}
    assert gates.apply(rec, [design()], use_yosys=False)["verdict"] == \
        "NECESSARY"
    many = dict(rec, invariants_compact=[f"w{k} == 1'b0" for k in range(70)],
                invariants=[f"w{k} == 1'b0" for k in range(70)])
    many.pop("gates", None)
    assert gates.apply(many, [design()], use_yosys=False)["verdict"] == \
        "TOO_MANY_FACTS"


def test_too_many_expanded_facts_are_refused_too():
    facts, err = extend.expand_facts(
        ["forall k in 0..63, m in 0..63: a[k] != b[m]"])
    assert facts is None and "expand" in err


def test_a_promoted_row_keeps_the_compact_facts():
    parent = dict(design(), id="c0_000")
    rec = {"extension_id": "e1", "ext_type": "structural", "move": "STORAGE",
           "parent_id": "c0_000", "verdict": "NECESSARY",
           "child_verilog": parent["verilog"] + "\n// grew",
           "invariants": ["a <= 4'd9"] + [f"q[{k}] + a <= 5'd20"
                                          for k in range(4)],
           "invariants_compact": ["a <= 4'd9",
                                  "forall k in 0..3: q[k] + a <= 5'd20"]}
    _, rows = promote.promote([parent], [rec], compute_metrics=False)
    assert rows[0]["invariants_compact"] == rec["invariants_compact"]
    assert rows[0]["invariants"] == rec["invariants"]


def test_a_debug_step_keeps_the_parents_compact_facts(monkeypatch):
    parent = dict(design(), id="c0_000",
                  invariants=["v[0]", "v[1]"],
                  invariants_compact=["forall k in 0..1: v[k]"])
    seen = {}
    monkeypatch.setattr(batch, "distractor_call",
                        lambda p: (seen.setdefault("prompt", p), None,
                                   {"input": 1, "output": 1}, "refusal")[1:])
    monkeypatch.setattr(batch, "dump", lambda r: None)
    task = {"task_id": 1, "attempt": 1, "ext_type": "distractor",
            "parent_id": "c0_000"}
    rec = batch._real_executor(task, [parent])
    assert rec["invariants_compact"] == ["forall k in 0..1: v[k]"]
    assert "forall k in 0..1: v[k]" in seen["prompt"]
