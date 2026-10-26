"""Making the good designs bigger (18 Oct 2026).

Hand review of the 18 Oct trial: the good designs were small (typically
135 lines) and the weak ones big (typically 219). From the 55 designs of
both trials, read by hand:
  - every good step grew its design 1.4 to 1.7 times, which is what the
    question asked for (+50-75%);
  - good steps ran out after about two: 11 of 20 first steps were good,
    7 of 16 second, 2 of 10 third, 1 of 9 later, and a step built on a
    weak design was good 3 times in 18;
  - every structural step had to change what the first property reads,
    so on a long chain the new parts piled up on one path (five buffers in
    a row in front of a fixed table, all weak);
  - small tweaks were almost never good (splitting a counter 0 of 5, a
    second running count 0 of 4).
Rishab's six changes ("do all 6 changes"):
  1. sub-unit and storage steps ask to double the design, and a sub-unit
     is written as its own module;
  2. a big step may grow the design sideways: a new part beside the old
     logic, with one assertion of its own;
  3. no more than two steps of one kind per family, and the question lists
     what the family already has;
  4. the second reviewer reads every step, and the run grows only what it
     keeps;
  5. no small tweaks (splitting, the second running count, a register
     limit, one-stage pipelines);
  6. a run can start from the biggest starting designs.
The model and the checker are faked here: no calls, no proofs."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import extend                                                   # noqa: E402
import promote                                                  # noqa: E402
import sense_check as S                                         # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict, Tier,  # noqa: E402
                          TripleResult)

CATALOG = json.loads((ROOT / "initiator" / "catalog" /
                      "kind_categories_2026-10-09.json").read_text())
_GROUP = {a["id"]: a["group"] for a in CATALOG["areas"]}


def kind_in(group):
    return next(k["name"] for k in CATALOG["kinds"]
                if _GROUP.get(k["category"]) == group)


def lines_row(n, id="c0_1", top="m"):
    # lines of real content, no two alike: sizes are counted with comments
    # left out and copies counted once since 19 Oct 2026
    body = [f"module {top} (input wire clk, output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(n - 2)] + \
        ["endmodule"]
    return {"id": id, "generation": 0, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": "\n".join(body), "property": ["a <= 4'd9"],
            "invariants": ["a <= 4'd9"], "metrics": {}}


# --- 1. a good step doubles the design ---------------------------------------

def test_sub_unit_and_storage_steps_ask_to_double_the_design():
    # 81 lines: 80 of content (endmodule is not counted)
    for move in ("SUBUNIT", "STORAGE"):
        text = extend.build_prompt(lines_row(81), "structural", move)
        assert "80 lines" in text and "80 to 100 new lines" in text, move
    peer = extend.build_prompt(lines_row(81), "structural", "PEER")
    assert "40 to 60 new lines" in peer


def test_a_sub_unit_is_written_as_its_own_module():
    text = " ".join(extend.MOVES["SUBUNIT"].split())
    assert "its own module" in text
    # its state reaches the facts through ports, never instance.signal
    assert "output port" in text and "instance.signal" in text


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


def ctr_row():
    return {"id": "c0_7", "top_module": "ctr", "clock": "clk",
            "antecedents": [], "sanity_covers": [], "verilog": CTR,
            "property": ["cnt <= 4'd9"], "invariants": ["lim == 4'd9"]}


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
+    always @(posedge clk) assert ({assertion});"""


def answer(assertion="wraps <= 3'd5",
           facts=("lim == 4'd9", "(wraps == 3'd0) || (lim == 4'd9)")):
    return {"patch": WRAPS.format(assertion=assertion),
            "invariants": list(facts),
            "dispositions": [{"clause": "lim == 4'd9", "status": "kept",
                              "replaced_by": [], "reason": ""}]}


@pytest.fixture
def checker(monkeypatch):
    """A checker that says NECESSARY, and that the new assertion alone
    needs facts (or proves alone, when `alone` is set to PROVEN)."""
    seen = {"alone": Tier.NOT_INDUCTIVE, "graded": 0}
    monkeypatch.setattr(extend, "state_bits", lambda v, top: len(v))

    def grade(path, prop, **kw):
        return GradeResult(tier=seen["alone"], reason="alone")

    def triple(payload, **kw):
        seen["graded"] += 1
        return TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")
    monkeypatch.setattr(extend, "grade", grade)
    monkeypatch.setattr(extend, "grade_triple_generated", triple)
    return seen


def step(move, out):
    record = {"ext_type": "structural", "move": move}
    return extend.grade_step4(ctr_row(), "structural", out, record)


def test_a_fact_that_reaches_inside_an_instance_is_turned_away(checker):
    out = answer(facts=("lim == 4'd9", "u_sub.cnt <= 4'd3"))
    rec = step("SUBUNIT", out)
    assert rec["verdict"] == "HIERARCHICAL_REF"
    assert checker["graded"] == 0


# --- 2. a big step may grow the design sideways -------------------------------

def test_big_steps_may_add_a_part_with_an_assertion_of_its_own():
    for move in extend.BIG_MOVES:
        text = " ".join(extend.build_prompt(lines_row(80), "structural",
                                            move).split())
        # 26 Oct 2026: a PEER step reads PEER_EFFECT_RULE, not the sideways
        # rule, because its new lane affects the old logic only through
        # the one thing the lanes share (change 5); it may still bring ONE
        # new assertion of its own
        if move == "PEER":
            assert "through the one thing the lanes share" in text
        else:
            assert "beside the existing logic" in text, move
        assert "ONE new assertion" in text, move
    stage = " ".join(extend.build_prompt(lines_row(80), "structural",
                                         "STAGE", k=4).split())
    assert "beside the existing logic" not in stage


def test_storage_no_longer_has_to_sit_on_the_property_path():
    assert "on the path the property depends on" not in \
        extend.MOVES["STORAGE"]


def test_a_part_beside_the_old_logic_with_its_own_assertion_is_graded(
        checker):
    rec = step("SUBUNIT", answer())
    assert rec["verdict"] == "NECESSARY"
    assert rec["own_property"] == "wraps <= 3'd5"
    assert rec["p2_new_ids"] == ["wraps"]
    assert checker["graded"] == 1


SUB_MODULE = """\
+module lap_meter (
+    input  wire       clk,
+    input  wire       lap,
+    output reg  [2:0] laps
+);
+    initial laps = 3'd0;
+    always @(posedge clk)
+        if (lap && laps != 3'd5)
+            laps <= laps + 3'd1;
+endmodule
@@ output reg  [3:0] cnt @@
= );
+    wire [2:0] laps_w;
+    lap_meter u_laps (.clk(clk), .lap(en && cnt == lim), .laps(laps_w));
@@ always @(posedge clk) assert (cnt <= 4'd9); @@
+    always @(posedge clk) assert (laps_w != 3'd7);"""


def test_a_sub_modules_state_reaches_its_assertion_through_a_wire(checker):
    """Found grading a hand-made step with the real checker: the sub-unit's
    register lives in its own module, so the assertion in the top module
    reads a wire, and the rule that counted only registers turned it
    away."""
    out = {"patch": SUB_MODULE,
           "invariants": ["lim == 4'd9", "laps_w <= 3'd5",
                          "(laps_w == 3'd0) || (lim == 4'd9)"],
           "dispositions": answer()["dispositions"]}
    rec = step("SUBUNIT", out)
    assert rec["verdict"] == "NECESSARY"
    assert rec["p2_new_ids"] == ["laps_w"]
    # a wire made from old state alone is still not new state
    out["patch"] = SUB_MODULE.replace("lap_meter u_laps", "// ").replace(
        "wire [2:0] laps_w;", "wire [2:0] laps_w = cnt[2:0];")
    assert step("SUBUNIT", out)["verdict"] == "P2_SAME_SUPPORT"


def test_a_small_step_still_may_not_add_an_assertion(checker):
    assert step("STAGE", answer())["verdict"] == "PROPERTY_CHANGED"


def test_the_new_assertion_must_read_new_state(checker):
    rec = step("STORAGE", answer(assertion="cnt != 4'd15"))
    assert rec["verdict"] == "P2_SAME_SUPPORT"
    assert checker["graded"] == 0


def test_the_new_part_must_share_a_fact_with_the_old_one(checker):
    rec = step("PEER", answer(facts=("lim == 4'd9", "wraps != 3'd7")))
    assert rec["verdict"] == "NO_COUPLING"
    assert checker["graded"] == 0


def test_an_assertion_that_proves_without_facts_teaches_nothing(checker):
    checker["alone"] = Tier.PROVEN
    rec = step("SUBUNIT", answer())
    assert rec["verdict"] == "P2_PROVES_UNAIDED"
    assert checker["graded"] == 0


def test_only_one_new_assertion_per_step(checker):
    out = answer()
    out["patch"] += "\n+    always @(posedge clk) assert (wraps != 3'd7);"
    assert step("SUBUNIT", out)["verdict"] == "PROPERTY_CHANGED"


def test_the_new_assertion_may_sit_anywhere_in_the_file(checker):
    out = answer()
    out["patch"] = ("+module helper (input wire clk);\n"
                    "+endmodule\n" + out["patch"])
    out["patch"] = out["patch"].replace(
        "@@ always @(posedge clk) assert (cnt <= 4'd9); @@\n"
        "+    always @(posedge clk) assert (wraps <= 3'd5);", "")
    out["patch"] = out["patch"].replace(
        "@@ reg [3:0] lim; @@",
        "@@ reg [3:0] lim; @@\n+    always @(posedge clk) "
        "assert (wraps <= 3'd5);")
    rec = step("SUBUNIT", out)
    assert rec["verdict"] == "NECESSARY"
    assert rec["own_property"] == "wraps <= 3'd5"


# --- 3. no repeats, and the question says what the family has ---------------

def chain(*kinds):
    """A starting design and one child per step, oldest first."""
    rows = [lines_row(60)]
    for i, kind in enumerate(kinds, 1):
        structural = kind in extend.MOVES
        rows.append(dict(lines_row(60, id=f"c{i}_000"), generation=i,
                         parent=rows[-1]["id"],
                         ext_type="structural" if structural else kind,
                         move=kind if structural else None,
                         added=f"step {i}"))
    return rows


def kinds_drawn(rows, n=3000, seed=3):
    rng = random.Random(seed)
    out = set()
    for _ in range(n):
        t = batch.sample_task(rng, rows[-1], rows)
        out.add(t["move"] if t["ext_type"] == "structural" else
                t["ext_type"])
    return out


def test_a_family_gets_no_more_than_two_steps_of_one_kind():
    """Sub-units and storage are not limited since 19 Oct
    (test_extender_real_systems); the other kinds still are."""
    assert batch.MAX_SAME_KIND == 2
    drawn = kinds_drawn(chain("PEER", "SUBUNIT", "PEER"))
    assert "PEER" not in drawn
    assert "SUBUNIT" in drawn


def test_the_limit_counts_every_kind_of_step():
    drawn = kinds_drawn(chain("distractor", "second", "second",
                              "distractor"))
    assert not drawn & {"distractor", "second"}
    assert "SUBUNIT" in drawn


def test_a_family_that_has_used_every_kind_is_finished(monkeypatch):
    # sub-units and storage have no limit since 19 Oct: with them drawn,
    # a family is never finished, so take them out to test the rest
    for unlimited in batch.NO_LIMIT_KINDS:
        monkeypatch.setitem(extend.MOVE_WEIGHTS, unlimited, 0)
    used = [k for k, w in extend.MOVE_WEIGHTS.items() if w] + \
        ["distractor", "second", "replicate", "compose"]
    rows = chain(*[k for k in used for _ in range(2)])
    assert batch.sample_task(random.Random(1), rows[-1], rows) is None
    calls = []
    batch.run_loop(rows, lambda t, c: calls.append(t) or
                   {"verdict": "PATCH_ERROR"}, max_calls=5, max_gen=99,
                   rng=random.Random(0), start_ids=[rows[-1]["id"]])
    assert calls == []                           # no call spent on it


def test_the_family_history_lists_each_step_oldest_first():
    rows = chain("STORAGE", "distractor", "SUBUNIT")
    by_id = {r["id"]: r for r in rows}
    assert promote.history(rows[-1], by_id) == [
        "STORAGE: step 1", "distractor: step 2", "SUBUNIT: step 3"]
    assert promote.history(rows[0], by_id) == []


def test_a_combined_design_has_the_history_of_both_its_parts():
    a, b = chain("STORAGE"), chain("PEER")
    b = [dict(r, id=r["id"].replace("c", "d")) for r in b]
    b[1]["parent"] = b[0]["id"]
    both = dict(lines_row(60, id="c2_000"), generation=2, parent=a[-1]["id"],
                parent2=b[-1]["id"], ext_type="compose", move=None,
                added="glue")
    by_id = {r["id"]: r for r in a + b + [both]}
    assert promote.history(both, by_id) == [
        "STORAGE: step 1", "PEER: step 1", "compose: glue"]


def test_the_question_lists_what_the_family_already_has():
    hist = ["STORAGE: a prefetch queue", "SUBUNIT: a retry controller"]
    for ext_type, move in (("structural", "PEER"), ("second", None)):
        text = " ".join(extend.build_prompt(lines_row(60), ext_type, move,
                                            history=hist).split())
        assert "1. STORAGE: a prefetch queue" in text, ext_type
        assert "something it does not have yet" in text, ext_type
    plain = extend.build_prompt(lines_row(60), "structural", "PEER")
    assert "Earlier steps" not in plain


def test_a_promoted_row_records_what_its_step_added():
    parent = dict(lines_row(40, id="c0_000"))
    rec = {"extension_id": "e1", "ext_type": "structural", "move": "STORAGE",
           "parent_id": "c0_000", "verdict": "NECESSARY",
           "child_verilog": parent["verilog"] + "\n// grew",
           "invariants": ["a <= 4'd9", "q_cnt + a == 4'd9"],
           "new_state": "A 4-entry queue in front of the input. " + "x" * 400}
    _, rows = promote.promote([parent], [rec], compute_metrics=False)
    assert rows[0]["added"] == "A 4-entry queue in front of the input."
    short = dict(rec, extension_id="e2", child_verilog=rec["child_verilog"]
                 + "\n", ext_type="second", claim="no response without a "
                 "request")
    _, rows = promote.promote([parent], [short], compute_metrics=False)
    assert rows[0]["added"] == "no response without a request"


def test_the_run_puts_the_family_history_in_the_question(monkeypatch):
    rows = chain("STORAGE")
    seen = {}

    def model(prompt, schema):
        seen["prompt"] = prompt
        return None, {"input": 1, "output": 1}, "refusal"
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    task = {"task_id": 1, "attempt": 1, "ext_type": "structural",
            "move": "SUBUNIT", "parent_id": rows[-1]["id"], "k": 1}
    batch._real_executor(task, rows)
    assert "1. STORAGE: step 1" in seen["prompt"]


# --- 4. the second reviewer reads every step -----------------------------------

@pytest.fixture(autouse=True)
def no_checker_notes(monkeypatch):
    # 26 Oct 2026: the reviewer now reads the checker's notes
    # (nothing_lost.reviewer_notes) because change 6 shows it what the
    # checker's own tools found; these records carry none
    monkeypatch.setattr(S, "checker_notes", lambda record: [])


def reply(**kw):
    # 26 Oct 2026: three questions added (value_cut_or_padded,
    # only_a_delayed_copy, item_can_be_dropped) because change 6 asks about
    # cut or padded values, delayed copies and lost items; a reply without
    # them is now a failed try
    a = {"parts_connected_sensibly": True,
         "value_used_against_its_meaning": False,
         "value_cut_or_padded": False,
         "new_logic_only_copies_existing_state": False,
         "only_a_delayed_copy": False,
         "results_of_parts_used": True,
         "new_part_has_its_own_job": True,
         "item_can_be_dropped": False,
         "breaks_what_it_did": False,
         "engineer_would_build_this": True,
         "why": "a real feature"}
    a.update(kw)
    return a


CLAIMS_OPENING = "You reviewed a hardware design"


class Model:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        # 26 Oct 2026: the second call (the builder's claims) is answered
        # without taking a reply because change 6 added that call; the
        # replies stay the design call's answers
        if prompt.startswith(CLAIMS_OPENING):
            return {"claims_match_design": True, "false_claims": ""}
        return self.replies.pop(0) if self.replies else None


def proved(ext_type="structural", **kw):
    r = {"verdict": "NECESSARY", "ext_type": ext_type,
         "move": "STORAGE" if ext_type == "structural" else None,
         "child_verilog": "module top; endmodule\n",
         "patch": "@@ module top; @@\n+  reg [3:0] q_cnt;",
         "new_state": "a 4-entry queue in front of the input"}
    r.update(kw)
    return r


def test_every_proved_step_is_reviewed():
    for ext_type in ("structural", "second", "distractor", "compose",
                     "replicate"):
        model = Model(reply())
        rec = S.review(proved(ext_type), call=model)
        # 26 Oct 2026: two calls for the structural step, one for the rest,
        # because change 6 checks the builder's words in a second call and
        # only the structural record here has any (its new_state)
        assert len(model.prompts) == (2 if ext_type == "structural"
                                      else 1), ext_type
        assert rec["verdict"] == "NECESSARY"


def test_a_step_that_did_not_prove_is_not_reviewed():
    model = Model(reply())
    for verdict in ("DECORATIVE", "NOT_PROVEN", "NO_NEW_SHAPE"):
        assert S.review(proved(verdict=verdict), call=model)["verdict"] == \
            verdict
    assert model.prompts == []


def test_the_reviewer_sees_the_change_and_the_earlier_steps():
    model = Model(reply())
    S.review(proved(), call=model,
             history=["STORAGE: a prefetch queue", "SUBUNIT: a retry unit"])
    # 26 Oct 2026: the builder's own words are in the second question, not
    # the first, because change 6 has the reviewer judge the code first
    prompt, claims = model.prompts
    assert "+  reg [3:0] q_cnt;" in prompt                  # the change
    assert "module top; endmodule" in prompt                # the design
    assert "a 4-entry queue in front of the input" not in prompt
    assert "a 4-entry queue in front of the input" in claims
    assert "1. STORAGE: a prefetch queue" in prompt
    assert "new storage" in prompt                          # what kind


def test_a_part_with_no_job_of_its_own_is_rejected():
    rec = S.review(proved(), call=Model(reply(
        new_part_has_its_own_job=False,
        why="a third buffer in a row in front of the same output")))
    assert rec["verdict"] == "NONSENSE"
    assert "third buffer" in rec["reason"]


def test_a_change_that_breaks_what_the_design_did_is_rejected():
    rec = S.review(proved(), call=Model(reply(
        breaks_what_it_did=True, why="body pieces are dropped")))
    assert rec["verdict"] == "NONSENSE"


def test_a_failed_review_is_tried_once_more_then_rejected():
    model = Model(None, None)
    rec = S.review(proved(), call=model)
    assert len(model.prompts) == 2
    assert rec["verdict"] == "UNREVIEWED"
    assert rec["sense_check"]["keep"] is False
    assert promote.route("UNREVIEWED") == "reject"
    again = Model(None, reply())
    assert S.review(proved(), call=again)["verdict"] == "NECESSARY"


def test_the_run_reviews_with_the_family_history(monkeypatch):
    rows = chain("STORAGE")
    seen = {}
    monkeypatch.setattr(batch, "call_model",
                        lambda p, s: (None, {"input": 1, "output": 1},
                                      "refusal"))
    monkeypatch.setattr(batch, "dump", lambda r: None)
    monkeypatch.setattr(batch, "SENSE_CHECK", True)
    monkeypatch.setattr(batch.sense_check, "review",
                        lambda rec, history=None: seen.update(h=history))
    task = {"task_id": 1, "attempt": 1, "ext_type": "structural",
            "move": "SUBUNIT", "parent_id": rows[-1]["id"], "k": 1}
    batch._real_executor(task, rows)
    assert seen["h"] == ["STORAGE: step 1"]


def _trial(tmp_path, name, labels):
    """A trial folder with hand labels and extension records, and its
    corpus file next to the folder, the way the trials are laid out."""
    d = tmp_path / "logs" / name
    d.mkdir(parents=True)
    rows = chain("STORAGE", "SUBUNIT")
    recs = []
    for i, (cid, label) in enumerate(labels.items()):
        recs.append(proved(extension_id=f"{name}_e{i}"))
        for r in rows:
            if r["id"] == cid:
                r["source_extension_id"] = f"{name}_e{i}"
    (d / "extensions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs))
    (d / "hand_labels.json").write_text(json.dumps(
        {cid: {"label": lab, "extension_id": f"{name}_e{i}"}
         for i, (cid, lab) in enumerate(labels.items())}))
    (tmp_path / f"corpus_{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_the_review_is_tried_on_both_hand_labelled_trials(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(S, "HERE", tmp_path)
    a = _trial(tmp_path, "trial_a", {"c1_000": "good", "c2_000": "weak"})
    b = _trial(tmp_path, "trial_b", {"c2_000": "wrong"})
    model = Model(reply(), reply(new_part_has_its_own_job=False),
                  reply(value_used_against_its_meaning=True))
    table = S.validate([a, b], call=model, workers=1)
    assert table == {"good": {"kept": 1, "rejected": 0},
                     "weak": {"kept": 0, "rejected": 1},
                     "wrong": {"kept": 0, "rejected": 1}}
    # each design is shown with the steps before it
    # 26 Oct 2026: the claims questions are left out because change 6 adds
    # one after each kept design; the second design question is the one
    design = [p for p in model.prompts if not p.startswith(CLAIMS_OPENING)]
    assert "1. STORAGE: step 1" in design[1]
    for d in (a, b):
        assert (d / "sense_check_every_step.jsonl").exists()


def test_the_18_oct_labels_cover_all_27_designs():
    path = ROOT / "extender" / "logs" / "growth_trial_2026-10-18" / \
        "hand_labels.json"
    if not path.exists():                       # local data, not in git
        return
    labels = json.loads(path.read_text())
    assert len(labels) == 27
    counts = {}
    for v in labels.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
        assert v["extension_id"]
    assert counts == {"good": 11, "weak": 15, "wrong": 1}


# --- 5. no small tweaks ---------------------------------------------------------

def test_small_tweaks_are_no_longer_drawn():
    w = extend.MOVE_WEIGHTS
    for small in ("SPLIT", "LEDGER", "BOUND", "GUARD"):
        assert w[small] == 0, small
    # their turns went to the two moves that stayed good; STAGE off since
    # 27 Oct 2026 (Rishab approved): all 3 reviewed were only delayed copies
    assert (w["SUBUNIT"], w["STORAGE"], w["PEER"], w["STAGE"]) == (7, 6, 3, 0)
    drawn = kinds_drawn([lines_row(60)])
    assert not drawn & {"SPLIT", "LEDGER", "BOUND", "GUARD"}


def test_pipelines_are_never_a_single_stage(monkeypatch):
    # STAGE is off since 27 Oct 2026; turned back on, it still comes in
    # runs of 4 or 8
    monkeypatch.setitem(batch.MOVE_WEIGHTS, "STAGE", 1)
    rng = random.Random(2)
    ks = {t["k"] for t in (batch.sample_task(rng, lines_row(40),
                                             [lines_row(40)])
                           for _ in range(3000))
          if t.get("move") == "STAGE"}
    assert ks == {4, 8}


# --- 6. starting from the biggest designs ------------------------------------

def start(id, lines, group, gen=0):
    r = lines_row(lines, id=id, top=f"m_{id}")
    r["generation"] = gen
    r["construct"] = f"{kind_in(group)}: x"
    return r


def test_a_run_can_start_from_the_biggest_designs_one_per_group():
    rows = [start("c0_1", 180, "memory"), start("c0_2", 170, "memory"),
            start("c0_3", 150, "crypto"), start("c0_4", 90, "DSP"),
            start("c0_5", 400, "DSP", gen=1), start("c0_6", 120, "control")]
    rows[5]["metrics"] = {"without_main_s": 900.0}      # too slow to extend
    assert batch.largest_starts(rows, 3) == ["c0_1", "c0_3", "c0_4"]


def test_a_design_whose_check_the_extender_cannot_read_is_not_a_start():
    """29 of the 252 starting designs write `assert property (...)`, which
    the extender does not read, so it sees no property at all; two of the
    five biggest are among them (found 18 Oct 2026)."""
    rows = [start("c0_1", 180, "memory"), start("c0_3", 150, "crypto")]
    rows[0]["property"] = []
    assert batch.largest_starts(rows, 2) == ["c0_3"]


def test_the_command_line_offers_the_biggest_starts(monkeypatch, tmp_path):
    seen = {}

    def fake_loop(rows, executor, **kw):
        seen["start"] = kw["start_ids"]
        return []
    monkeypatch.setattr(batch, "run_loop", fake_loop)
    corpus = tmp_path / "corpus_trial.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in [
        start("c0_1", 180, "memory"), start("c0_3", 150, "crypto"),
        start("c0_4", 90, "DSP")]))
    batch.main(["--max-calls", "1", "--corpus", str(corpus), "--log-dir",
                str(tmp_path / "logs"), "--start-largest", "2"])
    assert seen["start"] == ["c0_1", "c0_3"]
    with pytest.raises(SystemExit):
        batch.main(["--max-calls", "1", "--corpus", str(corpus),
                    "--log-dir", str(tmp_path / "logs"),
                    "--start-largest", "2", "--start", "c0_1"])
