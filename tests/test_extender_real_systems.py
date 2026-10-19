"""Growing designs into real systems (19 Oct 2026).

Hand review of the 18 Oct good-growth trial (13 kept of 40 tries; 11 good,
2 weak):
  - second properties were kept 0 times in 8 and copies 0 in 3: neither
    is drawn now;
  - three sub-units that doubled their designs with real features were
    turned away only because their facts had familiar shapes: a big step
    the reviewer keeps, grown by half or more, now passes without a new
    kind of fact;
  - two designs grew mostly by writing one block once per slot (61% and
    83% of their new lines): slots are written as arrays, size is counted
    without copied lines, and a step whose new lines are mostly copies is
    rejected;
  - the reviewer rejected the same credit pool three times on one design,
    and two sibling designs added the same length guard: the question now
    lists what was rejected on the design and what its other children
    added;
  - sub-units and storage are no longer limited to two per family;
  - combine partners were drawn at random: the model now picks one from a
    short list of starting designs, or says none fits;
  - a new sub-module may connect to the modules already in the design;
  - 29 of the 252 starting designs write `assert property (...)`, which
    the extender could not read.
Rishab: "let's do all 4 and then rerun". The model and checker are faked."""
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import build_corpus                                             # noqa: E402
import extend                                                   # noqa: E402
import gates                                                    # noqa: E402
import promote                                                  # noqa: E402

needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None,
                                 reason="yosys not on PATH (run hwtools)")

CATALOG = json.loads((ROOT / "initiator" / "catalog" /
                      "kind_categories_2026-10-09.json").read_text())
_GROUP = {a["id"]: a["group"] for a in CATALOG["areas"]}


def kind_in(group):
    return next(k["name"] for k in CATALOG["kinds"]
                if _GROUP.get(k["category"]) == group)


def design(n, id="c0_1", top="m", gen=0, group=None, facts=("a <= 4'd9",)):
    """A design of n lines, every line different (no copies)."""
    body = [f"module {top} (input wire clk, input wire en, "
            "output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(n - 2)] + \
        ["endmodule"]
    row = {"id": id, "generation": gen, "top_module": top, "clock": "clk",
           "antecedents": [], "sanity_covers": [],
           "verilog": "\n".join(body), "property": ["a <= 4'd9"],
           "invariants": list(facts), "metrics": {}}
    if group:
        row["construct"] = f"{kind_in(group)}: does its job"
    return row


@pytest.fixture(autouse=True)
def fresh_run_memory(monkeypatch):
    monkeypatch.setattr(batch, "REJECTED", defaultdict(list), raising=False)


# --- 1. the quick fixes ----------------------------------------------------------

def test_second_properties_and_copies_are_no_longer_drawn():
    assert extend.SECOND_PROPERTY_RATE == 0
    assert batch.COPIES is False
    rng = random.Random(3)
    row = design(60)
    kinds = {batch.sample_task(rng, row, [row])["ext_type"]
             for _ in range(3000)}
    assert not kinds & {"second", "replicate"}


def proved_big_step(move="SUBUNIT", grow_to=100, parent_lines=60):
    parent = design(parent_lines, id="c0_000")
    child = design(grow_to, id="x")["verilog"]      # new lines, none copied
    return parent, {
        "extension_id": "e1", "ext_type": "structural", "move": move,
        "parent_id": "c0_000", "verdict": "NECESSARY",
        "child_verilog": child, "patch": "",
        "invariants": ["a <= 4'd7"]}         # the same kind of fact


def test_a_big_step_needs_no_new_kind_of_fact_if_the_reviewer_will_read_it():
    parent, rec = proved_big_step()
    promote.ratchet(rec, [parent], review_waiver=True)
    assert rec["verdict"] == "NECESSARY"
    assert rec["shape_waiver"]
    # a later check of the same record (the run checks again) keeps it
    promote.ratchet(rec, [parent])
    assert rec["verdict"] == "NECESSARY"


def test_without_the_reviewer_the_old_rule_holds():
    parent, rec = proved_big_step()
    assert promote.ratchet(rec, [parent])["verdict"] == "NO_NEW_SHAPE"


def test_the_pass_is_only_for_big_steps_that_grew_by_half():
    parent, small = proved_big_step(move="STAGE")
    assert promote.ratchet(small, [parent], review_waiver=True)["verdict"] \
        == "NO_NEW_SHAPE"
    parent, little = proved_big_step(grow_to=75)              # +25%
    assert promote.ratchet(little, [parent], review_waiver=True)["verdict"] \
        == "NO_NEW_SHAPE"


def test_the_run_gives_the_pass_only_when_the_reviewer_is_on(monkeypatch):
    parent, rec = proved_big_step()
    out = {"new_state": "a retire queue", "coupling": "c",
           "induction_gap": "g", "why_parent_insufficient": "w",
           "patch": "", "invariants": rec["invariants"], "dispositions": []}

    def grade(p, ext_type, o, record):
        record.update(verdict="NECESSARY", child_verilog=rec["child_verilog"],
                      invariants=rec["invariants"], patch="")
    monkeypatch.setattr(batch, "call_model",
                        lambda p, s: (out, {"input": 1, "output": 1}, "ok"))
    monkeypatch.setattr(batch, "grade_step4", grade)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    task = {"task_id": 1, "attempt": 1, "ext_type": "structural",
            "move": "SUBUNIT", "parent_id": "c0_000", "k": 1}
    assert batch._real_executor(dict(task), [parent])["verdict"] == \
        "NO_NEW_SHAPE"
    monkeypatch.setattr(batch, "SENSE_CHECK", True)
    reviewed = []
    monkeypatch.setattr(batch.sense_check, "review",
                        lambda r, history=None: reviewed.append(r) or r)
    assert batch._real_executor(dict(task), [parent])["verdict"] == \
        "NECESSARY"
    assert reviewed                     # the reviewer read the passed step


# copied text

SLOTS = "\n".join(
    f"+    q{k}_len <= 6'd0;\n+    q{k}_csum <= 8'd0;\n"
    f"+    q{k}_stat <= ST_NONE;\n+    q{k}_par <= 1'b0;" for k in range(8))
TABLE = "\n".join(f"+        5'd{k}: enc6 = 6'b{k:06b};" for k in range(32))


def test_copied_lines_are_counted_once():
    copies = [line[1:] for line in SLOTS.splitlines()]
    assert build_corpus.copied_lines(copies) == 28          # 32 lines, 4 kinds
    # a lookup table's rows differ in their values: no copies
    rows = [line[1:] for line in TABLE.splitlines()]
    assert build_corpus.copied_lines(rows) == 0
    text = "module m;\n" + "\n".join(copies) + "\nendmodule\n"
    assert build_corpus.honest_lines(text) == 1 + 4    # endmodule: no content


def proved_patch(patch, ext_type="structural"):
    return {"verdict": "NECESSARY", "ext_type": ext_type, "move": "STORAGE",
            "patch": patch, "child_verilog": "module m; endmodule",
            "invariants": ["a <= 4'd9"]}


def test_a_step_whose_new_lines_are_mostly_copies_is_rejected():
    rec = gates.apply(proved_patch(SLOTS), [design(40)], use_yosys=False)
    assert rec["verdict"] == "COPIED_TEXT"
    assert promote.route("COPIED_TEXT") == "reject"
    table = gates.apply(proved_patch(TABLE), [design(40)], use_yosys=False)
    assert table["verdict"] == "NECESSARY"


def test_repeated_slots_are_asked_for_as_arrays():
    text = " ".join(extend.SYSTEM_PROMPT.split())
    assert "array" in text and "for loop" in text and "once per slot" in text
    stage = extend.STAGE_K_TEXT
    assert "no generate loops" not in stage and "spelled out" not in stage
    assert "array" in stage


def test_size_is_counted_without_copied_lines():
    copied = design(40)
    copied["verilog"] = copied["verilog"].replace(
        "endmodule", "\n".join(line[1:] for line in SLOTS.splitlines()
                               * 10) + "\nendmodule")
    n, lo, hi = extend.size_target(copied["verilog"], "SUBUNIT")
    assert n == build_corpus.honest_lines(copied["verilog"]) < 50
    parent = design(40, id="c0_000")
    rec = {"child_verilog": copied["verilog"]}
    assert promote.growth(rec, [parent]) < 0.2       # copies are not growth
    # the stop at about one training example still counts every line
    assert len(copied["verilog"].splitlines()) > 300


def test_the_training_example_stop_still_counts_every_line():
    long = design(40)
    long["verilog"] = long["verilog"].replace(
        "endmodule", "\n".join(line[1:] for line in SLOTS.splitlines()
                               * 32) + "\nendmodule")
    assert len(long["verilog"].splitlines()) > batch.MAX_LINES
    assert not batch.extendable(long, None)


# what was tried and what the siblings have

def test_the_question_lists_siblings_and_rejected_ideas():
    text = " ".join(extend.build_prompt(
        design(60), "structural", "SUBUNIT",
        history=["STORAGE: a prefetch queue"],
        siblings=["SUBUNIT: a frame-length guard"],
        rejected=["SUBUNIT: a credit pool. Rejected: it drops data"]).split())
    assert "1. STORAGE: a prefetch queue" in text
    assert "SUBUNIT: a frame-length guard" in text
    assert "it drops data" in text
    assert "do not try" in text.lower()


def test_the_other_designs_grown_from_a_design_are_listed():
    start = design(60)
    kids = [dict(design(60, id=f"c1_00{i}"), parent="c0_1",
                 ext_type="structural", move=m, added=a, generation=1)
            for i, (m, a) in enumerate([("SUBUNIT", "a length guard"),
                                        ("STORAGE", "a frame list")])]
    by_id = {r["id"]: r for r in [start] + kids}
    assert promote.children(start, by_id) == [
        "SUBUNIT: a length guard", "STORAGE: a frame list"]
    assert promote.children(kids[0], by_id) == []


def test_what_a_step_added_keeps_two_sentences():
    """A first sentence alone can say nothing of the job ("A new sub-module
    fm_pkt_guard (instance u_guard) is added.", 18 Oct trial)."""
    rec = {"ext_type": "structural", "new_state": (
        "A new sub-module fm_pkt_guard is added. It caps payloads at 48 "
        "bytes. It also counts runts.")}
    assert promote.what_was_added(rec) == (
        "A new sub-module fm_pkt_guard is added. It caps payloads at 48 "
        "bytes.")


def test_the_run_remembers_what_the_reviewer_rejected(monkeypatch):
    parent = design(60, id="c0_000")
    prompts = []
    out = {"new_state": "A credit pool in front of the input.",
           "coupling": "c", "induction_gap": "g",
           "why_parent_insufficient": "w", "patch": "", "invariants": [],
           "dispositions": []}

    def model(prompt, schema):
        prompts.append(prompt)
        return out, {"input": 1, "output": 1}, "ok"

    def grade(p, ext_type, o, record):
        record.update(verdict="NECESSARY", child_verilog="module m; endmodule",
                      invariants=["v + a == 4'd9"], patch="")

    def review(record, history=None):
        record.update(verdict="NONSENSE",
                      reason="sense check: it drops valid input data")
        return record
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "grade_step4", grade)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    monkeypatch.setattr(batch, "SENSE_CHECK", True)
    monkeypatch.setattr(batch.sense_check, "review", review)
    task = {"task_id": 1, "attempt": 1, "ext_type": "structural",
            "move": "SUBUNIT", "parent_id": "c0_000", "k": 1}
    batch._real_executor(dict(task), [parent])
    batch._real_executor(dict(task, task_id=2), [parent])
    assert "drops valid input data" not in prompts[0]
    assert "A credit pool in front of the input." in prompts[1]
    assert "drops valid input data" in prompts[1]


def test_sub_units_and_storage_have_no_per_family_limit():
    rows = [design(60)]
    for i, kind in enumerate(["SUBUNIT", "STORAGE"] * 3 + ["PEER"] * 2, 1):
        rows.append(dict(design(60, id=f"c{i}_000"), generation=i,
                         parent=rows[-1]["id"], ext_type="structural",
                         move=kind, added=f"step {i}"))
    rng = random.Random(5)
    drawn = {t.get("move") for t in (batch.sample_task(rng, rows[-1], rows)
                                     for _ in range(2000))}
    assert {"SUBUNIT", "STORAGE"} <= drawn
    assert "PEER" not in drawn


# --- 2. combine, with a partner the model chooses -----------------------------

def pool():
    """A memory design and possible partners: fitting starting designs,
    one too big, one grown, one from a group that does not fit."""
    a = design(80, id="c0_1", top="mem_a", group="memory")
    fit = [design(60 + i, id=f"c0_{10 + i}", top=f"if_{i}", group="interface")
           for i in range(10)]
    big = design(260, id="c0_30", top="if_big", group="interface")
    grown = dict(design(70, id="c1_000", top="if_grown", group="interface"),
                 generation=1, parent="c0_10")
    far = design(60, id="c0_40", top="pw", group="clocking/reset/power")
    return a, [a] + fit + [big, grown, far]


def test_a_combination_offers_a_short_list_of_starting_designs():
    a, rows = pool()
    rng = random.Random(1)
    tasks = [t for t in (batch.sample_task(rng, a, rows) for _ in range(400))
             if t["ext_type"] == "compose"]
    assert tasks
    for t in tasks:
        assert t["parent2_id"] is None
        assert 1 <= len(t["candidates"]) <= batch.PARTNER_CHOICES
        assert set(t["candidates"]) <= {f"c0_{10 + i}" for i in range(10)}


def test_the_partner_question_shows_purpose_and_ports():
    a, rows = pool()
    text = extend.partner_prompt(a, rows[1:3], {r["id"]: r for r in rows})
    for r in [a] + rows[1:3]:
        assert r["top_module"] in text
    assert "does its job" in text                     # the purpose
    assert "en" in text and "a[3:0]" in text          # the ports
    assert '"none"' in text


def compose_task():
    return {"task_id": 1, "attempt": 1, "ext_type": "compose",
            "parent_id": "c0_1", "parent2_id": None, "move": None,
            "candidates": ["c0_10", "c0_11"]}


def test_the_model_may_say_no_partner_fits(monkeypatch):
    a, rows = pool()
    calls = []

    def model(prompt, schema):
        calls.append(schema)
        return ({"choice": "none", "system": "nothing here takes memory "
                 "reads", "connection": ""}, {"input": 1, "output": 1}, "ok")
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    rec = batch._real_executor(compose_task(), rows)
    assert rec["verdict"] == "NO_PARTNER"
    assert len(calls) == 1
    assert promote.route("NO_PARTNER") == "pass"


def test_a_chosen_partner_is_combined_as_planned(monkeypatch):
    a, rows = pool()
    prompts, graded = [], {}
    plan = {"choice": "c0_11", "system": "a memory with a host interface",
            "connection": "mem_a's read data feeds if_1's transmit data"}

    def model(prompt, schema):
        prompts.append(prompt)
        if len(prompts) == 1:
            return plan, {"input": 1, "output": 1}, "ok"
        return ({"glue_scheme": "g", "induction_gap": "i",
                 "why_parents_insufficient": "w", "top_module": "t",
                 "wrapper": "", "antecedents": [], "sanity_covers": [],
                 "invariants": []}, {"input": 1, "output": 1}, "ok")

    def grade(pa, pb, out, record):
        graded["partner"] = pb["id"]
        record["verdict"] = "NOT_PROVEN"
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "grade_compose", grade)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    task = compose_task()
    rec = batch._real_executor(task, rows)
    assert graded["partner"] == "c0_11"
    assert task["parent2_id"] == rec["parent2_id"] == "c0_11"
    assert "mem_a's read data feeds if_1's transmit data" in prompts[1]
    assert rec["partner_choice"]["system"] == plan["system"]


def test_no_partner_does_not_count_against_the_design(monkeypatch):
    a, rows = pool()
    calls = []

    def exe(task, corpus):
        calls.append((task["parent_id"], task["attempt"]))
        return {"verdict": "NO_PARTNER", "extension_id": f"e{len(calls)}",
                "parent_id": task["parent_id"]}
    batch.run_loop(rows, exe, max_calls=12, rng=random.Random(0),
                   start_ids=["c0_1"])
    # not retried as a format failure, and the design is not retired
    # after three: every call is a fresh task on it
    assert calls == [("c0_1", 1)] * 12


# --- 3. modules that work together ----------------------------------------

TWO_MODULES = """module guard (
    input  wire clk,
    input  wire i_open,
    output reg  o_open
);
    always @(posedge clk) o_open <= i_open;
endmodule

module framer (input wire clk, input wire sof, output wire busy);
    wire g_open;
    guard u_guard (.clk(clk), .i_open(sof), .o_open(g_open));
    assign busy = g_open;
    always @(posedge clk) assert (!busy || sof || g_open);
endmodule
"""


def test_the_question_lists_the_modules_already_in_the_design():
    row = dict(design(40), top_module="framer", verilog=TWO_MODULES,
               property=["!busy || sof || g_open"])
    text = " ".join(extend.build_prompt(row, "structural", "SUBUNIT").split())
    assert "u_guard (guard): inputs clk, i_open; outputs o_open" in text
    assert "drive their inputs" in text
    plain = extend.build_prompt(design(40), "structural", "SUBUNIT")
    assert "Sub-modules already in this design" not in plain


# --- 4. starting designs that write `assert property` ------------------------

CONCURRENT = """module c (input wire clk, input wire en, output reg [3:0] n);
    initial n = 4'd0;
    always @(posedge clk) if (en) n <= (n == 4'd9) ? 4'd0 : n + 4'd1;
    assert property (n != 4'd12);
    always @(posedge clk) assert (n <= 4'd9);
endmodule
"""


def test_a_concurrent_check_is_read():
    assert build_corpus.extract_asserts(CONCURRENT) == ["n != 4'd12",
                                                        "n <= 4'd9"]


def test_a_concurrent_check_can_be_taken_out():
    out = extend.remove_asserts(CONCURRENT, ["n != 4'd12"])
    assert build_corpus.extract_asserts(out) == ["n <= 4'd9"]
    assert "property" not in out


@needs_yosys
def test_taking_out_a_concurrent_check_leaves_legal_verilog():
    out = extend.remove_asserts(CONCURRENT, ["n != 4'd12"])
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "c.sv"
        f.write_text(out)
        r = subprocess.run(["yosys", "-p", f"read_verilog -sv {f}; "
                            "hierarchy -top c; proc"], capture_output=True,
                           text=True)
    assert r.returncode == 0, r.stdout[-500:]


def test_a_run_reads_the_checks_the_corpus_missed(monkeypatch, tmp_path):
    seen = {}

    def fake_loop(rows, executor, **kw):
        seen["rows"] = rows
        return []
    monkeypatch.setattr(batch, "run_loop", fake_loop)
    row = dict(design(40), verilog=CONCURRENT, top_module="c", property=[])
    corpus = tmp_path / "corpus_trial.jsonl"
    corpus.write_text(json.dumps(row) + "\n")
    batch.main(["--max-calls", "1", "--corpus", str(corpus), "--log-dir",
                str(tmp_path / "logs")])
    assert seen["rows"][0]["property"] == ["n != 4'd12", "n <= 4'd9"]
