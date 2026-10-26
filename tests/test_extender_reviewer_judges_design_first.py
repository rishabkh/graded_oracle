"""The reviewer judges the design from its code first, and the builder's
words only after (26 Oct 2026, Rishab approved, change 6).

In the 25 Oct study 8 of 14 combined and second-lane designs lost data and
5 of them were kept, because the loss was counted or flagged; the reviewer
read the builder's words before the code and repeated them ("never
refuses"). Now the first call sees the code, the change, the earlier steps
and what the checker's own tools found (the nothing-lost search, new input
ports), never the builder's words, and it is also asked about values cut
or padded, delayed copies and lost items, counted or not. Only a design it
keeps, and whose builder said something, goes to a second call that checks
the builder's claims against the code. For now (decision 2 of the 26 Oct
spec) that call only records what it finds, as record["claims_check"], and
rejects nothing; --validate shows it.

The model is faked here: no calls. The checker's notes come from
nothing_lost.reviewer_notes, written alongside this change; they are faked
too, except in one test that runs only once that file exists."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import sense_check as S                                         # noqa: E402

REAL_CHECKER_NOTES = S.checker_notes
CLAIMS_OPENING = "You reviewed a hardware design"
FACTS_HEADING = "What the checker's own tools found (not the builder's words):"
MATCH = {"claims_match_design": True, "false_claims": ""}
# two lines worded the way nothing_lost.reviewer_notes words them (spec 5.12)
NOTHING_LOST_NOTE = (
    "the checker's own nothing-is-lost search, stream 'requests into the "
    "queue' (handed over when `en && in_valid` and `in_ready`; delivered "
    "q_pop; held q_count): no item lost in any input sequence of 48 cycles "
    "from the start")
TAKES_EVERYTHING_NOTE = (
    "stream 'results into the checker' counts each item delivered in the "
    "cycle it is offered and holds none, so it assumes the receiving design "
    "takes every item in every state")


def squash(text):
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def notes(monkeypatch):
    """The checker's notes on the record under review, faked; a test fills
    the list it is given."""
    lines = []
    monkeypatch.setattr(S, "checker_notes", lambda record: list(lines))
    return lines


def structural(**kw):
    r = {"verdict": "NECESSARY", "ext_type": "structural", "move": "STORAGE",
         "child_verilog": ("module top (input wire clk, input wire en);\n"
                           "  reg [2:0] q_count;\nendmodule\n"),
         "patch": "@@ module top @@\n+  reg [2:0] q_count;",
         "new_state": "a 4-entry request queue that never refuses a request"}
    r.update(kw)
    return r


def combination(**kw):
    r = {"verdict": "NECESSARY", "ext_type": "compose",
         "child_verilog": ("module a; endmodule\nmodule b; endmodule\n"
                           "module pair; endmodule\n"),
         "wrapper": "module pair; endmodule\n",
         "glue_scheme": "a 2-entry buffer holds A's results until B is idle",
         "partner_choice": {"choice": "c0_2",
                            "system": "a parser feeding a checksum unit",
                            "connection": "A's result drives B's start; B "
                                          "never refuses"}}
    r.update(kw)
    return r


def answer(**kw):
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
         "why": "a queue that makes the sender wait"}
    a.update(kw)
    return a


class Model:
    """A fake reviewer. The design call gets each reply in turn (None when
    none is left), the claims call each claims reply in turn (a match when
    none is left); a reply that is an exception is raised. Every question
    asked is kept."""

    def __init__(self, *replies, claims=()):
        self.replies, self.claims, self.prompts = \
            list(replies), list(claims), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith(CLAIMS_OPENING):
            reply = self.claims.pop(0) if self.claims else MATCH
        else:
            reply = self.replies.pop(0) if self.replies else None
        if isinstance(reply, Exception):
            raise reply
        return reply


def builders_words(rec):
    plan = rec.get("partner_choice") or {}
    return [w for w in (rec.get("new_state"), rec.get("glue_scheme"),
                        plan.get("system"), plan.get("connection")) if w]


# --- the first call: the code alone ------------------------------------------

def test_the_first_call_has_no_builder_words_and_has_the_checkers_notes(
        notes):
    notes[:] = [NOTHING_LOST_NOTE, "the step adds the input ports b_valid"]
    for rec in (structural(), combination()):
        model = Model(answer())
        S.review(rec, call=model)
        design = model.prompts[0]
        assert rec["child_verilog"] in design
        for words in builders_words(rec):
            assert words not in design, words
        assert "In the builder's words" not in design
        assert "Planned system" not in design
        assert "Planned connection" not in design
        facts = design.split(FACTS_HEADING)[1].split("1. parts_connected")[0]
        assert f"  - {NOTHING_LOST_NOTE}" in facts
        assert "  - the step adds the input ports b_valid" in facts
    # nothing found, no section
    notes[:] = []
    model = Model(answer())
    S.review(structural(), call=model)
    assert FACTS_HEADING not in model.prompts[0]


def test_the_first_call_asks_the_missing_questions():
    text = squash(S.PROMPT)
    assert "counted, flagged or raised as an error bit" in text
    assert "cut down or padded" in text
    assert "delayed copy of a signal" in text
    assert "does the change stop the old part doing anything" in text
    assert "silently" not in text
    # the numbered questions are the answer's fields, in the same order
    fields = S._FLAGS + ("why",)
    assert S.SCHEMA["required"] == list(fields)
    for i, field in enumerate(fields, 1):
        assert f"{i}. {field}:" in text, field
    assert "reasoning" not in S.SCHEMA["properties"]


@pytest.mark.parametrize("flag", ["item_can_be_dropped",
                                  "only_a_delayed_copy",
                                  "value_cut_or_padded"])
def test_each_new_answer_can_reject(flag):
    model = Model(answer(**{flag: True}, why=f"rejected for {flag}"))
    rec = S.review(structural(), call=model)
    assert rec["verdict"] == "NONSENSE"
    assert rec["sense_check"]["keep"] is False
    assert rec["reason"] == f"sense check: rejected for {flag}"
    assert len(model.prompts) == 1


def test_the_two_questions_outside_the_rule_still_reject_nothing():
    """Asked and recorded, but outside the keep rule, as before."""
    for flag, value in (("new_logic_only_copies_existing_state", True),
                        ("results_of_parts_used", False)):
        rec = S.review(structural(), call=Model(answer(**{flag: value})))
        assert rec["verdict"] == "NECESSARY", flag
        assert rec["sense_check"][flag] is value


def test_an_answer_missing_a_question_counts_as_no_answer():
    short = answer()
    del short["item_can_be_dropped"]
    model = Model(short, dict(short))
    rec = S.review(structural(), call=model)
    assert rec["verdict"] == "UNREVIEWED"
    assert len(model.prompts) == 2
    assert rec["sense_check"]["keep"] is False
    assert "item_can_be_dropped" in rec["sense_check"]["error"]
    assert "item_can_be_dropped" in rec["reason"]
    # a reply that is not a JSON object is no answer either
    rec = S.review(structural(), call=Model(["not", "an", "object"], "no"))
    assert rec["verdict"] == "UNREVIEWED"
    # a short reply, then a whole one: the second try counts
    assert S.review(structural(), call=Model(short, answer()))["verdict"] \
        == "NECESSARY"


# --- the second call: the builder's claims, checked against the code ---------

def test_the_second_call_sees_the_code_the_first_answers_and_the_builders_words(
        notes):
    notes[:] = [TAKES_EVERYTHING_NOTE]
    first = answer(why="a buffer that makes A wait while B is busy")
    rec = combination()
    model = Model(first)
    S.review(rec, call=model)
    design, claims = model.prompts
    assert claims.startswith(CLAIMS_OPENING)
    assert not design.startswith(CLAIMS_OPENING)
    assert rec["child_verilog"] in claims                   # the code
    assert TAKES_EVERYTHING_NOTE in claims                  # the notes
    shown = claims.split("Your answers about it:")[1].split(
        "Now read what the builder says")[0]
    assert json.loads(shown) == {k: first[k] for k in S._FLAGS + ("why",)}
    words = claims.split("Now read what the builder says about the design "
                         "and the change:")[1]
    assert rec["glue_scheme"] in words
    assert "Planned system: a parser feeding a checksum unit" in words
    assert "Planned connection: A's result drives B's start; B never " \
        "refuses" in words
    # a patched step shows its change in the second call too
    rec = structural()
    model = Model(answer())
    S.review(rec, call=model)
    claims = model.prompts[1]
    assert "+  reg [2:0] q_count;" in claims
    assert rec["new_state"] in claims.split("Now read what the builder")[1]


def test_a_false_claim_about_behaviour_is_recorded_and_the_design_kept():
    """Decision 2 of the 26 Oct spec: recorded, not a rejection, until the
    claims call has been tried on the hand-labelled trials."""
    false = {"claims_match_design": False,
             "false_claims": '"never refuses a request": the queue ignores '
                             'a push while it is full'}
    model = Model(answer(), claims=[false])
    rec = S.review(structural(), call=model)
    assert len(model.prompts) == 2
    assert rec["verdict"] == "NECESSARY"
    assert rec["sense_check"]["keep"] is True
    assert rec["claims_check"] == false
    assert "reason" not in rec


def test_a_failed_second_call_is_recorded_and_the_design_kept():
    model = Model(answer(), claims=[RuntimeError("no answer (refusal)"),
                                    None])
    rec = S.review(structural(), call=model)
    assert len(model.prompts) == 3                  # one design, two claims
    assert rec["verdict"] == "NECESSARY"
    assert rec["sense_check"]["keep"] is True
    assert rec["claims_check"] == {
        "error": "RuntimeError: no answer (refusal)", "tries": S.TRIES}
    assert "could not be checked" in S.claims_line(rec["claims_check"])
    # a claims reply missing a field is a failed try too
    half = {"claims_match_design": False}
    rec = S.review(structural(), call=Model(answer(), claims=[half, half]))
    assert rec["verdict"] == "NECESSARY"
    assert "false_claims" in rec["claims_check"]["error"]


def test_a_design_rejected_by_the_first_call_has_no_second_call():
    model = Model(answer(item_can_be_dropped=True,
                         why="a push is ignored while the queue is full"))
    rec = S.review(structural(), call=model)
    assert rec["verdict"] == "NONSENSE"
    assert len(model.prompts) == 1
    assert "claims_check" not in rec
    # nor one the first call could not review
    model = Model(None, None)
    rec = S.review(structural(), call=model)
    assert rec["verdict"] == "UNREVIEWED"
    assert len(model.prompts) == 2
    assert "claims_check" not in rec


def test_a_kept_design_with_no_builder_words_needs_no_second_call():
    for rec in (structural(new_state=""), structural(new_state="  \n"),
                combination(glue_scheme="", partner_choice=None),
                {"verdict": "NECESSARY", "ext_type": "second",
                 "child_verilog": "module top; endmodule\n",
                 "patch": "@@ endmodule @@\n+ // p2"}):
        model = Model(answer())
        S.review(rec, call=model)
        assert len(model.prompts) == 1, rec
        assert rec["verdict"] == "NECESSARY"
        assert "claims_check" not in rec
    # a combination's plan alone is something the builder said
    model = Model(answer())
    S.review(combination(glue_scheme=""), call=model)
    assert len(model.prompts) == 2
    assert "Planned connection:" in model.prompts[1]


def test_both_calls_use_the_reviewer_system_text_and_their_own_schema(
        monkeypatch):
    """Both count as reviews in the realism report, which tells the calls
    apart by their system text."""
    sent = []

    def call_claude(**kw):
        sent.append(kw)
        reply = MATCH if kw["schema"] is S.CLAIMS_SCHEMA else answer()
        return json.dumps(reply), {"input": 1, "output": 1}, "ok"
    monkeypatch.setattr(S.llm_client, "call_claude", call_claude)
    rec = S.review(structural())
    assert [kw["schema"] for kw in sent] == [S.SCHEMA, S.CLAIMS_SCHEMA]
    assert all(kw["system"] == S.SYSTEM for kw in sent)
    assert all(kw["model"] == S.MODEL for kw in sent)
    assert sent[0]["user"].startswith("This design was built by")
    assert sent[1]["user"].startswith(CLAIMS_OPENING)
    assert rec["verdict"] == "NECESSARY" and rec["claims_check"] == MATCH
    # a strict schema of its own, with no field named "reasoning"
    assert S.CLAIMS_SCHEMA["required"] == ["claims_match_design",
                                           "false_claims"]
    assert S.CLAIMS_SCHEMA["additionalProperties"] is False
    assert "reasoning" not in S.CLAIMS_SCHEMA["properties"]


# --- what the checker's tools found ------------------------------------------

def test_the_reviewer_is_told_when_a_stream_assumes_the_receiver_takes_everything(
        notes):
    notes[:] = [TAKES_EVERYTHING_NOTE]
    model = Model(answer())
    S.review(combination(), call=model)
    design, claims = model.prompts
    facts = design.split(FACTS_HEADING)[1].split("1. parts_connected")[0]
    assert TAKES_EVERYTHING_NOTE in facts
    assert TAKES_EVERYTHING_NOTE in claims.split(FACTS_HEADING)[1]
    # and it is asked whether an item can be lost in some state
    assert "8. item_can_be_dropped:" in design


def test_the_notes_are_asked_of_the_nothing_lost_checker(monkeypatch):
    """checker_notes hands the record to nothing_lost.reviewer_notes (spec
    5.12), once per review, and both calls show what it says."""
    seen = []
    fake = types.ModuleType("nothing_lost")
    fake.reviewer_notes = lambda record: (
        seen.append(record) or ["the step adds the input ports b_valid"])
    monkeypatch.setitem(sys.modules, "nothing_lost", fake)
    monkeypatch.setattr(S, "checker_notes", REAL_CHECKER_NOTES)
    rec = structural(move="PEER", new_inputs=["b_valid"])
    model = Model(answer())
    S.review(rec, call=model)
    assert len(seen) == 1 and seen[0] is rec
    for prompt in model.prompts:
        assert "  - the step adds the input ports b_valid" in prompt


def test_the_checker_words_its_notes_from_the_record(monkeypatch):
    """The record fields the graders write (spec 5.11 and 5.12) become the
    reviewer's notes, through nothing_lost.py itself; skipped until that
    file exists."""
    pytest.importorskip("nothing_lost")
    monkeypatch.setattr(S, "checker_notes", REAL_CHECKER_NOTES)
    rec = combination(
        streams=[{"name": "results into the checker",
                  "inputs": [{"offer": "a_done", "ready": ""}],
                  "delivered": "a_done", "held": "0", "flush": ""}],
        nothing_lost={"result": "nothing_lost", "depth": 48, "seconds": 1.0,
                      "reset": "rst", "step": None, "stream": None,
                      "checked_to": None, "why": None, "workdir": None},
        glue_identifiers=[])
    model = Model(answer())
    S.review(rec, call=model)
    design = squash(model.prompts[0])
    assert "no item lost in any input sequence of 48 cycles" in design
    assert ("assumes the receiving design takes every item in every state"
            in design)
    assert "the part between the two designs has no registers" in design
    # a step with none of these fields (logged before 26 Oct 2026) has none
    model = Model(answer())
    S.review(structural(), call=model)
    assert FACTS_HEADING not in model.prompts[0]


# --- the report --------------------------------------------------------------

def test_the_report_shows_what_the_claims_call_found(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(S, "HERE", tmp_path)        # no corpus: no history
    d = tmp_path / "logs" / "trial_x"
    d.mkdir(parents=True)
    stale = {"claims_match_design": False, "false_claims": "from a run"}
    recs = [structural(extension_id="e0"), structural(extension_id="e1"),
            structural(extension_id="e2", new_state="", claims_check=stale)]
    (d / "extensions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs))
    (d / "hand_labels.json").write_text(json.dumps(
        {"c1_000": {"label": "good", "extension_id": "e0"},
         "c1_001": {"label": "weak", "extension_id": "e1"},
         "c1_002": {"label": "weak", "extension_id": "e2"}}))
    false = {"claims_match_design": False,
             "false_claims": '"never refuses a request"'}
    model = Model(answer(), answer(), answer(), claims=[MATCH, false])
    table = S.validate([d], call=model, workers=1)
    # the claims call decides nothing: all three are kept
    assert table == {"good": {"kept": 1, "rejected": 0},
                     "weak": {"kept": 2, "rejected": 0}}
    rows = {r["id"]: r for r in map(json.loads, (
        d / "sense_check_every_step.jsonl").read_text().splitlines())}
    assert rows["c1_000"]["claims_check"] == MATCH
    assert rows["c1_001"]["claims_check"] == false
    assert rows["c1_001"]["verdict"] == "NECESSARY"
    assert rows["c1_002"]["claims_check"] is None   # nothing said, no call
    out = capsys.readouterr().out
    assert ("the builder claims what the code does not do (recorded, not a "
            "rejection): \"never refuses a request\"") in out
    assert "the builder's claims match the code" in out
    assert "good  match 1, false 0, not checked 0" in out
    assert "weak  match 0, false 1, not checked 0" in out
    assert "from a run" not in out
