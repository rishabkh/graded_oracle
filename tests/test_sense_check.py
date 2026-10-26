"""A second model reads each combination or copy and says whether it makes
sense as hardware (18 Oct 2026).

The checks in gates.py are structural. The one wrong design of the 17 Oct
trial they cannot see was consistent and meaningless at once: CRC report
frames sent with zeros where the check bits go, and a property proving
exactly that. Before this review is trusted it is tried on the 15 combined
and copied designs of that trial, against the hand labels (--validate).
The model is faked here: no calls."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import promote                                                  # noqa: E402
import sense_check as S                                         # noqa: E402


@pytest.fixture(autouse=True)
def no_checker_notes(monkeypatch):
    # 26 Oct 2026: the reviewer now reads the checker's notes
    # (nothing_lost.reviewer_notes) because change 6 shows it what the
    # checker's own tools found; these records carry none, and the
    # reviewer is tested here on its own
    monkeypatch.setattr(S, "checker_notes", lambda record: [])


def wrapped(**kw):
    r = {"verdict": "NECESSARY", "ext_type": "compose",
         "child_verilog": "module a; endmodule\nmodule top; endmodule\n",
         "wrapper": "module top; endmodule\n",
         "glue_scheme": "a one-entry report buffer feeds the CRC checker"}
    r.update(kw)
    return r


def answer(**kw):
    # 26 Oct 2026: three questions added (value_cut_or_padded,
    # only_a_delayed_copy, item_can_be_dropped) because change 6 asks about
    # cut or padded values, delayed copies and lost items; an answer
    # without them is now a failed try
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
         "why": "a sensible pipeline"}
    a.update(kw)
    return a


class Model:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        # 26 Oct 2026: the second call (the builder's claims) gets a claims
        # answer because change 6 reads the builder's words in a call of
        # their own, after the design is judged
        if prompt.startswith("You reviewed a hardware design"):
            return {"claims_match_design": True, "false_claims": ""}
        return self.reply


def test_the_reviewer_sees_the_whole_design_and_how_it_was_built():
    model = Model(answer())
    S.review(wrapped(), call=model)
    # 26 Oct 2026: the builder's words moved from the first question to the
    # second because change 6 has the reviewer judge the code before it
    # reads them
    design, claims = model.prompts
    assert "module a; endmodule" in design and "module top" in design
    assert "a one-entry report buffer feeds the CRC checker" not in design
    assert "a one-entry report buffer feeds the CRC checker" in claims


def test_a_sensible_design_is_kept():
    rec = S.review(wrapped(), call=Model(answer()))
    assert rec["verdict"] == "NECESSARY"
    assert rec["sense_check"]["keep"] is True


def test_a_value_used_against_its_meaning_is_rejected():
    reply = answer(value_used_against_its_meaning=True,
                   why="zeros sent where the CRC check bits belong")
    rec = S.review(wrapped(), call=Model(reply))
    assert rec["verdict"] == "NONSENSE"
    assert "check bits" in rec["reason"]


def test_parts_wired_without_sense_or_unbuildable_designs_are_rejected():
    for bad in ({"parts_connected_sensibly": False},
                {"engineer_would_build_this": False}):
        rec = S.review(wrapped(), call=Model(answer(**bad)))
        assert rec["verdict"] == "NONSENSE", bad


def test_only_a_proved_design_is_reviewed():
    """Every kind of proved step since 18 Oct 2026 (combinations and copies
    before): test_extender_good_growth."""
    model = Model(answer(parts_connected_sensibly=False))
    for rec in (wrapped(verdict="DECORATIVE"), wrapped(verdict="NOT_PROVEN")):
        assert S.review(rec, call=model)["verdict"] == rec["verdict"]
    assert model.prompts == []


def test_a_failed_review_call_rejects_the_design_and_says_so():
    """It kept the design until 18 Oct 2026: the one kept that way in the
    18 Oct trial (its review was refused) was weak."""
    rec = S.review(wrapped(), call=Model(None))
    assert rec["verdict"] == "UNREVIEWED"
    assert rec["sense_check"]["error"]


def test_the_answer_field_is_not_named_reasoning():
    """A strict schema with a field named "reasoning" is refused before it
    reaches the model (distractor.py, 21 Sep 2026)."""
    assert "reasoning" not in S.SCHEMA["properties"]


def test_nonsense_is_a_rejection_not_a_repair():
    assert promote.route("NONSENSE") == "reject"


def test_agreement_with_the_hand_labels_is_counted(tmp_path):
    labels = {"c1": {"label": "good"}, "c2": {"label": "wrong"},
              "c3": {"label": "weak"}, "c4": {"label": "wrong"}}
    kept = {"c1": True, "c2": False, "c3": False, "c4": True}
    table = S.agreement(labels, kept)
    assert table == {"good": {"kept": 1, "rejected": 0},
                     "weak": {"kept": 0, "rejected": 1},
                     "wrong": {"kept": 1, "rejected": 1}}


def test_the_trial_labels_cover_all_28_designs():
    path = ROOT / "extender" / "logs" / "growth_trial_2026-10-17" / \
        "hand_labels.json"
    if not path.exists():                      # local data, not in git
        return
    labels = json.loads(path.read_text())
    assert len(labels) == 28
    assert {v["label"] for v in labels.values()} == {"good", "weak", "wrong"}


def test_the_review_is_off_until_switched_on():
    import inspect
    import batch
    assert batch.SENSE_CHECK is False
    assert "--sense-check" in inspect.getsource(batch.main)


def test_question_8_judges_only_the_change():
    """27 Oct 2026: asked about any input of the design, the reviewer had
    to turn down every step on a design that already ignores an input in
    some state (r0_054, r0_093 and r0_098 of the GPT-6.1 Sol run)."""
    text = " ".join(S.PROMPT.split())
    assert "any input of the design" not in text
    assert "can an item handed to the part the change adds or changes" in text
    assert "is not a loss of this change" in text


def test_a_review_may_think_as_long_as_the_builder():
    """27 Oct 2026: with GPT-6.1 Sol the limit counts its hidden thinking,
    and its design answers in the 26 Oct test used up to 16,700 tokens."""
    import extend
    assert S.MAX_TOKENS >= extend.MAX_TOKENS
