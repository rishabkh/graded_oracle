"""The streams question gives the checker's own stream rules (26 Oct 2026,
Rishab approved: "fix these things"). The first real answers (GPT-6.1 Sol,
3 of 3) were sent back for rules the question never gave: two named the
reset as the flush, and one combination declared 4 streams.
Reads text only: no model, no checker."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import extend                                                   # noqa: E402
import nothing_lost                                             # noqa: E402


def squash(text):
    return " ".join(text.split())


def test_the_question_gives_the_checkers_limits():
    text = squash(extend.STREAMS_TEXT)
    assert f"at most {nothing_lost.MAX_STREAMS} streams" in text
    assert f"at most {nothing_lost.MAX_INPUTS} inputs" in text


def test_the_question_says_what_a_flush_may_not_read():
    """nothing_lost.declaration_error turns down a flush that reads the
    reset, the enable or the stream's own offer or ready signals."""
    text = squash(extend.STREAMS_TEXT)
    for words in ("an input port that empties the stream on purpose",
                  "Never the reset", "Never the enable",
                  "never the stream's own offer or ready signals"):
        assert words in text, words


def test_every_question_with_streams_carries_the_rules():
    for kind in ("STORAGE", "PEER", "SUBUNIT", "compose"):
        text = squash(extend.streams_text(kind))
        assert f"at most {nothing_lost.MAX_STREAMS} streams" in text, kind
        assert "Never the reset" in text, kind
        assert "start the stream where items come into that logic" in text, \
            kind


def test_the_enable_goes_in_only_when_the_design_waits_for_it():
    """27 Oct 2026: r0_062 takes items whatever its enable, and offer
    "ce && in_valid", the old advice word for word, was refused as
    counting wrong. The lesson after a lost item says the same."""
    text = squash(extend.STREAMS_TEXT)
    for words in ("only when the design takes items only while the enable "
                  "is high", 'the valid alone ("in_valid")',
                  "only when taking an item waits for the enable"):
        assert words in text, words
    streams = [{"name": "requests", "inputs": [{"offer": "in_valid",
                                                "ready": "in_ready"}],
                "delivered": "pop", "held": "used", "flush": ""}]
    lost = nothing_lost.lesson({"result": "lost", "stream": 0, "step": 3},
                               streams)
    assert "only when the design takes items only while it is high" in lost


def test_rule_14_is_told_with_what_to_do_for_a_new_logic_sender():
    """27 Oct 2026: GPT-6.1 Sol's journal and state-machine streams, sent
    by the step's own new logic, were refused by rule 14 of
    nothing_lost.declaration_error, which the question never gave."""
    text = squash(extend.STREAMS_TEXT)
    assert ("The offer may not read state your step adds that delivered "
            "or held also read") in text
    assert ("start the stream where items come into that logic, or leave "
            "that stream out") in text
