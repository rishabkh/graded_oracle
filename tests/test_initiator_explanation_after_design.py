"""The initiator's answer form since 26 Oct 2026: the design first, then
the fake state and a short explanation of the finished answer.

Until then the question said "the reasoning fields come FIRST. Work out the
state S ... BEFORE writing the design", and the answer's first field was
`cti_reasoning`. Opus 5.5 stopped 7 of the 41 calls of the 25 Oct run, all
with stop category "reasoning_extraction". Anthropic's refusals page says
that category means the request "asks the model to reproduce its internal
reasoning in the response text", names a scratchpad the model fills in
before it answers and a `reasoning` field in JSON output as its usual
causes, and still allows "a short explanation". Opus 5.5 thinks before it
answers whatever the question says, so nothing is lost by moving it."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

from prompts import SYSTEM_PROMPT, prompt_version                  # noqa: E402
from schema import TRIPLE_SCHEMA                                    # noqa: E402

FIELDS = list(TRIPLE_SCHEMA["properties"])


def test_the_answer_form_ends_with_the_fake_state_and_the_explanation():
    assert FIELDS[-2:] == ["cti_state", "cti_explanation"]
    assert FIELDS.index("verilog") < FIELDS.index("cti_state")
    assert TRIPLE_SCHEMA["required"] == FIELDS


def test_no_field_of_the_answer_form_is_named_for_reasoning():
    assert not [f for f in FIELDS
                if any(w in f for w in ("reasoning", "thinking", "trace"))]


def test_the_question_no_longer_asks_for_working_out_in_the_answer():
    text = SYSTEM_PROMPT.lower()
    assert "reasoning" not in text
    assert "come first" not in text


def test_the_question_asks_for_a_short_explanation_after_the_design():
    flat = " ".join(SYSTEM_PROMPT.split())
    assert "- `cti_explanation`: a short explanation of the finished answer" \
        in flat
    notes = [flat.index(f"- `{f}`") for f in
             ("invariants", "verilog", "cti_state", "cti_explanation")]
    assert notes == sorted(notes)


def test_the_example_answers_follow_the_answer_form():
    """The question shows one of these as its worked example, field for
    field in this order."""
    exemplars = json.loads((ROOT / "initiator" / "exemplars.json").read_text())
    assert exemplars
    for name, ex in exemplars.items():
        assert list(ex) == FIELDS, name
        assert ex["cti_explanation"].strip(), name


def test_the_stamp_names_the_new_answer_form():
    assert prompt_version().endswith("+explanation-after-design")
    old = ("Field order is deliberate: the reasoning fields come FIRST.\n"
           "- `cti_reasoning`: one or two sentences: why this state satisfies P")
    assert prompt_version(old) == "nocap"
