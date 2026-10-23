"""A retry says why the answer before it was sent back (22 Oct 2026, review
finding B1).

A step sent back for a format failure (STORAGE_TOO_DEEP, FACT_FORMAT,
PATCH_ERROR, ...) is retried, up to three attempts; the retry asked the
very same question, and the error, which says how to fix the answer, went
only to the log. promote.py has always said a format failure means "retry
with the lesson". The retry's question now ends with a short section:
"Your previous answer to this question was sent back: <error>".

The saved training example keeps the question WITHOUT that section (the
training builder trains on the saved messages as they are, and the
benchmark never asks such a question); its metadata keeps the lesson as
retry_lesson, so the question as sent can still be told exactly.
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
import distill_log                                              # noqa: E402
import llm_client                                               # noqa: E402
import promote                                                  # noqa: E402

USAGE = {"input": 1, "output": 2}
DEEP = ("the step adds storage of more than 16 entries: hist in module ctr "
        "(32 entries). Declare the depth as a parameter.")
SYSTEM_FN = "facts may not use system functions ($past): write each fact " \
    "about one cycle"


def design(id, lines=40):
    top = f"m_{id}"
    body = [f"module {top} (input wire clk, input wire en, "
            "output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(lines - 2)] \
        + ["endmodule"]
    return {"id": id, "generation": 0, "parent": None, "parent2": None,
            "top_module": top, "clock": "clk", "antecedents": [],
            "sanity_covers": [], "verilog": "\n".join(body),
            "property": ["a <= 4'd9"], "invariants": ["a <= 4'd9"],
            "metrics": {}}


PARENT, PARTNER = design("c0_1"), design("c0_2")


# --- the lesson a failed record gives ----------------------------------------

def test_the_lesson_is_the_error_of_the_answer_sent_back():
    rec = {"verdict": "STORAGE_TOO_DEEP", "error": DEEP, "usage": USAGE}
    assert batch.retry_lesson(rec) == DEEP


def test_a_cut_off_or_unreadable_answer_gets_a_plain_lesson():
    cut = batch.retry_lesson({"verdict": "TRUNCATED", "usage": USAGE})
    assert cut.startswith("it was cut off at the length limit")
    assert "write a shorter answer" in cut
    unread = batch.retry_lesson({"verdict": "UNPARSEABLE", "usage": USAGE})
    assert "could not be read as the JSON reply" in unread


def test_no_lesson_when_the_question_was_not_answered_or_was_refused():
    # the partner choice failed: the extension question was never asked
    assert batch.retry_lesson({"verdict": "TRUNCATED",
                               "partner_usage": USAGE}) is None
    assert batch.retry_lesson({"verdict": "REFUSED", "usage": USAGE}) is None


def test_a_long_error_is_cut_and_spacing_is_tidied():
    rec = {"verdict": "DISPOSITION_ERROR", "usage": USAGE,
           "error": "silent   deletion:\n" + "x" * 5000}
    lesson = batch.retry_lesson(rec)
    assert lesson.startswith("silent deletion: x")
    assert len(lesson) <= batch.LESSON_MAX + 4 and lesson.endswith("...")
    assert batch.retry_lesson({"verdict": "YOSYS_ERROR", "usage": USAGE,
                               "error": "child does not read in yosys"}) \
        == "child does not read in yosys."


def test_the_section_reads_as_the_review_asked():
    section = batch.retry_section(DEEP)
    assert section == ("\n\nYour previous answer to this question was sent "
                       f"back: {DEEP}\nFix that in a new, complete answer.")
    assert batch.retry_section(None) == ""


# --- the loop hands the lesson to the next attempt ---------------------------

def loop_with(verdicts):
    """Run one task through the loop; each attempt's record is the next of
    `verdicts` ((verdict, error) pairs). Returns the lesson each attempt
    was given."""
    given = []

    def executor(task, corpus):
        given.append(task.get("lesson"))
        verdict, error = verdicts[len(given) - 1]
        rec = {"verdict": verdict, "parent_id": task["parent_id"],
               "extension_id": f"e{task['task_id']}.{task['attempt']}",
               "usage": USAGE, "result": {}}
        if error:
            rec["error"] = error
        return rec
    batch.run_loop([dict(PARENT)], executor, max_calls=len(verdicts),
                   rng=random.Random(0), start_ids=["c0_1"])
    return given


def test_each_retry_gets_the_error_of_the_attempt_before_it():
    assert promote.route("STORAGE_TOO_DEEP") == "reformat"
    given = loop_with([("STORAGE_TOO_DEEP", DEEP),
                       ("FACT_FORMAT", SYSTEM_FN),
                       ("PATCH_ERROR", "anchor not found")])
    assert given == [None, DEEP, SYSTEM_FN + "."]


def test_a_refused_attempt_leaves_no_lesson_for_the_next():
    given = loop_with([("STORAGE_TOO_DEEP", DEEP), ("REFUSED", None),
                       ("PATCH_ERROR", "anchor not found")])
    assert given == [None, DEEP, None]


# --- the question the retry asks ---------------------------------------------

STRUCTURAL = {"new_state": "a retire queue", "coupling": "c",
              "induction_gap": "g", "why_parent_insufficient": "w",
              "patch": "@@ endmodule @@\n+ // grew",
              "invariants": ["a <= 4'd9", "q + a <= 5'd20"],
              "dispositions": []}
ANSWERS = {
    "structural": STRUCTURAL,
    "second": {"claim": "c", "shared_state": "s", "new_state": "n",
               "not_inductive_alone": "x", "shared_clause": "c",
               "patch": "@@ endmodule @@\n+ // p2",
               "invariants": ["a <= 4'd9", "q <= 4'd8"], "dispositions": []},
    "distractor": {"note": "a status counter",
                   "patch": "@@ endmodule @@\n+ // debug"},
    "compose": {"glue_scheme": "g", "induction_gap": "i",
                "why_parents_insufficient": "w", "top_module": "pair",
                "wrapper": "module pair; endmodule", "antecedents": [],
                "sanity_covers": [], "invariants": ["a <= 4'd9"]},
    "replicate": {"pool_scheme": "p", "why_aggregate_needed": "w",
                  "top_module": "pool", "wrapper": "module pool; endmodule",
                  "antecedents": [], "sanity_covers": [],
                  "invariants": ["a <= 4'd9"]},
}
GRADERS = {"structural": "grade_step4", "second": "grade_step4",
           "distractor": "grade_extension", "compose": "grade_compose",
           "replicate": "grade_replicate"}


def task(kind, **kw):
    t = {"task_id": 1, "attempt": 1, "ext_type": kind, "parent_id": "c0_1",
         "parent2_id": None, "move": None, "k": 1}
    if kind == "structural":
        t.update(move="STORAGE")
    if kind == "compose":
        t["parent2_id"] = "c0_2"
    if kind == "replicate":
        t["instances"] = 4
    t.update(kw)
    return t


class Model:
    """A fake model: each call gets the next answer; questions are kept."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, prompt, schema=None):
        self.asked.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, tuple):
            return answer[0], USAGE, answer[1]
        return answer, USAGE, "ok"


def sent_back(verdict="STORAGE_TOO_DEEP", error=DEEP):
    """A fake check that sends the answer back."""
    def grade(*args):
        record = next(a for a in args
                      if isinstance(a, dict) and "extension_id" in a)
        record.update(verdict=verdict, error=error)
        return record
    return grade


@pytest.fixture
def saved(monkeypatch, tmp_path):
    path = tmp_path / "distill_extensions.jsonl"
    monkeypatch.setattr(batch, "DISTILL_LOG", path)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    return lambda: distill_log.read(path) if path.exists() else []


def answering(monkeypatch, model):
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "distractor_call", model)
    return model


@pytest.mark.parametrize("kind", sorted(ANSWERS))
def test_a_retry_question_ends_with_the_lesson(saved, monkeypatch, kind):
    ask = answering(monkeypatch, Model(ANSWERS[kind], ANSWERS[kind]))
    monkeypatch.setattr(batch, GRADERS[kind], sent_back())
    first = batch._real_executor(task(kind), [PARENT, PARTNER])
    again = batch._real_executor(task(kind, attempt=2, lesson=DEEP),
                                 [PARENT, PARTNER])
    section = batch.retry_section(DEEP)
    assert not ask.asked[0].endswith(section)
    assert "was sent back" not in ask.asked[0]
    assert ask.asked[1] == ask.asked[0] + section
    assert "retry_lesson" not in first
    assert again["retry_lesson"] == DEEP


@pytest.mark.parametrize("kind", sorted(ANSWERS))
def test_the_saved_example_keeps_the_question_without_the_lesson(
        saved, monkeypatch, kind):
    ask = answering(monkeypatch, Model(ANSWERS[kind]))
    monkeypatch.setattr(batch, GRADERS[kind], sent_back())
    batch._real_executor(task(kind, attempt=2, lesson=DEEP),
                         [PARENT, PARTNER])
    [ex] = saved()
    question = ex["metadata"]["messages"][-1]
    assert question["role"] == "user"
    assert question["content"] + batch.retry_section(DEEP) == ask.asked[0]
    assert "was sent back" not in question["content"]
    assert ex["metadata"]["retry_lesson"] == DEEP
    assert ex["metadata"]["batch_attempt"] == 2


def test_a_first_attempts_example_has_no_lesson(saved, monkeypatch):
    answering(monkeypatch, Model(STRUCTURAL))
    monkeypatch.setattr(batch, "grade_step4", sent_back())
    batch._real_executor(task("structural"), [PARENT])
    [ex] = saved()
    assert "retry_lesson" not in ex["metadata"]


def test_a_compose_retry_skips_the_partner_and_asks_with_the_lesson(
        saved, monkeypatch):
    """The partner chosen on the first attempt is kept, so the retry asks
    only the combination question, with the lesson."""
    plan = {"choice": "c0_2", "system": "s", "connection": "c"}
    ask = answering(monkeypatch, Model(plan, ANSWERS["compose"],
                                       ANSWERS["compose"]))
    monkeypatch.setattr(batch, "grade_compose", sent_back())
    t = task("compose", parent2_id=None, candidates=["c0_2"])
    first = batch._real_executor(t, [PARENT, PARTNER])
    t.update(attempt=2, lesson=batch.retry_lesson(first))
    batch._real_executor(t, [PARENT, PARTNER])
    assert len(ask.asked) == 3
    assert ask.asked[2] == ask.asked[1] + batch.retry_section(DEEP)


# --- through the real client --------------------------------------------------

@pytest.fixture
def api(monkeypatch):
    """The real client, with only the request to Anthropic faked."""
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    monkeypatch.setattr(llm_client, "_gate", None)
    monkeypatch.setattr(llm_client, "_call_log_path", None)
    sent = []

    def anthropic(kwargs):
        sent.append(kwargs)
        text = json.dumps(STRUCTURAL)
        return text, {"input": 10, "output": 20}, "ok", text
    monkeypatch.setattr(llm_client, "_call_anthropic", anthropic)
    return sent


def test_the_request_carries_the_lesson_and_the_example_does_not(
        saved, api, monkeypatch):
    monkeypatch.setattr(batch, "grade_step4", sent_back())
    batch._real_executor(task("structural", attempt=2, lesson=DEEP),
                         [PARENT])
    [request] = api
    sent_question = request["messages"][-1]["content"]
    assert sent_question.endswith(batch.retry_section(DEEP))
    [ex] = saved()
    assert ex["metadata"]["messages"][0] == {"role": "system",
                                             "content": request["system"]}
    assert ex["metadata"]["messages"][-1]["content"] + \
        batch.retry_section(DEEP) == sent_question


def test_without_section_leaves_other_questions_alone():
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "the question"}]
    assert batch.without_section(messages, "\n\nnot there") == messages
    assert batch.without_section(messages, "") == messages
    cut = batch.without_section(
        [{"role": "user", "content": "q" + batch.retry_section(DEEP)}],
        batch.retry_section(DEEP))
    assert cut == [{"role": "user", "content": "q"}]


# --- a whole loop, with the real executor ------------------------------------

def test_a_run_retries_with_the_lesson_and_saves_three_examples(
        saved, monkeypatch):
    ask = answering(monkeypatch, Model(*[STRUCTURAL] * 3))
    checks = iter([("STORAGE_TOO_DEEP", DEEP), ("FACT_FORMAT", SYSTEM_FN),
                   ("NOT_PROVEN", None)])

    def grade(parent, ext_type, out, record):
        verdict, error = next(checks)
        record.update(verdict=verdict, invariants=out["invariants"])
        if error:
            record["error"] = error
        else:
            record["reason"] = "with-invariants grade is NOT_INDUCTIVE"
    monkeypatch.setattr(batch, "grade_step4", grade)
    monkeypatch.setattr(batch, "MOVE_WEIGHTS", {"STORAGE": 1})
    monkeypatch.setattr(batch, "DISTRACTOR_RATE", 0)
    monkeypatch.setattr(batch, "MULTIPLIER_RATE", 0)
    batch.run_loop([dict(PARENT)], batch._real_executor, max_calls=3,
                   rng=random.Random(0), start_ids=["c0_1"])
    assert len(ask.asked) == 3
    assert ask.asked[1] == ask.asked[0] + batch.retry_section(DEEP)
    assert ask.asked[2] == ask.asked[0] + batch.retry_section(
        SYSTEM_FN + ".")
    examples = saved()
    assert [e["metadata"].get("retry_lesson") for e in examples] == \
        [None, DEEP, SYSTEM_FN + "."]
    # every saved question is the plain one
    assert {e["metadata"]["messages"][-1]["content"] for e in examples} == \
        {ask.asked[0]}
