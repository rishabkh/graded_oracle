"""Every answer to an extension question is saved as a training example
(21 Oct 2026, Rishab: "make all 1-8 changes", item 8a).

initiator/run.py saves each generator call as an example in Formal Disco's
shape (distill_log.py); the extender's answers were kept only in its own
log, which the training builder does not read. Each answered extension
question (a structural step, a second property, a debug block, a
combination, a set of copies) is now one `extend` example in
distill_extensions.jsonl next to the extension log: the question exactly
as sent, the answer exactly as the model wrote it, and the step's final
verdict (after the second check of a step that ran out of time, the shape
rule and the design checks), so distill_log can say success, fail or
error. Its metadata names the grown design (child_verilog), the rows it
grew from (parent_ids) and the extension id, which the training builder
reads to hold back test rows. The partner choice and the reviewer's calls
are not extension answers and are not saved.

The model is always faked: either the extender's own call, or only the
last step of the client, so the real request is built and noted with no
network call. The checker is faked too."""
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import distill_log                                              # noqa: E402
import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
import llm_client                                               # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict,        # noqa: E402
                          RunEvidence, Tier, TripleResult)

USAGE = {"input": 1, "output": 2}


def design(id, lines=40, gen=0, parent=None, parent2=None,
           facts=("a <= 4'd9",)):
    """A design of `lines` lines, no two alike, with its own module name."""
    top = f"m_{id}"
    body = [f"module {top} (input wire clk, input wire en, "
            "output reg [3:0] a);"] + \
        [f"    localparam [15:0] P{i} = 16'd{i};" for i in range(lines - 2)] \
        + ["endmodule"]
    return {"id": id, "generation": gen, "parent": parent,
            "parent2": parent2, "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": "\n".join(body), "property": ["a <= 4'd9"],
            "invariants": list(facts), "metrics": {}}


PARENT, PARTNER = design("c0_1"), design("c0_2")

STRUCTURAL = {"new_state": "a retire queue", "coupling": "c",
              "induction_gap": "g", "why_parent_insufficient": "w",
              "patch": "@@ endmodule @@\n+ // grew",
              "invariants": ["a <= 4'd9", "q + a <= 5'd20"],
              "dispositions": []}
ANSWERS = {
    "structural": STRUCTURAL,
    "second": {"claim": "the queue never overflows", "shared_state": "s",
               "new_state": "n", "not_inductive_alone": "x",
               "shared_clause": "c", "patch": "@@ endmodule @@\n+ // p2",
               "invariants": ["a <= 4'd9", "q <= 4'd8"],
               "dispositions": []},
    "distractor": {"note": "a status counter nobody reads",
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
GROWN = "module grown; endmodule"


def task(kind, **kw):
    t = {"task_id": 1, "attempt": 1, "ext_type": kind, "parent_id": "c0_1",
         "parent2_id": None, "move": None, "k": 1}
    if kind == "structural":
        t.update(move="STAGE", k=4)
    if kind == "compose":
        t["parent2_id"] = "c0_2"
    if kind == "replicate":
        t["instances"] = 4
    t.update(kw)
    return t


class Model:
    """A fake model: each call gets the next answer, an answer or
    (None, how the call ended); the questions asked are kept."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, prompt, schema=None):
        self.asked.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, tuple):
            return answer[0], USAGE, answer[1]
        return answer, USAGE, "ok"


def checked(verdict="NOT_PROVEN", child=GROWN, **fields):
    """A fake check of an answer: the record gets `verdict`, the grown
    design and any other `fields`."""
    def grade(*args):
        record = next(a for a in args
                      if isinstance(a, dict) and "extension_id" in a)
        record.update(verdict=verdict, child_verilog=child,
                      reason=("with-invariants grade is NOT_INDUCTIVE, "
                              "not PROVEN"), **fields)
        return record
    return grade


@pytest.fixture
def saved(monkeypatch, tmp_path):
    """Examples go to a temporary file and the extension log line is
    dropped; gives a reader of the examples saved so far."""
    path = tmp_path / "distill_extensions.jsonl"
    monkeypatch.setattr(batch, "DISTILL_LOG", path)
    monkeypatch.setattr(batch, "dump", lambda r: None)
    return lambda: distill_log.read(path) if path.exists() else []


def answering(monkeypatch, model):
    monkeypatch.setattr(batch, "call_model", model)
    monkeypatch.setattr(batch, "distractor_call", model)
    return model


# --- what one example holds ---------------------------------------------------

@pytest.mark.parametrize("kind", sorted(ANSWERS))
def test_every_kind_of_answer_is_saved_once(saved, monkeypatch, kind):
    ask = answering(monkeypatch, Model(ANSWERS[kind]))
    monkeypatch.setattr(batch, GRADERS[kind], checked())
    rec = batch._real_executor(task(kind), [PARENT, PARTNER])
    [ex] = saved()
    assert ex["prompt"] == "extend"
    assert ex["arguments"] == {
        "parent_id": "c0_1",
        "parent2_id": "c0_2" if kind == "compose" else None,
        "ext_type": kind, "move": rec["move"], "k": rec["k"]}
    assert ex["response"] == json.dumps(ANSWERS[kind])
    question = {"role": "user", "content": ask.asked[0]}
    if kind == "distractor":          # its question has no system message
        assert ex["metadata"]["messages"] == [question]
    else:
        assert ex["metadata"]["messages"] == [
            {"role": "system", "content": extend.SYSTEM_PROMPT}, question]
    assert ex["metadata"]["verdict"] == rec["verdict"] == "NOT_PROVEN"
    assert ex["outcome"] == "fail"
    # what the training builder reads to hold back test rows
    meta = ex["metadata"]
    assert meta["child_verilog"] == GROWN
    assert meta["parent_ids"] == (["c0_1", "c0_2"] if kind == "compose"
                                  else ["c0_1"])
    assert meta["extension_id"] == rec["extension_id"]


def test_the_arguments_are_the_steps_own_inputs_only(saved, monkeypatch):
    answering(monkeypatch, Model(STRUCTURAL))
    monkeypatch.setattr(batch, "grade_step4", checked())
    batch._real_executor(task("structural", move="SUBUNIT", k=1), [PARENT])
    [ex] = saved()
    assert ex["arguments"] == {"parent_id": "c0_1", "parent2_id": None,
                               "ext_type": "structural", "move": "SUBUNIT",
                               "k": 1}


def test_the_extension_log_line_does_not_grow(saved, monkeypatch):
    """The question is long; it goes in the example, not on the record."""
    answering(monkeypatch, Model(STRUCTURAL))
    monkeypatch.setattr(batch, "grade_step4", checked())
    rec = batch._real_executor(task("structural"), [PARENT])
    assert not {"messages", "response", "kept"} & set(rec)


def test_a_step_with_no_grown_design_saves_none(saved, monkeypatch):
    answering(monkeypatch, Model(STRUCTURAL))

    def bad_patch(parent, ext_type, out, record):
        record.update(verdict="PATCH_ERROR", error="anchor not found")
    monkeypatch.setattr(batch, "grade_step4", bad_patch)
    batch._real_executor(task("structural"), [PARENT])
    [ex] = saved()
    assert ex["metadata"]["child_verilog"] is None
    assert ex["metadata"]["error"] == "anchor not found"
    assert ex["outcome"] == "fail"


# --- which calls are saved -----------------------------------------------------

def test_the_partner_choice_is_not_saved(saved, monkeypatch):
    plan = {"choice": "c0_2", "system": "a memory with a host interface",
            "connection": "the memory's read data feeds the interface"}
    ask = answering(monkeypatch, Model(plan, ANSWERS["compose"]))
    monkeypatch.setattr(batch, "grade_compose", checked())
    rec = batch._real_executor(
        task("compose", parent2_id=None, candidates=["c0_2"]),
        [PARENT, PARTNER])
    assert len(ask.asked) == 2 and rec["parent2_id"] == "c0_2"
    [ex] = saved()
    assert ex["metadata"]["messages"][-1]["content"] == ask.asked[1]
    assert ex["response"] == json.dumps(ANSWERS["compose"])
    assert ex["arguments"]["parent2_id"] == "c0_2"
    assert ex["metadata"]["parent_ids"] == ["c0_1", "c0_2"]


@pytest.mark.parametrize("partner_answer", [
    {"choice": "none", "system": "nothing here fits", "connection": ""},
    (None, "refusal")])
def test_no_combination_question_no_example(saved, monkeypatch,
                                            partner_answer):
    answering(monkeypatch, Model(partner_answer))
    rec = batch._real_executor(
        task("compose", parent2_id=None, candidates=["c0_2"]),
        [PARENT, PARTNER])
    assert rec["verdict"] in ("NO_PARTNER", "REFUSED")
    assert saved() == []


def test_the_reviewers_call_is_not_saved(saved, api, monkeypatch):
    """The real reviewer asks its own question after the step; only the
    step's answer is an example, and the reviewer's rejection is its
    final verdict."""
    sent, replies = api
    review = {"parts_connected_sensibly": True,
              "value_used_against_its_meaning": False,
              "new_logic_only_copies_existing_state": False,
              "results_of_parts_used": True,
              "new_part_has_its_own_job": True, "breaks_what_it_did": True,
              "engineer_would_build_this": False,
              "why": "it drops valid input data"}
    replies.extend([(json.dumps(STRUCTURAL), "ok"),
                    (json.dumps(review), "ok")])
    monkeypatch.setattr(batch, "grade_step4", checked(
        verdict="NECESSARY", invariants=STRUCTURAL["invariants"]))
    monkeypatch.setattr(batch, "SENSE_CHECK", True)
    rec = batch._real_executor(task("structural"), [PARENT])
    assert len(sent) == 2                       # the step, then the review
    [ex] = saved()
    assert ex["response"] == json.dumps(STRUCTURAL)
    assert ex["metadata"]["messages"][-1] == sent[0]["messages"][-1]
    assert ex["metadata"]["verdict"] == rec["verdict"] == "NONSENSE"
    assert ex["outcome"] == "fail"


def test_a_refused_question_is_saved_as_an_error(saved, monkeypatch):
    answering(monkeypatch, Model((None, "refusal")))
    rec = batch._real_executor(task("structural"), [PARENT])
    [ex] = saved()
    assert rec["verdict"] == ex["metadata"]["verdict"] == "REFUSED"
    assert ex["response"] is None
    assert ex["outcome"] == "error"
    assert ex["metadata"]["messages"][-1]["role"] == "user"
    assert ex["metadata"]["child_verilog"] is None
    assert ex["metadata"]["parent_ids"] == ["c0_1"]


def test_without_a_place_for_examples_nothing_is_saved(monkeypatch):
    """A test or a tool calling the executor directly saves nothing; only
    a run (main) says where examples go."""
    monkeypatch.setattr(batch, "DISTILL_LOG", None)
    monkeypatch.setattr(batch, "dump", lambda r: None)

    def must_not_save(*a, **k):
        raise AssertionError("an example was saved")
    monkeypatch.setattr(distill_log, "record", must_not_save)
    answering(monkeypatch, Model(STRUCTURAL))
    monkeypatch.setattr(batch, "grade_step4", checked())
    assert batch._real_executor(task("structural"), [PARENT])["verdict"] \
        == "NOT_PROVEN"


# --- the final verdict ---------------------------------------------------------

def test_a_proved_step_no_harder_than_its_parent_is_saved_as_rejected(
        saved, monkeypatch):
    """The checker said NECESSARY, but the step needs no new kind of fact;
    the example says what the run did with it."""
    answering(monkeypatch, Model(STRUCTURAL))
    monkeypatch.setattr(batch, "grade_step4", checked(
        verdict="NECESSARY", invariants=["a <= 4'd8"]))
    rec = batch._real_executor(task("structural"), [PARENT])
    [ex] = saved()
    assert ex["metadata"]["verdict"] == rec["verdict"] == "NO_NEW_SHAPE"
    assert ex["outcome"] == "fail"


def leg(tier, rc):
    run = RunEvidence(mode="prove", rc=rc, depth=20, engine="smtbmc yices",
                      duration_s=1.0, workdir=Path("job"), log_excerpt="")
    return GradeResult(tier=tier, reason=tier.name.lower(), runs=[run])


RAN_OUT = TripleResult(NecessityVerdict.INCONCLUSIVE,
                       "without-invariants grade is TIMEOUT - necessity not "
                       "established", with_invariants=leg(Tier.PROVEN, 0),
                       without_invariants=leg(Tier.TIMEOUT, 8))
PROVED = TripleResult(NecessityVerdict.NECESSARY,
                      "strengthening is load-bearing: PROVEN with "
                      "invariants, NOT_INDUCTIVE without",
                      with_invariants=leg(Tier.PROVEN, 0),
                      without_invariants=leg(Tier.NOT_INDUCTIVE, 4))

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

LAST_COUNT = """\
@@ reg [3:0] lim; @@
+    reg [3:0] last;
+    initial last = 4'd0;
@@ cnt <= (cnt == lim) ? 4'd0 : cnt + 4'd1; @@
+    always @(posedge clk)
+        if (en) last <= cnt;"""


def test_the_verdict_saved_is_the_one_after_the_second_check(saved,
                                                             monkeypatch):
    """The real step check, with the checker faked: the first check runs
    out of time, the second (longer) one proves the step."""
    counter = {"id": "c0_7", "top_module": "ctr", "clock": "clk",
               "antecedents": [], "sanity_covers": [], "verilog": CTR,
               "property": ["cnt <= 4'd9"], "invariants": ["lim == 4'd9"]}
    answer = {"patch": LAST_COUNT,
              "invariants": ["lim == 4'd9", "last <= 4'd9"],
              "dispositions": [{"clause": "lim == 4'd9", "status": "kept",
                                "replaced_by": [], "reason": ""}],
              "new_state": "the last count", "coupling": "c",
              "induction_gap": "g", "why_parent_insufficient": "w"}
    for module in (extend, distractor):
        monkeypatch.setattr(module, "state_bits", lambda v, top: len(v))
    answers = [RAN_OUT, PROVED]
    monkeypatch.setattr(extend, "grade_triple_generated",
                        lambda payload, **kw: answers.pop(0))
    answering(monkeypatch, Model(answer))
    rec = batch._real_executor(task("structural", parent_id="c0_7",
                                    move="STORAGE", k=1), [counter])
    assert rec["first_result"]["verdict"] == "INCONCLUSIVE"
    [ex] = saved()
    assert ex["metadata"]["verdict"] == rec["verdict"] == "NECESSARY"
    assert ex["metadata"]["reason"] == PROVED.reason
    assert ex["metadata"]["rechecked"] is True
    assert ex["outcome"] == "success"
    assert ex["metadata"]["child_verilog"] == rec["child_verilog"]
    assert "last <= cnt" in ex["metadata"]["child_verilog"]


# --- the rows a design grew from -----------------------------------------------

LINEAGE = [design("c0_1"), design("c0_2"),
           design("c1_000", gen=1, parent="c0_1"),
           design("c2_000", gen=2, parent="c1_000", parent2="c0_2")]


def test_every_row_a_design_grew_from_is_named_nearest_first():
    by_id = {r["id"]: r for r in LINEAGE}
    assert batch.grown_from(["c2_000", None], by_id) == \
        ["c2_000", "c1_000", "c0_2", "c0_1"]
    assert batch.grown_from(["c1_000", "c0_2"], by_id) == \
        ["c1_000", "c0_2", "c0_1"]
    # a row whose parent is not in the corpus is still named
    assert batch.grown_from(["c9_000"], by_id) == ["c9_000"]
    looped = {"x": {"id": "x", "parent": "y"}, "y": {"id": "y", "parent": "x"}}
    assert batch.grown_from(["x"], looped) == ["x", "y"]


def test_a_step_on_a_grown_design_names_its_whole_lineage(saved,
                                                          monkeypatch):
    """A design grown from a held-back row's grandchild still carries most
    of its text and facts, so the test-row hold-back needs every row it
    grew from, not only its own parent."""
    answering(monkeypatch, Model(STRUCTURAL))
    monkeypatch.setattr(batch, "grade_step4", checked())
    batch._real_executor(task("structural", parent_id="c2_000"), LINEAGE)
    [ex] = saved()
    assert ex["arguments"]["parent_id"] == "c2_000"
    assert ex["metadata"]["parent_ids"] == ["c2_000", "c1_000", "c0_2",
                                            "c0_1"]


# --- through the real client, with only the request faked ----------------------

@pytest.fixture
def api(monkeypatch):
    """The extender's real calls and the real client: only the request to
    Anthropic is faked, so the request is built and noted as in a run.
    Replies are (text, how the call ended) or an exception, in turn; every
    request sent is kept."""
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    monkeypatch.setattr(llm_client, "_gate", None)
    monkeypatch.setattr(llm_client, "_call_log_path", None)
    sent, replies = [], []

    def anthropic(kwargs):
        sent.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text, stop = reply
        usage = {"input": 10, "output": 20}
        if stop == "refusal":
            return None, usage, "refusal", None
        if stop == "length":
            return None, usage, "length", text
        return text, usage, "ok", text
    monkeypatch.setattr(llm_client, "_call_anthropic", anthropic)
    return sent, replies


# spaced as no JSON writer would space it
RAW = ('{"new_state": "a retire queue",   "coupling": "c",\n'
       '  "induction_gap": "g", "why_parent_insufficient": "w",\n'
       '  "patch": "@@ endmodule @@\\n+ // grew",\n'
       '  "invariants": ["a <= 4\'d9", "q + a <= 5\'d20"],\n'
       '  "dispositions": []}')


def test_the_answer_is_saved_exactly_as_the_model_wrote_it(saved, api,
                                                            monkeypatch):
    sent, replies = api
    replies.append((RAW, "ok"))
    monkeypatch.setattr(batch, "grade_step4", checked())
    batch._real_executor(task("structural"), [PARENT])
    [ex] = saved()
    assert ex["response"] == RAW
    assert ex["response"] != json.dumps(json.loads(RAW))     # not rewritten
    # the messages exactly as the request carried them
    [request] = sent
    assert request["system"] == extend.SYSTEM_PROMPT
    assert ex["metadata"]["messages"] == \
        [{"role": "system", "content": request["system"]}] \
        + request["messages"]
    assert ex["metadata"]["call"]["usage"] == {"input": 10, "output": 20}
    assert ex["metadata"]["call"]["stop"] == "ok"


def test_a_debug_block_question_is_saved_as_sent_without_a_system_message(
        saved, api, monkeypatch):
    sent, replies = api
    replies.append((json.dumps(ANSWERS["distractor"]), "ok"))
    monkeypatch.setattr(batch, "grade_extension", checked())
    batch._real_executor(task("distractor"), [PARENT])
    [ex] = saved()
    [request] = sent
    assert "system" not in request
    assert ex["metadata"]["messages"] == request["messages"]


def test_a_cut_off_answer_is_saved_as_written_and_counts_as_an_error(
        saved, api):
    _, replies = api
    replies.append(('{"new_state": "a retire q', "length"))
    rec = batch._real_executor(task("structural"), [PARENT])
    [ex] = saved()
    assert rec["verdict"] == ex["metadata"]["verdict"] == "TRUNCATED"
    assert ex["response"] == '{"new_state": "a retire q'
    assert ex["outcome"] == "error"


def test_a_refusal_is_saved_with_no_answer(saved, api):
    _, replies = api
    replies.append((None, "refusal"))
    rec = batch._real_executor(task("structural"), [PARENT])
    [ex] = saved()
    assert rec["verdict"] == "REFUSED"
    assert ex["response"] is None and ex["outcome"] == "error"


def test_an_answer_that_could_not_be_read_is_still_saved(saved, api):
    _, replies = api
    replies.append(("this is not JSON at all", "ok"))
    rec = batch._real_executor(task("structural"), [PARENT])
    assert rec["verdict"] == "ERROR"
    [ex] = saved()
    assert ex["response"] == "this is not JSON at all"
    assert ex["outcome"] == "error"
    assert "JSONDecodeError" in ex["metadata"]["error"]


def test_a_failed_request_has_no_answer_to_save(saved, api):
    _, replies = api
    replies.append(RuntimeError("overloaded"))
    rec = batch._real_executor(task("structural"), [PARENT])
    assert rec["verdict"] == "ERROR" and "overloaded" in rec["error"]
    assert saved() == []


def test_workers_do_not_mix_up_their_answers(saved, api, monkeypatch):
    """Each worker reads its own call's note (llm_client.last_call is kept
    per thread), even when the other worker's note was written after its
    own and before it read it back."""
    sent, _ = api

    def anthropic(kwargs):
        sent.append(kwargs)
        user = kwargs["messages"][0]["content"]
        which = "c0_1" if "module m_c0_1 " in user else "c0_2"
        text = json.dumps(dict(STRUCTURAL, new_state=f"grown from {which}"))
        return text, {"input": 1, "output": 1}, "ok", text
    monkeypatch.setattr(llm_client, "_call_anthropic", anthropic)
    # both notes are written before either worker goes on to read its own
    both_noted = threading.Barrier(2, timeout=10)
    note = llm_client._record_call

    def note_then_wait(entry, t0):
        note(entry, t0)
        both_noted.wait()
    monkeypatch.setattr(llm_client, "_record_call", note_then_wait)
    monkeypatch.setattr(batch, "grade_step4", checked())
    threads = [threading.Thread(target=batch._real_executor,
                                args=(task("structural", parent_id=p,
                                           task_id=i), [PARENT, PARTNER]))
               for i, p in enumerate(["c0_1", "c0_2"], 1)]
    [t.start() for t in threads]
    [t.join(timeout=20) for t in threads]
    examples = saved()
    assert len(examples) == 2 and len(sent) == 2
    for ex in examples:
        parent = ex["arguments"]["parent_id"]
        assert json.loads(ex["response"])["new_state"] == \
            f"grown from {parent}"
        assert f"module m_{parent} " in \
            ex["metadata"]["messages"][-1]["content"]


# --- where a run saves them ----------------------------------------------------

@pytest.fixture
def logs_restored(monkeypatch, tmp_path):
    """main() moves the log paths for its run; put them back after."""
    monkeypatch.setattr(distractor, "OUT_LOG", tmp_path / "extensions.jsonl")
    monkeypatch.setattr(batch, "QUEUE_LOG", tmp_path / "batch_queue.jsonl")
    monkeypatch.setattr(batch, "FIXER_QUEUE", tmp_path / "fixer_queue.jsonl")
    corpus = tmp_path / "corpus_trial.jsonl"
    corpus.write_text(json.dumps(PARENT) + "\n")
    return corpus


def watch(monkeypatch):
    seen = {}

    def fake_loop(rows, executor, **kw):
        seen["examples"] = batch.DISTILL_LOG
        return []
    monkeypatch.setattr(batch, "run_loop", fake_loop)
    return seen


def test_a_run_saves_examples_in_its_log_folder(logs_restored, monkeypatch,
                                                tmp_path, capsys):
    seen = watch(monkeypatch)
    logs = tmp_path / "logs"
    batch.main(["--max-calls", "1", "--corpus", str(logs_restored),
                "--log-dir", str(logs)])
    assert seen["examples"] == logs / "distill_extensions.jsonl"
    assert batch.DISTILL_LOG is None               # put back after the run
    assert f"training examples: {logs / 'distill_extensions.jsonl'}" in \
        capsys.readouterr().out


def test_without_a_log_folder_examples_sit_next_to_the_extension_log(
        logs_restored, monkeypatch, tmp_path):
    seen = watch(monkeypatch)
    monkeypatch.setattr(batch, "CORPUS", logs_restored)   # the default one
    batch.main(["--max-calls", "1"])
    assert seen["examples"] == tmp_path / "distill_extensions.jsonl"
    assert batch.distill_path(tmp_path / "x" / "extensions.jsonl") == \
        tmp_path / "x" / "distill_extensions.jsonl"


def test_a_whole_run_saves_one_example_per_answer(logs_restored,
                                                  monkeypatch, tmp_path):
    """main() with the real loop and executor; the model and the checker
    are faked, and every step fails, so the corpus is not touched."""
    monkeypatch.setattr(batch, "call_model", Model(*[STRUCTURAL] * 3))
    monkeypatch.setattr(batch, "distractor_call",
                        Model(*[ANSWERS["distractor"]] * 3))
    monkeypatch.setattr(batch, "grade_step4", checked())
    monkeypatch.setattr(batch, "grade_extension", checked())
    logs = tmp_path / "logs"
    before = logs_restored.read_text()
    batch.main(["--max-calls", "2", "--corpus", str(logs_restored),
                "--log-dir", str(logs), "--seed", "3"])
    records = [json.loads(l) for l in
               (logs / "extensions.jsonl").read_text().splitlines()]
    examples = distill_log.read(logs / "distill_extensions.jsonl")
    assert len(records) == len(examples) == 2
    for rec, ex in zip(records, examples):
        assert ex["metadata"]["extension_id"] == rec["extension_id"]
        assert ex["metadata"]["verdict"] == rec["verdict"] == "NOT_PROVEN"
        assert "messages" not in rec
    assert logs_restored.read_text() == before
