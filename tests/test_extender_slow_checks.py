"""Steps lost to the checker's time limit get a second check, and a run's
proofs stay next to its logs (21 Oct 2026, Rishab: "make all 1-8 changes").

  2.  In the 21 Oct real-systems trial 9 of the 40 steps were lost because
      the checker hit its 300 s limit. Checked again by hand with 1200 s,
      2 of them proved (their proof without facts took 539 s and 625 s).
      A step whose verdict came from the time limit is now checked once
      more with a longer limit (900 s, batch.py --recheck-timeout-s)
      before it is routed, and the first check stays on the record.
  8b. distractor.PROOF_ROOT was never set, so every batch run's proofs went
      to the shared runs/ folder and no record said where its proofs were.
      With --log-dir they now go in <log-dir>/proofs, and each record's
      proof_dir is a path inside the log folder.

The model is faked, and so is the checker except in one test, which runs
a deliberately slow design for a few seconds in a temporary folder."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
from build_corpus import extract_asserts                        # noqa: E402
from oracle import grade_triple_generated                       # noqa: E402
from oracle.grading import INDUCTION_FAIL_NOTE                  # noqa: E402
from oracle.sby import sby_available                            # noqa: E402
from oracle.types import (GradeResult, NecessityVerdict,        # noqa: E402
                          RunEvidence, Tier, TripleResult)
from tests.test_patch import TOKEN_BUCKET                       # noqa: E402

requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH - activate hwtools first")


# --- the checker's answers ---------------------------------------------------

def leg(tier, notes=()):
    run = RunEvidence(mode="prove", rc=8 if tier is Tier.TIMEOUT else 0,
                      depth=20, engine="smtbmc yices", duration_s=1.0,
                      workdir=Path("job"), log_excerpt="", notes=list(notes))
    return GradeResult(tier=tier, reason=tier.name.lower(), runs=[run])


def with_proof_ran_out(notes=()):
    """The proof with the facts hit the time limit."""
    return TripleResult(NecessityVerdict.NOT_PROVEN,
                        "with-invariants grade is TIMEOUT, not PROVEN",
                        with_invariants=leg(Tier.TIMEOUT, notes))


def without_proof_ran_out():
    """The proof without the facts hit the time limit."""
    return TripleResult(NecessityVerdict.INCONCLUSIVE,
                        "without-invariants grade is TIMEOUT - necessity "
                        "not established",
                        with_invariants=leg(Tier.PROVEN),
                        without_invariants=leg(Tier.TIMEOUT))


def proved():
    return TripleResult(NecessityVerdict.NECESSARY,
                        "strengthening is load-bearing: PROVEN with "
                        "invariants, NOT_INDUCTIVE without",
                        with_invariants=leg(Tier.PROVEN),
                        without_invariants=leg(Tier.NOT_INDUCTIVE))


def with_proof(tier, verdict="NOT_PROVEN"):
    return TripleResult(NecessityVerdict[verdict],
                        f"with-invariants grade is {tier.name}, not PROVEN",
                        with_invariants=leg(tier))


NOT_FROM_THE_TIME_LIMIT = {
    "necessary": proved(),
    "decorative": TripleResult(NecessityVerdict.DECORATIVE, "decorative",
                               with_invariants=leg(Tier.PROVEN),
                               without_invariants=leg(Tier.PROVEN)),
    "false": with_proof(Tier.FALSE),
    "did not close": with_proof(Tier.NOT_INDUCTIVE),
    "tool failed": with_proof(Tier.ERROR),
    "tool failed without facts": TripleResult(
        NecessityVerdict.INCONCLUSIVE,
        "without-invariants grade is ERROR - necessity not established",
        with_invariants=leg(Tier.PROVEN), without_invariants=leg(Tier.ERROR)),
    "malformed answer": TripleResult(
        NecessityVerdict.NOT_PROVEN,
        "generator output contract violation: 'verilog' missing",
        with_invariants=GradeResult(Tier.ERROR, "contract violation")),
    "no facts": TripleResult(NecessityVerdict.NO_INVARIANTS,
                             "no invariants supplied"),
    # induction had already failed when the time ran out: more time
    # cannot make the proof close
    "induction failed first": with_proof_ran_out([INDUCTION_FAIL_NOTE]),
}


class Checker:
    """A fake checker: gives its answers in turn and keeps the settings of
    every check it was asked for."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, payload, **kw):
        self.calls.append(kw)
        return self.answers.pop(0)


# --- which verdicts came from the time limit --------------------------------

@pytest.mark.parametrize("result", [with_proof_ran_out(),
                                    without_proof_ran_out()])
def test_a_verdict_from_the_time_limit_is_found(result):
    assert distractor.ran_out_of_time(result)


@pytest.mark.parametrize("name", sorted(NOT_FROM_THE_TIME_LIMIT))
def test_other_verdicts_did_not_come_from_the_time_limit(name):
    assert not distractor.ran_out_of_time(NOT_FROM_THE_TIME_LIMIT[name])


# --- the second check --------------------------------------------------------

def test_a_step_that_ran_out_of_time_is_checked_again_with_a_longer_limit():
    check = Checker(without_proof_ran_out(), proved())
    distractor.necessity_check(check, {}, {}, "checking")
    assert distractor.RECHECK_TIMEOUT_S == 900
    assert [c["timeout_s"] for c in check.calls] == [300, 900]
    # only the main limit grows; the side checks keep theirs
    assert all(c["pdr_timeout_s"] == 120 and c["cover_depth"] == "auto"
               for c in check.calls)
    # the shared settings every worker reads are left alone
    assert distractor.GRADE_KWARGS["timeout_s"] == 300


def test_the_second_verdict_replaces_the_first_and_the_first_is_kept():
    record = {}
    result = distractor.necessity_check(
        Checker(without_proof_ran_out(), proved()), {}, record, "checking")
    assert result.verdict is NecessityVerdict.NECESSARY
    assert record["verdict"] == "NECESSARY"
    assert record["reason"] == proved().reason
    assert record["result"]["verdict"] is NecessityVerdict.NECESSARY
    first = record["first_result"]
    assert first["verdict"] == "INCONCLUSIVE"
    assert first["reason"] == without_proof_ran_out().reason
    assert first["result"]["without_invariants"]["tier"] == Tier.TIMEOUT
    assert isinstance(first["grade_wall_s"], float)
    assert record["rechecked"] is True
    assert record["recheck_timeout_s"] == 900
    assert record["recheck_wall_s"] == record["grade_wall_s"]
    json.dumps(record, default=str)          # the log line can be written


@pytest.mark.parametrize("name", sorted(NOT_FROM_THE_TIME_LIMIT))
def test_only_a_verdict_from_the_time_limit_is_checked_again(name):
    answer = NOT_FROM_THE_TIME_LIMIT[name]
    check = Checker(answer)
    record = {}
    distractor.necessity_check(check, {}, record, "checking")
    assert len(check.calls) == 1
    assert record["verdict"] == answer.verdict.name
    assert not ({"rechecked", "first_result", "recheck_wall_s"} & set(record))


def test_a_step_is_checked_again_only_once():
    check = Checker(with_proof_ran_out(), with_proof_ran_out())
    record = {}
    distractor.necessity_check(check, {}, record, "checking")
    assert len(check.calls) == 2
    assert record["verdict"] == "NOT_PROVEN" and record["rechecked"]
    assert record["first_result"]["verdict"] == "NOT_PROVEN"
    # checked again on this record: the second check is not repeated
    again = Checker(with_proof_ran_out(), proved())
    distractor.necessity_check(again, {}, record, "checking")
    assert [c["timeout_s"] for c in again.calls] == [300]


def test_zero_turns_the_second_check_off(monkeypatch):
    monkeypatch.setattr(distractor, "RECHECK_TIMEOUT_S", 0)
    check = Checker(without_proof_ran_out(), proved())
    record = {}
    distractor.necessity_check(check, {}, record, "checking")
    assert len(check.calls) == 1
    assert record["verdict"] == "INCONCLUSIVE"
    assert "rechecked" not in record and "first_result" not in record


def test_the_second_checks_proofs_go_in_the_records_own_folder(
        monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    monkeypatch.setattr(distractor, "PROOF_ROOT", logs / "proofs")
    record = {"extension_id": "2026-10-21_09h19m17s_t035",
              "parent_id": "c2_002"}
    check = Checker(without_proof_ran_out(), proved())
    distractor.necessity_check(check, {}, record, "checking")
    assert record["proof_dir"].startswith(
        "proofs/2026-10-21_09h19m17s_t035_c2_002_")
    first, second = (c["workdir_root"] for c in check.calls)
    assert first == logs / record["proof_dir"]          # found from the log
    assert second == first / "recheck"


def test_without_a_proof_folder_both_checks_use_the_shared_one(monkeypatch):
    monkeypatch.setattr(distractor, "PROOF_ROOT", None)
    record = {"extension_id": "e1", "parent_id": "c0_000"}
    check = Checker(without_proof_ran_out(), proved())
    distractor.necessity_check(check, {}, record, "checking")
    assert all("workdir_root" not in c for c in check.calls)
    assert "proof_dir" not in record


def test_the_longer_limit_can_be_asked_for_one_check_only(monkeypatch,
                                                          tmp_path):
    monkeypatch.setattr(distractor, "PROOF_ROOT", tmp_path / "proofs")
    record = {"extension_id": "e1", "parent_id": "c0_000"}
    longer = distractor.grade_kwargs(record, leg="recheck", timeout_s=900)
    assert longer["timeout_s"] == 900
    assert distractor.grade_kwargs(record)["timeout_s"] == 300
    assert distractor.GRADE_KWARGS["timeout_s"] == 300


# --- every kind of step gets the second check --------------------------------

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

KEPT = [{"clause": "lim == 4'd9", "status": "kept", "replaced_by": [],
         "reason": ""}]

SRC = """module src (
    input wire clk, input wire step,
    output wire [3:0] hot
);
    reg [1:0] sel;
    initial sel = 2'd0;
    always @(posedge clk)
        if (step) sel <= sel + 2'd1;
    assign hot = 4'b0001 << sel;
endmodule
"""

SINK = """module sink (
    input wire clk, input wire in_valid, input wire [3:0] in_data,
    output reg [3:0] held
);
    initial held = 4'b0001;
    always @(posedge clk)
        if (in_valid) held <= in_data;
    always @(posedge clk) assert (held != 4'b0000);
endmodule
"""

LINK = """module link (
    input wire clk, input wire step, input wire in_valid,
    output wire [3:0] held_o
);
    wire [3:0] hotw;
    reg [3:0] buf_q;
    initial buf_q = 4'b0001;
    always @(posedge clk)
        if (step) buf_q <= hotw;
    src a (.clk(clk), .step(step), .hot(hotw));
    sink b (.clk(clk), .in_valid(in_valid), .in_data(buf_q), .held(held_o));
    always @(posedge clk)
        if (in_valid) assert (held_o != 4'b1111);
endmodule
"""

PAIR = """module pair (
    input wire clk, input wire step,
    output wire [3:0] h0, output wire [3:0] h1
);
    reg [3:0] seen;
    initial seen = 4'd0;
    src u0 (.clk(clk), .step(step), .hot(h0));
    src u1 (.clk(clk), .step(step), .hot(h1));
    always @(posedge clk) seen <= h0 | h1;
    always @(posedge clk) assert (h0 != 4'b0000);
endmodule
"""


def design(top, verilog, facts=(), id=None):
    return {"id": id or f"x_{top}", "top_module": top, "clock": "clk",
            "antecedents": [], "sanity_covers": [], "verilog": verilog,
            "property": extract_asserts(verilog), "invariants": list(facts)}


def counter():
    return design("ctr", CTR, ["lim == 4'd9"], id="c0_7")


def structural_answer():
    return {"patch": LAST_COUNT, "invariants": ["lim == 4'd9", "last <= 4'd9"],
            "dispositions": KEPT, "new_state": "the last count",
            "coupling": "c", "induction_gap": "g",
            "why_parent_insufficient": "w"}


def structural_step(record):
    return extend.grade_step4(counter(), "structural", structural_answer(),
                              dict(record, ext_type="structural",
                                   move="STORAGE"))


def compose_step(record):
    out = {"wrapper": LINK, "top_module": "link", "antecedents": [],
           "sanity_covers": [], "invariants": ["buf_q != 4'b0000"]}
    return extend.grade_compose(design("src", SRC), design("sink", SINK),
                                out, record)


def replicate_step(record):
    out = {"wrapper": PAIR, "top_module": "pair", "antecedents": [],
           "sanity_covers": [],
           "invariants": ["(h0 != 4'b0000) || (h1 != 4'b0000)"]}
    return extend.grade_replicate(design("src", SRC), out, record, 2)


def distractor_step(record):
    parent = design("token_bucket", TOKEN_BUCKET, ["tokens <= 3'd4"])
    return distractor.grade_extension(parent, distractor.SELFTEST_PATCH,
                                      record)


STEPS = {"structural": structural_step, "compose": compose_step,
         "replicate": replicate_step, "distractor": distractor_step}


@pytest.fixture
def checker(monkeypatch):
    """No yosys and no proofs: sizes are text lengths (so new logic is
    live), the new-property-alone check does not close, and the necessity
    check is the Checker the test puts in `install`."""
    for module in (extend, distractor):
        monkeypatch.setattr(module, "state_bits", lambda v, top: len(v))
    monkeypatch.setattr(extend, "grade",
                        lambda *a, **k: GradeResult(Tier.NOT_INDUCTIVE,
                                                    "alone"))

    def install(*answers):
        check = Checker(*answers)
        for module in (extend, distractor):
            monkeypatch.setattr(module, "grade_triple_generated", check)
        return check
    return install


@pytest.mark.parametrize("kind", sorted(STEPS))
def test_every_kind_of_step_is_checked_again(checker, kind):
    check = checker(without_proof_ran_out(), proved())
    record = STEPS[kind]({})
    assert record["verdict"] == "NECESSARY", record.get("error")
    assert record["first_result"]["verdict"] == "INCONCLUSIVE"
    assert record["rechecked"] is True
    assert [c["timeout_s"] for c in check.calls] == [300, 900]


@pytest.mark.parametrize("kind", sorted(STEPS))
def test_every_kind_of_step_is_checked_once_when_it_finished(checker, kind):
    check = checker(with_proof(Tier.NOT_INDUCTIVE))
    record = STEPS[kind]({})
    assert record["verdict"] == "NOT_PROVEN"
    assert len(check.calls) == 1 and "rechecked" not in record


def test_a_combination_saved_by_the_second_check_is_judged_on_it(checker):
    """A combination that proves without its facts is told its combined
    assertion needs no facts; that is read off the second verdict."""
    decorative = TripleResult(NecessityVerdict.DECORATIVE, "decorative",
                              with_invariants=leg(Tier.PROVEN),
                              without_invariants=leg(Tier.PROVEN))
    checker(without_proof_ran_out(), decorative)
    record = compose_step({})
    assert record["verdict"] == "DECORATIVE"
    # 26 Oct 2026: "glue not stateful enough" became "proves without any
    # facts", because the part between the designs no longer has to hold
    # registers (change 4)
    assert "proves without any facts" in record["error"]


# --- the batch run ------------------------------------------------------------

def test_the_second_check_counts_against_the_grading_slots(checker,
                                                           monkeypatch):
    """In batch mode only GRADE_SLOTS answers are checked at once (8 Oct
    2026); a second check is checking too, and holds its slot."""
    live, most, records = [0], [0], []
    lock = threading.Lock()

    def slow_check(payload, **kw):
        with lock:
            live[0] += 1
            most[0] = max(most[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return proved() if kw["timeout_s"] > 300 else without_proof_ran_out()
    monkeypatch.setattr(extend, "grade_triple_generated", slow_check)
    monkeypatch.setattr(batch, "call_model",
                        lambda p, s: (structural_answer(),
                                      {"input": 1, "output": 1}, "ok"))
    monkeypatch.setattr(batch, "dump", records.append)
    monkeypatch.setattr(batch, "GRADE_SLOTS", threading.BoundedSemaphore(2))
    tasks = [{"task_id": i, "attempt": 1, "ext_type": "structural",
              "move": "STORAGE", "parent_id": "c0_7", "k": 1}
             for i in range(6)]
    threads = [threading.Thread(target=batch._real_executor,
                                args=(t, [counter()])) for t in tasks]
    [t.start() for t in threads]
    [t.join(timeout=20) for t in threads]
    assert most[0] == 2
    assert len(records) == 6
    assert all(r["rechecked"] and r["verdict"] == "NECESSARY"
               for r in records)


def test_the_task_line_says_when_a_step_was_checked_again(checker,
                                                          monkeypatch,
                                                          capsys):
    checker(without_proof_ran_out(), proved())
    monkeypatch.setattr(batch, "call_model",
                        lambda p, s: (structural_answer(),
                                      {"input": 1, "output": 1}, "ok"))
    monkeypatch.setattr(batch, "dump", lambda r: None)
    task = {"task_id": 1, "attempt": 1, "ext_type": "structural",
            "move": "STORAGE", "parent_id": "c0_7", "k": 1}
    batch._real_executor(task, [counter()])
    line = capsys.readouterr().out
    assert ("-> NECESSARY (checked again with a 900 s limit; the first "
            "check said INCONCLUSIVE)") in line
    assert batch.recheck_note({"verdict": "NOT_PROVEN"}) == ""


def _corpus(tmp_path):
    corpus = tmp_path / "corpus_trial.jsonl"
    corpus.write_text(json.dumps({"id": "c0_000", "generation": 0}) + "\n")
    return corpus


def _watch_the_run(monkeypatch):
    """A run that does nothing but note the checker's settings while it
    runs, and a record's proof folder."""
    seen = {}

    def fake_loop(rows, executor, **kw):
        seen["limit"] = distractor.RECHECK_TIMEOUT_S
        seen["root"] = distractor.PROOF_ROOT
        record = {"extension_id": "e1", "parent_id": "c0_000"}
        seen["workdir"] = distractor.grade_kwargs(record).get("workdir_root")
        seen["proof_dir"] = record.get("proof_dir")
        return []
    monkeypatch.setattr(batch, "run_loop", fake_loop)
    return seen


@pytest.mark.parametrize("flag, limit", [([], 900),
                                         (["--recheck-timeout-s", "1200"],
                                          1200),
                                         (["--recheck-timeout-s", "0"], 0)])
def test_the_command_line_sets_the_second_checks_limit_for_the_run(
        monkeypatch, tmp_path, flag, limit):
    seen = _watch_the_run(monkeypatch)
    batch.main(["--max-calls", "1", "--corpus", str(_corpus(tmp_path)),
                "--log-dir", str(tmp_path / "logs")] + flag)
    assert seen["limit"] == limit
    assert distractor.RECHECK_TIMEOUT_S == 900     # put back after the run


@pytest.mark.parametrize("value", ["300", "120", "-5"])
def test_a_second_check_no_longer_than_the_first_is_refused(
        monkeypatch, tmp_path, value):
    seen = _watch_the_run(monkeypatch)
    with pytest.raises(SystemExit):
        batch.main(["--max-calls", "1", "--corpus", str(_corpus(tmp_path)),
                    "--log-dir", str(tmp_path / "logs"),
                    "--recheck-timeout-s", value])
    assert seen == {}                              # nothing was run


def test_a_log_dir_keeps_the_runs_proofs_next_to_its_logs(monkeypatch,
                                                          tmp_path):
    monkeypatch.setattr(distractor, "PROOF_ROOT", None)
    seen = _watch_the_run(monkeypatch)
    logs = tmp_path / "logs"
    batch.main(["--max-calls", "1", "--corpus", str(_corpus(tmp_path)),
                "--log-dir", str(logs)])
    assert seen["root"] == (logs / "proofs").resolve()
    assert seen["proof_dir"].startswith("proofs/e1_c0_000_")
    assert seen["workdir"] == (logs / seen["proof_dir"]).resolve()
    assert distractor.PROOF_ROOT is None           # put back after the run


def test_without_a_log_dir_the_proofs_stay_in_the_shared_folder(monkeypatch,
                                                                 tmp_path):
    monkeypatch.setattr(distractor, "PROOF_ROOT", None)
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(batch, "CORPUS", corpus)   # the default corpus
    monkeypatch.setattr(batch, "QUEUE_LOG", tmp_path / "batch_queue.jsonl")
    monkeypatch.setattr(batch, "FIXER_QUEUE", tmp_path / "fixer_queue.jsonl")
    monkeypatch.setattr(distractor, "OUT_LOG", tmp_path / "extensions.jsonl")
    seen = _watch_the_run(monkeypatch)
    batch.main(["--max-calls", "1"])
    assert seen["root"] is None and seen["workdir"] is None
    assert seen["proof_dir"] is None


# --- a real check, in a temporary folder --------------------------------------

@requires_sby
def test_a_real_check_that_ran_out_of_time_is_checked_again(monkeypatch,
                                                            tmp_path):
    """The deliberately slow design of tests/triples (a 64-bit factoring
    question) runs out of a 1 s limit and then of a 2 s second check. Both
    checks' proofs are in the record's own folder inside the log folder,
    each run with its own limit."""
    logs = tmp_path / "logs"
    monkeypatch.setattr(distractor, "PROOF_ROOT", logs / "proofs")
    monkeypatch.setattr(distractor, "GRADE_KWARGS",
                        dict(distractor.GRADE_KWARGS, timeout_s=1))
    monkeypatch.setattr(distractor, "RECHECK_TIMEOUT_S", 2)
    payload = {"verilog": (ROOT / "tests" / "triples" /
                           "slow_factor.sv").read_text(),
               "top_module": "slow_factor", "clock": "clk",
               "antecedents": [], "sanity_covers": [],
               "invariants": ["a == a"]}
    record = {"extension_id": "2026-10-21_10h00m00s_t001",
              "parent_id": "c0_000"}
    distractor.necessity_check(grade_triple_generated, payload, record,
                               "slow design")
    assert record["rechecked"] is True
    assert record["first_result"]["verdict"] == "NOT_PROVEN"
    assert "TIMEOUT" in record["first_result"]["reason"]
    assert record["verdict"] == "NOT_PROVEN"
    assert record["result"]["with_invariants"]["tier"] == Tier.TIMEOUT
    folder = logs / record["proof_dir"]
    first = Path(record["first_result"]["result"]["with_invariants"]
                 ["runs"][0]["workdir"]).parent
    second = Path(record["result"]["with_invariants"]["runs"][0]
                  ["workdir"]).parent
    assert first.is_relative_to(folder / "with")
    assert second.is_relative_to(folder / "recheck" / "with")
    assert "timeout 1\n" in (first / "job.sby").read_text()
    assert "timeout 2\n" in (second / "job.sby").read_text()
