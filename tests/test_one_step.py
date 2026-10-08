"""The 1-step check: is an answer inductive under the textbook rule (one
step back), and which of its facts does that rule need? Uses sby; skips
when the toolchain is not loaded."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import one_step                                                  # noqa: E402

needs_sby = pytest.mark.skipif(shutil.which("sby") is None,
                               reason="sby not on PATH")

# Two 3-bit counters stepped together. The property only looks at a == 3,
# so it is not inductive alone: a=2, b=0 holds it and one step breaks it.
# "a == b" is the needed fact; "a <= 3'd7" is true of any 3-bit value and
# does nothing (padding).
ROW = {"id": "toy", "generation": 0, "top_module": "toy", "clock": "clk",
       "verilog": """module toy (input wire clk, input wire i_ce);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    always @(posedge clk) if (i_ce) begin a <= a + 3'd1; b <= b + 3'd1; end
    always @(posedge clk) assert (!(a == 3'd3) || (b == 3'd3));
endmodule
""",
       "invariants": ["a == b", "a <= 3'd7"]}


@needs_sby
def test_the_full_answer_is_inductive_at_one_step(tmp_path):
    r = one_step.prove(ROW, ROW["invariants"], tmp_path, depth=1)
    assert r["tier"] == "INDUCTIVE"


@needs_sby
def test_dropping_the_needed_fact_breaks_it_with_a_counterexample(tmp_path):
    r = one_step.prove(ROW, ["a <= 3'd7"], tmp_path, depth=1)
    assert r["tier"] == "NOT_INDUCTIVE"
    assert r["trace"] and "a" in r["trace"]


@needs_sby
def test_dropping_padding_leaves_it_inductive(tmp_path):
    r = one_step.prove(ROW, ["a == b"], tmp_path, depth=1)
    assert r["tier"] == "INDUCTIVE"


@needs_sby
def test_the_run_writes_one_line_per_proof_and_resumes(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(ROW) + "\n")
    out = tmp_path / "out.jsonl"
    one_step.run(corpus, out, workers=2)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    # -1: full answer at 20 steps; -2: full answer at 1 step; then each
    # fact dropped, at 1 step
    assert sorted(l["dropped"] for l in lines) == [-2, -1, 0, 1]
    one_step.run(corpus, out, workers=2)                    # nothing left to do
    assert len(out.read_text().splitlines()) == 4


def test_summary_reads_the_verdicts(tmp_path):
    out = tmp_path / "out.jsonl"
    rows = [
        {"id": "x", "dropped": -1, "tier": "INDUCTIVE", "trace": None},
        {"id": "x", "dropped": -2, "tier": "INDUCTIVE", "trace": None},
        {"id": "x", "dropped": 0, "tier": "NOT_INDUCTIVE", "trace": "t",
         "clause": "a == b"},
        {"id": "x", "dropped": 1, "tier": "INDUCTIVE", "trace": None,
         "clause": "a <= 3'd7"},
        {"id": "y", "dropped": -1, "tier": "INDUCTIVE", "trace": None},
        {"id": "y", "dropped": -2, "tier": "NOT_INDUCTIVE", "trace": "t"},
    ]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = one_step.summary(out)
    assert s["passes_one_step"] == {"x"}
    assert s["needed"] == {"x": [0]}
    assert s["counterexample"][("x", 0)] == "t"


# --- the whole proof, for facts not yet proven true ---------------------
# prove() runs the induction step only, which is enough for subsets of an
# answer already proven. A NEW fact (task 4: Opus strengthening a too-weak
# answer) has not been proven, so it needs the base case too, and a
# rejection has to say which fact failed and why.

@needs_sby
def test_a_true_one_step_answer_is_proven(tmp_path):
    r = one_step.prove_true(ROW, ["a == b"], tmp_path)
    assert r["tier"] == "PROVEN"


@needs_sby
def test_a_true_but_too_weak_answer_gets_its_counterexample(tmp_path):
    r = one_step.prove_true(ROW, ["a <= 3'd7"], tmp_path)
    assert r["tier"] == "NOT_INDUCTIVE"
    assert r["fact"] is None                      # the property broke
    assert r["trace"] and "trace_induct" in r["trace"]


@needs_sby
def test_a_false_fact_is_named_with_a_run_from_reset(tmp_path):
    r = one_step.prove_true(ROW, ["a == b", "a == b + 3'd1"], tmp_path)
    assert r["tier"] == "FALSE"
    assert r["fact"] == "a == b + 3'd1"
    assert r["trace"] and "trace.vcd" in r["trace"]


@needs_sby
@pytest.mark.parametrize("fact, says", [
    ("a == q", "q"),                               # yosys would invent a wire
    ("a == b +", "syntax error"),
    ("$past(a) == b", "$past"),
])
def test_an_unreadable_fact_is_named_not_called_false(tmp_path, fact, says):
    r = one_step.prove_true(ROW, ["a == b", fact], tmp_path)
    assert r["tier"] == "BAD_FACT"
    assert r["fact"] == fact and says in r["detail"]


# --- keep everything (8 Oct 2026): proof folders were deleted after every
# check, so 5,255 rows of 1-step results had no proof behind them ---

@needs_sby
def test_the_step_check_keeps_its_proof_folder(tmp_path):
    r = one_step.prove(ROW, ["a <= 3'd7"], tmp_path, depth=1,
                       name="toy_d0")
    folder = tmp_path / r["proof_dir"]
    assert r["proof_dir"].startswith("toy_d0_")
    assert (folder / "job" / "logfile.txt").exists()
    assert (folder / "sby_stderr.txt").exists()
    assert list((folder / "job").rglob("trace_induct.vcd"))


@needs_sby
def test_the_whole_check_keeps_its_proof_folder(tmp_path):
    r = one_step.prove_true(ROW, ["a == b"], tmp_path, name="toy_fix")
    assert r["tier"] == "PROVEN"
    assert (tmp_path / r["proof_dir"] / "job" / "logfile.txt").exists()


def test_a_hung_check_keeps_what_it_printed(tmp_path, monkeypatch):
    import subprocess

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("sby", 1, output="half a log",
                                        stderr="")
    monkeypatch.setattr(one_step.subprocess, "run", hang)
    r = one_step.prove(ROW, ["a == b"], tmp_path, name="toy_hang")
    assert r["tier"] == "TIMEOUT"
    assert (tmp_path / r["proof_dir"] / "sby_stdout.txt").read_text() == \
        "half a log"


def test_a_run_keeps_proofs_next_to_its_log(tmp_path, monkeypatch):
    seen = []

    def fake_prove(row, facts, work, depth=1, name=None):
        seen.append((Path(work), name))
        return {"tier": "INDUCTIVE", "trace": None, "broke": None,
                "secs": 0, "proof_dir": f"{name}_x"}
    monkeypatch.setattr(one_step, "prove", fake_prove)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(ROW) + "\n")
    out = tmp_path / "logs" / "one_step.jsonl"
    one_step.run(corpus, out, workers=1)
    assert {w for w, _ in seen} == {tmp_path / "logs" / "one_step_proofs"}
    assert sorted(n for _, n in seen) == ["toy_d-1", "toy_d-2", "toy_d0",
                                          "toy_d1"]
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["proof_dir"].startswith("one_step_proofs/")
