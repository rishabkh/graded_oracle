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


# --- the clock-edge fix (21 Oct 2026, Rishab approved) -------------------
# yosys reads an assertion inside always @(posedge clk) one step late, so
# at depth 1 the old check never assumed the property in the first state
# and asked the facts alone to imply it. These answers are 1-inductive by
# the textbook rule (and by EBMC, the benchmark's judge); the old check
# called them too weak. A truly weak answer must still fail.

import os                                                        # noqa: E402

# b copies a one step behind; a counts 0..5. The property is clocked.
# "a <= 4'd5" is a good answer: with the property, it carries over a step.
LAG = {"id": "lag", "generation": 0, "top_module": "lag", "clock": "clk",
       "verilog": """module lag (input wire clk);
    reg [3:0] a = 4'd0;
    reg [3:0] b = 4'd0;
    always @(posedge clk) begin
        a <= (a == 4'd5) ? 4'd0 : a + 4'd1;
        b <= a;
    end
    always @(posedge clk) assert (b <= 4'd5);
endmodule
""", "invariants": ["a <= 4'd5"]}

# the same idea through $past: the property looks back one step
PAST_LAG = {"id": "pastlag", "generation": 0, "top_module": "pastlag",
            "clock": "clk", "verilog": """module pastlag (input wire clk);
    reg [3:0] a = 4'd0;
    reg past_valid = 1'b0;
    always @(posedge clk) begin
        past_valid <= 1'b1;
        a <= (a == 4'd5) ? 4'd0 : a + 4'd1;
    end
    always @(posedge clk) if (past_valid) assert ($past(a) <= 4'd5);
endmodule
""", "invariants": ["a <= 4'd5"]}

# both kinds in one design: a same-step assertion and a clocked one
MIXED = {"id": "mixed", "generation": 0, "top_module": "mixed",
         "clock": "clk", "verilog": """module mixed (input wire clk);
    reg [3:0] a = 4'd0;
    reg [3:0] b = 4'd0;
    always @(posedge clk) begin
        a <= (a == 4'd5) ? 4'd0 : a + 4'd1;
        b <= a;
    end
    assert property (a <= 4'd5);
    always @(posedge clk) assert (b <= 4'd5);
endmodule
""", "invariants": []}

GOOD, WEAK = ["a <= 4'd5"], ["a <= 4'd15"]       # 15: a can be 6..15


@needs_sby
@pytest.mark.parametrize("row", [LAG, PAST_LAG, MIXED],
                         ids=["clocked", "looks_back", "mixed"])
def test_the_old_check_called_a_good_answer_too_weak_the_fix_does_not(
        tmp_path, row):
    facts = row["invariants"]
    old = one_step.prove(row, facts, tmp_path, clock_edge_fix=False)
    new = one_step.prove(row, facts, tmp_path)
    assert old["tier"] == "NOT_INDUCTIVE"                    # the quirk
    assert new["tier"] == "INDUCTIVE"


@needs_sby
@pytest.mark.parametrize("row", [LAG, PAST_LAG], ids=["clocked", "looks_back"])
def test_a_truly_weak_answer_fails_with_and_without_the_fix(tmp_path, row):
    for fix in (False, True):
        r = one_step.prove(row, WEAK, tmp_path, clock_edge_fix=fix)
        assert r["tier"] == "NOT_INDUCTIVE"
        assert r["trace"] and "a" in r["trace"]


@needs_sby
@pytest.mark.parametrize("row", [LAG, PAST_LAG], ids=["clocked", "looks_back"])
def test_with_the_fix_the_fact_is_still_needed(tmp_path, row):
    assert one_step.prove(row, [], tmp_path)["tier"] == "NOT_INDUCTIVE"


@needs_sby
def test_the_whole_proof_takes_the_fix_too(tmp_path):
    """prove_true (fix_weak.py's check) had the same quirk."""
    old = one_step.prove_true(LAG, GOOD, tmp_path, clock_edge_fix=False)
    assert old["tier"] == "NOT_INDUCTIVE"
    assert one_step.prove_true(LAG, GOOD, tmp_path)["tier"] == "PROVEN"
    weak = one_step.prove_true(LAG, WEAK, tmp_path)
    assert weak["tier"] == "NOT_INDUCTIVE" and weak["fact"] is None


@needs_sby
def test_with_the_fix_a_clocked_property_is_read_in_the_first_state(
        tmp_path):
    """The base case now reads a clocked property on the start state
    itself; the old one-step-late reading never looked at it at depth 1."""
    row = dict(LAG, verilog=LAG["verilog"].replace(
        "reg [3:0] b = 4'd0;", "reg [3:0] b = 4'd9;"))
    r = one_step.prove_true(row, GOOD, tmp_path)
    assert r["tier"] == "FALSE" and r["fact"] is None


def test_without_the_fix_the_job_is_the_one_the_6_oct_log_was_made_with():
    assert one_step._job("toy", 1, "smtbmc --induction yices", False) == (
        "[options]\nmode prove\ndepth 1\ntimeout 120\n\n"
        "[engines]\nsmtbmc --induction yices\n\n"
        "[script]\nread -formal design.sv\nprep -top toy\n\n"
        "[files]\ndesign.sv\n")
    fixed = one_step._job("toy", 1, "smtbmc --induction yices", True)
    assert "prep -top toy\n" + one_step.SAME_STEP + "\n[files]" in fixed


# --- checked against EBMC, the benchmark's judge ---------------------------

EBMC_BIN = os.environ.get("EBMC_PATH")
needs_ebmc = pytest.mark.skipif(
    not (EBMC_BIN and Path(EBMC_BIN).is_file()),
    reason="needs EBMC_PATH pointing at the ebmc binary")

# LAG in the benchmark's own file shape: the property as `prop`, so the
# scorer's adapter adds the facts as lemmas and runs its 1-step rule
LAG_BENCH = LAG["verilog"].replace(
    "always @(posedge clk) assert (b <= 4'd5);",
    "property prop; @(posedge clk) (b <= 4'd5); endproperty")


@needs_ebmc
@needs_sby
@pytest.mark.parametrize("facts, ebmc_says, we_say", [
    (GOOD, "PROVEN", "INDUCTIVE"),
    (WEAK, "INCONCLUSIVE", "NOT_INDUCTIVE")])
def test_ebmc_and_the_fixed_check_agree_on_a_clocked_property(
        tmp_path, monkeypatch, facts, ebmc_says, we_say):
    sys.path.insert(0, str(ROOT / "initiator"))
    import ebmc_eval
    monkeypatch.setattr(ebmc_eval, "EBMC", EBMC_BIN)
    bench = tmp_path / "lag.sv"
    bench.write_text(LAG_BENCH)
    judged = ebmc_eval.run(bench, facts, "one_inductive_with_prop", tmp_path)
    assert judged["verdict"] == ebmc_says
    assert one_step.prove(LAG, facts, tmp_path)["tier"] == we_say


@needs_ebmc
@needs_sby
@pytest.mark.parametrize("facts, ebmc_says, we_say", [
    (GOOD, "PROVEN", "INDUCTIVE"),
    (WEAK, "INCONCLUSIVE", "NOT_INDUCTIVE")])
def test_ebmc_and_the_fixed_check_agree_on_a_property_that_looks_back(
        tmp_path, facts, ebmc_says, we_say):
    """EBMC cannot read $past inside a named property (the benchmark's
    shape), so facts and property go into one module-level assertion,
    judged with the same 1-step rule."""
    import subprocess
    sys.path.insert(0, str(ROOT / "initiator"))
    from ebmc_eval import parse_verdict
    conj = " && ".join(f"({f})" for f in facts) + \
        " && (!past_valid || ($past(a) <= 4'd5))"
    src = PAST_LAG["verilog"].replace(
        "always @(posedge clk) if (past_valid) assert ($past(a) <= 4'd5);",
        f"assert property (@(posedge clk) ({conj}));")
    path = tmp_path / "pastlag.sv"
    path.write_text(src)
    r = subprocess.run([EBMC_BIN, str(path), "--k-induction", "--bound", "1"],
                       capture_output=True, text=True, timeout=120)
    assert parse_verdict(r.stdout, r.stderr,
                         "one_inductive_with_prop") == ebmc_says
    assert one_step.prove(PAST_LAG, facts, tmp_path)["tier"] == we_say


# --- old and new verdicts are never mixed in one log ------------------------

def _fake_prove(row, facts, work, depth=1, name=None):
    return {"tier": "INDUCTIVE", "trace": None, "broke": None, "secs": 0,
            "proof_dir": f"{name}_x"}


def test_every_line_says_it_was_made_with_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(one_step, "prove", _fake_prove)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(ROW) + "\n")
    out = tmp_path / "out.jsonl"
    one_step.run(corpus, out, workers=1)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 4 and all(l["clock_edge_fix"] is True
                                   for l in lines)
    one_step.run(corpus, out, workers=1)                    # resumes
    assert len(out.read_text().splitlines()) == 4


def test_a_run_never_adds_to_a_log_from_before_the_fix(tmp_path,
                                                       monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("no proof should run")
    monkeypatch.setattr(one_step, "prove", must_not_run)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(ROW) + "\n")
    out = tmp_path / "one_step.jsonl"
    before = json.dumps({"id": "toy", "dropped": -1, "tier": "INDUCTIVE"})
    out.write_text(before + "\n")                       # a 6 Oct line
    with pytest.raises(SystemExit) as e:
        one_step.run(corpus, out, workers=1)
    assert "clock-edge fix" in str(e.value) and "--out" in str(e.value)
    assert out.read_text() == before + "\n"


def test_the_default_log_is_a_new_file_not_the_6_oct_one():
    assert one_step.OUT.name != "one_step.jsonl"
    assert "clock_edge_fix" in one_step.OUT.name


def test_the_report_warns_about_lines_from_before_the_fix(tmp_path, capsys):
    rows = [{"id": "x", "dropped": -1, "tier": "INDUCTIVE", "trace": None},
            {"id": "x", "dropped": -2, "tier": "INDUCTIVE", "trace": None}]
    old = tmp_path / "old.jsonl"
    old.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = one_step.summary(old)
    assert s["lines_before_fix"] == 2
    one_step.report(old)
    assert "WARNING: 2 line(s)" in capsys.readouterr().out
    new = tmp_path / "new.jsonl"
    new.write_text("".join(json.dumps(dict(r, clock_edge_fix=True)) + "\n"
                           for r in rows))
    assert one_step.summary(new)["lines_before_fix"] == 0
    assert one_step.before_fix_warning(one_step.summary(new)) is None
    one_step.report(new)
    assert "WARNING" not in capsys.readouterr().out


# --- a toolchain that cannot take the fix stops the run ---------------------
# The fix selects yosys's $check cells; a yosys without them would select
# nothing and quietly keep the quirk, so a run proves a toy first.

@needs_sby
def test_this_toolchain_takes_the_fix_and_the_probe_tells_the_two_apart(
        tmp_path):
    assert one_step.fix_takes_effect()
    old = one_step.prove(one_step.TIMING_PROBE,
                         one_step.TIMING_PROBE["invariants"], tmp_path,
                         clock_edge_fix=False)
    assert old["tier"] == "NOT_INDUCTIVE"


def test_a_run_stops_when_the_fix_does_not_take_effect(tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(one_step.shutil, "which", lambda name: "/bin/sby")
    monkeypatch.setattr(one_step, "prove", lambda *a, **k: {
        "tier": "NOT_INDUCTIVE", "trace": None, "broke": None, "secs": 0})
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(ROW) + "\n")
    out = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit) as e:
        one_step.main(["--corpus", str(corpus), "--out", str(out)])
    assert "one step late" in str(e.value)
    assert not out.exists()
