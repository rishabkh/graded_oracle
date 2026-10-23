"""The fixer's repaired facts and its proofs (22 Oct 2026, review findings
F9 and F11, fixer part, and B2).

F9/F11  A repaired list is promoted into the corpus the training file is
        built from, and the trained model is scored by EBMC 6.0, which
        cannot read a fact with a system function ($past, $countones, ...),
        the compact `forall` form or a dotted name. The fixer's question
        now forbids them, and a repaired list that uses one is turned away
        before it is checked, as FACT_FORMAT. Before, a $past fact was
        turned away only by accident (as an "out-of-scope identifier
        past"), and a $signed fact went on to be checked.
B2      The fixer never set a proof folder: its checks went to the shared
        runs/ folder while its attempt log copied the record's proof_dir,
        which points into the batch run's own folder. Its checks now go in
        proofs/ next to its own log, and each attempt's line names its
        own folder.
The model and the checker are faked: no calls, no proofs."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import extend                                                   # noqa: E402
import fact_format                                              # noqa: E402
import fix                                                      # noqa: E402
from oracle.types import NecessityVerdict, TripleResult         # noqa: E402

DESIGN = """module acc (
    input  wire       clk,
    input  wire       en,
    output reg  [3:0] cnt
);
    reg signed [4:0] acc;
    initial cnt = 4'd0;
    initial acc = 5'sd0;
    always @(posedge clk)
        if (en && cnt != 4'd9) begin
            cnt <= cnt + 4'd1;
            acc <= acc + 5'sd1;
        end
    always @(posedge clk) assert (cnt <= 4'd9);
endmodule
"""
PARENT = {"id": "c0_3", "top_module": "acc", "clock": "clk",
          "antecedents": [], "sanity_covers": [], "verilog": DESIGN,
          "invariants": ["cnt != 4'd15"]}
RECORD = {"extension_id": "2026-10-21_07h44m12s_t008",
          "ext_type": "structural", "parent_id": "c0_3",
          "verdict": "NOT_PROVEN", "child_verilog": DESIGN,
          "invariants": ["cnt != 4'd15"],
          # where the batch run kept its own proofs, relative to its logs
          "proof_dir": "proofs/2026-10-21_07h44m12s_t008_c0_3_abc123",
          "result": {"with_invariants": {"runs": [
              {"mode": "prove", "rc": 4, "notes": [],
               "trace_text": "cnt = 4'd3, acc = -5'sd2"}]}}}

PROVED = TripleResult(verdict=NecessityVerdict.NECESSARY, reason="ok")
NOT_YET = TripleResult(verdict=NecessityVerdict.NOT_PROVEN,
                       reason="with-invariants grade is NOT_INDUCTIVE")


@pytest.fixture
def fixer(monkeypatch, tmp_path):
    """The attempt log in a temporary folder; the checker faked, keeping
    each check's settings."""
    monkeypatch.setattr(fix, "ATTEMPT_LOG", tmp_path / "fixer_attempts.jsonl")
    checks = []
    verdicts = []

    def check(payload, **kw):
        checks.append({"facts": json.loads(payload)["invariants"], **kw})
        return verdicts.pop(0) if verdicts else PROVED
    monkeypatch.setattr(fix, "grade_triple_generated", check)

    def attempts():
        log = tmp_path / "fixer_attempts.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] \
            if log.exists() else []
    return {"checks": checks, "verdicts": verdicts, "attempts": attempts,
            "logs": tmp_path}


def answers(*lists):
    queue = [{"diagnosis": "d", "invariants": list(x)} for x in lists]
    return lambda prompt, n: queue.pop(0)


# --- F9/F11: facts the benchmark's checker can read --------------------------

def test_the_fixer_question_forbids_system_functions():
    text = " ".join(fix.FIXER_PROMPT.split())
    assert "Each clause is one same-cycle expression" in text
    for name in ("$past", "$stable", "$rose", "$fell", "$changed",
                 "$countones", "$onehot", "$onehot0", "$isunknown",
                 "$signed", "$clog2"):
        assert name in text, name
    assert "turned away before it is checked" in text
    assert "a count of set bits is written as a sum (v[0] + v[1] + v[2])" \
        in text
    # still a template the fixer can fill
    fix.FIXER_PROMPT.format(verilog="v", property="p", invariants="i",
                            pdr="", cti="c")


@pytest.mark.parametrize("fact, problem", [
    ("cnt == $past(cnt) + 4'd1 || !en", "system function $past"),
    ("$signed(acc) >= 0", "system function $signed"),
    ("$countones(cnt) <= 3'd4", "system function $countones"),
    ("forall k in 0..3: cnt != 4'd15", "compact forall form"),
    ("u_core.cnt <= 4'd9", "dotted name"),
])
def test_an_unreadable_repair_is_turned_away_before_any_check(
        fixer, fact, problem):
    won = fix.run_task(RECORD, [PARENT],
                       call=answers(["cnt != 4'd15", fact]), max_attempts=1)
    assert won is None
    assert fixer["checks"] == []                    # nothing was checked
    [attempt] = fixer["attempts"]()
    assert attempt["verdict"] == "FACT_FORMAT"
    assert problem in attempt["error"]
    assert fact in attempt["error"]
    assert "same-cycle expression" in attempt["error"]


def test_signed_went_on_to_be_checked_before_this_fix():
    """Why the scope check alone was not enough: `signed` is a Verilog
    word, so the scope check sees only `acc`, a signal of the design."""
    assert extend.out_of_scope(["$signed(acc) >= 0"], DESIGN, "acc") == set()
    assert extend.out_of_scope(["cnt == $past(cnt)"], DESIGN, "acc") == \
        {"past"}


def test_a_readable_repair_after_an_unreadable_one_is_checked(fixer):
    won = fix.run_task(RECORD, [PARENT],
                       call=answers(["cnt == $past(cnt)"],
                                    ["cnt != 4'd15", "acc <= 5'sd9"]))
    assert won is not None and won["verdict"] == "NECESSARY"
    assert [a["verdict"] for a in fixer["attempts"]()] == ["FACT_FORMAT",
                                                         "NECESSARY"]
    assert [c["facts"] for c in fixer["checks"]] == [["cnt != 4'd15",
                                                      "acc <= 5'sd9"]]


def test_the_fixer_and_the_extender_share_one_definition():
    assert extend._SYSTEM_FUNCTION is fact_format.SYSTEM_FUNCTION
    assert fix.fact_format_error(["cnt <= 4'd9"]) is None


# --- B2: the fixer's own proof folder ----------------------------------------

def test_the_checks_go_next_to_the_fixers_own_log(fixer):
    won = fix.run_task(RECORD, [PARENT],
                       call=answers(["cnt != 4'd15", "acc <= 5'sd9"]))
    [check] = fixer["checks"]
    folder = Path(check["workdir_root"])
    assert folder.parent.parent == (fixer["logs"] / "proofs").resolve()
    assert folder.name == "fix_a1"
    assert folder.parent.name.startswith(
        "2026-10-21_07h44m12s_t008_fix_c0_3_")
    # the same settings as every other check
    for key, value in fix.GRADE_KWARGS.items():
        assert check[key] == value
    # the log names this attempt's folder, relative to the log's folder,
    # not the batch run's folder the record came with
    assert won["proof_dir"] == f"proofs/{folder.parent.name}/fix_a1"
    assert (fixer["logs"] / won["proof_dir"]).resolve() == folder
    assert won["proof_dir"] != RECORD["proof_dir"]
    assert RECORD["proof_dir"] == \
        "proofs/2026-10-21_07h44m12s_t008_c0_3_abc123"   # left as it was


def test_each_attempt_has_its_own_folder_in_the_tasks_folder(fixer):
    fixer["verdicts"].extend([NOT_YET, PROVED])
    fix.run_task(RECORD, [PARENT],
                 call=answers(["cnt != 4'd15", "acc <= 5'sd9"],
                              ["cnt != 4'd15", "acc <= 5'sd8"]))
    first, second = (Path(c["workdir_root"]) for c in fixer["checks"])
    assert first.parent == second.parent
    assert (first.name, second.name) == ("fix_a1", "fix_a2")
    assert [a["proof_dir"] for a in fixer["attempts"]()] == [
        f"proofs/{first.parent.name}/fix_a1",
        f"proofs/{first.parent.name}/fix_a2"]


def test_two_tasks_never_share_a_folder(fixer):
    fix.run_task(RECORD, [PARENT], call=answers(["acc <= 5'sd9"]))
    fix.run_task(RECORD, [PARENT], call=answers(["acc <= 5'sd9"]))
    a, b = (Path(c["workdir_root"]).parent for c in fixer["checks"])
    assert a != b
