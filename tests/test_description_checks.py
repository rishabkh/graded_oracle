"""Checks written from the description alone (27 Oct 2026, Rishab: "Have a
model that hasn't seen the design read the description and write a few
'this should be possible' and 'this should never happen' checks. If the
design fails one, look at it. Don't auto-mark it wrong").

The question (description and ports, never the code), the answer's
handling, where the checks go in the design and how each result is read
are tested without tools, with a stand-in for the model and for sby. One
small design runs through sby itself; it skips without hwtools."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import description_checks as dc                                # noqa: E402
from oracle.sby import SbyOutcome, sby_available               # noqa: E402

DESIGN = """module pulse (input wire clk, input wire rst, input wire go,
                       output wire busy, output wire done);
    reg [1:0] secret_phase_counter = 2'd0;
    always @(posedge clk)
        if (rst) secret_phase_counter <= 2'd0;
        else if (go && secret_phase_counter == 2'd0) secret_phase_counter <= 2'd1;
        else if (secret_phase_counter != 2'd0) secret_phase_counter <= secret_phase_counter + 2'd1;
    assign busy = secret_phase_counter != 2'd0;
    assign done = secret_phase_counter == 2'd3;
    always @(posedge clk) assert (secret_phase_counter <= 2'd3);
endmodule
"""
PORTS = [("clk", "input", 1), ("rst", "input", 1), ("go", "input", 1),
         ("busy", "output", 1), ("done", "output", 1)]
ROW = {"key": "k1", "verilog": DESIGN, "top_module": "pulse", "clock": "clk",
       "description": "What it is: a one-shot timer: go starts it, done pulses after three cycles."}


def test_question_has_the_description_and_ports_but_never_the_code():
    q = dc.build(ROW, PORTS)
    assert "a one-shot timer" in q
    assert "input  [" not in q.split("PORTS")[1].splitlines()[1]   # 1-bit ports have no range
    assert "output done" in q and "input  go" in q
    assert "secret_phase_counter" not in q and "always" not in q


def test_answer_keeps_usable_checks_and_retries_an_empty_one():
    answers = iter([({"checks": []}, {}),
                    ({"checks": [{"kind": "possible", "expr": "done", "why": "a"},
                                 {"kind": "sometimes", "expr": "busy", "why": "b"},
                                 {"kind": "never", "expr": "  ", "why": "c"},
                                 {"kind": "never", "expr": "done && !busy", "why": "d"}]},
                     {"input": 10, "output": 5})])
    row = dc.ask_one(ROW, PORTS, call=lambda prompt: next(answers))
    assert [c["expr"] for c in row["checks"]] == ["done", "done && !busy"]
    assert row["usage"] == {"input": 10, "output": 5}


def test_no_usable_answer_in_two_tries_is_an_error_not_a_crash():
    def broken(prompt):
        raise RuntimeError("cut off")
    row = dc.ask_one(ROW, PORTS, call=broken)
    assert row == {"key": "k1", "error": "RuntimeError: cut off"}


def test_checks_go_in_on_the_clock_edge_and_never_checks_are_negated():
    inj = dc.with_checks(DESIGN, "pulse", "clk", "never", ["done && !busy"])
    assert "always @(posedge clk) begin" in inj.text
    line = inj.text.splitlines()[next(iter(inj.line_map)) - 1]
    assert "assert (!(done && !busy));" in line
    inj = dc.with_checks(DESIGN, "pulse", "clk", "possible", ["done"])
    assert "cover (done);" in inj.text.splitlines()[next(iter(inj.line_map)) - 1]


def test_a_name_that_is_not_a_port_makes_the_check_bad():
    # Verilog would make it a free wire that passes every check
    assert dc.unknown_names("done && !busy", PORTS) == []
    assert dc.unknown_names("$past(go) && done == 1'b1 && 4'hA != 4'd3", PORTS) == []
    assert dc.unknown_names("phase == 2'd3 || done_flag", PORTS) == ["done_flag", "phase"]


def _fake_sby(outcomes):
    """sby stand-in: answers by the run's name, reading the injected file to
    name the cover line it reports."""
    def sby(name, sv_path, top, mode, depth, timeout_s, workdir):
        rc, how = outcomes[name]
        log = ""
        if mode == "cover":
            line = next(i for i, t in enumerate(Path(sv_path).read_text().splitlines(), 1)
                        if "cover (" in t)
            log = (f"Reached cover statement in step 3 at pulse: {Path(sv_path).name}:{line}.5-{line}.20"
                   if how == "reached" else
                   f"Unreached cover statement at pulse: {Path(sv_path).name}:{line}.5-{line}.20")
        return SbyOutcome(rc=rc, duration_s=0.1, workdir=Path(workdir), log_text=log)
    return sby


def test_each_result_is_read_and_a_failed_check_only_flags(tmp_path):
    checks = [{"kind": "possible", "expr": "done", "why": "a"},
              {"kind": "possible", "expr": "busy && done", "why": "b"},
              {"kind": "never", "expr": "done && !busy", "why": "c"},
              {"kind": "never", "expr": "done && go", "why": "d"},
              {"kind": "never", "expr": "nonsense_port", "why": "e"}]
    sby = _fake_sby({"possible0": (0, "reached"), "possible1": (2, "unreached"),
                     "never2": (0, None), "never3": (2, None), "never4": (1, None)})
    row = dc.run_one(ROW, checks, tmp_path, sby=sby, ports=PORTS)
    assert [c["result"] for c in row["checks"]] == \
        ["reached", "unreached", "held", "happened", "bad"]
    assert row["flagged"] is True
    assert row["flags"] == ["possible check unreached: busy && done",
                            "never check happened: done && go"]
    assert "label" not in row                       # never a label by itself


@pytest.mark.skipif(not sby_available(), reason="sby not on PATH (run hwtools)")
def test_checks_run_on_a_real_design(tmp_path):
    checks = [{"kind": "possible", "expr": "done", "why": "a"},
              {"kind": "never", "expr": "!rst && done && !busy", "why": "b"},
              {"kind": "never", "expr": "!rst && $past(done) && done", "why": "c"},
              {"kind": "possible", "expr": "no_such_port", "why": "d"}]
    row = dc.run_one(ROW, checks, tmp_path, depth=8, timeout_s=60)
    assert [c["result"] for c in row["checks"]] == ["reached", "held", "held", "bad"]
    assert row["flagged"] is False
