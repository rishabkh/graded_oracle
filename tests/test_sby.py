import subprocess
from pathlib import Path

import pytest

from oracle.sby import (DEFAULT_ENGINE, PDR_ENGINE, SbyOutcome, make_sby_text,
                        run_sby, sby_available)

requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH - activate hwtools first")


def test_make_sby_text_golden():
    text = make_sby_text("m.sv", "m", "prove", 20, 300)
    assert text == (
        "[options]\n"
        "mode prove\n"
        "depth 20\n"
        "timeout 300\n"
        "\n"
        "[engines]\n"
        "smtbmc yices\n"
        "\n"
        "[script]\n"
        "read -formal m.sv\n"
        "prep -top m\n"
        "\n"
        "[files]\n"
        "m.sv\n"
    )


# 26 Oct 2026, Rishab approved (change 3): the nothing-lost search turns
# traces off, since rebuilding abc's trace ran past the time limit
def test_extra_options_go_in_the_options_section(monkeypatch, tmp_path):
    text = make_sby_text("m.sv", "m", "bmc", 48, 120,
                         engine="smtbmc yices\nabc bmc3",
                         options=("vcd off",))
    lines = text.splitlines()
    assert lines[:6] == ["[options]", "mode bmc", "depth 48", "timeout 120",
                         "vcd off", ""]
    assert lines[6:9] == ["[engines]", "smtbmc yices", "abc bmc3"]
    # no options writes exactly the text it wrote before
    assert make_sby_text("m.sv", "m", "prove", 20, 300, options=()) == \
        make_sby_text("m.sv", "m", "prove", 20, 300)
    # and run_sby passes them on to the job file
    monkeypatch.setattr(
        "oracle.sby.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=0,
                                                    stdout="", stderr=""))
    sv = tmp_path / "m.sv"
    sv.write_text("module m; endmodule\n")
    out = run_sby("m", sv, "m", "bmc", 48, 120, tmp_path / "runs",
                  options=("vcd off",))
    job = (out.workdir.parent / "job.sby").read_text()
    assert "timeout 120\nvcd off\n" in job


def test_outer_guard_returns_rc_none(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sby", timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr("oracle.sby.subprocess.run", fake_run)
    sv = tmp_path / "m.sv"
    sv.write_text("module m; endmodule\n")
    out = run_sby("m", sv, "m", "prove", 5, 1, tmp_path / "runs")
    assert out.rc is None
    assert out.duration_s >= 0


def test_outcome_records_engine(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("oracle.sby.subprocess.run", fake_run)
    sv = tmp_path / "m.sv"
    sv.write_text("module m; endmodule\n")
    a = run_sby("m", sv, "m", "prove", 5, 60, tmp_path / "runs")
    b = run_sby("m", sv, "m", "prove", 5, 60, tmp_path / "runs",
                engine=PDR_ENGINE)
    assert a.engine == DEFAULT_ENGINE
    assert b.engine == "abc pdr"
    assert "abc pdr" in make_sby_text("m.sv", "m", "prove", 5, 60,
                                      engine=PDR_ENGINE)


def test_unique_rundirs(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("oracle.sby.subprocess.run", fake_run)
    sv = tmp_path / "m.sv"
    sv.write_text("module m; endmodule\n")
    a = run_sby("m", sv, "m", "prove", 5, 60, tmp_path / "runs")
    b = run_sby("m", sv, "m", "prove", 5, 60, tmp_path / "runs")
    assert a.workdir != b.workdir


@requires_sby
def test_run_sby_smoke_trivial_pass(tmp_path):
    sv = tmp_path / "t.sv"
    sv.write_text(
        "module t (input wire clk);\n"
        "    always @(posedge clk) assert (1'b1);\n"
        "endmodule\n")
    out = run_sby("t", sv, "t", "prove", 5, 60, tmp_path / "runs")
    assert out.rc == 0
    assert "PASS" in out.log_text


# --- keep everything (8 Oct 2026): the tool's own output is saved next to
# its proof folder, and the evidence points at the full log ---

def test_tool_output_is_saved_next_to_the_proof(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        job = Path(kwargs["cwd"]) / "job"
        job.mkdir()
        (job / "logfile.txt").write_text("line\n" * 100)
        return subprocess.CompletedProcess(args=args, returncode=0,
                                           stdout="screen text",
                                           stderr="python crash text")
    monkeypatch.setattr("oracle.sby.subprocess.run", fake_run)
    sv = tmp_path / "m.sv"
    sv.write_text("module m; endmodule\n")
    out = run_sby("m", sv, "m", "prove", 5, 60, tmp_path / "runs")
    rundir = out.workdir.parent
    assert (rundir / "sby_stderr.txt").read_text() == "python crash text"
    assert (rundir / "sby_stdout.txt").read_text() == "screen text"
    assert out.log_path == out.workdir / "logfile.txt"
    assert out.log_path.read_text().count("line") == 100


def test_partial_output_of_a_hung_run_is_saved(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sby", timeout=1,
                                        output=b"partial out",
                                        stderr=b"partial err")
    monkeypatch.setattr("oracle.sby.subprocess.run", fake_run)
    sv = tmp_path / "m.sv"
    sv.write_text("module m; endmodule\n")
    out = run_sby("m", sv, "m", "prove", 5, 1, tmp_path / "runs")
    rundir = out.workdir.parent
    assert out.rc is None
    assert (rundir / "sby_stderr.txt").read_text() == "partial err"
    assert (rundir / "sby_stdout.txt").read_text() == "partial out"
    assert out.log_path is None          # sby never wrote its log
