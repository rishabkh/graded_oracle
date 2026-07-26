import subprocess
from pathlib import Path

import pytest

from oracle.sby import DEFAULT_ENGINE, SbyOutcome, make_sby_text, run_sby, sby_available

requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH — activate hwtools first")


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


def test_outer_guard_returns_rc_none(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sby", timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr("oracle.sby.subprocess.run", fake_run)
    sv = tmp_path / "m.sv"
    sv.write_text("module m; endmodule\n")
    out = run_sby("m", sv, "m", "prove", 5, 1, tmp_path / "runs")
    assert out.rc is None
    assert out.duration_s >= 0


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
