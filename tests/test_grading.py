import re
import subprocess
from pathlib import Path

import pytest

import oracle.grading as grading
from oracle import GradeResult, PropertyInfo, Tier
from oracle.grading import grade
from oracle.sby import DEFAULT_ENGINE, SbyOutcome

SV = """module m (
    input wire clk,
    input wire a,
    input wire b
);
    always @(posedge clk) assert (1'b1);
endmodule
"""


@pytest.fixture
def sv_file(tmp_path):
    p = tmp_path / "m.sv"
    p.write_text(SV)
    return p


def fake_run_sby(prove_rc, cover_plan=None, cover_rc_override=None):
    """cover_plan: dict expr -> 'reached' | 'unreached' | 'missing'.
    The fake reads the injected file it is handed, finds the oracle's
    cover lines, and fabricates a log naming those exact line numbers.
    """
    def fake(name, sv_path, top_module, mode, depth, timeout_s,
             workdir_root, engine=DEFAULT_ENGINE):
        wd = Path(workdir_root) / f"{name}_{mode}_fake" / "job"
        wd.mkdir(parents=True, exist_ok=True)
        if mode == "prove":
            traces = []
            if prove_rc == 2:
                traces = [wd / "engine_0" / "trace.vcd"]
            if prove_rc == 4:
                traces = [wd / "engine_0" / "trace_induct.vcd"]
            return SbyOutcome(rc=prove_rc, duration_s=0.1, workdir=wd,
                              log_text="prove log", trace_paths=traces)
        assert mode == "cover"
        log_lines = []
        any_unreached = False
        for i, line in enumerate(sv_path.read_text().splitlines(), 1):
            m = re.search(r"cover \((.+)\);", line)
            if not m:
                continue
            state = (cover_plan or {}).get(m.group(1), "reached")
            if state == "reached":
                log_lines.append(
                    f"engine_0: ##   0:00:00  Reached cover statement in "
                    f"step 1 at {top_module}: {sv_path.name}:{i}.5-{i}.26 (_witness_.c)")
            elif state == "unreached":
                any_unreached = True
                log_lines.append(
                    f"engine_0: ##   0:00:00  Unreached cover statement at "
                    f"{top_module}: {sv_path.name}:{i}.5-{i}.26 (_witness_.c)")
            # 'missing' -> no log line at all
        rc = cover_rc_override if cover_rc_override is not None else (
            2 if any_unreached else 0)
        return SbyOutcome(rc=rc, duration_s=0.1, workdir=wd,
                          log_text="\n".join(log_lines), trace_paths=[])
    return fake


def _grade(sv_file, tmp_path, monkeypatch, prove_rc, antecedents=None,
           sanity=None, cover_plan=None, cover_rc_override=None):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby",
                        fake_run_sby(prove_rc, cover_plan, cover_rc_override))
    prop = PropertyInfo(top_module="m", antecedents=antecedents or [],
                        sanity_covers=sanity or [])
    return grade(sv_file, prop, workdir_root=tmp_path / "runs")


def test_sby_missing_is_error(sv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: False)
    r = grade(sv_file, PropertyInfo(top_module="m"),
              workdir_root=tmp_path / "runs")
    assert r.tier is Tier.ERROR and "sby" in r.reason


def test_missing_file_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    r = grade(tmp_path / "nope.sv", PropertyInfo(top_module="m"),
              workdir_root=tmp_path / "runs")
    assert r.tier is Tier.ERROR


def test_prove_outer_guard_timeout(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=None)
    assert r.tier is Tier.TIMEOUT
    assert r.runs[0].timeout_source == "outer_guard"


def test_prove_sby_timeout(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=8)
    assert r.tier is Tier.TIMEOUT
    assert r.runs[0].timeout_source == "sby"


def test_false_carries_trace(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=2)
    assert r.tier is Tier.FALSE
    assert r.runs[0].trace_paths


def test_unexpected_rc_is_error(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=16)
    assert r.tier is Tier.ERROR


def test_proven_no_antecedent_skips_cover(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0)
    assert r.tier is Tier.PROVEN
    assert len(r.runs) == 1
    assert any("not_applicable" in n for n in r.runs[0].notes)


def test_not_inductive_no_antecedent(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=4)
    assert r.tier is Tier.NOT_INDUCTIVE
    assert any("trace_induct" in p.name for p in r.runs[0].trace_paths)


def test_proven_with_reachable_antecedent(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], sanity=["b"])
    assert r.tier is Tier.PROVEN
    assert len(r.runs) == 2
    assert "a" in r.runs[1].reached_covers
    assert "b" in r.runs[1].reached_covers


def test_vacuous_demotes_full_pass(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_plan={"a": "unreached"})
    assert r.tier is Tier.VACUOUS
    assert "a" in r.runs[1].unreached_covers


def test_vacuous_demotes_rc4_too(sv_file, tmp_path, monkeypatch):
    # The settled rule: an unreachable antecedent outranks NOT_INDUCTIVE.
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=4,
               antecedents=["a"], cover_plan={"a": "unreached"})
    assert r.tier is Tier.VACUOUS


def test_sanity_unreached_is_note_not_tier_change(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], sanity=["b"], cover_plan={"b": "unreached"})
    assert r.tier is Tier.PROVEN
    assert any("sanity_cover_unreached" in n for n in r.runs[1].notes)


def test_antecedent_missing_from_log_is_error(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_plan={"a": "missing"})
    assert r.tier is Tier.ERROR


def test_cover_timeout_blocks_verdict(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_rc_override=8)
    assert r.tier is Tier.TIMEOUT
    assert r.runs[1].timeout_source == "sby"


def test_cover_error_blocks_verdict(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_rc_override=16)
    assert r.tier is Tier.ERROR


def test_injection_failure_is_error(sv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(0))
    prop = PropertyInfo(top_module="wrong_name", antecedents=["a"])
    r = grade(sv_file, prop, workdir_root=tmp_path / "runs")
    # prove stage uses the fake (doesn't care about the name);
    # injection then fails on the missing module name.
    assert r.tier is Tier.ERROR
    assert "injection" in r.reason.lower()


def test_keep_workdirs_false_removes_rundirs(sv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(0))
    prop = PropertyInfo(top_module="m")
    r = grade(sv_file, prop, workdir_root=tmp_path / "runs",
              keep_workdirs=False)
    assert r.tier is Tier.PROVEN
    assert not r.runs[0].workdir.parent.exists()
    assert any("workdir removed" in n for n in r.runs[0].notes)
