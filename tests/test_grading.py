import re
import subprocess
from pathlib import Path

import pytest

import json

import oracle.grading as grading
from oracle import GradeResult, NecessityVerdict, PropertyInfo, Tier
from oracle.grading import grade, grade_generated, grade_triple, grade_triple_generated
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


MINI_VCD = """\
$var integer 32 t smt_step $end
$scope module m $end
$var wire 4 q count $end
$upscope $end
$enddefinitions $end
#0
b0101 q
b00000000000000000000000000000000 t
#10
b0110 q
b00000000000000000000000000000001 t
"""


def fake_run_sby(prove_rc, cover_plan=None, cover_rc_override=None,
                 pdr_rc=None, pdr_unreach_rc=None, false_line=None):
    """cover_plan: dict expr -> 'reached' | 'unreached' | 'missing'.
    The fake reads the injected file it is handed, finds the oracle's
    cover lines, and fabricates a log naming those exact line numbers.
    prove_rc may be an int, or a callable(sv_text) -> int so tests can
    make the verdict depend on whether injected invariants are present.
    pdr_rc answers `abc pdr` second-opinion runs; pdr_unreach_rc answers
    `abc pdr` unreachability runs (recognized by the injected-invariant
    header in a stripped file). Defaulting to prove_rc keeps old tests
    on the unknown path.
    """
    def fake(name, sv_path, top_module, mode, depth, timeout_s,
             workdir_root, engine=DEFAULT_ENGINE):
        wd = Path(workdir_root) / f"{name}_{mode}_fake" / "job"
        wd.mkdir(parents=True, exist_ok=True)
        if engine == "abc pdr":
            text = sv_path.read_text()
            is_unreach = "ORACLE-INJECTED INVARIANTS" in text
            rc = pdr_unreach_rc if is_unreach else pdr_rc
            if rc is None:
                rc = prove_rc
            if callable(rc):
                rc = rc(text)
            traces = [wd / "engine_0" / "trace.vcd"] if rc == 2 else []
            return SbyOutcome(rc=rc, duration_s=0.1, workdir=wd,
                              log_text="pdr log", trace_paths=traces,
                              engine=engine)
        if mode == "prove":
            text = sv_path.read_text()
            rc = prove_rc(text) if callable(prove_rc) else prove_rc
            traces = []
            if rc == 2:
                traces = [wd / "engine_0" / "trace.vcd"]
            if rc == 4:
                traces = [wd / "engine_0" / "trace_induct.vcd"]
            for t in traces:
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_text(MINI_VCD)
            log = "prove log"
            if rc == 2 and false_line is not None:
                marker = ("// invariant" if false_line == "invariant"
                          else "assert (1'b1)")
                lineno = next(i for i, l in enumerate(text.splitlines(), 1)
                              if marker in l)
                log = (f"engine_0.basecase: ##   0:00:00  Assert failed in "
                       f"{top_module}: {sv_path.name}:{lineno}.9-{lineno}.31 "
                       f"(_witness_.check)")
            return SbyOutcome(rc=rc, duration_s=0.1, workdir=wd,
                              log_text=log, trace_paths=traces,
                              engine=engine)
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
                          log_text="\n".join(log_lines), trace_paths=[],
                          engine=engine)
    return fake


def _grade(sv_file, tmp_path, monkeypatch, prove_rc, antecedents=None,
           sanity=None, cover_plan=None, cover_rc_override=None,
           pdr_rc=None, pdr_unreach_rc=None):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby",
                        fake_run_sby(prove_rc, cover_plan, cover_rc_override,
                                     pdr_rc=pdr_rc,
                                     pdr_unreach_rc=pdr_unreach_rc))
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
               antecedents=["a"], cover_plan={"a": "unreached"},
               pdr_unreach_rc=0)
    assert r.tier is Tier.VACUOUS
    assert "a" in r.runs[1].unreached_covers


def test_vacuous_demotes_rc4_too(sv_file, tmp_path, monkeypatch):
    # The settled rule: an unreachable antecedent outranks NOT_INDUCTIVE.
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=4,
               antecedents=["a"], cover_plan={"a": "unreached"},
               pdr_rc=0, pdr_unreach_rc=0)
    assert r.tier is Tier.VACUOUS


# --- Fix A: unbounded vacuity via PDR unreachability ---

def test_vacuous_requires_pdr_proof_of_unreachability(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_plan={"a": "unreached"},
               pdr_unreach_rc=0)
    assert r.tier is Tier.VACUOUS
    assert "abc pdr" in r.reason or "unreachable for all time" in r.reason
    pdr_ev = [ev for ev in r.runs if ev.engine == "abc pdr"][0]
    assert any("proven unreachable" in n for n in pdr_ev.notes)


def test_deep_antecedent_is_not_vacuous(sv_file, tmp_path, monkeypatch):
    # Antecedent unreached at depth D but PDR finds a deeper witness:
    # the old bounded check would have falsely discarded this triple.
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_plan={"a": "unreached"},
               pdr_unreach_rc=2)
    assert r.tier is Tier.PROVEN
    pdr_ev = [ev for ev in r.runs if ev.engine == "abc pdr"][0]
    assert pdr_ev.trace_paths  # the reachability witness
    assert any("reachable beyond depth" in n for n in pdr_ev.notes)


def test_unreach_check_file_is_stripped_and_negated(sv_file, tmp_path, monkeypatch):
    seen = {}
    inner = fake_run_sby(0, cover_plan={"a": "unreached"}, pdr_unreach_rc=0)

    def spy(name, sv_path, top_module, mode, depth, timeout_s,
            workdir_root, engine=DEFAULT_ENGINE):
        if engine == "abc pdr":
            seen["text"] = sv_path.read_text()
        return inner(name, sv_path, top_module, mode, depth, timeout_s,
                     workdir_root, engine)

    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", spy)
    prop = PropertyInfo(top_module="m", antecedents=["a"])
    grade(sv_file, prop, workdir_root=tmp_path / "runs")
    assert "assert (1'b1)" not in seen["text"]   # original assert stripped
    assert "assert (!(a));" in seen["text"]      # negated antecedent only


def test_unreach_pdr_timeout_is_timeout_tier(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_plan={"a": "unreached"},
               pdr_unreach_rc=8)
    assert r.tier is Tier.TIMEOUT
    assert "vacuity undecided" in r.reason


def test_unreach_pdr_error_is_error_tier(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a"], cover_plan={"a": "unreached"},
               pdr_unreach_rc=16)
    assert r.tier is Tier.ERROR


def test_mixed_antecedents_one_deep_one_dead(sv_file, tmp_path, monkeypatch):
    # 'a' has a deep witness, 'c' is provably dead -> VACUOUS names 'c'.
    unreach_rc = lambda text: 2 if "!(a)" in text else 0  # noqa: E731
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=0,
               antecedents=["a", "c"],
               cover_plan={"a": "unreached", "c": "unreached"},
               pdr_unreach_rc=unreach_rc)
    assert r.tier is Tier.VACUOUS
    assert "c" in r.reason


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


# --- necessity criterion (grade_triple) ---

INV_LOADBEARING = lambda text: 0 if "// invariant" in text else 4  # noqa: E731


def _triple(sv_file, tmp_path, monkeypatch, prove_rc, invariants=None,
            antecedents=None, cover_plan=None):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(prove_rc, cover_plan))
    prop = PropertyInfo(top_module="m", antecedents=antecedents or [],
                        invariants=invariants if invariants is not None
                        else ["sa == sb"])
    return grade_triple(sv_file, prop, workdir_root=tmp_path / "runs")


def test_necessary_when_invariant_is_load_bearing(sv_file, tmp_path, monkeypatch):
    r = _triple(sv_file, tmp_path, monkeypatch, INV_LOADBEARING)
    assert r.verdict is NecessityVerdict.NECESSARY
    assert r.with_invariants.tier is Tier.PROVEN
    assert r.without_invariants.tier is Tier.NOT_INDUCTIVE
    # without-run must skip the cover stage: its only job is the rc
    # (a pdr second-opinion run may accompany the prove run)
    assert not any(ev.mode == "cover" for ev in r.without_invariants.runs)


def test_decorative_when_proven_without_help(sv_file, tmp_path, monkeypatch):
    r = _triple(sv_file, tmp_path, monkeypatch, prove_rc=0)
    assert r.verdict is NecessityVerdict.DECORATIVE
    assert r.without_invariants.tier is Tier.PROVEN


def test_not_proven_when_with_run_fails(sv_file, tmp_path, monkeypatch):
    r = _triple(sv_file, tmp_path, monkeypatch, prove_rc=4)
    assert r.verdict is NecessityVerdict.NOT_PROVEN
    assert r.with_invariants.tier is Tier.NOT_INDUCTIVE
    assert r.without_invariants is None  # second call never made


def test_not_proven_when_with_run_vacuous(sv_file, tmp_path, monkeypatch):
    r = _triple(sv_file, tmp_path, monkeypatch, INV_LOADBEARING,
                antecedents=["a"], cover_plan={"a": "unreached"})
    assert r.verdict is NecessityVerdict.NOT_PROVEN
    assert r.with_invariants.tier is Tier.VACUOUS


def test_inconclusive_when_without_run_times_out(sv_file, tmp_path, monkeypatch):
    rc_fn = lambda text: 0 if "// invariant" in text else 8  # noqa: E731
    r = _triple(sv_file, tmp_path, monkeypatch, rc_fn)
    assert r.verdict is NecessityVerdict.INCONCLUSIVE
    assert r.without_invariants.tier is Tier.TIMEOUT


def test_no_invariants_short_circuits(sv_file, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby",
                        lambda *a, **k: calls.append(a) or None)
    r = grade_triple(sv_file, PropertyInfo(top_module="m"),
                     workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NO_INVARIANTS
    assert calls == []  # nothing was run


def test_triple_injection_failure_is_not_proven(sv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(0))
    prop = PropertyInfo(top_module="wrong_name", invariants=["x"])
    r = grade_triple(sv_file, prop, workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NOT_PROVEN
    assert r.with_invariants.tier is Tier.ERROR


# --- Fix B: PDR second opinion on rc=4 ---

def _grade_pdr(sv_file, tmp_path, monkeypatch, **fake_kwargs):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(**fake_kwargs))
    prop = PropertyInfo(top_module="m")
    return grade(sv_file, prop, workdir_root=tmp_path / "runs")


def test_rc4_pdr_proves_stays_not_inductive(sv_file, tmp_path, monkeypatch):
    r = _grade_pdr(sv_file, tmp_path, monkeypatch, prove_rc=4, pdr_rc=0)
    assert r.tier is Tier.NOT_INDUCTIVE
    pdr_runs = [ev for ev in r.runs if ev.engine == "abc pdr"]
    assert len(pdr_runs) == 1
    assert any("pdr_second_opinion" in n and "true" in n
               for n in pdr_runs[0].notes)


def test_rc4_pdr_refutes_is_false_not_fixer_food(sv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(4, pdr_rc=2))
    prop = PropertyInfo(top_module="m", antecedents=["a"])
    r = grade(sv_file, prop, workdir_root=tmp_path / "runs")
    assert r.tier is Tier.FALSE
    assert "deep" in r.reason
    pdr_ev = [ev for ev in r.runs if ev.engine == "abc pdr"][0]
    assert pdr_ev.trace_paths  # the deep counterexample witness
    assert not any(ev.mode == "cover" for ev in r.runs)  # no cover run


def test_rc4_pdr_unknown_keeps_tier_with_note(sv_file, tmp_path, monkeypatch):
    r = _grade_pdr(sv_file, tmp_path, monkeypatch, prove_rc=4, pdr_rc=8)
    assert r.tier is Tier.NOT_INDUCTIVE
    pdr_ev = [ev for ev in r.runs if ev.engine == "abc pdr"][0]
    assert any("inconclusive" in n for n in pdr_ev.notes)


# --- rc=2 attribution + readable traces ---

def test_false_evidence_carries_lines_and_trace_text(sv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby",
                        fake_run_sby(2, false_line="property"))
    r = grade(sv_file, PropertyInfo(top_module="m"),
              workdir_root=tmp_path / "runs")
    assert r.tier is Tier.FALSE
    assert r.runs[0].failed_assert_lines  # which assert, by line
    assert "Trace summary" in r.runs[0].trace_text
    assert "count = 4'h5" in r.runs[0].trace_text


def test_cti_trace_text_on_rc4(sv_file, tmp_path, monkeypatch):
    r = _grade(sv_file, tmp_path, monkeypatch, prove_rc=4)
    assert r.tier is Tier.NOT_INDUCTIVE
    assert "Trace summary" in r.runs[0].trace_text
    assert "At start state (step 0)" in r.runs[0].trace_text


def _triple_false(sv_file, tmp_path, monkeypatch, false_line):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby",
                        fake_run_sby(2, false_line=false_line))
    prop = PropertyInfo(top_module="m", invariants=["bad_inv"])
    return grade_triple(sv_file, prop, workdir_root=tmp_path / "runs")


def test_falsified_invariant_attributed(sv_file, tmp_path, monkeypatch):
    r = _triple_false(sv_file, tmp_path, monkeypatch, "invariant")
    assert r.verdict is NecessityVerdict.NOT_PROVEN
    assert "falsified invariant" in r.reason
    assert "bad_inv" in r.reason
    assert "property was not shown false" in r.reason


def test_false_property_attributed(sv_file, tmp_path, monkeypatch):
    r = _triple_false(sv_file, tmp_path, monkeypatch, "property")
    assert r.verdict is NecessityVerdict.NOT_PROVEN
    assert "property itself is false" in r.reason
    assert "leave the invariants alone" in r.reason


def test_false_without_attribution_notes_it(sv_file, tmp_path, monkeypatch):
    r = _triple_false(sv_file, tmp_path, monkeypatch, None)
    assert r.verdict is NecessityVerdict.NOT_PROVEN
    assert "not attributable" in r.reason


# --- generator output contract boundary ---

GENERATED = json.dumps({
    "verilog": SV,
    "top_module": "m",
    "antecedents": ["a"],
    "sanity_covers": ["b"],
})


def test_grade_generated_valid_reaches_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(0))
    r = grade_generated(GENERATED, workdir_root=tmp_path / "runs")
    assert r.tier is Tier.PROVEN
    assert "a" in r.runs[1].reached_covers


def test_grade_generated_malformed_is_error_without_running_sby(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby",
                        lambda *a, **k: calls.append(a) or None)
    r = grade_generated("not json at all", workdir_root=tmp_path / "runs")
    assert r.tier is Tier.ERROR
    assert "contract violation" in r.reason
    assert calls == []


def test_grade_triple_generated_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(INV_LOADBEARING))
    text = json.dumps({"verilog": SV, "top_module": "m",
                       "invariants": ["sa == sb"]})
    r = grade_triple_generated(text, workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NECESSARY


def test_grade_triple_generated_malformed_is_not_proven(tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    r = grade_triple_generated("{\"verilog\": 3}",
                               workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NOT_PROVEN
    assert r.with_invariants.tier is Tier.ERROR
    assert "contract violation" in r.with_invariants.reason


def test_keep_workdirs_false_removes_rundirs(sv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "sby_available", lambda: True)
    monkeypatch.setattr(grading, "run_sby", fake_run_sby(0))
    prop = PropertyInfo(top_module="m")
    r = grade(sv_file, prop, workdir_root=tmp_path / "runs",
              keep_workdirs=False)
    assert r.tier is Tier.PROVEN
    assert not r.runs[0].workdir.parent.exists()
    assert any("workdir removed" in n for n in r.runs[0].notes)
