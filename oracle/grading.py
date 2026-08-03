"""Orchestration: two sequenced sby runs combined by the decision table.

One sby invocation per mode — never a multi-task .sby file, whose OR'd
return codes would re-create the exact ambiguity this oracle exists to
eliminate. ERROR/TIMEOUT are non-verdicts; PROVEN is never granted
without completed, fully-accounted cover evidence.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .contract import ContractViolation, parse_generator_output
from .inject import InjectionError, inject_covers, inject_invariants
from .parse import parse_cover_log, tail
from .sby import DEFAULT_ENGINE, SbyOutcome, run_sby, sby_available
from .types import (GradeResult, NecessityVerdict, PropertyInfo, RunEvidence,
                    Tier, TripleResult)

NA_NOTE = "vacuity_check: not_applicable (no antecedent)"
SANITY_NOTE = ("sanity_cover_unreached: instrument suspect — "
               "distrust the vacuity verdict for this design")


def _evidence(mode: str, out: SbyOutcome, depth: int, *,
              timeout_source: str | None = None) -> RunEvidence:
    return RunEvidence(
        mode=mode, rc=out.rc, depth=depth, engine=DEFAULT_ENGINE,
        duration_s=out.duration_s, workdir=out.workdir,
        log_excerpt=tail(out.log_text), trace_paths=out.trace_paths,
        timeout_source=timeout_source)


def grade(verilog_file: Path | str, prop: PropertyInfo, *,
          depth: int = 20, timeout_s: int = 300,
          workdir_root: Path | None = None,
          keep_workdirs: bool = True) -> GradeResult:
    result = _grade(Path(verilog_file), prop, depth, timeout_s,
                    Path(workdir_root) if workdir_root is not None
                    else Path(__file__).resolve().parent.parent / "runs")
    if not keep_workdirs:
        for ev in result.runs:
            shutil.rmtree(ev.workdir.parent, ignore_errors=True)
            ev.notes.append("workdir removed (keep_workdirs=False)")
    return result


def grade_generated(output_text: str, *,
                    depth: int = 20, timeout_s: int = 300,
                    workdir_root: Path | None = None,
                    keep_workdirs: bool = True) -> GradeResult:
    """Grade raw generator (LLM) output: one JSON object per attempt.

    Malformed output is a contract violation and grades ERROR — a
    non-verdict the loop can route — never an exception. sby is not
    invoked for output that fails the contract.
    """
    try:
        gen = parse_generator_output(output_text)
    except ContractViolation as exc:
        return GradeResult(Tier.ERROR,
                           f"generator output contract violation: {exc}")
    with tempfile.TemporaryDirectory() as td:
        sv = Path(td) / f"{gen.prop.top_module}.sv"
        sv.write_text(gen.verilog)
        return grade(sv, gen.prop, depth=depth, timeout_s=timeout_s,
                     workdir_root=workdir_root, keep_workdirs=keep_workdirs)


def grade_triple_generated(output_text: str, *,
                           depth: int = 20, timeout_s: int = 300,
                           workdir_root: Path | None = None,
                           keep_workdirs: bool = True) -> TripleResult:
    """Necessity-check raw generator output (see grade_generated)."""
    try:
        gen = parse_generator_output(output_text)
    except ContractViolation as exc:
        err = GradeResult(Tier.ERROR,
                          f"generator output contract violation: {exc}")
        return TripleResult(NecessityVerdict.NOT_PROVEN, err.reason,
                            with_invariants=err)
    with tempfile.TemporaryDirectory() as td:
        sv = Path(td) / f"{gen.prop.top_module}.sv"
        sv.write_text(gen.verilog)
        return grade_triple(sv, gen.prop, depth=depth, timeout_s=timeout_s,
                            workdir_root=workdir_root,
                            keep_workdirs=keep_workdirs)


def grade_triple(verilog_file: Path | str, prop: PropertyInfo, *,
                 depth: int = 20, timeout_s: int = 300,
                 workdir_root: Path | None = None,
                 keep_workdirs: bool = True) -> TripleResult:
    """The necessity criterion: grade twice, with and without invariants.

    A triple is Stage-4-worthy only if the strengthening is load-bearing:
    PROVEN with the invariants injected AND NOT_INDUCTIVE with them
    stripped. PROVEN both ways means the invariant was decorative — the
    plumbing worked but the premise wasn't tested.

    The without-run strips the antecedents too: its only job is "does
    induction close unaided", so the cover stage would be wasted compute.
    """
    kwargs = dict(depth=depth, timeout_s=timeout_s,
                  workdir_root=workdir_root, keep_workdirs=keep_workdirs)
    if not prop.invariants:
        return TripleResult(NecessityVerdict.NO_INVARIANTS,
                            "no invariants supplied — nothing to test "
                            "necessity of")

    verilog_file = Path(verilog_file)
    if not verilog_file.is_file():
        err = GradeResult(Tier.ERROR,
                          f"verilog file not found: {verilog_file}")
        return TripleResult(NecessityVerdict.NOT_PROVEN, err.reason,
                            with_invariants=err)
    try:
        inj = inject_invariants(verilog_file.read_text(), prop.top_module,
                                prop.invariants)
    except InjectionError as exc:
        err = GradeResult(Tier.ERROR, f"invariant injection failed: {exc}")
        return TripleResult(NecessityVerdict.NOT_PROVEN, err.reason,
                            with_invariants=err)

    with tempfile.TemporaryDirectory() as td:
        strengthened = Path(td) / verilog_file.name
        strengthened.write_text(inj.text)
        with_res = grade(strengthened, prop, **kwargs)

    if with_res.tier is not Tier.PROVEN:
        return TripleResult(
            NecessityVerdict.NOT_PROVEN,
            f"with-invariants grade is {with_res.tier.name}, not PROVEN",
            with_invariants=with_res)

    prop_without = PropertyInfo(top_module=prop.top_module, clock=prop.clock)
    without_res = grade(verilog_file, prop_without, **kwargs)

    if without_res.tier is Tier.NOT_INDUCTIVE:
        return TripleResult(
            NecessityVerdict.NECESSARY,
            "strengthening is load-bearing: PROVEN with invariants, "
            "NOT_INDUCTIVE without",
            with_invariants=with_res, without_invariants=without_res)
    if without_res.tier is Tier.PROVEN:
        return TripleResult(
            NecessityVerdict.DECORATIVE,
            "invariants are decorative: the property proves without them",
            with_invariants=with_res, without_invariants=without_res)
    return TripleResult(
        NecessityVerdict.INCONCLUSIVE,
        f"without-invariants grade is {without_res.tier.name} — necessity "
        "not established",
        with_invariants=with_res, without_invariants=without_res)


def _grade(verilog_file: Path, prop: PropertyInfo, depth: int,
           timeout_s: int, root: Path) -> GradeResult:
    runs: list[RunEvidence] = []
    if not sby_available():
        return GradeResult(Tier.ERROR,
                           "sby not found on PATH (activate hwtools)", runs)
    if not verilog_file.is_file():
        return GradeResult(Tier.ERROR,
                           f"verilog file not found: {verilog_file}", runs)

    prove = run_sby(verilog_file.stem, verilog_file, prop.top_module,
                    "prove", depth, timeout_s, root)

    if prove.rc is None:
        runs.append(_evidence("prove", prove, depth,
                              timeout_source="outer_guard"))
        return GradeResult(Tier.TIMEOUT,
                           "prove run killed by outer guard "
                           "(sby itself hung)", runs)
    if prove.rc == 8:
        runs.append(_evidence("prove", prove, depth, timeout_source="sby"))
        return GradeResult(Tier.TIMEOUT,
                           f"prove run hit sby timeout ({timeout_s}s)", runs)
    if prove.rc == 2:
        runs.append(_evidence("prove", prove, depth))
        return GradeResult(Tier.FALSE,
                           "counterexample reachable from reset "
                           "(trace in evidence)", runs)
    if prove.rc not in (0, 4):
        runs.append(_evidence("prove", prove, depth))
        return GradeResult(Tier.ERROR,
                           f"judge did not run (prove rc={prove.rc})", runs)

    pass_tier = Tier.PROVEN if prove.rc == 0 else Tier.NOT_INDUCTIVE
    pass_reason = (
        f"proven by k-induction at depth {depth}"
        if prove.rc == 0 else
        f"no bug to depth {depth} but induction did not close "
        "with the supplied invariants (CTI trace in evidence)")

    prove_ev = _evidence("prove", prove, depth)
    runs.append(prove_ev)

    if not prop.antecedents:
        prove_ev.notes.append(NA_NOTE)
        return GradeResult(pass_tier,
                           f"{pass_reason}; vacuity check not applicable "
                           "(no antecedent)", runs)

    try:
        inj = inject_covers(verilog_file.read_text(), prop.top_module,
                            prop.clock, prop.antecedents, prop.sanity_covers)
    except InjectionError as exc:
        return GradeResult(Tier.ERROR, f"cover injection failed: {exc}", runs)

    with tempfile.TemporaryDirectory() as td:
        inj_path = Path(td) / verilog_file.name
        inj_path.write_text(inj.text)
        cover = run_sby(verilog_file.stem, inj_path, prop.top_module,
                        "cover", depth, timeout_s, root)

    if cover.rc is None:
        runs.append(_evidence("cover", cover, depth,
                              timeout_source="outer_guard"))
        return GradeResult(Tier.TIMEOUT,
                           "cover run killed by outer guard — no verdict "
                           "without completed vacuity evidence", runs)
    if cover.rc == 8:
        runs.append(_evidence("cover", cover, depth, timeout_source="sby"))
        return GradeResult(Tier.TIMEOUT,
                           f"cover run hit sby timeout ({timeout_s}s) — no "
                           "verdict without completed vacuity evidence", runs)
    if cover.rc not in (0, 2):  # rc=2 just means some cover unreached
        runs.append(_evidence("cover", cover, depth))
        return GradeResult(Tier.ERROR,
                           f"cover run failed to run (rc={cover.rc})", runs)

    cover_ev = _evidence("cover", cover, depth)
    runs.append(cover_ev)
    reached_lines, unreached_lines = parse_cover_log(cover.log_text)

    unreached_antecedents: list[str] = []
    missing_antecedents: list[str] = []
    sanity_ok = True
    for lineno, (kind, expr) in inj.line_map.items():
        if lineno in reached_lines:
            cover_ev.reached_covers.append(expr)
        elif lineno in unreached_lines:
            cover_ev.unreached_covers.append(expr)
            if kind == "antecedent":
                unreached_antecedents.append(expr)
            else:
                sanity_ok = False
        else:
            if kind == "antecedent":
                missing_antecedents.append(expr)
            else:
                sanity_ok = False

    if not sanity_ok:
        cover_ev.notes.append(SANITY_NOTE)

    if missing_antecedents:
        return GradeResult(
            Tier.ERROR,
            "cover result missing from log for antecedent(s): "
            f"{', '.join(missing_antecedents)} — instrument integrity "
            "failure, no verdict without complete evidence", runs)

    if unreached_antecedents:
        return GradeResult(
            Tier.VACUOUS,
            "antecedent(s) unreachable within depth "
            f"{depth}: {', '.join(unreached_antecedents)} — "
            "the property proves nothing", runs)

    return GradeResult(pass_tier,
                       f"{pass_reason}; all antecedent(s) reachable", runs)
