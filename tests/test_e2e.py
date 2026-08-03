"""End-to-end: every exemplar triple through real sby into its tier.

Requires sby on PATH (activate hwtools first); skips otherwise.
"""
import json
from pathlib import Path

import pytest

from oracle import NecessityVerdict, PropertyInfo, Tier, grade, grade_triple
from oracle.sby import sby_available

TRIPLES = Path(__file__).parent / "triples"

requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH — activate hwtools first")


def run_triple(name: str, tmp_path: Path):
    info = json.loads((TRIPLES / f"{name}.json").read_text())
    prop = PropertyInfo(
        top_module=info["top_module"],
        clock=info.get("clock", "clk"),
        antecedents=info.get("antecedents", []),
        sanity_covers=info.get("sanity_covers", []))
    result = grade(TRIPLES / info["sv"], prop,
                   workdir_root=tmp_path / "runs",
                   **info.get("grade_kwargs", {}))
    assert result.tier.name == info["expected_tier"], result.reason
    return result


@requires_sby
def test_proven_with_antecedent(tmp_path):
    r = run_triple("counter_proven", tmp_path)
    cover = r.runs[1]
    assert cover.mode == "cover"
    assert "past_rst" in cover.reached_covers
    assert "count == 4'd15" in cover.reached_covers
    assert cover.unreached_covers == []


@requires_sby
def test_proven_no_antecedent_skips_cover(tmp_path):
    r = run_triple("kitest_strong", tmp_path)
    assert len(r.runs) == 1
    assert any("not_applicable" in n for n in r.runs[0].notes)


@requires_sby
def test_not_inductive_carries_cti(tmp_path):
    r = run_triple("kitest_weak", tmp_path)
    assert r.runs[0].rc == 4
    assert any("induct" in p.name for p in r.runs[0].trace_paths)


@requires_sby
def test_false_carries_cex(tmp_path):
    r = run_triple("counter_false", tmp_path)
    assert r.runs[0].rc == 2
    assert r.runs[0].trace_paths


@requires_sby
def test_vacuous_names_the_antecedent(tmp_path):
    r = run_triple("counter_vacuous", tmp_path)
    cover = r.runs[1]
    assert "count == 5'd20" in cover.unreached_covers
    assert "count == 4'd15" in cover.reached_covers
    assert not any("sanity_cover_unreached" in n for n in cover.notes)


@requires_sby
def test_error_on_broken_syntax(tmp_path):
    r = run_triple("broken", tmp_path)
    assert r.runs[0].rc not in (0, 2, 4, 8, None)
    assert r.runs[0].log_excerpt  # the Fixer's food: what went wrong


@requires_sby
def test_timeout_records_source(tmp_path):
    r = run_triple("slow_factor", tmp_path)
    assert r.runs[0].timeout_source in ("sby", "outer_guard")


# --- necessity criterion, live ---

@requires_sby
def test_necessary_kitest_strengthening_is_load_bearing(tmp_path):
    # kitest_weak.sv = design + property only; the strengthening is
    # supplied separately and injected by the oracle. This is the real
    # Stage-4 triple shape.
    prop = PropertyInfo(top_module="kitest", clock="i_clk",
                        invariants=["sa == sb"])
    r = grade_triple(TRIPLES / "kitest_weak.sv", prop,
                     workdir_root=tmp_path / "runs", timeout_s=120)
    assert r.verdict is NecessityVerdict.NECESSARY, r.reason
    assert r.with_invariants.tier is Tier.PROVEN
    assert r.without_invariants.tier is Tier.NOT_INDUCTIVE
    assert any("induct" in p.name
               for p in r.without_invariants.runs[0].trace_paths)


@requires_sby
def test_decorative_tautology_invariant_is_rejected(tmp_path):
    # count <= 4'd15 is true by the type system — the counter proves its
    # property with or without it. Exactly the reviewer's trap.
    prop = PropertyInfo(top_module="counter",
                        invariants=["count <= 4'd15"])
    r = grade_triple(TRIPLES / "counter_proven.sv", prop,
                     workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.DECORATIVE, r.reason
    assert r.with_invariants.tier is Tier.PROVEN
    assert r.without_invariants.tier is Tier.PROVEN
