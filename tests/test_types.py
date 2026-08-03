from pathlib import Path

from oracle.types import (GradeResult, NecessityVerdict, PropertyInfo,
                          RunEvidence, Tier, TripleResult)


def test_tier_ordering_matches_spec():
    assert (Tier.ERROR < Tier.TIMEOUT < Tier.FALSE < Tier.VACUOUS
            < Tier.NOT_INDUCTIVE < Tier.BOUNDED < Tier.PROVEN)


def test_property_info_defaults_are_not_shared():
    a = PropertyInfo(top_module="m")
    b = PropertyInfo(top_module="n")
    assert a.clock == "clk"
    a.antecedents.append("x")
    a.sanity_covers.append("y")
    assert b.antecedents == [] and b.sanity_covers == []


def test_run_evidence_defaults():
    ev = RunEvidence(mode="prove", rc=0, depth=20, engine="smtbmc yices",
                     duration_s=1.0, workdir=Path("/tmp/x"), log_excerpt="")
    assert ev.trace_paths == [] and ev.reached_covers == []
    assert ev.unreached_covers == [] and ev.notes == []
    assert ev.timeout_source is None
    assert ev.failed_assert_lines == []
    assert ev.trace_text is None


def test_grade_result_defaults():
    r = GradeResult(tier=Tier.PROVEN, reason="ok")
    assert r.runs == []


def test_property_info_invariants_default_not_shared():
    a = PropertyInfo(top_module="m")
    b = PropertyInfo(top_module="n")
    a.invariants.append("x == y")
    assert b.invariants == []


def test_necessity_verdict_members():
    assert {v.name for v in NecessityVerdict} == {
        "NECESSARY", "DECORATIVE", "NOT_PROVEN", "INCONCLUSIVE",
        "NO_INVARIANTS"}


def test_triple_result_shape():
    r = TripleResult(verdict=NecessityVerdict.NO_INVARIANTS,
                     reason="no invariants supplied")
    assert r.with_invariants is None and r.without_invariants is None
