"""End-to-end: every exemplar triple through real sby into its tier.

Requires sby on PATH (activate hwtools first); skips otherwise.
"""
import json
from pathlib import Path

import pytest

from oracle import (NecessityVerdict, PropertyInfo, Tier, grade,
                    grade_generated, grade_triple)
from oracle.sby import sby_available

TRIPLES = Path(__file__).parent / "triples"

requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH - activate hwtools first")


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
    # The CTI as prompt-ready text: both registers, differing values.
    text = r.runs[0].trace_text
    assert "Trace summary" in text
    assert "sa = 32'h" in text and "sb = 32'h" in text
    sa = [l for l in text.splitlines() if "sa = " in l]
    sb = [l for l in text.splitlines() if "sb = " in l]
    assert sa[0] != sb[0].replace("sb", "sa")  # the bogus differing state


@requires_sby
def test_false_carries_cex(tmp_path):
    r = run_triple("counter_false", tmp_path)
    assert r.runs[0].rc == 2
    assert r.runs[0].trace_paths
    assert r.runs[0].failed_assert_lines == [15]  # the assert's source line
    assert "Trace summary" in r.runs[0].trace_text


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


# --- depth problem, live ---

@requires_sby
def test_deep_antecedent_rescued_by_pdr_witness(tmp_path):
    # Antecedent reachable only at cycle 40 (> depth 20). The bounded
    # check alone would emit a false VACUOUS; the PDR witness proves the
    # antecedent live and the triple grades PROVEN.
    r = run_triple("deep_antecedent", tmp_path)
    cover = [ev for ev in r.runs if ev.mode == "cover"][0]
    assert "count == 6'd40" in cover.unreached_covers   # bounded probe missed it
    pdr = [ev for ev in r.runs if ev.engine == "abc pdr"][0]
    assert pdr.rc == 2                                  # reachability witness
    assert pdr.trace_paths
    assert any("reachable beyond depth" in n for n in pdr.notes)


@requires_sby
def test_vacuous_verdict_is_now_pdr_proven(tmp_path):
    # counter_vacuous still grades VACUOUS, but the verdict is now an
    # unbounded proof, not a depth guess.
    r = run_triple("counter_vacuous", tmp_path)
    pdr = [ev for ev in r.runs if ev.engine == "abc pdr"][0]
    assert pdr.rc == 0
    assert any("proven unreachable" in n for n in pdr.notes)
    assert "for all time" in r.reason


@requires_sby
def test_kitest_weak_gets_pdr_second_opinion(tmp_path):
    # rc=4 now carries the corpus-protecting second opinion: PDR proves
    # the property true (unbounded), so NOT_INDUCTIVE is legitimate
    # Fixer food — and the necessity of the strengthening is scoped to
    # k-induction, measured, not assumed.
    r = run_triple("kitest_weak", tmp_path)
    pdr = [ev for ev in r.runs if ev.engine == "abc pdr"][0]
    assert pdr.rc == 0
    assert any("proven true" in n for n in pdr.notes)


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


# --- rc=2 attribution, live ---

@requires_sby
def test_falsified_invariant_attributed_live(tmp_path):
    # True property, false invariant (count reaches 9): the Fixer must
    # be told to fix the invariant, not the property.
    prop = PropertyInfo(top_module="counter",
                        invariants=["count != 4'd9"])
    r = grade_triple(TRIPLES / "counter_proven.sv", prop,
                     workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NOT_PROVEN, r.reason
    assert "falsified invariant" in r.reason
    assert "count != 4'd9" in r.reason


@requires_sby
def test_false_property_attributed_live(tmp_path):
    # False property, harmless true invariant: opposite repair route.
    prop = PropertyInfo(top_module="counter_false",
                        invariants=["count <= 4'd15"])
    r = grade_triple(TRIPLES / "counter_false.sv", prop,
                     workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NOT_PROVEN, r.reason
    assert "property itself is false" in r.reason


# --- generator output contract, live ---

@requires_sby
def test_generated_json_grades_proven(tmp_path):
    # Simulates a well-formed Initiator emission: inline verilog +
    # structured metadata, graded end-to-end through real sby.
    output = json.dumps({
        "verilog": (TRIPLES / "counter_proven.sv").read_text(),
        "top_module": "counter",
        "clock": "clk",
        "antecedents": ["past_rst"],
        "sanity_covers": ["count == 4'd15"],
    })
    r = grade_generated(output, workdir_root=tmp_path / "runs")
    assert r.tier is Tier.PROVEN, r.reason
    assert "past_rst" in r.runs[1].reached_covers


def test_malformed_generated_output_is_error_without_sby(tmp_path):
    # No @requires_sby: contract violations are classified before sby
    # is ever needed.
    r = grade_generated("```json\n{\"verilog\": \"module m\"\n```",
                        workdir_root=tmp_path / "runs")
    assert r.tier is Tier.ERROR
    assert "contract violation" in r.reason


@requires_sby
def test_necessary_fifo_count_pointer_link(tmp_path):
    # Hidden facts: count <= 4 and count[1:0] == wptr - rptr, true by
    # construction. Weak claim (empty => pointers agree) follows from
    # them but cannot prove itself.
    prop = PropertyInfo(top_module="fifo",
                        antecedents=["count == 3'd0"],
                        sanity_covers=["count == 3'd2"],
                        invariants=["count <= 3'd4",
                                    "count[1:0] == (wptr - rptr)"])
    r = grade_triple(TRIPLES / "fifo.sv", prop,
                     workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NECESSARY, r.reason
    cover = [ev for ev in r.with_invariants.runs if ev.mode == "cover"][0]
    assert "count == 3'd0" in cover.reached_covers


@requires_sby
def test_necessary_onehot_fsm(tmp_path):
    # Hidden fact: exactly one bit set. Weak claim (bits 0 and 2 never
    # both set) is broken by the legal-but-unreachable state 4'b1010
    # rotating into 4'b0101.
    prop = PropertyInfo(top_module="onehot_fsm",
                        invariants=["(state == 4'b0001) || "
                                    "(state == 4'b0010) || "
                                    "(state == 4'b0100) || "
                                    "(state == 4'b1000)"])
    r = grade_triple(TRIPLES / "onehot_fsm.sv", prop,
                     workdir_root=tmp_path / "runs")
    assert r.verdict is NecessityVerdict.NECESSARY, r.reason
    assert r.without_invariants.runs[0].trace_text  # the CTI, readable


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


# --- composition semantics (the COMPOSE de-risk pair): a plain wire
# between two verified parts destroys necessity (structurally-constrained
# input makes the property inductive unaided); stateful glue that HOLDS
# under idle restores it, because the glue state is new territory no
# parent invariant covers ---

@requires_sby
def test_compose_plain_wire_is_decorative():
    r = grade_triple(TRIPLES / "compose_wire.sv",
                     PropertyInfo(top_module="compose_wire",
                                  invariants=["held_o != 4'b0000"]),
                     timeout_s=120, keep_workdirs=False)
    assert r.verdict is NecessityVerdict.DECORATIVE


@requires_sby
def test_compose_stateful_glue_is_necessary():
    r = grade_triple(
        TRIPLES / "compose_buffer.sv",
        PropertyInfo(top_module="compose_buffer",
                     invariants=["buf_d == 4'b0001 || buf_d == 4'b0010 || "
                                 "buf_d == 4'b0100 || buf_d == 4'b1000"]),
        timeout_s=120, keep_workdirs=False)
    assert r.verdict is NecessityVerdict.NECESSARY
    assert r.without_invariants.tier is Tier.NOT_INDUCTIVE
