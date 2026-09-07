"""Tests for the EBMC adapter (no ebmc binary needed): lemma wrapping,
insertion point, directive assembly, and verdict parsing - all ported to
match large_lemma_miners/src/evaluation.py, not our own conventions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from ebmc_eval import (build_variant, clocking_prefix, ebmc_command,
                       insert_index, parse_verdict, wrap_lemmas)

BENCH = """module main(input clk, input rst);
reg [3:0] x;
property prop;
  x <= 4'd8;
endproperty
endmodule
"""


def test_wrap_lemmas_clocked_like_their_fewshot_pool():
    # their canonical form: property lemma_1; @(posedge clk) disable iff
    # (rst) (<expr>); endproperty
    out = wrap_lemmas(["a == b"], "@(posedge clk) disable iff (rst)")
    assert out[0] == ("property lemma0; @(posedge clk) disable iff (rst) "
                      "(a == b); endproperty")
    assert wrap_lemmas(["x"], "") == ["property lemma0; (x); endproperty"]


def test_clocking_prefix_stolen_from_the_files_own_prop():
    assert clocking_prefix(BENCH) == ""              # BENCH prop is bare
    src = BENCH.replace("x <= 4'd8;",
                        "@(posedge clk) disable iff (!rstN) x <= 4'd8;")
    assert clocking_prefix(src) == "@(posedge clk) disable iff (!rstN)"


def test_insert_index_is_after_last_endproperty():
    lines = BENCH.splitlines(keepends=True)
    assert insert_index(lines) == 5      # line after 'endproperty'


def test_build_variant_close_mode_assumes_lemmas_asserts_prop():
    v = build_variant(BENCH, ["a == b"], "timing_with_lemmas")
    assert "property lemma0; (a == b); endproperty" in v
    assert "assume property (lemma0);" in v
    assert "assert property (prop);" in v
    # lemmas inserted after the benchmark's endproperty, before endmodule
    assert v.index("endproperty") < v.index("lemma0")
    assert v.index("assert property (prop)") < v.index("endmodule")


def test_build_variant_correctness_asserts_lemma_conjunction():
    v = build_variant(BENCH, ["a == b", "c == d"], "correctness")
    assert "assert property(lemma0 and lemma1);" in v
    assert "assume property" not in v


def test_ebmc_command_matches_their_modes():
    assert ebmc_command("f.sv", "k_induction", rst=True) == \
        "ebmc f.sv --k-induction --bound 5 --reset main.rst --trace"
    assert ebmc_command("f.sv", "timing_with_lemmas", rst=False) == \
        "ebmc f.sv --bound 30 --trace"
    assert ebmc_command("f.sv", "one_inductive_with_prop", rst=False) == \
        "ebmc f.sv --k-induction --bound 1 --trace"
    assert ebmc_command("f.sv", "correctness", rst=False) == \
        "ebmc f.sv --trace"


def test_parse_verdict_ported_semantics():
    assert parse_verdict("** Results: ... PROVED", "", "k_induction") == "PROVEN"
    assert parse_verdict("blah REFUTED\nCounterexample:\n  x=1", "", "k_induction") == "CEX"
    assert parse_verdict("INCONCLUSIVE", "", "k_induction") == "INCONCLUSIVE"
    assert parse_verdict("FAILURE: line 3: bad", "", "k_induction") == "ERROR"
    assert parse_verdict("", "boom", "k_induction") == "ERROR"
    # bounded-only proof counts as not-proven (their TIMEOUT bucket)
    assert parse_verdict("** Results: PROVED up to bound 30", "",
                         "timing_with_lemmas") == "TIMEOUT"
    # correctness mode demands the strict Results-line match
    assert parse_verdict("some PROVED text", "", "correctness") == "TIMEOUT"
