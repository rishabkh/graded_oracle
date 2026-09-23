"""Tests for the EBMC adapter (no ebmc binary needed): lemma wrapping,
insertion point, directive assembly, and verdict parsing - all ported to
match large_lemma_miners/src/evaluation.py, not our own conventions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from ebmc_eval import (build_variant, clocking_prefix, ebmc_command,
                       top_module,
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


TWO_MODULES = """module fifo (input clk, input rst);
endmodule
module inv_80 (input clk, input rst);
  property prop; @(posedge clk) x <= 4; endproperty
  assert property (prop);
endmodule
"""


def test_top_module_is_main_when_present_else_the_last_one():
    assert top_module("module main (input clk, input rst);\nendmodule\n") == "main"
    assert top_module(TWO_MODULES) == "inv_80"


def test_reset_flag_names_the_files_own_top_module():
    cmd = ebmc_command("f.sv", "k_induction", rst=True, source=TWO_MODULES)
    assert "--reset inv_80.rst" in cmd


def test_timeout_is_a_setting_and_defaults_to_their_protocol(monkeypatch):
    """Their paper uses 120s. Changing it is allowed but must be visible:
    both sides of a before/after comparison have to use the same value,
    and the deviation has to be reportable."""
    import importlib
    import ebmc_eval as E
    monkeypatch.delenv("EBMC_TIMEOUT_S", raising=False)
    importlib.reload(E)
    assert E.TIMEOUT_S == 120
    monkeypatch.setenv("EBMC_TIMEOUT_S", "240")
    importlib.reload(E)
    assert E.TIMEOUT_S == 240
    monkeypatch.delenv("EBMC_TIMEOUT_S")
    importlib.reload(E)


def test_judgeability_is_measured_once_and_cached(tmp_path):
    """A file whose own property EBMC cannot k-induct errors with no
    lemmas attached, so no answer can ever succeed on it. That fact
    belongs in the record, not in somebody's head: 9 of 78 easy files
    and 5 of 31 hard ones are in this class."""
    from ebmc_eval import judgeable
    calls = []

    def fake_run(path, lemmas, mode, workdir):
        calls.append(path)
        return {"verdict": "ERROR" if "bad" in str(path) else "INCONCLUSIVE"}

    cache = tmp_path / "judgeable.json"
    assert judgeable(tmp_path / "good.sv", cache, _run=fake_run) is True
    assert judgeable(tmp_path / "bad.sv", cache, _run=fake_run) is False
    assert len(calls) == 2
    # second time comes from the cache, not from ebmc
    assert judgeable(tmp_path / "bad.sv", cache, _run=fake_run) is False
    assert len(calls) == 2
    assert "bad.sv" in cache.read_text()
