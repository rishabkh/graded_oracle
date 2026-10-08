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


# --- the one-bit enum crash (6 Oct 2026) --------------------------------

import os                                                        # noqa: E402
import shutil                                                    # noqa: E402

import pytest                                                    # noqa: E402

from ebmc_eval import normalize                                  # noqa: E402

ENUM_FILE = (Path(__file__).resolve().parent.parent.parent
             / "large_lemma_miners" / "benchmarks" / "hard"
             / "gulwani_fig1a_ebmc.sv")


def test_a_one_bit_enum_is_written_with_its_bit():
    src = "typedef enum logic {LOOP, DONE} state_t;"
    assert normalize(src) == "typedef enum logic [0:0] {LOOP, DONE} state_t;"


def test_wider_enums_and_other_text_are_left_alone():
    for src in ("typedef enum logic [1:0] {A, B, C} s_t;",
                "typedef enum {A, B} s_t;", "logic x; // enum logic {"):
        assert normalize(src.split("//")[0]) == src.split("//")[0]


def test_every_variant_sent_to_ebmc_is_normalized():
    src = ENUM_FILE.read_text() if ENUM_FILE.exists() else (
        "module m(input clk);\ntypedef enum logic {A, B} s_t;\n"
        "property prop; @(posedge clk) 1; endproperty\nendmodule\n")
    out = build_variant(src, ["1"], "one_inductive_with_prop")
    assert "enum logic {" not in out and "enum logic [0:0] {" in out


@pytest.mark.skipif(not (os.environ.get("EBMC_PATH") and ENUM_FILE.exists()),
                    reason="needs EBMC_PATH and the benchmark checkout")
def test_ebmc_judges_the_enum_family_instead_of_crashing(tmp_path):
    from ebmc_eval import run
    r = run(ENUM_FILE, [], "one_inductive_with_prop", tmp_path)
    assert r["verdict"] == "INCONCLUSIVE"


# --- keep everything (8 Oct 2026): rows kept only the last 400 characters

def test_a_scoring_run_keeps_the_whole_tool_output(tmp_path, monkeypatch):
    import subprocess
    import ebmc_eval
    bench = tmp_path / "b.sv"
    bench.write_text(BENCH)
    long_out = "step line\n" * 500
    monkeypatch.setattr(ebmc_eval.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, stdout=long_out,
                                                    stderr="warn"))
    monkeypatch.setattr(ebmc_eval, "parse_verdict", lambda o, e, m: "PROVEN")
    res = ebmc_eval.run(bench, [], "one_inductive_with_prop", tmp_path)
    assert res["stdout"] == long_out and res["stderr"] == "warn"
    assert res["stdout_tail"] == long_out[-400:]          # unchanged


def test_a_hung_scoring_run_keeps_what_it_printed(tmp_path, monkeypatch):
    import subprocess
    import ebmc_eval
    bench = tmp_path / "b.sv"
    bench.write_text(BENCH)

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("ebmc", 1, output="partial",
                                        stderr=None)
    monkeypatch.setattr(ebmc_eval.subprocess, "run", hang)
    res = ebmc_eval.run(bench, [], "one_inductive_with_prop", tmp_path)
    assert res["verdict"] == "TIMEOUT" and res["stdout"] == "partial"
