"""Tests for the riscv-formal induction survey (no sby): the prove-mode
variant transform and the verdict classification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from riscv_survey import classify, prove_variant, survey_files

SBY = """[options]
mode bmc
expect pass,fail
append 0
depth 31
skip 30

[engines]
smtbmc boolector
"""


def test_prove_variant_flips_mode_depth_and_expect():
    out = prove_variant(SBY, k=10)
    assert "mode prove" in out and "mode bmc" not in out
    assert "depth 10" in out and "depth 31" not in out
    assert "expect pass,fail,unknown" in out
    assert "skip" not in out          # skip is a bmc-only knob
    assert "smtbmc boolector" in out  # engine untouched


def test_classify_reads_sby_outcomes():
    assert classify(0, "successful proof by k-induction.") == "PROVED"
    assert classify(4, "engine_0 returned FAIL for induction") == "NEEDS_INVARIANT"
    assert classify(2, "returned FAIL for basecase") == "BASECASE_FAIL"
    assert classify(None, "") == "TIMEOUT"
    assert classify(16, "ERROR: something") == "ERROR"


def test_survey_files_skips_leftovers(tmp_path):
    for name in ["reg_ch0", "reg_ch0_induct", "reg_ch0_prove10",
                 "cover", "insn_add_ch0"]:
        (tmp_path / f"{name}.sby").write_text("")
    assert [f.stem for f in survey_files(tmp_path)] == ["insn_add_ch0",
                                                        "reg_ch0"]
