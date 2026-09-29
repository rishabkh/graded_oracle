"""What the model is shown on a riscv-formal problem.

Found 28 Sep 2026: the 'native' prompt shows one file per core and the
check's top file. For serv that one file is serv_top.v, which only wires
eleven sub-modules together (no always block at all), and for every core
the check's top file is four include lines, so no model was ever shown
the assertion it was asked to strengthen. The 'full' condition shows
every file the proof tool reads. 'native' stays byte-identical, because
every result so far was measured with it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

import riscv_score as S                                         # noqa: E402

SERV_TOP = """\
module serv_top (input wire clk, input wire i_rst);
   wire cnt0to3;
   wire cnt0;
   serv_state state (.i_clk (clk), .o_cnt0to3 (cnt0to3), .o_cnt0 (cnt0));
endmodule
"""
SERV_STATE = """\
module serv_state (input wire i_clk, output wire o_cnt0to3,
                   output wire o_cnt0);
   reg [4:2] o_cnt;
   reg [3:0] cnt_lsb;
   assign o_cnt0to3 = (o_cnt[4:2] == 3'd0);
   assign o_cnt0    = (o_cnt[4:2] == 3'd0) & cnt_lsb[0];
endmodule
"""
CHECKER = """\
module rvfi_causal_check (input clock, reset, check);
   reg found_non_causal = 0;
   always @(posedge clock) if (check) assert(!found_non_causal);
endmodule
"""
INCLUDES = '`include "defines.sv"\n`include "rvfi_causal_check.sv"\n'


@pytest.fixture
def root(tmp_path):
    """A tiny riscv-formal tree with the same layout as the real one:
    the sby read line names the design by absolute (and unnormalised)
    path, and the check's own sources sit in its unpacked src folder."""
    rtl = tmp_path / "cores/serv/serv-src/rtl"
    rtl.mkdir(parents=True)
    (rtl / "serv_top.v").write_text(SERV_TOP)
    (rtl / "serv_state.v").write_text(SERV_STATE)
    (tmp_path / "cores/serv/wrapper.sv").write_text(
        "module rvfi_wrapper (input clock); endmodule\n")
    checks = tmp_path / "cores/serv/checks"
    src = checks / "causal_ch0_prove10/src"
    src.mkdir(parents=True)
    base = f"{tmp_path}/cores/serv/../../cores/serv"
    (checks / "causal_ch0.sby").write_text(
        "[options]\nmode bmc\n\n[script]\n"
        f"read -sv causal_ch0.sv {base}/wrapper.sv "
        f"{base}/serv-src/rtl/serv_state.v {base}/serv-src/rtl/serv_top.v\n"
        "prep -flatten -nordff -top rvfi_testbench\n")
    (src / "causal_ch0.sv").write_text(INCLUDES)
    (src / "defines.sv").write_text("`define RISCV_FORMAL_CHANNEL_IDX 0\n")
    (src / "rvfi_causal_check.sv").write_text(CHECKER)
    (src / "rvfi_macros.vh").write_text("`define X 1\n" * 5000)
    return tmp_path


def test_native_is_unchanged_one_file_and_the_include_lines(root):
    design, check = S.prompt_texts("serv", "causal_ch0", "native", root)
    assert design == SERV_TOP
    assert check == INCLUDES


def test_full_shows_the_sub_modules_that_define_the_wires(root):
    design, _ = S.prompt_texts("serv", "causal_ch0", "full", root)
    assert "assign o_cnt0to3 = (o_cnt[4:2] == 3'd0);" in design
    assert "module serv_top" in design and "module rvfi_wrapper" in design


def test_full_shows_the_assertion_being_strengthened(root):
    _, check = S.prompt_texts("serv", "causal_ch0", "full", root)
    assert "assert(!found_non_causal);" in check
    assert "RISCV_FORMAL_CHANNEL_IDX" in check


def test_full_leaves_out_the_generated_macro_library(root):
    _, check = S.prompt_texts("serv", "causal_ch0", "full", root)
    assert "`define X 1" not in check


def test_every_file_is_shown_under_a_header_naming_it(root):
    design, check = S.prompt_texts("serv", "causal_ch0", "full", root)
    for name in ("serv_top.v", "serv_state.v", "wrapper.sv"):
        assert f"// ===== {name} =====" in design
    assert "// ===== rvfi_causal_check.sv =====" in check


def test_the_design_files_are_the_absolute_paths_on_the_read_line(root):
    sby = (root / "cores/serv/checks/causal_ch0.sby").read_text()
    names = [Path(p).name for p in S.design_files(sby)]
    assert names == ["wrapper.sv", "serv_state.v", "serv_top.v"]


def test_an_unbuilt_condition_is_refused_rather_than_mislabelled(root):
    """focused and cut are listed but were never built; passing one
    used to produce the native prompt under the wrong label."""
    with pytest.raises(ValueError):
        S.prompt_texts("serv", "causal_ch0", "cut", root)


def test_a_sub_module_register_is_caught_before_the_proof(root):
    """cnt_lsb lives in serv_state. Injected into serv_top it would
    become a free wire and fail for a reason that has nothing to do
    with the model's reasoning."""
    assert S.names_outside_core(["cnt_lsb == 4'd0"], "serv", root) \
        == ["cnt_lsb"]
    assert S.names_outside_core(["!cnt0to3 || !cnt0"], "serv", root) == []
