"""Plugging a model's answer into a real riscv-formal check.

The model is asked for invariants over the core's own signals, so they
are injected INSIDE the core module, guarded by its reset, and the check
then runs unchanged. Everything here is text work: no sby, no yosys."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from riscv_adapter import CORES, inject_invariants

TWO_MODULES = """module helper (input a);
  reg x;
endmodule

module nerv #(
  parameter integer NUMREGS = 32
) (
  input clock,
  input reset,
  output trap
);
  reg [31:0] pc;
  always @(posedge clock) pc <= pc + 4;
endmodule
"""


def test_invariants_land_inside_the_named_module():
    out = inject_invariants(TWO_MODULES, "nerv", "clock", "reset",
                            ["pc[1:0] == 2'b00"])
    body = out.split("module nerv")[1]
    assert "pc[1:0] == 2'b00" in body          # in nerv, not in helper
    assert out.count("endmodule") == 2         # both modules still closed


def test_each_invariant_becomes_a_reset_guarded_assertion():
    out = inject_invariants(TWO_MODULES, "nerv", "clock", "reset",
                            ["pc[1:0] == 2'b00", "pc < 32'h1000"])
    assert "always @(posedge clock)" in out
    assert "!reset" in out
    assert out.count("assert (") == 2


def test_no_invariants_leaves_the_source_untouched():
    assert inject_invariants(TWO_MODULES, "nerv", "clock", "reset",
                             []) == TWO_MODULES


def test_trailing_semicolons_and_wrapping_are_tolerated():
    out = inject_invariants(TWO_MODULES, "nerv", "clock", "reset",
                            ["  (pc < 32'h1000) ;  "])
    assert "assert ((pc < 32'h1000));" in out or \
           "assert (pc < 32'h1000);" in out
    assert ";;" not in out


def test_an_unknown_module_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError):
        inject_invariants(TWO_MODULES, "picorv32", "clk", "resetn",
                          ["1'b1"])


def test_the_three_cores_are_described():
    for core in ("nerv", "serv", "picorv32"):
        entry = CORES[core]
        assert entry["file"] and entry["module"]
        assert entry["clock"] and entry["reset"]


SBY_LINE = ("[script]\n"
            "read -sv bus_dmem_ch0.sv "
            "/a/cores/nerv/../../cores/nerv/wrapper.sv "
            "/a/cores/nerv/../../cores/nerv/nerv.sv\n"
            "prep -top rvfi_testbench\n")


def test_retarget_catches_the_unnormalised_path_the_checks_use():
    """The generated checks reference the core as
    cores/nerv/../../cores/nerv/nerv.sv, so a plain swap of the resolved
    path misses, the patched core is never read, and the model is scored
    on a design that never saw its invariants."""
    from riscv_adapter import retarget
    out = retarget(SBY_LINE, "nerv.sv", "/tmp/work/nerv.sv")
    assert "/tmp/work/nerv.sv" in out
    assert "cores/nerv/nerv.sv" not in out
    assert "wrapper.sv" in out            # other files untouched
    assert "bus_dmem_ch0.sv" in out
