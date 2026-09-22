"""Cutting a riscv-formal check down to what the property depends on.

The sliced text is what the MODEL reads, never what the checker runs:
grading always uses the original files, so the slice is allowed to be
unparseable Verilog. It has to be readable and honest about what it
removed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from slice_check import parse_src, slice_source, yosys_script

SBY = """[options]
mode prove
depth 10

[engines]
smtbmc boolector

[script]
read -sv check.sv core.v
prep -flatten -top rvfi_testbench
chformal -early

[files]
core.v
"""

CORE = """module nerv (
  input clk,
  input reset,
  output reg [31:0] pc
);
  reg [31:0] regfile [0:31];
  wire [31:0] insn;

  always @(posedge clk) begin
    pc <= pc + 4;
  end

  always @(posedge clk) begin
    trace <= 1;
  end
endmodule
"""


def test_yosys_script_takes_only_the_script_section():
    assert yosys_script(SBY) == ["read -sv check.sv core.v",
                                 "prep -flatten -top rvfi_testbench",
                                 "chformal -early"]


def test_parse_src_reads_every_location_in_one_attribute():
    got = parse_src("core.v:12.3-15.7|wrap.sv:4|core.v:20")
    assert got == [("core.v", 12, 15), ("wrap.sv", 4, 4), ("core.v", 20, 20)]


def test_parse_src_ignores_rubbish():
    assert parse_src("") == []
    assert parse_src("no-location-here") == []


def test_slice_keeps_the_cone_and_every_declaration():
    out = slice_source(CORE, keep={9, 10, 11})       # the pc always block
    assert "pc <= pc + 4;" in out
    for decl in ("module nerv (", "input clk,", "output reg [31:0] pc",
                 "reg [31:0] regfile [0:31];", "wire [31:0] insn;",
                 "endmodule"):
        assert decl in out, decl
    assert "trace <= 1;" not in out                  # outside the cone


def test_slice_says_what_it_removed():
    out = slice_source(CORE, keep={9, 10, 11})
    assert "not in the cone" in out


def test_slice_keeps_source_order_and_no_duplicates():
    out = slice_source(CORE, keep={9, 10, 11})
    kept = [l for l in out.splitlines() if "not in the cone" not in l]
    assert kept == sorted(set(kept), key=kept.index)
    assert kept.index("module nerv (") < kept.index("endmodule")


MACROS = """`define KEEP_ME  /* wanted */ \\
    field_a, \\
    field_b
`define OTHER  /* not wanted */ \\
    field_c
module m (input clk);
  reg [3:0] count;
  always @(posedge clk) count <= count + 1;
endmodule
"""


def test_a_kept_macro_brings_its_continuation_lines():
    out = slice_source(MACROS, keep={1})
    assert "field_a," in out and "field_b" in out


def test_a_macro_outside_the_cone_is_not_kept_just_for_being_a_macro():
    out = slice_source(MACROS, keep={1})
    assert "`define OTHER" not in out
    assert "field_c" not in out


def test_declarations_are_still_kept_without_being_in_the_cone():
    out = slice_source(MACROS, keep={9})
    assert "reg [3:0] count;" in out and "module m (input clk);" in out
