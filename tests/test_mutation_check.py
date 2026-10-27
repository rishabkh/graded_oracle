"""The breaking test (27 Oct 2026, Rishab: "Find weak ones by breaking the
design on purpose ... see if the property notices. If it doesn't notice,
it's weak. A free tool called MCY does this").

What MCY is told is tested without tools: the changes it may make (never
to the property's own logic), how each change is tagged, the side-by-side
copy that shows whether a change is visible at the outputs, and the score.
One small design is run through MCY itself; it skips without hwtools."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import mutation_check                                          # noqa: E402

PORTS = [("clk", "input", 1), ("din", "input", 8), ("go", "input", 1),
         ("dout", "output", 8), ("busy", "output", 1)]


def test_changes_skip_the_property_own_logic():
    text = mutation_check.config_text("top")
    select = next(line for line in text.splitlines() if line.startswith("select "))
    # all cells minus the cells that reach the checks but no output
    assert select.startswith("select c:* t:$check t:$assert")
    assert select.endswith("%ci* o:* %ci* %d %d")
    assert "t:$assume" in select


def test_each_change_is_tagged_by_the_property_alone_by_default():
    text = mutation_check.config_text("top", size=7, seed=3)
    assert "size 7" in text and "seed 3" in text and "prep -top top" in text
    assert 'tag("NOTICED")' in text and 'tag("UNNOTICED")' in text
    assert "test_eq" not in text            # the slow output comparison is off


def test_compare_outputs_splits_off_changes_that_show_nowhere():
    text = mutation_check.config_text("top", compare=True)
    assert "[test test_eq]" in text and 'tag("NOCHANGE")' in text
    # the outputs are compared only when the property did not notice
    logic = mutation_check.LOGIC_COMPARE
    assert logic.index('result("test_fm")') < logic.index('result("test_eq")')


def test_side_by_side_copy_compares_every_output():
    sv = mutation_check.miter_sv("top", PORTS)
    assert "input wire clk," in sv and "input wire [7:0] din," in sv
    assert "input wire go\n" in sv
    assert "top ref (.mutsel(8'd0), .clk(clk), .din(din), .go(go)" in sv
    assert "top uut (.mutsel(8'd1)" in sv
    assert "assert (ref_dout == uut_dout);" in sv
    assert "assert (ref_busy == uut_busy);" in sv


def test_only_the_comparison_test_adds_the_switch_input():
    assert "create_mutated.sh -c -o mutated.il" in mutation_check.test_script("test_eq", ctrl=True)
    fm = mutation_check.test_script("test_fm", ctrl=False)
    assert "create_mutated.sh -o mutated.il" in fm and "-c" not in fm.split("create_mutated.sh")[1].split("\n")[0]


def test_score_counts_only_changes_that_show():
    s = mutation_check.parse_score("Coverage...\nMCYSCORE noticed=6 unnoticed=2 nochange=15 error=1\n")
    assert s == {"noticed": 6, "unnoticed": 2, "nochange": 15, "error": 1, "score": 0.75}
    assert mutation_check.parse_score("MCYSCORE noticed=0 unnoticed=0 nochange=4 error=0")["score"] is None
    assert mutation_check.parse_score("no line here") is None


# a counter that wraps at 5 and a property about it: most changes to the
# counter break the property
COUNTER = """module ctr (input wire clk, input wire en, output wire [2:0] q);
    reg [2:0] c = 3'd0;
    always @(posedge clk) if (en) c <= (c == 3'd4) ? 3'd0 : c + 3'd1;
    assign q = c;
    always @(posedge clk) assert (c <= 3'd4);
endmodule
"""


@pytest.mark.skipif(shutil.which("mcy") is None or shutil.which("sby") is None,
                    reason="mcy/sby not on PATH (run hwtools)")
def test_mcy_runs_and_the_property_notices_changes_to_the_counter(tmp_path):
    row = mutation_check.check_design(
        {"key": "ctr", "verilog": COUNTER, "top_module": "ctr"}, tmp_path,
        size=6, depth=12, timeout_s=30)
    assert row["status"] == "ok", row
    assert row["noticed"] + row["unnoticed"] + row["nochange"] + row["error"] == 6
    assert row["noticed"] >= 1
