import pytest

from oracle.inject import (Injection, InjectionError, inject_covers,
                           inject_invariants)

SIMPLE = """module m (
    input wire clk,
    output reg q
);
    always @(posedge clk) q <= ~q;
endmodule
"""

TWO_MODULES = """module top (input wire clk);
    wire w;
    helper h (.clk(clk));
endmodule

module helper (input wire clk);
    reg r;
endmodule
"""

COMMENT_TRAP = """module m (input wire clk);
    // a fake endmodule in a comment
    /* endmodule also here */
    reg r;
endmodule
"""


def _line(text: str, n: int) -> str:
    return text.splitlines()[n - 1]


def test_basic_injection_and_line_map():
    inj = inject_covers(SIMPLE, "m", "clk", ["a == 1"], ["b == 2"])
    lines = inj.text.splitlines()
    assert any("ORACLE-INJECTED VACUITY CHECK" in l for l in lines)
    assert len(inj.line_map) == 2
    kinds = []
    for lineno, (kind, expr) in sorted(inj.line_map.items()):
        assert f"cover ({expr});" in _line(inj.text, lineno)
        kinds.append(kind)
    assert kinds == ["antecedent", "sanity"]
    # block sits inside the module, before endmodule
    assert inj.text.rstrip().endswith("endmodule")


def test_injects_into_named_module_not_last():
    inj = inject_covers(TWO_MODULES, "top", "clk", ["w"], [])
    end_of_top = inj.text.index("endmodule")
    cover_pos = inj.text.index("cover (w);")
    assert cover_pos < end_of_top  # inside `top`, not appended to `helper`


def test_endmodule_in_comment_is_ignored():
    inj = inject_covers(COMMENT_TRAP, "m", "clk", ["r"], [])
    # cover must land after the real code, before the real endmodule
    assert inj.text.index("cover (r);") > inj.text.index("reg r;")


def test_clock_name_is_used():
    inj = inject_covers(SIMPLE, "m", "i_clk", ["a"], [])
    assert "always @(posedge i_clk) begin" in inj.text


def test_missing_top_module_raises():
    with pytest.raises(InjectionError):
        inject_covers(SIMPLE, "nope", "clk", ["a"], [])


def test_missing_endmodule_raises():
    with pytest.raises(InjectionError):
        inject_covers("module m (input wire clk);\n  reg r;\n", "m", "clk", ["a"], [])


def test_invariant_injection_combinational_asserts():
    inj = inject_invariants(SIMPLE, "m", ["sa == sb", "cnt <= 4'd9"])
    assert "ORACLE-INJECTED INVARIANTS" in inj.text
    assert "always @(*) begin" in inj.text
    for lineno, (kind, expr) in sorted(inj.line_map.items()):
        assert kind == "invariant"
        assert f"assert ({expr});" in _line(inj.text, lineno)
    assert len(inj.line_map) == 2
    assert inj.text.rstrip().endswith("endmodule")


def test_invariant_injection_targets_named_module():
    inj = inject_invariants(TWO_MODULES, "top", ["w"])
    assert inj.text.index("assert (w);") < inj.text.index("endmodule")


def test_invariant_injection_missing_module_raises():
    with pytest.raises(InjectionError):
        inject_invariants(SIMPLE, "nope", ["a"])


def test_invariant_injection_empty_list_is_identity():
    inj = inject_invariants(SIMPLE, "m", [])
    assert inj.text == SIMPLE
    assert inj.line_map == {}
