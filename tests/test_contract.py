import json

import pytest

from oracle.contract import ContractViolation, GeneratorOutput, parse_generator_output

VALID = {
    "verilog": "module counter (input wire clk);\n    always @(posedge clk) assert (1'b1);\nendmodule\n",
    "top_module": "counter",
    "clock": "clk",
    "antecedents": ["past_rst"],
    "sanity_covers": ["count == 4'd15"],
    "invariants": ["count <= 4'd15"],
}


def test_valid_full_object():
    out = parse_generator_output(json.dumps(VALID))
    assert isinstance(out, GeneratorOutput)
    assert out.verilog == VALID["verilog"]
    assert out.prop.top_module == "counter"
    assert out.prop.clock == "clk"
    assert out.prop.antecedents == ["past_rst"]
    assert out.prop.sanity_covers == ["count == 4'd15"]
    assert out.prop.invariants == ["count <= 4'd15"]


def test_optional_fields_default():
    minimal = {"verilog": "module m ();\nendmodule\n", "top_module": "m"}
    out = parse_generator_output(json.dumps(minimal))
    assert out.prop.clock == "clk"
    assert out.prop.antecedents == []
    assert out.prop.sanity_covers == []
    assert out.prop.invariants == []


def test_markdown_fenced_json_accepted():
    fenced = "```json\n" + json.dumps(VALID) + "\n```"
    out = parse_generator_output(fenced)
    assert out.prop.top_module == "counter"


def test_not_json_is_violation():
    with pytest.raises(ContractViolation):
        parse_generator_output("module m (); endmodule")


def test_json_array_is_violation():
    with pytest.raises(ContractViolation):
        parse_generator_output(json.dumps([VALID]))


def test_missing_verilog_is_violation():
    bad = {k: v for k, v in VALID.items() if k != "verilog"}
    with pytest.raises(ContractViolation, match="verilog"):
        parse_generator_output(json.dumps(bad))


def test_empty_verilog_is_violation():
    bad = dict(VALID, verilog="   \n")
    with pytest.raises(ContractViolation, match="verilog"):
        parse_generator_output(json.dumps(bad))


def test_missing_top_module_is_violation():
    bad = {k: v for k, v in VALID.items() if k != "top_module"}
    with pytest.raises(ContractViolation, match="top_module"):
        parse_generator_output(json.dumps(bad))


def test_invalid_identifier_top_module_is_violation():
    bad = dict(VALID, top_module="2bad name")
    with pytest.raises(ContractViolation, match="identifier"):
        parse_generator_output(json.dumps(bad))


def test_top_module_absent_from_verilog_is_violation():
    bad = dict(VALID, top_module="other")
    with pytest.raises(ContractViolation, match="not found"):
        parse_generator_output(json.dumps(bad))


def test_top_module_only_in_comment_is_violation():
    bad = dict(VALID, verilog="// module other\n" + VALID["verilog"],
               top_module="other")
    with pytest.raises(ContractViolation, match="not found"):
        parse_generator_output(json.dumps(bad))


def test_non_list_antecedents_is_violation():
    bad = dict(VALID, antecedents="past_rst")
    with pytest.raises(ContractViolation, match="antecedents"):
        parse_generator_output(json.dumps(bad))


def test_empty_string_in_list_is_violation():
    bad = dict(VALID, invariants=["sa == sb", ""])
    with pytest.raises(ContractViolation, match="invariants"):
        parse_generator_output(json.dumps(bad))


# --- hierarchical references: yosys's open frontend silently turns
# `u0.count` into a NEW dangling wire (warning: implicitly declared),
# so a dotted expression is semantically inert, never what was meant ---

def test_hierarchical_ref_in_invariants_rejected():
    with pytest.raises(ContractViolation, match="hierarchical"):
        parse_generator_output(json.dumps({
            "verilog": "module m (input wire clk); always @(*) assert (1); endmodule",
            "top_module": "m",
            "invariants": ["u0.count <= 4'd12"]}))


def test_hierarchical_ref_in_antecedents_rejected():
    with pytest.raises(ContractViolation, match="hierarchical"):
        parse_generator_output(json.dumps({
            "verilog": "module m (input wire clk); always @(*) assert (1); endmodule",
            "top_module": "m",
            "antecedents": ["u1.busy"]}))


def test_plain_expressions_still_accepted():
    out = parse_generator_output(json.dumps({
        "verilog": "module m (input wire clk); always @(*) assert (1); endmodule",
        "top_module": "m",
        "invariants": ["c0 <= 4'd12", "{1'b0, a} + {1'b0, b} == 5'd8"]}))
    assert out.prop.invariants[0] == "c0 <= 4'd12"
