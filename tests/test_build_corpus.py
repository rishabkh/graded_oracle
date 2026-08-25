"""Unit tests for the corpus flattener (extender step 1) — the pure parts:
template normalization, property extraction, record flattening. The yosys
state-bits metric is exercised by the live build, not unit-tested here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extender.build_corpus import (   # noqa: E402
    count_dff_bits, extract_asserts, flatten_record, pdr_wall_s, template)


# --- stat parsing: verbatim block from a real yosys 0.67 run ---

STAT_OUTPUT = """\
7. Printing statistics.

=== sorted_pair ===

        +----------Local Count, excluding submodules.
        |
       73 wires
      202 wire bits
       12 public wires
       41 public wire bits
        7 ports
       15 port bits
      180 cells
       31   $_AND_
       60   $_MUX_
       20   $_NOT_
       24   $_OR_
       26   $_SDFFE_PP0P_
       18   $_XOR_
        1   $check

End of script. Logfile hash: a4f9d6bc21
"""


def test_count_dff_bits_from_real_stat_output():
    # only the 26 flop cells count; $_AND_/$_MUX_/wire counts must not
    assert count_dff_bits(STAT_OUTPUT) == 26


def test_count_dff_bits_sums_multiple_dff_types():
    out = "       10   $_DFF_P_\n        5   $_SDFFE_PP0P_\n        3   $_AND_\n"
    assert count_dff_bits(out) == 15


def test_count_dff_bits_none_when_no_flops():
    assert count_dff_bits("       31   $_AND_\n       60   $_MUX_\n") is None


# --- template: identifiers -> id, constants -> N, structure preserved ---

def test_template_same_family_same_template():
    # the one-hot-shift family: different names/widths, one shape
    assert template("mask == (8'h01 << ptr)") == template("pri == (4'b0001 << idx)")


def test_template_shape_is_stable():
    assert template("gray == (bin ^ (bin >> 1))") == "id == ( id ^ ( id >> N ) )"


def test_template_sized_literals_and_bare_numbers_both_mask():
    assert template("units <= 3'd4") == template("units <= 4")


def test_template_keeps_system_functions():
    t = template("$onehot(active)")
    assert "$onehot" in t
    assert "active" not in t


def test_template_distinct_shapes_stay_distinct():
    assert template("a + b == c") != template("a == (b << c)")


# --- extract_asserts: the property lives inside the verilog ---

def test_extract_asserts_nested_parens():
    v = "module m; always @(*) begin if (x) assert ((a + b) == c); end endmodule"
    assert extract_asserts(v) == ["(a + b) == c"]


def test_extract_asserts_ignores_commented_asserts():
    v = "module m; // assert (bogus);\nalways @(*) assert (a); endmodule"
    assert extract_asserts(v) == ["a"]


def test_extract_asserts_multiple_in_order():
    v = "module m; always @(posedge clk) begin assert (a); assert (b > 1); end endmodule"
    assert extract_asserts(v) == ["a", "b > 1"]


# --- flatten_record: one attempts.jsonl row -> one corpus row ---

RECORD = {
    "run_id": "2026-08-14_19h03m22s", "attempt": 3, "verdict": "NECESSARY",
    "raw_json": json.dumps({
        "verilog": "module m; always @(*) assert (a == b); endmodule",
        "top_module": "m", "clock": "clk",
        "antecedents": [], "sanity_covers": ["a"],
        "invariants": ["a == b"],
    }),
    "result": {"without_invariants": {"runs": [
        {"mode": "prove", "engine": "smtbmc yices", "duration_s": 0.3, "notes": []},
        {"mode": "prove", "engine": "abc pdr", "duration_s": 1.25,
         "notes": ["pdr_second_opinion: property proven true (unbounded) by "
                   "abc pdr — induction failure is a strengthening gap"]},
    ]}},
}


def test_flatten_record_corpus_fields():
    row = flatten_record(RECORD, idx=5)
    assert row["id"] == "g0_005"
    assert row["generation"] == 0
    assert row["parent"] is None
    assert row["source_run_id"] == "2026-08-14_19h03m22s"
    assert row["source_attempt"] == 3
    assert row["top_module"] == "m"
    assert row["invariants"] == ["a == b"]
    assert row["property"] == ["a == b"]


def test_flatten_record_metrics():
    row = flatten_record(RECORD, idx=0)
    m = row["metrics"]
    assert m["clause_count"] == 1
    assert m["pdr_wall_s"] == 1.25
    assert m["invariant_templates"] == ["id == id"]
    assert m["state_bits"] is None   # yosys runs later, in main()


def test_pdr_wall_s_absent_when_no_pdr_run():
    rec = {"result": {"without_invariants": {"runs": [
        {"mode": "prove", "engine": "smtbmc yices", "duration_s": 0.3, "notes": []}]}}}
    assert pdr_wall_s(rec) is None


# --- zero-extension normalization: {1'b0, x} is width plumbing, not shape ---

def test_template_zero_extension_collapses_to_plain_sum():
    assert template("({1'b0, sent} + {1'b0, remaining}) == 6'd20") == \
           template("(sent + remaining) == 20")


def test_template_zero_extension_multibit_zeros():
    assert template("sq == ({3'b000, cnt} * {3'b000, cnt})") == \
           template("sq == (cnt * cnt)")


def test_template_nonzero_concat_is_not_zero_extension():
    # {1'b1, x} really concatenates a one — a different value, different shape
    assert template("a == {1'b1, x}") != template("a == x")


def test_template_trailing_constant_concat_is_not_zero_extension():
    # odd == {cnt, 1'b1} is shift-and-set, not width extension
    assert template("odd == {cnt, 1'b1}") != template("odd == cnt")
