"""Tests for the verdict router / promotion step (no API, no sby):
routing table, g+1 promotion with parent pointers, content-hash
idempotency, and top-module derivation for wrapper types.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from promote import child_top, promote, route   # noqa: E402


def test_route_covers_the_spec():
    assert route("NECESSARY") == "accept"
    assert route("DECORATIVE") == "reject"        # no repair, by design
    assert route("NOT_PROVEN") == "fixer"
    assert route("NOT_INDUCTIVE") == "fixer"
    assert route("FALSE") == "drop"
    assert route("ERROR") == "retire"
    assert route("TIMEOUT") == "retire"
    assert route("INCONCLUSIVE") == "retire"


def test_route_machinery_verdicts_are_reformat():
    for v in ("PATCH_ERROR", "DISPOSITION_ERROR", "PROPERTY_COPY",
              "NO_AGGREGATE", "NO_GLUE_CLAUSE", "HIERARCHICAL_REF",
              "TRUNCATED", "UNPARSEABLE", "REFUSED", "WRAPPER_ERROR"):
        assert route(v) == "reformat", v


PARENT = {"id": "g0_014", "generation": 0, "top_module": "token_bucket",
          "clock": "clk", "antecedents": [], "sanity_covers": [],
          "verilog": "module token_bucket; endmodule",
          "invariants": ["tokens <= 3'd4"]}

CHILD_V = ("module token_bucket; always @(*) assert (a); endmodule")


def rec(verdict="NECESSARY", **kw):
    base = {"extension_id": "e1", "ext_type": "structural", "move": "GUARD",
            "parent_id": "g0_014", "verdict": verdict,
            "child_verilog": CHILD_V, "invariants": ["deep == (a >= 1)"]}
    base.update(kw)
    return base


def test_promote_writes_generation_one_row():
    buckets, rows = promote([PARENT], [rec()], compute_metrics=False)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "g1_000"
    assert row["generation"] == 1
    assert row["parent"] == "g0_014"
    assert row["invariants"] == ["deep == (a >= 1)"]
    assert row["source_extension_id"] == "e1"


def test_promote_is_idempotent_on_content():
    _, first = promote([PARENT], [rec()], compute_metrics=False)
    # same child again (a rerun of promote over the same log)
    _, second = promote([PARENT] + first, [rec()], compute_metrics=False)
    assert second == []


def test_promote_dedupes_identical_children_in_one_pass():
    _, rows = promote([PARENT], [rec(extension_id="a"), rec(extension_id="b")],
                      compute_metrics=False)
    assert len(rows) == 1


def test_promote_sequences_ids_and_routes_rejects():
    records = [rec(extension_id="a"),
               rec(extension_id="b",
                   child_verilog=CHILD_V.replace("(a)", "(b)")),
               rec(extension_id="c", verdict="NOT_INDUCTIVE"),
               rec(extension_id="d", verdict="FALSE")]
    buckets, rows = promote([PARENT], records, compute_metrics=False)
    assert [r["id"] for r in rows] == ["g1_000", "g1_001"]
    assert [r["extension_id"] for r in buckets["fixer"]] == ["c"]
    assert [r["extension_id"] for r in buckets["drop"]] == ["d"]


def test_promote_child_of_g1_lands_in_g2():
    g1 = {"id": "g1_000", "generation": 1, "top_module": "token_bucket",
          "clock": "clk", "antecedents": [], "sanity_covers": [],
          "verilog": CHILD_V, "invariants": ["x"]}
    _, rows = promote([PARENT, g1], [rec(parent_id="g1_000")],
                      compute_metrics=False)
    assert rows[0]["id"] == "g2_000"
    assert rows[0]["generation"] == 2


def test_child_top_wrapper_types_use_last_module():
    r = rec(ext_type="compose",
            child_verilog="module a; endmodule\nmodule b; endmodule\n"
                          "module glue_top; endmodule")
    assert child_top(r, PARENT) == "glue_top"
    assert child_top(rec(), PARENT) == "token_bucket"
