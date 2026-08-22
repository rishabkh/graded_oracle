"""Tests for the extender's guard paths (no API, no sby):
a patch that changes the property must be caught, a parent that fails
yosys must be an error not a silent live=False, and a bad patch stays
PATCH_ERROR. All early returns — the oracle is never reached.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

import extend_one                              # noqa: E402
from extend_one import SELFTEST_PATCH, grade_extension   # noqa: E402
from build_corpus import extract_asserts       # noqa: E402
from tests.test_patch import TOKEN_BUCKET      # noqa: E402


def make_parent():
    return {"id": "g0_test", "top_module": "token_bucket", "clock": "clk",
            "antecedents": [], "sanity_covers": [],
            "verilog": TOKEN_BUCKET,
            "property": extract_asserts(TOKEN_BUCKET),
            "invariants": ["tokens <= 3'd4"]}


PATCH_SWAPS_ASSERT = """\
- if (tokens_hi) assert (tokens != 3'd0);
+         if (tokens_hi) assert (tokens != 3'd1);"""


def test_property_change_is_caught_not_graded():
    rec = grade_extension(make_parent(), PATCH_SWAPS_ASSERT, {})
    assert rec["verdict"] == "PROPERTY_CHANGED"
    assert "3'd0" in rec["error"] and "3'd1" in rec["error"]
    # caught before any yosys or oracle work
    assert "parent_state_bits" not in rec
    assert "result" not in rec


def test_unreadable_parent_is_yosys_error(monkeypatch):
    monkeypatch.setattr(extend_one, "state_bits", lambda *a: None)
    rec = grade_extension(make_parent(), SELFTEST_PATCH, {})
    assert rec["verdict"] == "YOSYS_ERROR"
    assert "parent" in rec["error"]      # blamed on machinery, not the model


def test_malformed_patch_is_patch_error():
    rec = grade_extension(make_parent(), "@@ nowhere @@\n+ x", {})
    assert rec["verdict"] == "PATCH_ERROR"
    assert "child_verilog" not in rec
