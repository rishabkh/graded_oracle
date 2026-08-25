"""Tests for the step-4 extension machinery (no API, no sby):
disposition auditing (silent clause deletion is rejected before grading),
second-property assert-set discipline, selective assert removal for the
P2-proves-unaided check, and coupling-clause detection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from extend import (                     # noqa: E402
    check_dispositions, split_second_property_asserts, remove_asserts,
    has_coupling_clause)

PARENT_INVS = ["tokens_hi == (tokens >= 3'd2)", "tokens <= 3'd4"]


def d(clause, status, replaced_by="", reason=""):
    return {"clause": clause, "status": status,
            "replaced_by": replaced_by, "reason": reason}


# --- dispositions: every parent clause accounted for, or rejected ---

def test_dispositions_all_kept_ok():
    new = PARENT_INVS + ["pending <= 2'd2"]
    err = check_dispositions(PARENT_INVS, new,
                             [d(PARENT_INVS[0], "kept"),
                              d(PARENT_INVS[1], "kept")])
    assert err is None


def test_dispositions_missing_entry_is_silent_deletion():
    err = check_dispositions(PARENT_INVS, PARENT_INVS[:1],
                             [d(PARENT_INVS[0], "kept")])
    assert err is not None and "tokens <= 3'd4" in err


def test_dispositions_kept_but_absent_from_list():
    err = check_dispositions(PARENT_INVS, PARENT_INVS[:1],
                             [d(PARENT_INVS[0], "kept"),
                              d(PARENT_INVS[1], "kept")])
    assert err is not None and "kept" in err


def test_dispositions_superseded_ok():
    new = ["tokens_hi == (tokens >= 3'd2 || boost)", "tokens <= 3'd4"]
    err = check_dispositions(
        PARENT_INVS, new,
        [d(PARENT_INVS[0], "superseded",
           replaced_by="tokens_hi == (tokens >= 3'd2 || boost)",
           reason="tokens_hi has a second source now"),
         d(PARENT_INVS[1], "kept")])
    assert err is None


def test_dispositions_superseded_but_replacement_missing():
    err = check_dispositions(
        PARENT_INVS, ["tokens <= 3'd4"],
        [d(PARENT_INVS[0], "superseded", replaced_by="something_else == 1",
           reason="r"),
         d(PARENT_INVS[1], "kept")])
    assert err is not None


def test_dispositions_superseded_needs_reason():
    new = ["stronger", "tokens <= 3'd4"]
    err = check_dispositions(
        PARENT_INVS, new,
        [d(PARENT_INVS[0], "superseded", replaced_by="stronger", reason=""),
         d(PARENT_INVS[1], "kept")])
    assert err is not None and "reason" in err


def test_dispositions_unknown_clause_rejected():
    err = check_dispositions(
        PARENT_INVS, PARENT_INVS,
        [d(PARENT_INVS[0], "kept"), d(PARENT_INVS[1], "kept"),
         d("invented == 1", "kept")])
    assert err is not None and "invented" in err


# --- second property: parent asserts verbatim + exactly one new ---

def test_second_property_split_ok():
    parent = ["tokens != 3'd0"]
    child = ["tokens != 3'd0", "free <= 3'd4"]
    new, err = split_second_property_asserts(parent, child)
    assert err is None
    assert new == "free <= 3'd4"


def test_second_property_missing_new_assert():
    new, err = split_second_property_asserts(["a"], ["a"])
    assert err is not None


def test_second_property_parent_modified():
    new, err = split_second_property_asserts(["a == 1"], ["a == 2", "b"])
    assert err is not None


def test_second_property_two_new_asserts():
    new, err = split_second_property_asserts(["a"], ["a", "b", "c"])
    assert err is not None


# --- remove_asserts: blank a specific assertion, keep the file legal ---

def test_remove_asserts_removes_only_the_named_one():
    v = ("module m;\n"
         "always @(posedge clk) if (g) assert (a == b);\n"
         "always @(posedge clk) assert (c);\n"
         "endmodule\n")
    out = remove_asserts(v, ["a == b"])
    assert "a == b" not in out
    assert "assert (c);" in out
    assert "endmodule" in out


def test_remove_asserts_guarded_position_stays_legal():
    v = "module m;\nalways @(posedge clk) if (g) assert (x);\nendmodule\n"
    out = remove_asserts(v, ["x"])
    # the guarded statement still has a body (empty statement)
    assert "if (g) ;" in " ".join(out.split())


# --- coupling clause: mentions both new and existing identifiers ---

PARENT_IDS = {"tokens", "tokens_hi", "drip", "spend", "clk", "rst"}
NEW_IDS = {"pending"}


def test_coupling_clause_detected():
    assert has_coupling_clause(
        ["pending <= 2'd2", "tokens + pending <= 3'd4"], PARENT_IDS, NEW_IDS)


def test_no_coupling_when_new_state_isolated():
    assert not has_coupling_clause(
        ["pending <= 2'd2"], PARENT_IDS, NEW_IDS)
