"""Tests for the solver baseline harness (no API, no sby): stratified
corpus sampling and the two-rate report (raw vs conditional on
well-formed output).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

import random

from solver_baseline import (parse_invariants, rates, replay_slate,   # noqa: E402
                             stratify)


def row(id, gen, ext=None):
    return {"id": id, "generation": gen, "ext_type": ext, "top_module": id,
            "clock": "clk", "antecedents": [], "sanity_covers": [],
            "verilog": "module m; endmodule", "invariants": ["a <= 1"]}


def test_stratify_takes_per_gen_across_ext_types():
    pool = ([row(f"g0_{i}", 0) for i in range(30)]
            + [row(f"g1_{i}", 1, "structural") for i in range(20)]
            + [row(f"g1_x{i}", 1, "second") for i in range(4)]
            + [row(f"g2_{i}", 2, "structural") for i in range(6)])
    out = stratify(pool, per_gen=8, gens=(0, 1, 2), rng=random.Random(0))
    by_gen = {}
    for r in out:
        by_gen.setdefault(r["generation"], []).append(r)
    assert len(by_gen[0]) == 8
    assert len(by_gen[1]) == 8
    assert len(by_gen[2]) == 6           # fewer than asked -> take all
    # spread across ext types, not clustered: both g1 types present
    assert {r["ext_type"] for r in by_gen[1]} == {"structural", "second"}


def test_stratify_is_deterministic():
    pool = [row(f"g0_{i}", 0) for i in range(30)]
    a = stratify(pool, per_gen=5, gens=(0,), rng=random.Random(7))
    b = stratify(pool, per_gen=5, gens=(0,), rng=random.Random(7))
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_rates_split_raw_and_conditional():
    # 2 NECESSARY, 1 NOT_PROVEN, 1 MALFORMED: raw 2/4, conditional 2/3
    records = [{"verdict": "NECESSARY"}, {"verdict": "NECESSARY"},
               {"verdict": "NOT_PROVEN"}, {"verdict": "MALFORMED"}]
    raw, cond, n, n_wf = rates(records)
    assert (n, n_wf) == (4, 3)
    assert abs(raw - 0.5) < 1e-9
    assert abs(cond - 2 / 3) < 1e-9


def test_rates_empty_and_all_malformed():
    assert rates([])[:2] == (None, None)
    raw, cond, n, n_wf = rates([{"verdict": "MALFORMED"}])
    assert raw == 0.0 and cond is None


def test_parse_invariants_repairs_illegal_quote_escape():
    # qwen writes \' inside JSON strings (illegal escape); valid JSON
    # never contains it, so repairing is safe and model-neutral
    raw = """{"invariants": ["valid == 4\\'b0000 || $onehot(valid)", "count <= 4\\'d9"]}"""
    assert parse_invariants(raw) == [
        "valid == 4'b0000 || $onehot(valid)", "count <= 4'd9"]


def test_parse_invariants_still_rejects_actual_garbage():
    assert parse_invariants('{"invariants": ["count <= 4') is None
    assert parse_invariants("no json here at all") is None


def test_replay_slate_rebuilds_the_original_run_in_order():
    old = [
        {"baseline_id": "B1", "source_run_id": "r1", "source_attempt": 5},
        {"baseline_id": "B1", "source_run_id": "r1", "source_attempt": 2},
        {"baseline_id": "OTHER", "source_run_id": "r1", "source_attempt": 9},
        {"baseline_id": "B1", "source_run_id": "r1", "source_attempt": 5},  # dup
    ]
    triples = [
        {"source_run_id": "r1", "source_attempt": 2, "top_module": "a"},
        {"source_run_id": "r1", "source_attempt": 5, "top_module": "b"},
        {"source_run_id": "r1", "source_attempt": 9, "top_module": "c"},
    ]
    out = replay_slate(old, triples, "B1")
    # original order, no duplicates, OTHER run excluded
    assert [t["top_module"] for t in out] == ["b", "a"]


def test_replay_slate_missing_pair_fails_loudly():
    import pytest
    with pytest.raises(SystemExit):
        replay_slate([{"baseline_id": "B1", "source_run_id": "rX",
                       "source_attempt": 0}], [], "B1")
