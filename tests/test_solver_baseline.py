"""Tests for the solver baseline harness (no API, no sby): stratified
corpus sampling and the two-rate report (raw vs conditional on
well-formed output).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

import random

from solver_baseline import (out_of_scope, parse_invariants, rates,   # noqa: E402
                             replay_slate, stratify)


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


TWO_MOD = """module inner (input wire clk, output reg [2:0] on_events);
reg heat_on;
endmodule
module outer (input wire clk, output wire [2:0] on0);
inner u0 (.clk(clk), .on_events(on0));
endmodule"""


def test_out_of_scope_catches_submodule_names():
    # the right fact in the wrong namespace: graded at top level these
    # names become free wires and the clause is trivially "falsified"
    bad = out_of_scope(["heat_on || (on_events == 3'd0)"], TWO_MOD, "outer")
    assert "heat_on" in bad and "on_events" in bad


def test_out_of_scope_accepts_top_level_names():
    assert out_of_scope(["on0 == 3'd1"], TWO_MOD, "outer") == set()


def test_solving_checks_get_the_deeper_look_but_keep_their_time_limit():
    """8 Oct 2026: a big design's condition may need more than 20 steps;
    without the deeper look it went to pdr and could time out. The 120 s
    limit stays, so earlier scores remain comparable."""
    import solver_baseline as sb
    assert sb.GRADE_KWARGS == {"timeout_s": 120, "cover_depth": "auto"}


# --- our own storage test (6 Nov 2026) ---------------------------------------
# The storage designs set aside from training, asked of a served model and
# judged by the same checker, with the limit the designs were accepted at.

def test_the_set_aside_storage_designs_can_be_asked(tmp_path, capsys):
    import json
    import solver_baseline as sb
    corpus = tmp_path / "storage.jsonl"
    rows = [dict(row(f"storage_queue_{i:03d}", 0), held_out=(i == 10),
                 verdict="NECESSARY") for i in (0, 10, 21)]
    rows[2]["held_out"] = True
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sb.main(["--solver", "qwen", "--corpus", str(corpus), "--held-out", "--dry"])
    out = capsys.readouterr().out
    assert "2 set-aside" in out
    assert "storage_queue_010" in out and "storage_queue_021" in out
    assert "storage_queue_000" not in out


def test_the_grading_limit_can_be_raised_to_the_one_designs_were_accepted_at():
    import solver_baseline as sb
    assert sb.grade_kwargs(None) == sb.GRADE_KWARGS
    assert sb.grade_kwargs(300)["timeout_s"] == 300
