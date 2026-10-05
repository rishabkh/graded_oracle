"""The scorer's main loop, with the model and EBMC both faked: what each
saved row records. No network, no EBMC."""
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import benchmark_solve as bs                                   # noqa: E402

REPLY = ('{"reasoning": "Fake state: c = 4\'d12. c jumps past 9.", '
         '"invariants": ["c <= 4\'d9"]}')


def _run(tmp_path, monkeypatch, solve):
    bench = tmp_path / "bench" / "hard"
    bench.mkdir(parents=True)
    (bench / "toy.sv").write_text("module main(); endmodule\n")
    monkeypatch.setattr(bs, "BENCH_ROOT", tmp_path / "bench")
    monkeypatch.setattr(bs, "OUT_LOG", tmp_path / "out.jsonl")
    monkeypatch.setattr(bs, "solve_qwen", solve)
    monkeypatch.setattr(bs, "judgeable", lambda f: True)
    monkeypatch.setattr(bs, "ebmc_run", lambda *a, **k: {
        "verdict": "INCONCLUSIVE", "time": 0.1})
    monkeypatch.setenv("EBMC_PATH", "/bin/true")
    monkeypatch.setenv("QWEN_BASE_URL", "http://localhost:8000/v1")
    fake_openai = types.SimpleNamespace(OpenAI=lambda **kw: types.SimpleNamespace(
        models=types.SimpleNamespace(list=lambda: None)))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    import riscv_score
    monkeypatch.setattr(riscv_score, "served_model", lambda listing: "runs/v3")
    monkeypatch.setattr(sys, "argv", ["benchmark_solve.py", "--solver", "qwen",
                                      "--set", "hard", "--n", "1"])
    bs.main()
    return [json.loads(l) for l in (tmp_path / "out.jsonl").read_text()
            .splitlines()]


def test_every_row_keeps_the_full_reply_and_why_it_ended(tmp_path, monkeypatch):
    """Found 5 Oct 2026: v3 (trained with reasoning) scored far below v2
    and the log held only the extracted invariants, so nobody could read
    what the model actually wrote. The whole reply is now kept."""
    def solve(prompt):
        solve.last_raw, solve.last_finish = REPLY, "stop"
        return ["c <= 4'd9"], None
    solve.last_raw = solve.last_finish = None
    [row] = _run(tmp_path, monkeypatch, solve)
    assert row["raw"] == REPLY
    assert row["finish"] == "stop"
    assert row["lemmas"] == ["c <= 4'd9"] and row["verdict"] == "INCONCLUSIVE"


def test_an_unreadable_reply_is_kept_too(tmp_path, monkeypatch):
    def solve(prompt):
        solve.last_raw, solve.last_finish = "no json here", "length"
        return None, "truncated at 16000 tokens"
    solve.last_raw = solve.last_finish = None
    [row] = _run(tmp_path, monkeypatch, solve)
    assert row["verdict"] == "NO_ANSWER"
    assert row["raw"] == "no json here" and row["finish"] == "length"


def test_a_call_that_dies_does_not_inherit_the_last_reply(tmp_path, monkeypatch):
    def solve(prompt):
        raise ConnectionError("server went away")
    solve.last_raw, solve.last_finish = "an older file's reply", "stop"
    [row] = _run(tmp_path, monkeypatch, solve)
    assert row["verdict"] == "NO_ANSWER"
    assert row["raw"] is None and row["finish"] is None
