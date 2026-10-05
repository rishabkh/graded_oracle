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


def _run(tmp_path, monkeypatch, solve, extra=(), one_shot="INCONCLUSIVE"):
    bench = tmp_path / "bench" / "hard"
    bench.mkdir(parents=True)
    (bench / "toy.sv").write_text("module main(); endmodule\n")
    monkeypatch.setattr(bs, "BENCH_ROOT", tmp_path / "bench")
    monkeypatch.setattr(bs, "OUT_LOG", tmp_path / "out.jsonl")
    monkeypatch.setattr(bs, "solve_qwen", solve)
    monkeypatch.setattr(bs, "judgeable", lambda f: True)
    monkeypatch.setattr(bs, "ebmc_run", lambda *a, **k: {
        "verdict": one_shot, "time": 0.1})
    monkeypatch.setenv("EBMC_PATH", "/bin/true")
    monkeypatch.setenv("QWEN_BASE_URL", "http://localhost:8000/v1")
    fake_openai = types.SimpleNamespace(OpenAI=lambda **kw: types.SimpleNamespace(
        models=types.SimpleNamespace(list=lambda: None)))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    import riscv_score
    monkeypatch.setattr(riscv_score, "served_model", lambda listing: "runs/v3")
    monkeypatch.setattr(sys, "argv", ["benchmark_solve.py", "--solver", "qwen",
                                      "--set", "hard", "--n", "1", *extra])
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


# --- the authors' repair loop on top of the one-shot answer -------------

import repair_loop                                               # noqa: E402


def _first(reply=REPLY):
    def solve(prompt):
        solve.last_raw, solve.last_finish = reply, "stop"
        return ["c <= 4'd9"], None
    solve.last_raw = solve.last_finish = None
    return solve


def _no_ebmc(monkeypatch):
    monkeypatch.setattr(repair_loop, "run_mode",
                        lambda bench, lemmas, mode, workdir, buechi=False:
                        {"verdict": "INCONCLUSIVE"})


def test_without_rounds_rows_are_todays(tmp_path, monkeypatch):
    [row] = _run(tmp_path, monkeypatch, _first())
    for field in ("round", "per_lemma", "kept", "solved", "final"):
        assert field not in row


def test_a_file_solved_one_shot_asks_nothing_more(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(bs, "make_ask", lambda: lambda m, t: asked.append(m))
    rows = _run(tmp_path, monkeypatch, _first(), extra=("--rounds", "5"),
                one_shot="PROVEN")
    assert asked == [] and len(rows) == 1
    row = rows[0]
    assert row["verdict"] == "PROVEN" and row["round"] == 0
    assert row["solved"] is True and row["final"] is True
    assert row["stop_reason"] == "solved" and row["solved_round"] == 0


def test_repair_rounds_follow_the_one_shot_row(tmp_path, monkeypatch):
    _no_ebmc(monkeypatch)
    replies = ['{"invariants": ["c <= 4\'d8"]}', '{"invariants": ["c != 4\'d9"]}']
    seen = []

    def ask(messages, max_tokens):
        seen.append(messages)
        return replies.pop(0), "stop"
    monkeypatch.setattr(bs, "make_ask", lambda: ask)
    rows = _run(tmp_path, monkeypatch, _first(), extra=("--rounds", "3",
                                                        "--show-cex"))
    assert [r["round"] for r in rows] == [0, 1, 2]
    zero = rows[0]
    assert zero["verdict"] == "INCONCLUSIVE"         # today's one-shot verdict
    assert zero["raw"] == REPLY and zero["lemmas"] == ["c <= 4'd9"]
    assert zero["per_lemma"][0]["lemma"] == "c <= 4'd9"
    assert rows[1]["verdict"] == "NOT_SOLVED" and rows[1]["bench"] == "toy.sv"
    assert rows[1]["served_model"] == "runs/v3"
    assert rows[-1]["final"] is True and rows[-1]["stop_reason"] == "max_iterations"
    assert seen[0][0]["content"] == bs.PROMPT.format(
        verilog="module main(); endmodule\n")


def test_rounds_need_the_qwen_solver(tmp_path, monkeypatch):
    import pytest
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["benchmark_solve.py", "--solver",
                                          "opus", "--rounds", "2"])
        bs.main()
