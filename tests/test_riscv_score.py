"""Scoring a model on the riscv-formal problem set.

The survey logged 237 verdicts but not which core produced each row, and
check names repeat across cores, so the problem set has to be recovered
by matching each run's check names against the cores' directories."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from riscv_score import build_prompt, core_of_run, problem_set

ROWS = [
    {"run_id": "r1", "check": "insn_add_ch0", "verdict": "NEEDS_INVARIANT"},
    {"run_id": "r1", "check": "pc_fwd_ch0", "verdict": "PROVED"},
    {"run_id": "r1", "check": "causal_ch0", "verdict": "NEEDS_INVARIANT"},
    {"run_id": "r2", "check": "insn_add_ch0", "verdict": "NEEDS_INVARIANT"},
    {"run_id": "r2", "check": "bus_dmem_ch0", "verdict": "NEEDS_INVARIANT"},
]
DIRS = {"serv": {"insn_add_ch0", "pc_fwd_ch0", "causal_ch0"},
        "nerv": {"insn_add_ch0", "bus_dmem_ch0", "pc_fwd_ch0"}}


def test_a_run_is_matched_to_the_core_that_has_all_its_checks():
    assert core_of_run(ROWS, "r1", DIRS) == "serv"
    assert core_of_run(ROWS, "r2", DIRS) == "nerv"


def test_a_run_matching_no_core_is_unknown_rather_than_guessed():
    rows = [{"run_id": "r9", "check": "not_a_check", "verdict": "PROVED"}]
    assert core_of_run(rows, "r9", DIRS) is None


def test_the_problem_set_is_the_unproved_checks_of_one_core():
    assert problem_set(ROWS, "serv", DIRS) == ["causal_ch0", "insn_add_ch0"]
    assert problem_set(ROWS, "nerv", DIRS) == ["bus_dmem_ch0",
                                               "insn_add_ch0"]


def test_the_prompt_asks_for_facts_about_the_core_not_the_wrapper():
    out = build_prompt("module nerv (input clock); endmodule",
                       "assert (pc[1:0] == 2'b00);", core_module="nerv")
    assert "module nerv" in out and "assert (pc[1:0] == 2'b00);" in out
    assert "nerv" in out
    assert '{"invariants"' in out


def test_the_prompt_forbids_the_things_the_checker_cannot_take():
    out = build_prompt("module nerv (input clock); endmodule", "assert (1);",
                       core_module="nerv")
    for banned in ("$past", "->", "SVA"):
        assert banned in out          # named as forbidden, not used


def test_malformed_expressions_are_caught_before_a_proof_is_run():
    """Measured 22 Sep: one serv check drew 9 and then 15 invariants, all
    of them using `==>`, which is not Verilog. Running the prover on them
    wastes a proof and records ERROR, which reads like a wrong answer
    rather than an unusable one."""
    from riscv_score import malformed
    assert malformed(["a ==> b"])
    assert malformed(["(a && b"])               # unbalanced
    assert malformed(["a -> b"])
    assert malformed(["$past(a) == b"])
    assert not malformed(["!a || b", "count <= 4'd9"])


def test_a_wholly_malformed_answer_is_no_answer():
    from riscv_score import usable
    assert usable(["!a || b", "x ==> y"]) == ["!a || b"]
    assert usable(["x ==> y"]) == []


def test_summarise_groups_a_run_and_counts_what_matters():
    from riscv_score import summarise
    rows = [
        {"run_id": "r1", "core": "serv", "condition": "native",
         "model": "llm", "check": "a", "verdict": "PROVED"},
        {"run_id": "r1", "core": "serv", "condition": "native",
         "model": "llm", "check": "b", "verdict": "BASECASE_FAIL"},
        {"run_id": "r1", "core": "serv", "condition": "native",
         "model": "llm", "check": "c", "verdict": "NO_ANSWER"},
        {"run_id": "r0", "core": "serv", "condition": "native",
         "model": "llm", "check": "a", "verdict": "PROVED"},
    ]
    s = summarise(rows, "r1")
    assert s["proved"] == 1 and s["total"] == 3
    assert s["counts"]["BASECASE_FAIL"] == 1
    assert s["core"] == "serv" and s["model"] == "llm"
    assert s["condition"] == "native"


def test_a_dead_endpoint_stops_the_run_before_it_writes_anything():
    """22 Sep: a run started while the server was still loading and wrote
    NO_ANSWER rows that looked like model failures. A score file must
    never contain rows produced by a connection error."""
    import riscv_score

    def refuses():
        raise ConnectionError("connection refused")

    ok, why = riscv_score.endpoint_ready(refuses)
    assert ok is False and "refused" in why

    ok, why = riscv_score.endpoint_ready(lambda: ["llm"])
    assert ok is True


def test_an_empty_api_key_is_caught_before_the_run(monkeypatch):
    """OpenRouter lists models without a key, so a models.list() probe
    passes and every question then comes back 401. Ten rows of NO_ANSWER
    looked exactly like a model scoring zero."""
    import riscv_score
    monkeypatch.setenv("QWEN_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("QWEN_API_KEY", "")
    ok, why = riscv_score.credentials_ready()
    assert ok is False and "QWEN_API_KEY" in why

    monkeypatch.setenv("QWEN_API_KEY", "sk-or-v1-something")
    assert riscv_score.credentials_ready()[0] is True


def test_a_local_endpoint_needs_no_key(monkeypatch):
    import riscv_score
    monkeypatch.setenv("QWEN_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("QWEN_API_KEY", "none")
    assert riscv_score.credentials_ready()[0] is True


def test_solve_records_why_an_answer_was_empty():
    """Twice now an empty answer was logged as NO_ANSWER with no way to
    tell whether the model refused, ran out of budget, or spent it all on
    thinking. The reply's own fields say which."""
    import types
    import riscv_score

    msg = types.SimpleNamespace(content="", refusal=None,
                                reasoning="thinking and thinking")
    usage = types.SimpleNamespace(
        completion_tokens=16000, prompt_tokens=10660,
        completion_tokens_details=types.SimpleNamespace(
            reasoning_tokens=16000))
    reply = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="length")],
        usage=usage, provider="Claude Platform on AWS")

    meta = riscv_score.reply_meta(reply)
    assert meta["finish_reason"] == "length"
    assert meta["reasoning_tokens"] == 16000
    assert meta["completion_tokens"] == 16000
    assert meta["provider"] == "Claude Platform on AWS"
    assert meta["had_reasoning_text"] is True


def test_the_served_model_is_recorded_not_just_its_alias():
    """Both the untrained model and the fine-tuned one are served under
    the alias 'llm', so the log could not say which produced a run. The
    server reports the real path; record it."""
    import types
    import riscv_score
    listing = types.SimpleNamespace(data=[
        types.SimpleNamespace(id="llm",
                              root="/n/home01/x/graded_oracle/runs/v1/merged")])
    assert riscv_score.served_model(listing).endswith("runs/v1/merged")

    listing = types.SimpleNamespace(data=[
        types.SimpleNamespace(id="llm", root="Qwen/Qwen2.5-Coder-32B-Instruct")])
    assert riscv_score.served_model(listing) == "Qwen/Qwen2.5-Coder-32B-Instruct"

    assert riscv_score.served_model(types.SimpleNamespace(data=[])) is None
