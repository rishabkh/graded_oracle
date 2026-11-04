"""What a run spends (3 Nov 2026). Rishab chose a cost estimate before
every paid run over a limit that could stop a run part-way, so llm_client
only counts: every paid call goes through call_claude, and the run prints
its total when it ends, to set against the estimate. The API is faked."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm_client                                                # noqa: E402

# claude-opus-5-5 list prices: $4 in, $20 out per million tokens
IN, OUT = 4 / 1e6, 20 / 1e6


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


def anthropic_route(monkeypatch, i=10, o=1000, fail=False):
    """The Anthropic route with a fake server. Returns the requests that
    reached it."""
    sent = []

    class Stream:
        def __init__(self, kw):
            self.kw = kw

        def __enter__(self):
            sent.append(self.kw)
            if fail:
                raise RuntimeError("overloaded")
            return types.SimpleNamespace(
                get_final_message=lambda: types.SimpleNamespace(
                    content=[_Text('{"a": 1}')], stop_reason="end_turn",
                    usage=types.SimpleNamespace(input_tokens=i,
                                                output_tokens=o)))

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(llm_client, "_an_client", types.SimpleNamespace(
        messages=types.SimpleNamespace(stream=lambda **kw: Stream(kw))))
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    return sent


def ask(model="claude-opus-5-5"):
    return llm_client.call_claude(model=model, max_tokens=1000, user="u")


def test_a_paid_call_needs_no_spending_setting(monkeypatch):
    sent = anthropic_route(monkeypatch)
    monkeypatch.delenv("LLM_MAX_DOLLARS", raising=False)
    assert ask()[2] == "ok" and len(sent) == 1


def test_each_call_is_counted_at_its_models_list_price(monkeypatch):
    anthropic_route(monkeypatch)
    ask()
    ask()
    s = llm_client.spending()
    assert s["calls"] == 2
    assert s["spent"] == pytest.approx(2 * (10 * IN + 1000 * OUT))


def test_half_price_batches_are_counted_at_half(monkeypatch):
    anthropic_route(monkeypatch)
    monkeypatch.setattr(llm_client, "_gate", types.SimpleNamespace(
        call=lambda kw: ("ok", '{"a": 1}', {"input": 10, "output": 1000},
                         '{"a": 1}', {})))
    ask()
    assert llm_client.spending()["spent"] == \
        pytest.approx((10 * IN + 1000 * OUT) / 2)


def test_openai_calls_are_priced_as_the_model_that_answers(monkeypatch):
    body = {"choices": [{"message": {"content": '{"a": 1}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1000}}
    reply = types.SimpleNamespace(model_dump=lambda: body)
    monkeypatch.setattr(llm_client, "_oa_client", types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **kw: reply))))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    ask()
    # GPT-6.1 Sol: $2 in, $10 out per million tokens
    assert llm_client.spending()["spent"] == \
        pytest.approx((10 * 2 + 1000 * 10) / 1e6)


def test_a_failed_call_is_not_counted(monkeypatch):
    anthropic_route(monkeypatch, fail=True)
    with pytest.raises(RuntimeError):
        ask()
    assert llm_client.spending() == {"spent": 0.0, "calls": 0, "unpriced": 0}


def test_a_model_with_no_known_price_still_answers_and_is_flagged(monkeypatch):
    anthropic_route(monkeypatch)
    assert ask(model="claude-unpriced-9")[2] == "ok"
    s = llm_client.spending()
    assert s["calls"] == 1 and s["unpriced"] == 1 and s["spent"] == 0
    assert "not counted" in llm_client.spending_report()


def test_the_run_says_what_it_spent(monkeypatch):
    anthropic_route(monkeypatch)
    ask()
    ask()
    report = llm_client.spending_report()
    assert "$0.04" in report and "2 calls" in report
