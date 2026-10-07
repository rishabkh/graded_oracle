"""Tests for the provider switch (no network): provider selection,
lenient JSON extraction for the OpenRouter path, and usage mapping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_client                      # noqa: E402
from llm_client import extract_json, provider, model_label   # noqa: E402


def test_provider_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    assert provider() == "anthropic"


def test_provider_openrouter(monkeypatch):
    monkeypatch.setenv("CLAUDE_PROVIDER", "openrouter")
    assert provider() == "openrouter"


def test_model_label_names_the_provider(monkeypatch):
    monkeypatch.setenv("CLAUDE_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_CLAUDE_MODEL", raising=False)
    assert model_label("claude-opus-5").startswith("openrouter:")
    monkeypatch.setenv("CLAUDE_PROVIDER", "anthropic")
    assert model_label("claude-opus-5") == "claude-opus-5"


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_with_prose():
    text = 'Here you go:\n```json\n{"patch": "x", "invariants": ["a"]}\n```\ndone'
    assert extract_json(text) == {"patch": "x", "invariants": ["a"]}


def test_extract_json_nested_braces():
    text = 'note {"outer": {"inner": [1, 2]}, "s": "with } brace in string"}'
    obj = extract_json(text)
    assert obj["outer"] == {"inner": [1, 2]}
    assert "}" in obj["s"]


def test_extract_json_none_when_absent():
    assert extract_json("no json here") is None
    assert extract_json("{broken") is None


def test_anthropic_call_streams_so_a_long_generation_is_allowed(monkeypatch):
    """The SDK refuses a plain create() whose max_tokens could run past ten
    minutes, which is every initiator call at a 32k budget."""
    import types
    import llm_client

    seen = {}

    class FakeMessage:
        stop_reason = "end_turn"
        usage = types.SimpleNamespace(input_tokens=1, output_tokens=2)
        content = [types.SimpleNamespace(type="text", text='{"ok": 1}')]

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return FakeMessage()

    class FakeMessages:
        def create(self, **kw):
            raise AssertionError("must not call create(): use stream()")

        def stream(self, **kw):
            seen.update(kw)
            return FakeStream()

    monkeypatch.setattr(llm_client, "_an_client",
                        types.SimpleNamespace(messages=FakeMessages()))
    text, usage, stop = llm_client._call_anthropic(
        model="m", max_tokens=32000, user="u", system=None, schema=None,
        effort=None)
    assert stop == "ok" and text == '{"ok": 1}'
    assert usage == {"input": 1, "output": 2}
    assert seen["max_tokens"] == 32000


def test_openrouter_reply_without_a_usage_block_does_not_crash(monkeypatch):
    """Four calls in the last batch died on this: some provider routes
    return a response with usage=None, and a paid call became an ERROR
    that the retry path could not rescue."""
    import types
    import llm_client

    choice = types.SimpleNamespace(
        finish_reason="stop",
        message=types.SimpleNamespace(content=None))
    reply = types.SimpleNamespace(choices=[choice], usage=None)

    class FakeCompletions:
        def create(self, **kw):
            return reply

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(llm_client, "_or_client", types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions())))
    text, usage, stop = llm_client._call_openrouter(
        model="m", max_tokens=100, user="u", system=None, schema=None,
        effort=None)
    assert text is None
    assert usage == {"input": 0, "output": 0}
    assert stop == "ok"        # retryable, not a dead ERROR


# --- prices (Oct 2026: everything moved to Opus 5.5) ---------------------

def test_each_model_is_priced_at_its_own_list_price():
    import pytest
    assert llm_client.dollars("claude-opus-5", 1_000_000, 1_000_000) == 30.0
    assert llm_client.dollars("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    # a logged label is priced like the bare id
    assert llm_client.dollars("openrouter:anthropic/claude-opus-5",
                              1_000_000, 0) == 5.0
    with pytest.raises(ValueError):
        llm_client.dollars("some-new-model", 1, 1)


def test_every_script_that_calls_opus_uses_opus_5_5():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))
    import catalog_kinds, distractor, fix_weak, regen_reasoning, run
    import solver_baseline
    for mod, name in ((run, "MODEL"), (distractor, "MODEL"),
                      (regen_reasoning, "MODEL"), (fix_weak, "MODEL"),
                      (catalog_kinds, "MODEL"), (solver_baseline, "OPUS_MODEL")):
        assert getattr(mod, name) == "claude-opus-5-5", mod.__name__


# --- batch mode (8 Oct 2026): half price, same request, same reading ----
import types                                                    # noqa: E402


class _Text:
    def __init__(self, text):
        self.type, self.text = "text", text


def _msg(text, stop="end_turn", i=10, o=20):
    return types.SimpleNamespace(
        content=[_Text(text)], stop_reason=stop,
        usage=types.SimpleNamespace(input_tokens=i, output_tokens=o))


def test_a_reply_is_read_the_same_way_in_both_modes():
    assert llm_client.read_message(_msg('{"a": 1}')) == (
        '{"a": 1}', {"input": 10, "output": 20}, "ok", '{"a": 1}')
    refused = llm_client.read_message(_msg("", stop="refusal"))
    assert refused[0] is None and refused[2] == "refusal"
    cut = llm_client.read_message(_msg('{"a": ', stop="max_tokens"))
    assert cut[0] is None and cut[2] == "length" and cut[3] == '{"a": '


def test_a_batch_request_is_exactly_the_direct_request(monkeypatch):
    kw = llm_client.request_kwargs(model="claude-opus-5-5", max_tokens=32000,
                                   user="U", system="S",
                                   schema={"type": "object"}, effort="high")
    assert kw == {"model": "claude-opus-5-5", "max_tokens": 32000,
                  "messages": [{"role": "user", "content": "U"}],
                  "system": "S",
                  "output_config": {"effort": "high", "format": {
                      "type": "json_schema", "schema": {"type": "object"}}}}
    sent = {}
    batches = types.SimpleNamespace(create=lambda requests: (
        sent.update(requests=requests), types.SimpleNamespace(id="b1"))[1])
    monkeypatch.setattr(llm_client, "_an_client", types.SimpleNamespace(
        messages=types.SimpleNamespace(batches=batches)))
    assert llm_client.batch_submit([("a0000", kw)]) == "b1"
    assert sent["requests"] == [{"custom_id": "a0000", "params": kw}]


def test_batch_results_cover_every_outcome(monkeypatch):
    def res(cid, kind, **kw):
        return types.SimpleNamespace(custom_id=cid, result=types.SimpleNamespace(
            type=kind, **kw))
    out = [res("a", "succeeded", message=_msg('{"x": 1}')),
           res("b", "succeeded", message=_msg("", stop="refusal")),
           res("c", "errored", error=types.SimpleNamespace(
               error=types.SimpleNamespace(type="overloaded_error"))),
           res("d", "expired")]
    batches = types.SimpleNamespace(results=lambda bid: iter(out))
    monkeypatch.setattr(llm_client, "_an_client", types.SimpleNamespace(
        messages=types.SimpleNamespace(batches=batches)))
    got = dict(llm_client.batch_results("b1"))
    assert got["a"][0] == "ok" and got["a"][1] == '{"x": 1}'
    assert got["b"][0] == "refusal" and got["b"][1] is None
    assert got["c"][0] == "error:overloaded_error" and got["c"][1] is None
    assert got["d"][0] == "error:expired"


def test_batch_tokens_cost_half():
    assert llm_client.dollars("claude-opus-5-5", 1_000_000, 1_000_000,
                              batch=True) == 12.0


def test_a_refusal_does_not_leave_the_previous_reply_behind(monkeypatch):
    replies = iter([_msg('{"first": 1}'), _msg("", stop="refusal")])

    class Stream:
        def __enter__(self):
            return types.SimpleNamespace(get_final_message=lambda: next(replies))

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(llm_client, "_an_client", types.SimpleNamespace(
        messages=types.SimpleNamespace(stream=lambda **kw: Stream())))
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    llm_client.call_claude(model="m", max_tokens=1, user="u")
    assert llm_client.LAST_RAW == '{"first": 1}'
    assert llm_client.call_claude(model="m", max_tokens=1, user="u")[2] == "refusal"
    assert llm_client.LAST_RAW is None
