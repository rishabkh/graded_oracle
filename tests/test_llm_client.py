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
