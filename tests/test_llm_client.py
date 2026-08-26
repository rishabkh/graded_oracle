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
