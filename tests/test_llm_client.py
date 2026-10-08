"""Tests for the provider switch (no network): provider selection,
lenient JSON extraction for the OpenRouter path, and usage mapping.
"""
import json
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
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    text, usage, stop = llm_client.call_claude(model="m", max_tokens=32000,
                                               user="u")
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


# --- the batch gate (8 Oct 2026): many threads, one half-price batch -----
# The extender runs many workers, each making ordinary call_claude calls.
# With a gate open, those calls are gathered and sent as one Message
# Batch; each thread blocks until its own answer is back. Nothing else in
# the caller changes.
import threading                                                 # noqa: E402
import time as _time                                             # noqa: E402


def _fake_batches(monkeypatch, answers=None, fail=()):
    sent = []

    def submit(items):
        sent.append(items)
        return f"batch_{len(sent)}"

    def results(bid):
        for cid, kw in sent[int(bid.split("_")[1]) - 1]:
            user = kw["messages"][0]["content"]
            if user in fail:
                yield cid, ("error:overloaded_error", None, {"input": 0, "output": 0}, None)
            else:
                text = (answers or {}).get(user, f'{{"echo": "{user}"}}')
                yield cid, ("ok", text, {"input": 1, "output": 2}, text)
    monkeypatch.setattr(llm_client, "batch_submit", submit)
    monkeypatch.setattr(llm_client, "batch_status", lambda bid: ("ended", {}))
    monkeypatch.setattr(llm_client, "batch_results", results)
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    return sent


def _call_in_threads(n):
    out = [None] * n

    def one(i):
        out[i] = llm_client.call_claude(model="claude-opus-5-5", max_tokens=9,
                                        user=f"q{i}", system="S")
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    return out


def test_concurrent_calls_through_the_gate_go_as_one_batch(monkeypatch, tmp_path):
    sent = _fake_batches(monkeypatch)
    with llm_client.batch_gate(flush_after_s=0.3, poll_s=0,
                               manifest_dir=tmp_path):
        out = _call_in_threads(5)
    assert len(sent) == 1 and len(sent[0]) == 5
    # every thread got its own answer, read exactly like a direct reply
    assert [o[0] for o in out] == [f'{{"echo": "q{i}"}}' for i in range(5)]
    assert all(o[2] == "ok" and o[1] == {"input": 1, "output": 2} for o in out)
    # the request is the direct request, byte for byte
    kw = sent[0][0][1]
    assert kw == llm_client.request_kwargs(model="claude-opus-5-5", max_tokens=9,
                                           user=kw["messages"][0]["content"],
                                           system="S")


def test_a_failed_batch_entry_raises_in_its_own_thread_only(monkeypatch, tmp_path):
    _fake_batches(monkeypatch, fail={"q1"})
    errors = []

    def one(i):
        try:
            llm_client.call_claude(model="claude-opus-5-5", max_tokens=9,
                                   user=f"q{i}")
        except RuntimeError as e:
            errors.append((i, str(e)))
    with llm_client.batch_gate(flush_after_s=0.3, poll_s=0,
                               manifest_dir=tmp_path):
        ts = [threading.Thread(target=one, args=(i,)) for i in range(3)]
        [t.start() for t in ts]
        [t.join(timeout=10) for t in ts]
    assert errors == [(1, "batch error:overloaded_error")]


def test_every_batch_is_written_down_before_it_is_sent(monkeypatch, tmp_path):
    sent = _fake_batches(monkeypatch)
    with llm_client.batch_gate(flush_after_s=0.3, poll_s=0,
                               manifest_dir=tmp_path):
        _call_in_threads(2)
    [m] = list(tmp_path.glob("batch_*.json"))
    rec = json.loads(m.read_text())
    assert rec["batch_id"] == "batch_1" and len(rec["requests"]) == 2
    assert rec["usage"] == {"input": 2, "output": 4}


def test_with_the_gate_closed_calls_go_direct_again(monkeypatch, tmp_path):
    sent = _fake_batches(monkeypatch)
    with llm_client.batch_gate(flush_after_s=0.3, poll_s=0, manifest_dir=tmp_path):
        pass
    assert llm_client._gate is None


# --- the call log (8 Oct 2026): every API call, saved at the source --------
# Whatever script makes a call, the client itself appends one line: the
# exact request, the raw answer (even a refusal or a cut-off one), tokens,
# how it ended, timing, and for a batch its batch id and custom id.

def _stream_returning(monkeypatch, *replies):
    replies = list(replies)

    class Stream:
        def __enter__(self):
            return types.SimpleNamespace(get_final_message=lambda: replies.pop(0))

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(llm_client, "_an_client", types.SimpleNamespace(
        messages=types.SimpleNamespace(stream=lambda **kw: Stream())))
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)


def test_every_direct_call_is_logged_with_request_and_raw_answer(monkeypatch,
                                                                tmp_path):
    log = tmp_path / "calls.jsonl"
    _stream_returning(monkeypatch, _msg('{"a": 1}'), _msg("half", stop="refusal"))
    with llm_client.call_log(log):
        llm_client.call_claude(model="claude-opus-5-5", max_tokens=9, user="U1",
                               system="S", schema={"type": "object"}, effort="high")
        llm_client.call_claude(model="claude-opus-5-5", max_tokens=9, user="U2")
    a, b = [json.loads(l) for l in log.read_text().splitlines()]
    assert a["request"] == llm_client.request_kwargs(
        model="claude-opus-5-5", max_tokens=9, user="U1", system="S",
        schema={"type": "object"}, effort="high")
    assert a["raw"] == '{"a": 1}' and a["stop"] == "ok"
    assert a["usage"] == {"input": 10, "output": 20} and a["seconds"] >= 0
    assert a["started"] and a["batch_id"] is None
    assert b["stop"] == "refusal" and b["request"]["messages"][0]["content"] == "U2"
    assert llm_client._call_log_path is None          # closed again


def test_the_last_call_is_kept_per_thread(monkeypatch):
    _stream_returning(monkeypatch, _msg('{"t": 1}'), _msg('{"t": 2}'))
    got = {}

    def one(name):
        llm_client.call_claude(model="m", max_tokens=1, user=name)
        got[name] = llm_client.last_call()["request"]["messages"][0]["content"]
    t1 = threading.Thread(target=one, args=("first",))
    t1.start(); t1.join()
    t2 = threading.Thread(target=one, args=("second",))
    t2.start(); t2.join()
    assert got == {"first": "first", "second": "second"}


def test_a_batch_call_is_logged_with_its_batch_and_custom_id(monkeypatch, tmp_path):
    _fake_batches(monkeypatch)
    log = tmp_path / "calls.jsonl"
    with llm_client.call_log(log), llm_client.batch_gate(
            flush_after_s=0.2, poll_s=0, manifest_dir=tmp_path / "batches"):
        _call_in_threads(2)
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(lines) == 2
    assert all(l["batch_id"] == "batch_1" and l["custom_id"].startswith("r")
               and l["batch"] is True for l in lines)
    assert {l["raw"] for l in lines} == {'{"echo": "q0"}', '{"echo": "q1"}'}


def test_a_failed_batch_entry_is_logged_before_it_raises(monkeypatch, tmp_path):
    _fake_batches(monkeypatch, fail={"q0"})
    log = tmp_path / "calls.jsonl"
    with llm_client.call_log(log), llm_client.batch_gate(
            flush_after_s=0.2, poll_s=0, manifest_dir=tmp_path / "batches"):
        try:
            llm_client.call_claude(model="m", max_tokens=1, user="q0")
        except RuntimeError:
            pass
    [line] = [json.loads(l) for l in log.read_text().splitlines()]
    assert line["stop"] == "error:overloaded_error" and line["raw"] is None
