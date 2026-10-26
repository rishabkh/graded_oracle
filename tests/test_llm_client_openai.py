"""The OpenAI route (26 Oct 2026): GPT-6.1 Sol directly, and its Batch API
through the same batch gate as Claude's Message Batches.

The Anthropic key expired after 30 days, and Rishab approved running the
extender on OpenAI's counterpart to Opus 5.5 meanwhile. Every test here
uses a stand-in for OpenAI: no network, no tokens."""
import json
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))
sys.path.insert(0, str(ROOT / "initiator"))

import llm_client                                               # noqa: E402

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}},
          "required": ["x"], "additionalProperties": False}


def completion(content='{"x": 1}', finish="stop", refusal=None,
               prompt=11, done=7):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": content,
                                     "refusal": refusal}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": done}}


class FakeOpenAI:
    """Chat Completions, Files and Batches, enough for llm_client."""

    def __init__(self, reply=None, batch_status="completed", output=None,
                 errors_file=None, failed_reasons=None):
        self.sent, self.uploads, self.batches_made = [], [], []
        self.reply = reply or completion()
        self.status = batch_status
        self.output, self.errors_file = output, errors_file
        self.failed_reasons = failed_reasons or []
        outer = self

        class _Completions:
            def create(self, **body):
                outer.sent.append(body)
                return types.SimpleNamespace(
                    model_dump=lambda: outer.reply)

        class _Files:
            def create(self, *, file, purpose):
                outer.uploads.append((file[0], file[1].decode(), purpose))
                return types.SimpleNamespace(id="file-in")

            def content(self, file_id):
                text = {"file-out": outer.output,
                        "file-err": outer.errors_file}[file_id]
                return types.SimpleNamespace(text=text)

        class _Batches:
            def create(self, **kw):
                outer.batches_made.append(kw)
                return types.SimpleNamespace(id="batch_1")

            def retrieve(self, batch_id):
                errors = types.SimpleNamespace(data=[
                    types.SimpleNamespace(message=m)
                    for m in outer.failed_reasons])
                return types.SimpleNamespace(
                    status=outer.status,
                    request_counts=types.SimpleNamespace(
                        total=2, completed=1, failed=1),
                    output_file_id="file-out" if outer.output else None,
                    error_file_id="file-err" if outer.errors_file else None,
                    errors=errors)

        self.chat = types.SimpleNamespace(completions=_Completions())
        self.files = _Files()
        self.batches = _Batches()


@pytest.fixture
def openai_route(monkeypatch):
    monkeypatch.setenv("CLAUDE_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    def use(fake):
        monkeypatch.setattr(llm_client, "_oa_client", fake)
        return fake
    return use


# --- the direct call ----------------------------------------------------------

def test_the_route_sends_one_strict_chat_request(openai_route):
    fake = openai_route(FakeOpenAI())
    llm_client.call_claude(model="claude-opus-5-5", max_tokens=64000,
                           user="grow it", system="You extend designs.",
                           schema=SCHEMA, effort="high")
    [body] = fake.sent
    assert body["model"] == "gpt-6.1-sol"
    # system first, as a training example rebuilds the turns
    assert body["messages"] == [
        {"role": "system", "content": "You extend designs."},
        {"role": "user", "content": "grow it"}]
    assert body["max_completion_tokens"] == 64000
    assert body["reasoning_effort"] == "high"
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "strict": True, "schema": SCHEMA}}


def test_an_answer_is_read_like_a_claude_answer(openai_route, tmp_path):
    openai_route(FakeOpenAI(reply=completion(
        content='Here it is: {"x": 3} done', prompt=40, done=9)))
    with llm_client.call_log(tmp_path / "calls.jsonl"):
        text, usage, stop = llm_client.call_claude(
            model="claude-opus-5-5", max_tokens=100, user="u",
            schema=SCHEMA)
    assert json.loads(text) == {"x": 3} and stop == "ok"
    assert usage == {"input": 40, "output": 9}
    [entry] = [json.loads(l) for l in
               (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert entry["provider"] == "openai"
    assert entry["model"] == "openai:gpt-6.1-sol"
    assert entry["batch"] is False
    assert entry["raw"] == 'Here it is: {"x": 3} done'


def test_a_refusal_and_a_cut_off_answer(openai_route):
    openai_route(FakeOpenAI(reply=completion(content=None,
                                             refusal="I can't help.")))
    assert llm_client.call_claude(model="m", max_tokens=9, user="u")[2] \
        == "refusal"
    assert llm_client.last_call()["refusal"] == {
        "category": "refusal", "explanation": "I can't help."}
    openai_route(FakeOpenAI(reply=completion(content='{"x": ',
                                             finish="length")))
    text, _, stop = llm_client.call_claude(model="m", max_tokens=9, user="u")
    assert text is None and stop == "length"


@pytest.mark.parametrize("ours, theirs", [("high", "high"), ("max", "high"),
                                          ("medium", "medium")])
def test_effort_names_map_to_openais(ours, theirs):
    body = llm_client.openai_request_body(max_tokens=1, user="u",
                                          effort=ours)
    assert body["reasoning_effort"] == theirs


def test_the_model_can_be_chosen_and_is_named_in_the_logs(monkeypatch):
    monkeypatch.setenv("CLAUDE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-astra")
    assert llm_client.openai_request_body(max_tokens=1, user="u")["model"] \
        == "gpt-6-astra"
    assert llm_client.model_label("claude-opus-5-5") == "openai:gpt-6-astra"


def test_the_counterpart_is_priced_at_its_own_list_price():
    assert llm_client.dollars("openai:gpt-6.1-sol", 1_000_000, 1_000_000) \
        == pytest.approx(12.0)
    assert llm_client.dollars("openai:gpt-6.1-sol", 1_000_000, 1_000_000,
                              batch=True) == pytest.approx(6.0)


# --- the Batch API ----------------------------------------------------------

def test_a_batch_uploads_one_line_per_request(openai_route):
    fake = openai_route(FakeOpenAI())
    bodies = [("r00001", llm_client.openai_request_body(max_tokens=5,
                                                        user="a")),
              ("r00002", llm_client.openai_request_body(max_tokens=5,
                                                        user="b"))]
    assert llm_client.batch_submit(bodies) == "batch_1"
    [(name, text, purpose)] = fake.uploads
    assert purpose == "batch" and name.endswith(".jsonl")
    lines = [json.loads(l) for l in text.splitlines()]
    assert [l["custom_id"] for l in lines] == ["r00001", "r00002"]
    assert all(l["method"] == "POST" and l["url"] == "/v1/chat/completions"
               for l in lines)
    assert lines[1]["body"] == bodies[1][1]
    assert fake.batches_made == [{"input_file_id": "file-in",
                                  "endpoint": "/v1/chat/completions",
                                  "completion_window": "24h"}]


@pytest.mark.parametrize("theirs, ours", [
    ("validating", "in_progress"), ("in_progress", "in_progress"),
    ("finalizing", "in_progress"), ("cancelling", "canceling"),
    ("completed", "ended"), ("failed", "ended"), ("expired", "ended"),
    ("cancelled", "ended"), ("a_state_added_later", "ended")])
def test_every_batch_state_reads_as_the_gate_expects(openai_route, theirs,
                                                     ours):
    openai_route(FakeOpenAI(batch_status=theirs))
    state, counts = llm_client.batch_status("batch_1")
    assert state == ours and counts == {"total": 2, "completed": 1,
                                        "failed": 1}


def test_batch_results_read_every_line(openai_route):
    ok = {"custom_id": "r00001", "error": None,
          "response": {"status_code": 200, "body": completion('{"x": 5}')}}
    refused = {"custom_id": "r00002", "error": None,
               "response": {"status_code": 200,
                            "body": completion(None, refusal="no")}}
    bad = {"custom_id": "r00003", "error": None,
           "response": {"status_code": 400, "body": {"error": {
               "type": "invalid_request_error", "code": "bad_schema"}}}}
    expired = {"custom_id": "r00004", "response": None,
               "error": {"code": "batch_expired", "message": "too late"}}
    openai_route(FakeOpenAI(
        output="\n".join(json.dumps(x) for x in (ok, refused)) + "\n",
        errors_file="\n".join(json.dumps(x) for x in (bad, expired))))
    got = dict(llm_client.batch_results("batch_1"))
    assert got["r00001"][0] == "ok" and json.loads(got["r00001"][1]) == {
        "x": 5}
    assert got["r00001"][2] == {"input": 11, "output": 7}
    assert got["r00002"][0] == "refusal" and got["r00002"][4]["explanation"] \
        == "no"
    assert got["r00003"][0] == "error:bad_schema"
    assert got["r00004"][0] == "error:batch_expired"


def test_a_batch_openai_turned_away_says_why(openai_route):
    openai_route(FakeOpenAI(batch_status="failed",
                            failed_reasons=["Line 1: unknown model"]))
    with pytest.raises(RuntimeError, match="unknown model"):
        list(llm_client.batch_results("batch_1"))


def test_the_gate_sends_openai_calls_as_one_half_price_batch(openai_route,
                                                            tmp_path):
    fake = openai_route(FakeOpenAI())

    def answer_both():
        lines = [json.loads(l) for l in fake.uploads[-1][1].splitlines()]
        fake.output = "".join(json.dumps({
            "custom_id": l["custom_id"], "error": None,
            "response": {"status_code": 200, "body": completion(
                json.dumps({"x": int(l["body"]["messages"][-1]["content"])}))}
        }) + "\n" for l in lines)
    original = fake.batches.create

    def create_and_answer(**kw):
        made = original(**kw)
        answer_both()
        return made
    fake.batches.create = create_and_answer
    got = {}

    def ask(n):
        got[n] = llm_client.call_claude(model="claude-opus-5-5",
                                        max_tokens=9, user=str(n),
                                        schema=SCHEMA)

    with llm_client.call_log(tmp_path / "calls.jsonl"):
        with llm_client.batch_gate(flush_after_s=0.05, poll_s=0.01,
                                   manifest_dir=tmp_path / "batches"):
            threads = [threading.Thread(target=ask, args=(n,))
                       for n in (4, 6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    assert {n: json.loads(v[0])["x"] for n, v in got.items()} == {4: 4, 6: 6}
    assert len(fake.uploads) == 1           # both questions, one batch
    entries = [json.loads(l) for l in
               (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert {e["batch"] for e in entries} == {True}
    assert {e["batch_id"] for e in entries} == {"batch_1"}
    assert {e["model"] for e in entries} == {"openai:gpt-6.1-sol"}


def test_batches_still_refuse_a_route_without_them(monkeypatch):
    monkeypatch.setenv("CLAUDE_PROVIDER", "openrouter")
    with pytest.raises(RuntimeError, match="Anthropic or OpenAI"):
        with llm_client.batch_gate():
            pass


# --- the answer formats meet OpenAI's strict rules ---------------------------

UNSUPPORTED = {"minItems", "maxItems", "minLength", "maxLength", "pattern",
               "format", "minimum", "maximum", "uniqueItems",
               "patternProperties", "default", "oneOf", "allOf", "not"}


def strict_problems(schema, path="$"):
    out = []
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            props = set((schema.get("properties") or {}))
            if set(schema.get("required") or []) != props:
                out.append(f"{path}: not every property is required")
            if schema.get("additionalProperties") is not False:
                out.append(f"{path}: extra properties allowed")
        out += [f"{path}: {k}" for k in schema if k in UNSUPPORTED]
        for k, v in schema.items():
            out += strict_problems(v, f"{path}.{k}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            out += strict_problems(v, f"{path}[{i}]")
    return out


def every_schema():
    import batch
    import extend
    import sense_check
    import schema as initiator_schema
    found = {}
    for mod in (extend, sense_check, batch, initiator_schema):
        for name, value in vars(mod).items():
            if name.isupper() and isinstance(value, dict) \
                    and value.get("type") == "object":
                found[f"{mod.__name__}.{name}"] = value
    for move in extend.MOVES:
        found[f"extend.structural_schema({move})"] = \
            extend.structural_schema(move)
    return found


def test_every_answer_format_meets_openais_strict_rules():
    schemas = every_schema()
    assert len(schemas) >= 10
    bad = {name: strict_problems(s) for name, s in schemas.items()
           if strict_problems(s)}
    assert bad == {}


def test_the_report_finds_the_system_text_of_an_openai_request():
    import realism_report
    import sense_check
    entry = {"request": llm_client.openai_request_body(
        max_tokens=1, user="u", system=sense_check.SYSTEM)}
    assert realism_report.call_kind(entry) == "reviews"
