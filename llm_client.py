"""One switch for how Claude is reached: the Anthropic API directly
(default) or through OpenRouter — so the harness runs on whichever key
the project has.

  # default: direct Anthropic (exact current behaviour)
  export ANTHROPIC_API_KEY=sk-ant-...

  # via OpenRouter:
  export CLAUDE_PROVIDER=openrouter
  export OPENROUTER_API_KEY=sk-or-...
  export OPENROUTER_CLAUDE_MODEL=anthropic/claude-opus-5   # optional; check
                                                           # openrouter.ai/models
                                                           # for the exact slug

  # OpenAI directly (26 Oct 2026), batches included:
  export CLAUDE_PROVIDER=openai
  export OPENAI_API_KEY=sk-proj-...
  export OPENAI_MODEL=gpt-6.1-sol        # optional; the default

Differences that matter, handled here:
- Anthropic enforces the JSON schema server-side (output_config); OpenRouter
  forwards an OpenAI-style response_format when the provider supports it and
  we parse leniently as a fallback, so a model that wraps JSON in prose or
  fences still yields the object.
- Refusals: Anthropic signals stop_reason=="refusal"; OpenAI-style signals
  finish_reason=="content_filter".
- Usage is normalised to {"input": .., "output": ..} either way.
"""
import contextlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_or_client = None
_an_client = None

# Raw text of the last reply, kept so an unparseable/truncated answer can
# be logged instead of vanishing with the money that bought it.
LAST_RAW = None


def provider():
    return os.environ.get("CLAUDE_PROVIDER", "anthropic")


# 26 Oct 2026, Rishab approved: OpenAI as a third route, for extender runs
# while the Anthropic key was out (it expired after 30 days). GPT-6.1 Sol is
# OpenAI's tier below its top model, as Opus 5.5 is below Fable; it came
# out on 29 Sep 2026. Every call site keeps passing its Claude model id;
# this route ignores it and calls OPENAI_MODEL, and the logs name the
# model that really answered.
OPENAI_MODEL_DEFAULT = "gpt-6.1-sol"


def openai_model():
    return os.environ.get("OPENAI_MODEL", OPENAI_MODEL_DEFAULT)


def model_label(anthropic_model):
    """What to record in logs: the model actually called, provider-tagged."""
    if provider() == "openrouter":
        slug = os.environ.get("OPENROUTER_CLAUDE_MODEL",
                              "anthropic/claude-opus-5")
        return f"openrouter:{slug}"
    if provider() == "openai":
        return f"openai:{openai_model()}"
    return anthropic_model


# Published list prices, dollars per million tokens (input, output).
# Opus 5.5 replaced Opus 5 for every call on 8 Oct 2026 at 20% less.
# GPT-6.1 Sol: $2 in, $10 out (OpenAI's pricing page as quoted on 4 and 11
# Oct 2026); its Batch API is half price too.
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-opus-5-5": (4.0, 20.0),
          "gpt-6.1-sol": (2.0, 10.0)}


def dollars(model, tokens_in, tokens_out, batch=False):
    """What a call cost at its own model's list price, halved for the
    Message Batches API. A logged label ("openrouter:anthropic/claude-opus-5")
    prices like the bare id; an unknown model raises, so a new one is
    never priced silently wrong."""
    key = str(model).split("/")[-1].split(":")[-1]
    if key not in PRICES:
        raise ValueError(f"no list price for {model!r}; add it to PRICES")
    p_in, p_out = PRICES[key]
    cost = tokens_in * p_in / 1e6 + tokens_out * p_out / 1e6
    return cost / 2 if batch else cost


def extract_json(text):
    """First complete JSON object in `text`, or None. Balanced-brace scan,
    string- and escape-aware, so prose or fences around the object are
    harmless."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif in_str:
                if c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def call_claude(*, model, max_tokens, user, system=None, schema=None,
                effort=None):
    """One structured-output call. Returns (raw_json_text | None, usage,
    stop) where stop is "ok" or "refusal" and usage is
    {"input": int, "output": int}. raw_json_text is None only on refusal
    or (OpenRouter path) unparseable output."""
    global LAST_RAW
    entry = {"started": datetime.now(timezone.utc).isoformat(),
             "provider": provider(),
             "model": model_label(model) if provider() == "openai" else model,
             "batch": _gate is not None and provider() in BATCH_PROVIDERS,
             "batch_id": None, "custom_id": None, "request": None,
             "raw": None, "stop": None, "usage": None, "refusal": None}
    t0 = time.monotonic()
    try:
        if provider() == "openai":
            body = openai_request_body(max_tokens=max_tokens, user=user,
                                       system=system, schema=schema,
                                       effort=effort)
            entry["request"] = body
            if _gate is not None:
                stop, text, usage, raw, info = _gate.call(body)
                entry.update(info)
                if stop.startswith("error"):
                    entry.update(stop=stop, usage=usage)
                    raise RuntimeError(f"batch {stop}")
            else:
                text, usage, stop, raw, refusal = _call_openai(body)
                entry["refusal"] = refusal
            LAST_RAW = raw
        elif provider() == "openrouter":
            entry["request"] = {"model": model, "max_tokens": max_tokens,
                                "system": system, "user": user,
                                "schema": schema, "effort": effort}
            text, usage, stop = _call_openrouter(
                model=model, max_tokens=max_tokens, user=user, system=system,
                schema=schema, effort=effort)
            raw = LAST_RAW
        else:
            kwargs = request_kwargs(model=model, max_tokens=max_tokens,
                                    user=user, system=system, schema=schema,
                                    effort=effort)
            entry["request"] = kwargs
            if _gate is not None:
                stop, text, usage, raw, info = _gate.call(kwargs)
                entry.update(info)
                if stop.startswith("error"):
                    entry.update(stop=stop, usage=usage)
                    raise RuntimeError(f"batch {stop}")
            else:
                text, usage, stop, raw, *reason = _call_anthropic(kwargs)
                entry["refusal"] = reason[0] if reason else None
            LAST_RAW = raw
    except Exception as exc:
        if not entry["stop"]:
            entry["stop"] = f"exception: {type(exc).__name__}: {exc}"[:300]
        _record_call(entry, t0)
        raise
    entry.update(raw=raw, stop=stop, usage=usage)
    _record_call(entry, t0)
    return text, usage, stop


# --- the call log (8 Oct 2026) ----------------------------------------------
# Every call, from any script, saved at the source: the exact request, the
# raw answer (even a refusal or a cut-off one), tokens, how it ended,
# timing, and for a batch its batch id and custom id. Scripts open it with
# `with call_log(path):`; last_call() gives the current thread's last entry.

_call_log_path = None
_call_log_lock = threading.Lock()
_thread = threading.local()


def append_call(path, entry):
    """One call line, for answers that never pass through call_claude (a
    one-call batch collected straight from the batch)."""
    line = json.dumps(entry, default=str) + "\n"
    with _call_log_lock:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a") as f:
            f.write(line)


def _record_call(entry, t0):
    entry["seconds"] = round(time.monotonic() - t0, 2)
    _thread.call = entry
    if _call_log_path is not None:
        append_call(_call_log_path, entry)


def last_call():
    """This thread's most recent call entry, or None."""
    return getattr(_thread, "call", None)


@contextlib.contextmanager
def call_log(path):
    global _call_log_path
    previous, _call_log_path = _call_log_path, Path(path)
    try:
        yield Path(path)
    finally:
        _call_log_path = previous


def request_kwargs(*, model, max_tokens, user, system=None, schema=None,
                   effort=None):
    """The Messages API request, built once for both the direct call and a
    batch entry, so the two cannot drift apart."""
    output_config = {}
    if effort:
        output_config["effort"] = effort
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs = dict(model=model, max_tokens=max_tokens,
                  messages=[{"role": "user", "content": user}])
    if output_config:
        kwargs["output_config"] = output_config
    if system:
        kwargs["system"] = system
    return kwargs


def read_message(resp):
    """One reply read the same way in both modes: (raw_json_text | None,
    usage, stop, raw_text) with stop "ok", "refusal" or "length" (cut off
    with no complete JSON object). raw_text keeps what came back even
    when it is unusable, so a paid answer can still be logged."""
    usage = {"input": resp.usage.input_tokens,
             "output": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        return None, usage, "refusal", None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    if resp.stop_reason == "max_tokens" and extract_json(text) is None:
        return None, usage, "length", text
    return text, usage, "ok", text


def refusal_reason(resp):
    """Anthropic's stated reason for a refusal, {"category", "explanation"},
    or None for any other reply. Either value may be None: stop_details can
    be missing even on a refusal. The call log keeps it since 26 Oct 2026;
    the 7 refusals of the 25 Oct initiator run had to be read back from
    their batches by hand."""
    if resp.stop_reason != "refusal":
        return None
    details = getattr(resp, "stop_details", None)
    return {"category": getattr(details, "category", None),
            "explanation": getattr(details, "explanation", None)}


def _client():
    global _an_client
    if _an_client is None:
        import anthropic
        _an_client = anthropic.Anthropic()
    return _an_client


def _call_anthropic(kwargs):
    """(text, usage, stop, raw, refusal). raw is None on a refusal: before
    8 Oct 2026 LAST_RAW kept the previous reply then, so a refused attempt
    was logged with another attempt's text. refusal is refusal_reason's."""
    # stream, not create: the SDK refuses a plain create() whose max_tokens
    # could run past ten minutes, which is every call at our 32k budget
    with _client().messages.stream(**kwargs) as s:
        resp = s.get_final_message()
    return read_message(resp) + (refusal_reason(resp),)


# --- Message Batches API: the same requests at half price, answered within
# 24 hours (usually under one). Anthropic direct, and since 26 Oct 2026
# OpenAI's Batch API through the same three functions.

BATCH_PROVIDERS = ("anthropic", "openai")


def batch_submit(items):
    """items: [(custom_id, request_kwargs(...))], or OpenAI request
    bodies on that route. Returns the batch id."""
    if provider() == "openai":
        return _openai_batch_submit(items)
    batch = _client().messages.batches.create(
        requests=[{"custom_id": cid, "params": kw} for cid, kw in items])
    return batch.id


def batch_status(batch_id):
    """("in_progress" | "canceling" | "ended", request counts)."""
    if provider() == "openai":
        return _openai_batch_status(batch_id)
    b = _client().messages.batches.retrieve(batch_id)
    c = b.request_counts
    return b.processing_status, {k: getattr(c, k, 0) for k in (
        "processing", "succeeded", "errored", "canceled", "expired")}


def batch_results(batch_id):
    """Yields (custom_id, (stop, raw_json_text | None, usage, raw_text,
    refusal)), a succeeded entry read exactly as a direct reply
    (read_message, refusal_reason); an errored, canceled or expired one as
    stop "error:<kind>"."""
    if provider() == "openai":
        yield from _openai_batch_results(batch_id)
        return
    zero = {"input": 0, "output": 0}
    for r in _client().messages.batches.results(batch_id):
        kind = r.result.type
        if kind == "succeeded":
            text, usage, stop, raw = read_message(r.result.message)
            yield r.custom_id, (stop, text, usage, raw,
                                refusal_reason(r.result.message))
        elif kind == "errored":
            err = getattr(r.result, "error", None)
            etype = getattr(getattr(err, "error", None), "type", None) \
                or getattr(err, "type", None) or "unknown"
            yield r.custom_id, (f"error:{etype}", None, zero, None, None)
        else:
            yield r.custom_id, (f"error:{kind}", None, zero, None, None)


def _call_openrouter(*, model, max_tokens, user, system, schema, effort):
    global _or_client
    from openai import OpenAI
    if _or_client is None:
        _or_client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL",
                                    "https://openrouter.ai/api/v1"),
            api_key=os.environ["OPENROUTER_API_KEY"])
    slug = os.environ.get("OPENROUTER_CLAUDE_MODEL",
                          "anthropic/claude-opus-5")
    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": user}]
    kwargs = dict(model=slug, max_tokens=max_tokens, messages=messages)
    if effort:
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "strict": True,
                            "schema": schema}}
    try:
        resp = _or_client.chat.completions.create(**kwargs)
    except Exception:
        # Some provider routes reject response_format; retry without it and
        # rely on the schema instructions already present in the prompt.
        kwargs.pop("response_format", None)
        resp = _or_client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    # some provider routes answer with no usage block at all; that must
    # not turn a paid call into an ERROR the retry path cannot rescue
    u = getattr(resp, "usage", None)
    usage = {"input": getattr(u, "prompt_tokens", 0) or 0,
             "output": getattr(u, "completion_tokens", 0) or 0}
    if choice.finish_reason == "content_filter":
        return None, usage, "refusal"
    global LAST_RAW
    text = choice.message.content or ""
    LAST_RAW = text
    obj = extract_json(text)
    if obj is None:
        stop = "length" if choice.finish_reason == "length" else "ok"
        return None, usage, stop
    # Re-serialise so callers always receive clean JSON text, exactly as
    # the Anthropic structured-output path would have produced.
    return json.dumps(obj), usage, "ok"


# --- OpenAI directly (26 Oct 2026) ------------------------------------------
# The same structured-output call through OpenAI's Chat Completions API,
# and the same half-price batches through its Batch API (a file of
# requests, answered within 24 hours). The system text goes in as a
# "system" message, so a saved training example rebuilds the same chat
# turns as a Claude call (batch.sent_messages). All our schemas already
# meet OpenAI's strict rules (every property required, no extra
# properties; tests/test_llm_client_openai.py checks each one).

_oa_client = None

# our effort names, as OpenAI's reasoning_effort reads them
OPENAI_EFFORT = {"low": "low", "medium": "medium", "high": "high",
                 "xhigh": "high", "max": "high"}


def _openai():
    global _oa_client
    if _oa_client is None:
        from openai import OpenAI
        # a long answer at high effort can take many minutes
        _oa_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                            timeout=3600)
    return _oa_client


def openai_request_body(*, max_tokens, user, system=None, schema=None,
                        effort=None):
    """The Chat Completions request, built once for both the direct call
    and a batch line, so the two cannot drift apart."""
    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": user}]
    body = {"model": openai_model(), "messages": messages,
            "max_completion_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = OPENAI_EFFORT.get(effort, effort)
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "strict": True,
                            "schema": schema}}
    return body


def read_openai_completion(body):
    """One chat completion (as a dict) read like read_message reads a
    Claude reply: (raw_json_text | None, usage, stop, raw_text, refusal),
    stop "ok", "refusal" or "length" (cut off with no complete JSON
    object). The JSON text is re-serialised clean; raw_text keeps what
    came back."""
    u = body.get("usage") or {}
    usage = {"input": u.get("prompt_tokens") or 0,
             "output": u.get("completion_tokens") or 0}
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    if message.get("refusal") or finish == "content_filter":
        return None, usage, "refusal", None, {
            "category": "refusal" if message.get("refusal")
            else "content_filter",
            "explanation": message.get("refusal")}
    text = message.get("content") or ""
    obj = extract_json(text)
    if obj is None:
        return None, usage, ("length" if finish == "length" else "ok"), \
            text, None
    return json.dumps(obj), usage, "ok", text, None


def _call_openai(body):
    """(text, usage, stop, raw, refusal), as _call_anthropic gives."""
    resp = _openai().chat.completions.create(**body)
    return read_openai_completion(resp.model_dump())


def _openai_batch_submit(items):
    lines = "".join(json.dumps({"custom_id": cid, "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": body}) + "\n"
                    for cid, body in items)
    upload = _openai().files.create(
        file=("batch_requests.jsonl", lines.encode()), purpose="batch")
    batch = _openai().batches.create(input_file_id=upload.id,
                                     endpoint="/v1/chat/completions",
                                     completion_window="24h")
    return batch.id


# OpenAI's batch states, as the gate reads Anthropic's: a state not listed
# here counts as ended, so a new one can never hold the run forever.
OPENAI_BATCH_STATES = {"validating": "in_progress", "in_progress": "in_progress",
                       "finalizing": "in_progress", "cancelling": "canceling",
                       "completed": "ended", "failed": "ended",
                       "expired": "ended", "cancelled": "ended"}


def _openai_batch_status(batch_id):
    b = _openai().batches.retrieve(batch_id)
    c = b.request_counts
    return OPENAI_BATCH_STATES.get(b.status, "ended"), {
        k: getattr(c, k, 0) or 0 for k in ("total", "completed", "failed")}


def _openai_batch_results(batch_id):
    """Yields (custom_id, (stop, raw_json_text | None, usage, raw_text,
    refusal)) from the batch's output and error files; a request that
    failed reads as stop "error:<code>". A batch OpenAI turned away whole
    (a request it could not read) raises with OpenAI's reasons."""
    b = _openai().batches.retrieve(batch_id)
    if b.status == "failed":
        errors = getattr(getattr(b, "errors", None), "data", None) or []
        reasons = "; ".join(str(getattr(e, "message", e)) for e in errors)
        raise RuntimeError(f"OpenAI batch failed: {reasons[:600]}")
    zero = {"input": 0, "output": 0}
    for file_id in (b.output_file_id, b.error_file_id):
        if not file_id:
            continue
        for line in _openai().files.content(file_id).text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            response = row.get("response") or {}
            if response.get("status_code") == 200 \
                    and isinstance(response.get("body"), dict):
                text, usage, stop, raw, refusal = read_openai_completion(
                    response["body"])
                yield row["custom_id"], (stop, text, usage, raw, refusal)
                continue
            err = row.get("error") or \
                (response.get("body") or {}).get("error") or {}
            code = err.get("code") or err.get("type") \
                or response.get("status_code") or "unknown"
            yield row["custom_id"], (f"error:{code}", None, zero, None, None)


# --- the batch gate (8 Oct 2026) -------------------------------------------
# Many threads making ordinary call_claude calls, sent as Message Batches at
# half price. Inside `with batch_gate(...)`, a call queues its request and
# blocks its own thread; once no new request has arrived for flush_after_s
# (or max_size are queued), the queue goes out as one batch, and each
# thread gets back its own answer, read by read_message exactly like a
# direct reply. A failed entry raises in its own thread, as a failed direct
# call would. Every batch is written to manifest_dir before it is sent, and
# again with its id and token use, so paid work can always be traced.

import contextlib                                              # noqa: E402
import threading                                               # noqa: E402
import time                                                    # noqa: E402
from datetime import datetime, timezone                        # noqa: E402
from pathlib import Path                                       # noqa: E402

_gate = None


class _Gate:
    def __init__(self, flush_after_s, poll_s, manifest_dir, max_size):
        self.flush_after_s, self.poll_s, self.max_size = \
            flush_after_s, poll_s, max_size
        self.manifest_dir = Path(manifest_dir) if manifest_dir else None
        self.lock = threading.Lock()
        self.pending, self.senders = [], []
        self.seq, self.last = 0, time.monotonic()
        self.closing = threading.Event()
        self.flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self.flusher.start()

    def call(self, kwargs):
        slot = {"event": threading.Event(), "result": None}
        with self.lock:
            self.seq += 1
            self.pending.append((f"r{self.seq:05d}", kwargs, slot))
            self.last = time.monotonic()
        slot["event"].wait()
        if isinstance(slot["result"], Exception):
            raise slot["result"]
        return slot["result"]

    def _flush_loop(self):
        while True:
            time.sleep(min(0.05, self.flush_after_s))
            with self.lock:
                quiet = time.monotonic() - self.last >= self.flush_after_s
                if self.pending and (quiet or len(self.pending) >= self.max_size
                                     or self.closing.is_set()):
                    take, self.pending = self.pending, []
                elif self.closing.is_set() and not self.pending:
                    return
                else:
                    continue
            sender = threading.Thread(target=self._send, args=(take,),
                                      daemon=True)
            sender.start()
            self.senders.append(sender)

    def _write(self, path, rec):
        if path is not None:
            path.write_text(json.dumps(rec, indent=1))

    def _send(self, take):
        items = [(cid, kw) for cid, kw, _ in take]
        slots = {cid: slot for cid, _, slot in take}
        rec = {"created": datetime.now(timezone.utc).isoformat(),
               "batch_id": None,
               "requests": [{"custom_id": c, "params": kw} for c, kw in items]}
        path = None
        if self.manifest_dir is not None:
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            path = self.manifest_dir / f"batch_{stamp}_{items[0][0]}.json"
        self._write(path, rec)
        try:
            rec["batch_id"] = batch_submit(items)
            self._write(path, rec)
            while batch_status(rec["batch_id"])[0] != "ended":
                time.sleep(self.poll_s)
            got = dict(batch_results(rec["batch_id"]))
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            self._write(path, rec)
            for slot in slots.values():
                slot["result"] = RuntimeError(f"batch failed: {exc}")
                slot["event"].set()
            return
        usage = {"input": 0, "output": 0}
        results = {}
        for cid in slots:
            out = got.get(cid, ("error:no_result", None,
                                {"input": 0, "output": 0}, None))
            usage["input"] += out[2].get("input", 0)
            usage["output"] += out[2].get("output", 0)
            results[cid] = out
        rec.update(usage=usage, ended=datetime.now(timezone.utc).isoformat())
        self._write(path, rec)
        for cid, slot in slots.items():
            stop, text, use, raw, *reason = results[cid]
            info = {"batch_id": rec["batch_id"], "custom_id": cid}
            if reason and reason[0]:
                info["refusal"] = reason[0]
            slot["result"] = (stop, text, use, raw, info)
            slot["event"].set()

    def close(self):
        self.closing.set()
        self.flusher.join()
        for sender in self.senders:
            sender.join()


@contextlib.contextmanager
def batch_gate(flush_after_s=20, poll_s=60, manifest_dir=None,
               max_size=10000):
    """Inside the block, call_claude calls from any thread go out as
    batches at half price (see the notes above): Anthropic's Message
    Batches, or OpenAI's Batch API on that route (26 Oct 2026)."""
    global _gate
    if provider() not in BATCH_PROVIDERS:
        raise RuntimeError("batch mode needs the Anthropic or OpenAI API, "
                           f"not {provider()}")
    gate = _Gate(flush_after_s, poll_s, manifest_dir, max_size)
    _gate = gate
    try:
        yield gate
    finally:
        gate.close()
        _gate = None
