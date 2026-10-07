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

Differences that matter, handled here:
- Anthropic enforces the JSON schema server-side (output_config); OpenRouter
  forwards an OpenAI-style response_format when the provider supports it and
  we parse leniently as a fallback, so a model that wraps JSON in prose or
  fences still yields the object.
- Refusals: Anthropic signals stop_reason=="refusal"; OpenAI-style signals
  finish_reason=="content_filter".
- Usage is normalised to {"input": .., "output": ..} either way.
"""
import json
import os

_or_client = None
_an_client = None

# Raw text of the last reply, kept so an unparseable/truncated answer can
# be logged instead of vanishing with the money that bought it.
LAST_RAW = None


def provider():
    return os.environ.get("CLAUDE_PROVIDER", "anthropic")


def model_label(anthropic_model):
    """What to record in logs: the model actually called, provider-tagged."""
    if provider() == "openrouter":
        slug = os.environ.get("OPENROUTER_CLAUDE_MODEL",
                              "anthropic/claude-opus-5")
        return f"openrouter:{slug}"
    return anthropic_model


# Published list prices, dollars per million tokens (input, output).
# Opus 5.5 replaced Opus 5 for every call on 8 Oct 2026 at 20% less.
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-opus-5-5": (4.0, 20.0)}


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
    if provider() == "openrouter":
        return _call_openrouter(model=model, max_tokens=max_tokens,
                                user=user, system=system, schema=schema,
                                effort=effort)
    return _call_anthropic(model=model, max_tokens=max_tokens, user=user,
                           system=system, schema=schema, effort=effort)


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


def _client():
    global _an_client
    if _an_client is None:
        import anthropic
        _an_client = anthropic.Anthropic()
    return _an_client


def _call_anthropic(*, model, max_tokens, user, system, schema, effort):
    kwargs = request_kwargs(model=model, max_tokens=max_tokens, user=user,
                            system=system, schema=schema, effort=effort)
    # stream, not create: the SDK refuses a plain create() whose max_tokens
    # could run past ten minutes, which is every call at our 32k budget
    with _client().messages.stream(**kwargs) as s:
        resp = s.get_final_message()
    global LAST_RAW
    text, usage, stop, raw = read_message(resp)
    # None on a refusal: before 8 Oct 2026 it kept the previous reply, so
    # a refused attempt was logged with another attempt's text
    LAST_RAW = raw
    return text, usage, stop


# --- Message Batches API: the same requests at half price, answered within
# 24 hours (usually under one). Anthropic direct only.

def batch_submit(items):
    """items: [(custom_id, request_kwargs(...))]. Returns the batch id."""
    batch = _client().messages.batches.create(
        requests=[{"custom_id": cid, "params": kw} for cid, kw in items])
    return batch.id


def batch_status(batch_id):
    """("in_progress" | "canceling" | "ended", request counts)."""
    b = _client().messages.batches.retrieve(batch_id)
    c = b.request_counts
    return b.processing_status, {k: getattr(c, k, 0) for k in (
        "processing", "succeeded", "errored", "canceled", "expired")}


def batch_results(batch_id):
    """Yields (custom_id, (stop, raw_json_text | None, usage, raw_text)),
    a succeeded entry read exactly as a direct reply (read_message); an
    errored, canceled or expired one as stop "error:<kind>"."""
    zero = {"input": 0, "output": 0}
    for r in _client().messages.batches.results(batch_id):
        kind = r.result.type
        if kind == "succeeded":
            text, usage, stop, raw = read_message(r.result.message)
            yield r.custom_id, (stop, text, usage, raw)
        elif kind == "errored":
            err = getattr(r.result, "error", None)
            etype = getattr(getattr(err, "error", None), "type", None) \
                or getattr(err, "type", None) or "unknown"
            yield r.custom_id, (f"error:{etype}", None, zero, None)
        else:
            yield r.custom_id, (f"error:{kind}", None, zero, None)


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
