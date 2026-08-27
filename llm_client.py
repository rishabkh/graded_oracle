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


def _call_anthropic(*, model, max_tokens, user, system, schema, effort):
    global _an_client
    import anthropic
    if _an_client is None:
        _an_client = anthropic.Anthropic()
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
    resp = _an_client.messages.create(**kwargs)
    usage = {"input": resp.usage.input_tokens,
             "output": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        return None, usage, "refusal"
    global LAST_RAW
    text = next((b.text for b in resp.content if b.type == "text"), "")
    LAST_RAW = text
    if resp.stop_reason == "max_tokens" and extract_json(text) is None:
        return None, usage, "length"
    return text, usage, "ok"


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
    usage = {"input": resp.usage.prompt_tokens,
             "output": resp.usage.completion_tokens}
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
