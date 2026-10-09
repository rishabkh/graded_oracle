"""Training examples in Formal Disco's shape, one line per model call.

Formal Disco saves every call as a DistillExample (formal-disco/
distill_common.py): `prompt` (the call type), `arguments` (its inputs),
`response`, `outcome` and `metadata`, and trains one model on all the call
types together. Ours uses the same five fields, so data from both systems
reads the same way, with two additions in `metadata`: the exact chat
`messages` that were sent (a later change to a prompt template must never
silently change what an example says), and our checker's own `verdict`.

Call types, Formal Disco's and their counterpart here:
  idea       README -> idea text            README + kind + size seeds -> design idea
  implement  idea -> program                idea -> design, property, helper facts
  initiate   README -> idea + program       our one-step generator
  generate   README -> program, self-repaired   the generator with self-repair;
                                            the response is the FINAL design
  repair     broken program + notes -> fix  broken design + checker notes -> fixed design
  extend     program -> diff                the extender (every type, combining too)
  solve      lemma_synth / hint removal     design -> helper facts (our test task)

Outcomes follow Formal Disco's: success (our NECESSARY, or PROVEN for a
solved design), fail (the checker judged the design and rejected it),
error (no design to judge: refused, cut off, unreadable, broken tool).

  record(log, "initiate", arguments, response, verdict, messages, **meta)
"""
import json
import threading
from pathlib import Path

TYPES = ("idea", "implement", "initiate", "generate", "repair", "extend",
         "solve")
SUCCESS = {"NECESSARY", "PROVEN"}
# no design reached the checker, or the checker itself could not judge
# (INCONCLUSIVE: the check without helper facts gave no verdict)
ERROR = {"REFUSED", "TRUNCATED", "UNPARSEABLE", "ERROR", "TIMEOUT",
         "CRASH", "check_error", "INCONCLUSIVE"}

_lock = threading.Lock()


def outcome_of(verdict, reason=None):
    """A check that ran out of time says nothing about the design (9 Oct
    2026: bigger designs make it common), so it is an error, not a
    failure the model should learn to avoid."""
    if verdict in SUCCESS:
        return "success"
    if verdict is None or verdict in ERROR or "TIMEOUT" in (reason or ""):
        return "error"
    return "fail"


def record(log, kind, arguments, response, verdict, messages, reason=None,
           **metadata):
    """Append one example. `messages` are the exact chat messages sent;
    `reason` is the checker's, which tells a timeout from a rejection."""
    if kind not in TYPES:
        raise ValueError(f"unknown call type {kind!r}; one of {TYPES}")
    if reason is not None:
        metadata["reason"] = reason
    line = {"prompt": kind, "arguments": arguments, "response": response,
            "outcome": outcome_of(verdict, reason),
            "metadata": dict(metadata, verdict=verdict, messages=messages)}
    text = json.dumps(line, default=str) + "\n"
    log = Path(log)
    with _lock:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(text)


def read(log):
    out = []
    for line in Path(log).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a line cut short by a killed run
    return out
