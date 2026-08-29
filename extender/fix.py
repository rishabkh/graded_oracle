"""The Fixer harness — DORMANT BY DESIGN. Nothing imports or calls this;
it stays unplugged until the fixer queue holds enough genuine proof
failures to read (the directive: collect them, read twenty, THEN write
the prompt against what you actually see). The prompt below is
PROVISIONAL plumbing-test text and is expected to be rewritten at that
point.

Contract (fixed now, because it is structural, not speculative):
- Input per task: design, property, the failed invariant list, and the
  summarised CTI from the grading evidence.
- Output: a NEW INVARIANT LIST ONLY. The harness rebuilds the graded
  payload from the original record's design and property itself, so the
  model cannot touch either — the cheapest repair is always to weaken
  what you are proving, and the corpus rots quietly if that is allowed.
- Three attempts per task, then the branch dies. Later attempts see the
  fresh CTI from the previous attempt's grading.
- EVERY attempt is logged (task id, attempt number, invariants, verdict,
  usage) to logs/fixer_attempts.jsonl — failures included, because
  failed-vs-fixed pairs on the same task are future preference data and
  regeneration is the expensive thing.

  venv/bin/python extender/fix.py --dry            # print prompt for first queued task
  venv/bin/python extender/fix.py --selftest       # hand repair through the grade path (no API)
  venv/bin/python extender/fix.py --task <ext_id>  # run ONE task (up to 3 attempts)
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import extract_asserts                 # noqa: E402
from extend import property_copy                         # noqa: E402
from extend_one import GRADE_KWARGS, MODEL, EFFORT, Spinner  # noqa: E402
from promote import child_top                            # noqa: E402

import llm_client                                        # noqa: E402
from oracle import grade_triple_generated                # noqa: E402

CORPUS = HERE / "corpus.jsonl"
QUEUE = HERE / "logs" / "fixer_queue.jsonl"
ATTEMPT_LOG = HERE / "logs" / "fixer_attempts.jsonl"
MAX_ATTEMPTS = 3
MAX_TOKENS = 16000

# ---------------------------------------------------------------------------
# PROVISIONAL PROMPT — written before real failures existed. Rewrite this
# against the first twenty genuine queue entries; do not tune it before then.
# ---------------------------------------------------------------------------
FIXER_PROMPT = """\
You are repairing the strengthening invariants of a formally verified
SystemVerilog module. The design and its property are FIXED — you may not
change either. Only the invariant list is yours.

The module:

{verilog}

The property (assertion in the source), which stays exactly as it is:
{property}

The invariant list that FAILED to close the proof:
{invariants}

The counterexample-to-induction the prover found — a state it may legally
start from under the current invariants, from which the property fails:

{cti}

Work it out in order:
1. diagnosis: why the shown state satisfies the current invariants yet
   breaks the proof — which missing fact would exclude it.
2. Then the COMPLETE corrected invariant list. The whole conjunction must
   be inductive; keep clauses as strong as the originals; add no clause
   you cannot justify; never copy the property text into the list; plain
   Verilog boolean expressions over top-level signals only (no inst.signal,
   no prose).

Reply with JSON: {{"diagnosis": "...", "invariants": ["..."]}}"""

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "invariants": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["diagnosis", "invariants"],
    "additionalProperties": False,
}


# --- pure helpers (unit-tested) ---

def failing_cti(record):
    """The summarised CTI from the grading evidence: prefer the
    with-invariants leg (the run the failed list actually lost), fall
    back to the without-leg. Returns (text | None, leg_name)."""
    result = record.get("result") or {}
    for leg in ("with_invariants", "without_invariants"):
        for run in ((result.get(leg) or {}).get("runs") or []):
            if run.get("trace_text"):
                return run["trace_text"], leg
    return None, None


def fixer_context(record, corpus_rows):
    """Everything the prompt and the grading payload need, resolved from
    the record + its parent corpus row. The design and property come from
    the RECORD, never from the model."""
    parent = next(r for r in corpus_rows if r["id"] == record["parent_id"])
    verilog = record["child_verilog"]
    wrapper_type = record.get("ext_type") in ("replicate", "compose")
    return {
        "verilog": verilog,
        "top_module": child_top(record, parent),
        "clock": parent.get("clock", "clk"),
        "antecedents": (record.get("antecedents") if wrapper_type
                        else parent.get("antecedents", [])) or [],
        "sanity_covers": (record.get("sanity_covers") if wrapper_type
                          else parent.get("sanity_covers", [])) or [],
        "property": extract_asserts(verilog),
        "invariants": record.get("invariants", []),
    }


def pre_gate(candidate, failed, properties):
    """Cheap rejections before any grading run: an unchanged list would
    re-lose the same proof, and a property copy is the forbidden repair."""
    norm = lambda xs: sorted("".join(x.split()) for x in xs)  # noqa: E731
    if norm(candidate) == norm(failed):
        return "candidate list is identical to the failed list"
    copied = property_copy(candidate, properties)
    if copied:
        return f"candidate copies a property verbatim: {copied!r}"
    return None


# --- the attempt loop ---

def run_task(record, corpus_rows, call=None, max_attempts=MAX_ATTEMPTS):
    """Up to three attempts; every attempt logged; branch dies after the
    last. `call` is injectable so the selftest can run without an API."""
    ctx = fixer_context(record, corpus_rows)
    cti, leg = failing_cti(record)
    task_id = record.get("extension_id")
    current_invs = ctx["invariants"]
    current_cti = cti or "(no trace captured — reason: " + \
        str((record.get("result") or {}).get("with_invariants", {})
            .get("reason", "unknown")) + ")"

    for attempt in range(1, max_attempts + 1):
        prompt = FIXER_PROMPT.format(
            verilog=ctx["verilog"],
            property="\n".join(ctx["property"]),
            invariants="\n".join(current_invs),
            cti=current_cti)
        log = {"task_id": task_id, "attempt": attempt,
               "fixed_at": datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
               "model": llm_client.model_label(MODEL), "cti_leg": leg}

        if call is None:
            with Spinner(f"[{task_id} #{attempt}] {MODEL} repairing invariants"):
                text, usage, stop = llm_client.call_claude(
                    model=MODEL, max_tokens=MAX_TOKENS, user=prompt,
                    schema=FIX_SCHEMA, effort=EFFORT)
            log["usage"] = usage
            if text is None:
                log["verdict"] = "REFUSED" if stop == "refusal" else "UNPARSEABLE"
                _log(log)
                continue
            out = json.loads(text)
        else:
            out = call(prompt, attempt)

        candidate = out["invariants"]
        log["diagnosis"] = out.get("diagnosis")
        log["invariants"] = candidate

        err = pre_gate(candidate, current_invs, ctx["property"])
        if err:
            log["verdict"] = "PRE_GATE"
            log["error"] = err
            _log(log)
            continue

        payload = {"verilog": ctx["verilog"], "top_module": ctx["top_module"],
                   "clock": ctx["clock"], "antecedents": ctx["antecedents"],
                   "sanity_covers": ctx["sanity_covers"],
                   "invariants": candidate}
        t0 = time.monotonic()
        with Spinner(f"[{task_id} #{attempt}] oracle grading repair"):
            result = grade_triple_generated(json.dumps(payload), **GRADE_KWARGS)
        log["grade_wall_s"] = round(time.monotonic() - t0, 2)
        log["verdict"] = result.verdict.name
        log["reason"] = result.reason
        log["result"] = asdict(result)
        _log(log)
        print(f"  attempt {attempt}: {result.verdict.name} — {result.reason[:80]}")

        if result.verdict.name == "NECESSARY":
            return log
        # feed the fresh CTI to the next attempt
        new_cti, _ = failing_cti({"result": log["result"]})
        if new_cti:
            current_cti = new_cti
        current_invs = candidate

    print(f"  branch dies: {max_attempts} attempts exhausted")
    return None


def _log(entry):
    ATTEMPT_LOG.parent.mkdir(exist_ok=True)
    with ATTEMPT_LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", help="extension_id of ONE queued task to repair")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--selftest", action="store_true",
                   help="hand repair of a deliberately broken triple "
                        "through the full grade path — no API")
    args = p.parse_args()

    corpus_rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]

    if args.selftest:
        parent = next(r for r in corpus_rows if r["id"] == "g0_014")
        broken = {"extension_id": "selftest", "ext_type": "structural",
                  "parent_id": "g0_014", "verdict": "NOT_PROVEN",
                  "child_verilog": parent["verilog"],
                  "invariants": ["tokens <= 3'd4"],   # hi-relation missing
                  "result": {"with_invariants": {
                      "runs": [{"mode": "prove", "rc": 4, "notes": [],
                                "trace_text": "tokens_hi = 1, tokens = 0"}]}}}
        fixed = parent["invariants"]                  # the known-good list
        won = run_task(broken, corpus_rows,
                       call=lambda prompt, n: {"diagnosis": "selftest",
                                               "invariants": fixed})
        print("SELFTEST " + ("PASSED: hand repair graded NECESSARY"
                             if won else "FAILED"))
        return

    queue = [json.loads(l) for l in QUEUE.read_text().splitlines()] \
        if QUEUE.exists() else []
    if not queue:
        sys.exit("fixer queue is empty — nothing to repair")
    tasks = [r for r in queue if not args.task
             or r.get("extension_id") == args.task]
    if args.dry:
        ctx = fixer_context(tasks[0], corpus_rows)
        cti, _ = failing_cti(tasks[0])
        print(FIXER_PROMPT.format(
            verilog=ctx["verilog"], property="\n".join(ctx["property"]),
            invariants="\n".join(ctx["invariants"]), cti=cti or "(none)"))
        return
    if not args.task:
        sys.exit("refusing to run the whole queue: the Fixer is dormant "
                 "until the prompt is rewritten against real failures. "
                 "Run one task explicitly with --task <extension_id>.")
    for rec in tasks:
        print(f"task {rec.get('extension_id')} ({rec.get('verdict')})")
        run_task(rec, corpus_rows)


if __name__ == "__main__":
    main()
