"""Fresh "fake state" reasoning for every corpus row, kept only when the
checker proves the fake state real.

Why: the next training run differs from v2 in one thing only, the
reasoning in front of each answer. Today only the 228 generation-0 rows
have it (the generator's own note), the extensions have none of this
kind, and nobody has checked that the 228 notes are true. Mixing two
sources, half unchecked, would blur that one difference. So every row
gets new reasoning from one prompt, and every fake state goes through
training/cti_check.py: all assertions hold there, the invariant list R
is false there, and one clock step breaks an assertion. A rejected state
is sent back with the checker's verdict, up to --max-attempts; rows that
never pass are dropped.

The field notes are copied from initiator/prompts.py, so the new
reasoning reads the way the generator was asked to write it. --originals
runs the same checker over the 228 original notes, so the two sources
can be compared on how often each one is real.

Every attempt is one line in extender/logs/reasoning_regen.jsonl. A row
with an accepted line is skipped next time, so a rerun never pays twice.

  source ~/Desktop/HarvardResearch/oss-cad-suite/environment
  venv/bin/python training/regen_reasoning.py --dry --n 5     # no calls
  venv/bin/python training/regen_reasoning.py --n 5           # spends money
  venv/bin/python training/regen_reasoning.py --report        # no calls
  venv/bin/python training/regen_reasoning.py --originals     # no calls
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extender"))

import llm_client                                             # noqa: E402
from distractor import Spinner                                # noqa: E402
from schema import TRIPLE_SCHEMA                              # noqa: E402

CORPUS = HERE.parent / "extender" / "corpus.jsonl"
ATTEMPTS = HERE.parent / "initiator" / "logs" / "attempts.jsonl"
LOG = HERE.parent / "extender" / "logs" / "reasoning_regen.jsonl"
ORIG_LOG = HERE.parent / "extender" / "logs" / "reasoning_originals_check.jsonl"

MODEL = "claude-opus-5-5"
# log lines written before the model was recorded came from Opus 5
OLD_MODEL = "claude-opus-5"

# the generator's own two reasoning fields, same definitions, nothing else
FIELDS = ("cti_reasoning", "cti_state")
SCHEMA = {
    "type": "object",
    "properties": {k: TRIPLE_SCHEMA["properties"][k] for k in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}

PROMPT = """\
You are given a small SystemVerilog design from a formal verification dataset.

P is the design's assertions, all of them together. P is true of the design but
CANNOT be proven by k-induction on its own. R is the strengthening invariant: the
conjunction of the clauses listed below. P and R together are inductive.

P is NOT inductive on its own. Concretely, there exists a state S such that:
      - S satisfies P  (every assertion holds)
      - S violates R  (at least one clause of R is false, so S is unreachable in
        real operation, but a solver performing induction may start there)
      - S transitions in ONE clock step, with some input, to a state that
        VIOLATES P  (some assertion fails)

    S must also be a state the design can IDLE IN — one it can remain in for
    arbitrarily many cycles by holding the enable low. This matters: a one-step
    CTI only shows P is not 1-inductive, but the prover uses a 20-state window.
    If the design can stall in S, the solver builds a window of 20 copies of S and
    then takes the violating transition, so the property fails induction at ANY
    depth rather than only at depth 1. Without a stall, P may survive 20-induction
    and the triple is rejected.

Your task is to name that state S.

## Design (top module `{top}`, clock `{clock}`)

```
{verilog}
```

## P: the design's assertions

Each is checked as written in the source above, under its guard if it has one.

{assertions}

## R: the strengthening invariants (R is their conjunction)

{invariants}

## Output

Your output is emitted through the API's structured-output mode, with exactly two
fields.

Field notes:

{field_notes}

Rules for `cti_state`:
- Top-level signal names only: signals declared in module `{top}`, not names
  inside submodule instances.
- Every value is a sized constant, e.g. 4'd3 or 1'b0.
- Give each signal whole, never a bit slice such as `c[1]`; a memory word is
  named like `mem[1]`.
- Every signal needed to evaluate the assertions and R must appear, including
  any register that guards an assertion.
- Two moments, kept apart. S is where the design sits: it can stay in S with
  its enable held low (the idle condition above). The step that breaks P is
  the next clock step, taken with the enable ON. If you list an input in
  `cti_state`, give its value on that breaking step, never at its idle value
  (that step would change nothing). You may leave inputs out; every input
  value is then tried.
- Do not modify the design. S is a state of the design exactly as written.
"""

# Copied from initiator/prompts.py. Changed only where this task differs:
# "the state S from condition (2)" became "the state S above", and R is
# given here rather than the model's own, so the self-check says "the
# given R" and "fix the state".
FIELD_NOTES = """\
- `cti_reasoning`: one or two sentences: why this state satisfies P, why it
  violates R, and what its successor does that breaks P.

- `cti_state`: the state S above, as a list of {signal, value}
  pairs with concrete values. Every signal needed to evaluate both P and R must
  appear. Vague answers are rejected: "pointers disagree" is not acceptable;
  [{"signal": "count", "value": "3'd1"}, {"signal": "wptr", "value": "2'd2"},
  {"signal": "rptr", "value": "2'd0"}] is.

  Self-check before you answer: evaluate R at cti_state. It MUST come out false.
  If your named state does not violate the given R, you have contradicted
  yourself — fix the state."""

FEEDBACK = """
## Your previous answer was rejected

Previous fake state:
{state}

Checker verdict: {verdict}
{detail}

Give a corrected fake state and reasoning.
"""

# The checker could not judge the state (it crashed, errored or ran out
# of time). That says nothing about the state, so the row stops there
# instead of paying for a new one.
TOOL_FAILURES = {"check_error", "ERROR", "TIMEOUT"}

_log_lock = threading.Lock()
_ASSERT = re.compile(r"\bassert\s*(?:property\s*)?\(")


def check(row, state):
    """The fake-state checker. Imported late so tests can replace this
    function without training/cti_check.py existing."""
    from cti_check import check_state
    return check_state(row, state)


def preflight():
    """Why the checker cannot run, or None. Asked before any paid call,
    so a missing prover does not turn every row into a paid crash."""
    if shutil.which("sby") is None:
        return ("sby is not on PATH. Run first: "
                "source ~/Desktop/HarvardResearch/oss-cad-suite/environment")
    try:
        import cti_check                                      # noqa: F401
    except ImportError as e:
        return f"training/cti_check.py cannot be imported ({e})"
    return None


def assertions(row):
    """Every assertion expression in the source. The corpus `property`
    field misses the `assert property (...)` form (empty on 35 rows), so
    the source is scanned, with the field as the fallback."""
    src = re.sub(r"/\*.*?\*/", " ", row["verilog"], flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    found = []
    for m in _ASSERT.finditer(src):
        depth, i = 1, m.end()
        while i < len(src) and depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        found.append(" ".join(src[m.end():i - 1].split()))
    return found or list(row.get("property") or [])


def build_prompt(row):
    return PROMPT.format(
        top=row["top_module"], clock=row.get("clock", "clk"),
        verilog=row["verilog"].rstrip("\n"),
        assertions="\n".join(f"- {a}" for a in assertions(row)),
        invariants="\n".join(f"- {c}" for c in row["invariants"]),
        field_notes=FIELD_NOTES)


def feedback(state, chk):
    pairs = "\n".join(f"  {p['signal']} = {p['value']}" for p in state)
    return FEEDBACK.format(state=pairs, verdict=chk.get("verdict"),
                           detail=chk.get("detail") or "")


def parse_answer(text):
    """The two fields, or None when the reply is missing, malformed, or
    names no state at all."""
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = llm_client.extract_json(text)
    if not isinstance(obj, dict):
        return None
    reasoning, state = obj.get("cti_reasoning"), obj.get("cti_state")
    if not isinstance(reasoning, str) or not isinstance(state, list) \
            or not state:
        return None
    for p in state:
        if not (isinstance(p, dict) and isinstance(p.get("signal"), str)
                and isinstance(p.get("value"), str)):
            return None
    return {"cti_reasoning": reasoning, "cti_state": state}


def safe_check(row, state):
    """A crash in the checker is a tool failure, not a verdict on the
    state; it is recorded as one so it is never mistaken for a rejection."""
    try:
        return check(row, state)
    except Exception as e:
        return {"ok": False, "verdict": "check_error",
                "detail": f"{type(e).__name__}: {e}"}


def append(path, record):
    with _log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as f:
            # a killed run can leave a cut line with no newline; start a
            # fresh line so this record is not glued onto it and lost
            if f.seek(0, 2):
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write((json.dumps(record, default=str) + "\n").encode())


def read_log(path):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a line cut short by a killed run
    return out


def now():
    return datetime.now(timezone.utc).isoformat()


def verdict_of(rec):
    if rec.get("check"):
        return rec["check"].get("verdict")
    stop = rec.get("stop")
    return stop if stop and stop != "ok" else "unparseable"


def cost(tokens_in, tokens_out, model=None):
    """At the list price of `model` (default: the model this script calls)."""
    return llm_client.dollars(model or MODEL, tokens_in, tokens_out)


def line_cost(rec):
    """One log line at the price of the model it records, so an old Opus 5
    log is not re-priced as Opus 5.5, nor the reverse."""
    u = rec.get("usage") or {}
    return cost(u.get("input", 0), u.get("output", 0),
                rec.get("model") or OLD_MODEL)


def progress_label(done, total, active, dollars):
    return (f"regenerating reasoning: {done}/{total} rows done, "
            f"{active} in progress, ${dollars:.2f} so far")


def say(line):
    """A progress line, printed clear of the spinner: wipe the spinner's
    line first, and the spinner redraws below on its next frame."""
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
    print(line, flush=True)


def regen_row(row, *, effort, max_tokens, max_attempts, first_attempt=1):
    """Ask, check, and retry with the checker's verdict until one state
    passes or the attempts run out. Every attempt is logged."""
    base = build_prompt(row)
    prompt = base
    spent = {"input": 0, "output": 0}
    verdict, accepted, n = None, False, first_attempt - 1
    for n in range(first_attempt, first_attempt + max_attempts):
        rec = {"id": row["id"], "attempt": n, "prompt_kind": "regen",
               "model": llm_client.model_label(MODEL), "effort": effort}
        try:
            text, usage, stop = llm_client.call_claude(
                model=MODEL, max_tokens=max_tokens, user=prompt,
                schema=SCHEMA, effort=effort)
        except Exception as e:
            # an API error (overload, no credit) repeats on retry; stop
            # the row and let a rerun pick it up
            rec.update(usage={"input": 0, "output": 0}, stop="error",
                       error=f"{type(e).__name__}: {e}", cti_reasoning=None,
                       cti_state=None, check=None, accepted=False,
                       timestamp=now())
            append(LOG, rec)
            return {"id": row["id"], "attempts": n - first_attempt + 1,
                    "accepted": False, "verdict": "error", "usage": spent}
        spent["input"] += usage.get("input", 0)
        spent["output"] += usage.get("output", 0)
        answer = parse_answer(text)
        chk = safe_check(row, answer["cti_state"]) if answer else None
        accepted = bool(chk and chk.get("ok"))
        rec.update(usage=usage, stop=stop,
                   cti_reasoning=answer and answer["cti_reasoning"],
                   cti_state=answer and answer["cti_state"],
                   check=chk, accepted=accepted, timestamp=now())
        append(LOG, rec)
        verdict = verdict_of(rec)
        if accepted or verdict in TOOL_FAILURES:
            break
        if chk:
            prompt = base + feedback(answer["cti_state"], chk)
    return {"id": row["id"], "attempts": n - first_attempt + 1,
            "accepted": accepted, "verdict": verdict, "usage": spent}


def report(log_path, corpus_ids):
    """Counts and cost from the log alone; makes no calls."""
    lines = read_log(log_path)
    wanted = set(corpus_ids)
    verdicts = Counter(verdict_of(r) for r in lines)
    first, accepted = {}, {}
    for r in lines:
        first[r["id"]] = min(first.get(r["id"], r["attempt"]), r["attempt"])
        if r.get("accepted") and r["id"] in wanted:
            accepted[r["id"]] = min(accepted.get(r["id"], r["attempt"]),
                                    r["attempt"])
    first_try = sum(1 for i, a in accepted.items() if a == first[i])
    tokens_in = sum((r.get("usage") or {}).get("input", 0) for r in lines)
    tokens_out = sum((r.get("usage") or {}).get("output", 0) for r in lines)
    dollars = sum(line_cost(r) for r in lines)
    rep = {"verdicts": dict(verdicts), "first_try": first_try,
           "after_retries": len(accepted) - first_try,
           "missing": len(wanted) - len(accepted), "rows": len(wanted),
           "input": tokens_in, "output": tokens_out, "cost": dollars,
           "cost_per_accepted": dollars / len(accepted) if accepted else None}
    print("attempts by verdict: " + (", ".join(
        f"{v} {c}" for v, c in verdicts.most_common()) or "none"))
    print(f"rows: {rep['first_try']} accepted first try, "
          f"{rep['after_retries']} after retries, {rep['missing']} still "
          f"missing (of {rep['rows']})")
    per = (f"${rep['cost_per_accepted']:.3f} per accepted row"
           if accepted else "no accepted rows")
    print(f"tokens: {tokens_in:,} in, {tokens_out:,} out = ${dollars:.2f} "
          f"({per})")
    return rep


def load_originals(rows, attempts_path):
    """(row, cti_reasoning, cti_state) for each generation-0 row, read
    from the generator attempt that produced it: (source_run_id,
    source_attempt) == (run_id, attempt) in attempts.jsonl."""
    keys = {(r.get("source_run_id"), r.get("source_attempt"))
            for r in rows if r.get("generation") == 0}
    raw = {}
    with open(attempts_path) as f:
        for line in f:
            a = json.loads(line)
            key = (a.get("run_id"), a.get("attempt"))
            if key in keys and a.get("raw_json"):
                raw[key] = a["raw_json"]
    out = []
    for r in rows:
        key = (r.get("source_run_id"), r.get("source_attempt"))
        if r.get("generation") != 0 or key not in raw:
            continue
        try:
            obj = json.loads(raw[key])
        except json.JSONDecodeError:
            continue
        out.append((r, obj.get("cti_reasoning"), obj.get("cti_state")))
    return out


def run_originals(rows, workers):
    found = load_originals(rows, ATTEMPTS)
    gen0 = sum(1 for r in rows if r.get("generation") == 0)
    print(f"{len(found)} of {gen0} generation-0 rows linked to their "
          f"original reasoning")
    done = {"n": 0}

    def one(item):
        row, reasoning, state = item
        chk = safe_check(row, state)
        append(ORIG_LOG, {"id": row["id"], "prompt_kind": "original",
                          "source_run_id": row.get("source_run_id"),
                          "source_attempt": row.get("source_attempt"),
                          "cti_reasoning": reasoning, "cti_state": state,
                          "check": chk, "accepted": bool(chk.get("ok")),
                          "timestamp": now()})
        with _log_lock:
            done["n"] += 1
            print(f"[{done['n']}/{len(found)}] {row['id']}  "
                  f"{chk.get('verdict')}", flush=True)
        return chk.get("verdict"), bool(chk.get("ok"))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, found))
    counts = Counter(v for v, _ in results)
    real = sum(1 for _, ok in results if ok)
    print("original reasoning by verdict: " + (", ".join(
        f"{v} {c}" for v, c in counts.most_common()) or "none"))
    print(f"real: {real} of {len(results)}")
    return counts


def require_key():
    need = ("OPENROUTER_API_KEY" if llm_client.provider() == "openrouter"
            else "ANTHROPIC_API_KEY")
    if not os.environ.get(need):
        print(f"{need} is not set, so no calls were made. Set it, or use "
              f"--dry / --report / --originals.", file=sys.stderr)
        sys.exit(2)


def select(rows, n=None, ids=None):
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        unknown = set(wanted) - {r["id"] for r in rows}
        if unknown:
            print(f"unknown ids: {', '.join(sorted(unknown))}",
                  file=sys.stderr)
            sys.exit(2)
        rows = [r for r in rows if r["id"] in set(wanted)]
    return rows[:n] if n is not None else rows


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Regenerate and check the fake-state reasoning.")
    p.add_argument("--n", type=int, help="first N corpus rows")
    p.add_argument("--ids", help="comma-separated corpus ids")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--effort", default="medium")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--dry", action="store_true",
                   help="print the first prompt and the row count; no calls")
    p.add_argument("--report", action="store_true",
                   help="counts and cost from the log; no calls")
    p.add_argument("--originals", action="store_true",
                   help="check the 228 original notes; no calls")
    args = p.parse_args(argv)

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()
            if l.strip()]
    if args.report:
        report(LOG, [r["id"] for r in rows])
        return
    if args.originals:
        problem = preflight()
        if problem:
            print(problem, file=sys.stderr)
            sys.exit(2)
        run_originals(rows, args.workers)
        return

    selected = select(rows, args.n, args.ids)
    log = read_log(LOG)
    accepted = {r["id"] for r in log if r.get("accepted")}
    todo = [r for r in selected if r["id"] not in accepted]
    if args.dry:
        if selected:
            print(build_prompt(selected[0]))
        print(f"\n{len(todo)} rows would be called "
              f"({len(selected) - len(todo)} already accepted, skipped)")
        return

    require_key()
    problem = preflight()
    if problem:
        print(problem, file=sys.stderr)
        sys.exit(2)

    last = Counter()
    for r in log:
        last[r["id"]] = max(last[r["id"]], r.get("attempt", 0))
    running = {"done": 0, "active": 0, "input": 0, "output": 0}
    # one spinner for the whole run: the workers share it
    spinner = Spinner(progress_label(0, len(todo), 0, 0.0), always=True)

    def refresh():
        spinner.label = progress_label(
            running["done"], len(todo), running["active"],
            cost(running["input"], running["output"]))

    def one(row):
        with _log_lock:
            running["active"] += 1
            refresh()
        out = regen_row(row, effort=args.effort, max_tokens=args.max_tokens,
                        max_attempts=args.max_attempts,
                        first_attempt=last[row["id"]] + 1)
        with _log_lock:
            running["active"] -= 1
            running["done"] += 1
            running["input"] += out["usage"]["input"]
            running["output"] += out["usage"]["output"]
            dollars = cost(running["input"], running["output"])
            say(f"[{running['done']}/{len(todo)}] {row['id']}  "
                f"{out['attempts']} attempt(s)  {out['verdict']}  "
                f"${dollars:.2f}")
            refresh()
        return out

    print(f"{len(todo)} rows to call ({len(selected) - len(todo)} already "
          f"accepted, skipped)")
    with spinner:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
    report(LOG, [r["id"] for r in selected])


if __name__ == "__main__":
    main()
