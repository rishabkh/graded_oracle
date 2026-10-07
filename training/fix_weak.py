"""Opus strengthens the corpus answers that are true but too weak under
the benchmark's 1-step rule; a new answer is kept only when the proof
tool proves it true and 1-step inductive.

Why: 207 of 665 corpus answers pass the corpus's 20-step proof and fail
the benchmark's 1-step one (one_step.py). Training without them (run 4a)
cut "too weak" answers but raised false ones (hard set, 7 Oct 2026: too
weak 44.6% -> 35.4%, false 30.0% -> 36.9%); examples that are strong AND
true are the next step.

104 of the 207 are left out: their property looks back a cycle ($past,
$stable; 104 of the corpus's 105 such rows fail at one step, against 103
of 554 others). The 1-step check keeps the previous value in a hidden
register that no fact can name, so no answer can pass it there: g0_041's
counterexample holds every fact and breaks the property only through a
hidden "previous peak" of 19 beside a real peak of 16, a history that
cannot happen. This fixes the other 103.

Each row: the design, its property, its current facts and the proof
tool's 1-step counterexample go to Opus, which returns a full new list.
The list is kept only if the whole proof passes at one step
(one_step.prove_true): every fact holds from reset, and the facts with
the property carry over one step. So every kept fact is proven true, not
just plausible. A rejected list goes back with the tool's answer (which
fact is false and a run from reset that breaks it, the new
counterexample, or the name or syntax the tool cannot read), up to
--max-attempts.

Every attempt is one line in extender/logs/fix_weak.jsonl; a row with an
accepted line is skipped next time, so a rerun never pays twice.

  source ~/Desktop/HarvardResearch/oss-cad-suite/environment
  venv/bin/python training/fix_weak.py --dry          # no calls
  venv/bin/python training/fix_weak.py --n 3          # spends money
  venv/bin/python training/fix_weak.py                # all 103
  venv/bin/python training/fix_weak.py --report       # no calls
"""
import argparse
import json
import shutil
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extender"))

import llm_client                                             # noqa: E402
from distractor import Spinner                                # noqa: E402
from regen_reasoning import (append, assertions, cost, now,  # noqa: E402
                             read_log, require_key, say)

CORPUS = HERE.parent / "extender" / "corpus.jsonl"
ONE_STEP = HERE.parent / "extender" / "logs" / "one_step.jsonl"
LOG = HERE.parent / "extender" / "logs" / "fix_weak.jsonl"

MODEL = "claude-opus-5"
LOOKS_BACK = ("$past", "$stable")
SCHEMA = {
    "type": "object",
    "properties": {"why": {"type": "string"},
                   "invariants": {"type": "array",
                                  "items": {"type": "string"}}},
    "required": ["why", "invariants"],
    "additionalProperties": False,
}

PROMPT = """\
You are given a small SystemVerilog design from a formal verification
dataset, the property it asserts, and a list of facts about the states it
can reach.

Every fact is true, but together they are too weak: the property plus the
facts is not 1-inductive. The proof tool found a state where the property
and every fact hold, yet one clock step later the property or a fact fails:

{counterexample}

Write a new list of facts such that:
1. every fact is true in every state the design can reach from reset;
2. the property and the facts together are 1-inductive: from ANY state
   where all of them hold, one clock step, with any inputs, leads to a
   state where all of them hold again.

The list you return replaces the current one: keep, change or drop facts
as you need. Usually the counterexample state breaks a relation between
registers that the design keeps but no fact states; add that relation.
Keep each fact as simple as it can be; do not list reachable states one
by one.

Each fact is checked in every state as
    always @(*) assert (<fact>);
placed at the end of module `{top}`. So each fact:
- is a plain SystemVerilog boolean expression over signals visible in
  module `{top}` (its ports, regs and wires; no dotted paths into
  submodules);
- uses no system functions at all ($past, $stable, $countones, $onehot,
  ...);
- never uses `->`; write `!a || b` instead;
- uses sized constants, e.g. 4'd9;
- bounds a register whose declared width admits values it can never
  reach (a counter that saturates at 8 in a 4-bit reg): induction would
  otherwise start from an impossible value.

## Design (top module `{top}`, clock `{clock}`)

```verilog
{verilog}
```

## Property (the design's own assertions)

{assertions}

## Current facts

{facts}

## Output

JSON with two fields:
- `why`: one or two sentences: what the counterexample state gets wrong,
  and which fact rules it out.
- `invariants`: the full new list, one fact per string.
"""

FEEDBACK = """
## Your previous answer was rejected

Your facts:
{facts}

{what}

Give a corrected full list.
"""

# The tool could not judge the list (it crashed, errored or ran out of
# time). That says nothing about the facts, so the row stops there
# instead of paying for a new list.
TOOL_FAILURES = {"TIMEOUT", "ERROR", "check_error"}

_lock = threading.Lock()
_work = {}


def targets(rows, one_step_log):
    """(row, counterexample) for each row whose answer passed at 20 steps
    and failed at 1, leaving out properties that look back a cycle."""
    by = {}
    for r in one_step_log:
        by.setdefault(r["id"], {})[r["dropped"]] = r
    out = []
    for row in rows:
        d = by.get(row["id"], {})
        if d.get(-1, {}).get("tier") != "INDUCTIVE" or \
                d.get(-2, {}).get("tier") != "NOT_INDUCTIVE":
            continue
        if any(f in a for a in assertions(row) for f in LOOKS_BACK):
            continue
        out.append((row, d[-2].get("trace")))
    return out


def _bullets(items):
    return "\n".join(f"- {x}" for x in items)


def build_prompt(row, counterexample):
    return PROMPT.format(
        counterexample=counterexample or "(the tool gave no trace)",
        top=row["top_module"], clock=row.get("clock", "clk"),
        verilog=row["verilog"].rstrip("\n"),
        assertions=_bullets(assertions(row)),
        facts=_bullets(row["invariants"]))


def feedback(facts, chk):
    tier, fact = chk.get("tier"), chk.get("fact")
    who = f"`{fact}`" if fact else "the property"
    if tier == "FALSE":
        what = (f"The proof tool found that {who} is false: this run from "
                f"reset breaks it.\n\n{chk.get('trace') or ''}")
    elif tier == "BAD_FACT":
        what = f"The proof tool cannot read {who}: {chk.get('detail')}."
    else:
        what = ("Still not 1-inductive: in this state the property and "
                f"every fact hold, but one step later {who} fails.\n\n"
                f"{chk.get('trace') or ''}")
    return FEEDBACK.format(facts=_bullets(facts), what=what.rstrip())


def parse_answer(text):
    """The new list of facts, or None when the reply is missing,
    malformed, or empty."""
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = llm_client.extract_json(text)
    facts = obj.get("invariants") if isinstance(obj, dict) else None
    if not isinstance(facts, list) or not facts or \
            not all(isinstance(f, str) and f.strip() for f in facts):
        return None
    return facts


def check(row, facts):
    """The whole proof at one step. Imported late so tests can replace
    this function without the toolchain."""
    from one_step import prove_true
    with _lock:
        if "dir" not in _work:
            _work["dir"] = tempfile.mkdtemp(prefix="fix_weak_")
    return prove_true(row, facts, _work["dir"])


def safe_check(row, facts):
    """A crash in the tool is a tool failure, not a verdict on the facts."""
    try:
        return check(row, facts)
    except Exception as e:
        return {"tier": "check_error", "detail": f"{type(e).__name__}: {e}"}


def preflight():
    """Why the proof tool cannot run, or None. Asked before any paid
    call, so a missing prover does not turn every row into a paid crash."""
    if shutil.which("sby") is None:
        return ("sby is not on PATH. Run first: "
                "source ~/Desktop/HarvardResearch/oss-cad-suite/environment")
    return None


def fix_row(row, counterexample, *, effort, max_tokens, max_attempts,
            first_attempt=1):
    """Ask, prove, and retry with the tool's answer until one list is
    proven or the attempts run out. Every attempt is logged."""
    base = build_prompt(row, counterexample)
    prompt = base
    spent = {"input": 0, "output": 0}
    verdict, accepted, n = None, False, first_attempt - 1
    for n in range(first_attempt, first_attempt + max_attempts):
        rec = {"id": row["id"], "attempt": n,
               "model": llm_client.model_label(MODEL), "effort": effort,
               "original": row["invariants"]}
        try:
            text, usage, stop = llm_client.call_claude(
                model=MODEL, max_tokens=max_tokens, user=prompt,
                schema=SCHEMA, effort=effort)
        except Exception as e:
            # an API error (overload, no credit) repeats on retry; stop
            # the row and let a rerun pick it up
            append(LOG, dict(rec, usage={"input": 0, "output": 0},
                             stop="error", error=f"{type(e).__name__}: {e}",
                             invariants=None, check=None, accepted=False,
                             timestamp=now()))
            return {"id": row["id"], "attempts": n - first_attempt + 1,
                    "accepted": False, "verdict": "error", "usage": spent}
        spent["input"] += usage.get("input", 0)
        spent["output"] += usage.get("output", 0)
        facts = parse_answer(text)
        chk = safe_check(row, facts) if facts else None
        accepted = bool(chk and chk.get("tier") == "PROVEN")
        why = None
        if facts:
            try:
                why = json.loads(text).get("why")
            except (json.JSONDecodeError, AttributeError):
                pass
        append(LOG, dict(rec, usage=usage, stop=stop, why=why,
                         invariants=facts, check=chk, accepted=accepted,
                         timestamp=now()))
        verdict = chk.get("tier") if chk else (
            stop if stop and stop != "ok" else "unparseable")
        if accepted or verdict in TOOL_FAILURES:
            break
        if chk:
            prompt = base + feedback(facts, chk)
    return {"id": row["id"], "attempts": n - first_attempt + 1,
            "accepted": accepted, "verdict": verdict, "usage": spent}


def report(log_path, ids):
    """Counts and cost from the log alone; makes no calls."""
    lines = read_log(log_path)
    wanted = set(ids)
    verdicts = Counter((r.get("check") or {}).get("tier")
                       or r.get("stop") or "unparseable" for r in lines)
    first, accepted = {}, {}
    for r in lines:
        first[r["id"]] = min(first.get(r["id"], r["attempt"]), r["attempt"])
        if r.get("accepted") and r["id"] in wanted:
            accepted.setdefault(r["id"], r)
    first_try = sum(1 for i, r in accepted.items() if r["attempt"] == first[i])
    tokens_in = sum((r.get("usage") or {}).get("input", 0) for r in lines)
    tokens_out = sum((r.get("usage") or {}).get("output", 0) for r in lines)
    dollars = cost(tokens_in, tokens_out)
    print("attempts by verdict: " + (", ".join(
        f"{v} {c}" for v, c in verdicts.most_common()) or "none"))
    print(f"rows: {first_try} fixed first try, {len(accepted) - first_try} "
          f"after retries, {len(wanted) - len(accepted)} not fixed "
          f"(of {len(wanted)})")
    if accepted:
        before = sum(len(r["original"]) for r in accepted.values())
        after = sum(len(r["invariants"]) for r in accepted.values())
        print(f"facts per fixed answer: {before / len(accepted):.1f} before, "
              f"{after / len(accepted):.1f} after")
    per = (f"${dollars / len(accepted):.3f} per fixed row" if accepted
           else "no fixed rows")
    print(f"tokens: {tokens_in:,} in, {tokens_out:,} out = ${dollars:.2f} "
          f"({per})")
    return {"fixed": len(accepted), "first_try": first_try, "cost": dollars}


def progress_label(done, total, active, dollars):
    return (f"fixing too-weak answers: {done}/{total} rows done, "
            f"{active} in progress, ${dollars:.2f} so far")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Strengthen the corpus answers that fail at one step.")
    p.add_argument("--n", type=int, help="first N rows to fix")
    p.add_argument("--ids", help="comma-separated corpus ids")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--effort", default="high")
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--dry", action="store_true",
                   help="print the first question and the row count; no calls")
    p.add_argument("--report", action="store_true",
                   help="counts and cost from the log; no calls")
    args = p.parse_args(argv)

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()
            if l.strip()]
    picked = targets(rows, read_log(ONE_STEP))
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        unknown = wanted - {r["id"] for r, _ in picked}
        if unknown:
            sys.exit(f"not among the {len(picked)} rows to fix: "
                     f"{', '.join(sorted(unknown))}")
        picked = [(r, c) for r, c in picked if r["id"] in wanted]
    picked = picked[:args.n] if args.n is not None else picked
    if args.report:
        report(LOG, [r["id"] for r, _ in picked])
        return

    log = read_log(LOG)
    accepted = {r["id"] for r in log if r.get("accepted")}
    todo = [(r, c) for r, c in picked if r["id"] not in accepted]
    if args.dry:
        if picked:
            print(build_prompt(*picked[0]))
        print(f"\n{len(todo)} rows would be called "
              f"({len(picked) - len(todo)} already fixed, skipped)")
        return

    require_key()
    problem = preflight()
    if problem:
        sys.exit(problem)

    last = Counter()
    for r in log:
        last[r["id"]] = max(last[r["id"]], r.get("attempt", 0))
    running = {"done": 0, "active": 0, "input": 0, "output": 0, "fixed": 0}
    spinner = Spinner(progress_label(0, len(todo), 0, 0.0), always=True)

    def refresh():
        spinner.label = progress_label(
            running["done"], len(todo), running["active"],
            cost(running["input"], running["output"]))

    def one(item):
        row, cex = item
        with _lock:
            running["active"] += 1
            refresh()
        out = fix_row(row, cex, effort=args.effort,
                      max_tokens=args.max_tokens,
                      max_attempts=args.max_attempts,
                      first_attempt=last[row["id"]] + 1)
        with _lock:
            running["active"] -= 1
            running["done"] += 1
            running["fixed"] += out["accepted"]
            running["input"] += out["usage"]["input"]
            running["output"] += out["usage"]["output"]
            say(f"[{running['done']}/{len(todo)}] {row['id']}  "
                f"{out['attempts']} attempt(s)  {out['verdict']}  "
                f"fixed so far {running['fixed']}  "
                f"${cost(running['input'], running['output']):.2f}")
            refresh()
        return out

    print(f"{len(todo)} rows to call ({len(picked) - len(todo)} already "
          f"fixed, skipped)")
    with spinner:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
    report(LOG, [r["id"] for r, _ in picked])


if __name__ == "__main__":
    main()
