"""The Fixer — the third worker class. Drains the fixer queue that
promote.py and batch.py feed with NOT_PROVEN / NOT_INDUCTIVE records:
proof failures where the design and property are sound but the invariant
list is too weak. The prompt was written against the first real failures
in the log (two induction CTIs with a PDR second opinion proving the
property true — pure strengthening problems).

Contract:
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
- A repaired list with a fact the benchmark's checker cannot read (a
  system function, the compact forall form, a dotted name) is turned away
  before it is checked, as FACT_FORMAT (22 Oct 2026), and so is one with
  a fact that reads an asynchronous reset, as READS_RESET (23 Oct 2026).
- After a list is turned away before any check, the next attempt's
  question ends with why (23 Oct 2026); its log line keeps that reason as
  retry_lesson.
- Each attempt's checks are kept in logs/proofs/<task>_fix_<parent>_<id>/
  fix_a<N>, and its log line names that folder as proof_dir, relative to
  the log's folder (22 Oct 2026).
- A winning repair is promoted into the corpus immediately, at the
  child's generation — a fixed branch rejoins the ladder.

  venv/bin/python extender/fix.py --dry            # print prompt for first pending task
  venv/bin/python extender/fix.py --selftest       # hand repair through the grade path (no API)
  venv/bin/python extender/fix.py --task <ext_id>  # run ONE task (up to 3 attempts)
  venv/bin/python extender/fix.py                  # drain: every pending task, skip attempted
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
from extend import out_of_scope, property_copy           # noqa: E402
from distractor import MODEL, EFFORT, GRADE_KWARGS, Spinner  # noqa: E402
from promote import child_top, promote                   # noqa: E402

from fact_format import unreadable_facts                 # noqa: E402
import llm_client                                        # noqa: E402
from oracle import grade_triple_generated                # noqa: E402
from reset_reads import step_reset_error                 # noqa: E402

CORPUS = HERE / "corpus.jsonl"
QUEUE = HERE / "logs" / "fixer_queue.jsonl"
ATTEMPT_LOG = HERE / "logs" / "fixer_attempts.jsonl"
MAX_ATTEMPTS = 3
MAX_TOKENS = 64000       # was 16000: one answer already stopped there (8 Oct 2026)

FIXER_PROMPT = """\
You are repairing the strengthening invariants of a formally verified
SystemVerilog module. k-induction FAILED to close with the invariant list
below. The design and the property are FIXED — you may not change either.
Only the invariant list is yours.

The module:

{verilog}

The property (assertion in the source), which stays exactly as it is:
{property}

The invariant list that FAILED to close the proof:
{invariants}
{pdr}
The counterexample to induction. The prover found a START STATE that
satisfies every invariant above, yet drives the property to a violation
within k steps. That start state is unreachable in the real design; your
job is to add the fact that excludes it:

{cti}

Work it out in order:

1. diagnosis: name the signal values in the CTI start state that no
   reachable state could hold together — a counter above its real bound,
   a sum of parts exceeding the whole, a flag inconsistent with the value
   it mirrors. That missing relationship is the missing invariant.

2. Check each EXISTING clause against the CTI: keep every clause that is
   true of the design. Most repairs ADD the missing fact; rewriting old
   clauses weaker is almost always wrong.

3. Then the COMPLETE corrected invariant list (old clauses plus the fix).

Rules for the list: the whole conjunction must be inductive; never copy
the property text into the list; each clause is a plain synthesizable
Verilog boolean expression over this module's own signals — no
hierarchical refs (a.b), no prose, no |->, write !a || b for implication;
match widths when adding or comparing — zero-extend narrow terms with
{{N'b0, x}} exactly as the existing clauses do. Each clause is one
same-cycle expression: it reads the signals as they are in one cycle,
with no system function ($past, $stable, $rose, $fell, $changed,
$countones, $onehot, $onehot0, $isunknown, $signed, $clog2). The checker
that scores the facts cannot read them, and a list that uses one is
turned away before it is checked. A relation across cycles is written
between the registers that carry the value from one cycle to the next;
a count of set bits is written as a sum (v[0] + v[1] + v[2]). No clause
reads the reset, directly or through a wire that copies it: checkers
disagree about what an asynchronous reset does within a cycle, and a list
that reads one is turned away before it is checked.

Reply with JSON: {{"diagnosis": "...", "invariants": ["..."]}}"""

PDR_TRUE_LINE = """
A PDR engine has separately proven this property TRUE, unbounded. The
design and property are correct; the ONLY defect is that the invariant
list is too weak to close k-induction. A repair exists.
"""

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
            if run.get("mode") == "prove" and run.get("trace_text"):
                return run["trace_text"], leg
    return None, None


def pdr_says_true(record):
    """True when the with-leg evidence carries the PDR second opinion
    that the property holds unbounded — both real failures so far did.
    The prompt then tells the model a repair is known to exist."""
    runs = ((record.get("result") or {}).get("with_invariants")
            or {}).get("runs") or []
    return any("pdr_second_opinion" in str(n) and "proven true" in str(n)
               for r in runs for n in (r.get("notes") or []))


def repaired_record(record, invariants, result):
    """The winning attempt folded back into an extension-shaped record so
    promote() accepts it exactly like a first-try NECESSARY."""
    fixed = dict(record)
    fixed["verdict"] = "NECESSARY"
    fixed["invariants"] = invariants
    fixed["result"] = result
    fixed["extension_id"] = f"{record.get('extension_id')}_fix"
    fixed["fixed_from"] = record.get("verdict")
    return fixed


def pending_tasks(queue, attempted):
    """Queue entries not yet attempted, first occurrence per id (the
    queue file is append-only and may hold duplicates). A record the
    fixer cannot act on (a tool error, or a check that ran out of time)
    is skipped here too, not only in promote.py (8 Oct 2026 review)."""
    from promote import fixable
    seen, out = set(), []
    for rec in queue:
        eid = rec.get("extension_id")
        if eid in attempted or eid in seen or not fixable(rec):
            continue
        seen.add(eid)
        out.append(rec)
    return out


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


# 22 Oct 2026, review findings F9 and F11: a repaired list is promoted
# into the corpus the training file is built from, and the trained model
# is scored by EBMC 6.0, which cannot read a fact with a system function,
# the compact forall form or a dotted name (fact_format.py). Such a list is
# turned away before it is checked, as FACT_FORMAT. Before, a fact with
# $past was turned away only by accident, as an "out-of-scope identifier
# past", and one with $signed went on to be checked and could be promoted.
def fact_format_error(candidate):
    """Why the benchmark's checker could not read some of the facts, or
    None when it can read them all."""
    bad = unreadable_facts(candidate)
    if not bad:
        return None
    return ("facts the benchmark's checker cannot read: "
            + "; ".join(f"{fact} ({', '.join(problems)})"
                        for fact, problems in bad)
            + ". Each fact is one same-cycle expression over this module's "
            "own signals, with no system function: write a count of set "
            "bits as a sum (v[0] + v[1] + v[2]).")


# 23 Oct 2026, review finding F4: a repaired fact that reads an
# asynchronous reset can make our checker and EBMC disagree, as a step's
# can (reset_reads.py). The design and its assertions are the fixer's to
# keep, so only the facts are judged: the design is passed as its own
# parent. Such a list is turned away before it is checked, as READS_RESET.
def reset_error(candidate, ctx):
    """Why some of the facts read an asynchronous reset of the design, or
    None when none does."""
    return step_reset_error(ctx["verilog"], ctx["clock"], candidate, [],
                            [(ctx["verilog"], ctx["clock"], [], [])])


# 23 Oct 2026, review finding: after a list was turned away before any
# check (FACT_FORMAT, READS_RESET, PRE_GATE), the next attempt got exactly
# the same question, so it could repeat the mistake (a probe's attempts 1
# and 2 were both FACT_FORMAT, with identical questions). The next question
# now ends with why, in the batch runner's words (batch.retry_section, its
# review finding B1). A checked attempt needs none: the next question
# carries its list and its new counterexample.
def rejection_lesson(error):
    """The reason a list was turned away, as one paragraph for the next
    question: spaces made single, cut at batch.LESSON_MAX characters, and
    ending in a full stop."""
    from batch import LESSON_MAX
    text = " ".join(str(error).split())
    if len(text) > LESSON_MAX:
        text = text[:LESSON_MAX].rstrip() + " ..."
    return text if text.endswith((".", "...", "!", "?")) else text + "."


# 22 Oct 2026, review finding B2: the fixer never set a proof folder, so
# its checks went to the shared runs/ folder, while its attempt log copied
# the record's proof_dir, which since 21 Oct points into the batch run's
# own folder. Its checks now go in proofs/ next to its own log, the batch
# runner's way (<log-dir>/proofs), and each attempt names its own folder.
def proof_folder(record):
    """A new folder for one task's repair checks: (its path as the
    attempt log records it, relative to the folder of ATTEMPT_LOG, and its
    full path). Each attempt's checks go in a fix_a<N> folder inside."""
    import uuid
    rel = (f"proofs/{record.get('extension_id', 'x')}_fix_"
           f"{record.get('parent_id', '')}_{uuid.uuid4().hex[:6]}")
    return rel, (ATTEMPT_LOG.parent / rel).resolve()


# --- the attempt loop ---

def run_task(record, corpus_rows, call=None, max_attempts=MAX_ATTEMPTS):
    """Up to three attempts; every attempt logged; branch dies after the
    last. `call` is injectable so the selftest can run without an API."""
    ctx = fixer_context(record, corpus_rows)
    cti, leg = failing_cti(record)
    task_id = record.get("extension_id")
    proof_dir, proof_root = proof_folder(record)
    current_invs = ctx["invariants"]
    current_cti = cti or "(no trace captured - reason: " + \
        str((record.get("result") or {}).get("with_invariants", {})
            .get("reason", "unknown")) + ")"
    from batch import retry_section
    lesson = None             # why the last list was turned away, if it was

    for attempt in range(1, max_attempts + 1):
        prompt = FIXER_PROMPT.format(
            verilog=ctx["verilog"],
            property="\n".join(ctx["property"]),
            invariants="\n".join(current_invs),
            pdr=PDR_TRUE_LINE if pdr_says_true(record) else "",
            cti=current_cti) + retry_section(lesson)
        log = {"task_id": task_id, "attempt": attempt,
               "fixed_at": datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
               "model": llm_client.model_label(MODEL), "cti_leg": leg}
        if lesson:
            log["retry_lesson"] = lesson

        if call is None:
            with Spinner(f"repair try {attempt}/3: model writing"):
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

        # before the scope check, which would name `past` in `$past(x)`
        # as an unknown signal
        err = fact_format_error(candidate)
        if err:
            log["verdict"] = "FACT_FORMAT"
            log["error"] = err
            _log(log)
            lesson = rejection_lesson(err)
            continue

        err = reset_error(candidate, ctx)
        if err:
            log["verdict"] = "READS_RESET"
            log["error"] = err
            _log(log)
            lesson = rejection_lesson(err)
            continue

        bad = out_of_scope(candidate, ctx["verilog"], ctx["top_module"])
        err = (f"out-of-scope identifiers {sorted(bad)}: sub-module "
               "namespace - grading would score a fake FALSE") if bad \
            else pre_gate(candidate, current_invs, ctx["property"])
        if err:
            log["verdict"] = "PRE_GATE"
            log["error"] = err
            _log(log)
            lesson = rejection_lesson(err)
            continue
        lesson = None             # checked: the next question shows why

        payload = {"verilog": ctx["verilog"], "top_module": ctx["top_module"],
                   "clock": ctx["clock"], "antecedents": ctx["antecedents"],
                   "sanity_covers": ctx["sanity_covers"],
                   "invariants": candidate}
        t0 = time.monotonic()
        with Spinner(f"repair try {attempt}/3: oracle checking"):
            result = grade_triple_generated(
                json.dumps(payload), **dict(
                    GRADE_KWARGS,
                    workdir_root=proof_root / f"fix_a{attempt}"))
        log["grade_wall_s"] = round(time.monotonic() - t0, 2)
        log["verdict"] = result.verdict.name
        log["reason"] = result.reason
        log["result"] = asdict(result)
        # this attempt's own checks, relative to the attempt log's folder
        log["proof_dir"] = f"{proof_dir}/fix_a{attempt}"
        _log(log)
        print(f"  attempt {attempt}: {result.verdict.name} - {result.reason[:80]}")

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
        sys.exit("fixer queue is empty - nothing to repair")
    attempted = set()
    if ATTEMPT_LOG.exists():
        attempted = {json.loads(l).get("task_id")
                     for l in ATTEMPT_LOG.read_text().splitlines()}
    if args.task:
        tasks = [r for r in queue if r.get("extension_id") == args.task]
        if not tasks:
            sys.exit(f"{args.task!r} is not in the queue")
        tasks = tasks[:1]
    else:
        tasks = pending_tasks(queue, attempted)
        if not tasks:
            sys.exit("every queued task already has attempts logged - "
                     "rerun one explicitly with --task <extension_id>")

    if args.dry:
        ctx = fixer_context(tasks[0], corpus_rows)
        cti, _ = failing_cti(tasks[0])
        print(FIXER_PROMPT.format(
            verilog=ctx["verilog"], property="\n".join(ctx["property"]),
            invariants="\n".join(ctx["invariants"]),
            pdr=PDR_TRUE_LINE if pdr_says_true(tasks[0]) else "",
            cti=cti or "(none)"))
        return

    promoted_total = 0
    for rec in tasks:
        print(f"task {rec.get('extension_id')} ({rec.get('verdict')}, "
              f"{rec.get('ext_type')} on {rec.get('parent_id')})")
        won = run_task(rec, corpus_rows)
        if not won:
            continue
        fixed = repaired_record(rec, won["invariants"], won["result"])
        _, rows = promote(corpus_rows, [fixed])
        corpus_rows.extend(rows)
        with CORPUS.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        for row in rows:
            promoted_total += 1
            print(f"  promoted {row['id']} gen={row['generation']} "
                  f"(repaired {rec.get('extension_id')})")
    print(f"drain complete: {promoted_total} repair(s) promoted")


if __name__ == "__main__":
    main()
