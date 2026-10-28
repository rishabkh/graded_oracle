"""Corpus rows -> training pairs.

One rule governs the shape: the training question is the SAME text the
evaluation asks (`solver_baseline.SOLVER_PROMPT`), so the model is
trained on the task it is scored on. If that prompt changes, this file
follows it automatically, because it imports it rather than copying it.

Run 1 is answers only, per Nada: design and property in, invariant list
out, nothing else. Run 2 adds reasoning, and the repair pairs written by
`--repairs` are its most interesting source - a failed attempt, the
counterexample that refuted it, and the fix that closed the proof, all
produced by the pipeline rather than narrated after the fact.

Run v3 is v2 with reasoning: the same rows and the same question, and
only the answer differs. The reasoning goes INSIDE the answer object,
first: {"reasoning": ..., "invariants": [...]}. The scorer's reader
(`solver_baseline.parse_invariants`) takes everything from the first
"{" to the last "}" and reads "invariants"; reasoning written as prose
before the JSON would often carry Verilog braces like {2'b00, x} and
make the answer unreadable. Inside a JSON string those braces are
harmless. Only reasoning from the regeneration prompt is accepted, so
the generator's original notes cannot leak into this file, and every
pair is read back through the real scorer before anything is written.

  venv/bin/python training/build_sft.py --out extender/sft_train.jsonl
  venv/bin/python training/build_sft.py --repairs extender/sft_repairs.jsonl
  venv/bin/python training/build_sft.py \\
      --reasoning extender/logs/reasoning_regen.jsonl \\
      --out extender/sft_train_v3.jsonl

Generation too (8 Oct 2026): `--distill` adds the generator's own calls
(idea, implement, initiate, generate, repair, extend) from the training
examples it saved (distill_log.py), under Formal Disco's rules
(formal-disco/distill.py build_sft_records): successes only; an idea counts
only if a design built from it passed; the question is the exact chat
messages that were sent, the answer one assistant turn. Solving pairs still
come from the corpus, asked exactly as the test asks. Any record that shows
a design v2 holds back is held back too, so nothing held back leaks in
through a generation answer. An extend record (21 Oct 2026) is also held
back when its grown design (metadata "child_verilog") is a held-back row,
whatever its spacing, or when it grew from one (metadata "parent_ids").

  venv/bin/python training/build_sft.py \\
      --extra-corpus extender/corpus_2026-10-09.jsonl \\
      --distill initiator/logs/distill_attempts_2026-10-09.jsonl \\
      --out extender/sft_train_with_design_writing.jsonl

Designs the automatic engine cannot prove alone count for more (21 Oct
2026, Rishab approved): for about 4 in 5 new designs the engine built
into modern proof tools finds its own helper facts (198 of 252 on 9 Oct,
15 of 19 grown designs on 21 Oct), so nobody would write facts for them;
the ones where it runs out of time are where people still do.
`--engine-hard-copies K` writes each such solving row K times (rows with
metrics["engine_alone"] == "timed_out"). The others stay in, once: they
still teach the patterns of helper facts. A held-back row is never
repeated, and a file with repeats carries a holdout mark on every line,
so the trainer never splits copies of one question across its two sides.

Only facts the benchmark's checker can read (22 Oct 2026, review findings
F2, F3 and F9). The trained model is scored by EBMC 6.0, which cannot read
a fact with a system function ($onehot, $countones, $past, ...), the
compact `forall k in 0..7:` form, or a dotted name (fact_format.py). Five
answers in corpus.jsonl use $onehot(st) (g0_051, g1_001, g1_089, g2_075,
g3_073), c0_144 in corpus_2026-10-09.jsonl uses $countones, and a saved
extension answer can hold compact facts, so every answer this file writes
is checked first. An extend answer whose metadata has "invariants_expanded"
(the plain list the checker proved) gets that list in place of the facts
it wrote, the rest of the answer kept. Any other answer with a fact EBMC
cannot read is left out, never rewritten, and the run prints how many and
which; so is an extend answer whose list of kept and replaced parent
facts names one (23 Oct 2026, listed_facts). Held-back rows are chosen on the full list before anything is left
out, so leaving a row out never changes which other rows are held back;
a file that used to leave that choice to the trainer (the answers-only
file, the repairs file) carries a holdout mark on every line once a row
is left out, and stays as it was when nothing is.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))

from solver_baseline import SOLVER_PROMPT, parse_invariants  # noqa: E402
from sft_data import split                                   # noqa: E402
from fact_format import unreadable_facts                     # noqa: E402
import distill_log                                           # noqa: E402

CORPUS = HERE.parent / "extender" / "corpus.jsonl"
FIXER_LOG = HERE.parent / "extender" / "logs" / "fixer_attempts.jsonl"
V2_FILE = HERE.parent / "extender" / "sft_train_v2.jsonl"
REGEN_KIND = "regen"

REPAIR_PROMPT = """\
{base}
A previous attempt gave these invariants:
{attempt}

The prover refused them: {diagnosis}
Counterexample state: {cti}

Give a corrected invariant list in the same JSON format."""


def build_pair(row):
    """One (question, answer) from one corpus row."""
    invariants = row.get("invariants") or []
    if not invariants:
        raise ValueError(f"{row.get('id')}: no invariants to learn from")
    return {"prompt": SOLVER_PROMPT.format(verilog=row["verilog"]),
            "completion": json.dumps({"invariants": invariants}),
            "id": row.get("id"), "generation": row.get("generation")}


def build_pairs(rows):
    """Every row, minus designs we would otherwise train on twice. The
    corpus has no duplicates today; extensions that only change the
    invariant list would create them."""
    seen, out = set(), []
    for row in rows:
        design = row.get("verilog", "")
        if design in seen:
            continue
        seen.add(design)
        try:
            out.append(build_pair(row))
        except ValueError:
            continue
    return out


# 23 Oct 2026, review finding: every failed fixer attempt became a repair
# example saying "The prover refused them", but an attempt turned away
# before any check (FACT_FORMAT, READS_RESET, PRE_GATE) or never answered
# (REFUSED, UNPARSEABLE) never reached the prover, and one whose check ran
# out of time or broke was not refused for its facts. Only an attempt the
# prover checked and refused is paired now. And "Counterexample state" was
# filled with the attempt's cti_leg, the name of the check its question's
# counterexample came from ("with_invariants"), not a counterexample: it
# now holds the counterexample that refuted the attempt itself, from that
# attempt's own check. The fixer log had no such example yet (7 attempts,
# 0 repair pairs), so no training file changes.
def refused_by_the_prover(attempt):
    """Was this fixer attempt checked, and its facts refused by the
    prover (NOT_PROVEN, not from a check that ran out of time or broke)?"""
    reason = str(attempt.get("reason", ""))
    return (attempt.get("verdict") == "NOT_PROVEN"
            and "grade is ERROR" not in reason
            and "grade is TIMEOUT" not in reason)


def attempt_counterexample(attempt):
    """The counterexample that refuted an attempt's facts: the trace of the
    proof run of its own check with the facts (the with-invariants leg), or
    None when none was kept."""
    leg = ((attempt.get("result") or {}).get("with_invariants") or {})
    for run in leg.get("runs") or []:
        if run.get("mode") == "prove" and run.get("trace_text"):
            return run["trace_text"]
    return None


def repair_pairs(attempts, rows_by_task):
    """Pair each fixer attempt the prover refused with the answer that
    finally worked. Tasks that never got fixed contribute nothing: an
    unsolved failure is not a lesson; nor does an attempt that never
    reached the prover (refused_by_the_prover)."""
    by_task = {}
    for a in attempts:
        by_task.setdefault(a.get("task_id"), []).append(a)

    out = []
    for task, tries in by_task.items():
        row = rows_by_task.get(task)
        winner = next((t for t in tries
                       if t.get("verdict") == "NECESSARY"), None)
        if row is None or winner is None:
            continue
        for t in tries:
            if t is winner or not refused_by_the_prover(t):
                continue
            prompt = REPAIR_PROMPT.format(
                base=SOLVER_PROMPT.format(verilog=row["verilog"]),
                attempt=json.dumps({"invariants": t.get("invariants") or []}),
                diagnosis=t.get("diagnosis") or "the proof did not close",
                cti=attempt_counterexample(t) or "(none recorded)")
            out.append({"prompt": prompt,
                        "completion": json.dumps(
                            {"invariants": winner.get("invariants") or []}),
                        "task_id": task})
    return out


# --- only facts the benchmark's checker can read ---------------------------
# (22 Oct 2026, review findings F3/F9: five corpus answers with $onehot(st)
# and one with $countones reached the training files, and EBMC, the
# benchmark's checker, reads neither; F2: so could an extension answer's
# compact `forall` facts.) Every writer below leaves such an answer out
# after the held-back rows are chosen, and says which.

def _answer_object(text):
    """The JSON object an answer holds, or None. The whole text when it is
    one object, else read like the scorer reads (solver_baseline.
    parse_invariants: from the first { to the last }), so a fence, prose
    or a list around the object hides no fact."""
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if not isinstance(obj, dict):
        m = re.search(r"\{.*\}", text, flags=re.S)
        try:
            obj = json.loads(m.group(0).replace("\\'", "'")) if m else None
        except ValueError:
            obj = None
    return obj if isinstance(obj, dict) else None


def answer_facts(text):
    """The fact list in an answer (its "invariants"), or None when it has
    none: an idea, a distractor's patch, a reply that is not JSON."""
    obj = _answer_object(text)
    facts = obj.get("invariants") if obj is not None else None
    return [str(f) for f in facts] if isinstance(facts, list) else None


def leave_out_unreadable(pairs, answer="completion", name="id"):
    """(kept, left_out): the pairs whose answer (the `answer` field) has
    only facts EBMC can read, in order, and the others as (the pair's
    `name` field, [(fact, problems)]). Nothing is rewritten."""
    kept, left_out = [], []
    for p in pairs:
        bad = unreadable_facts(answer_facts(p[answer]) or [])
        if bad:
            left_out.append((p.get(name), bad))
        else:
            kept.append(p)
    return kept, left_out


def say_left_out(what, left_out):
    """Print how many `what` were left out for facts EBMC cannot read,
    which ones and why, or that none were: a left-out answer is never
    silent."""
    if not left_out:
        print(f"  {what}: EBMC can read every fact")
        return
    names = list(dict.fromkeys(str(n) for n, _ in left_out))
    print(f"  left out {len(left_out)} {what} with facts EBMC cannot read "
          f"(not rewritten): {', '.join(names)}")
    shown = set()
    for n, bad in left_out:
        if str(n) in shown:
            continue
        shown.add(str(n))
        for fact, problems in bad:
            print(f"    {n}: {fact}  ({'; '.join(problems)})")


def mark_trainer_split(pairs):
    """Every pair marked with the held-back choice the trainer would make
    on this full list (sft_data.split with its defaults, 10% and seed 0,
    which train_lora.py uses unless told otherwise), for a file that left
    the choice to the trainer: once a pair is left out, a shuffle of the
    shorter list would hold back other pairs."""
    held = {id(p) for p in split(pairs)[1]}
    return [dict(p, holdout=id(p) in held) for p in pairs]


def load_reasoning(path):
    """id -> the last accepted record, in file order, so a rerun that
    restarts its attempt count still wins. Any line not written by the
    regeneration prompt stops the build: the generator's original notes
    cover only some rows and were written differently, and mixing the
    two would blur the one thing v3 changes."""
    out = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {n}: not JSON ({exc})")
        if rec.get("prompt_kind") != REGEN_KIND:
            raise ValueError(
                f"{path} line {n}: prompt_kind {rec.get('prompt_kind')!r}, "
                f"expected {REGEN_KIND!r}; only regenerated reasoning may "
                "enter the training file")
        if rec.get("accepted") is not True:
            continue
        state = rec.get("cti_state")
        if (not rec.get("id") or not isinstance(rec.get("cti_reasoning"), str)
                or not rec["cti_reasoning"].strip()
                or not isinstance(state, list) or not state
                or not all(isinstance(s, dict) and "signal" in s
                           and "value" in s for s in state)):
            raise ValueError(f"{path} line {n}: accepted but missing an id, "
                             "the reasoning or the fake state")
        out[rec["id"]] = rec
    return out


def reasoning_text(record):
    state = ", ".join(f"{s['signal']} = {s['value']}"
                      for s in record["cti_state"])
    return f"Fake state: {state}. " + record["cti_reasoning"]


def build_reasoning_pairs(rows, reasoning, missing="fail"):
    """The v2 pairs with reasoning added to each answer. Returns the
    pairs and the ids that had no accepted reasoning; `missing` says
    what happens to those: fail, drop, or keep them answers-only.

    Each pair carries v2's own held-back choice (`holdout`), made on the
    full v2 list, so dropping rows leaves the held-back set unchanged."""
    base = build_pairs(rows)
    held = {p["prompt"] for p in split(base)[1]}
    pairs, absent = [], []
    for pair in base:
        pair = dict(pair, holdout=pair["prompt"] in held)
        rec = reasoning.get(pair["id"])
        if rec is None:
            absent.append(pair["id"])
            if missing == "answers":
                pairs.append(dict(pair, reasoning=False))
            continue
        invariants = json.loads(pair["completion"])["invariants"]
        pairs.append(dict(pair, reasoning=True, completion=json.dumps(
            {"reasoning": reasoning_text(rec), "invariants": invariants})))
    if absent and missing == "fail":
        raise ValueError(f"{len(absent)} rows have no accepted reasoning: "
                         + ", ".join(map(str, absent)))
    return pairs, absent


def check_parity(pairs, rows):
    """Every answer must read back, through the scorer's own reader, to
    exactly its row's invariants, and every question must be the v2
    question for that row. Returns the problems; empty means fine."""
    by_id = {r.get("id"): r for r in rows}
    problems = []
    for pair in pairs:
        row = by_id.get(pair["id"])
        if row is None:
            problems.append(f"{pair['id']}: not in the corpus")
            continue
        if parse_invariants(pair["completion"]) != row["invariants"]:
            problems.append(f"{pair['id']}: the scorer does not read back "
                            "this row's invariants")
        if pair["prompt"] != build_pair(row)["prompt"]:
            problems.append(f"{pair['id']}: question differs from v2")
    return problems


def write_reasoning_file(args, rows):
    """Build, check, and only then write: a file that fails the parity
    check is never left on disk."""
    if not Path(args.reasoning).exists():
        sys.exit(f"no reasoning file at {args.reasoning}")
    try:
        reasoning = load_reasoning(args.reasoning)
        pairs, absent = build_reasoning_pairs(rows, reasoning, args.missing)
    except ValueError as exc:
        sys.exit(f"refused: {exc}")
    problems = check_parity(pairs, rows)
    if problems:
        sys.exit(f"parity check failed on {len(problems)} pairs, nothing "
                 "written:\n  " + "\n  ".join(problems))

    kept = sum(1 for p in pairs if p["reasoning"])
    unused = set(reasoning) - {r.get("id") for r in rows}
    print(f"reasoning from {args.reasoning}")
    print(f"  rows {kept + len(absent)}, with reasoning {kept}, "
          f"missing {len(absent)}, policy {args.missing} "
          f"-> {len(pairs)} pairs")
    if unused:
        print(f"  {len(unused)} reasoning ids not in the corpus, ignored")
    if pairs:
        chars = sorted(len(p["prompt"]) + len(p["completion"])
                       for p in pairs)
        print(f"  median {chars[len(chars)//2]} chars, longest {chars[-1]} "
              f"(~{chars[-1] * 10 // 36} tokens)")
    print("  parity: every answer reads back to its row's invariants, "
          "every question matches v2")
    # every pair already carries v2's held-back mark, made on all rows
    # (22 Oct 2026: so leaving a pair out moves no other pair across)
    pairs, left_out = leave_out_unreadable(pairs)
    say_left_out("solving answers", left_out)
    pairs = weight_engine_hard(pairs, rows, args.engine_hard_copies)
    if args.out:
        Path(args.out).write_text(
            "".join(json.dumps({"prompt": p["prompt"],
                                "completion": p["completion"],
                                "holdout": p["holdout"]}) + "\n"
                    for p in pairs))
        print(f"  output {args.out} ({len(pairs)} lines)")
    else:
        print("  output: none (no --out given)")


def held_back(rows):
    """v2's own held-back questions, chosen on the full v2 list, so a file
    built from fewer rows still holds back exactly v2's rows."""
    return {p["prompt"] for p in split(build_pairs(rows))[1]}


def held_rows(rows):
    """v2's held-back rows: every row whose question is held back."""
    prompts = held_back(rows)
    return [r for r in rows if r.get("invariants")
            and build_pair(r)["prompt"] in prompts]


def held_designs(rows):
    """The Verilog of v2's held-back rows."""
    return {r["verilog"] for r in held_rows(rows)}


def held_ids(rows):
    """The ids of v2's held-back rows. A grown design names the rows it
    grew from by id (the extender's training examples, 21 Oct 2026)."""
    return {r["id"] for r in held_rows(rows) if r.get("id") is not None}


ENGINE_HARD = "timed_out"


def engine_alone(row):
    """What the automatic engine did with the property alone, recorded
    on rows built from 21 Oct 2026: "proved", "timed_out", "error" or
    "not_tried". None on rows built before."""
    return (row.get("metrics") or {}).get("engine_alone")


def engine_hard_questions(rows):
    """The training questions of rows the engine ran out of time on."""
    return {SOLVER_PROMPT.format(verilog=r["verilog"]) for r in rows
            if engine_alone(r) == ENGINE_HARD and r.get("verilog")}


def repeat_engine_hard(pairs, hard, copies):
    """Each trained pair whose question is in `hard`, written `copies`
    times, the copies right after it. A held-back pair is never repeated:
    a copy in training would leak it. Returns the pairs, how many pairs
    were repeated, and how many engine-hard pairs are held back."""
    if copies > 1 and not all("holdout" in p for p in pairs):
        raise ValueError("mark every pair held back or not before "
                         "repeating any")
    out, repeated, held = [], 0, 0
    for p in pairs:
        out.append(p)
        if p["prompt"] not in hard:
            continue
        if p.get("holdout"):
            held += 1
        elif copies > 1:
            out += [dict(p) for _ in range(copies - 1)]
            repeated += 1
    return out, repeated, held


def engine_alone_line(rows):
    """How the rows split by what the engine did alone, or None when no
    row records it (every corpus built before 21 Oct 2026)."""
    counts = collections.Counter(engine_alone(r) or "not recorded"
                                 for r in rows)
    if set(counts) <= {"not recorded"}:
        return None
    return "  the engine alone: " + ", ".join(
        f"{k} {v}" for k, v in sorted(counts.items()))


def weight_engine_hard(pairs, rows, copies):
    """repeat_engine_hard on solving pairs built from `rows`, saying what
    it did. With copies 1 the pairs come back unchanged."""
    if engine_alone_line(rows):
        print(engine_alone_line(rows))
    out, repeated, held = repeat_engine_hard(
        pairs, engine_hard_questions(rows), copies)
    if copies > 1:
        print(f"  engine-hard rows (the engine ran out of time alone): "
              f"{repeated} repeated, each written {copies} times "
              f"(+{len(out) - len(pairs)} lines); {held} held back, "
              "never repeated")
    return out


# 23 Oct 2026, review finding: rebuilt in place with this builder,
# sft_train_v2.jsonl has 660 lines (the 5 answers EBMC cannot read left
# out, every line marked held back or not), and the combined build then
# stopped, as it compared line for line with the 665 lines first written.
# It now accepts either: the file as first written, or the file this
# builder writes today from the same rows.
def v2_file_matches(lines, v2_pairs):
    """How the v2 file's `lines` match v2's pairs as built now, each
    carrying v2's held-back mark: "as first written" (every pair, its
    question and answer only, in order) or "rebuilt with unreadable
    answers left out" (only the pairs whose facts EBMC can read, in order,
    a line's held-back mark, when it has one, being the pair's own); None
    when neither."""
    if lines == [json.dumps({"prompt": q["prompt"],
                             "completion": q["completion"]})
                 for q in v2_pairs]:
        return "as first written"
    kept, _ = leave_out_unreadable(v2_pairs)
    try:
        got = [json.loads(line) for line in lines]
    except ValueError:
        return None
    if len(got) != len(kept):
        return None
    for line, pair in zip(got, kept):
        if not isinstance(line, dict) \
                or (line.get("prompt"), line.get("completion")) != \
                (pair["prompt"], pair["completion"]) \
                or line.get("holdout", pair["holdout"]) != pair["holdout"]:
            return None
    return "rebuilt with unreadable answers left out"


def build_combined_pairs(rows, extra_rows):
    """v2's pairs with v2's held-back marks, then a separate corpus's pairs
    (the catalog run, 8 Oct 2026), all trained on. Appending unmarked rows
    to v2's file would make the trainer re-pick the held-back set (60 of
    v2's 66 held-back rows would be trained on); every row is marked
    instead. A new question equal to a v2 question is refused."""
    held = held_back(rows)
    base = [dict(p, holdout=p["prompt"] in held) for p in build_pairs(rows)]
    known = {p["prompt"] for p in base}
    extra = build_pairs(extra_rows)
    clash = [p["id"] for p in extra if p["prompt"] in known]
    if clash:
        raise ValueError(f"questions already in v2's file: {clash[:5]}")
    return base + [dict(p, holdout=False) for p in extra]


GEN_TYPES = ("idea", "implement", "initiate", "generate", "repair", "extend")
CHARS_PER_TOKEN = 3.6     # the estimate this file has always printed


def design_of(response):
    """The Verilog inside a design answer, or None (an idea, a cut-off
    reply)."""
    try:
        return json.loads(response).get("verilog")
    except (TypeError, ValueError, AttributeError):
        return None


def idea_text(response):
    """An idea answer is the text, or JSON {"idea": text} when the call
    forced that shape."""
    try:
        obj = json.loads(response)
    except (TypeError, ValueError):
        return str(response)
    return str(obj.get("idea")) if isinstance(obj, dict) else str(response)


def _flat(text):
    """The text with every run of spaces, tabs and newlines made one
    space, so a design copied with other spacing still matches."""
    return " ".join(str(text).split())


def held_marks(designs, ids=()):
    """What gives a held-back row away: its design as written, as escaped
    JSON text and with its spacing flattened, and its id. A blank design
    is skipped: it would match every example."""
    designs = [d for d in designs if d and str(d).strip()]
    return {"forms": [f for d in designs for f in (d, json.dumps(d)[1:-1])],
            "flat": [_flat(d) for d in designs],
            "ids": {str(i) for i in ids}}


def _parts(example):
    """Every text of an example that can show a design: the answer, the
    design the answer made (metadata "design"; "child_verilog" for a
    grown design, whose answer is only a patch), and every message of the
    question."""
    meta = example.get("metadata") or {}
    parts = [example.get("response"), meta.get("design"),
             meta.get("child_verilog")]
    parts += [m.get("content", "") for m in meta.get("messages") or []]
    return [str(p) for p in parts if p]


def _parent_ids(example):
    """The corpus ids a grown design came from: metadata "parent_ids",
    and the same key in the call's arguments in case it is kept there."""
    ids = set()
    for where in (example.get("metadata") or {},
                  example.get("arguments") or {}):
        got = where.get("parent_ids") if isinstance(where, dict) else None
        if isinstance(got, (list, tuple, set)):
            ids |= {str(i) for i in got if i is not None}
        elif got:
            ids.add(str(got))
    return ids


def _shows(example, marks):
    """Does the example show a held-back row? Some part of it (question,
    answer, or the design an answer made) contains a held-back design,
    as written, as escaped JSON text, or with its spacing flattened; or
    the design it grew came from a held-back row. (Found by the 10 Oct
    review: an extend answer is a patch and a fact list, so a grown design
    that is a held-back row never showed in the answer itself, and 44 of
    v2's 66 held-back rows are grown designs.)"""
    parts = _parts(example)
    if any(form in part for part in parts for form in marks["forms"]):
        return True
    flat = [_flat(p) for p in parts]
    if any(d in part for part in flat for d in marks["flat"]):
        return True
    return bool(marks["ids"] & _parent_ids(example))


def with_expanded_facts(answer, expanded):
    """An extend answer with its "invariants" replaced by `expanded`, the
    plain list the checker proved (metadata "invariants_expanded"), and
    every other field of the answer kept as it was, in its place; written
    back as one JSON object. None when there is nothing to replace: no
    such list, no fact list in the answer (a distractor's patch), or the
    same list already, so such an answer stays exactly as written.
    (22 Oct 2026, review finding F2: an answer kept as written taught the
    compact `forall k in 0..7:` form, which EBMC cannot read.)"""
    if not (isinstance(expanded, list) and expanded
            and all(isinstance(f, str) and f.strip() for f in expanded)):
        return None
    obj = _answer_object(answer)
    if obj is None or not isinstance(obj.get("invariants"), list) \
            or obj["invariants"] == expanded:
        return None
    return json.dumps(dict(obj, invariants=expanded), ensure_ascii=False)


# An extend answer also says, for each parent fact as its question showed
# it, whether the fact was kept or replaced and by which facts (its
# "dispositions"); the extender checks that list against the facts as
# written. A parent's compact `forall` fact, or a compact replacing fact,
# stays in that list when the answer is rebuilt with the plain facts: the
# answer then calls a fact kept that its own list no longer holds, and
# still writes the compact form, which EBMC cannot read. Rishab chose (23
# Oct 2026) to leave such answers out, not rewrite their lists: 1 of the 19
# kept steps of the 21 Oct trial (2026-10-21_08h58m35s_t030, which kept
# its parent's `forall k in 0..7: ...`). A list that names only plain facts
# still agrees once the compact ones are spelled out, since a plain fact
# is the same either way.

def listed_facts(text):
    """The facts an extend answer's kept/replaced list names: each parent
    fact it declares kept or replaced, then each fact named as its
    replacement, in order; [] when the answer has no such list."""
    obj = _answer_object(text)
    items = obj.get("dispositions") if obj is not None else None
    out = []
    for item in items if isinstance(items, list) else ():
        if not isinstance(item, dict):
            continue
        by = item.get("replaced_by") or []
        for fact in [item.get("clause")] + ([by] if isinstance(by, str)
                                            else list(by)):
            if isinstance(fact, str) and fact.strip():
                out.append(fact)
    return out


def example_name(example, n):
    """A name to print for a saved example: the extender's step id, the
    generator's run and attempt, or else its place in the input."""
    meta = example.get("metadata") or {}
    if meta.get("extension_id"):
        return str(meta["extension_id"])
    for run, attempt in (("run_id", "attempt"),
                         ("source_run_id", "source_attempt")):
        if meta.get(run):
            return f"{meta[run]} attempt {meta.get(attempt)}"
    return f"example {n + 1} of the input"


def generation_records(examples, held_designs=frozenset(),
                       held_ids=frozenset(), fact_report=None):
    """Formal Disco's rules over saved call examples. Returns the records,
    in input order, and per-type counts of kept and skipped examples.
    `held_designs` are the Verilog texts of held-back solving rows and
    `held_ids` their ids: a record that shows one anywhere (its question,
    its answer, or the design its answer made, also as escaped JSON text
    or with other spacing), or whose grown design came from one, is held
    back too, and so is an idea whose kept design is one. Whether a record
    is held back never depends on which other records are left out.

    Facts EBMC can read only (22 Oct 2026, review findings F2 and F9): an
    extend answer whose metadata has "invariants_expanded" gets that list
    in place of its own facts (with_expanded_facts); any answer still
    holding a fact EBMC cannot read is left out, counted as skipped, and
    so is an extend answer whose kept/replaced list names one (23 Oct
    2026, see listed_facts). `fact_report`, when a dict, is filled with
    what this did: "rebuilt", the names of the kept rebuilt answers;
    "left_out", [(kind and name, [(fact, problems)])]; and "lists", the
    names of the extend answers left out for their kept/replaced list."""
    report = {"rebuilt": [], "left_out": [], "lists": []}
    marks = held_marks(held_designs, held_ids)
    good_ideas = {str(e["arguments"].get("idea")) for e in examples
                  if e.get("prompt") == "implement"
                  and e.get("outcome") == "success"
                  and e.get("arguments", {}).get("idea")}
    held_ideas = {str(e["arguments"].get("idea")) for e in examples
                  if e.get("prompt") == "implement"
                  and e.get("outcome") == "success"
                  and _shows(e, marks)}
    records, counts = [], {}
    for n, e in enumerate(examples):
        kind = e.get("prompt")
        if kind not in GEN_TYPES:
            continue
        c = counts.setdefault(kind, {"kept": 0, "skipped": 0})
        response = e.get("response")
        ok = (idea_text(response) in good_ideas if kind == "idea"
              else e.get("outcome") == "success")
        messages = (e.get("metadata") or {}).get("messages")
        if not ok or response is None or not messages:
            c["skipped"] += 1
            continue
        answer = str(response)
        rebuilt = (with_expanded_facts(answer, (e.get("metadata") or {})
                                       .get("invariants_expanded"))
                   if kind == "extend" else None)
        answer = rebuilt or answer
        bad = unreadable_facts(answer_facts(answer) or [])
        listed = ([(fact, [p + ", in the kept/replaced list"
                           for p in problems])
                   for fact, problems in unreadable_facts(
                       listed_facts(answer))]
                  if kind == "extend" else [])
        if bad or listed:
            report["left_out"].append((f"{kind} {example_name(e, n)}",
                                       bad + listed))
            if listed:
                report["lists"].append(example_name(e, n))
            c["skipped"] += 1
            continue
        if rebuilt:
            report["rebuilt"].append(example_name(e, n))
        held = (idea_text(response) in held_ideas if kind == "idea"
                else _shows(e, marks))
        records.append({"prompt": messages,
                        "completion": [{"role": "assistant",
                                        "content": answer}],
                        "holdout": held, "kind": kind})
        c["kept"] += 1
    if fact_report is not None:
        fact_report.update(report)
    return records, counts


def record_tokens(rec):
    """A rough token count of one training line (question and answer)."""
    def chars(x):
        if isinstance(x, str):
            return len(x)
        return sum(len(m.get("content", "")) for m in x)
    return int((chars(rec["prompt"]) + chars(rec["completion"]))
               / CHARS_PER_TOKEN)


def report_lengths(lines, max_len):
    """Training drops every line over its length limit; say so here, by
    kind, before any GPU time is spent."""
    over = {}
    for line in lines:
        if record_tokens(line) > max_len:
            kind = line.get("kind", "solve")
            over[kind] = over.get(kind, 0) + 1
    n = sum(over.values())
    if n:
        print(f"  WARNING: {n} record(s) over {max_len} tokens (estimated), "
              f"which training would drop: {over}")
    else:
        print(f"  every record fits in {max_len} tokens (estimated)")
    return over


def build_clean_pairs(rows, passing):
    """Run 4a: v2's pairs, keeping only answers that pass the 1-step rule
    (training/one_step.py). 207 of 665 do not: by the benchmark's own
    standard they are true but too weak, the habit v1 and v2 show."""
    held = held_back(rows)
    return [dict(p, holdout=p["prompt"] in held) for p in build_pairs(rows)
            if p["id"] in passing]


def build_preference_pairs(rows, s):
    """Run 4b: for each answer that passes the 1-step rule, one pair per
    needed fact: the full answer (chosen, v2's own completion) against the
    same answer without that fact (rejected: still true, too weak by the
    1-step rule, with the proof tool's counterexample attached)."""
    held = held_back(rows)
    by_id = {r.get("id"): r for r in rows}
    out = []
    for p in build_pairs(rows):
        if p["id"] not in s["passes_one_step"]:
            continue
        inv = by_id[p["id"]]["invariants"]
        for i in s["needed"].get(p["id"], []):
            out.append({"prompt": p["prompt"], "chosen": p["completion"],
                        "rejected": json.dumps(
                            {"invariants": inv[:i] + inv[i + 1:]}),
                        "holdout": p["prompt"] in held, "id": p["id"],
                        "dropped": inv[i],
                        "counterexample": s["counterexample"].get((p["id"], i))})
    return out


def at_least_trained(pairs, n):
    """Release held-back pairs, first ones first, until n are trained on
    (29 Oct 2026, Rishab: "no test should have less than 200 training
    examples"; the 217-pair sets held back 21 and trained on 196). The
    held-back slice only watches the loss; scores never come from it."""
    need = n - sum(not p["holdout"] for p in pairs)
    for p in pairs:
        if need <= 0:
            break
        if p["holdout"]:
            p["holdout"] = False
            need -= 1
    return pairs


def write_one_step_files(args, rows):
    """Build, check that every answer reads back through the scorer, and
    only then write."""
    import one_step
    s = one_step.summary(args.one_step)
    by_id = {r.get("id"): r for r in rows}
    v2_prompts = {p["id"]: p["prompt"] for p in build_pairs(rows)}
    problems = []
    clean = build_clean_pairs(rows, s["passes_one_step"])
    for p in clean:
        if parse_invariants(p["completion"]) != by_id[p["id"]]["invariants"]:
            problems.append(f"{p['id']}: clean answer does not read back")
    pref = build_preference_pairs(rows, s)
    for p in pref:
        full = by_id[p["id"]]["invariants"]
        if parse_invariants(p["chosen"]) != full:
            problems.append(f"{p['id']}: chosen does not read back")
        rej = parse_invariants(p["rejected"])
        if rej is None or len(rej) != len(full) - 1:
            problems.append(f"{p['id']}: rejected does not read back")
        if p["prompt"] != v2_prompts[p["id"]]:
            problems.append(f"{p['id']}: question differs from v2")
    if problems:
        sys.exit(f"check failed on {len(problems)} items, nothing written:\n  "
                 + "\n  ".join(problems[:20]))
    print(f"1-step rule ({args.one_step}): {len(s['passes_one_step'])} of "
          f"{len(rows)} answers pass")
    if one_step.before_fix_warning(s):
        print("  " + one_step.before_fix_warning(s))
    # both lists carry v2's held-back marks, made on all rows; the chosen
    # answer is the one a preference pair teaches (the rejected one is it
    # minus one fact)
    clean, clean_left_out = leave_out_unreadable(clean)
    pref, pref_left_out = leave_out_unreadable(pref, answer="chosen")
    if args.clean:
        say_left_out("clean answers", clean_left_out)
        clean = weight_engine_hard(clean, rows, args.engine_hard_copies)
        clean = at_least_trained(clean, args.min_trained)
        Path(args.clean).write_text("".join(
            json.dumps({"prompt": p["prompt"], "completion": p["completion"],
                        "holdout": p["holdout"]}) + "\n" for p in clean))
        print(f"  clean pairs: {len(clean)} "
              f"({sum(not p['holdout'] for p in clean)} trained on) -> {args.clean}")
    if args.preference:
        say_left_out("preference pairs", pref_left_out)
        Path(args.preference).write_text("".join(
            json.dumps(p) + "\n" for p in pref))
        print(f"  preference pairs: {len(pref)} from "
              f"{len({p['id'] for p in pref})} answers "
              f"({sum(not p['holdout'] for p in pref)} trained on) -> "
              f"{args.preference}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default=None,
                   help="pairs file: answers only (run 1, v2), or with "
                        "reasoning when --reasoning is given (v3)")
    p.add_argument("--repairs", default=None,
                   help="failed-then-fixed pairs, run 2")
    p.add_argument("--reasoning", default=None,
                   help="regenerated reasoning log "
                        "(extender/logs/reasoning_regen.jsonl)")
    p.add_argument("--missing", choices=["fail", "drop", "answers"],
                   default="fail",
                   help="rows with no accepted reasoning: stop, leave "
                        "them out, or keep them answers-only")
    p.add_argument("--one-step", default=None,
                   help="1-step check log (extender/logs/one_step.jsonl)")
    p.add_argument("--clean", default=None,
                   help="with --one-step: answers that pass the 1-step "
                        "rule only (run 4a)")
    p.add_argument("--preference", default=None,
                   help="with --one-step: chosen/rejected pairs, one per "
                        "needed fact (run 4b)")
    p.add_argument("--extra-corpus", default=None,
                   help="with --out: v2's rows with v2's held-back marks, "
                        "plus every row of this corpus trained on (the "
                        "catalog run); every row carries a holdout mark")
    p.add_argument("--distill", nargs="+", default=None,
                   help="with --out: also the generator's own calls from "
                        "these saved-example files (distill_*.jsonl), "
                        "under Formal Disco's rules")
    p.add_argument("--max-len", type=int, default=16384,
                   help="the training length limit, to report records "
                        "that training would drop")
    p.add_argument("--min-trained", type=int, default=0,
                   help="with --clean: release held-back pairs until this many are trained on")
    p.add_argument("--engine-hard-copies", type=int, default=1,
                   metavar="K",
                   help="write each solving row the automatic engine "
                        "could not prove alone (metrics engine_alone = "
                        "timed_out) K times; held-back rows never repeat. "
                        "1 (the default) changes nothing")
    args = p.parse_args(argv)
    if args.engine_hard_copies < 1:
        sys.exit("--engine-hard-copies must be 1 or more; nothing was "
                 "written")
    if args.engine_hard_copies > 1 and not (args.clean if args.one_step
                                            else args.out):
        # only solving pairs repeat; with no solving file the option
        # would do nothing, silently
        sys.exit("--engine-hard-copies needs a solving file (--out, or "
                 "--clean with --one-step); nothing was written")
    if (args.extra_corpus or args.distill) and not args.out:
        sys.exit("--extra-corpus and --distill need --out")
    if (args.extra_corpus or args.distill) and (args.reasoning
                                                or args.one_step):
        # they write different files; one would be silently skipped
        sys.exit("--extra-corpus and --distill cannot be combined with "
                 "--reasoning or --one-step; nothing was written")
    if (args.clean or args.preference) and not args.one_step:
        sys.exit("--clean and --preference need --one-step")

    rows = [json.loads(l) for l in
            Path(args.corpus).read_text().splitlines() if l.strip()]
    pairs = build_pairs(rows)
    chars = sorted(len(p["prompt"]) + len(p["completion"]) for p in pairs)
    print(f"{len(rows)} rows -> {len(pairs)} pairs")
    print(f"  median {chars[len(chars)//2]} chars, longest {chars[-1]} "
          f"(~{chars[-1] * 10 // 36} tokens)")
    by_gen = {}
    for pair in pairs:
        by_gen[pair["generation"]] = by_gen.get(pair["generation"], 0) + 1
    print("  by generation:", dict(sorted(by_gen.items())))

    if args.one_step:
        write_one_step_files(args, rows)
    elif args.reasoning:
        write_reasoning_file(args, rows)
    elif args.extra_corpus or args.distill:
        extra = ([json.loads(l) for l in
                  Path(args.extra_corpus).read_text().splitlines()
                  if l.strip()] if args.extra_corpus else [])
        combined = build_combined_pairs(rows, extra)
        n_base = len(pairs)
        if Path(args.corpus).resolve() == CORPUS.resolve() and V2_FILE.exists():
            how = v2_file_matches(V2_FILE.read_text().splitlines(),
                                  combined[:n_base])
            if how is None:
                sys.exit(f"the v2 part differs from {V2_FILE.name}; "
                         "nothing was written")
            print(f"  v2 part matches {V2_FILE.name} line for line ({how})")
        print(f"  {n_base} v2 pairs ({sum(q['holdout'] for q in combined)} "
              f"held back) + {len(combined) - n_base} new pairs, all trained")
        # after the line-for-line check above, which is against all of
        # v2; every pair already carries its held-back mark
        combined, left_out = leave_out_unreadable(combined)
        say_left_out("solving answers", left_out)
        combined = weight_engine_hard(combined, rows + extra,
                                      args.engine_hard_copies)
        lines = [{"prompt": q["prompt"], "completion": q["completion"],
                  "holdout": q["holdout"]} for q in combined]
        if args.distill:
            examples = [e for f in args.distill for e in distill_log.read(f)]
            # held back by v2's choice on every row, before any solving
            # answer was left out above (22 Oct 2026)
            fact_report = {}
            gen, counts = generation_records(examples, held_designs(rows),
                                             held_ids(rows),
                                             fact_report=fact_report)
            print(f"  generation records from {len(args.distill)} file(s): "
                  + ", ".join(f"{k} {c['kept']} kept / {c['skipped']} skipped"
                              for k, c in sorted(counts.items())))
            print(f"  {len(fact_report['rebuilt'])} extend answer(s) rebuilt "
                  "with the plain fact list the checker proved "
                  "(metadata invariants_expanded)")
            print(f"  {len(fact_report['lists'])} extend answer(s) left out "
                  "because their kept/replaced list names a fact EBMC cannot "
                  "read (a compact `forall` fact of the parent or of the "
                  "answer)")
            say_left_out("generation answers (counted as skipped)",
                         fact_report["left_out"])
            print(f"  {sum(r['holdout'] for r in gen)} generation record(s) "
                  "held back because they show a held-back design or grew "
                  "from one")
            lines += gen
        report_lengths(lines, args.max_len)
        Path(args.out).write_text("".join(json.dumps(l) + "\n"
                                          for l in lines))
        print(f"  {len(lines)} lines -> {args.out}")
    elif args.out:
        keys = ("prompt", "completion")
        held = held_back(rows)          # v2's choice, made on every row
        pairs, left_out = leave_out_unreadable(
            [dict(p, holdout=p["prompt"] in held) for p in pairs])
        say_left_out("solving answers", left_out)
        if args.engine_hard_copies > 1 or left_out:
            # every line carries v2's held-back choice: left to itself the
            # trainer would shuffle the list it is given, so copies of one
            # question could land on both sides, and a list with rows left
            # out (22 Oct 2026) would hold back other rows than v2's
            keys += ("holdout",)
        pairs = weight_engine_hard(pairs, rows, args.engine_hard_copies)
        Path(args.out).write_text(
            "".join(json.dumps({k: p[k] for k in keys}) + "\n"
                    for p in pairs))
        print(f"  wrote {len(pairs)} lines to {args.out}")

    if args.repairs:
        if not FIXER_LOG.exists():
            sys.exit("no fixer attempts logged yet")
        attempts = [json.loads(l) for l in
                    FIXER_LOG.read_text().splitlines() if l.strip()]
        # the fixer records a task id; corpus rows carry the same id on
        # the record that produced them
        rows_by_task = {r.get("source_task"): r for r in rows
                        if r.get("source_task") is not None}
        out = repair_pairs(attempts, rows_by_task)
        # held-back marks chosen on the full list, as the trainer would,
        # so leaving a pair out moves no other pair across (22 Oct 2026);
        # with nothing left out the file stays unmarked, as it was
        kept, left_out = leave_out_unreadable(mark_trainer_split(out),
                                              name="task_id")
        say_left_out("repair answers", left_out)
        if not left_out:
            kept = out
        Path(args.repairs).write_text(
            "".join(json.dumps(o) + "\n" for o in kept))
        print(f"  {len(attempts)} fixer attempts -> {len(kept)} repair pairs "
              f"-> {args.repairs}")


if __name__ == "__main__":
    main()
