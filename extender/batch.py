"""Step 7 — close the loop. A JSONL work queue and a while loop; not an
agenda framework.

Stopping conditions, per the directive:
- a branch stops extending when the main proof of its without-run
  reaches the checker's own limit (300 s; until 17 Oct 2026 it was 60 s
  on the slowest run, which the 120 s second opinion alone could cross),
  when the design passes 1000 lines, or after three consecutive failures
  on that branch;
- the run stops at a HARD CAP of LLM calls, because the loop is
  recursive and an unbounded one is how a budget disappears overnight.

Every attempt is logged with task_id + attempt number (format failures
retry the same task up to three attempts — the step-6 requirement, so
failed-vs-fixed pairs are future preference data). Accepted children
promote immediately and join the frontier, which is what makes the loop
recursive: children of children get extended in the same run.

  venv/bin/python extender/batch.py --dry --max-calls 20      # plan only
  venv/bin/python extender/batch.py --max-calls 20 --seed 1   # spend money
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from extend import (MOVE_WEIGHTS, SECOND_PROPERTY_RATE,          # noqa: E402
                    REPLICATE_SCHEMA, SECOND_SCHEMA, STRUCTURAL_SCHEMA,
                    COMPOSE_SCHEMA, PARTNER_SCHEMA, build_prompt,
                    call_model, compose_prompt, expose_for_wrapper,
                    grade_compose, grade_replicate, grade_step4,
                    keep_reasoning, partner_prompt, wrappable)
import llm_client                                                 # noqa: E402
from build_corpus import (extract_asserts, honest_lines,          # noqa: E402
                          without_main_s)
from distractor import (Spinner, dump, OUT_LOG, MODEL, EFFORT,    # noqa: E402
                        GRADE_KWARGS, build_prompt as distractor_prompt,
                        call_model as distractor_call, grade_extension)
from promote import (FIXER_QUEUE, children, history, lineage,     # noqa: E402
                     promote, ratchet, route, step_kind,
                     what_was_added)
import distractor                                                # noqa: E402
import gates                                                     # noqa: E402
import sense_check                                               # noqa: E402

import contextlib

# How many answers are checked at once. None (no limit) unless --batch sets
# it: a batch returns many answers together, and checking them all at the
# same time would push checks past their time limit and change verdicts.
GRADE_SLOTS = None

# The second-model review (sense_check.py) of every proved step; a step it
# rejects is not kept, so only designs it keeps are grown further (every
# step since 18 Oct 2026, combinations and copies before). Off until it has
# been tried on the hand-labelled trials; --sense-check.
SENSE_CHECK = False


def _grading():
    return GRADE_SLOTS if GRADE_SLOTS is not None else contextlib.nullcontext()


# with --workers > 1, main() sets Spinner.quiet (per-step spinners would
# draw over each other) and runs one pool-wide spinner fed by PROGRESS
PROGRESS = {"done": 0, "total": 0, "spinner": None}
_progress_lock = threading.Lock()

CORPUS = HERE / "corpus.jsonl"
QUEUE_LOG = HERE / "logs" / "batch_queue.jsonl"

# branch stop: the main proof without facts reached the checker's own
# limit (was 60 s on any run: 7 of 17 children stopped on 15 Oct, 4 of them
# only because the 120 s second opinion was slow)
MAX_WITHOUT_S = float(GRADE_KWARGS["timeout_s"])
MAX_LINES = 1000          # branch stop: about what one training example holds
BRANCH_FAILS = 3          # branch stop: three consecutive failures
MAX_GEN = 12              # was 6 (and 3 before it): real blocks are about
                          # 1,900 lines; ours start at about 70 (17 Oct 2026)
MAX_ATTEMPTS = 3          # per-task retries on format failures
MULTIPLIER_RATE = 0.25    # replicate/compose share when eligible (0.15, then
                          # 0.4 on 17 Oct; 0.25 since 18 Oct: they grow a
                          # design 2-4x a step, but gave 1 good design in 15)
DISTRACTOR_RATE = 1 / 6   # the control: irrelevant logic, invariants verbatim.
                          # The only move that lowers coi_ratio — without it
                          # every extension lands inside the cone and the
                          # metric drifts the wrong way with depth.
MAX_SAME_KIND = 2         # steps of one kind per family (18 Oct 2026: one
                          # family got five buffers in a row, all weak)
# 19 Oct 2026: the limit counted the step's label, not the job it does, so
# it also stopped a third sub-unit with a different job; real blocks have
# about 5 modules. The history in the question and the reviewer's own-job
# question guard against repeats for these two.
NO_LIMIT_KINDS = ("SUBUNIT", "STORAGE")
COPIES = False            # replicate: 2 good designs in 13, then 0 kept in 3
                          # in the 18 Oct good-growth trial (19 Oct 2026)
# combine: the model picks its partner from a short list of small starting
# designs, or says none fits (extend.partner_prompt, 19 Oct 2026)
PARTNER_CHOICES = 8
PARTNER_MAX_LINES = 200

# What the reviewer and the design checks turned down on each design in
# this run, for the next question about it (19 Oct 2026: the reviewer
# rejected the same credit pool three times on one design).
REJECTED = defaultdict(list)
_rejected_lock = threading.Lock()
# turned down for what the step does, not for how the answer was written
REMEMBERED = ("NONSENSE",) + gates.VERDICTS


def rejection_note(record):
    """`kind: what it added. Rejected: why` for a turned-down step."""
    kind = record.get("move") or record.get("ext_type")
    why = str(record.get("reason") or "").replace("sense check: ", "")
    return " ".join(f"{kind}: {what_was_added(record)} Rejected: "
                    f"{why[:300]}".split())


def without_wall(record):
    """Main proof wall time on the without leg — the number the branch
    stop reads (build_corpus.without_main_s)."""
    return without_main_s(record)


def extendable(row, without_s, max_gen=MAX_GEN):
    """`without_s` is the time just measured; a row carried into a new run
    has its own on file (metrics.without_main_s), so the stop holds across
    runs as well as within one."""
    if row.get("generation", 0) >= max_gen:
        return False
    if len(row["verilog"].splitlines()) > MAX_LINES:
        return False
    if without_s is None:
        without_s = (row.get("metrics") or {}).get("without_main_s")
    if without_s is not None and without_s > MAX_WITHOUT_S:
        return False
    return True


# Which hardware groups are combined (18 Oct 2026). Partners drawn from the
# whole corpus gave 0 good combinations out of 7 in the 17 Oct trial: a
# power sequencer feeding a DCT, a CRC checker or a random-number generator,
# a breakpoint unit feeding a multiplier. A design may now be combined only
# with one from its own group or a group it is commonly wired to; a
# judgement call over the catalog's 17 groups, read both ways.
_NEAR = {
    "memory": {"interface", "interconnect", "processor", "DSP",
               "video/image", "networking", "accelerators",
               "data integrity", "test/debug"},
    "interface": {"peripherals", "networking", "data integrity",
                  "interconnect", "crypto", "video/image"},
    "interconnect": {"processor", "networking", "test/debug",
                     "accelerators"},
    "processor": {"peripherals", "control", "arithmetic", "test/debug"},
    "DSP": {"arithmetic", "video/image", "accelerators"},
    "arithmetic": {"crypto", "accelerators"},
    "crypto": {"data integrity"},
    "data integrity": {"networking"},
    "clocking/reset/power": {"control", "peripherals"},
    "control": {"peripherals"},
}
GROUP_FIT = {}
for _g, _near in _NEAR.items():
    for _h in _near:
        GROUP_FIT.setdefault(_g, set()).add(_h)
        GROUP_FIT.setdefault(_h, set()).add(_g)
CATALOG = HERE.parent / "initiator" / "catalog" / \
    "kind_categories_2026-10-09.json"
_KIND_GROUP = None


def _kind_group(kind):
    global _KIND_GROUP
    if _KIND_GROUP is None:
        cat = json.loads(CATALOG.read_text()) if CATALOG.exists() else \
            {"areas": [], "kinds": []}
        group = {a["id"]: a["group"] for a in cat["areas"]}
        _KIND_GROUP = {k["name"].strip(): group.get(k["category"])
                       for k in cat["kinds"]}
    return _KIND_GROUP.get(kind)


def design_groups(row, by_id, _seen=None):
    """The hardware groups a design belongs to: its kind's group, or for a
    grown design the groups of the designs it was grown from."""
    seen = _seen if _seen is not None else set()
    if row is None or row["id"] in seen:
        return set()
    seen.add(row["id"])
    kind = (row.get("construct") or "").split(":", 1)[0].strip()
    if kind and _kind_group(kind):
        return {_kind_group(kind)}
    out = set()
    for pid in (row.get("parent"), row.get("parent2")):
        if pid:
            out |= design_groups(by_id.get(pid), by_id, seen)
    return out


def fits(a, b, by_id):
    """May these two designs be combined? Same group, or groups commonly
    wired together; never when either group is unknown."""
    ga, gb = design_groups(a, by_id), design_groups(b, by_id)
    return any(x == y or y in GROUP_FIT.get(x, ()) for x in ga for y in gb)


def copies_that_fit(n_facts):
    """How many copies keep the restated fact list (one renamed set per
    copy, plus the pool's own) within gates.MAX_FACTS."""
    return [n for n in (4, 6) if n * n_facts + 1 <= gates.MAX_FACTS]


_HONEST = {}


def _honest(row):
    """honest_lines of a row, remembered: the sampler asks it of every
    possible partner, every task."""
    key = (row.get("id"), hash(row["verilog"]))
    if key not in _HONEST:
        _HONEST[key] = honest_lines(row["verilog"])
    return _HONEST[key]


def kinds_used(row, by_id):
    """How many steps of each kind made this design (promote.step_kind)."""
    return Counter(step_kind(r) for r in lineage(row, by_id))


def sample_task(rng, parent, corpus_rows):
    """One weighted task for a parent, or None when its family has used
    every limited kind of step MAX_SAME_KIND times and the unlimited ones
    cannot be drawn. GUARD, SPLIT, BOUND and LEDGER are weight 0; the
    distractor runs at ~1-in-6; a combination only for a parent a wrapper
    can reach, after the port step (extend.wrappable), offering the model
    up to PARTNER_CHOICES small starting designs that fit (fits) while the
    facts stay within gates.MAX_FACTS; copies only with COPIES."""
    eligible_self = wrappable(parent)
    n_facts = len(parent["invariants"])
    by_id = {r["id"]: r for r in corpus_rows}
    used = kinds_used(parent, by_id)

    def allowed(kind):
        return kind in NO_LIMIT_KINDS or used[kind] < MAX_SAME_KIND

    partners = [r for r in corpus_rows
                if r["id"] != parent["id"]
                and r.get("generation", 0) == 0
                and _honest(r) <= PARTNER_MAX_LINES
                and r["top_module"] != parent["top_module"]
                and n_facts + len(r["invariants"]) + 1 <= gates.MAX_FACTS
                and wrappable(r) and fits(parent, r, by_id)] \
        if eligible_self and allowed("compose") else []
    copies = copies_that_fit(n_facts) \
        if COPIES and eligible_self and allowed("replicate") else []
    can_multiply = bool(partners or copies) and \
        rng.random() < MULTIPLIER_RATE
    if can_multiply and partners and (not copies or rng.random() < 0.5):
        shortlist = rng.sample(partners, min(PARTNER_CHOICES, len(partners)))
        return {"ext_type": "compose", "parent_id": parent["id"],
                "parent2_id": None, "move": None,
                "candidates": [r["id"] for r in shortlist]}
    if can_multiply:
        return {"ext_type": "replicate", "parent_id": parent["id"],
                "parent2_id": None, "move": None,
                "instances": rng.choice(copies)}
    if rng.random() < DISTRACTOR_RATE and allowed("distractor"):
        return {"ext_type": "distractor", "parent_id": parent["id"],
                "parent2_id": None, "move": None}
    if rng.random() < SECOND_PROPERTY_RATE and allowed("second"):
        return {"ext_type": "second", "parent_id": parent["id"],
                "parent2_id": None, "move": None}
    moves = [m for m, w in MOVE_WEIGHTS.items() if allowed(m)
             for _ in range(w)]
    # pipeline runs of several stages; a single stage was a small tweak
    # (18 Oct 2026)
    k = rng.choice([4, 4, 8])
    if not moves:
        rest = [t for t, on in (("distractor", DISTRACTOR_RATE),
                                ("second", SECOND_PROPERTY_RATE))
                if on and allowed(t)]
        if not rest:
            return None
        return {"ext_type": rng.choice(rest), "parent_id": parent["id"],
                "parent2_id": None, "move": None}
    move = rng.choice(moves)
    return {"ext_type": "structural", "parent_id": parent["id"],
            "parent2_id": None, "move": move,
            "k": k if move == "STAGE" else 1}


def largest_starts(corpus_rows, n, max_gen=MAX_GEN):
    """The n biggest starting designs that can still be extended, no two
    from the same hardware group, so a trial grows several kinds of
    hardware. 18 Oct 2026: good designs stayed close to their starting
    size, and the trials had started from designs of 62 to 84 lines.
    A design with no property the extender can read is skipped: 29 of the
    252 write `assert property (...)`, which extract_asserts does not
    read, and two of the five biggest are among them."""
    by_id = {r["id"]: r for r in corpus_rows}
    rows = sorted((r for r in corpus_rows if r.get("generation", 0) == 0
                   and r.get("property") and extendable(r, None, max_gen)),
                  key=lambda r: (-len(r["verilog"].splitlines()), r["id"]))
    out, taken = [], set()
    for r in rows:
        groups = design_groups(r, by_id)
        if groups & taken:
            continue
        taken |= groups
        out.append(r["id"])
        if len(out) == n:
            break
    return out


def run_loop(corpus_rows, executor, *, max_calls, max_gen=MAX_GEN,
             rng=None, on_task=None, on_fixer=None, workers=1,
             on_promote=None, order="shuffle", per_family=3,
             start_ids=None):
    """The while loop, now a worker pool. `executor(task, corpus_rows) ->
    record` is injectable so tests run without API or sby. `workers`
    threads share one frontier under one lock; the executor runs
    unlocked (model calls are network-bound, grading is subprocesses,
    so threads parallelise both). A call is charged when issued, so
    in-flight work never exceeds the hard cap. `on_promote(row)` fires
    under the lock for every accepted row — the checkpoint hook: a
    killed run keeps everything promoted before the kill.
    `start_ids` names the only rows the run starts from (every row stays
    a possible partner for a composition); order "chase" extends each
    accepted child next, so a lineage goes deep within one run (17 Oct
    2026). Returns promoted rows."""
    rng = rng or random.Random()
    corpus_rows = list(corpus_rows)
    if start_ids:
        by_id = {r["id"]: r for r in corpus_rows}
        missing = [i for i in start_ids if i not in by_id]
        if missing:
            raise ValueError(f"no such starting rows: {missing}")
        # in the order given
        start = [by_id[i] for i in start_ids
                 if extendable(by_id[i], None, max_gen)]
    else:
        start = [r for r in corpus_rows if extendable(r, None, max_gen)]
        if order == "depth":
            # deepest and rarest first, round-robin across families so one
            # lineage cannot absorb the run (see frontier.py)
            from frontier import order_frontier
            start = order_frontier(start, {r["id"]: r for r in corpus_rows},
                                   per_family=per_family)
        else:
            rng.shuffle(start)   # sample the corpus, not the oldest rows first
    frontier = deque(start)
    branch_fails = Counter()
    dead_branches = set()
    new_rows = []
    state = {"calls": 0, "task_seq": 0}
    lock = threading.Lock()

    def next_task():
        """Reserve budget and a parent, or None. Lock held by caller."""
        while frontier:
            if state["calls"] >= max_calls:
                return None
            parent = frontier.popleft()
            if parent["id"] in dead_branches:
                continue
            task = sample_task(rng, parent, corpus_rows)
            if task is None:          # its family has used every kind
                continue
            state["task_seq"] += 1
            task["task_id"] = state["task_seq"]
            task["attempt"] = 1
            return parent, task
        return None

    def settle(parent, task, record):
        """Route one graded record. Lock held by caller. Returns True
        when the same task should be retried (reformat)."""
        record["task_id"] = task["task_id"]
        record["attempt"] = task["attempt"]
        # here as well as in the executor, so a child no harder than its
        # parent always counts as a failed attempt; otherwise the parent
        # is put back as healthy and drawn until the budget runs out
        p2 = next((r for r in corpus_rows
                   if r["id"] == task.get("parent2_id")), None)
        ratchet(record, [parent, p2])
        action = route(record.get("verdict", "ERROR"))
        if action == "accept":
            _, promoted = promote(corpus_rows, [record])
            corpus_rows.extend(promoted)
            new_rows.extend(promoted)
            branch_fails[parent["id"]] = 0
            w = without_wall(record)
            for child in promoted:
                if on_promote:
                    on_promote(child)
                if extendable(child, w, max_gen):
                    if order == "chase":
                        frontier.appendleft(child)   # the newest goes next
                    else:
                        frontier.append(child)
            frontier.append(parent)      # a healthy parent stays live
            return False
        if action == "pass":             # nothing fit (NO_PARTNER): not a
            frontier.append(parent)      # failure of the design
            return False
        if action == "fixer" and on_fixer:
            on_fixer(record)      # the branch still counts the failure
        if action == "reformat" and task["attempt"] < MAX_ATTEMPTS \
                and state["calls"] < max_calls:
            task["attempt"] += 1
            return True
        branch_fails[parent["id"]] += 1
        if branch_fails[parent["id"]] >= BRANCH_FAILS:
            dead_branches.add(parent["id"])
        else:
            frontier.append(parent)
        return False

    def worker():
        while True:
            with lock:
                nxt = next_task()
                if nxt is None:
                    return
                parent, task = nxt
                state["calls"] += 1
                if on_task:
                    on_task(task)
            while True:
                record = executor(task, corpus_rows)
                with lock:
                    retry = settle(parent, task, record)
                    if retry:
                        state["calls"] += 1
                        if on_task:
                            on_task(task)
                    else:
                        break

    if workers <= 1:
        worker()
    else:
        threads = [threading.Thread(target=worker) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return new_rows


def finalize_distractor(record):
    """A distractor whose logic yosys swept is the parent with dead text:
    NECESSARY is true and worthless. Route it as a format failure so the
    loop retries with a fresh call instead of promoting a duplicate."""
    if record.get("verdict") == "NECESSARY" and not record.get("live"):
        record["verdict"] = "DEAD_LOGIC"
        record["error"] = "added logic did not survive yosys opt (state_bits unchanged)"
    return record


def _ready_for_wrapper(record, rows):
    """The rows a wrapper will instantiate, after the port step; the
    record keeps which internal registers got a mirror port."""
    ready = [expose_for_wrapper(r) for r in rows]
    for r, out in zip(rows, ready):
        if out is None:
            raise ValueError(f"{r['id']} cannot be wrapped: a fact needs a "
                             "memory or a function no port can mirror")
        if out.get("exposed"):
            record.setdefault("mirrored_ports", {})[r["id"]] = out["exposed"]
    return ready


def _choose_partner(task, record, parent, by_id):
    """Ask the model which candidate (if any) a real chip would wire to
    the parent (19 Oct 2026). Sets task["parent2_id"] and task["plan"], or
    the record's verdict: NO_PARTNER when none fits, a format verdict when
    the call fails."""
    cands = [by_id[i] for i in task.get("candidates", []) if i in by_id]
    label = (f"task {task['task_id']}.{task['attempt']} compose "
             f"{task['parent_id']}: choosing a partner")
    with Spinner(label):
        out, usage, stop = call_model(
            partner_prompt(parent, cands, by_id, history(parent, by_id)),
            PARTNER_SCHEMA)
    record["partner_usage"] = usage
    if out is None:
        record["verdict"] = ("REFUSED" if stop == "refusal"
                             else "TRUNCATED" if stop == "length"
                             else "UNPARSEABLE")
        return
    record["partner_choice"] = out
    if out.get("choice") not in {c["id"] for c in cands}:
        record["verdict"] = "NO_PARTNER"
        record["reason"] = out.get("system") or "no candidate fits"
        return
    task["parent2_id"] = out["choice"]
    task["plan"] = {"system": out.get("system", ""),
                    "connection": out.get("connection", "")}


def _real_executor(task, corpus_rows):
    by_id = {r["id"]: r for r in corpus_rows}
    parent = by_id[task["parent_id"]]
    record = {"extension_id": (time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
                               + f"_t{task['task_id']:03d}"),
              "ext_type": task["ext_type"], "move": task.get("move"),
              "k": task.get("k", 1), "parent_id": task["parent_id"],
              "parent2_id": task.get("parent2_id"),
              "batch_task": task["task_id"], "batch_attempt": task["attempt"],
              # who wrote this row: a corpus grown by two different models
              # is training data that cannot be reasoned about otherwise
              "model": llm_client.model_label(MODEL), "effort": EFFORT}
    if llm_client._gate is not None:
        record["batch"] = True            # sent at half price
    try:
        if task["ext_type"] == "distractor":
            record["invariants"] = parent["invariants"]     # verbatim
            label = (f"task {task['task_id']}.{task['attempt']} distractor "
                     f"on {task['parent_id']}")
            with Spinner(f"{label}: model writing"):
                out, usage, stop = distractor_call(distractor_prompt(parent))
            record["usage"] = usage
            if out is None:
                record["verdict"] = ("REFUSED" if stop == "refusal"
                                     else "TRUNCATED" if stop == "length"
                                     else "UNPARSEABLE")
            else:
                record["reasoning"] = out["note"]   # kept under the old key
                with _grading():
                    finalize_distractor(grade_extension(parent, out["patch"],
                                                        record))
        elif task["ext_type"] == "compose":
            if not task.get("parent2_id"):
                _choose_partner(task, record, parent, by_id)
            if "verdict" not in record:
                record["parent2_id"] = task["parent2_id"]
                parent, p2 = _ready_for_wrapper(
                    record, [parent, by_id[task["parent2_id"]]])
                prompt = compose_prompt(parent, p2, task.get("plan"))
                label = (f"task {task['task_id']}.{task['attempt']} compose "
                         f"{task['parent_id']}+{task['parent2_id']}")
                with Spinner(f"{label}: model writing"):
                    out, usage, stop = call_model(prompt, COMPOSE_SCHEMA)
                record["usage"] = usage
                if out is None:
                    record["verdict"] = ("REFUSED" if stop == "refusal"
                                         else "TRUNCATED" if stop == "length"
                                         else "UNPARSEABLE")
                else:
                    keep_reasoning(out, record)
                    with _grading():
                        grade_compose(parent, p2, out, record)
        else:
            if task["ext_type"] == "replicate":
                parent, = _ready_for_wrapper(record, [parent])
            with _rejected_lock:
                rejected = list(REJECTED.get(parent["id"], []))
            prompt = build_prompt(parent, task["ext_type"], task.get("move"),
                                  n=task.get("instances", 6),
                                  k=task.get("k", 1),
                                  history=history(parent, by_id),
                                  siblings=children(parent, by_id),
                                  rejected=rejected)
            schema = {"structural": STRUCTURAL_SCHEMA,
                      "second": SECOND_SCHEMA,
                      "replicate": REPLICATE_SCHEMA}[task["ext_type"]]
            label = (f"task {task['task_id']}.{task['attempt']} "
                     f"{task['ext_type']} {task.get('move') or ''} "
                     f"on {task['parent_id']}")
            with Spinner(f"{label}: model writing"):
                out, usage, stop = call_model(prompt, schema)
            record["usage"] = usage
            if out is None:
                record["verdict"] = ("REFUSED" if stop == "refusal"
                                     else "TRUNCATED" if stop == "length"
                                     else "UNPARSEABLE")
            else:
                keep_reasoning(out, record)
                with _grading():
                    if task["ext_type"] == "replicate":
                        grade_replicate(parent, out, record,
                                        task.get("instances", 6))
                    else:
                        grade_step4(parent, task["ext_type"], out, record)
    except Exception as exc:
        record["verdict"] = "ERROR"
        record["error"] = f"{type(exc).__name__}: {exc}"
    # before the log line, so the log and the terminal show the real verdict
    by_id = {r["id"]: r for r in corpus_rows}
    # a big step with no new kind of fact may pass only if the reviewer
    # reads it, so the pass is offered only when the reviewer is on
    ratchet(record, [by_id.get(task["parent_id"]),
                     by_id.get(task.get("parent2_id"))],
            review_waiver=SENSE_CHECK)
    if SENSE_CHECK:
        sense_check.review(record, history=history(
            by_id.get(task["parent_id"]), by_id))
    if record.get("verdict") in REMEMBERED:
        with _rejected_lock:
            REJECTED[task["parent_id"]].append(rejection_note(record))
    dump(record)
    with _progress_lock:
        PROGRESS["done"] += 1
        if PROGRESS["spinner"]:
            PROGRESS["spinner"].label = (f"batch: {PROGRESS['done']}/"
                                         f"{PROGRESS['total']} calls done")
    clear = "\r\033[K" if sys.stdout.isatty() else ""
    print(f"{clear}  task {task['task_id']}.{task['attempt']} "
          f"{task['ext_type']:10s} on {task['parent_id']}"
          + (f"+{task['parent2_id']}" if task.get("parent2_id") else "")
          + f" -> {record.get('verdict')}")
    return record


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--max-calls", type=int, required=True,
                   help="HARD cap on LLM calls this run")
    p.add_argument("--max-gen", type=int, default=MAX_GEN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel workers (threads); model calls are "
                        "network-bound and grading is subprocesses, so "
                        "N workers is close to N-fold")
    p.add_argument("--order", default="shuffle",
                   choices=["shuffle", "depth", "chase"],
                   help="shuffle samples the whole corpus; depth draws the "
                        "deepest and rarest rows first, round-robin across "
                        "families, which is what grows generation 2 and 3; "
                        "chase extends every accepted child next, so a "
                        "lineage grows many rounds in one run")
    p.add_argument("--start", default=None,
                   help="comma-separated row ids to start from (the only "
                        "lineages the run grows; every row can still be a "
                        "partner for a composition)")
    p.add_argument("--start-largest", type=int, default=None, metavar="N",
                   help="start from the N biggest starting designs, one "
                        "per hardware group (largest_starts); instead of "
                        "--start")
    p.add_argument("--per-family", type=int, default=3,
                   help="frontier entries one lineage may contribute, with "
                        "--order depth. Raise it to drive a single lineage "
                        "deep, which is the only way to see whether size "
                        "keeps growing past the generations we have")
    p.add_argument("--corpus", type=Path, default=CORPUS,
                   help="corpus to read and append to. Point it at a copy "
                        "to run a probe without writing into the corpus the "
                        "training file is built from")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="write the extension, queue and fixer logs here "
                        "instead of extender/logs; required with any "
                        "--corpus other than extender/corpus.jsonl")
    p.add_argument("--batch", action="store_true",
                   help="send the workers' calls as half-price Message "
                        "Batches (llm_client.batch_gate); --workers sets how "
                        "many requests can wait in one batch")
    p.add_argument("--grade-slots", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2),
                   help="with --batch: answers checked at the same time")
    p.add_argument("--flush-after", type=float, default=30.0,
                   help="with --batch: send the waiting requests once no new "
                        "one has arrived for this many seconds")
    p.add_argument("--dry", action="store_true",
                   help="print the planned first tasks, call nothing")
    p.add_argument("--sense-check", action="store_true",
                   help="have a second model review every proved step "
                        "(sense_check.py) and keep only what it keeps; "
                        "paid, one call per proved step")
    args = p.parse_args(argv)
    if args.start and args.start_largest:
        p.error("--start and --start-largest cannot be used together")
    # 8 Oct 2026: --corpus used to change only the corpus. The logs stayed
    # v2's, so a later promote or fix run could copy another corpus's
    # designs into v2's corpus under clashing names.
    global QUEUE_LOG, FIXER_QUEUE, GRADE_SLOTS, SENSE_CHECK
    SENSE_CHECK = args.sense_check
    if not args.dry and args.log_dir is None and \
            args.corpus.resolve() != CORPUS.resolve():
        sys.exit("a corpus other than extender/corpus.jsonl needs its own "
                 "--log-dir; nothing was run")
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        distractor.OUT_LOG = args.log_dir / "extensions.jsonl"
        QUEUE_LOG = args.log_dir / "batch_queue.jsonl"
        FIXER_QUEUE = args.log_dir / "fixer_queue.jsonl"

    corpus_rows = [json.loads(l)
                   for l in args.corpus.read_text().splitlines() if l.strip()]
    # 29 of the 252 starting designs write `assert property (...)`, which
    # was not read when the corpus was built, so they carry no property;
    # read it now (19 Oct 2026)
    for row in corpus_rows:
        if not row.get("property") and row.get("verilog"):
            row["property"] = extract_asserts(row["verilog"])
    rng = random.Random(args.seed)
    if args.start_largest:
        args.start = ",".join(largest_starts(corpus_rows, args.start_largest,
                                             args.max_gen))
        print("starting from the biggest designs, one per hardware group: "
              + ", ".join(
                  f"{i} ({len(r['verilog'].splitlines())} lines)"
                  for i in args.start.split(",") for r in corpus_rows
                  if r["id"] == i))

    if args.dry:
        frontier = [r for r in corpus_rows
                    if extendable(r, None, args.max_gen)]
        if args.start:
            wanted = args.start.split(",")
            frontier = [r for i in wanted for r in frontier if r["id"] == i]
        else:
            rng.shuffle(frontier)
        print(f"frontier: {len(frontier)} extendable rows; "
              f"first {min(args.max_calls, 15)} planned tasks:")
        for parent in frontier[:min(args.max_calls, 15)]:
            t = sample_task(rng, parent, corpus_rows)
            print(f"  {t['ext_type']:10s} {t.get('move') or '':6s} "
                  f"on {parent['id']} ({parent['top_module']})"
                  + (f" + {t['parent2_id']}" if t.get("parent2_id") else ""))
        return

    def log_task(task):
        QUEUE_LOG.parent.mkdir(exist_ok=True)
        with QUEUE_LOG.open("a") as f:
            f.write(json.dumps(task) + "\n")

    queued = set()
    if FIXER_QUEUE.exists():
        queued = {json.loads(l).get("extension_id")
                  for l in FIXER_QUEUE.read_text().splitlines()}

    def to_fixer_queue(record):
        if record.get("extension_id") in queued:
            return
        queued.add(record.get("extension_id"))
        FIXER_QUEUE.parent.mkdir(exist_ok=True)
        with FIXER_QUEUE.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    if args.workers > 1:
        Spinner.quiet = True
        PROGRESS["total"] = args.max_calls
        PROGRESS["spinner"] = Spinner(f"batch: 0/{args.max_calls} calls done",
                                      always=True)

    def checkpoint(row):
        # append immediately, under run_loop's lock: a killed run keeps
        # every row promoted before the kill
        with args.corpus.open("a") as f:
            f.write(json.dumps(row) + "\n")

    gate = contextlib.nullcontext()
    if args.batch:
        GRADE_SLOTS = threading.BoundedSemaphore(args.grade_slots)
        gate = llm_client.batch_gate(
            flush_after_s=args.flush_after, poll_s=60,
            manifest_dir=(args.log_dir or QUEUE_LOG.parent) / "batches")
    try:
        with llm_client.call_log((args.log_dir or QUEUE_LOG.parent)
                                 / "calls.jsonl"), \
                gate, PROGRESS["spinner"] or contextlib.nullcontext():
            new_rows = run_loop(corpus_rows, _real_executor,
                                max_calls=args.max_calls,
                                max_gen=args.max_gen, rng=rng,
                                on_task=log_task, on_fixer=to_fixer_queue,
                                workers=args.workers, on_promote=checkpoint,
                                order=args.order, per_family=args.per_family,
                                start_ids=(args.start.split(",")
                                           if args.start else None))
    finally:
        GRADE_SLOTS = None
    print(f"\nrun complete: {len(new_rows)} promoted "
          f"(corpus now {len(corpus_rows) + len(new_rows)} rows); "
          f"log: {distractor.OUT_LOG}")


if __name__ == "__main__":
    main()
