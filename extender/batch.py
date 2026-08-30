"""Step 7 — close the loop. A JSONL work queue and a while loop; not an
agenda framework.

Stopping conditions, per the directive:
- a branch stops extending when its without-run exceeds 60 seconds
  (grading is about to stop being free), when the design passes 1000
  lines, or after three consecutive failures on that branch;
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
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from extend import (MOVE_WEIGHTS, SECOND_PROPERTY_RATE,          # noqa: E402
                    REPLICATE_SCHEMA, SECOND_SCHEMA, STRUCTURAL_SCHEMA,
                    COMPOSE_SCHEMA, COMPOSE_TEMPLATE, build_prompt,
                    call_model, compose_hidden_signals, grade_compose,
                    grade_replicate, grade_step4)
from extend_one import Spinner, dump, OUT_LOG                    # noqa: E402
from promote import promote, route                               # noqa: E402

CORPUS = HERE / "corpus.jsonl"
QUEUE_LOG = HERE / "logs" / "batch_queue.jsonl"

MAX_WITHOUT_S = 60.0      # branch stop: grading about to get expensive
MAX_LINES = 1000          # branch stop: design too big to read
BRANCH_FAILS = 3          # branch stop: three consecutive failures
MAX_GEN = 3               # depth of the ratchet for this phase
MAX_ATTEMPTS = 3          # per-task retries on format failures
MULTIPLIER_RATE = 0.15    # replicate/compose share when eligible


def without_wall(record):
    """Longest prove-run wall time on the without leg — the number the
    60-second branch stop reads."""
    runs = ((record.get("result") or {}).get("without_invariants")
            or {}).get("runs") or []
    times = [r.get("duration_s") for r in runs
             if r.get("mode") == "prove" and r.get("duration_s") is not None]
    return max(times) if times else None


def extendable(row, without_s, max_gen=MAX_GEN):
    if row.get("generation", 0) >= max_gen:
        return False
    if len(row["verilog"].splitlines()) > MAX_LINES:
        return False
    if without_s is not None and without_s > MAX_WITHOUT_S:
        return False
    return True


def sample_task(rng, parent, corpus_rows):
    """One weighted task for a parent. GUARD is weight 0; second-property
    runs at ~1-in-6; multipliers only between eligible parents."""
    eligible_self = not compose_hidden_signals(parent)
    partners = [r for r in corpus_rows
                if r["id"] != parent["id"]
                and r["top_module"] != parent["top_module"]
                and not compose_hidden_signals(r)]
    can_multiply = eligible_self and rng.random() < MULTIPLIER_RATE
    if can_multiply and partners and rng.random() < 0.5:
        partner = rng.choice(partners)
        return {"ext_type": "compose", "parent_id": parent["id"],
                "parent2_id": partner["id"], "move": None}
    if can_multiply:
        return {"ext_type": "replicate", "parent_id": parent["id"],
                "parent2_id": None, "move": None,
                "instances": rng.choice([4, 6])}
    if rng.random() < SECOND_PROPERTY_RATE:
        return {"ext_type": "second", "parent_id": parent["id"],
                "parent2_id": None, "move": None}
    moves = [m for m, w in MOVE_WEIGHTS.items() for _ in range(w)]
    k = rng.choice([1, 1, 1, 4])      # occasional magnitude on STAGE
    move = rng.choice(moves)
    return {"ext_type": "structural", "parent_id": parent["id"],
            "parent2_id": None, "move": move,
            "k": k if move == "STAGE" else 1}


def run_loop(corpus_rows, executor, *, max_calls, max_gen=MAX_GEN,
             rng=None, on_task=None):
    """The while loop. `executor(task, corpus_rows) -> record` is
    injectable so tests run without API or sby. Returns promoted rows."""
    rng = rng or random.Random()
    corpus_rows = list(corpus_rows)
    start = [r for r in corpus_rows if extendable(r, None, max_gen)]
    rng.shuffle(start)   # sample the whole corpus, not the oldest rows first
    frontier = deque(start)
    branch_fails = Counter()
    dead_branches = set()
    new_rows, calls, task_seq = [], 0, 0

    while calls < max_calls and frontier:
        parent = frontier.popleft()
        if parent["id"] in dead_branches:
            continue
        task = sample_task(rng, parent, corpus_rows)
        task_seq += 1
        task["task_id"] = task_seq
        task["attempt"] = 1

        while calls < max_calls:
            if on_task:
                on_task(task)
            record = executor(task, corpus_rows)
            calls += 1
            record["task_id"] = task["task_id"]
            record["attempt"] = task["attempt"]
            action = route(record.get("verdict", "ERROR"))

            if action == "accept":
                _, promoted = promote(corpus_rows, [record])
                corpus_rows.extend(promoted)
                new_rows.extend(promoted)
                branch_fails[parent["id"]] = 0
                w = without_wall(record)
                for child in promoted:
                    if extendable(child, w, max_gen):
                        frontier.append(child)
                frontier.append(parent)      # a healthy parent stays live
                break
            if action == "reformat" and task["attempt"] < MAX_ATTEMPTS:
                task["attempt"] += 1
                continue
            branch_fails[parent["id"]] += 1
            if branch_fails[parent["id"]] >= BRANCH_FAILS:
                dead_branches.add(parent["id"])
            else:
                frontier.append(parent)
            break

    return new_rows


def _real_executor(task, corpus_rows):
    by_id = {r["id"]: r for r in corpus_rows}
    parent = by_id[task["parent_id"]]
    record = {"extension_id": time.strftime("%Y-%m-%d_%Hh%Mm%Ss"),
              "ext_type": task["ext_type"], "move": task.get("move"),
              "k": task.get("k", 1), "parent_id": task["parent_id"],
              "parent2_id": task.get("parent2_id"),
              "batch_task": task["task_id"], "batch_attempt": task["attempt"]}
    try:
        if task["ext_type"] == "compose":
            p2 = by_id[task["parent2_id"]]
            prompt = COMPOSE_TEMPLATE.format(
                verilog_a=parent["verilog"], verilog_b=p2["verilog"],
                property_a="; ".join(parent["property"]),
                property_b="; ".join(p2["property"]),
                invariants_a="; ".join(parent["invariants"]) or "(none)",
                invariants_b="; ".join(p2["invariants"]) or "(none)")
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
                grade_compose(parent, p2, out, record)
        else:
            prompt = build_prompt(parent, task["ext_type"], task.get("move"),
                                  n=task.get("instances", 6),
                                  k=task.get("k", 1))
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
            elif task["ext_type"] == "replicate":
                grade_replicate(parent, out, record,
                                task.get("instances", 6))
            else:
                grade_step4(parent, task["ext_type"], out, record)
    except Exception as exc:
        record["verdict"] = "ERROR"
        record["error"] = f"{type(exc).__name__}: {exc}"
    dump(record)
    print(f"  task {task['task_id']}.{task['attempt']} "
          f"{task['ext_type']:10s} on {task['parent_id']}"
          + (f"+{task['parent2_id']}" if task.get("parent2_id") else "")
          + f" -> {record.get('verdict')}")
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-calls", type=int, required=True,
                   help="HARD cap on LLM calls this run")
    p.add_argument("--max-gen", type=int, default=MAX_GEN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry", action="store_true",
                   help="print the planned first tasks, call nothing")
    args = p.parse_args()

    corpus_rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    rng = random.Random(args.seed)

    if args.dry:
        frontier = [r for r in corpus_rows
                    if extendable(r, None, args.max_gen)]
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

    new_rows = run_loop(corpus_rows, _real_executor,
                        max_calls=args.max_calls, max_gen=args.max_gen,
                        rng=rng, on_task=log_task)
    with CORPUS.open("a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nrun complete: {len(new_rows)} promoted "
          f"(corpus now {len(corpus_rows) + len(new_rows)} rows); "
          f"log: {OUT_LOG.name}")


if __name__ == "__main__":
    main()
