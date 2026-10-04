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
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))

from solver_baseline import SOLVER_PROMPT, parse_invariants  # noqa: E402
from sft_data import split                                   # noqa: E402

CORPUS = HERE.parent / "extender" / "corpus.jsonl"
FIXER_LOG = HERE.parent / "extender" / "logs" / "fixer_attempts.jsonl"
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


def repair_pairs(attempts, rows_by_task):
    """Pair each failed fixer attempt with the answer that finally
    worked. Tasks that never got fixed contribute nothing: an unsolved
    failure is not a lesson."""
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
            if t is winner or t.get("verdict") == "NECESSARY":
                continue
            prompt = REPAIR_PROMPT.format(
                base=SOLVER_PROMPT.format(verilog=row["verilog"]),
                attempt=json.dumps({"invariants": t.get("invariants") or []}),
                diagnosis=t.get("diagnosis") or "the proof did not close",
                cti=t.get("cti_leg") or "(none recorded)")
            out.append({"prompt": prompt,
                        "completion": json.dumps(
                            {"invariants": winner.get("invariants") or []}),
                        "task_id": task})
    return out


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
    if args.out:
        Path(args.out).write_text(
            "".join(json.dumps({"prompt": p["prompt"],
                                "completion": p["completion"],
                                "holdout": p["holdout"]}) + "\n"
                    for p in pairs))
        print(f"  output {args.out}")
    else:
        print("  output: none (no --out given)")


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
    args = p.parse_args(argv)

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

    if args.reasoning:
        write_reasoning_file(args, rows)
    elif args.out:
        Path(args.out).write_text(
            "".join(json.dumps({"prompt": p["prompt"],
                                "completion": p["completion"]}) + "\n"
                    for p in pairs))
        print(f"  wrote {args.out}")

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
        Path(args.repairs).write_text(
            "".join(json.dumps(o) + "\n" for o in out))
        print(f"  {len(attempts)} fixer attempts -> {len(out)} repair pairs "
              f"-> {args.repairs}")


if __name__ == "__main__":
    main()
