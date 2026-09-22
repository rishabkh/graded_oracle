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

  venv/bin/python training/build_sft.py --out extender/sft_train.jsonl
  venv/bin/python training/build_sft.py --repairs extender/sft_repairs.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))

from solver_baseline import SOLVER_PROMPT                    # noqa: E402

CORPUS = HERE.parent / "extender" / "corpus.jsonl"
FIXER_LOG = HERE.parent / "extender" / "logs" / "fixer_attempts.jsonl"

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default=None, help="answers-only pairs, run 1")
    p.add_argument("--repairs", default=None,
                   help="failed-then-fixed pairs, run 2")
    args = p.parse_args()

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

    if args.out:
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
