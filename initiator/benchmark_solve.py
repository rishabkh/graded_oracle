"""Run a solver model against the large_lemma_miners benchmarks.

Per file: show the model the design (it already contains the property),
ask for strengthening lemmas as plain boolean expressions, inject them
through the EBMC adapter, and judge with THEIR solved criterion
(one_inductive_with_prop: lemmas AND prop 1-inductive, 120s).

  venv/bin/python initiator/benchmark_solve.py --solver opus --set hard --n 5 --dry
  venv/bin/python initiator/benchmark_solve.py --solver opus --set hard --n 5
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from ebmc_eval import (EBMC, TIMEOUT_S as ebmc_timeout_s, judgeable,
                       run as ebmc_run)


def ebmc_timeout():
    return ebmc_timeout_s
from solver_baseline import INV_SCHEMA, Spinner, solve_opus, solve_qwen

BENCH_ROOT = HERE.parent.parent / "large_lemma_miners" / "benchmarks"
OUT_LOG = HERE / "logs" / "benchmark_solve.jsonl"

PROMPT = """\
Here is a SystemVerilog design. It contains one property (the `prop`
block) which is true of the design but NOT provable by 1-induction on
its own.

{verilog}

Find strengthening lemma(s): facts about the design's reachable states
that hold in every reachable state and, asserted together with prop,
make the set 1-inductive.

Reply with JSON: {{"invariants": ["<expr>", ...]}} where each entry is a
plain SystemVerilog boolean expression over the design's signals - the
harness adds the clocking itself. No implication arrows, no prose;
sized constants; complete expressions only. NO system functions at all
($past, $countones, $onehot, ...) - the checker does not implement
them; write a popcount as an explicit sum instead:
busy[0] + busy[1] + ... + busy[N].

Scope: your lemmas are inserted next to the file's own property, so use
only names visible THERE. A signal inside another module is not visible
by a dotted path from that scope - if the property's module sees the
value through a port, use the port name.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["opus", "qwen"], required=True)
    p.add_argument("--set", default="hard",
                   choices=["hard", "main_experiment"])
    p.add_argument("--n", type=int, default=5, help="how many files")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--rounds", type=int, default=0,
                   help="0: one answer, as always. N: the benchmark "
                        "authors' repair loop, up to N answers in all "
                        "(theirs is 5); see repair_loop.py")
    p.add_argument("--show-cex", action="store_true",
                   help="repair feedback shows the run that breaks an "
                        "incorrect lemma (their show_cex option)")
    p.add_argument("--ebmc-workers", type=int, default=4,
                   help="EBMC checks run at once during repair")
    p.add_argument("--no-feedback", action="store_true",
                   help="the repair loop's control: between answers the "
                        "model hears only that the design is not solved yet "
                        "(repair_loop.NO_FEEDBACK); all else is the same")
    args = p.parse_args()
    if args.rounds and args.solver != "qwen":
        sys.exit("--rounds needs --solver qwen (a served model); nothing "
                 "was run")
    if args.no_feedback and not args.rounds:
        sys.exit("--no-feedback needs --rounds; nothing was run")
    if args.no_feedback and args.show_cex:
        sys.exit("--show-cex has nothing to show with --no-feedback; "
                 "nothing was run")

    files = sorted((BENCH_ROOT / args.set).glob("*.sv"))[:args.n]
    print(f"{len(files)} benchmark file(s) from {args.set}/")
    if args.dry:
        for f in files:
            print(f"  {f.name:32s} {len(f.read_text().splitlines())} lines")
        return
    if not os.access(EBMC, os.X_OK) and not os.environ.get("EBMC_PATH"):
        sys.exit("EBMC_PATH not set - export it first, nothing was called")

    solve = solve_opus if args.solver == "opus" else solve_qwen
    # which model actually answered: the base model and the fine-tune are
    # both served under the alias "llm", so the alias cannot tell a
    # before-run from an after-run
    serving = None
    if args.solver == "qwen":
        from openai import OpenAI
        from riscv_score import served_model
        serving = served_model(OpenAI(
            base_url=os.environ["QWEN_BASE_URL"],
            api_key=os.environ.get("QWEN_API_KEY", "none")).models.list())
        print(f"endpoint is serving: {serving or 'unknown'}")
    run_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    solved = repaired = 0
    for i, f in enumerate(files):
        rec = {"run_id": run_id, "solver": args.solver, "bench": f.name,
               "set": args.set, "served_model": serving,
               "ebmc_timeout_s": ebmc_timeout(),
               # a file that errors with no lemmas is nobody's failure
               "file_judgeable": judgeable(f)}
        if args.rounds:
            # both versions can run in one job, so each row says which
            rec["feedback_mode"] = "none" if args.no_feedback else "per_lemma"
        t0 = time.monotonic()
        # cleared first, so a call that dies early cannot leave the
        # previous file's reply recorded against this one
        solve.last_raw = solve.last_finish = None
        try:
            with Spinner(f"[{i}] {args.solver} on {f.stem}"):
                lemmas, err = solve(PROMPT.format(verilog=f.read_text()))
        except Exception as exc:
            lemmas, err = None, f"{type(exc).__name__}: {exc}"
        rec["solve_wall_s"] = round(time.monotonic() - t0, 2)
        # the whole reply, not just the invariants pulled out of it: v3's
        # answers carry reasoning, and a score nobody can read back to
        # what the model wrote cannot be diagnosed
        rec["raw"] = getattr(solve, "last_raw", None)
        rec["finish"] = getattr(solve, "last_finish", None)
        if lemmas is None:
            rec["verdict"] = "NO_ANSWER"
            rec["error"] = str(err)[:300]
        else:
            rec["lemmas"] = lemmas
            with tempfile.TemporaryDirectory() as d:
                with Spinner(f"[{i}] ebmc judging {f.stem}"):
                    res = ebmc_run(f, lemmas, "one_inductive_with_prop", d)
            rec["verdict"] = res["verdict"]
            rec["ebmc_time"] = res["time"]
            solved += res["verdict"] == "PROVEN"
        print(f"[{i}] {f.stem:28s} {rec['verdict']:10s} "
              f"({rec['solve_wall_s']}s)")
        if not args.rounds:
            write_row(rec)
            continue
        repaired += repair_file(f, rec, lemmas, args, i)
    print(f"\nsolved {solved}/{len(files)} "
          f"({100 * solved / max(len(files), 1):.0f}%)")
    if args.rounds:
        print(f"after the repair loop (up to {args.rounds} answers"
              f"{', no feedback' if args.no_feedback else ''}): "
              f"{repaired}/{len(files)} "
              f"({100 * repaired / max(len(files), 1):.0f}%)")


def write_row(row):
    OUT_LOG.parent.mkdir(exist_ok=True)
    with OUT_LOG.open("a") as out:
        out.write(json.dumps(row) + "\n")


def make_ask():
    """The served model, asked with a whole conversation: solve_qwen's
    client and model, a reply budget that fits the window."""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["QWEN_BASE_URL"],
                    api_key=os.environ.get("QWEN_API_KEY", "none"))

    def ask(messages, max_tokens):
        for _ in range(3):
            resp = client.chat.completions.create(
                model=os.environ["QWEN_MODEL"], max_tokens=max_tokens,
                messages=messages)
            text = resp.choices[0].message.content or ""
            fin = resp.choices[0].finish_reason
            if fin != "error":
                break
            time.sleep(2)
        return text, fin
    return ask


def repair_file(f, rec, lemmas, args, i):
    """Round 0 is the row above; the authors' loop takes it from there.
    Returns 1 if the file ends solved, by one shot or by repair."""
    import repair_loop
    if rec["verdict"] == "PROVEN":
        write_row(dict(rec, round=0, solved=True, final=True,
                       stop_reason="solved", solved_round=0))
        return 1
    if rec.get("raw") is None:            # the call itself failed
        write_row(dict(rec, round=0, solved=False, final=True,
                       stop_reason="prompt_error", solved_round=None))
        return 0
    fields = {k: rec[k] for k in ("run_id", "solver", "bench", "set",
                                  "served_model", "ebmc_timeout_s",
                                  "file_judgeable", "feedback_mode")}

    def log(row):
        if row["round"] == 0:
            extra = {k: v for k, v in row.items()
                     if k not in ("raw", "finish", "lemmas", "solve_wall_s")}
            write_row(dict(rec, **extra))
        else:
            write_row(dict(fields, **row,
                           verdict="PROVEN" if row["solved"] else "NOT_SOLVED"))
    with tempfile.TemporaryDirectory() as d:
        with Spinner(f"[{i}] repairing {f.stem}"):
            out = repair_loop.repair(
                f, PROMPT.format(verilog=f.read_text()), rec["raw"], lemmas,
                make_ask(), rounds=args.rounds, show_cex=args.show_cex,
                workdir=d, log=log, workers=args.ebmc_workers,
                give_feedback=not args.no_feedback)
    print(f"[{i}] {f.stem:28s} repair: {out['stop_reason']} after "
          f"{out['answers']} answer(s)"
          + (f", solved at round {out['solved_round']}" if out["solved"]
             else ""))
    return int(out["solved"])


if __name__ == "__main__":
    main()
