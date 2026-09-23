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
    args = p.parse_args()

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
    solved = 0
    for i, f in enumerate(files):
        rec = {"run_id": run_id, "solver": args.solver, "bench": f.name,
               "set": args.set, "served_model": serving,
               "ebmc_timeout_s": ebmc_timeout(),
               # a file that errors with no lemmas is nobody's failure
               "file_judgeable": judgeable(f)}
        t0 = time.monotonic()
        try:
            with Spinner(f"[{i}] {args.solver} on {f.stem}"):
                lemmas, err = solve(PROMPT.format(verilog=f.read_text()))
        except Exception as exc:
            lemmas, err = None, f"{type(exc).__name__}: {exc}"
        rec["solve_wall_s"] = round(time.monotonic() - t0, 2)
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
        OUT_LOG.parent.mkdir(exist_ok=True)
        with OUT_LOG.open("a") as out:
            out.write(json.dumps(rec) + "\n")
    print(f"\nsolved {solved}/{len(files)} "
          f"({100 * solved / max(len(files), 1):.0f}%)")


if __name__ == "__main__":
    main()
