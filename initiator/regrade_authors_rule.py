"""Re-grade saved one-try answers with the benchmark authors' rule.

Our one-try score (benchmark_solve.py) checks all of an answer's facts
together, so one false fact sinks an answer that also holds the right
one. The authors' scorer (large_lemma_miners scripts/evaluate_non_agentic.py
-> evaluate_lemmas -> src/evaluation.py evaluate_subsets) checks each fact
alone, drops the ones it refutes or cannot finish in 120 s, and tries the
rest together; repair_loop.evaluate_subsets copies it. The two rules can
differ either way (checked 31 Oct 2026), and the paper's numbers are on
theirs, so every saved answer gets both grades.

Reads the scoring log, keeps the one-try answers of the models asked for
(the last 3 complete runs of each set per model, as the job summaries
count them) and writes one row per answer as it finishes, so a stopped
job carries on where it left off:

  venv/bin/python initiator/regrade_authors_rule.py --dry
  venv/bin/python initiator/regrade_authors_rule.py --sets hard --workers 8 \\
      --out initiator/logs/regraded_seven_sets_authors_rule_2026-11-03.jsonl
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_loop as rl                                         # noqa: E402

BENCH_ROOT = HERE.parent.parent / "large_lemma_miners" / "benchmarks"
LOG = HERE / "logs" / "benchmark_solve.jsonl"
SIZES = {"hard": 31, "main_experiment": 78}
KEY = ("run_id", "set", "bench", "served_model")
# the seven training sets of 29 Oct 2026, by the names Rishab uses
LABELS = {"good_open_source_2026-10-29": "R1",
          "good_weak148_2026-10-29": "R2",
          "weak69_open_source_2026-10-29": "R3",
          "good_weak_2026-10-28": "R4",
          "good_weak244_open_source_2026-10-29": "R5",
          "good_weak67_wrong_open_source_2026-10-29": "R6",
          "good_weak_open_source_unlabeled_2026-10-29": "R7"}


def model_name(served_model):
    """The model's folder: .../runs/<name>/merged -> <name>."""
    if not served_model:
        return None
    p = Path(served_model)
    return p.parent.name if p.name == "merged" else p.name


def one_try_rows(rows, match):
    """One-try answers of the models whose folder holds `match`. The
    feedback-loop jobs write their rounds to the same log, marked by
    `round` and `feedback_mode`."""
    return [r for r in rows
            if "round" not in r and "feedback_mode" not in r
            and match in (model_name(r.get("served_model")) or "")]


def pick_runs(rows, sizes=SIZES, keep=3):
    """The last `keep` complete runs of each set for each model. A job
    bumped mid-pass leaves a run with files missing; its restart writes
    complete runs of its own, and the job summaries count only those."""
    def run_of(r):
        return model_name(r["served_model"]), r["set"], r["run_id"]
    files = defaultdict(set)
    for r in rows:
        files[run_of(r)].add(r["bench"])
    complete = defaultdict(list)
    for (model, set_, run_id), benches in files.items():
        if len(benches) >= sizes.get(set_, 0):
            complete[model, set_].append(run_id)
    kept = {(model, set_, run_id) for (model, set_), run_ids in
            complete.items() for run_id in sorted(run_ids)[-keep:]}
    return [r for r in rows if run_of(r) in kept]


def key_fields(r):
    return {k: r.get(k) for k in KEY}


def _facts(singletons):
    out, seen = [], set()
    for e in singletons:
        lemma = e["lemma"][0]
        if lemma in seen:
            continue
        seen.add(lemma)
        out.append({"lemma": lemma, "correct": rl.booleans(e)["correct"],
                    "correctness": e["correct"]["verdict"],
                    "one_induction": e["1-inductive"]["verdict"],
                    "with_prop": e["1-inductive with property"]["verdict"]})
    return out


def regrade_row(row, run, bench_root, workdir, workers=4):
    """One saved answer, graded both ways. An empty answer or a file EBMC
    cannot judge at all costs no checks."""
    g = dict(key_fields(row), model=model_name(row.get("served_model")),
             ours_verdict=row.get("verdict"),
             ours_solved=row.get("verdict") == "PROVEN",
             judgeable=row.get("file_judgeable", True) is not False,
             authors_solved=False, facts=[])
    lemmas = row.get("lemmas") or []
    if not g["judgeable"] or not lemmas:
        return g
    bench = Path(bench_root) / row["set"] / row["bench"]
    entries = rl.evaluate_subsets(bench, lemmas, workdir, run=run,
                                  workers=workers)
    g["authors_solved"] = any(rl.solves(e) for e in entries)
    g["facts"] = _facts([e for e in entries if len(e["lemma"]) == 1])
    return g


def grade_all(rows, out_path, run, bench_root, workdir, row_workers=1,
              workers=4):
    """Every row not already in `out_path`, appended as it finishes. A
    check already made on a design is not run again: the seven sets'
    models often write the same fact."""
    out_path = Path(out_path)
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                done[tuple(g.get(k) for k in KEY)] = g
    todo = [r for r in rows if tuple(r.get(k) for k in KEY) not in done]
    caches, lock = {}, threading.Lock()

    def cached_run(bench_path, lemmas, mode, workdir, buechi=False):
        with lock:
            design = caches.setdefault(str(bench_path), rl.remembered(run))
        return design(bench_path, lemmas, mode, workdir, buechi=buechi)

    def one(r):
        g = regrade_row(r, cached_run, bench_root, workdir, workers)
        with lock:
            with out_path.open("a") as out:
                out.write(json.dumps(g) + "\n")
        return g

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, row_workers)) as ex:
        new = list(ex.map(one, todo))
    return list(done.values()) + new


def summarize(graded):
    """model -> set -> per pass, in time order: answers solved on our
    rule, on theirs, and how many files EBMC could judge."""
    passes = defaultdict(lambda: defaultdict(
        lambda: {"ours": 0, "authors": 0, "judgeable": 0}))
    for g in graded:
        p = passes[g["model"], g["set"]][g["run_id"]]
        if g["judgeable"]:
            p["judgeable"] += 1
            p["ours"] += bool(g["ours_solved"])
            p["authors"] += bool(g["authors_solved"])
    out = defaultdict(dict)
    for (model, set_), runs in passes.items():
        order = sorted(runs)
        out[model][set_] = {k: [runs[r][k] for r in order]
                            for k in ("ours", "authors", "judgeable")}
    return dict(out)


def set_name(model):
    return re.sub(r"_seed\d+$", "", model)


def table(summary):
    """Each model's passes, then each set's average per pass."""
    lines = ["model / set / solved per pass: ours, then the authors' rule"]
    for model in sorted(summary):
        for set_ in sorted(summary[model]):
            d = summary[model][set_]
            lines.append(f"{model:50s} {set_:16s} ours {d['ours']}  "
                         f"authors {d['authors']}  (of {d['judgeable']})")
    sets = defaultdict(lambda: defaultdict(lambda: {"ours": [], "authors": []}))
    for model, by_set in summary.items():
        for set_, d in by_set.items():
            sets[set_name(model)][set_]["ours"] += d["ours"]
            sets[set_name(model)][set_]["authors"] += d["authors"]
    lines += ["", f"{'set':45s} {'hard ours':>9s} {'theirs':>7s} "
                  f"{'easy ours':>9s} {'theirs':>7s}"]
    for name in sorted(sets):
        cells = []
        for set_ in ("hard", "main_experiment"):
            d = sets[name].get(set_)
            cells += ([f"{mean(d['ours']):9.2f}", f"{mean(d['authors']):7.2f}"]
                      if d and d["ours"] else [f"{'-':>9s}", f"{'-':>7s}"])
        lines.append(f"{name:45s} {' '.join(cells)}  {LABELS.get(name, '')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", default=str(LOG), help="the scoring log")
    ap.add_argument("--out", default=str(
        HERE / "logs" / f"regraded_authors_rule_{date.today()}.jsonl"),
        help="one row per answer; a rerun with the same file carries on")
    ap.add_argument("--match", default="_seed",
                    help="models whose folder holds this (default: the "
                         "seven training sets of 29 Oct 2026)")
    ap.add_argument("--sets", nargs="+", default=sorted(SIZES),
                    choices=sorted(SIZES))
    ap.add_argument("--workers", type=int, default=4,
                    help="answers checked at once; each runs up to 4 EBMC "
                         "checks at once")
    ap.add_argument("--dry", action="store_true",
                    help="count the answers, check nothing")
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.log).read_text().splitlines()
            if l.strip()]
    rows = pick_runs([r for r in one_try_rows(rows, args.match)
                      if r["set"] in args.sets])
    runs = defaultdict(set)
    for r in rows:
        runs[model_name(r["served_model"]), r["set"]].add(r["run_id"])
    print(f"{len(rows)} answers, {len({m for m, _ in runs})} models")
    for (model, set_), run_ids in sorted(runs.items()):
        print(f"  {model:50s} {set_:16s} {len(run_ids)} runs")
    if args.dry:
        return
    if not os.access(rl.ebmc_eval.EBMC, os.X_OK):
        sys.exit(f"EBMC_PATH ({rl.ebmc_eval.EBMC}) is not a program - "
                 "export it first; nothing was checked")
    with tempfile.TemporaryDirectory() as workdir:
        graded = grade_all(rows, args.out, rl.run_mode, BENCH_ROOT, workdir,
                           row_workers=args.workers)
    print(table(summarize(graded)))


if __name__ == "__main__":
    main()
