"""Which corpus answers hold up under the textbook induction rule, and
which of their facts that rule needs.

Our corpus accepted an answer if the property plus the facts are
k-inductive at depth 20 (oracle.grade). The benchmark judges with depth 1
(EBMC --k-induction --bound 1), the textbook definition of an inductive
invariant and a stricter rule: measured 6 Oct 2026, 207 of 665 corpus
answers pass at 20 and fail at 1, so about a third of what v1 and v2
learned from is "true but too weak" by the benchmark's own standard.

For every row this records: the full answer at depth 20 (the control,
dropped=-1), the full answer at depth 1 (dropped=-2), and the answer
without each fact in turn at depth 1 (dropped=i). A fact whose removal
breaks the depth-1 proof is NEEDED; the proof tool's counterexample for
that break is kept, ready for preference pairs and repair examples.

Only the induction step runs. Every set tried is a subset of an answer
already proven true, so the base case cannot fail, and the grader's
full run spent its whole 120 s on the base case (g0_000, 6 Oct 2026).
An induction-only sby run always ends "DONE (UNKNOWN)" because the base
case never ran; the verdict is the engine's "returned ... for induction"
line.

  source ~/Desktop/HarvardResearch/oss-cad-suite/environment
  venv/bin/python training/one_step.py              # writes extender/logs/one_step.jsonl
  venv/bin/python training/one_step.py --report
"""
import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from oracle.inject import inject_invariants                    # noqa: E402
from oracle.trace import summarize_vcd                         # noqa: E402

CORPUS = HERE.parent / "extender" / "corpus.jsonl"
OUT = HERE.parent / "extender" / "logs" / "one_step.jsonl"
TIMEOUT_S = 120
TRACE_CAP = 3000


def _folder(work, name):
    """A kept proof folder (8 Oct 2026: they used to be deleted after
    every check). `name` makes it findable: <row id>_d<dropped fact>."""
    return Path(tempfile.mkdtemp(dir=work, prefix=f"{name}_" if name else None))


def _sby(td):
    """Run sby in `td` and keep what it printed next to its log. Returns
    (stdout, timed out)."""
    try:
        r = subprocess.run(["sby", "-f", "job.sby"], cwd=td,
                           capture_output=True, text=True,
                           timeout=TIMEOUT_S * 1.5 + 10)
        out, err, hung = r.stdout or "", r.stderr or "", False
    except subprocess.TimeoutExpired as e:
        out, err, hung = _text(e.stdout), _text(e.stderr), True
    (td / "sby_stdout.txt").write_text(out)
    (td / "sby_stderr.txt").write_text(err)
    return out, hung


def _text(x):
    if x is None:
        return ""
    return x.decode(errors="replace") if isinstance(x, bytes) else x


def prove(row, invariants, work, depth=1, name=None):
    """Induction step only: the design's assertions plus these facts,
    k-inductive at `depth`? The proof folder is kept under `work`; its
    name is in "proof_dir"."""
    td = _folder(work, name)
    src = inject_invariants(row["verilog"], row["top_module"],
                            list(invariants)).text
    (td / "design.sv").write_text(src)
    (td / "job.sby").write_text(
        f"[options]\nmode prove\ndepth {depth}\ntimeout {TIMEOUT_S}\n\n"
        "[engines]\nsmtbmc --induction yices\n\n"
        f"[script]\nread -formal design.sv\nprep -top {row['top_module']}\n\n"
        "[files]\ndesign.sv\n")
    t0 = time.monotonic()
    _, hung = _sby(td)
    if hung:
        return {"tier": "TIMEOUT", "trace": None, "broke": None,
                "secs": round(time.monotonic() - t0, 1), "proof_dir": td.name}
    logfile = td / "job" / "logfile.txt"
    log = logfile.read_text() if logfile.exists() else ""
    m = re.search(r"returned (\w+) for induction", log)
    status = m.group(1).upper() if m else (
        "TIMEOUT" if "timeout" in log.lower() else "ERROR")
    tier = {"PASS": "INDUCTIVE", "FAIL": "NOT_INDUCTIVE",
            "TIMEOUT": "TIMEOUT"}.get(status, "ERROR")
    trace = None
    vcds = sorted((td / "job").rglob("trace_induct.vcd"))
    if tier == "NOT_INDUCTIVE" and vcds:
        trace = summarize_vcd(vcds[0])[:TRACE_CAP]
    broke = re.search(r"failed assertion \S+ at (\S+)", log)
    return {"tier": tier, "trace": trace,
            "broke": broke.group(1) if broke else None,
            "secs": round(time.monotonic() - t0, 1), "proof_dir": td.name}


_FAILED_AT = re.compile(r"failed assertion \S+ at design\.sv:(\d+)\.")
_IMPLICIT = re.compile(r"design\.sv:(\d+): Warning: Identifier `\\?(\S+?)' "
                       r"is implicitly declared")
_ERROR_AT = re.compile(r"design\.sv:(\d+): ERROR: (.*)")


def prove_true(row, invariants, work, depth=1, name=None):
    """The whole proof, base case and induction step, for facts not yet
    proven true (prove() runs the step only, which is enough for subsets
    of an answer already proven). One of:
      PROVEN         every fact holds from reset, and the facts with the
                     property are inductive at `depth`
      FALSE          a run from reset breaks `fact` (None: the property)
      NOT_INDUCTIVE  true from reset, but a state holding everything
                     breaks `fact` (None: the property) one step later
      BAD_FACT       `fact` cannot be read: a name the design does not
                     have (yosys would quietly invent a free wire for it,
                     making the fact look false), bad syntax, or a system
                     function; `detail` says which
      TIMEOUT, ERROR the tool could not judge; says nothing of the facts
    with `trace`, the proof tool's run, for FALSE and NOT_INDUCTIVE. The
    proof folder is kept under `work`; its name is in "proof_dir"."""
    td = _folder(work, name)
    inj = inject_invariants(row["verilog"], row["top_module"],
                            list(invariants))
    facts = {line: expr for line, (_, expr) in inj.line_map.items()}
    (td / "design.sv").write_text(inj.text)
    (td / "job.sby").write_text(
        f"[options]\nmode prove\ndepth {depth}\ntimeout {TIMEOUT_S}\n\n"
        "[engines]\nsmtbmc yices\n\n"
        f"[script]\nread -formal design.sv\nprep -top {row['top_module']}\n\n"
        "[files]\ndesign.sv\n")
    t0 = time.monotonic()
    out = {"tier": None, "fact": None, "detail": None, "trace": None,
           "proof_dir": td.name}
    log, hung = _sby(td)
    if hung:
        return dict(out, tier="TIMEOUT", secs=round(time.monotonic() - t0, 1))
    logfile = td / "job" / "logfile.txt"
    if logfile.exists():
        log += logfile.read_text()
    out["secs"] = round(time.monotonic() - t0, 1)
    for pattern, why in ((_ERROR_AT, None), (_IMPLICIT, "unknown name")):
        for m in pattern.finditer(log):
            line = int(m.group(1))
            if line in facts:
                detail = (f"{why} `{m.group(2)}`: the design has no "
                          f"such signal in module {row['top_module']}"
                          if why else m.group(2).strip())
                return dict(out, tier="BAD_FACT", fact=facts[line],
                            detail=detail)
    status = {k: m.group(1).upper() for k in ("basecase", "induction")
              for m in [re.search(rf"returned (\w+) for {k}", log)] if m}
    failed = _FAILED_AT.search(log)
    fact = facts.get(int(failed.group(1))) if failed else None
    if status.get("basecase") == "FAIL":
        tier, vcd = "FALSE", "trace.vcd"
    elif status.get("induction") == "FAIL":
        tier, vcd = "NOT_INDUCTIVE", "trace_induct.vcd"
    elif "DONE (PASS" in log:
        return dict(out, tier="PROVEN")
    else:
        return dict(out, tier="TIMEOUT" if "timeout" in log.lower()
                    else "ERROR", detail=log[-600:])
    vcds = sorted((td / "job").rglob(vcd))
    trace = summarize_vcd(vcds[0])[:TRACE_CAP] if vcds else None
    return dict(out, tier=tier, fact=fact, trace=trace)


def run(corpus=CORPUS, out=OUT, workers=6):
    rows = [json.loads(l) for l in Path(corpus).read_text().splitlines()
            if l.strip()]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["id"], r["dropped"]))
            except (json.JSONDecodeError, KeyError):
                continue          # a line cut short by a killed run
    tasks = [(r, i) for r in rows
             for i in [-1, -2] + list(range(len(r["invariants"])))
             if (r["id"], i) not in done]
    print(f"{len(rows)} rows, {len(tasks)} proofs to run "
          f"({len(done)} already done)", flush=True)
    # kept next to the log (8 Oct 2026); they used to be deleted
    work = out.parent / f"{out.stem}_proofs"
    work.mkdir(parents=True, exist_ok=True)
    lock, count = threading.Lock(), {"n": 0}

    def one(task):
        r, i = task
        facts = r["invariants"] if i < 0 else \
            r["invariants"][:i] + r["invariants"][i + 1:]
        try:
            res = prove(r, facts, work, depth=20 if i == -1 else 1,
                        name=f"{r['id']}_d{i}")
            if res.get("proof_dir"):
                res["proof_dir"] = f"{work.name}/{res['proof_dir']}"
        except Exception as e:                  # a crash is not a verdict
            res = {"tier": "CRASH", "trace": None, "broke": None, "secs": 0,
                   "error": f"{type(e).__name__}: {e}"[:300]}
        rec = {"id": r["id"], "generation": r.get("generation", 0),
               "n_clauses": len(r["invariants"]), "dropped": i,
               "clause": None if i < 0 else r["invariants"][i], **res}
        with lock:
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            count["n"] += 1
            if count["n"] % 500 == 0 or count["n"] == len(tasks):
                print(f"{count['n']}/{len(tasks)} proofs done", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, tasks))


def summary(out=OUT):
    """passes_one_step: ids whose full answer is 1-inductive (and whose
    20-step control passed); needed: id -> clause indices whose removal
    breaks the 1-step proof; counterexample: (id, i) -> trace."""
    by = collections.defaultdict(dict)
    for l in Path(out).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            by[r["id"]][r["dropped"]] = r
    passes = {i for i, d in by.items()
              if d.get(-1, {}).get("tier") == "INDUCTIVE"
              and d.get(-2, {}).get("tier") == "INDUCTIVE"}
    needed = {i: sorted(k for k, r in by[i].items()
                        if k >= 0 and r["tier"] == "NOT_INDUCTIVE")
              for i in passes}
    cex = {(i, k): by[i][k]["trace"] for i in passes for k in needed[i]}
    return {"rows": dict(by), "passes_one_step": passes, "needed": needed,
            "counterexample": cex}


def report(out=OUT):
    s = summary(out)
    by = s["rows"]
    print(f"rows checked: {len(by)}")
    print("control, full answer at 20 steps:",
          dict(collections.Counter(d[-1]["tier"] for d in by.values() if -1 in d)))
    print("full answer at 1 step (the benchmark's rule):",
          dict(collections.Counter(d[-2]["tier"] for d in by.values() if -2 in d)))
    facts = [r for i in s["passes_one_step"] for k, r in by[i].items() if k >= 0]
    print(f"in the {len(s['passes_one_step'])} answers that pass: facts needed "
          f"{sum(r['tier'] == 'NOT_INDUCTIVE' for r in facts)} of {len(facts)}")
    print(f"one-needed-fact-short answers: {len(s['counterexample'])}, with a "
          f"counterexample: {sum(bool(t) for t in s['counterexample'].values())}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    if not args.report:
        if shutil.which("sby") is None:
            sys.exit("sby is not on PATH: source the oss-cad-suite "
                     "environment first; nothing was run")
        run(args.corpus, args.out, args.workers)
    report(args.out)


if __name__ == "__main__":
    main()
