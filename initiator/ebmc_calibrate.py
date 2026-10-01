"""Prove a rebuilt EBMC judges exactly like the one past scores came from.

Scoring moves to the cluster so a run needs no laptop, which means a
rebuilt EBMC binary. The first two models were scored on the laptop's
EBMC (6.0, commit 0308d417) with a 300 s limit per problem. A score from
any other binary counts only after stored answers, replayed through it,
all come back with the verdict they got the first time. Only answers
judged well under the limit are replayed, so a slower or faster machine
cannot flip a verdict by timing alone.

  # laptop, once: choose the answers to replay, written to results/
  venv/bin/python initiator/ebmc_calibrate.py --make
  # any machine: replay them through $EBMC_PATH; exits 1 on any mismatch
  EBMC_PATH=... EBMC_TIMEOUT_S=300 python initiator/ebmc_calibrate.py
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CALIBRATION = HERE.parent / "results" / "ebmc_calibration.jsonl"
SOURCE_LOG = HERE / "logs" / "benchmark_solve.jsonl"
KEEP = ("run_id", "set", "bench", "lemmas", "verdict", "ebmc_time",
        "ebmc_timeout_s")


def select(rows, per_verdict=10, max_time=20):
    """Up to per_verdict answers of each verdict, judged in under
    max_time seconds on files EBMC can judge; one per file first so the
    replay covers as many designs as it can; the same choice whatever
    order the log is in."""
    ok = [r for r in rows
          if r.get("lemmas") and r.get("file_judgeable")
          and r.get("ebmc_time") is not None and r["ebmc_time"] < max_time
          and r.get("verdict") not in (None, "TIMEOUT", "NO_ANSWER")]
    key = lambda r: (r["set"], r["bench"], r.get("run_id", ""),   # noqa: E731
                     json.dumps(r["lemmas"]))
    by = defaultdict(list)
    for r in sorted(ok, key=key):
        by[r["verdict"]].append({k: r[k] for k in KEEP if k in r})
    chosen = []
    for verdict in sorted(by):
        seen, first, rest = set(), [], []
        for r in by[verdict]:
            (rest if r["bench"] in seen else first).append(r)
            seen.add(r["bench"])
        pool = first + rest
        n = min(per_verdict, len(pool))
        # evenly spaced, so no one set or alphabet range dominates
        chosen += [pool[i * len(pool) // n] for i in range(n)]
    return chosen


def check(rows, run):
    """(row, verdict now, matches) for every stored answer replayed."""
    out = []
    for r in rows:
        got = run(r)
        out.append((r, got, got == r["verdict"]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--make", action="store_true",
                   help="choose the answers to replay from the local log")
    p.add_argument("--per-verdict", type=int, default=10)
    args = p.parse_args(argv)

    if args.make:
        rows = [json.loads(l) for l in SOURCE_LOG.read_text().splitlines()
                if l.strip()]
        chosen = select(rows, args.per_verdict)
        CALIBRATION.write_text("".join(json.dumps(r) + "\n" for r in chosen))
        print(f"wrote {len(chosen)} answers to {CALIBRATION}")
        return

    from benchmark_solve import BENCH_ROOT
    from ebmc_eval import EBMC, TIMEOUT_S, run as ebmc_run
    rows = [json.loads(l) for l in CALIBRATION.read_text().splitlines()
            if l.strip()]
    wanted = {r.get("ebmc_timeout_s") for r in rows} - {None}
    if wanted and TIMEOUT_S not in wanted:
        sys.exit(f"EBMC_TIMEOUT_S is {TIMEOUT_S}; the stored verdicts were "
                 f"judged at {sorted(wanted)}. Set it to match first.")
    version = subprocess.run([EBMC, "--version"], capture_output=True,
                             text=True).stdout.strip()
    print(f"EBMC {version} ({EBMC}), limit {TIMEOUT_S}s, "
          f"replaying {len(rows)} stored answers")

    def replay(r):
        with tempfile.TemporaryDirectory() as d:
            return ebmc_run(BENCH_ROOT / r["set"] / r["bench"], r["lemmas"],
                            "one_inductive_with_prop", d)["verdict"]

    results = check(rows, replay)
    bad = [(r, got) for r, got, ok in results if not ok]
    for r, got in bad:
        print(f"  MISMATCH {r['set']}/{r['bench']}: stored {r['verdict']}, "
              f"now {got}")
    print(f"{len(results) - len(bad)}/{len(results)} verdicts match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
