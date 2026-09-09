"""Induction survey over riscv-formal checks: which of the generated
checks close by k-induction unaided, and which NEED a strengthening
invariant. The latter are the industrial problem set.

Per check: copy the .sby with mode prove at a small window (failing at
small k is what makes an invariant valuable), run sby with a per-check
timeout, classify. No model calls, no money - CPU only.

  venv/bin/python initiator/riscv_survey.py --checks-dir \\
      ~/Desktop/HarvardResearch/riscv-formal/cores/picorv32/checks \\
      --k 10 --workers 8 --timeout 300
"""
import argparse
import itertools
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_LOG = HERE / "logs" / "riscv_survey.jsonl"


def prove_variant(sby_text, k):
    """mode bmc -> mode prove at window k; unknown is a legal outcome;
    the bmc-only skip knob goes away. Everything else stays theirs."""
    out = []
    for line in sby_text.splitlines():
        s = line.strip()
        if s == "mode bmc":
            out.append("mode prove")
        elif s.startswith("expect "):
            out.append("expect pass,fail,unknown")
        elif s.startswith("depth "):
            out.append(f"depth {k}")
        elif s.startswith("skip "):
            continue
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def classify(rc, log_tail):
    """sby rc + summary text -> survey verdict."""
    if rc is None:
        return "TIMEOUT"
    if "successful proof by k-induction" in log_tail:
        return "PROVED"
    if "FAIL for basecase" in log_tail:
        return "BASECASE_FAIL"
    if "FAIL for induction" in log_tail or rc == 4:
        return "NEEDS_INVARIANT"
    if rc == 0:
        return "PROVED"
    return "ERROR"


def run_one(sby_path, k, timeout):
    text = sby_path.read_text()
    variant = sby_path.with_name(sby_path.stem + f"_prove{k}.sby")
    variant.write_text(prove_variant(text, k))
    t0 = time.monotonic()
    try:
        r = subprocess.run(["sby", "-f", variant.name],
                           cwd=variant.parent, capture_output=True,
                           text=True, timeout=timeout)
        rc, tail = r.returncode, r.stdout[-600:]
    except subprocess.TimeoutExpired:
        rc, tail = None, ""
    return {"check": sby_path.stem, "k": k,
            "verdict": classify(rc, tail), "rc": rc,
            "wall_s": round(time.monotonic() - t0, 1)}


class Progress:
    """One braille spinner for the whole pool: shows done/total and a
    clock. Individual per-worker spinners would garble the line."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CYAN, DIM, RESET = "\033[36m", "\033[2m", "\033[0m"

    def __init__(self, total):
        self.total = total
        self.done = 0
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        start = time.monotonic()
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            el = int(time.monotonic() - start)
            clock = f"{el // 60:02d}:{el % 60:02d}"
            sys.stderr.write(f"\r{self.CYAN}{frame}{self.RESET} "
                             f"surveying {self.done}/{self.total} "
                             f"{self.DIM}{clock}{self.RESET} ")
            sys.stderr.flush()
            self._stop.wait(0.08)
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()

    def __enter__(self):
        if sys.stderr.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checks-dir", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=int, default=300,
                   help="per-check seconds; a slow unaided proof is not "
                        "a problem-set candidate anyway")
    args = p.parse_args()

    d = Path(args.checks_dir).expanduser()
    files = sorted(f for f in d.glob("*.sby")
                   if "_prove" not in f.stem and "cover" not in f.stem)
    print(f"{len(files)} checks, k={args.k}, {args.workers} workers, "
          f"{args.timeout}s each")
    run_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    results = []
    with Progress(len(files)) as prog, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(lambda f: run_one(f, args.k, args.timeout),
                            files):
            res["run_id"] = run_id
            results.append(res)
            prog.done += 1
            sys.stderr.write("\r" + " " * 40 + "\r")
            print(f"  {res['check']:32s} {res['verdict']:16s} "
                  f"({res['wall_s']}s)")
            OUT_LOG.parent.mkdir(exist_ok=True)
            with OUT_LOG.open("a") as f:
                f.write(json.dumps(res) + "\n")

    from collections import Counter
    tally = Counter(r["verdict"] for r in results)
    print("\n=== survey ===")
    for v, n in tally.most_common():
        print(f"  {v:16s} {n}")
    need = [r["check"] for r in results if r["verdict"] == "NEEDS_INVARIANT"]
    print(f"\nproblem set ({len(need)}):", ", ".join(need[:20]),
          "..." if len(need) > 20 else "")


if __name__ == "__main__":
    main()
