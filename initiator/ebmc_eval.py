"""EBMC adapter for the large_lemma_miners benchmarks.

Injects OUR invariants into THEIR benchmark files exactly the way their
evaluator does (src/evaluation.py in the large_lemma_miners repo), and
judges with THEIR commands and verdict rules - so a number produced here
is comparable to their tables, not a private redefinition of the task.

Their conventions, ported:
- a lemma is a named block:  property lemmaN; <expr>; endproperty
- lemmas are inserted after the LAST endproperty of the file
- modes: correctness (assert lemma conjunction, unbounded engine),
  k_induction (lemma set inductive, --k-induction --bound 5),
  one_inductive_with_prop (lemmas AND prop 1-inductive),
  timing_with_lemmas (assume lemmas, assert prop, BMC bound 30)
- --reset rst appended when the file has an rst input; --trace always
- timeout 120s; "PROVED up to bound" counts as TIMEOUT, not PROVEN

  venv/bin/python initiator/ebmc_eval.py --file <bench.sv> --mode k_induction \\
      --lemmas "x <= 4'd8" "y == x"           # needs EBMC_PATH
  ... --dry      # print instrumented file + command, run nothing
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EBMC = os.getenv("EBMC_PATH", "ebmc")
TIMEOUT_S = 120        # their EBMC_TIMEOUT
K_INDUCTION_BOUND = 5  # their EBMC_K_FOR_INDUCTION
BMC_BOUND = 30         # their EBMC_BMC_BOUND

_ENDPROPERTY = re.compile(r"^.*endproperty\s*(//.*)?$")
_MODULE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.M)
_RST = re.compile(r"\binput(?:\s+(?:reg|logic|wire))?\s+rst\b", re.M)


def wrap_lemmas(exprs, prefix=""):
    """Their canonical few-shot form: the lemma carries the same clocking
    prefix as the file's own property."""
    p = (prefix + " ") if prefix else ""
    return [f"property lemma{i}; {p}({e}); endproperty"
            for i, e in enumerate(exprs)]


def clocking_prefix(source):
    """The @(posedge ...) disable iff (...) prefix of the file's own
    prop, so injected lemmas are clocked identically. Empty if bare."""
    m = re.search(r"property\s+prop\s*;\s*(.*?)\s*endproperty", source, re.S)
    if not m:
        return ""
    body = m.group(1).strip()
    pm = re.match(r"(@\([^)]*\)(?:\s*disable\s+iff\s*\([^)]*\))?)", body)
    return pm.group(1).strip() if pm else ""


def insert_index(lines):
    """Line index just after the file's last endproperty - their
    _find_insert_at_index."""
    hits = [i for i, l in enumerate(lines) if _ENDPROPERTY.match(l)]
    if not hits:
        raise ValueError("no endproperty in benchmark file")
    return hits[-1] + 1


def directives(names, mode):
    ands = " and ".join(names)
    if mode in ("correctness", "correctness_bounded"):
        return [f"assert property({ands}); \n"]
    if mode in ("one_induction", "k_induction"):
        return [f"assert property({ands}); \n"]
    if mode == "one_inductive_with_prop":
        return [f"assert property({' and '.join(names + ['prop'])}); \n"]
    if mode == "timing_with_lemmas":
        return [f"assume property ({n}); \n" for n in names] + \
               ["assert property (prop); \n"]
    if mode == "timing_without_lemmas":
        return ["assert property (prop); \n"]
    raise ValueError(f"unknown mode {mode!r}")


def build_variant(source, lemma_exprs, mode):
    lines = source.splitlines(keepends=True)
    at = insert_index(lines)
    blocks = [b + "\n" for b in wrap_lemmas(lemma_exprs,
                                             clocking_prefix(source))]
    lines[at:at] = blocks
    at += len(blocks)
    names = [f"lemma{i}" for i in range(len(lemma_exprs))]
    lines[at:at] = directives(names, mode)
    return "".join(lines)


def top_module(source):
    """The reset has to be scoped to the file's own top module. Most of
    their files declare `module main`, but the hard set names the module
    after the design, and `--reset main.rst` there dies with "failed to
    parse reset constraint" - which we were scoring as the model's miss."""
    mods = _MODULE.findall(source)
    if "main" in mods:
        return "main"
    return mods[-1] if mods else "main"


def ebmc_command(path, mode, rst, ebmc="ebmc", source=""):
    cmd = {
        "correctness": f"{ebmc} {path}",
        "one_induction": f"{ebmc} {path} --k-induction --bound 1",
        "one_inductive_with_prop": f"{ebmc} {path} --k-induction --bound 1",
        "correctness_bounded": f"{ebmc} {path} --bound {BMC_BOUND}",
        "k_induction": f"{ebmc} {path} --k-induction --bound {K_INDUCTION_BOUND}",
        "timing_with_lemmas": f"{ebmc} {path} --bound {BMC_BOUND}",
        "timing_without_lemmas": f"{ebmc} {path} --bound {BMC_BOUND}",
    }[mode]
    if rst:
        # ebmc 6.0 wants the module-scoped name; every benchmark file in
        # large_lemma_miners declares `module main`
        cmd += f" --reset {top_module(source)}.rst"
    return cmd + " --trace"


def parse_verdict(stdout, stderr, mode):
    """Their run_ebmc result logic, condensed to a verdict string."""
    strict = r"^\*\* Results:.* PROVED\s*$"
    bounded = r"^\*\* Results:.*PROVED up to bound.*"
    if re.search(bounded, stdout, re.M | re.S):
        return "TIMEOUT"
    pattern = strict if mode == "correctness" else "PROVED"
    if re.search(pattern, stdout, re.M | re.S):
        return "PROVEN"
    if "REFUTED" in stdout:
        return "CEX"
    if "INCONCLUSIVE" in stdout:
        return "INCONCLUSIVE"
    if "FAILURE:" in stdout or "Assertion failure" in stdout:
        return "ERROR"
    if stderr:
        return "ERROR"
    return "TIMEOUT"


def run(bench_path, lemma_exprs, mode, workdir):
    source = Path(bench_path).read_text()
    variant = build_variant(source, lemma_exprs, mode)
    out = Path(workdir) / f"{Path(bench_path).stem}_{mode}.sv"
    out.write_text(variant)
    cmd = ebmc_command(out, mode, rst=bool(_RST.search(source)), ebmc=EBMC,
                       source=source)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           errors="replace", timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "time": TIMEOUT_S, "cmd": cmd}
    return {"verdict": parse_verdict(r.stdout, r.stderr, mode),
            "time": round(time.perf_counter() - t0, 2), "cmd": cmd,
            "stdout_tail": r.stdout[-400:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="benchmark .sv")
    p.add_argument("--mode", default="k_induction",
                   choices=["correctness", "one_induction",
                            "one_inductive_with_prop", "k_induction",
                            "timing_with_lemmas", "timing_without_lemmas"])
    p.add_argument("--lemmas", nargs="*", default=[],
                   help="invariant expressions (plain boolean SVA bodies)")
    p.add_argument("--dry", action="store_true",
                   help="print instrumented file + command, run nothing")
    args = p.parse_args()

    source = Path(args.file).read_text()
    if args.dry:
        print(build_variant(source, args.lemmas, args.mode))
        print("# command:", ebmc_command(args.file, args.mode,
                                         rst=bool(_RST.search(source)),
                                         ebmc=EBMC, source=source))
        return
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        res = run(args.file, args.lemmas, args.mode, d)
    print(f"{res['verdict']}  ({res['time']}s)")
    print("cmd:", res["cmd"])
    if res["verdict"] != "PROVEN":
        print(res.get("stdout_tail", ""))


if __name__ == "__main__":
    main()
