"""Does the property notice when the design is broken? MCY, per design.

27 Oct 2026, Rishab: "Find weak ones by breaking the design on purpose.
Make small changes, like flipping a condition or changing a number, and
see if the property notices. If it doesn't notice, it's weak. A free tool
called MCY does this."

MCY (Mutation Cover with Yosys, in oss-cad-suite) makes SIZE small changes
to a design: one signal bit flipped, or stuck at 0 or 1. It changes only
the design's own logic: a cell whose output reaches only the property and
the design's assumptions (never an output port) is the property's own
logic and is left alone, so "count <= 7" is never broken into
"count <= 6" and then counted as noticed. For each changed design:

  NOTICED    the property fails within DEPTH steps (bounded check)
  UNNOTICED  the property still holds
  ERROR      a tool failure

  score = NOTICED / (NOTICED + UNNOTICED)

A low score means most of the design could be wrong without the property
seeing it: a weak training example. No model is called; it costs nothing
but laptop time, one bounded check per change (about a second: the check
only looks at what the property depends on).

--compare-outputs also asks, for each change the property missed, whether
the outputs can differ from the original's at all (MCY's own split): if
not, the change is NOCHANGE and leaves the score. That check runs the
whole design twice side by side, and on the wide designs (64 and 128-bit
words, 16 to 128 entries) it ran past 60 s for 4 of 8 changes in the first
trial (27 Oct 2026), so it is off by default.

  hwtools
  venv/bin/python extender/mutation_check.py --keys 60a099f22252     # one design
  venv/bin/python extender/mutation_check.py --workers 6             # all, resumable
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE / "all_unique_designs_2026-10-27.jsonl"
OUT = HERE / "logs" / "mutation_check_2026-10-27" / "scores.jsonl"
SIZE = 24          # changes per design
DEPTH = 25         # steps of each bounded check; the proofs use k = 20
TIMEOUT_S = 60     # per bounded check
SEED = 20261027
MUTSEL_WIDTH = 8   # create_mutated.sh -c default

# yosys 0.67 reads an assertion as a $check cell (older ones as $assert)
FORMAL_CELLS = "t:$check t:$assert %u t:$assume %u t:$cover %u t:$live %u t:$fair %u"
# all cells, minus those that reach the property and assumptions but no
# output port: the property's own logic
SELECT = f"c:* {FORMAL_CELLS} %ci* o:* %ci* %d %d"

LOGIC = """\
fm = result("test_fm")
if fm == "FAIL":
    tag("NOTICED")
elif fm in ("PASS", "TIMEOUT", "UNKNOWN"):
    tag("UNNOTICED")
else:
    tag("ERROR")
"""
LOGIC_COMPARE = """\
fm = result("test_fm")
if fm == "FAIL":
    tag("NOTICED")
elif fm in ("PASS", "TIMEOUT", "UNKNOWN"):
    eq = result("test_eq")
    if eq == "FAIL":
        tag("UNNOTICED")
    elif eq in ("PASS", "TIMEOUT", "UNKNOWN"):
        tag("NOCHANGE")
    else:
        tag("ERROR")
else:
    tag("ERROR")
"""
REPORT = ('print("MCYSCORE noticed=%d unnoticed=%d nochange=%d error=%d" % '
          '(tags("NOTICED"), tags("UNNOTICED"), tags("NOCHANGE"), tags("ERROR")))')
STATUSES = "PASS FAIL TIMEOUT UNKNOWN ERROR"
_SCORE = re.compile(r"MCYSCORE noticed=(\d+) unnoticed=(\d+) nochange=(\d+) error=(\d+)")


def config_text(top, size=SIZE, seed=SEED, compare=False):
    eq = ["[test test_eq]", f"expect {STATUSES}", "run bash $PRJDIR/test_eq.sh", ""]
    return "\n".join([
        "[options]", f"size {size}", f"seed {seed}",
        "tags NOTICED UNNOTICED NOCHANGE ERROR", f"select {SELECT}", "",
        "[script]", "read -formal design.sv", f"prep -top {top}", "",
        "[files]", "design.sv", "",
        "[logic]", LOGIC_COMPARE if compare else LOGIC,
        "[report]", REPORT, "",
        "[test test_fm]", f"expect {STATUSES}", "run bash $PRJDIR/test_fm.sh", "",
        *(eq if compare else []),
    ])


def sby_text(script, files, depth=DEPTH, timeout_s=TIMEOUT_S):
    return "\n".join([
        "[options]", "mode bmc", f"depth {depth}", f"timeout {timeout_s}",
        "expect pass,fail,timeout,unknown,error", "",
        "[engines]", "smtbmc yices", "",
        "[script]", *script, "",
        "[files]", *files, "",
    ])


def test_script(name, ctrl):
    """One MCY test: make the changed design, run its bounded check, and
    append its status (PASS, FAIL, TIMEOUT, ...) for MCY to read."""
    links = " ".join(f"../../{f}" for f in
                     ([f"{name}.sby", f"{name}.sv"] if ctrl else [f"{name}.sby"]))
    return "\n".join([
        "#!/bin/bash", "exec 2>&1",
        f"bash $SCRIPTS/create_mutated.sh {'-c ' if ctrl else ''}-o mutated.il"
        " || { echo '1 ERROR' >> output.txt; exit 0; }",
        f"ln -sf {links} .",
        f"sby -f {name}.sby > sby_out.txt 2>&1",
        f"st=$(awk '{{print $1; exit}}' {name}/status 2>/dev/null)",
        'echo "1 ${st:-ERROR}" >> output.txt',
        "exit 0", "",
    ])


def miter_sv(top, ports, width=MUTSEL_WIDTH):
    """The original (ref) and the changed design (uut) side by side on the
    same inputs; every cycle each output must agree. `ports` is
    [(name, direction, width)] of the top module, mutsel excluded."""
    ins = [(n, w) for n, d, w in ports if d == "input"]
    outs = [(n, w) for n, d, w in ports if d == "output"]
    def decl(w):
        return f"[{w - 1}:0] " if w > 1 else ""
    lines = ["module miter ("]
    lines += [f"    input wire {decl(w)}{n}{',' if i < len(ins) - 1 else ''}"
              for i, (n, w) in enumerate(ins)]
    lines.append(");")
    for n, w in outs:
        lines.append(f"    wire {decl(w)}ref_{n}, uut_{n};")
    for inst, sel in (("ref", 0), ("uut", 1)):
        conns = [f".mutsel({width}'d{sel})"] + [f".{n}({n})" for n, _ in ins] \
            + [f".{n}({inst}_{n})" for n, _ in outs]
        lines.append(f"    {top} {inst} ({', '.join(conns)});")
    lines.append("    always @* begin")
    lines += [f"        assert (ref_{n} == uut_{n});" for n, _ in outs]
    lines += ["    end", "endmodule", ""]
    return "\n".join(lines)


def top_ports(verilog, top, workdir):
    """[(name, direction, width)] of the top module, as yosys reads it."""
    (workdir / "ports_design.sv").write_text(verilog)
    script = (f"read -formal ports_design.sv; prep -top {top}; "
              "write_json ports.json")
    proc = subprocess.run(["yosys", "-q", "-p", script], cwd=workdir,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError("yosys could not read the design: "
                           + (proc.stderr or proc.stdout)[-400:])
    mod = json.load(open(workdir / "ports.json"))["modules"][top]
    return [(n, p["direction"], len(p["bits"])) for n, p in mod["ports"].items()]


def write_project(design, pdir, size=SIZE, depth=DEPTH, seed=SEED, timeout_s=TIMEOUT_S,
                  compare=False):
    top = design["top_module"]
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "design.sv").write_text(design["verilog"])
    (pdir / "config.mcy").write_text(config_text(top, size, seed, compare))
    (pdir / "test_fm.sby").write_text(sby_text(
        ["read_rtlil mutated.il", f"prep -top {top}"], ["mutated.il"], depth, timeout_s))
    (pdir / "test_fm.sh").write_text(test_script("test_fm", ctrl=False))
    if not compare:
        return
    ports = top_ports(design["verilog"], top, pdir)
    if any(d not in ("input", "output") for _, d, _ in ports):
        raise RuntimeError("the top module has an inout port")
    (pdir / "test_eq.sby").write_text(sby_text(
        ["read_rtlil mutated.il", "chformal -assert -cover -remove",
         "read -formal test_eq.sv", "prep -top miter",
         "fmcombine -initeq -anyeq miter ref uut", "flatten", "opt -fast"],
        ["test_eq.sv", "mutated.il"], depth, timeout_s))
    (pdir / "test_eq.sv").write_text(miter_sv(top, ports))
    (pdir / "test_eq.sh").write_text(test_script("test_eq", ctrl=True))


def parse_score(text):
    m = _SCORE.search(text or "")
    if not m:
        return None
    noticed, unnoticed, nochange, error = map(int, m.groups())
    seen = noticed + unnoticed
    return {"noticed": noticed, "unnoticed": unnoticed, "nochange": nochange,
            "error": error, "score": round(noticed / seen, 3) if seen else None}


def check_design(design, root, size=SIZE, depth=DEPTH, seed=SEED,
                 timeout_s=TIMEOUT_S, keep=False, compare=False):
    """Run MCY on one design; one result row."""
    row = {"key": design["key"], "size": size, "depth": depth, "seed": seed,
           "compare_outputs": compare}
    t0 = time.monotonic()
    pdir = Path(root) / design["key"]
    if pdir.exists():
        shutil.rmtree(pdir)
    try:
        write_project(design, pdir, size, depth, seed, timeout_s, compare)
        for cmd in (["mcy", "init"], ["mcy", "run", "-j1"]):
            proc = subprocess.run(cmd, cwd=pdir, capture_output=True, text=True,
                                  timeout=size * 4 * timeout_s + 600)
            if proc.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)} failed: "
                                   + (proc.stdout + proc.stderr)[-600:])
        proc = subprocess.run(["mcy", "status"], cwd=pdir, capture_output=True,
                              text=True, timeout=120)
        score = parse_score(proc.stdout + proc.stderr)
        if score is None:
            raise RuntimeError("no MCYSCORE line: " + (proc.stdout + proc.stderr)[-400:])
        row.update(score, status="ok")
    except Exception as exc:                  # one design never stops the run
        row.update(status="error", error_text=str(exc)[-800:])
    row["seconds"] = round(time.monotonic() - t0, 1)
    if not keep and pdir.exists():
        shutil.rmtree(pdir, ignore_errors=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--designs", default=str(DESIGNS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keys", help="comma-separated design keys (default: all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--timeout-s", type=int, default=TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--keep", action="store_true", help="keep each MCY project folder")
    ap.add_argument("--compare-outputs", action="store_true",
                    help="also split off changes that make no visible difference (slow on wide designs)")
    args = ap.parse_args(argv)
    if shutil.which("mcy") is None or shutil.which("sby") is None:
        sys.exit("mcy/sby not on PATH: run hwtools first")

    designs = [json.loads(line) for line in open(args.designs)]
    if args.keys:
        wanted = set(args.keys.split(","))
        designs = [d for d in designs if d["key"] in wanted]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["key"] for line in open(out)} if out.exists() else set()
    todo = [d for d in designs if d["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    root = Path(tempfile.mkdtemp(prefix="mcy_", dir=out.parent)) if not args.keep \
        else out.parent / "projects"
    print(f"{len(todo)} designs to check ({len(done)} already in {out}), "
          f"{args.size} changes each, depth {args.depth}, {args.workers} at a time")
    with ThreadPoolExecutor(args.workers) as pool, open(out, "a") as fh:
        futs = [pool.submit(check_design, d, root, args.size, args.depth, args.seed,
                            args.timeout_s, args.keep, args.compare_outputs) for d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {row['key']} {row['status']} "
                  f"score={row.get('score')} noticed={row.get('noticed')} "
                  f"unnoticed={row.get('unnoticed')} nochange={row.get('nochange')} "
                  f"{row['seconds']}s", flush=True)
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
