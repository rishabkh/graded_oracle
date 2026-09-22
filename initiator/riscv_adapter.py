"""Score a model's invariants on a real riscv-formal check.

The survey (riscv_survey.py) found 144 checks across three cores whose
proofs do NOT close by k-induction alone. This turns one of those into a
question and a verdict: hand the model the design and the property, take
its invariants, inject them into the core, run the check in prove mode,
and see whether the proof now closes.

The invariants go INSIDE the core module, guarded by the core's own
reset, because that is the shape the corpus teaches: facts about the
design's own registers, not about the testbench wrapper.

A run is one of three conditions, and each must be reported as such:

  native    the whole design, every assertion        the real task
  focused   the whole design, one assertion          is the target buried?
  cut       the cone-sliced design (slice_check.py)  does length hurt?

Only the model's INPUT changes between conditions. What gets proven is
always the original, unsliced check.

  venv/bin/python initiator/riscv_adapter.py --core nerv \\
      --check bus_dmem_ch0 --invariants "pc[1:0] == 2'b00" --k 10
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from riscv_survey import classify, prove_variant            # noqa: E402

RISCV = Path.home() / "Desktop/HarvardResearch/riscv-formal"

# where each core's logic lives, and what it calls its clock and reset.
# The reset polarity matters: picorv32's resetn is active low.
CORES = {
    "nerv": {"file": "cores/nerv/nerv.sv", "module": "nerv",
             "clock": "clock", "reset": "reset"},
    "serv": {"file": "cores/serv/serv-src/rtl/serv_top.v",
             "module": "serv_top", "clock": "clk", "reset": "i_rst"},
    "picorv32": {"file": "cores/picorv32/picorv32.v", "module": "picorv32",
                 "clock": "clk", "reset": "!resetn"},
}


def inject_invariants(source, module, clock, reset, invariants):
    """Add the invariants as reset-guarded assertions inside `module`.

    Raises if the module is not in the text: a silent no-op would score
    the model as failing when in fact nothing was ever asked."""
    if not invariants:
        return source
    lines = source.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines)
                  if re.match(rf"^\s*module\s+{re.escape(module)}\b", l)),
                 None)
    if start is None:
        raise ValueError(f"module {module!r} not found in the source")
    end = next((i for i in range(start, len(lines))
                if re.match(r"^\s*endmodule\b", lines[i])), None)
    if end is None:
        raise ValueError(f"module {module!r} is never closed")

    guard = reset if reset.startswith("!") else f"!{reset}"
    body = ["\n", "`ifdef RISCV_FORMAL\n",
            "// strengthening invariants under test\n",
            f"always @(posedge {clock}) if ({guard}) begin\n"]
    for inv in invariants:
        expr = inv.strip().rstrip(";").strip()
        body.append(f"    assert ({expr});\n")
    body += ["end\n", "`endif\n"]
    return "".join(lines[:end] + body + lines[end:])


def retarget(sby_text, filename, replacement):
    """Point the check at our patched copy of one file.

    The generated checks name the core through an unnormalised path
    (cores/nerv/../../cores/nerv/nerv.sv), so swapping the resolved path
    misses, the patched core is never read, and the model gets scored on
    a design that never saw its invariants. Match on the basename."""
    return re.sub(rf"\S*{re.escape(filename)}\b", replacement, sby_text)


def run_check(core, check, invariants, k=10, timeout=900, condition="native"):
    """Inject, rewrite to prove mode, run sby. Returns a record."""
    spec = CORES[core]
    checks = RISCV / f"cores/{core}/checks"
    sby = checks / f"{check}.sby"
    if not sby.exists():
        raise FileNotFoundError(sby)

    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        src = RISCV / spec["file"]
        patched = inject_invariants(src.read_text(), spec["module"],
                                    spec["clock"], spec["reset"], invariants)
        # the check reads the core by absolute path, so shadow that path
        shadow = work / Path(spec["file"]).name
        shadow.write_text(patched)
        text = retarget(prove_variant(sby.read_text(), k),
                        Path(spec["file"]).name, str(shadow))
        run_sby = work / f"{check}_try.sby"
        run_sby.write_text(text)

        t0 = time.monotonic()
        try:
            r = subprocess.run(["sby", "-f", str(run_sby)], cwd=work,
                               capture_output=True, text=True,
                               timeout=timeout)
            rc, out = r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            rc, out = None, ""
        return {"core": core, "check": check, "condition": condition,
                "k": k, "n_invariants": len(invariants),
                "invariants": invariants,
                "verdict": classify(rc, out[-4000:]),
                "wall_s": round(time.monotonic() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core", choices=sorted(CORES), required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--invariants", nargs="*", default=[])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--condition", default="native",
                   choices=["native", "focused", "cut"])
    p.add_argument("--dry", action="store_true",
                   help="print the injected core and stop")
    args = p.parse_args()

    if args.dry:
        spec = CORES[args.core]
        text = (RISCV / spec["file"]).read_text()
        out = inject_invariants(text, spec["module"], spec["clock"],
                                spec["reset"], args.invariants)
        tail = [l for l in out.splitlines() if "assert (" in l]
        print("\n".join(tail[-10:]) or "(nothing injected)")
        return

    if shutil.which("sby") is None:
        sys.exit("sby not on PATH - run `hwtools` first")
    rec = run_check(args.core, args.check, args.invariants, args.k,
                    args.timeout, args.condition)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
