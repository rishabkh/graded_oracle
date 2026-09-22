"""Cut a riscv-formal check down to the logic its property depends on.

Why: the evaluation designs are far larger than the corpus ones.
picorv32 is 3,049 lines; nerv is 1,322. Measured cones (22 Sep 2026):

  nerv     bus_dmem_ch0   587 of 1,322 lines   ~6k tokens
  picorv32 insn_add_ch0 1,531 of 3,049 lines  ~18k tokens

so slicing brings nerv and serv inside an 8k budget outright, and halves
picorv32, which is then read at a longer context at evaluation time.

Per-assertion slicing was measured and does NOT help on a CPU: on that
picorv32 check the smallest single-assert cone was 1,407 lines against
1,531 for all 33 together, because the RVFI checks compare the whole
core against a model. So the cone is taken over all assertions.

IMPORTANT: the slice is what the MODEL reads. Grading always runs the
original, unsliced files, so the slice may be unparseable Verilog - it
is a readable extract, not a design. It keeps every declaration so the
names in it make sense, and it says where lines were removed.

  venv/bin/python initiator/slice_check.py path/to/check.sby --out sliced/
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# "core.v:12.3-15.7", "wrap.sv:4", several joined by "|"
_SRC = re.compile(r"([^|:]+):(\d+)(?:\.\d+)?(?:-(\d+)(?:\.\d+)?)?")

# lines worth keeping whatever the cone says: without them the extract
# reads as a list of statements about signals that were never declared
# `define is deliberately absent: a macro kept for its own sake drags in
# whole CSR tables the property never touches, and keeping its header
# without its backslash-continued body reads as broken code
_DECL = re.compile(
    r"^\s*(module\b|endmodule\b|input\b|output\b|inout\b|reg\b|wire\b|"
    r"logic\b|integer\b|genvar\b|parameter\b|localparam\b|\)\s*;|\);)")

ELISION = "  // ... {n} lines not in the cone ..."


def yosys_script(sby_text):
    """The [script] section of a .sby: the file list and the top module,
    which is exactly what we need to elaborate the same design sby does."""
    out, in_script = [], False
    for line in sby_text.splitlines():
        if line.strip().startswith("["):
            in_script = line.strip() == "[script]"
            continue
        if in_script and line.strip():
            out.append(line.strip())
    return out


def parse_src(attr):
    """Every (file, first line, last line) in one yosys src attribute."""
    spans = []
    for chunk in (attr or "").split("|"):
        m = _SRC.fullmatch(chunk.strip())
        if m:
            start = int(m.group(2))
            spans.append((m.group(1), start, int(m.group(3) or start)))
    return spans


def slice_source(text, keep):
    """The kept lines in source order, declarations always included, with
    a note wherever something was dropped. `keep` is 1-based line numbers."""
    lines = text.splitlines()
    chosen = {i for i, line in enumerate(lines, start=1)
              if i in keep or _DECL.match(line)}

    # a line ending in a backslash continues onto the next one: keep the
    # whole run, or the extract ends in dangling continuations
    for i in sorted(chosen):
        j = i
        while j <= len(lines) and lines[j - 1].rstrip().endswith("\\"):
            j += 1
            chosen.add(j)

    wanted = [(i, lines[i - 1]) for i in sorted(chosen) if i <= len(lines)]

    out, previous = [], 0
    for n, (i, line) in enumerate(wanted):
        gap = i - previous - 1
        if gap > 0 and previous:
            out.append(ELISION.format(n=gap))
        out.append(line)
        previous = i
    return "\n".join(out) + "\n"


def cone_lines(sby_path, yosys="yosys", timeout=900):
    """Run the check's own yosys script, walk back from every assertion,
    and return {resolved file path: set of line numbers}."""
    sby_path = Path(sby_path)
    with tempfile.TemporaryDirectory() as d:
        js = Path(d) / "design.json"
        cmds = yosys_script(sby_path.read_text()) + [f"write_json {js}"]
        r = subprocess.run([yosys, "-p", "; ".join(cmds)],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=sby_path.parent)
        if r.returncode != 0:
            raise RuntimeError(f"yosys failed: {r.stderr[-300:]}")
        design = json.loads(js.read_text())

    top = next((n for n, m in design["modules"].items()
                if m.get("attributes", {}).get("top")), None)
    cells = design["modules"][top]["cells"]

    driver = {}
    for name, cell in cells.items():
        dirs = cell.get("port_directions", {})
        for port, bits in cell.get("connections", {}).items():
            if dirs.get(port) == "output":
                for bit in bits:
                    if isinstance(bit, int):
                        driver[bit] = name

    def is_assert(cell):
        if cell.get("type") == "$assert":
            return True
        if cell.get("type") == "$check":
            return "assert" in str(cell.get("parameters", {})
                                   .get("FLAVOR", "assert"))
        return False

    roots = [n for n, c in cells.items() if is_assert(c)]
    if not roots:
        raise RuntimeError("no assertions found: nothing to slice around")

    found, seen, queue = {}, set(roots), list(roots)
    while queue:
        cell = cells[queue.pop()]
        for f, a, b in parse_src(cell.get("attributes", {}).get("src")):
            path = str((sby_path.parent / f).resolve())
            found.setdefault(path, set()).update(range(a, b + 1))
        dirs = cell.get("port_directions", {})
        for port, bits in cell.get("connections", {}).items():
            if dirs.get(port) == "output":
                continue
            for bit in bits:
                if isinstance(bit, int) and bit in driver:
                    if driver[bit] not in seen:
                        seen.add(driver[bit])
                        queue.append(driver[bit])
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sby", help="the check's .sby file")
    p.add_argument("--out", help="directory for the sliced files")
    args = p.parse_args()

    found = cone_lines(args.sby)
    total_in, total_out = 0, 0
    for path, keep in sorted(found.items(), key=lambda kv: -len(kv[1])):
        src = Path(path)
        if not src.exists():
            continue
        text = src.read_text()
        sliced = slice_source(text, keep)
        n_in = len(text.splitlines())
        n_out = len([l for l in sliced.splitlines()
                     if "not in the cone" not in l])
        total_in += n_in
        total_out += n_out
        print(f"  {src.name:28s} {n_in:6d} -> {n_out:5d} lines")
        if args.out:
            d = Path(args.out)
            d.mkdir(parents=True, exist_ok=True)
            (d / src.name).write_text(sliced)
    print(f"  {'TOTAL':28s} {total_in:6d} -> {total_out:5d} lines "
          f"(~{total_out * 12 // 1000}k tokens)")


if __name__ == "__main__":
    main()
