"""Cone-of-influence ratio (step 8): lines inside the property's COI
over total lines.

The one direct measure of whether distractors are working — irrelevant
logic grows the file without growing the cone, so the ratio falls — and
the number that says whether the invariant is still readable off the
structure: a ratio near 1 means everything in the file bears on the
proof; a low ratio means the property's support is buried.

Mechanics: yosys elaborates and flattens the design (flatten so that
replicated/composed children walk correctly), write_json dumps the cell
graph with per-cell `src` attributes, and we BFS backward from every
$assert cell through the bit-level driver map, collecting the source
lines of every cell that can influence the assertion.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SRC_LINE = re.compile(r":(\d+)\.")


def _src_lines(attrs):
    src = attrs.get("src", "")
    return {int(m) for m in _SRC_LINE.findall(src)}


def coi_ratio(verilog, top_module):
    """len(source lines in the property's cone) / len(total lines),
    or None if yosys cannot process the design."""
    if shutil.which("yosys") is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        sv = Path(d) / "design.sv"
        out = Path(d) / "design.json"
        sv.write_text(verilog)
        script = (f"read_verilog -formal -sv {sv}; "
                  f"hierarchy -top {top_module}; proc; flatten; "
                  f"write_json {out}")
        try:
            r = subprocess.run(["yosys", "-p", script], capture_output=True,
                               text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not out.exists():
            return None
        design = json.loads(out.read_text())

    mod = design.get("modules", {}).get(top_module)
    if mod is None:
        return None
    cells = mod.get("cells", {})

    # bit -> driving cell, via each cell's output ports
    driver = {}
    for name, cell in cells.items():
        dirs = cell.get("port_directions", {})
        for port, bits in cell.get("connections", {}).items():
            if dirs.get(port) == "output":
                for bit in bits:
                    if isinstance(bit, int):
                        driver[bit] = name

    def is_assert(cell):
        # yosys 0.67 lowers immediate assertions to $check cells with a
        # FLAVOR parameter; older flows emit $assert. Cover/assume
        # flavours must NOT root the cone.
        if cell.get("type") == "$assert":
            return True
        if cell.get("type") == "$check":
            flavor = cell.get("parameters", {}).get("FLAVOR", "assert")
            return "assert" in str(flavor)
        return False

    roots = [n for n, c in cells.items() if is_assert(c)]
    if not roots:
        return None

    cone_lines, visited, queue = set(), set(roots), list(roots)
    while queue:
        name = queue.pop()
        cell = cells[name]
        cone_lines |= _src_lines(cell.get("attributes", {}))
        dirs = cell.get("port_directions", {})
        for port, bits in cell.get("connections", {}).items():
            if dirs.get(port) == "output":
                continue
            for bit in bits:
                if isinstance(bit, int) and bit in driver:
                    d = driver[bit]
                    if d not in visited:
                        visited.add(d)
                        queue.append(d)

    total = len(verilog.splitlines())
    if not total:
        return None
    return round(len(cone_lines) / total, 3)


if __name__ == "__main__":
    path, top = sys.argv[1], sys.argv[2]
    print(coi_ratio(Path(path).read_text(), top))
