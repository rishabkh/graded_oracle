"""Render an sby .vcd trace as Fixer-readable plain text.

A CTI (rc=4) or CEX (rc=2) trace is the repair instruction — for kitest,
the arbitrary induction start state `sa = 32'h00000000, sb = 32'h40000000`
is exactly what points a model at `assert(sa == sb)`. But sby writes it
as a VCD waveform, which cannot go in a prompt. This parser extracts the
signal values at the first and last step and prints them as text.

Format handled (Yosys-SMTBMC output): `$var` declarations inside module
`$scope` blocks, then `#<time>` markers with `b<bits> <id>` value
changes. The `smt_step` integer outside any module scope carries the
step number; it and `smt_clock` are bookkeeping, not design state.
"""
from __future__ import annotations

import re
from pathlib import Path

_VAR = re.compile(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+).*?\$end")


def _fmt(name: str, width: int, bits: str) -> str:
    if width == 1:
        return f"{name} = {bits}"
    if set(bits) <= {"0", "1"}:
        return f"{name} = {width}'h{int(bits, 2):0{(width + 3) // 4}x}"
    return f"{name} = {width}'b{bits}"   # x/z bits: show raw


def summarize_vcd(vcd_path: Path) -> str:
    vars_by_id: dict[str, tuple[str, int]] = {}   # id -> (name, width)
    step_id: str | None = None
    in_module_scope = 0
    values: dict[str, str] = {}
    first_step_values: dict[str, str] | None = None
    times_seen = 0

    for line in Path(vcd_path).read_text().splitlines():
        line = line.strip()
        if line.startswith("$scope"):
            in_module_scope += 1
            continue
        if line.startswith("$upscope"):
            in_module_scope -= 1
            continue
        m = _VAR.match(line)
        if m:
            width, vid, name = int(m.group(1)), m.group(2), m.group(3)
            if in_module_scope > 0:
                vars_by_id[vid] = (name, width)
            elif name == "smt_step":
                step_id = vid
            continue
        if line.startswith("#"):
            times_seen += 1
            # values at the first timestamp = the (arbitrary) start state
            if first_step_values is None and times_seen > 1 and values:
                first_step_values = dict(values)
            continue
        if line.startswith("b"):
            bits, _, vid = line[1:].partition(" ")
            values[vid] = bits
        elif line and line[0] in "01xz" and len(line) > 1:
            values[line[1:]] = line[0]

    if not values or not vars_by_id:
        return f"trace summary unavailable: no signal data in {vcd_path.name}"
    if first_step_values is None:
        first_step_values = dict(values)

    step_bits = values.get(step_id or "", "")
    final_step = (int(step_bits, 2)
                  if step_bits and set(step_bits) <= {"0", "1"} else None)

    def block(state: dict[str, str]) -> list[str]:
        out = []
        for vid, (name, width) in vars_by_id.items():
            if vid in state:
                out.append("    " + _fmt(name, width, state[vid]))
        return out

    lines = [f"Trace summary ({vcd_path.name}):"]
    lines.append("  At start state (step 0):")
    lines += block(first_step_values)
    label = f"step {final_step}" if final_step is not None else "final step"
    lines.append(f"  At failure ({label}):")
    lines += block(values)
    return "\n".join(lines)
