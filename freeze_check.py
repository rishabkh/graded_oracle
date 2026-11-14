"""The checker's freeze check (8 Nov 2026, Rishab approved).

On 27 Oct 2026, 6 of the 7 storage steps the checker kept were labeled
wrong, 3 of them the same way: a new part drained only on another
output's ready, so a receiver that says ready only after it sees valid
(allowed, and fine for the parent) froze the design for good. The proof
and the nothing-lost search both passed them, since nothing is lost: the
items just never move again.

  channels    each `x_valid` paired with its own `x_ready` by name (also
              xvalid/xready, x_vld/x_rdy, AXI's tvalid/tready, and names
              that say which way the port points: i_x_valid/o_x_ready,
              x_valid_i/x_ready_o); a channel's side is "in" when the
              design takes items, "out" when it hands them over
  testbench   random inputs, every sender holding its item and its data
              until the item is taken, every receiver saying ready the
              cycle after it sees valid, any reset held for 3 cycles and
              no item offered before then
  check       a simulation per seed (iverilog); "frozen" when, after items
              moved, nothing moved again to the end of the run under every
              seed, nor in the last 4,000 cycles of a run 10 times as long
              (a slow design moves again). A design that never moved runs
              again with receivers that are always ready: "waits_for_ready"
              when items then move under every seed, since it waits for a
              ready before anything moves, and "never_moved" when not. A
              design with no channel is "not_checked" and one the tools
              cannot run is "error". Only frozen and waits_for_ready turn
              a design away; the rest do not say the design is wrong.
  lesson      what the check saw, for the retry

The trial of 8 Nov 2026 (40 designs) found these gaps, fixed the same day:
7 designs named their ports by the way they point (i_req_valid with
o_req_ready) and went unchecked; a channel named only i (i_valid/i_ready)
held every i_ input with its items, the clock enable i_ce too, so a good
design looked frozen; 2 designs that took items only while their output's
receiver already said ready never moved, which passed; and, once checked,
a slow design looked frozen.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))       # the oracle package, beside this file

from oracle.inject import _mask_comments, strip_assertions      # noqa: E402

# off unless a run switches it on (extender/batch.py --freeze-check,
# initiator/run.py pilot --freeze-check): runs without it are as before
ON = False
CYCLES = 2000       # simulated per seed
STILL = 400         # cycles with nothing moving, to the end, that are a freeze
LONGER = 10         # a design that looks frozen runs again 10 times as long,
                    # and is frozen only when still for the last 10 x STILL
                    # cycles of that run too (8 Nov 2026: a slow design, about
                    # 1,550 cycles an item, looked frozen)
SEEDS = (1, 2, 3)   # frozen only when frozen under every one
TIMEOUT_S = 60
PREFIX = "oracle_fz"

_VALID = ("_valid", "valid", "_vld", "vld")      # longest first
_READY = ("_ready", "ready", "_rdy", "rdy")
_FRONT = {"i_": "o_", "o_": "i_"}     # the other way's mark, in front
_END = {"_i": "_o", "_o": "_i"}       # and at the end
# a channel with no name of its own: any input could be its data or a
# setting (8 Nov 2026: i_valid/i_ready held the clock enable i_ce), so it
# holds none of them with its items
_NAMELESS = ("main", "i", "o")
_HIGH = {"rst", "reset", "arst", "areset", "srst", "rst_i", "reset_i"}
_LOW = {"rst_n", "rstn", "reset_n", "resetn", "arst_n", "arstn", "areset_n",
        "aresetn", "rst_ni", "rst_b", "reset_b"}
_NAME = re.compile(r"[A-Za-z_]\w*")


def _base(name, suffixes):
    low = name.lower()
    for s in suffixes:
        if low.endswith(s):
            return name[:len(name) - len(s)]
    return None


def _ready_names(valid):
    """(the channel's name, the names its ready may have, in the order
    tried) for a port named as a valid, else None: ready for valid
    (x_valid/x_ready, i_valid/i_ready where i_ names the channel), then
    with the other way's mark in front (i_x_valid/o_x_ready, i_valid/o_ready,
    where i_ and o_ say which way the port points) or at the end, as Ibex
    names them (x_valid_i/x_ready_o)."""
    low = valid.lower()
    base = _base(valid, _VALID)
    if base is not None:
        names = [base + s for s in _READY]
        rest = _base(valid[2:], _VALID)
        if low[:2] in _FRONT and rest is not None:
            names += [_FRONT[low[:2]] + rest + s for s in _READY]
    elif low[-2:] in _END and _base(valid[:-2], _VALID) is not None:
        base = _base(valid[:-2], _VALID)
        names = [base + s + _END[low[-2:]] for s in _READY]
    else:
        return None
    return base.rstrip("_") or "main", names


def channels(ports):
    """[{name, valid, ready, side}] from {port: (direction, width)}, sorted
    by name. A valid and a ready pair up when the ready has one of the
    valid's ready names and they point opposite ways."""
    lower = {p.lower(): p for p in ports}
    out = []
    for valid, (direction, _) in ports.items():
        named = _ready_names(valid)
        if named is None:
            continue
        for candidate in named[1]:
            ready = lower.get(candidate.lower())
            if ready is None or ports[ready][0] == direction:
                continue
            side = "out" if direction == "output" else "in"
            out.append({"name": named[0], "valid": valid, "ready": ready,
                        "side": side})
            break
    return sorted(out, key=lambda c: c["name"])


def reset_level(name):
    """1 or 0 for the level that holds a reset named `name` (also with i_ in
    front, as in i_rst and i_reset), None when the name is not a reset's."""
    low = name.lower()
    if low.startswith("i_"):
        low = low[2:]
    return 1 if low in _HIGH else 0 if low in _LOW else None


_STATEMENT = re.compile(r"\b(?:cover|assume|restrict)\b(?:\s+property)?\s*\(")
_FORMAL_CALL = re.compile(r"\$(?:past|fell|rose|stable|changed|anyseq|anyconst)"
                          r"\s*\(|\$initstate\b")


def _close(masked, i):
    """The index just after the bracket that closes the one at `i`."""
    depth = 0
    for k in range(i, len(masked)):
        depth += {"(": 1, ")": -1}.get(masked[k], 0)
        if depth == 0:
            return k + 1
    return None


def _replaced(source, pattern, with_text, statement):
    masked, spans = _mask_comments(source), []
    for m in pattern.finditer(masked):
        if spans and m.start() < spans[-1][1]:
            continue                      # inside one already replaced
        end = m.end()
        if masked[end - 1] == "(":
            end = _close(masked, end - 1)
            if end is None:
                continue
        if statement:
            semi = masked.find(";", end)
            if semi == -1:
                continue
            end = semi + 1
        spans.append((m.start(), end))
    out, prev = [], 0
    for start, end in spans:
        out += [source[prev:start], with_text]
        prev = end
    return "".join(out + [source[prev:]])


def simulation_copy(verilog):
    """The design as a simulator can read it: its assertions, covers and
    assumes become empty statements, and what only a proof tool reads
    ($past, $fell, $rose, $stable, $changed, $initstate, $anyseq,
    $anyconst), which our designs use only around their checks, becomes 0."""
    text = _replaced(strip_assertions(verilog), _STATEMENT, ";", True)
    return _replaced(text, _FORMAL_CALL, "1'b0", False)


def _vector(width):
    return f"[{width - 1}:0] " if width > 1 else ""


def _random(name, width):
    """A random value for an input: any for a bit, 0 to 15 for a word. The
    8 Nov 2026 calibration: random 32-bit settings and item fields (a
    latency, a timer's period, a shift amount) made good designs wait
    billions of cycles, which looked like a freeze."""
    if width > 1:
        return f"{name} <= {{$random({PREFIX}_seed)}} % 16;"
    return f"{name} <= $random({PREFIX}_seed);"


def testbench(top, ports, clock, seed, cycles, always_ready=False):
    """A testbench module `oracle_fz_tb` for `top`: it prints one line,
    ORACLE_FZ followed by the counts, and stops after `cycles` cycles.
    With `always_ready` every receiver says ready on every cycle."""
    chans = channels(ports)
    valids = {c["valid"]: c for c in chans}
    readies = {c["ready"]: c for c in chans}
    data = {}                    # a sender's data, held with its item
    for c in chans:
        if c["side"] == "in" and c["name"].lower() not in _NAMELESS:
            for p, (d, _) in ports.items():
                if d == "input" and p != c["valid"] and p.lower().startswith(
                        c["name"].lower() + "_"):
                    data[p] = c
    p = PREFIX
    lines = ["`timescale 1ns/1ps", f"module {p}_tb;",
             f"    integer {p}_seed = {seed};",
             f"    integer {p}_cycle = 0;",
             f"    integer {p}_moves = 0, {p}_first = -1, {p}_last = -1;",
             f"    integer {p}_still = 0;",
             f"    reg {clock} = 1'b0;",
             f"    always #5 {clock} = ~{clock};"]
    for name, (direction, width) in sorted(ports.items()):
        if name == clock:
            continue
        kind = "reg" if direction == "input" else "wire"
        start = f" = {width}'d0" if direction == "input" else ""
        lines.append(f"    {kind} {_vector(width)}{name}{start};")
    lines.append(f"    {top} dut (" + ", ".join(
        f".{name}({name})" for name in sorted(ports)) + ");")
    for c in chans:
        if c["side"] == "in":
            lines.append(f"    reg {p}_took_{c['name']} = 1'b0;")
    moves = " || ".join(f"({c['valid']} && {c['ready']})" for c in chans) \
        or "1'b0"
    lines.append(f"    wire {p}_move = {moves};")
    # inputs change on the falling edge, so the rising edge sees them settled
    lines.append(f"    always @(negedge {clock}) begin")
    for name, (direction, width) in sorted(ports.items()):
        if direction != "input" or name == clock or name in data:
            continue
        level = reset_level(name)
        if level is not None:
            lines.append(f"        {name} <= ({p}_cycle < 3) ? 1'b{level} : "
                         f"1'b{1 - level};")
        elif name in readies and readies[name]["side"] == "out":
            said = "1'b1" if always_ready else readies[name]["valid"]
            lines.append(f"        {name} <= {said};")
        elif name in valids and valids[name]["side"] == "in":
            c = valids[name]
            held = [d for d, ch in data.items() if ch is c]
            # nothing offered before the count starts, after the reset
            lines.append(f"        if ({p}_cycle < 3) {name} <= 1'b0;")
            lines.append(f"        else if (!{name} || {p}_took_{c['name']}) begin")
            lines.append(f"            {_random(name, width)}")
            lines += [f"            {_random(d, ports[d][1])}" for d in sorted(held)]
            lines.append("        end")
        else:
            lines.append(f"        {_random(name, width)}")
    lines.append("    end")
    lines.append(f"    always @(posedge {clock}) begin")
    for c in chans:
        if c["side"] == "in":
            lines.append(f"        {p}_took_{c['name']} <= {c['valid']} && "
                         f"{c['ready']};")
    lines += [
        f"        if ({p}_cycle >= 3) begin",
        f"            if ({p}_move) begin",
        f"                {p}_moves = {p}_moves + 1;",
        f"                if ({p}_first < 0) {p}_first = {p}_cycle;",
        f"                {p}_last = {p}_cycle;",
        f"                {p}_still = 0;",
        f"            end else if ({p}_first >= 0)",
        f"                {p}_still = {p}_still + 1;",
        "        end",
        f"        {p}_cycle = {p}_cycle + 1;",
        f"        if ({p}_cycle == {cycles}) begin",
        f'            $display("ORACLE_FZ moves=%0d first=%0d last=%0d '
        f'still=%0d", {p}_moves, {p}_first, {p}_last, {p}_still);',
        "            $finish;",
        "        end",
        "    end",
        "endmodule", ""]
    return "\n".join(lines)


def _might_have_channels(verilog):
    names = set(_NAME.findall(verilog))
    lower = {n.lower() for n in names}
    for n in names:
        named = _ready_names(n)
        if named and any(r.lower() in lower for r in named[1]):
            return True
    return False


def _read(design, top, work):
    """(the top's ports {name: (direction, width)}, why not). Yosys, which
    reads every design for the proof, also writes the copy the simulator
    runs: flattened, every size worked out, so iverilog never meets a size
    it cannot work out itself (27 Oct 2026: a localparam in a port width)."""
    out, flat = work / "ports.json", work / "flat.v"
    r = subprocess.run(
        ["yosys", "-q", "-p", f"read_verilog -sv {design}; hierarchy -top "
         f"{top}; proc; flatten; opt_clean; write_json {out}; "
         f"write_verilog -noattr {flat}"],
        capture_output=True, text=True, timeout=TIMEOUT_S)
    if r.returncode != 0 or not out.exists():
        return None, (r.stderr or r.stdout).strip()[-300:]
    mod = json.loads(out.read_text())["modules"].get(top)
    if mod is None:
        return None, f"yosys has no module {top}"
    return {n: (p["direction"], len(p["bits"])) for n, p in mod["ports"].items()
            if p["direction"] in ("input", "output")}, None


def _run(design, top, ports, clock, seed, work, always_ready=False,
         cycles=CYCLES):
    tag = (f"{seed}" + ("_always_ready" if always_ready else "")
           + (f"_{cycles}_cycles" if cycles != CYCLES else ""))
    tb = work / f"tb_{tag}.v"
    tb.write_text(testbench(top, ports, clock, seed, cycles, always_ready))
    sim = work / f"sim_{tag}.vvp"
    r = subprocess.run(["iverilog", "-g2012", "-o", str(sim), "-s",
                        f"{PREFIX}_tb", str(tb), str(design)],
                       capture_output=True, text=True, timeout=TIMEOUT_S)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout).strip()[-300:]}
    try:
        r = subprocess.run(["vvp", "-n", str(sim)], capture_output=True,
                           text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"error": f"the simulation ran past {TIMEOUT_S} s"}
    m = re.search(r"ORACLE_FZ moves=(\d+) first=(-?\d+) last=(-?\d+) "
                  r"still=(\d+)", r.stdout)
    if not m:
        return {"error": (r.stdout + r.stderr).strip()[-300:]}
    moves, first, last, still = map(int, m.groups())
    return {"moves": moves, "first": first, "last": last, "still": still}


def check(verilog, top, clock="clk", work=None, seeds=SEEDS):
    """{"result": "frozen" | "waits_for_ready" | "moves" | "never_moved" |
    "not_checked" | "error", ...}; see the top of this file."""
    if not _might_have_channels(verilog):
        return {"result": "not_checked", "why": "no valid/ready pair"}
    if shutil.which("iverilog") is None or shutil.which("yosys") is None:
        return {"result": "error", "why": "iverilog or yosys is not on PATH"}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(work or tmp)
        work.mkdir(parents=True, exist_ok=True)
        design = work / "design.sv"
        design.write_text(simulation_copy(verilog))
        try:
            ports, why = _read(design, top, work)
        except subprocess.TimeoutExpired:
            ports, why = None, f"yosys ran past {TIMEOUT_S} s"
        if ports is None:
            return {"result": "error", "why": why}
        design = work / "flat.v"
        chans = channels(ports)
        if not chans:
            return {"result": "not_checked", "why": "no valid/ready pair"}
        if clock not in ports:
            return {"result": "error", "why": f"no clock port {clock}"}
        try:
            runs = [_run(design, top, ports, clock, s, work) for s in seeds]
            again = longer = None
            if any("error" in r for r in runs):
                pass
            elif all(r["first"] < 0 for r in runs):
                if any(c["side"] == "out" for c in chans):
                    # nothing moved: once more with receivers always ready
                    again = [_run(design, top, ports, clock, s, work,
                                  always_ready=True) for s in seeds]
            elif all(r["first"] >= 0 and r["still"] >= STILL for r in runs):
                # still to the end: slow, or stopped for good?
                longer = [_run(design, top, ports, clock, s, work,
                               cycles=CYCLES * LONGER) for s in seeds]
        except subprocess.TimeoutExpired:
            return {"result": "error", "why": f"iverilog ran past {TIMEOUT_S} s"}
    bad = [r["error"] for r in runs if "error" in r]
    if bad:
        return {"result": "error", "why": bad[0], "channels": chans}
    out = {"channels": chans, "runs": runs, "moved": min(r["moves"] for r in runs),
           "still": min(r["still"] for r in runs),
           "last_move": max(r["last"] for r in runs)}
    if all(r["first"] < 0 for r in runs):
        if again and all("error" not in r for r in again):
            out["always_ready_moved"] = min(r["moves"] for r in again)
            if out["always_ready_moved"] > 0:
                return dict(out, result="waits_for_ready")
        return dict(out, result="never_moved")
    if longer is None:
        return dict(out, result="moves")
    bad = [r["error"] for r in longer if "error" in r]
    if bad:
        return dict(out, result="error", why=bad[0])
    out["long_run_still"] = min(r["still"] for r in longer)
    frozen = out["long_run_still"] >= STILL * LONGER
    return dict(out, result="frozen" if frozen else "moves")


def lesson(result):
    """The retry's lesson for a design the check turns away."""
    sends = [c for c in result.get("channels", []) if c["side"] == "out"]
    named = ", ".join(f"{c['valid']}/{c['ready']}" for c in sends) or "its outputs"
    seen = (f"With every receiver that says ready only after it sees valid "
            f"(allowed: a receiver may wait to see an item before taking it) "
            f"on {named}, ")
    if result["result"] == "waits_for_ready":
        return seen + (
            f"nothing moved in the whole {CYCLES}-cycle run; with receivers "
            f"that are always ready, items moved. Something waits for an "
            f"output's ready before anything moves: raise each valid as soon "
            f"as its item is there, and take items while there is room, "
            f"without waiting for an output's ready.")
    return seen + (
        f"the design moved {result['moved']} items and then "
        f"nothing moved from cycle {result['last_move']} on, while new "
        f"items were offered, for the rest of the {CYCLES}-cycle run. "
        f"Something waits for a ready that only comes when another "
        f"output already has an item: empty each output with its own "
        f"ready, and keep taking items while the outputs are taken.")


def reason(result):
    """One line for the reviewer and the next question about the design."""
    if result["result"] == "waits_for_ready":
        return (f"waits for ready: with receivers that say ready only after "
                f"they see valid nothing moved; with receivers always ready, "
                f"{result['always_ready_moved']} items moved")
    return (f"froze: with receivers that say ready only after they see "
            f"valid, {result['moved']} items moved and then nothing from "
            f"cycle {result['last_move']} on")


KEPT = ("result", "moved", "still", "last_move", "long_run_still",
        "always_ready_moved", "why", "channels")
TURNED_AWAY = ("frozen", "waits_for_ready")


def gate(record, verilog, top, clock):
    """When the run switched the check on: check `verilog` once, keep what
    it saw as record["freeze_check"], and for a design that froze or waits
    for ready mark the record FREEZES, with the lesson as its error and a
    one-line reason, and return True (the step is sent back). Anything
    else changes nothing: a design that moves, has no handshake, never
    moved with either receiver or could not be simulated is not turned
    away by this check."""
    if not ON or "freeze_check" in record:
        return False
    result = check(verilog, top, clock)
    record["freeze_check"] = {k: result[k] for k in KEPT if k in result}
    if result["result"] not in TURNED_AWAY:
        return False
    record["verdict"] = "FREEZES"
    record["error"] = lesson(result)
    record["reason"] = reason(result)
    return True
