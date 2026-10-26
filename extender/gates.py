"""Stricter checks on a proved child (18 Oct 2026).

The checker proves what a design says, not whether it makes sense. Reading
all 28 children of the 17 Oct growth trial by hand found weak and wrong
designs it accepts; each check here rejects one kind, and together they
caught 5 of the 6 wrong and 8 of the 12 weak children while passing all 10
good ones (notes_2026-10-18/strict_checks):

  SHADOW_REGISTER  a fact says a new register equals an existing signal: the
                   new state only copies state the design already has
  IDLE_GLUE        a wrapper's own registers drive no input of the designs it
                   wraps, so its logic only watches them; for a combination,
                   an input of one design computed from the other's outputs
                   also counts (26 Oct 2026), a plain wire does not
  XOR_OUTPUTS      results of separate copies XOR-ed into one output, which
                   keeps them alive for the tools and means nothing
  UNUSED_COPIES    under 70% of the copied designs' flip-flops still do any
                   work once assertions are ignored: their results go
                   nowhere, or they never change (21 Oct 2026: the zeros of
                   a word padded into a wider input)
  TOO_MANY_FACTS   more than 64 facts, nearly always one fact list restated
                   per copy (one child reached 202)
  COPIED_TEXT      more than half of a step's new lines are copies of each
                   other: one block written once per slot (19 Oct 2026: two
                   designs of the 18 Oct trial, 61% and 83%)

Checks run on a NECESSARY child only, once (the result is kept on the
record), cheapest first; only UNUSED_COPIES needs yosys."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from build_corpus import (_counts, _mask_comments, copied_lines,
                          count_dff_bits)
from extend import _port_list_span, _split_outside_brackets, identifiers

VERDICTS = ("SHADOW_REGISTER", "IDLE_GLUE", "XOR_OUTPUTS", "UNUSED_COPIES",
            "TOO_MANY_FACTS", "COPIED_TEXT")
MAX_FACTS = 64
# 21 Oct 2026, kept at 0.7 after reading the two combinations of the 21 Oct
# trial turned away at 69%, one point under it (Rishab: "make all 1-8
# changes", item 6). Both padded a 64-bit word with 64 zeros into the
# 128-bit payload input of the same SPI chip-select sequencer, so its
# 24-entry queue held 1,536 flip-flops of zeros for ever; fed from a real
# source instead, both measure 101% (yosys, same steps as below). The check
# saw real waste, not a counting slip, and one of the two also lost data
# whenever its buffer was full (it only counted the loss). Neither was a
# good design wrongly turned away, so the line stays where it is.
MIN_STATE_USED = 0.7
MAX_COPIED = 0.5
MIN_LINES_FOR_COPIES = 20     # a short patch says too little to judge
WRAPPERS = ("compose", "replicate")

_KEYWORDS = {"input", "output", "inout", "wire", "reg", "logic", "signed",
             "unsigned", "integer", "bit"}


def _declaration_text(t, i):
    """From index i in masked text, the declaration up to the `;` that ends
    it, or up to the `)` closing a port list."""
    depth, j = 0, i
    while j < len(t):
        ch = t[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif ch == ";" and depth == 0:
            break
        j += 1
    return t[i:j]


def declared_state(text):
    """{name: width} of every reg/logic declared in `text`, port
    declarations included; width is the first packed range when it is
    numeric, else 1."""
    t = _mask_comments(text)
    out = {}
    for m in re.finditer(r"\b(?:reg|logic)\b", t):
        rest = _declaration_text(t, m.end()).strip()
        rest = re.sub(r"^signed\b\s*", "", rest)
        width = 1
        dims = re.match(r"((?:\[[^\]]*\]\s*)*)", rest).group(1)
        first = re.match(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", dims.strip())
        if first:
            width = abs(int(first.group(1)) - int(first.group(2))) + 1
        for item in _split_outside_brackets(rest[len(dims):]):
            s = item.strip()
            if re.match(r"(input|output|inout)\b", s):
                break
            name = re.match(r"([A-Za-z_]\w*)", s)
            if name and name.group(1) not in _KEYWORDS:
                out[name.group(1)] = width
    return out


def new_registers(record, parents):
    """State this step added: a wrapper's own registers, or the registers a
    patched child declares that no parent does."""
    if record.get("ext_type") in WRAPPERS:
        return set(declared_state(record.get("wrapper", "")))
    old = set()
    for p in parents:
        if p:
            old |= set(declared_state(p["verilog"]))
    return set(declared_state(record.get("child_verilog", ""))) - old


_ZERO_EXT = re.compile(r"\{\s*\d+'[bodhBODH]0+\s*,\s*([A-Za-z_]\w*)\s*\}")
_PLAIN_EQ = re.compile(r"^\(?\s*([A-Za-z_]\w*)\s*==\s*([A-Za-z_]\w*)\s*\)?$")


def shadow_facts(invariants, new_regs):
    """Facts `new == old` (zero-extension ignored) where exactly one side is
    a register this step added: the register copies existing state."""
    out = []
    for fact in invariants:
        m = _PLAIN_EQ.match(_ZERO_EXT.sub(r"\1", fact.strip()))
        if m and ((m.group(1) in new_regs) != (m.group(2) in new_regs)):
            out.append(fact)
    return out


def _modules(verilog):
    return re.findall(r"\bmodule\s+(\w+)", _mask_comments(verilog))


def instances(wrapper, child, top):
    """(module, instance, connection text) for every design the wrapper
    instantiates."""
    w = _mask_comments(wrapper)
    out = []
    for mod in _modules(child):
        if mod == top:
            continue
        for name, body in re.findall(
                r"\b%s\s+(?:#\s*\([^;]*?\)\s*)?(\w+)\s*\(([^;]*?)\)\s*;"
                % re.escape(mod), w, re.S):
            out.append((mod, name, body))
    return out


def port_directions(child, module):
    """{port: 'input' | 'output' | 'inout'} from an ANSI port list."""
    masked = _mask_comments(child)
    span = _port_list_span(masked, module)
    out, direction = {}, None
    if not span:
        return out
    for item in _split_outside_brackets(masked[span[0]:span[1]]):
        d = re.match(r"\s*(input|output|inout)\b", item)
        if d:
            direction = d.group(1)
        names = [x for x in re.findall(r"[A-Za-z_]\w*",
                                       re.sub(r"\[[^\]]*\]", " ", item))
                 if x not in _KEYWORDS]
        if names and direction:
            out[names[-1]] = direction
    return out


_CONNECTION = re.compile(r"\.(\w+)\s*\(([^()]*(?:\([^()]*\))?[^()]*)\)")
_DRIVEN = re.compile(r"(?:\bassign\s+|\bwire\s+(?:signed\s+)?"
                     r"(?:\[[^\]]*\]\s*)?)(\w+)\s*=\s*([^;]+);")


def glue_controls(wrapper, child, top):
    """Does anything the wrapper's own registers decide reach an input of
    a design it wraps?"""
    w = _mask_comments(wrapper)
    derived = set(declared_state(wrapper))
    driven = _DRIVEN.findall(w)
    while True:
        more = {lhs for lhs, rhs in driven
                if lhs not in derived and identifiers(rhs) & derived}
        if not more:
            break
        derived |= more
    for mod, _, body in instances(wrapper, child, top):
        dirs = port_directions(child, mod)
        for port, expr in _CONNECTION.findall(body):
            if dirs.get(port) == "input" and identifiers(expr) & derived:
                return True
    return False


# 26 Oct 2026, Rishab approved (change 4): a combination's part between the
# designs may be a conversion or an arbiter with no register, so IDLE_GLUE
# also accepts an input of one design computed from another's outputs; a
# plain wire, even through renaming wires, still fails.
def parts_linked(wrapper, child, top):
    """Does an input of one wrapped design take its value from another
    design's outputs through logic of the wrapper's own (a slice, an
    extension, a comparison, a mux), not only through wires that rename
    it?"""
    w = _mask_comments(wrapper)
    driven = dict(_DRIVEN.findall(w))
    parts = []
    for mod, name, body in instances(wrapper, child, top):
        dirs = port_directions(child, mod)
        conns = _CONNECTION.findall(body)
        outs = {i for p, e in conns if dirs.get(p) == "output"
                for i in identifiers(e)}
        parts.append((name, outs, [e for p, e in conns
                                   if dirs.get(p) == "input"]))
    return any(_computed_from(e, outs, driven)
               for name, outs, _ in parts
               for other, _, ins in parts if other != name for e in ins)


def _computed_from(expr, sources, driven, seen=()):
    """Does `expr` read a name in `sources` through an operation? A bare
    name is followed through its assign or wire line; a bare name that is
    a source is a plain wire."""
    e = expr.strip()
    if re.fullmatch(r"[A-Za-z_]\w*", e):
        if e in driven and e not in seen:
            return _computed_from(driven[e], sources, driven, seen + (e,))
        return False
    reached, todo = set(), set(identifiers(e))
    while todo:
        n = todo.pop()
        if n not in reached:
            reached.add(n)
            todo |= identifiers(driven.get(n, ""))
    return bool(reached & sources)


def xor_of_copies(wrapper, child, top):
    """`assign x = a ^ b ...` lines whose operands are outputs of two or
    more different wrapped designs."""
    owner = {}
    for mod, name, body in instances(wrapper, child, top):
        dirs = port_directions(child, mod)
        for port, expr in _CONNECTION.findall(body):
            if dirs.get(port) == "output" and re.fullmatch(r"\s*\w+\s*",
                                                           expr):
                owner[expr.strip()] = name
    out = []
    for line in re.findall(r"assign\s+\w+\s*=\s*([^;]+);",
                           _mask_comments(wrapper)):
        ops = [o.strip() for o in line.split("^")]
        if len(ops) >= 2 and all(re.fullmatch(r"\w+", o) for o in ops) and \
                len({owner[o] for o in ops if o in owner}) >= 2:
            out.append(line.strip())
    return out


def _flip_flops(verilog, top):
    """Flip-flop bits that still drive something once assertions are
    ignored, so a register kept only by a property does not count."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "design.sv"
        f.write_text(verilog)
        script = (f"read_verilog -sv -noassert -noassume {f}; "
                  f"hierarchy -top {top}; proc; flatten; opt; memory; opt; "
                  "techmap; opt; stat")
        try:
            out = subprocess.run(["yosys", "-p", script], capture_output=True,
                                 text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return None
    if out.returncode != 0:
        return None
    return count_dff_bits(out.stdout) or 0


def copied_state_used(child, top, wrapper):
    """Flip-flop bits of the whole design over the bits its parts would
    have on their own (each wrapped design times its copies, plus the
    wrapper's own registers). Near 1 means every copy's results are used."""
    if shutil.which("yosys") is None:
        return None
    count = {}
    for mod, _, _ in instances(wrapper, child, top):
        count[mod] = count.get(mod, 0) + 1
    expected = sum(declared_state(wrapper).values())
    for mod, n in count.items():
        bits = _flip_flops(child, mod)
        if bits is None:
            return None
        expected += bits * n
    actual = _flip_flops(child, top)
    if actual is None or not expected:
        return None
    return round(actual / expected, 3)


def new_lines(record):
    """The lines a step wrote: a patch's + lines, or a wrapper."""
    if record.get("ext_type") in WRAPPERS:
        return (record.get("wrapper") or "").splitlines()
    return [line[1:] for line in (record.get("patch") or "").splitlines()
            if line.startswith("+")]


def copied_share(record):
    """Share of a step's new lines that are copies of each other, or None
    when it wrote too little to judge. Copies (replicate) instantiate one
    design several times on purpose and are not judged."""
    if record.get("ext_type") == "replicate":
        return None
    lines = [line for line in _mask_comments(
        "\n".join(new_lines(record))).splitlines() if _counts(line)]
    if len(lines) < MIN_LINES_FOR_COPIES:
        return None
    return round(copied_lines(lines) / len(lines), 3)


def _fail(record, name, detail):
    record["gates"] = {"failed": name, "detail": detail}
    record["verdict"] = name
    record["reason"] = detail
    return record


def apply(record, parents, use_yosys=True):
    """Run the checks on a proved child, once. A failed check replaces
    the verdict with its own name (routed as a rejection by promote)."""
    if record.get("verdict") != "NECESSARY":
        return record
    if "gates" in record:
        return record
    facts = record.get("invariants") or []
    # one compact fact for every entry counts once (21 Oct 2026)
    written = record.get("invariants_compact") or facts
    if len(written) > MAX_FACTS:
        return _fail(record, "TOO_MANY_FACTS",
                     f"{len(written)} facts, more than {MAX_FACTS}")
    shadows = shadow_facts(facts, new_registers(record, parents))
    if shadows:
        return _fail(record, "SHADOW_REGISTER",
                     "a new register only copies existing state: "
                     + "; ".join(shadows))
    copied = copied_share(record)
    if copied is not None and copied > MAX_COPIED:
        return _fail(record, "COPIED_TEXT",
                     f"{copied:.0%} of the new lines are copies of each "
                     "other (one block written once per slot)")
    result = {"failed": None, "copied_share": copied}
    if record.get("ext_type") in WRAPPERS and record.get("wrapper"):
        child, wrapper = record["child_verilog"], record["wrapper"]
        top = _modules(child)[-1]
        # a combination may link its parts by a conversion or an arbiter
        # with no register (26 Oct 2026, Rishab approved, change 4); copies
        # keep the old rule and reason
        compose = record.get("ext_type") == "compose"
        if not glue_controls(wrapper, child, top) and not (
                compose and parts_linked(wrapper, child, top)):
            return _fail(record, "IDLE_GLUE",
                         "the wrapper's registers drive no input of the "
                         "designs it wraps, and no input of one design is "
                         "computed from the other's outputs: the parts sit "
                         "side by side, or are joined by a plain wire"
                         if compose else
                         "the wrapper's registers drive no input of the "
                         "designs it wraps")
        mixed = xor_of_copies(wrapper, child, top)
        if mixed:
            return _fail(record, "XOR_OUTPUTS",
                         "copies' results XOR-ed together: "
                         + "; ".join(mixed))
        if use_yosys:
            used = copied_state_used(child, top, wrapper)
            result["copied_state_used"] = used
            if used is not None and used < MIN_STATE_USED:
                # what yosys leaves out: flip-flops that drive nothing,
                # never change or repeat another (21 Oct 2026)
                return _fail(record, "UNUSED_COPIES",
                             f"only {used:.0%} of the copied designs' "
                             "flip-flops still do any work once they are "
                             "wired together: the others drive no output, "
                             "never change (such as the zeros of a word "
                             "padded into a wider input) or repeat another "
                             "flip-flop")
    record["gates"] = result
    return record
