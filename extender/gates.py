"""Stricter checks on a proved child (18 Oct 2026).

The checker proves what a design says, not whether it makes sense. Reading
all 28 children of the 17 Oct growth trial by hand found weak and wrong
designs it accepts; each check here rejects one kind, and together they
caught 5 of the 6 wrong and 8 of the 12 weak children while passing all 10
good ones (notes_2026-10-18/strict_checks):

  SHADOW_REGISTER  a fact says a new register equals an existing signal: the
                   new state only copies state the design already has
  IDLE_GLUE        a wrapper's own registers drive no input of the designs it
                   wraps, so its logic only watches them
  XOR_OUTPUTS      results of separate copies XOR-ed into one output, which
                   keeps them alive for the tools and means nothing
  UNUSED_COPIES    under 70% of the copied designs' flip-flops still drive an
                   output once assertions are ignored: their results go nowhere
  TOO_MANY_FACTS   more than 64 facts, nearly always one fact list restated
                   per copy (one child reached 202)

Checks run on a NECESSARY child only, once (the result is kept on the
record), cheapest first; only UNUSED_COPIES needs yosys."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from build_corpus import _mask_comments, count_dff_bits
from extend import _port_list_span, _split_outside_brackets, identifiers

VERDICTS = ("SHADOW_REGISTER", "IDLE_GLUE", "XOR_OUTPUTS", "UNUSED_COPIES",
            "TOO_MANY_FACTS")
MAX_FACTS = 64
MIN_STATE_USED = 0.7
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
    if len(facts) > MAX_FACTS:
        return _fail(record, "TOO_MANY_FACTS",
                     f"{len(facts)} facts, more than {MAX_FACTS}")
    shadows = shadow_facts(facts, new_registers(record, parents))
    if shadows:
        return _fail(record, "SHADOW_REGISTER",
                     "a new register only copies existing state: "
                     + "; ".join(shadows))
    result = {"failed": None}
    if record.get("ext_type") in WRAPPERS and record.get("wrapper"):
        child, wrapper = record["child_verilog"], record["wrapper"]
        top = _modules(child)[-1]
        if not glue_controls(wrapper, child, top):
            return _fail(record, "IDLE_GLUE",
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
                return _fail(record, "UNUSED_COPIES",
                             f"only {used:.0%} of the copied designs' "
                             "flip-flops drive an output")
    record["gates"] = result
    return record
