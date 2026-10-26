"""The checker's own nothing-is-lost search (26 Oct 2026).

Rishab approved (changes 2, 3, 5 and 7). The root-cause study of 25 Oct
2026 found 8 of 14 combined and second-lane designs losing data: an item
handed over was dropped when a part was full or busy, overwritten before
it was used, or ignored while a part waited, and 5 of the 8 were kept
because the loss was counted or flagged. Counting what the SENDER hands
over caught all 8, with no false alarm on the 5 good designs; counting
only the items the design chose to take caught none. So the model only
says where the items go (the "streams" a big step or a combination
declares), and the checker counts them itself, in a copy of the design:

  declaration_error    the rules a declaration must meet, read before any
                       tool runs (STREAM_INVALID)
  frozen_enable_error  a combination that drives a part's enable with
                       logic while that part's other control inputs come
                       straight from the wrapper's inputs, so settings,
                       table writes or requests sent during a pause are
                       lost (t033, c2_004; PAUSES_WHOLE_DESIGN)
  checked_copy         the design without its own assertions, plus the
                       checker's counters: for each stream, the items
                       handed over equal the items delivered plus the items
                       still held, and a sender inside the design that was
                       told to wait keeps its item
  check                a bounded search of that copy from the first
                       cycle, 48 to 80 cycles deep, with two engines,
                       before the proof (ITEMS_LOST; STREAM_INVALID when
                       the counts add up the other way or the copy does not
                       build). A search that cannot run or runs out of time
                       changes nothing: a non-verdict says nothing about
                       the design.
  reviewer_notes       what the search found, in plain lines for the
                       reviewer (sense_check.py)

extend.py calls declaration_error, frozen_enable_error and check from
grade_step4 and grade_compose, importing this module inside those
functions, since this module imports extend. It reads reset_reads.py's
tokens and its tracing of signals as they are: the generator shares that
file.
"""
import json
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import extend                                                   # noqa: E402
import gates                                                    # noqa: E402
import reset_reads                                              # noqa: E402
from build_corpus import _mask_comments                         # noqa: E402
from distractor import Spinner, grade_kwargs                    # noqa: E402
from fact_format import SYSTEM_FUNCTION                         # noqa: E402
from oracle.contract import _HIER_REF, _PROSE                   # noqa: E402
from oracle.inject import (InjectionError, _insert_point,       # noqa: E402
                           strip_assertions)
from oracle.parse import parse_assert_failures                  # noqa: E402
from oracle.sby import run_sby, sby_available                   # noqa: E402

# 26 Oct 2026, Rishab approved (change 3, depth 48 to 80): a real loss
# (sys_c1_007's) shows only at cycle 39: the search missed it at depth 30
# and found it at 48 in 3.0 s; deeper new storage searches longer
# (search_depth), never past 80
NOTHING_LOST_MIN_DEPTH = 48
NOTHING_LOST_MAX_DEPTH = 80
# GRADE_KWARGS["pdr_timeout_s"]; like that limit it cannot make a design
# pass: a search that runs out of time changes nothing
NOTHING_LOST_TIMEOUT_S = 120
NOTHING_LOST_ENGINES = "smtbmc yices\nabc bmc3"   # first answer wins
# no trace rebuild: abc found the loss in 0.01 s on all four real designs,
# but rebuilding its trace ran past the 120 s limit each time
NOTHING_LOST_OPTIONS = ("vcd off",)
# tests point it at tmp_path
NOTHING_LOST_ROOT = HERE.parent / "runs" / "nothing_lost"
MAX_STREAMS, MAX_INPUTS, MAX_EXPRESSION = 3, 4, 300
PREFIX = "oracle_nl"
# 190 of the 228 starting designs and 14 of the 32 new ones name their
# enable en, i_en, enable, ce or i_ce; 24 of the other 38 old ones use step
ENABLE_NAME = re.compile(
    r"(?i)^(i_)?(en|ena|enable|ce|cke|clken|clk_en|step|tick)(_i)?$")
RESET_WORDS = frozenset({"rst", "rstn", "reset", "resetn", "areset",
                         "aresetn", "arst", "arstn", "srst", "srstn", "nrst",
                         "nreset"})
LOSS_WORDS = frozenset({"drop", "drops", "dropped", "lost", "loss",
                        "overflow", "ovf", "overrun", "discard",
                        "discarded"})

# what each result of the search does to the step: a result missing here
# (nothing_lost, unfinished, error, not_run) changes nothing
VERDICT_OF = {"lost": "ITEMS_LOST", "gave_up": "ITEMS_LOST",
              "unattributed": "ITEMS_LOST", "count_off": "STREAM_INVALID",
              "build_error": "STREAM_INVALID"}
# the result kept on the record as record["nothing_lost"]; "input" is the
# input whose sender gave up its item (gave_up only)
RESULT_KEYS = ("result", "depth", "seconds", "reset", "step", "stream",
               "input", "checked_to", "why", "workdir")
HEADER = ("// ---- NOTHING-LOST CHECK: written by the checker, not part of "
          "the design ----")
SHOWN = 100      # most characters of one value a lesson repeats


# --- reading the design ------------------------------------------------------

def _words(name):
    """The words of a signal name, split on "_" and digits, lower case."""
    return {w for w in re.split(r"[_\d]+", name.lower()) if w}


def is_reset_name(name):
    """Does a word of `name` name a reset (rst_n, i_reset, areset_n)?
    Words are split on "_" and digits, so "burst_len" does not."""
    return bool(_words(name) & RESET_WORDS)


def top_text(verilog, top):
    """The top module's own text, comments masked ("" when it is
    missing)."""
    m = re.search(r"\bmodule\s+%s\b(.*?)\bendmodule\b" % re.escape(top),
                  _mask_comments(verilog), re.S)
    return f"module {top}{m.group(1)}endmodule\n" if m else ""


# a port connection `.name(`: the port's name belongs to the module the
# instance is of, not to the top module, so it is masked before tracing
_PORT_NAME = re.compile(r"\.\s*[A-Za-z_]\w*\s*\(")


@lru_cache(maxsize=16)
def _carriers(text):
    """(((names set, names read), ...), every name set) for each way a
    signal of the top module's `text` takes its value in the same cycle
    (reset_reads._carriers, port connections masked)."""
    masked = _PORT_NAME.sub("(", text)
    _, toks = reset_reads._tokens(masked)
    pairs = tuple((frozenset(sets), frozenset(reads))
                  for sets, reads in reset_reads._carriers(masked, toks))
    return pairs, frozenset().union(*(sets for sets, _ in pairs))


def same_cycle_sources(expr, text):
    """Every name `expr` reads in the same cycle: its own names and,
    through the top module's assigns, wires declared with a value and
    combinational blocks, what those are made of, down to registers, ports
    and instance outputs. A combinational block counts every signal it
    sets as made of everything it reads. That reads more than it must,
    which is the safe side for the rules below."""
    pairs, _ = _carriers(text)
    reached = set(extend.identifiers(expr))
    grown = True
    while grown:
        grown = False
        for sets, reads in pairs:
            if sets & reached and not reads <= reached:
                reached |= reads
                grown = True
    return reached


def _leaves(expr, text):
    """The names `expr` reads in the same cycle that nothing in the top
    module sets in that cycle: registers, ports, instance outputs and
    parameters."""
    return same_cycle_sources(expr, text) - _carriers(text)[1]


def ports(verilog, top):
    """(input ports, output ports) of module `top`."""
    dirs = gates.port_directions(verilog, top)
    return ({p for p, d in dirs.items() if d == "input"},
            {p for p, d in dirs.items() if d == "output"})


def _parameters(text):
    return {name for name, _, _ in extend.module_parameters(text)}


def _outside(offer, text, inputs, params):
    """Does the sender come from outside the design: is everything the
    offer reads in the same cycle an input port or a parameter? An offer
    made of input ports through a wire of the design counts too, since
    such a sender can still withdraw its item (26 Oct 2026)."""
    return _leaves(offer, text) <= (inputs | params)


def _parts(wrapper, child, top):
    """{instance: {"module", "outs": the wrapper's names on its output
    ports, "ins": [(port, connection text)] of its input ports}} for the
    designs a combination's wrapper instantiates."""
    out = {}
    for mod, inst, body in gates.instances(wrapper, child, top):
        dirs = gates.port_directions(child, mod)
        conns = gates._CONNECTION.findall(body)
        out[inst] = {
            "module": mod,
            "outs": {i for p, e in conns if dirs.get(p) == "output"
                     for i in extend.identifiers(e)},
            "ins": [(p, e.strip()) for p, e in conns
                    if dirs.get(p) == "input"]}
    return out


def _added_arrays(child, top, parent_verilog):
    """{(module, array): entries} for the arrays of the design built from
    `top` that the parent did not have."""
    before = extend.parent_sizes(parent_verilog, [top]) \
        if parent_verilog else {}
    return {key: n for key, n in extend.storage_depths(child, top).items()
            if key not in before}


# --- the declaration ---------------------------------------------------------
# 26 Oct 2026, Rishab approved (change 3): the rules D1 to D19 of the
# spec, read before any tool. D14 closes the study's miss: an offer that
# folds in the new state its count reads counts only what the design chose
# to take, which hid every one of the 8 losses.

class _Shape(Exception):
    """A declaration whose shape is not the schema's."""


def _clean(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _Shape
    return value.strip()


def _normalized(streams):
    """The declared streams with every value a stripped string ("" for one
    missing or null), each stream named ("stream 2" when it has no name;
    a name cut to 80 characters, since lessons repeat it) and its inputs a
    list ([] when missing), or None when the shape is not the schema's.
    Strict structured output always gives the schema's shape; this guards
    the answers of older runs and tests."""
    if not isinstance(streams, list):
        return None
    out = []
    try:
        for i, s in enumerate(streams):
            if not isinstance(s, dict):
                raise _Shape
            ins = s.get("inputs")
            ins = [] if ins is None else ins
            if not isinstance(ins, list) or \
                    not all(isinstance(e, dict) for e in ins):
                raise _Shape
            out.append({
                "name": _short(_clean(s.get("name")), 80)
                or f"stream {i + 1}",
                "inputs": [{"offer": _clean(e.get("offer")),
                            "ready": _clean(e.get("ready"))} for e in ins],
                "delivered": _clean(s.get("delivered")),
                "held": _clean(s.get("held")),
                "flush": _clean(s.get("flush"))})
    except _Shape:
        return None
    return out


def _short(text, n=SHOWN):
    """`text` on one line, cut to `n` characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


def _balanced(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _subtracts(text):
    """A minus outside an index bracket ([3:0] and [rp[1:0]-1] are
    fine)."""
    inner = re.compile(r"\[[^\[\]]*\]")
    while inner.search(text):
        text = inner.sub(" ", text)
    return "-" in text


_ASSIGN = re.compile(r"(?<![=!<>])=(?!=)")


def _form_error(text):
    """Why `text` is not a plain same-cycle expression (D5), or None."""
    if len(text) > MAX_EXPRESSION:
        return (f"it is longer than {MAX_EXPRESSION} characters; name a "
                "wire of the design instead")
    if ";" in text:
        return "it has a ;"
    if SYSTEM_FUNCTION.search(text):
        return "it uses a $ function"
    if _HIER_REF.search(text):
        return "it names an instance (instance.signal)"
    if _PROSE.search(text):
        return "it is prose"
    if not _balanced(text):
        return "its brackets do not match"
    if _ASSIGN.search(text):
        return "it assigns with ="
    return None


def _value_error(name, field, value, child, top, clock):
    """D4 to D7 for one value of stream `name`, or None."""
    if not value:
        if field in ("offer", "delivered", "held"):
            return (f"stream '{name}': {field} is empty. offer, delivered "
                    "and held must each be a Verilog expression (\"0\" for "
                    "held when the stream holds nothing).")
        return None
    shown = _short(value)
    why = _form_error(value)
    if why:
        return (f"stream '{name}': {field} ({shown}) is not a plain "
                "same-cycle Verilog expression over the top module's "
                f"signals: {why}.")
    ids = extend.identifiers(value)
    if field == "offer" and not ids:
        return f"stream '{name}': offer ({shown}) reads no signal."
    if field in ("delivered", "held") and _subtracts(value):
        return (f"stream '{name}': {field} ({shown}) subtracts. The checker "
                "reads counts at 16 bits, where a difference of wrapping "
                "pointers comes out wrong: name a wire of the design that "
                "holds the difference at the pointers' width (wire [2:0] "
                "used = wp - rp;).")
    if clock in ids:
        return f"stream '{name}': {field} reads the clock {clock}."
    unknown = sorted(extend.out_of_scope([value], child, top))
    if unknown:
        return (f"stream '{name}': {field} names {', '.join(unknown)}, which "
                f"module {top} does not declare. Use the names the design "
                "declares (a register of a sub-module is reached through a "
                "wire on its output port).")
    return None


def _touches(names, part):
    """Do `names` reach a part: one of its output wires, or a name in one
    of its input connections?"""
    in_names = {i for _, e in part["ins"] for i in extend.identifiers(e)}
    return bool(names & (part["outs"] | in_names))


def declaration_error(streams, kind, child, top, clock, parent=None,
                      wrapper=None):
    """Why the streams a step declares break the rules (STREAM_INVALID),
    or None. `kind` is the move ("STORAGE", "PEER", "SUBUNIT") or
    "compose"; `parent` is the parent row of a patch and `wrapper` a
    combination's own module text. The rules read the stream, never which
    design is A or B, so they hold whichever way a combination's plan
    sends (the plan reverses the direction in 9 of 21 cases). The first
    rule broken is the lesson; each is under batch.LESSON_MAX."""
    shaped = _normalized(streams)
    if shaped is None:                                              # D1
        return ('"streams" must be a list of the streams your step adds or '
                "changes, as the question explains.")
    streams = shaped
    if len(streams) > MAX_STREAMS:                                  # D2
        return (f"Declare at most {MAX_STREAMS} streams; this answer "
                f"declares {len(streams)}.")
    text = top_text(child, top)
    t_in, t_out = ports(child, top)
    t_in.discard(clock)
    params = _parameters(text)
    if wrapper is not None:
        new_state = extend.wrapper_regs(wrapper)
        parts = _parts(wrapper, child, top)
        new_inputs = set()
    else:
        new_state = extend.state_carriers(parent, child) if parent else set()
        parts = {}
        new_inputs = t_in - ports(parent["verilog"], top)[0] \
            if parent else set()

    for s in streams:                                               # D3-D7
        name, ins = s["name"], s["inputs"]
        if not ins:
            return (f"stream '{name}' has no inputs: give one entry for "
                    "each place items enter it, with offer and ready.")
        if len(ins) > MAX_INPUTS:
            return (f"stream '{name}' has {len(ins)} inputs; at most "
                    f"{MAX_INPUTS}.")
        fields = ([("offer", e["offer"]) for e in ins]
                  + [("ready", e["ready"]) for e in ins]
                  + [(f, s[f]) for f in ("delivered", "held", "flush")])
        for field, value in fields:
            err = _value_error(name, field, value, child, top, clock)
            if err:
                return err

    # what each kind of step must declare
    if kind == "STORAGE" and not streams:                           # D8
        return ("A STORAGE step must declare at least one stream through "
                "its storage (for a register file or lookup table, its "
                "writes); this answer declares none.")
    if kind == "SUBUNIT" and not streams and parent:                # D9
        added = sorted(a for _, a in _added_arrays(child, top,
                                                   parent["verilog"]))
        if added:
            return (f"This step adds the array {added[0]}, so it must "
                    "declare at least one stream through the new unit (for "
                    "a register bank, its writes).")
    if kind == "PEER":                                              # D10
        def reads_new(offer):
            return bool(same_cycle_sources(offer, text) & new_inputs)
        if not any(len(s["inputs"]) >= 2
                   and any(reads_new(e["offer"]) for e in s["inputs"])
                   and any(not reads_new(e["offer"]) for e in s["inputs"])
                   for s in streams):
            return ("A PEER step must declare the stream through the thing "
                    "its two lanes share, with at least two inputs: the new "
                    "lane's, whose offer reads its own new input ("
                    + (", ".join(sorted(new_inputs)) or "none added yet")
                    + "), and the existing lane's, whose offer reads the "
                    "inputs it had before.")
    if kind == "compose":                                           # D11
        def crosses(s):
            # the top-level inputs are left out: every part shares the
            # wrapper's enable and reset, which say nothing about whether
            # the receiver takes the item
            given = same_cycle_sources(s["delivered"], text) - t_in
            for e in s["inputs"]:
                taken = same_cycle_sources(e["offer"], text)
                for x, part in parts.items():
                    if part["outs"] & taken and any(
                            _touches(given, other)
                            for y, other in parts.items() if y != x):
                        return True
            return False
        if not any(crosses(s) for s in streams):
            return ("A combination must declare at least one stream from "
                    "one design into the other: an input whose offer reads "
                    "the sending design's output wire, and a delivered that "
                    "counts the receiving design taking the item.")

    for s in streams:                                               # D12-D19
        name = s["name"]
        counted = (same_cycle_sources(s["delivered"], text)
                   | same_cycle_sources(s["held"], text))
        stream_names = set()
        for e in s["inputs"]:
            offer, ready = e["offer"], e["ready"]
            r_ids = extend.identifiers(ready)
            stream_names |= extend.identifiers(offer) | r_ids
            if _outside(offer, text, t_in, params):
                if ready and not (r_ids <= (t_in | t_out | params)
                                  and r_ids & t_out):              # D12
                    return (f"stream '{name}': the offer {_short(offer)} "
                            "comes from outside the design (it reads only "
                            "input ports), and such a sender sees only the "
                            "design's ports: ready must read an output port "
                            'it can see ("in_ready", "!busy_o") and nothing '
                            'but ports, or be "" when the sender cannot be '
                            f"told to wait; {_short(ready)} does not.")
                continue
            o_src = same_cycle_sources(offer, text)
            if ready and o_src & t_in:                              # D13
                return (f"stream '{name}': the offer {_short(offer)} reads "
                        "the input port(s) "
                        f"{', '.join(sorted(o_src & t_in))}. For a sender "
                        "inside the design, offer is the sender's own "
                        "valid, which stays true while it waits; put the "
                        "design's enable in ready instead.")
            shared = sorted(o_src & new_state & counted)
            # 27 Oct 2026, Rishab approved: the last sentence covers a
            # sender that is the step's own new logic, whose own valid is
            # refused here too (GPT-6.1 Sol's journal and state machine)
            if shared:                                              # D14
                return (f"stream '{name}': the offer {_short(offer)} reads "
                        f"{', '.join(shared)}, which delivered or held also "
                        "reads, so it folds in whether the item will be "
                        "taken: an offer that counts only the items the "
                        "design chose to take hides every drop. Write the "
                        "sender's own valid in offer and put what makes it "
                        "wait in ready. When the sender is logic your step "
                        "adds, start the stream where items come into that "
                        "logic, or leave that stream out.")
            if wrapper is None:
                continue
            senders = sorted(k for k, v in parts.items() if v["outs"] & o_src)
            if len(senders) > 1:                                    # D15
                return (f"stream '{name}': the offer {_short(offer)} reads "
                        f"outputs of {' and '.join(senders)}; it must read "
                        "only the sending design's own outputs: reading the "
                        "receiver folds in whether it can take the item.")
            if senders and ready:                                   # D16
                waits = same_cycle_sources(ready, text) - t_in
                if not any(same_cycle_sources(x, text) & waits
                           for _, x in parts[senders[0]]["ins"]):
                    return (f"stream '{name}': ready {_short(ready)} reaches "
                            f"no input of {senders[0]}, the design that sends "
                            "the items, so it cannot make it wait. Name what "
                            "drives its ready, busy, request or valid input, "
                            'or write "" if it cannot wait.')
        held_in = sorted(extend.identifiers(s["held"]) & t_in)
        if held_in:                                                 # D17
            return (f"stream '{name}': held {_short(s['held'])} reads the "
                    f"input port(s) {', '.join(held_in)}; held counts the "
                    "items inside the stream, so it reads only the design's "
                    "own state.")
        for field in ("delivered", "held"):                        # D18
            loss = sorted(x for x in extend.identifiers(s[field])
                          if _words(x) & LOSS_WORDS)
            if loss:
                return (f"stream '{name}': {field} names {loss[0]}, a signal "
                        "that counts or flags lost items. A lost item is "
                        "neither delivered nor held: never count it there, "
                        "and make the sender wait instead of losing it.")
        flush = s["flush"]
        if flush:                                                   # D19
            bad = sorted(x for x in extend.identifiers(flush)
                         if x not in t_in or ENABLE_NAME.match(x)
                         or is_reset_name(x) or x in stream_names)
            if bad:
                return (f"stream '{name}': flush {_short(flush)} must read "
                        "only input ports that empty the stream on purpose "
                        "(a clear, abort or flush command), never the "
                        "enable, the reset or the stream's own offer or "
                        f"ready signals; it reads {', '.join(bad)}.")
    return None


# --- the frozen-enable rule (combinations) -----------------------------------
# 26 Oct 2026, Rishab approved (change 2, kept strict): the advice to hold
# the sender's enable low froze whole designs; t033 and c2_004 lost the
# route-table writes sent during a pause. On the 40 logged combinations
# the rule fires on 10, both hand-labelled ones among them (t018 "wrong",
# t021 "weak"); 11 before a part's input wired to the wrapper's reset was
# left out (21 Oct t032's .i_resync(rst)).

_ONE = re.compile(r"(?:\d*'[bBhHdDoO])?0*1|'1")
_ZERO = re.compile(r"(?:\d*'[bBhHdDoO])?0+|'0")


def _renames(wrapper):
    """{wire: the one name it is assigned} for the wrapper's wires that
    only rename another signal."""
    return {lhs: rhs.strip()
            for lhs, rhs in gates._DRIVEN.findall(_mask_comments(wrapper))
            if re.fullmatch(r"\s*[A-Za-z_]\w*\s*", rhs)}


def _straight_input(expr, inputs, renames):
    """The wrapper input `expr` is, as written or through wires that only
    rename it, or None."""
    name, seen = expr.strip(), set()
    while re.fullmatch(r"[A-Za-z_]\w*", name) and name not in seen:
        if name in inputs:
            return name
        seen.add(name)
        name = renames.get(name, "")
    return None


def frozen_enable_error(wrapper, child, top, clock):
    """Why a combination pauses a whole design (PAUSES_WHOLE_DESIGN), or
    None. For each part, each 1-bit input whose name looks like an enable
    (ENABLE_NAME) must come straight from one wrapper input or be a
    constant one, unless no other 1-bit input of that part comes straight
    from the wrapper's inputs (the clock, resets and other enables left
    out): a part fed wholly by the stream may be paused through its
    enable. It reads port names: a heuristic for a code guard, not a
    judgement of behaviour. On the 14 study combinations it turned away
    the 4 frozen enables a netlist reader found (t033, c1_004, c1_007,
    c2_004) and the skeptic's rebuild of sys_c1_003, and passed the other
    9."""
    t_in, _ = ports(child, top)
    renames = _renames(wrapper)
    for mod, inst, body in gates.instances(wrapper, child, top):
        dirs = gates.port_directions(child, mod)
        one_bit = {p for p in extend.port_summary(child, mod)[0]
                   if "[" not in p}
        conns = [(p, " ".join(e.split()))
                 for p, e in gates._CONNECTION.findall(body)
                 if dirs.get(p) == "input" and p in one_bit]
        straight = {p: _straight_input(e, t_in, renames) for p, e in conns}
        # the part's other 1-bit inputs that come straight from the
        # wrapper's inputs, the clock, resets and enables left out
        others = sorted(q for q, _ in conns
                        if straight[q] and clock not in (q, straight[q])
                        and not is_reset_name(q)
                        and not is_reset_name(straight[q])
                        and not ENABLE_NAME.match(q))
        for port, expr in conns:
            if not ENABLE_NAME.match(port) or straight[port] \
                    or _ONE.fullmatch(expr):
                continue
            if others:
                return (f"{inst} ({mod})'s input {port} looks like its "
                        "enable, the input that leaves all of its state "
                        "unchanged while it is low. The wrapper drives it "
                        f"with {_short(expr)}, so while that is low {mod} "
                        f"also ignores {', '.join(others)}, which come "
                        "straight from the wrapper's inputs: settings, table "
                        "writes or requests sent then are lost. Connect "
                        f"{port} straight to the wrapper's own enable input. "
                        "Make the sender wait on the stream's own path "
                        "instead (its ready or busy input, or the request, "
                        "start or valid input that begins its next item), "
                        "show the wait on a ready output of the wrapper, and "
                        f"feed {mod} its items through its own valid, start "
                        f"or data inputs, never through {port}.")
    return None


# --- the reset, the depth and the copy ---------------------------------------

def _compared_to(after):
    """The value a name is compared to with == or === (the tokens after
    it), or None."""
    if after[:2] != ["=", "="]:
        return None
    at = 3 if after[2:3] == ["="] else 2
    return after[at] if len(after) > at else None


def _active_low(name, toks):
    """Is reset `name` active low? An edge decides (negedge: low; posedge:
    high); otherwise the design's tests of it vote; on a tie a name ending
    in n, _n, _ni, _b or _l, or starting with n, is active low."""
    at = [k for k, t in enumerate(toks) if t == name]
    if any(k and toks[k - 1] == "negedge" for k in at):
        return True
    if any(k and toks[k - 1] == "posedge" for k in at):
        return False
    low = high = 0
    for k in at:
        before, after = toks[max(0, k - 2):k], toks[k + 1:k + 5]
        if before in (["(", "!"], ["(", "~"]) and after[:1] == [")"]:
            low += 1                                    # ( ! X )   ( ~ X )
        if before == ["if", "("] and (after[:1] == [")"] or after[:2] in (
                ["|", "|"], ["&", "&"])):
            high += 1                       # if ( X )   if ( X ||   if ( X &&
        value = _compared_to(after)
        if value is not None and _ZERO.fullmatch(value):
            low += 1                                  # X == 1'b0   X == 0
        elif value is not None and _ONE.fullmatch(value):
            high += 1                                 # X == 1'b1   X == 1
    if low != high:
        return low > high
    return bool(re.search(r"(n|_n|_ni|_b|_l)$", name)
                or name.lower().startswith("n"))


def reset_condition(child, top, clock):
    """The condition under which the design is in reset during the search,
    or None: the top module's 1-bit inputs other than the clock that are
    asynchronous resets anywhere in the design or have a reset word in
    their name, each with its polarity read from the whole design (so a
    wrapper's rst is read in the parents), joined with ||. The search holds
    it in the first cycle only. Clear, abort and flush inputs and the
    enable stay free. Kept on the record, since a wrong guess gives a
    false "lost"."""
    inputs = {p for p in extend.port_summary(child, top)[0]
              if "[" not in p} - {clock}
    _, toks = reset_reads._tokens(child)
    _, async_resets = reset_reads._clocks_and_resets(toks, clock)
    conds = [f"(!{x})" if _active_low(x, toks) else f"({x})"
             for x in sorted(inputs)
             if x in async_resets or is_reset_name(x)]
    return " || ".join(conds) or None


def search_depth(child, top, parents, wrapper=None):
    """How many cycles the search looks: 4 for each entry of the biggest
    array the step added, plus 8, between NOTHING_LOST_MIN_DEPTH and
    NOTHING_LOST_MAX_DEPTH (a 4-entry queue gets 48, a 16-entry one 72).
    For a patch the arrays the parent (parents[0]) did not have; for a
    combination the arrays of the wrapper module itself."""
    if wrapper is not None:
        new = [n for (mod, _), n in extend.storage_depths(child, top).items()
               if mod == top]
    else:
        rows = [p for p in parents or [] if p and p.get("verilog")]
        new = list(_added_arrays(child, top,
                                 rows[0]["verilog"] if rows else "").values())
    cap = max(new, default=0)
    return min(NOTHING_LOST_MAX_DEPTH,
               max(NOTHING_LOST_MIN_DEPTH, 4 * (cap + 2)))


def _expr(value):
    """A declared value as one line of the copy: comments dropped and
    whitespace joined, so every line of the check stays where the line map
    says."""
    return " ".join(_mask_comments(value).split())


def _operand_end(text, i):
    """Where the operand that starts at or after text[i] ends (a name with
    its index brackets, a bracketed expression, or a not of either), or
    None for anything else (a number)."""
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return None
    if text[i] in "({":
        close, depth = {"(": ")", "{": "}"}[text[i]], 0
        for k in range(i, len(text)):
            if text[k] == text[i]:
                depth += 1
            elif text[k] == close:
                depth -= 1
                if depth == 0:
                    return k + 1
        return None
    if text[i] in "!~":
        return _operand_end(text, i + 1)
    m = re.match(r"[A-Za-z_]\w*(?:\s*\[[^\]]*\])*", text[i:])
    return i + m.end() if m else None


def _self_sized_nots(value):
    """`value` with each bitwise not of a name or a bracketed expression
    put inside braces ({~empty}), where Verilog sizes it by itself. The
    checker reads delivered and held at 16 bits so that sums of 1-bit
    flags add up, and at 16 bits ~ of a 1-bit flag would come out as 65534
    or 65535 (26 Oct 2026). A reduction (~& ~| ~^) and an exclusive nor
    (^~) are left as they are."""
    out, i = [], 0
    while i < len(value):
        if value[i] == "~" and value[i + 1:i + 2] not in ("&", "|", "^") \
                and value[i - 1:i] != "^":
            end = _operand_end(value, i + 1)
            if end is not None:
                out.append("{~" + value[i + 1:end].strip() + "}")
                i = end
                continue
        out.append(value[i])
        i += 1
    return "".join(out)


def checked_copy(child, top, clock, streams, reset):
    """(the copy the search runs on, its line map {line: (stream index,
    "lost" | "count_off" | "gave_up", input index or None)}, the first and
    last line of the checker's block). The copy is the design without its
    own assertions, so no other failure can stand in for a loss, plus a
    block before the top module's endmodule that counts, for each stream,
    the items handed over (offer and ready both true) against the items
    delivered plus the items still held, at 16 bits (they cannot wrap in
    80 cycles), and for each input from inside the design with a ready,
    that a sender told to wait still offers its item the cycle after. The
    reset, when found, is held in the first cycle only; an input declared
    as flush empties the stream on purpose, so the count starts again from
    what is left. Raises InjectionError when the top module is missing."""
    streams = _normalized(streams)
    if streams is None:
        raise ValueError("streams are not in the schema's shape")
    src = strip_assertions(child)
    names = extend.identifiers(src)
    p, n = PREFIX, 2
    while any(x.startswith(p) for x in names):
        p, n = f"{PREFIX}{n}", n + 1
    masked = _mask_comments(child)
    edge = ("negedge"
            if re.search(rf"\bnegedge\s+{re.escape(clock)}\b", masked)
            and not re.search(rf"\bposedge\s+{re.escape(clock)}\b", masked)
            else "posedge")
    tick = f"always @({edge} {clock})"
    rst = reset or "1'b0"
    text = top_text(child, top)
    t_in, _ = ports(child, top)
    t_in.discard(clock)
    params = _parameters(text)
    lines, tags = [HEADER], []
    if reset:
        lines += [f"reg {p}_started = 1'b0;",
                  f"{tick} {p}_started <= 1'b1;",
                  f"always @(*) assume (({rst}) == !{p}_started);"]
    for i, s in enumerate(streams):
        takes = []
        for j, e in enumerate(s["inputs"]):
            ready = _expr(e["ready"]) or "1'b1"
            lines.append(f"wire {p}_take_{i}_{j} = (|({_expr(e['offer'])})) "
                         f"&& (|({ready}));")
            takes.append(f"{{15'd0, {p}_take_{i}_{j}}}")
        took = " + ".join(takes)
        flush = _expr(s["flush"]) or "1'b0"
        give = _self_sized_nots(_expr(s["delivered"]))
        held = _self_sized_nots(_expr(s["held"]))
        lines += [
            f"wire [15:0] {p}_give_{i} = {give};",
            f"wire [15:0] {p}_held_{i} = {held};",
            f"wire {p}_flush_{i} = |({flush});",
            f"reg [15:0] {p}_in_{i} = 16'd0;",
            f"reg [15:0] {p}_out_{i} = 16'd0;",
            # 1 at the start, so a design with no reset counts from what it
            # holds in the first cycle (26 Oct 2026); a reset clears first
            f"reg {p}_rebase_{i} = 1'b1;",
            f"{tick} begin",
            f"    {p}_rebase_{i} <= {p}_flush_{i};",
            f"    if ({rst}) begin {p}_in_{i} <= 16'd0; "
            f"{p}_out_{i} <= 16'd0; end",
            f"    else if ({p}_rebase_{i}) begin "
            f"{p}_in_{i} <= {p}_out_{i} + {p}_held_{i} + {took}; "
            f"{p}_out_{i} <= {p}_out_{i} + {p}_give_{i}; end",
            f"    else begin {p}_in_{i} <= {p}_in_{i} + {took}; "
            f"{p}_out_{i} <= {p}_out_{i} + {p}_give_{i}; end",
            "end",
            f"always @(*) if (!({rst}) && !{p}_rebase_{i}) begin"]
        tags.append((len(lines), i, "lost", None))
        lines.append(f"    assert ({p}_in_{i} <= {p}_out_{i} + {p}_held_{i});"
                     f"   // stream {i}: nothing lost")
        tags.append((len(lines), i, "count_off", None))
        lines.append(f"    assert ({p}_in_{i} >= {p}_out_{i} + {p}_held_{i});"
                     f"   // stream {i}: counts add up")
        lines.append("end")
        for j, e in enumerate(s["inputs"]):
            if not e["ready"] or _outside(e["offer"], text, t_in, params):
                continue    # the outside may take back an item never taken
            offer, ready = _expr(e["offer"]), _expr(e["ready"])
            lines += [
                f"reg {p}_wait_{i}_{j} = 1'b0;",
                f"{tick} {p}_wait_{i}_{j} <= !({rst}) && (|({offer})) "
                f"&& !(|({ready}));",
                f"always @(*) if (!({rst}) && {p}_wait_{i}_{j} && "
                f"!{p}_rebase_{i} && !{p}_flush_{i})"]
            tags.append((len(lines), i, "gave_up", j))
            lines.append(f"    assert (|({offer}));   // stream {i}, input "
                         f"{j}: kept while told to wait")
    at = _insert_point(src, top)
    first = src[:at].count("\n") + 1
    line_map = {first + k: (i, kind, j) for k, i, kind, j in tags}
    return (src[:at] + "\n".join(lines) + "\n" + src[at:], line_map,
            (first, first + len(lines) - 1))


# --- reading the search ------------------------------------------------------

_ABC_FAIL = re.compile(r"Output (\d+) of miter .* was asserted in frame (\d+)")
_STEP = re.compile(r"failed assertion .*? step (\d+)")
_CHECKED = re.compile(r"Checking assertions in step (\d+)")
_ABC_FRAME = re.compile(r"engine_\d+: +(\d+) \+ :")
_BUILD_ERROR = re.compile(
    r"([\w.\-]+\.sv):(\d+)(?:\.\d+(?:-\d+\.\d+)?)?: ERROR: (.*)")
_YWA_LINE = re.compile(r"_(\d+)_\d+$")


def _result(result, **fields):
    """A search result with every key of RESULT_KEYS."""
    out = dict.fromkeys(RESULT_KEYS)
    out.update(fields, result=result)
    return out


def _abc_lines(workdir, output):
    """The source line of abc's failing output `output`, read from the
    AIGER map sby writes (model/design_aiger.ywa): its name ends
    `..._sv_<line>_<column>`."""
    try:
        ywa = json.loads((Path(workdir) / "model" / "design_aiger.ywa")
                         .read_text())
        m = _YWA_LINE.search(str(ywa["asserts"][output][-1]))
    except (OSError, ValueError, LookupError, TypeError):
        return []
    return [int(m.group(1))] if m else []


def read_result(outcome, line_map, span):
    """The search's result from sby's outcome (RESULT_KEYS; depth, reset
    and workdir are the caller's). An engine's FAIL decides, even when
    sby's last line says it ran out of time: the failing line says what
    was found (lost before gave_up before count_off; none of them is
    "unattributed"). Without one, rc 0 is "nothing_lost", a time-out
    "unfinished" (with the last cycle an engine reached), an error at a
    line of the checker's block, or naming its signals, "build_error", and
    anything else "error"."""
    log = outcome.log_text or ""
    seconds = round(outcome.duration_s or 0.0, 1)
    if "Status returned by engine: FAIL" in log:
        lines = sorted(parse_assert_failures(log))
        step = _STEP.search(log)
        step = int(step.group(1)) if step else None
        abc = _ABC_FAIL.search(log)
        if abc and not lines:
            lines = _abc_lines(outcome.workdir, int(abc.group(1)))
        if abc and step is None:
            step = int(abc.group(2))
        hits = [line_map[n] for n in lines if n in line_map]
        for kind in ("lost", "gave_up", "count_off"):
            hit = next((h for h in hits if h[1] == kind), None)
            if hit:
                return _result(kind, seconds=seconds, step=step,
                               stream=hit[0], input=hit[2])
        return _result("unattributed", seconds=seconds, step=step)
    if outcome.rc == 0:
        return _result("nothing_lost", seconds=seconds)
    if outcome.rc in (8, None):
        reached = [int(x) for x in _CHECKED.findall(log)
                   + _ABC_FRAME.findall(log)]
        return _result("unfinished", seconds=seconds,
                       checked_to=max(reached) if reached else None)
    m = _BUILD_ERROR.search(log)
    if m and (span[0] <= int(m.group(2)) <= span[1]
              or PREFIX in m.group(3)):
        return _result("build_error", seconds=seconds,
                       why=m.group(3).strip())
    return _result("error", seconds=seconds,
                   why="\n".join(log.splitlines()[-6:]))


# --- what the step is told ---------------------------------------------------

def _offers(stream):
    """`OFFER` and `READY` (or `OFFER` when ready is "") for each input."""
    return _short(", or ".join(
        f"`{_short(e['offer'])}` and `{_short(e['ready'])}`" if e["ready"]
        else f"`{_short(e['offer'])}`" for e in stream["inputs"]), 240)


def _found(result, streams):
    """The stream the search blames (None when it blames none)."""
    i = result.get("stream")
    return streams[i] if isinstance(i, int) and 0 <= i < len(streams) \
        else None


def _at_cycle(result):
    k = result.get("step")
    return f" at cycle {k}" if k is not None else ""


def _within(result):
    k = result.get("step")
    return k if k is not None else result.get("depth")


def lesson(result, streams):
    """What the step is told when the search sends it back (it goes in
    record["error"], which batch.retry_lesson reads), or None for a result
    that sends nothing back."""
    streams = _normalized(streams) or []
    kind = result.get("result")
    s = _found(result, streams)
    # 27 Oct 2026, Rishab approved: the enable goes in only when the design
    # waits for it (r0_062 takes items whatever its enable)
    if kind == "lost" and s:
        cycle = _at_cycle(result)
        return ("the checker's own nothing-is-lost search (written by the "
                f"checker, not you) found stream '{s['name']}' losing an "
                f"item{cycle}{' after the start' if cycle else ''}: more "
                "items were handed over at its inputs (counted in a cycle "
                f"when {_offers(s)}) than were delivered "
                f"({_short(s['delivered'])}) plus still held "
                f"({_short(s['held'])}). An item is dropped, overwritten or "
                "ignored when a part is full or busy, or while it waits. "
                "Never drop work, counted or not: make the sender wait on "
                "the stream's own path, through a ready or busy output it "
                "reads, or by holding only the registers that carry the "
                "stream, never the module's enable for everything. If the "
                "declaration is what is wrong, fix it instead: offer is the "
                "sender's own valid (with the design's enable only when the "
                "design takes items only while it is high), ready is what "
                "the sender reads, held counts every item inside, and an "
                "input that empties the stream on purpose goes in flush.")
    if kind == "gave_up" and s:
        j = result.get("input")
        e = s["inputs"][j] if isinstance(j, int) and \
            0 <= j < len(s["inputs"]) else s["inputs"][0]
        return ("the checker's own nothing-is-lost search found that in "
                f"stream '{s['name']}' the sender ({_short(e['offer'])}) gave "
                f"up an item{_at_cycle(result)} while the stream told it to "
                f"wait (its ready {_short(e['ready'])} was false the cycle "
                "before), so the item is lost. A sender that cannot hold its "
                'item cannot be made to wait: write its ready as "" and make '
                "the part take every item it offers, or make the sender hold "
                "its item until ready is true.")
    if kind == "count_off" and s:
        return ("the checker's own nothing-is-lost search found the counts "
                f"of stream '{s['name']}' adding up the other "
                f"way{_at_cycle(result)}: fewer items were handed over (offer "
                "and ready both true) than were delivered plus held. Offer "
                "and ready must be true together exactly once per item the "
                "design takes, delivered must count each item once as it "
                "leaves, and held must count only items handed over and not "
                "yet delivered, starting from 0.")
    if kind in ("lost", "gave_up", "count_off", "unattributed"):
        names = _short(", ".join(f"'{x['name']}'" for x in streams), 240)
        return ("the checker's own nothing-is-lost search found the counts "
                f"of the declared stream(s) {names} not adding up within "
                f"{_within(result)} cycles (it could not tell which way in "
                "time): either an item handed over is lost (dropped, "
                "overwritten or ignored when a part is full or busy: make "
                "the sender wait instead), or the declaration does not "
                "match the design (offer once per item handed over, "
                "delivered once per item that leaves, held every item "
                "inside).")
    if kind == "build_error":
        why = _short(result.get("why") or "", 300).rstrip(".")
        return ("the checker could not build its count from the declared "
                f"streams: {why}. Write each value as a plain Verilog "
                "expression over the top module's signals.")
    return None


def reason(result, streams):
    """The one-line reason ("tried and rejected" notes read it), or None
    for a result that sends nothing back."""
    streams = _normalized(streams) or []
    kind = result.get("result")
    s = _found(result, streams)
    if kind == "lost" and s:
        return ("the checker's nothing-is-lost search found stream "
                f"'{s['name']}' losing an item{_at_cycle(result)}")
    if kind == "gave_up" and s:
        return ("the checker's nothing-is-lost search found the sender of "
                f"stream '{s['name']}' giving up an item it was told to "
                f"hold{',' if result.get('step') is not None else ''}"
                f"{_at_cycle(result)}")
    if kind == "count_off" and s:
        k = result.get("step")
        return (f"stream '{s['name']}' is declared so that its counts add "
                "up the other way" + (f" (cycle {k})" if k is not None
                                      else ""))
    if kind in ("lost", "gave_up", "count_off", "unattributed"):
        return ("the checker's nothing-is-lost search found the counts of a "
                f"declared stream not adding up within {_within(result)} "
                "cycles")
    if kind == "build_error":
        return ("the checker could not build its count from the declared "
                "streams")
    return None


# --- the search --------------------------------------------------------------

def check(record, child, top, clock, parents, wrapper=None):
    """The checker's own nothing-is-lost search on record["streams"], once.
    Keeps the result as record["nothing_lost"] (RESULT_KEYS); marks the
    record ITEMS_LOST or STREAM_INVALID, with the lesson as its error and
    a one-line reason, and returns True when the step is sent back. A
    search that cannot run or runs out of time changes nothing (a
    non-verdict says nothing about the design). Streams that are empty, or
    a record already searched, return False at once. `parents` are the
    parent rows ([parent] for a patch, [a, b] for a combination) and
    `wrapper` a combination's own module text."""
    streams = _normalized(record.get("streams"))
    if not streams or "nothing_lost" in record \
            or not all(s["inputs"] for s in streams):
        return False
    if not sby_available():
        record["nothing_lost"] = _result("not_run", seconds=0.0,
                                         why="sby is not on PATH")
        return False
    depth = search_depth(child, top, parents, wrapper)
    reset = reset_condition(child, top, clock)
    if clock not in ports(child, top)[0]:
        record["nothing_lost"] = _result(
            "error", seconds=0.0, depth=depth, reset=reset,
            why=f"the clock {clock} is not an input of module {top}")
        return False
    root = grade_kwargs(record, leg="nothing_lost").get("workdir_root") \
        or NOTHING_LOST_ROOT
    try:
        text, line_map, span = checked_copy(child, top, clock, streams, reset)
        with tempfile.TemporaryDirectory() as td:
            sv = Path(td) / f"{top}_nothing_lost.sv"
            sv.write_text(text)
            with Spinner("checking that nothing offered is lost"):
                outcome = run_sby("nothing_lost", sv, top, "bmc", depth,
                                  NOTHING_LOST_TIMEOUT_S, root,
                                  engine=NOTHING_LOST_ENGINES,
                                  options=NOTHING_LOST_OPTIONS)
    except (InjectionError, OSError) as exc:
        record["nothing_lost"] = _result("error", seconds=0.0, depth=depth,
                                         reset=reset, why=str(exc))
        return False
    result = read_result(outcome, line_map, span)
    result.update(depth=depth, reset=reset, workdir=str(outcome.workdir))
    record["nothing_lost"] = result
    verdict = VERDICT_OF.get(result["result"])
    if verdict is None:
        return False
    record["verdict"] = verdict
    record["error"] = lesson(result, streams)
    record["reason"] = reason(result, streams)
    return True


# --- what the reviewer is told -----------------------------------------------

def _squash(text):
    return "".join(text.split())


def _assumes_every_item_taken(stream):
    """A stream that counts each item delivered in the cycle it is offered
    and holds none: it assumes the receiver takes every item."""
    if not _ZERO.fullmatch(_squash(stream["held"])):
        return False
    given = _squash(stream["delivered"])
    return any(given == _squash(e["offer"])
               or (e["ready"] and given == _squash(
                   f"{e['offer']} && {e['ready']}"))
               for e in stream["inputs"])


def reviewer_notes(record):
    """One plain line for each thing the checker's own tools found that the
    reviewer should know (sense_check.py shows them as facts, never as the
    builder's words): the search's result for each declared stream (no
    line for a result that sent the step back: such a step never reaches
    the reviewer), a combination's stream that assumes the receiver takes
    every item (change 7), the input ports a step adds, and a combination
    whose part between the designs has no registers."""
    notes = []
    streams = _normalized(record.get("streams")) or []
    found = record.get("nothing_lost") or {}
    result, depth = found.get("result"), found.get("depth")
    for s in streams:
        declared = (f"handed over when {_offers(s)}; delivered "
                    f"{_short(s['delivered'])}; held {_short(s['held'])}"
                    + (f"; emptied by {_short(s['flush'])}" if s["flush"]
                       else ""))
        head = (f"the checker's own nothing-is-lost search, stream "
                f"'{s['name']}' ({declared})")
        if result == "nothing_lost":
            notes.append(f"{head}: no item lost in any input sequence of "
                         f"{depth} cycles from the start")
        elif result == "unfinished":
            reached = found.get("checked_to")
            notes.append(f"{head}: the search ran out of time "
                         + (f"at cycle {reached} of {depth}"
                            if reached is not None
                            else f"before cycle {depth}")
                         + ", so nothing is known about it")
        elif result in ("not_run", "error"):
            notes.append(f"{head}: the search could not run")
    compose = record.get("ext_type") == "compose"
    if compose:
        notes += [f"stream '{s['name']}' counts each item delivered in the "
                  "cycle it is offered and holds none, so it assumes the "
                  "receiving design takes every item in every state"
                  for s in streams if _assumes_every_item_taken(s)]
    if record.get("new_inputs"):
        notes.append("the step adds the input ports "
                     + ", ".join(record["new_inputs"]))
    if compose and record.get("glue_identifiers") == []:
        notes.append("the part between the two designs has no registers")
    return notes
