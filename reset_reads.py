"""Does a design read its reset where two checkers disagree? (23 Oct 2026)

In a cycle where an asynchronous reset is held, our checker (yosys, which
sby runs with async2sync) shows every register at its reset value; EBMC
6.0, which scores the trained model, shows the value the register held and
resets it at the next clock edge. An assertion, its guard, an antecedent
or a fact that reads the reset can then be true for one checker and false
for the other: the reviewer's probe of 22 Oct 2026, property
`(rst_n || a == 0) && b <= 4` with facts `a == b` and `a <= 4`, is
NECESSARY for ours and REFUTED by EBMC. Designs whose assertions and facts
leave the reset alone got one verdict from both.

This began as the generator's check in initiator/run.py (22 Oct 2026).
The review of 23 Oct 2026 found the extender had none: a step that added
`assert ((rst_n || a == 3'd0) && d <= 4'd8)` to a design with an
asynchronous reset was NECESSARY for ours and REFUTED by EBMC. So both
sides now ask this module, the way they ask fact_format.py:
  initiator/run.py       reset_gate: a new design
  extender/extend.py     step_reads_reset: what a step adds
  extender/distractor.py (step_reset_error)
  extender/fix.py        a repaired fact list, the same way
All of them count asynchronous resets only (the generator counted every
reset until 23 Oct 2026; see usual_names below).

What counts as a reset (reset_names):
- an asynchronous reset: a signal with an edge in a sensitivity list that
  is not the clock. In a block with two edges whose first statement is an
  if, the edge that if tests is the reset and the other one is the clock;
  a signal alone in a sensitivity list is a clock.
- anything that carries an asynchronous reset in the same cycle (23 Oct
  2026, review finding: `wire in_reset = !rst_n` passed the generator's
  check, and that design was NECESSARY for ours and REFUTED by EBMC): a
  wire or assign whose right side reads one, a signal set in a
  combinational block that reads one, a port connected to one, a `define
  that names one, and copies of those copies. Registers are not followed.
- with usual_names, also a signal with a usual reset name (rst, rst_n,
  ...), even with no edge: a synchronous reset. Its copies are not
  followed. The generator asked for this until 23 Oct 2026, when Rishab
  chose asynchronous resets only for it too: both checkers read a
  synchronous reset the same way, and 30 of the 917 stored designs guard
  their property with one (`if (pv && !rst)`; 11 of the 665 of
  September, 19 of the 252 of 9 Oct). Nothing asks for it now.

Known limits: a port connected by position, not by name, is not followed,
and a signal set in a function is not either.
"""
import re
from collections import Counter

from oracle.inject import _mask_comments

# The usual names of a reset input; any signal other than the clock with
# an edge in a sensitivity list (`negedge clr_n`) is one too.
RESET_NAMES = frozenset({"rst", "rst_n", "reset", "reset_n", "arst_n"})

# a sized number (4'd9, 'h1F) is one token, so its letters are never names
_TOKEN = re.compile(r"\$?[A-Za-z_][\w$]*"
                    r"|\d*'[sS]?[bodhBODH]?[0-9a-fA-FxXzZ_?]+|\d[\w.]*|::|\S")
_WORD = re.compile(r"[A-Za-z_][\w$]*$")
_NAME = re.compile(r"(?<![\w$'])[A-Za-z_][\w$]*")
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
_BLOCKS = ("always", "always_ff", "always_comb", "always_latch", "initial",
           "final")
# `define NAME body, the body running on over lines that end in a backslash
_DEFINE = re.compile(r"`define[ \t]+([A-Za-z_]\w*)(?:\([^)\n]*\))?"
                     r"((?:[^\n\\]|\\\n|\\)*)")
# words that are never a signal, so never a reset or a copy of one
_KEYWORDS = frozenset({
    "always", "always_comb", "always_ff", "always_latch", "and", "assert",
    "assign", "assume", "automatic", "begin", "bit", "case", "casex",
    "casez", "cover", "default", "disable", "else", "end", "endcase",
    "endfunction", "endgenerate", "endmodule", "final", "for", "forever",
    "function", "generate", "genvar", "if", "iff", "initial", "inout",
    "input", "int", "integer", "localparam", "logic", "module", "negedge",
    "not", "or", "output", "parameter", "posedge", "priority", "property",
    "reg", "repeat", "signed", "unique", "unique0", "unsigned", "while",
    "wire"})


class _Unread(Exception):
    """A block the statement reader below cannot follow."""


def _tokens(verilog):
    """(spans, tokens) of the design with comments and strings blanked:
    spans are (token, start, end) in the original text."""
    masked = _STRING.sub(lambda m: " " * len(m.group()),
                         _mask_comments(verilog))
    spans = [(m.group(), m.start(), m.end())
             for m in _TOKEN.finditer(masked)]
    return spans, [s[0] for s in spans]


def _group_end(toks, i):
    """toks[i] opens a bracket: the index just after the one closing it."""
    depth = 0
    for k in range(i, len(toks)):
        if toks[k] in ("(", "[", "{"):
            depth += 1
        elif toks[k] in (")", "]", "}"):
            depth -= 1
            if depth == 0:
                return k + 1
    raise _Unread


def _next_at_depth_zero(toks, i, stop):
    depth = 0
    for k in range(i, len(toks)):
        if toks[k] in ("(", "[", "{"):
            depth += 1
        elif toks[k] in (")", "]", "}"):
            depth -= 1
        elif toks[k] == stop and depth == 0:
            return k
    raise _Unread


def _statement(toks, i, guard, found):
    """Read the statement starting at toks[i] and return the index after
    it. Each assert met goes into `found` with the token spans of what it
    sits under: enclosing if conditions, case selectors and case labels."""
    if i >= len(toks):
        raise _Unread
    t = toks[i]
    if t == "begin":
        i += 1
        if i < len(toks) and toks[i] == ":":          # begin : name
            i += 2
        while i < len(toks) and toks[i] != "end":
            i = _statement(toks, i, guard, found)
        if i >= len(toks):
            raise _Unread
        i += 1
        if i < len(toks) and toks[i] == ":":          # end : name
            i += 2
        return i
    if t in ("unique", "unique0", "priority"):
        return _statement(toks, i + 1, guard, found)
    if t == "if":
        if i + 1 >= len(toks) or toks[i + 1] != "(":
            raise _Unread
        j = _group_end(toks, i + 1)
        inner = guard + [("condition", i + 1, j)]
        i = _statement(toks, j, inner, found)
        if i < len(toks) and toks[i] == "else":
            i = _statement(toks, i + 1, inner, found)
        return i
    if t in ("case", "casez", "casex"):
        if i + 1 >= len(toks) or toks[i + 1] != "(":
            raise _Unread
        j = _group_end(toks, i + 1)
        selector = ("condition", i + 1, j)
        i = j
        while i < len(toks) and toks[i] != "endcase":
            k = _next_at_depth_zero(toks, i, ":")
            i = _statement(toks, k + 1,
                           guard + [selector, ("condition", i, k)], found)
        if i >= len(toks):
            raise _Unread
        return i + 1
    if t in ("for", "while", "repeat"):
        if i + 1 >= len(toks) or toks[i + 1] != "(":
            raise _Unread
        return _statement(toks, _group_end(toks, i + 1), guard, found)
    if t == "forever":
        return _statement(toks, i + 1, guard, found)
    if t == "@":
        if i + 1 < len(toks) and toks[i + 1] == "(":
            return _statement(toks, _group_end(toks, i + 1), guard, found)
        return _statement(toks, i + 2, guard, found)
    if (_WORD.match(t) and t not in _KEYWORDS and i + 2 < len(toks)
            and toks[i + 1] == ":"):
        # a label: `p0: assert (...)` (23 Oct 2026, review finding: such
        # an assertion was skipped whole, so not even a direct read of
        # the reset in it was seen)
        return _statement(toks, i + 2, guard, found)
    end = _next_at_depth_zero(toks, i, ";")
    if t == "assert":
        found.append((list(guard), (i, end + 1)))
    return end + 1


def assertions(verilog):
    """Every assertion of the design and what it sits under, as
    [((text, names), [(kind, text, names), ...])]: kind is "condition" (an
    enclosing if condition, case selector or case label), "block" (the
    event list of the always block it is in) or "unread" (all of a block
    the reader could not follow, up to the assertion: it can only find
    more, never less). Comments and strings are skipped."""
    spans, toks = _tokens(verilog)
    found, i, unread_from = [], 0, None
    while i < len(toks):
        if toks[i] in _BLOCKS:
            j, guard = i + 1, []
            try:
                if j < len(toks) and toks[j] == "@":
                    if j + 1 < len(toks) and toks[j + 1] == "(":
                        k = _group_end(toks, j + 1)
                        guard, j = [("block", j + 1, k)], k
                    else:
                        j += 2                          # @* or @ *
                inside = []
                end = _statement(toks, j, guard, inside)
            except _Unread:
                unread_from, i = i, i + 1
                continue
            found += inside
            unread_from, i = None, end
            continue
        if toks[i] == "assert":                         # module level
            try:
                end = _next_at_depth_zero(toks, i, ";")
            except _Unread:
                end = len(toks) - 1
            guard = [] if unread_from is None else [("unread", unread_from,
                                                     i)]
            found.append((guard, (i, end + 1)))
            i = end + 1
            continue
        i += 1

    def piece(a, b):
        text = " ".join(verilog[spans[a][1]:spans[b - 1][2]].split())
        return text, {w for w in toks[a:b] if _WORD.match(w)}
    return [(piece(a, b), [(kind,) + piece(g0, g1)
                           for kind, g0, g1 in guard if g1 > g0])
            for guard, (a, b) in found]


# --- what counts as a reset ------------------------------------------------

def _names(toks):
    """The signal names among `toks`."""
    return {t for t in toks if _WORD.match(t) and t not in _KEYWORDS}


def _event_lists(toks):
    """(edge names, the edge names the block's first if tests) for every
    block with an event list, in order."""
    out = []
    for i, t in enumerate(toks):
        if t not in _BLOCKS or toks[i + 1:i + 3] != ["@", "("]:
            continue
        try:
            k = _group_end(toks, i + 2)
        except _Unread:
            continue
        edges = [toks[j + 1] for j in range(i + 3, k - 2)
                 if toks[j] in ("posedge", "negedge")
                 and _WORD.match(toks[j + 1])]
        j = k
        if j < len(toks) and toks[j] == "begin":
            j += 1
            if j < len(toks) and toks[j] == ":":         # begin : name
                j += 2
        tested = set()
        if toks[j:j + 2] == ["if", "("]:
            try:
                tested = set(toks[j + 2:_group_end(toks, j + 1) - 1]) & \
                    set(edges)
            except _Unread:
                pass
        out.append((edges, tested))
    return out


def _clocks_and_resets(toks, clock):
    """(clocks, asynchronous resets) from the sensitivity lists."""
    lists = _event_lists(toks)
    clocks = {clock}
    for edges, tested in lists:
        if len(set(edges)) == 1:
            clocks |= set(edges)
        elif len(set(edges)) == 2 and tested:
            clocks |= set(edges) - tested   # the edge the if does not test
    resets = set()
    for edges, tested in lists:
        if len(set(edges)) > 1:
            # with three edges or more, the ones the first if does not
            # test may be a second reset as well as the clock
            resets |= tested | (set(edges) - clocks)
    return clocks, resets - {clock}


def _is_assignment(toks, k):
    """Is toks[k] an `=` that assigns (not part of ==, <=, >=, != or
    ===)?"""
    return (toks[k] == "=" and (k == 0 or toks[k - 1] not in
                                ("=", "<", ">", "!"))
            and (k + 1 >= len(toks) or toks[k + 1] != "="))


def _target_names(toks):
    """The signals an assignment's left side sets: its names outside
    index brackets and delays (`x[i]` sets x, `{a, b}` sets a and b)."""
    out, depth = set(), 0
    for t in toks:
        if t in ("[", "("):
            depth += 1
        elif t in ("]", ")"):
            depth -= 1
        elif depth == 0 and _WORD.match(t) and t not in _KEYWORDS:
            out.add(t)
    return out


def _split_assignments(toks):
    """[(names set, names read)] for a list of `a = x, b = y` items."""
    out, start, depth = [], 0, 0
    for k, t in enumerate(toks + [","]):
        if t in ("(", "[", "{"):
            depth += 1
        elif t in (")", "]", "}"):
            depth -= 1
        elif t == "," and depth == 0:
            item = toks[start:k]
            start = k + 1
            at = next((n for n in range(len(item))
                       if _is_assignment(item, n)), None)
            if at is not None:
                out.append((_target_names(item[:at]),
                            _names(item[at + 1:])))
    return out


def _declaration_end(toks, i):
    """The index of the `;` that ends the declaration at toks[i], or of
    the bracket that closes the port list it is in."""
    depth = 0
    for k in range(i + 1, len(toks)):
        if toks[k] in ("(", "[", "{"):
            depth += 1
        elif toks[k] in (")", "]", "}"):
            depth -= 1
            if depth < 0:
                return k
        elif toks[k] == ";" and depth == 0:
            return k
    return len(toks)


def _lvalue(toks, k):
    """The names set by the assignment whose `=` is toks[k], read back
    from it over index brackets and braces."""
    j = k - 1
    while j >= 0 and toks[j] == "]":                 # x[i][j] = ...
        depth = 0
        while j >= 0:
            if toks[j] == "]":
                depth += 1
            elif toks[j] == "[":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        j -= 1
    if j >= 0 and toks[j] == "}":                    # {a, b} = ...
        depth, end = 0, j
        while j >= 0:
            if toks[j] == "}":
                depth += 1
            elif toks[j] == "{":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        return _target_names(toks[j + 1:end])
    if j >= 0 and _WORD.match(toks[j]) and toks[j] not in _KEYWORDS:
        return {toks[j]}
    return set()


def _combinational_end(toks, j):
    """Where the combinational block whose statement starts at toks[j]
    ends: after that statement, or, when it cannot be read, at the next
    block, assign or endmodule (so it errs towards counting more)."""
    try:
        return _statement(toks, j, [], [])
    except _Unread:
        for k in range(j, len(toks)):
            if toks[k] in _BLOCKS + ("assign", "endmodule"):
                return k
        return len(toks)


def _carriers(verilog, toks):
    """[(names set, names they take their value from)] for every way a
    signal carries another in the same cycle: wires declared with a value,
    assigns, combinational blocks (every signal the block sets, from
    everything it reads), ports connected by name (both ways: a port and
    what it is connected to are one signal) and `define."""
    out = []
    for i, t in enumerate(toks):
        if t == "assign":
            try:
                end = _next_at_depth_zero(toks, i, ";")
            except _Unread:
                continue
            out += _split_assignments(toks[i + 1:end])
        elif t == "wire":
            out += _split_assignments(toks[i + 1:_declaration_end(toks, i)])
        elif (t == "." and toks[i + 1:i + 2] and _WORD.match(toks[i + 1])
              and toks[i + 2:i + 3] == ["("]):
            try:
                end = _group_end(toks, i + 2)
            except _Unread:
                continue
            port, inner = {toks[i + 1]}, _names(toks[i + 3:end - 1])
            out += [(port, inner), (inner, port)]
        elif t in ("always", "always_comb", "always_latch"):
            j = i + 1
            combinational = t != "always"
            if toks[j:j + 1] == ["@"]:
                if toks[j + 1:j + 2] == ["("]:
                    try:
                        k = _group_end(toks, j + 1)
                    except _Unread:
                        continue
                    combinational = not any(
                        x in ("posedge", "negedge") for x in toks[j + 2:k])
                    j = k
                else:
                    combinational, j = True, j + 2      # @* or @ *
            if not combinational:
                continue
            body = toks[j:_combinational_end(toks, j)]
            sets = set()
            for k in range(len(body)):
                if _is_assignment(body, k):
                    sets |= _lvalue(body, k)
            if sets:
                out.append((sets, _names(body)))
    for m in _DEFINE.finditer(_mask_comments(verilog)):
        out.append(({m.group(1)}, _names(_TOKEN.findall(m.group(2)))))
    return out


def reset_names(verilog, clock="clk", usual_names=False):
    """{name: the reset inputs it carries} for every signal the design
    uses as a reset (see the top of this file): its asynchronous resets,
    everything that carries one in the same cycle, and, with
    `usual_names`, the usual names of a reset input."""
    _, toks = _tokens(verilog if isinstance(verilog, str) else "")
    clocks, resets = _clocks_and_resets(toks, clock)
    found = {r: {r} for r in resets}
    carriers = _carriers(verilog if isinstance(verilog, str) else "", toks)
    grown = True
    while grown:
        grown = False
        for sets, reads in carriers:
            hit = reads & found.keys()
            if not hit:
                continue
            carried = set().union(*(found[r] for r in hit))
            for name in sets - clocks:
                if not carried <= found.setdefault(name, set()):
                    found[name] |= carried
                    grown = True
    if usual_names:
        for name in RESET_NAMES - {clock}:
            found.setdefault(name, {name})
    return {name: frozenset(base) for name, base in found.items()}


# --- where a design reads one ----------------------------------------------

def _flat(text):
    return " ".join(str(text).split())


def _place(text, under):
    """An assertion's place: its text and the guards it sits under."""
    return text, tuple((kind, gtext) for kind, gtext, _ in under)


def reset_reads(verilog, clock="clk", facts=(), antecedents=(),
                usual_names=False, names=None):
    """Where the design reads a reset (reset_names, or `names` when
    given): [(kind, text, where, reset names read)], kind being
    "assertion" (the assertion itself), "condition", "block" or "unread"
    (what it sits under), "antecedent" or "fact". For the first four,
    `where` is (the assertion's place, how many assertions with that
    place come before it); None for the others. Strings that are not
    text are skipped."""
    verilog = verilog if isinstance(verilog, str) else ""
    names = reset_names(verilog, clock, usual_names) if names is None \
        else names
    resets = set(names)
    out, seen = [], Counter()
    for (text, words), under in assertions(verilog):
        place = _place(text, under)
        where = (place, seen[place])
        seen[place] += 1
        if words & resets:
            out.append(("assertion", text, where, frozenset(words & resets)))
        for kind, gtext, gwords in under:
            if gwords & resets:
                out.append((kind, gtext, where, frozenset(gwords & resets)))
    for kind, exprs in (("antecedent", antecedents), ("fact", facts)):
        for expr in exprs if isinstance(exprs, (list, tuple)) else ():
            if not isinstance(expr, str):
                continue
            hit = set(_NAME.findall(expr)) & resets
            if hit:
                out.append((kind, _flat(expr), None, frozenset(hit)))
    return out


def _short(text, n=100):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 3] + "..."


def describe(reads, names, what="property"):
    """(the reset inputs read, sorted; each place as a phrase, in order,
    without repeats). `what` is what an assertion is called: "property"
    for a new design, "assertion" for a step. A place that reads a copy
    says which signal carries the reset."""
    inputs, places = set(), []
    for kind, text, _, read in reads:
        base = set().union(*(names.get(n, {n}) for n in read))
        inputs |= base
        place = {
            "assertion": f"the {what} `{_short(text)}`",
            "condition": f"the {what}'s guard `{_short(text)}`",
            "block": f"the {what} sits in a block the reset triggers, "
                     f"`@{_short(text)}`",
            "unread": f"the {what}'s block, which names the reset",
            "antecedent": f"antecedent `{_short(text)}`",
            "fact": f"fact `{_short(text)}`",
        }[kind]
        through = sorted(read - base)
        if through:
            place += (f" (through {', '.join(through)}, which "
                      + ("carries" if len(through) == 1 else "carry")
                      + " the reset)")
        places.append(place)
    return sorted(inputs), list(dict.fromkeys(places))


# --- what a step of the extender adds (23 Oct 2026, review finding) --------

STEP_RULE = (
    "a new assertion, its guard and the facts describe the design in normal "
    "operation and never read an asynchronous reset, directly or through a "
    "signal that carries it: in a cycle where such a reset is held, the "
    "checkers disagree about what the registers hold, so a step that reads "
    "one can be proven by our checker and refuted by the one that scores "
    "the facts. Leave the reset out of them: put a new assertion in a "
    "block of its own clocked by the clock alone, and guard an assertion "
    "that must skip the cycles of a reset with a register the reset "
    "clears.")


def step_reset_error(child, clock="clk", facts=(), antecedents=(),
                     parents=()):
    """Why a grown design reads an asynchronous reset where the designs
    it grew from did not, or None. `parents` are (verilog, clock, facts,
    antecedents) of each of them. An assertion they already have, with
    the same text under the same guards, is not judged again: a step must
    keep it as it is. A fact or antecedent is not judged when one of them
    has it and read the same resets there. Only asynchronous resets count,
    so the older designs whose property's guard reads a synchronous reset
    (`if (pv && !rst)`) are not touched. The fixer, which changes only the
    facts, passes the design itself as its parent."""
    if not isinstance(child, str):
        return None
    names = reset_names(child, clock)
    if not names:
        return None
    old_places, old_exprs = Counter(), Counter()
    for verilog, pclock, pfacts, pants in parents:
        verilog = verilog if isinstance(verilog, str) else ""
        for (text, _), under in assertions(verilog):
            old_places[_place(text, under)] += 1
        old_exprs.update(r for r in reset_reads(verilog, pclock, pfacts,
                                                pants) if r[2] is None)
    new = []
    for read in reset_reads(child, clock, facts, antecedents, names=names):
        if read[2] is not None:                 # an assertion or its guard
            place, before = read[2]
            if before < old_places[place]:
                continue
        elif old_exprs[read]:                   # a fact or antecedent
            old_exprs[read] -= 1
            continue
        new.append(read)
    if not new:
        return None
    read, places = describe(new, names, what="assertion")
    return (f"the asynchronous reset {', '.join(read)} is read by "
            + "; ".join(places) + " - " + STEP_RULE)
