"""Extender step 4: structural and second-property extensions — the two
types where the invariant list is allowed to change.

New machinery beyond the distractor:
  dispositions   every parent invariant clause must be declared kept or
                 superseded; a silent deletion rejects the extension
                 BEFORE grading.
  P2-unaided     a second property that proves by k-induction without any
                 invariants teaches nothing; checked by stripping the
                 first property and grading the second alone.
  coupling       a structural extension must produce at least one clause
                 mentioning both new and existing state; its absence is
                 reported (the human reads it), not auto-rejected.

Port list is FIXED for these types (unlike the distractor): new state
must be observable through existing outputs.

  venv/bin/python extender/extend.py --plan                      # the 5-seed plan
  venv/bin/python extender/extend.py --parent g0_014 --type structural --move GUARD --dry
  venv/bin/python extender/extend.py --parent g0_014 --type structural --move GUARD
  venv/bin/python extender/extend.py --parent g0_014 --type second
"""
import argparse
from collections import Counter
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import (_mask_comments, extract_asserts,   # noqa: E402
                          state_bits)
from distractor import (DIFF_FORMAT, GRADE_KWARGS, MODEL, EFFORT,   # noqa: E402
                        Spinner, dump, load_parent)
from patch import PatchError, apply_patch                 # noqa: E402

import llm_client                                               # noqa: E402
from oracle import PropertyInfo, grade, grade_triple_generated  # noqa: E402

MAX_TOKENS = 32000

SYSTEM_PROMPT = """\
You are extending a formally verified SystemVerilog module.

The module has a property proven by k-induction with the strengthening
invariants listed in the user message. Your job is to grow the design so
that it remains provable, but is harder.

The result is training data for teaching a model to discover invariants.
So the logic you add should read as though it exists for its own functional
reason. A reader should not be able to guess the invariant by noticing which
part of the module looks like scaffolding. Write the extension the way a
hardware engineer would write the next feature, not the way a proof author
would write a hint.

HARD RULES

1. The existing property text survives verbatim. Do not modify, move, or
   delete the existing assertion. Do not add assume or cover statements.
2. Emit a PATCH in the diff format given below. Never a rewritten module.
3. The port list is fixed. Do not add, remove, or rename ports. New state
   must be observable through an existing output, directly or through
   existing logic.
4. Every new register gets a reset in the existing if (rst) branch and an
   initial value, matching the style already in the module.
5. Same Verilog subset as the module: synthesizable always @(posedge clk)
   or always @(*) logic, immediate assertions only, no concurrent SVA, no
   system tasks.
6. Do not use the -> implication operator. Yosys rejects it. Write !a || b.
7. Keep datapath widths narrow, four to eight bits. Wide arithmetic over
   free inputs makes the proof intractable and teaches nothing.
8. If a new register's declared width admits values outside its reachable
   range, add an invariant clause bounding it."""

MOVES = {
    "SPLIT": ("Take a quantity that is currently held in one register and "
              "divide it, so part of it lives in a new register some of "
              "the time."),
    "STAGE": ("Insert a pipeline register between a producer and a "
              "consumer. An update that currently completes in one cycle "
              "now takes two, with the in-flight value held in new state."),
    "GUARD": ("Add an enable register that gates an existing update. The "
              "existing transition fires only when the new enable is set."),
    "PEER": ("Add a second instance of similar logic that shares a "
             "resource with the existing logic."),
    "BOUND": ("Replace a constant limit with a register that can change, "
              "so the existing comparison is against state rather than a "
              "literal."),
}

# Batch-runner sampling policy, from measured state-bit returns per call:
# STAGE +9, SPLIT +9, PEER +7, BOUND +4, GUARD +1. GUARD is dead weight;
# second-property rarely grows the invariant list (kept for the shared-
# clause coupling case only).
MOVE_WEIGHTS = {"STAGE": 3, "SPLIT": 3, "PEER": 2, "BOUND": 1, "GUARD": 0}
SECOND_PROPERTY_RATE = 1 / 6

STAGE_K_TEXT = """
        Magnitude K={k}: insert a pipeline of {k} registers, written out as
        {k} explicit stages (no generate loops — each stage spelled out, so
        the lines are real). The update that completed in one cycle now
        takes {k}+1, with {k} in-flight values each held in its own new
        register, and the invariant list must pin every stage."""

INVARIANT_RULES = """\
- Emit the COMPLETE new invariant list, not just what changed.
  Inductiveness is a property of the whole conjunction, so satisfy
  yourself that the full list closes the proof.
- The list must be sufficient: the property must be provable by
  k-induction from the design plus these invariants.
- Keep clauses as strong as the parent's. Generalising a clause to cover
  new state is correct. Weakening it until it happens to still hold is not.
- Add no clause you cannot justify. A clause that could be deleted without
  breaking the proof is noise in the corpus.
- Never copy a property into the invariant list. An invariant clause
  syntactically identical to any assertion is rejected outright."""

DISPOSITION_BLOCK = """\
For every invariant clause in the parent list, record what happened to it:

  kept        the clause appears verbatim in your new list
  superseded  the clause is no longer true, or is subsumed by a stronger
              clause. Name the replacing clause and give a one-line reason.
              If several new clauses jointly replace it, list them all in
              replaced_by — do not paste them into one string.

A parent clause that is neither kept nor declared superseded is a silent
deletion, and the extension is rejected before it is graded. Deleting a
clause is allowed. Deleting it quietly is not.

Example: the parent has `fault == (timer == 4'd0)`. You add a register that
latches an external error signal, so the fault output now has a second
source and the parent clause is false whenever that latch is set. The
correct disposition is superseded by `fault == (timer == 4'd0 || err_latch)`,
reason "the fault condition now has a second source"."""

STRUCTURAL_TEMPLATE = """\
Extension type: STRUCTURAL.

Add new state that INTERACTS with the state the property depends on, so the
parent's invariants are no longer sufficient and the invariant list must
grow to describe the coupling.

The move you must make:
{move}

The module:

{verilog}

The property, which you must not touch:
{property}

The parent's strengthening invariants:
{invariants}

REQUIREMENTS SPECIFIC TO THIS TYPE

The new state must write, or gate the write of, something the property or
the parent invariants read. If your new logic only reads existing state and
never affects it, that is a distractor, not a structural extension.

Your new invariant list must contain at least one clause mentioning both a
new identifier and an existing one. That coupling clause is the point of
the extension. A clause about only new state is a fact bolted on the side.

WORK IT OUT BEFORE YOU WRITE THE PATCH

Answer these in order. The patch comes last, after you know what breaks.

1. new_state: which register or registers you are adding, and which existing
   signal each one writes or gates.

2. coupling: in one sentence, the relation that must hold between the new
   state and the existing state for the design to be correct.

3. induction_gap: a concrete state of the extended design that an induction
   engine will treat as a legal starting point, that the design can never
   actually reach, and from which the property fails within a cycle. The
   design must be able to sit in this state indefinitely with its enables
   held low, so that a counterexample starting there survives the full
   twenty-step induction window.

4. why_parent_insufficient: which parent invariant clauses fail to exclude
   that state, and why.

5. Then write the patch, then the full new invariant list.

{invariant_rules}

{disposition_block}

Diff format:
{diff_format}

Reply with JSON:
{{"new_state": "...", "coupling": "...", "induction_gap": "...",
  "why_parent_insufficient": "...", "patch": "...",
  "invariants": ["..."], "dispositions": [...]}}"""

SECOND_TEMPLATE = """\
Extension type: SECOND PROPERTY.

Add a SECOND assertion about this module, proven from the SAME invariant
list as the first.

The module:

{verilog}

The existing property, which you must not touch:
{property}

The parent's strengthening invariants:
{invariants}

REQUIREMENTS SPECIFIC TO THIS TYPE

The second property must be true of the design and must NOT be provable by
k-induction on its own without any invariants. If it proves unaided it
teaches nothing and the extension is rejected.

The second property must depend on state the first property already
depends on, so that at least one invariant clause is required by both. Two
properties needing disjoint clauses is two problems in one file, not an
extension.

The second property must ALSO read at least one register the first
property never mentions, so the invariant list has to say something new.
A second property whose signals are a subset of the first's is rejected:
it rides on the existing invariants and teaches nothing.

You may add new state if the second property needs it, but you do not have
to. If you do, the rules above about resets, initial values, and observing
through existing outputs still apply.

Add the assertion as its own block, in the same style as the existing one:

    always @(posedge clk)
        <optional guard> assert (<expression>);

WORK IT OUT BEFORE YOU WRITE THE PATCH

1. claim: what the second property asserts, in one sentence.

2. shared_state: which registers both properties depend on.

2b. new_state: which register the second property reads that the first
    does not (required — add state if the design has no candidate).

3. not_inductive_alone: a concrete state that satisfies the second property,
   violates the invariant list, and steps to a violation of the second
   property. The design must be able to hold this state indefinitely with
   enables low.

4. shared_clause: which invariant clause or clauses are needed by both
   properties, and why each is needed by each.

5. Then write the patch and the full invariant list.

{invariant_rules}

{disposition_block}

Diff format:
{diff_format}

Reply with JSON:
{{"claim": "...", "shared_state": "...", "new_state": "...",
  "not_inductive_alone": "...", "shared_clause": "...", "patch": "...",
  "invariants": ["..."], "dispositions": [...]}}"""

_DISPOSITION_ITEMS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "clause": {"type": "string"},
            "status": {"type": "string", "enum": ["kept", "superseded"]},
            "replaced_by": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["clause", "status", "replaced_by", "reason"],
        "additionalProperties": False,
    },
}


def _schema(reasoning_fields):
    props = {f: {"type": "string"} for f in reasoning_fields}
    props["patch"] = {"type": "string"}
    props["invariants"] = {"type": "array", "items": {"type": "string"}}
    props["dispositions"] = _DISPOSITION_ITEMS
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


COMPOSE_TEMPLATE = """\
Extension type: COMPOSE.

Wire the two verified modules below together: module A's behaviour feeds
module B, THROUGH STATEFUL GLUE. You write ONLY the glue wrapper; both
parent modules are appended verbatim by the harness — do not repeat or
modify them.

Module A:

{verilog_a}

A's property: {property_a}
A's invariants: {invariants_a}

Module B:

{verilog_b}

B's property: {property_b}
B's invariants: {invariants_b}

REQUIREMENTS

1. The wrapper instantiates A once and B once, and connects a value A
   produces to an input B consumes — NEVER through a plain wire. Put a
   register stage, a valid/data handshake pair, or a small buffer between
   them, and the glue must HOLD its contents while its enable is low. A
   plain wire makes B's proof trivial and the extension worthless; glue
   that refills itself every cycle unconditionally is a plain wire with
   extra steps.
2. Expose any per-parent signal your invariants mention as a named
   top-level wire. NEVER write inst.signal — hierarchical references are
   rejected (the toolchain silently turns them into dangling wires).
3. The wrapper contains exactly ONE new assertion about the composed
   behaviour, guarded by an antecedent where natural.
4. The invariant list must contain: every parent clause renamed to the
   wrapper's wires (one copy each), PLUS at least one clause about the
   glue state itself. The glue clause is the point — it is the state
   neither parent's invariants describe.
5. Same clock and reset names as the parents. Every new register gets a
   reset and an initial value. Widths stay narrow.
6. No assume, no cover, no -> implication, no system tasks.

WORK IT OUT BEFORE YOU WRITE THE WRAPPER

1. glue_scheme: what the glue holds and when it moves, two sentences.
2. induction_gap: a state with garbage in the glue that satisfies your
   property, can be held indefinitely with enables low, and steps to a
   property violation.
3. why_parents_insufficient: why neither parent's invariants exclude it.
4. Then the wrapper module, then the complete invariant list. The wrapper
   field is PLAIN Verilog module source — not a diff, no +/- line
   prefixes, no ---/+++/@@ headers.

FIELD NOTES for the JSON reply — antecedents and sanity_covers are
VERILOG BOOLEAN EXPRESSIONS over top-level signals, never descriptions:
- antecedents: the guard of your new assertion. If it is written
  `if (pool == 4'd0) assert (...)` the antecedent is "pool == 4'd0".
  If unconditional, use [].
- sanity_covers: one or two expressions you believe genuinely reachable
  within ~20 cycles, e.g. "pool == 4'd6". Prose like "pool reaches zero
  after all credits are held" is rejected before grading.

Reply with JSON: {{"glue_scheme": "...", "induction_gap": "...",
 "why_parents_insufficient": "...", "top_module": "...", "wrapper": "...",
 "antecedents": [...], "sanity_covers": [...], "invariants": ["..."]}}"""


def _compose_schema():
    props = {f: {"type": "string"} for f in
             ("glue_scheme", "induction_gap", "why_parents_insufficient",
              "top_module", "wrapper")}
    for f in ("antecedents", "sanity_covers", "invariants"):
        props[f] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


COMPOSE_SCHEMA = _compose_schema()

REPLICATE_TEMPLATE = """\
Extension type: REPLICATE.

Instantiate the verified module below N={n} times behind a SHARED CREDIT
POOL, and prove one property of the pool.

You write ONLY the wrapper module. The parent module is appended verbatim
by the harness — do not repeat it, do not modify it, do not rename it.

The parent module:

{verilog}

Its property (it lives inside every instance and must still prove):
{property}

Its strengthening invariants (true of each instance):
{invariants}

REQUIREMENTS

1. The wrapper instantiates the parent exactly {n} times and owns a credit
   pool: a shared budget register of TOTAL credits (TOTAL at most 8). An
   instance can only gain a unit of its resource when the pool has a free
   credit; releasing a unit returns the credit. The wrapper may rely only
   on signals the parent exposes through its ports.
2. Expose every per-instance signal your invariants mention as a named
   top-level wire with an instance-numbered name (t0..t{last}, h0..h{last}).
   NEVER write inst.signal — hierarchical references are rejected: the
   toolchain silently turns them into dangling wires.
3. The wrapper contains exactly ONE new assertion: a property of the pool,
   immediate assertion, guarded by an antecedent where natural.
4. The invariant list must contain, for EVERY parent clause, one renamed
   copy per instance ({n} copies each), PLUS at least one clause that is a
   genuine sum or reduction across all {n} instances (for example
   pool + t0 + ... + t{last} == TOTAL). Per-instance and pairwise clauses
   do not count as that aggregate.
5. Same clock and reset names as the parent. Every new register gets a
   reset in the wrapper's if (rst) branch and an initial value. Widths
   stay narrow. Watch summation width: zero-extend operands so the sum
   cannot wrap.
6. No assume, no cover, no -> implication, no system tasks.

WORK IT OUT BEFORE YOU WRITE THE WRAPPER

1. pool_scheme: how credits move between pool and instances, two sentences.
2. induction_gap: a state of pool + instances that satisfies your property,
   violates the aggregate clause, can be held indefinitely with all enables
   low, and steps to a property violation.
3. why_aggregate_needed: why no collection of per-instance clauses excludes
   that state.
4. Then the wrapper module, then the complete invariant list. The wrapper
   field is PLAIN Verilog module source — not a diff, no +/- line
   prefixes, no ---/+++/@@ headers.

FIELD NOTES for the JSON reply — antecedents and sanity_covers are
VERILOG BOOLEAN EXPRESSIONS over top-level signals, never descriptions:
- antecedents: the guard of your new assertion. If it is written
  `if (pool == 4'd0) assert (...)` the antecedent is "pool == 4'd0".
  If unconditional, use [].
- sanity_covers: one or two expressions you believe genuinely reachable
  within ~20 cycles, e.g. "pool == 4'd6". Prose like "pool reaches zero
  after all credits are held" is rejected before grading.

Reply with JSON: {{"pool_scheme": "...", "induction_gap": "...",
 "why_aggregate_needed": "...", "top_module": "...", "wrapper": "...",
 "antecedents": [...], "sanity_covers": [...], "invariants": ["..."]}}"""


def _replicate_schema():
    props = {f: {"type": "string"} for f in
             ("pool_scheme", "induction_gap", "why_aggregate_needed",
              "top_module", "wrapper")}
    for f in ("antecedents", "sanity_covers", "invariants"):
        props[f] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


REPLICATE_SCHEMA = _replicate_schema()

STRUCTURAL_SCHEMA = _schema(
    ["new_state", "coupling", "induction_gap", "why_parent_insufficient"])
SECOND_SCHEMA = _schema(
    ["claim", "shared_state", "new_state", "not_inductive_alone",
     "shared_clause"])

# One structural move per seed, five diverse seeds. Distractor and second
# property run on the same five, so the by-hand reading covers 15 results.
PLAN = [
    ("g0_014", "token_bucket", "GUARD"),
    ("g0_035", "pwm_dimmer", "STAGE"),
    ("g0_036", "watchdog", "BOUND"),
    ("g0_043", "stack_hwm", "SPLIT"),
    ("g0_046", "rr_arbiter_pri", "PEER"),
]


# --- pure checks (unit-tested) ---

def _norm(s):
    return " ".join(s.split())


def check_dispositions(parent_invs, new_invs, dispositions):
    """Every parent clause must be declared kept (and present verbatim) or
    superseded (absent, with a named present replacement and a reason).
    Returns an error string, or None if the audit passes."""
    new_normed = {_norm(x) for x in new_invs}
    parent_normed = {_norm(x): x for x in parent_invs}
    seen = set()
    for disp in dispositions:
        clause = _norm(disp.get("clause", ""))
        if clause not in parent_normed:
            return f"disposition names a non-parent clause: {disp.get('clause')!r}"
        if clause in seen:
            return f"duplicate disposition for clause: {disp.get('clause')!r}"
        seen.add(clause)
        status = disp.get("status")
        if status == "kept":
            if clause not in new_normed:
                return (f"clause declared kept but absent from the new "
                        f"list: {disp.get('clause')!r}")
        elif status == "superseded":
            if clause in new_normed:
                return (f"clause declared superseded but still present: "
                        f"{disp.get('clause')!r}")
            replacements = disp.get("replaced_by") or []
            if isinstance(replacements, str):   # tolerate the singular form
                replacements = [replacements]
            if not replacements:
                return (f"superseded with no replacing clause named: "
                        f"{disp.get('clause')!r}")
            missing = [r for r in replacements if _norm(r) not in new_normed]
            if missing:
                return (f"superseding clause not in the new list for: "
                        f"{disp.get('clause')!r} (missing: {missing})")
            if not disp.get("reason", "").strip():
                return f"superseded without a reason: {disp.get('clause')!r}"
        else:
            return f"unknown disposition status: {status!r}"
    missing = [orig for n, orig in parent_normed.items() if n not in seen]
    if missing:
        return ("silent deletion — no disposition for parent clause(s): "
                + "; ".join(repr(m) for m in missing))
    return None


def split_second_property_asserts(parent_props, child_asserts):
    """Child must carry the parent's asserts verbatim plus EXACTLY one new
    one. Returns (new_assert, None) or (None, error)."""
    p = [_norm(x) for x in parent_props]
    c = [_norm(x) for x in child_asserts]
    if len(c) != len(p) + 1:
        return None, (f"expected exactly one new assertion: parent has "
                      f"{len(p)}, child has {len(c)}")
    if c[:len(p)] != p:
        return None, (f"parent assertion(s) not preserved verbatim: "
                      f"{p} vs {c[:len(p)]}")
    return child_asserts[-1], None


def remove_asserts(source, exprs):
    """Blank the assert(...) statements whose expression matches one of
    `exprs` (whitespace-normalized), leaving legal Verilog — used to
    isolate the second property for the proves-unaided check."""
    targets = {_norm(e) for e in exprs}
    out = source
    for m in reversed(list(re.finditer(r"\bassert\b", source))):
        i = m.end()
        while i < len(source) and source[i].isspace():
            i += 1
        if i >= len(source) or source[i] != "(":
            continue
        depth, j = 0, i
        while j < len(source):
            if source[j] == "(":
                depth += 1
            elif source[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if _norm(source[i + 1:j]) in targets:
            out = out[:m.start()] + out[j + 1:]
    return out


_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VERILOG_NOISE = {"posedge", "negedge", "always", "assert", "module",
                  "endmodule", "begin", "end", "if", "else", "wire", "reg",
                  "input", "output", "initial", "assign",
                  "localparam", "parameter", "case", "casez", "casex",
                  "endcase", "default", "function", "endfunction",
                  "integer", "genvar", "generate", "endgenerate",
                  "signed", "unsigned", "logic", "assume", "cover"}


def identifiers(text):
    text = _mask_comments(text)   # comment words are not identifiers
    ids = set(_ID.findall(re.sub(r"\d+'[bodhBODH][0-9a-fA-FxXzZ_?]+", " ",
                                 text)))
    return ids - _VERILOG_NOISE


def p2_new_ids(parent_props, new_assert, child_verilog):
    """REGISTERS the second property reads that no parent property does.
    Only declared regs count — a new localparam or wire name is a new
    label, not new state. Empty means P2 rides entirely on the first
    property's support — the same-invariants, no-new-clause pattern —
    and is rejected."""
    new = identifiers(new_assert) - identifiers("\n".join(parent_props))
    return new & wrapper_regs(child_verilog)


def property_copy(invariants, asserts):
    """An invariant clause syntactically identical to a property (after
    whitespace normalisation) is the cheapest cheat on the second-property
    type: sound, but it teaches 'copy the property into the invariant
    list'. Returns the offending clause, or None. All whitespace is
    stripped before comparing, so spacing differences cannot hide a copy."""
    squash = lambda x: "".join(x.split())  # noqa: E731
    assert_set = {squash(a) for a in asserts}
    for clause in invariants:
        if squash(clause) in assert_set:
            return clause
    return None


# --- REPLICATE checks ---

_BIT_SELECT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]")
_IDX_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*?)_?(\d+)$")


def _instance_indices(expr):
    """Map base-name -> set of instance indices, read from suffixed
    identifiers (t0, credits_3) and constant bit-selects (gnt[2])."""
    masked = re.sub(r"\d+'[bodhBODH][0-9a-fA-FxXzZ_?]+", " ", expr)
    out = {}
    for m in _BIT_SELECT.finditer(masked):
        out.setdefault(m.group(1), set()).add(int(m.group(2)))
    for name in _ID.findall(masked):
        if name in _VERILOG_NOISE:
            continue
        m = _IDX_NAME.fullmatch(name)
        if m:
            out.setdefault(m.group(1), set()).add(int(m.group(2)))
    return out


def aggregate_clause(invariants, n_required):
    """The clause that makes a REPLICATE worth having: one clause that is
    a genuine sum/reduction across >= n_required instances. Per-instance
    copies never qualify (each touches one index); pairwise clauses never
    qualify (two). Returns the clause, or None."""
    for clause in invariants:
        for indices in _instance_indices(clause).values():
            if len(indices) >= n_required:
                return clause
    return None


def instance_coverage_gaps(parent_invs, new_invs, n_instances):
    """Each parent clause family must appear >= n_instances times in the
    new list (renamed per instance — template() already masks names, so
    family membership is template equality). Returns the parent clauses
    whose family is under-represented."""
    from build_corpus import template as _template
    counts = Counter(_template(x) for x in new_invs)
    return [p for p in parent_invs
            if counts[_template(p)] < n_instances]


_REG_DECL = re.compile(
    r"\breg\b(?:\s*\[[^\]]*\])?\s*"
    r"([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)")


def wrapper_regs(wrapper):
    """Names of registers DECLARED in the wrapper — the only things that
    count as glue state. Wires are not state: a renamed port wire passing
    a parent signal through is exactly the plain-wire trap."""
    names = set()
    for m in _REG_DECL.finditer(_mask_comments(wrapper)):
        for name in m.group(1).split(","):
            names.add(name.strip())
    return names - _VERILOG_NOISE


_PORT_HEADER = re.compile(r"module\s+(\w+)\s*\((.*?)\)\s*;", re.S)


def ports_of(verilog, top_module):
    """Names declared in the module's port header."""
    for m in _PORT_HEADER.finditer(_mask_comments(verilog)):
        if m.group(1) == top_module:
            return identifiers(m.group(2))
    return set()


def compose_hidden_signals(row):
    """Invariant identifiers a wrapper could never reach: mentioned in the
    parent's invariants but not exposed through its ports. Non-empty means
    the parent is ineligible for COMPOSE (and REPLICATE)."""
    ports = ports_of(row["verilog"], row["top_module"])
    used = set()
    for clause in row["invariants"]:
        used |= identifiers(clause)
    return used - ports


_DIFF_MARKER = re.compile(r"^(---\s|\+\+\+\s|@@)", re.M)


def strip_diff_decoration(text):
    """Recover a plain module from an answer wrapped in unified-diff
    syntax (---/+++/@@ headers, + line prefixes). Applied only when diff
    markers are present, so plain Verilog passes through untouched —
    same philosophy as fence-stripping: a formatting tic must not waste
    the call."""
    if not _DIFF_MARKER.search(text):
        return text
    out = []
    for line in text.splitlines():
        if re.match(r"^(---\s|\+\+\+\s|@@)", line) or line == "@@":
            continue
        if line.startswith("+"):
            out.append(line[1:])
        elif line.startswith("-"):
            continue
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def hierarchical_refs(out):
    """Dotted references in any expression field — named early with the
    correct fix, instead of surfacing later as a baffling coverage miss
    or a contract ERROR after grading started."""
    from oracle.contract import _HIER_REF
    bad = []
    for field in ("invariants", "antecedents", "sanity_covers"):
        for x in out.get(field, []):
            if _HIER_REF.search(x):
                bad.append(x)
    return bad


def glue_clause(invariants, glue_ids):
    """COMPOSE's point: at least one clause about the glue registers —
    the state neither parent's invariants describe. Returns it, or None."""
    for clause in invariants:
        if identifiers(clause) & glue_ids:
            return clause
    return None


def has_coupling_clause(new_invs, parent_ids, new_ids):
    for clause in new_invs:
        ids = identifiers(clause)
        if ids & new_ids and ids & parent_ids:
            return True
    return False


# --- prompt / call / grade ---

def build_prompt(parent, ext_type, move=None, n=6, k=1):
    if ext_type == "replicate":
        return REPLICATE_TEMPLATE.format(
            n=n, last=n - 1, verilog=parent["verilog"],
            property="\n".join(parent["property"]),
            invariants="\n".join(parent["invariants"]))
    tmpl = STRUCTURAL_TEMPLATE if ext_type == "structural" else SECOND_TEMPLATE
    fields = dict(
        verilog=parent["verilog"],
        property="\n".join(parent["property"]),
        invariants="\n".join(parent["invariants"]),
        invariant_rules=INVARIANT_RULES,
        disposition_block=DISPOSITION_BLOCK,
        diff_format=DIFF_FORMAT)
    if ext_type == "structural":
        fields["move"] = f"{move}   {MOVES[move]}"
        if move == "STAGE" and k > 1:
            fields["move"] += STAGE_K_TEXT.format(k=k)
    return tmpl.format(**fields)


def call_model(prompt_text, schema):
    text, usage, stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        user=prompt_text, schema=schema, effort=EFFORT)
    return (None if text is None else json.loads(text)), usage, stop


def grade_compose(pa, pb, out, record):
    """assemble (A verbatim + B verbatim + glue wrapper) -> property
    discipline -> coverage + glue-clause gates -> yosys -> oracle.
    A DECORATIVE verdict here means the glue was not stateful enough."""
    wrapper = strip_diff_decoration(out["wrapper"])
    top = out["top_module"]
    record["wrapper"] = wrapper
    record["invariants"] = out["invariants"]
    record["antecedents"] = out.get("antecedents", [])
    record["sanity_covers"] = out.get("sanity_covers", [])
    if top in (pa["top_module"], pb["top_module"]) or not re.search(
            r"\bmodule\s+" + re.escape(top) + r"\b",
            _mask_comments(wrapper)):
        record["verdict"] = "WRAPPER_ERROR"
        record["error"] = f"wrapper must define a new top module, got {top!r}"
        return record
    bad = hierarchical_refs(out)
    if bad:
        record["verdict"] = "HIERARCHICAL_REF"
        record["error"] = (
            f"expression fields contain hierarchical references {bad} - "
            "the toolchain silently turns inst.signal into a dangling "
            "wire. State a wrapper invariant can mention must be exposed "
            "through the instance's PORTS onto a named wrapper wire; if "
            "the parent keeps that state internal, the parent is not "
            "eligible for this extension type.")
        return record
    child = (pa["verilog"].rstrip() + "\n\n" + pb["verilog"].rstrip()
             + "\n\n" + wrapper)
    record["child_verilog"] = child

    parent_props = pa["property"] + pb["property"]
    new_assert, err = split_second_property_asserts(
        parent_props, extract_asserts(child))
    if err:
        record["verdict"] = "PROPERTY_CHANGED"
        record["error"] = err
        return record
    record["compose_property"] = new_assert

    copied = property_copy(out["invariants"], extract_asserts(child))
    if copied:
        record["verdict"] = "PROPERTY_COPY"
        record["error"] = f"invariant copies a property verbatim: {copied!r}"
        return record

    gaps = instance_coverage_gaps(pa["invariants"] + pb["invariants"],
                                  out["invariants"], 1)
    if gaps:
        record["verdict"] = "COMPOSE_COVERAGE"
        record["error"] = f"parent clause family missing from the list: {gaps}"
        return record

    glue_ids = wrapper_regs(wrapper)
    record["glue_identifiers"] = sorted(glue_ids)
    if not glue_ids:
        record["verdict"] = "NO_GLUE_CLAUSE"
        record["error"] = ("wrapper declares no registers - plain-wire "
                           "composition, the DECORATIVE trap")
        return record
    glue = glue_clause(out["invariants"], glue_ids)
    if glue is None:
        record["verdict"] = "NO_GLUE_CLAUSE"
        record["error"] = ("no invariant clause mentions the glue "
                           f"register(s) {sorted(glue_ids)} - glue state "
                           "left undescribed")
        return record
    record["glue_clause"] = glue

    pa_bits = state_bits(pa["verilog"], pa["top_module"])
    pb_bits = state_bits(pb["verilog"], pb["top_module"])
    child_bits = state_bits(child, top)
    record["parent_state_bits"] = [pa_bits, pb_bits]
    record["child_state_bits"] = child_bits
    if child_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = "child does not read in yosys"
        return record

    payload = {"verilog": child, "top_module": top, "clock": pa["clock"],
               "antecedents": out["antecedents"],
               "sanity_covers": out["sanity_covers"],
               "invariants": out["invariants"]}
    t0 = time.monotonic()
    with Spinner(f"oracle grading composed {top}"):
        result = grade_triple_generated(json.dumps(payload), **GRADE_KWARGS)
    record["grade_wall_s"] = round(time.monotonic() - t0, 2)
    record["verdict"] = result.verdict.name
    record["reason"] = result.reason
    record["result"] = asdict(result)
    if result.verdict.name == "DECORATIVE":
        record["error"] = ("glue not stateful enough: the composition "
                           "proves without any invariants (wire-equivalent)")
    return record


def grade_replicate(parent, out, record, n):
    """assemble (parent verbatim + wrapper) -> property discipline ->
    coverage + aggregate checks -> yosys -> oracle."""
    wrapper = strip_diff_decoration(out["wrapper"])
    top = out["top_module"]
    record["wrapper"] = wrapper
    record["invariants"] = out["invariants"]
    record["antecedents"] = out.get("antecedents", [])
    record["sanity_covers"] = out.get("sanity_covers", [])
    if top == parent["top_module"] or not re.search(
            r"\bmodule\s+" + re.escape(top) + r"\b",
            _mask_comments(wrapper)):
        record["verdict"] = "WRAPPER_ERROR"
        record["error"] = (f"wrapper must define a new top module named "
                           f"{top!r} distinct from the parent")
        return record
    bad = hierarchical_refs(out)
    if bad:
        record["verdict"] = "HIERARCHICAL_REF"
        record["error"] = (
            f"expression fields contain hierarchical references {bad} - "
            "the toolchain silently turns inst.signal into a dangling "
            "wire. State a wrapper invariant can mention must be exposed "
            "through the instance's PORTS onto a named wrapper wire; if "
            "the parent keeps that state internal, the parent is not "
            "eligible for this extension type.")
        return record
    child = parent["verilog"].rstrip() + "\n\n" + wrapper
    record["child_verilog"] = child

    new_assert, err = split_second_property_asserts(
        parent["property"], extract_asserts(child))
    if err:
        record["verdict"] = "PROPERTY_CHANGED"
        record["error"] = err
        return record
    record["pool_property"] = new_assert

    copied = property_copy(out["invariants"], extract_asserts(child))
    if copied:
        record["verdict"] = "PROPERTY_COPY"
        record["error"] = f"invariant copies a property verbatim: {copied!r}"
        return record

    gaps = instance_coverage_gaps(parent["invariants"], out["invariants"], n)
    if gaps:
        record["verdict"] = "REPLICATE_COVERAGE"
        record["error"] = ("parent clause family under-represented "
                           f"(need {n} renamed copies each): {gaps}")
        return record

    agg = aggregate_clause(out["invariants"], n)
    if agg is None:
        record["verdict"] = "NO_AGGREGATE"
        record["error"] = (f"no clause sums/reduces across all {n} "
                           "instances — per-instance copies alone teach "
                           "nothing new")
        return record
    record["aggregate_clause"] = agg

    parent_bits = state_bits(parent["verilog"], parent["top_module"])
    child_bits = state_bits(child, top)
    record["parent_state_bits"] = parent_bits
    record["child_state_bits"] = child_bits
    if parent_bits is None or child_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = ("parent" if parent_bits is None else "child") + \
            " does not read in yosys"
        return record

    payload = {"verilog": child, "top_module": top,
               "clock": parent["clock"],
               "antecedents": out["antecedents"],
               "sanity_covers": out["sanity_covers"],
               "invariants": out["invariants"]}
    t0 = time.monotonic()
    with Spinner(f"oracle grading {top} (N={n})"):
        result = grade_triple_generated(json.dumps(payload), **GRADE_KWARGS)
    record["grade_wall_s"] = round(time.monotonic() - t0, 2)
    record["verdict"] = result.verdict.name
    record["reason"] = result.reason
    record["result"] = asdict(result)
    return record


def grade_step4(parent, ext_type, out, record):
    """apply -> property discipline -> dispositions -> yosys -> oracle
    (-> P2-unaided for second property). Fills record in place."""
    record["patch"] = out["patch"]
    record["invariants"] = out["invariants"]
    record["dispositions"] = out["dispositions"]

    try:
        child = apply_patch(parent["verilog"], out["patch"])
    except PatchError as exc:
        record["verdict"] = "PATCH_ERROR"
        record["error"] = str(exc)
        return record
    record["child_verilog"] = child

    child_asserts = extract_asserts(child)
    if ext_type == "structural":
        if [_norm(a) for a in child_asserts] != \
                [_norm(a) for a in parent["property"]]:
            record["verdict"] = "PROPERTY_CHANGED"
            record["error"] = (f"structural patches must not touch asserts: "
                               f"parent {parent['property']} vs child "
                               f"{child_asserts}")
            return record
        child_props = parent["property"]
    else:
        new_assert, err = split_second_property_asserts(
            parent["property"], child_asserts)
        if err:
            record["verdict"] = "PROPERTY_CHANGED"
            record["error"] = err
            return record
        record["second_property"] = new_assert
        record["p2_new_ids"] = sorted(p2_new_ids(parent["property"],
                                                 new_assert, child))
        if not record["p2_new_ids"]:
            record["verdict"] = "P2_SAME_SUPPORT"
            record["error"] = ("second property reads no register the "
                               "first property does not already read — it "
                               "rides on the existing invariants and "
                               "forces no new clause; rejected")
            return record
        child_props = child_asserts

    err = check_dispositions(parent["invariants"], out["invariants"],
                             out["dispositions"])
    if err:
        record["verdict"] = "DISPOSITION_ERROR"
        record["error"] = err
        return record

    copied = property_copy(out["invariants"], child_asserts)
    if copied:
        record["verdict"] = "PROPERTY_COPY"
        record["error"] = (f"invariant clause is a verbatim copy of a "
                           f"property: {copied!r} - sound but teaches the "
                           "wrong lesson; rejected")
        return record

    parent_bits = state_bits(parent["verilog"], parent["top_module"])
    child_bits = state_bits(child, parent["top_module"])
    record["parent_state_bits"] = parent_bits
    record["child_state_bits"] = child_bits
    if parent_bits is None or child_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = ("parent" if parent_bits is None else "child") + \
            " does not read in yosys"
        return record
    record["state_grew"] = child_bits > parent_bits

    parent_ids = identifiers(parent["verilog"])
    new_ids = identifiers(child) - parent_ids
    record["new_identifiers"] = sorted(new_ids)
    record["has_coupling_clause"] = has_coupling_clause(
        out["invariants"], parent_ids, new_ids)

    if ext_type == "second":
        # P2 alone, no invariants: PROVEN here means it teaches nothing.
        p2_only = remove_asserts(child, parent["property"])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / f"{parent['top_module']}.sv"
            f.write_text(p2_only)
            with Spinner("checking the second property does not prove unaided"):
                solo = grade(f, PropertyInfo(top_module=parent["top_module"],
                                             clock=parent["clock"]),
                             **GRADE_KWARGS)
        record["p2_unaided_tier"] = solo.tier.name
        if solo.tier.name == "PROVEN":
            record["verdict"] = "P2_PROVES_UNAIDED"
            record["error"] = ("second property proves by k-induction with "
                               "no invariants — teaches nothing")
            return record

    payload = {"verilog": child, "top_module": parent["top_module"],
               "clock": parent["clock"], "antecedents": parent["antecedents"],
               "sanity_covers": parent["sanity_covers"],
               "invariants": out["invariants"]}
    t0 = time.monotonic()
    with Spinner(f"oracle grading extended {parent['top_module']}"):
        result = grade_triple_generated(json.dumps(payload), **GRADE_KWARGS)
    record["grade_wall_s"] = round(time.monotonic() - t0, 2)
    record["verdict"] = result.verdict.name
    record["reason"] = result.reason
    record["result"] = asdict(result)
    return record


def report(record, ext_type):
    if ext_type in ("replicate", "compose"):
        print(f"\nwrapper:\n{record.get('wrapper', '<none>')}\n")
        if "aggregate_clause" in record:
            print("aggregate:", record["aggregate_clause"])
        if "glue_clause" in record:
            print("glue clause:", record["glue_clause"])
        if "child_state_bits" in record:
            print(f"state bits: {record.get('parent_state_bits')} -> "
                  f"{record.get('child_state_bits')}")
    else:
        print(f"\npatch:\n{record.get('patch', '<none>')}\n")
    print("invariants:", json.dumps(record.get("invariants", []), indent=1))
    print("dispositions:", json.dumps(record.get("dispositions", []), indent=1))
    if "state_grew" in record:
        print(f"state: {record['parent_state_bits']} -> "
              f"{record['child_state_bits']} bits "
              f"(new ids: {', '.join(record['new_identifiers']) or 'none'})")
    if ext_type == "structural":
        print(f"coupling clause present: {record.get('has_coupling_clause')}")
    if ext_type == "second":
        print(f"second property: {record.get('second_property')}")
        print(f"P2 unaided tier: {record.get('p2_unaided_tier')}")
    print(f"verdict: {record['verdict']} - "
          f"{record.get('reason', record.get('error'))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent")
    p.add_argument("--type",
                   choices=["structural", "second", "replicate", "compose"])
    p.add_argument("--parent2", help="second parent for compose")
    p.add_argument("--move", choices=sorted(MOVES))
    p.add_argument("--instances", type=int, default=6,
                   help="N for replicate (default 6)")
    p.add_argument("--k", type=int, default=1,
                   help="magnitude for STAGE (default 1)")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--list-compose", action="store_true",
                   help="list corpus rows whose invariants are fully "
                        "port-visible (compose/replicate eligible)")
    args = p.parse_args()

    if args.list_compose:
        from distractor import CORPUS
        for line in CORPUS.read_text().splitlines():
            row = json.loads(line)
            hidden = compose_hidden_signals(row)
            mark = "OK " if not hidden else "no "
            extra = "" if not hidden else f"  hidden: {sorted(hidden)}"
            print(f"{mark} {row['id']} {row['top_module']}{extra}")
        return

    if args.plan:
        print("five seeds, one structural move each; distractor and second "
              "property run on the same five (15 reads total):\n")
        for pid, name, move in PLAN:
            print(f"  venv/bin/python extender/distractor.py --parent {pid}"
                  f"            # distractor ({name})")
            print(f"  venv/bin/python extender/extend.py --parent {pid} "
                  f"--type structural --move {move}")
            print(f"  venv/bin/python extender/extend.py --parent {pid} "
                  f"--type second")
        return

    if not args.parent or not args.type:
        p.error("--parent and --type required (or use --plan)")
    if args.type == "structural" and not args.move:
        p.error("--move required for structural extensions")

    parent = load_parent(args.parent)
    print(f"parent {parent['id']} {parent['top_module']} "
          f"invariants={parent['invariants']}")
    parent2 = None
    if args.type == "compose":
        if not args.parent2:
            p.error("--parent2 required for compose")
        parent2 = load_parent(args.parent2)
        if parent2["top_module"] == parent["top_module"]:
            p.error("compose parents must have distinct module names")
        for row in (parent, parent2):
            hidden = compose_hidden_signals(row)
            if hidden:
                p.error(
                    f"{row['id']} ({row['top_module']}) is not compose-"
                    f"eligible: invariants mention non-port signal(s) "
                    f"{sorted(hidden)} a wrapper cannot reach. Pick another "
                    "parent (see --list-compose).")
        print(f"parent2 {parent2['id']} {parent2['top_module']} "
              f"invariants={parent2['invariants']}")
        prompt = COMPOSE_TEMPLATE.format(
            verilog_a=parent["verilog"], verilog_b=parent2["verilog"],
            property_a="; ".join(parent["property"]),
            property_b="; ".join(parent2["property"]),
            invariants_a="; ".join(parent["invariants"]) or "(none)",
            invariants_b="; ".join(parent2["invariants"]) or "(none)")
    else:
        prompt = build_prompt(parent, args.type, args.move,
                              n=args.instances, k=args.k)
    if args.dry:
        print("\n=== SYSTEM ===\n" + SYSTEM_PROMPT)
        print("\n=== USER ===\n" + prompt)
        return

    schema = {"structural": STRUCTURAL_SCHEMA, "second": SECOND_SCHEMA,
              "replicate": REPLICATE_SCHEMA,
              "compose": COMPOSE_SCHEMA}[args.type]
    record = {"extension_id": datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
              "ext_type": args.type, "move": args.move, "k": args.k,
              "parent_id": parent["id"],
              "parent2_id": parent2["id"] if parent2 else None,
              "model": llm_client.model_label(MODEL), "effort": EFFORT}
    with Spinner(f"{MODEL} writing a {args.type} extension"):
        out, usage, stop = call_model(prompt, schema)
    record["usage"] = usage
    if out is None:
        record["verdict"] = {"refusal": "REFUSED",
                             "length": "TRUNCATED"}.get(stop, "UNPARSEABLE")
        record["raw_text"] = llm_client.LAST_RAW
        dump(record)
        print(f"{record['verdict'].lower()} - logged (raw text kept)")
        return
    for k in ("new_state", "coupling", "induction_gap",
              "why_parent_insufficient", "claim", "shared_state",
              "not_inductive_alone", "shared_clause", "pool_scheme",
              "why_aggregate_needed", "glue_scheme",
              "why_parents_insufficient"):
        if k in out:
            record[k] = out[k]
    if args.type == "compose":
        grade_compose(parent, parent2, out, record)
    elif args.type == "replicate":
        record["n_instances"] = args.instances
        grade_replicate(parent, out, record, args.instances)
    else:
        grade_step4(parent, args.type, out, record)
    dump(record)
    report(record, args.type)


if __name__ == "__main__":
    main()
