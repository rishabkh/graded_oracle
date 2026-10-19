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
                 reported (the human reads it), not auto-rejected - except
                 for a part with an assertion of its own, where it is the
                 only link to the old design and is required (18 Oct 2026).
  sideways       a big move may add a part beside the existing logic with
                 ONE assertion of its own, held to the second-property
                 rules (new state read, needs facts alone) (18 Oct 2026).

Inputs stay fixed; new state may get new OUTPUT ports (18 Oct 2026).

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

from build_corpus import (_mask_comments, assert_spans,      # noqa: E402
                          extract_asserts, honest_lines, state_bits)
from distractor import (DIFF_FORMAT, MODEL, EFFORT,   # noqa: E402
                        Spinner, dump, grade_kwargs, load_parent)
from patch import PatchError, apply_patch                 # noqa: E402

import llm_client                                               # noqa: E402
from oracle import PropertyInfo, grade, grade_triple_generated  # noqa: E402

MAX_TOKENS = 64000       # was 32000: on 30 Sep 2026, 12 answers on deep parents
                         # stopped at exactly 32,000 and were thrown away

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
3. Keep every existing port as it is: do not remove, rename or retype one,
   and add no inputs. You may add OUTPUT ports for new state. New state must
   be observable through an output, directly or through existing logic.
4. Every new register gets a reset in the existing if (rst) branch and an
   initial value, matching the style already in the module.
5. Same Verilog subset as the module: synthesizable always @(posedge clk)
   or always @(*) logic, immediate assertions only, no concurrent SVA, no
   system tasks.
6. Do not use the -> implication operator. Yosys rejects it. Write !a || b.
7. Match the widths already in the module for data: a new register that
   holds the module's data takes the module's data width. Counters, indexes
   and credit counts are sized to the range they hold, as an engineer would
   (six credits need 3 bits, not 64). Do not multiply or divide wide free
   inputs; that makes the proof intractable.
8. If a new register's declared width admits values outside its reachable
   range, add an invariant clause bounding it.
9. Do not add a register that only copies one the module already has: a
   fact saying `new == old` means the new register adds nothing, and the
   extension is rejected.
10. Write repeated slots, lanes or stages as an array with a for loop
   (reg [7:0] len [0:7]), never as the same block written once per slot
   (q0_len, q1_len, ... q7_len). Copied text is not growth: a patch whose
   new lines are mostly copies of each other is rejected."""

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
    "LEDGER": ("Add a PAIR of event counters that log the two sides of an "
               "existing transition: issued/returned, pushed/popped, "
               "acquired/released, on/off. The counters are free-running "
               "and wrap at their width (do NOT guard the increments). "
               "Their difference is not stored in any register - it is an "
               "emergent fact - yet it must exactly track an existing "
               "piece of state (an occupancy, a pointer difference, a "
               "level flag), modulo the counter width. The invariant list "
               "MUST contain that difference relation, e.g. "
               "(n_in - n_out) == occupancy. Do not add a register that "
               "holds the difference; the relation must live only in the "
               "invariant. EVERY new register you add - the counters and "
               "any flag - must be an output port: a future wrapper must "
               "be able to state every invariant from outside the module, "
               "and a hidden register makes the child a dead end."),
    # 17 Oct 2026: an accepted step added a median of 12 lines, so six
    # rounds took a 73-line design to about 200 against real blocks of
    # about 1,900. These two grow a design by half or more in one step.
    # 18 Oct 2026: real blocks are about 5 modules and ours were 1, so the
    # new unit is now a module of its own (Rishab: "do all 6 changes").
    "SUBUNIT": ("Add a complete new functional unit that works with the "
                "existing logic, the way the next feature of a real block "
                "is added: for example a request queue in front of an "
                "existing input, a second channel that shares an existing "
                "resource, a control and status register bank that "
                "configures an existing behaviour, or a retry or backoff "
                "controller around an existing operation. It holds real "
                "state of its own (several registers and the logic that "
                "drives them). Write it as its own module, with its own "
                "ports, and instantiate it once inside the existing "
                "module, the way real blocks are built from sub-modules; "
                "a patch may begin with + lines, before any anchor, to put "
                "the new module at the top of the file. Facts and "
                "assertions can only name signals of the existing (top) "
                "module: bring each register of the new module that a "
                "fact needs out on an output port, connect that port to a "
                "wire in the existing module, and name the wire. Never "
                "write instance.signal. If the design already has "
                "sub-modules, the new one may work with them directly "
                "(see the list below)."),
    # 18 Oct 2026: "on the path the property depends on" piled every
    # storage step onto one path (five buffers in a row in front of a
    # fixed table in the 18 Oct trial, all weak)
    "STORAGE": ("Add storage the design needs: a register file, FIFO or "
                "lookup table (an array with a write side and a read "
                "side), sized with the module's own widths, that holds "
                "values on their way to existing state or serves a new "
                "part with an assertion of its own. The invariant list "
                "must relate the stored entries, or how many of them are "
                "in use, to the state around them."),
}

# the moves that add a whole piece of design; their question says how big
BIG_MOVES = ("SUBUNIT", "STORAGE", "PEER")

SIZE_TEXT = """
        Size: this is a substantial addition. The module now has {n} lines;
        your patch should add roughly {lo} to {hi} new lines ({share}),
        written out as real logic, not padding, comments, copies of
        existing lines or one block repeated per slot. Lines are counted
        with copies counted once."""

# 18 Oct 2026, hand review of two trials: every good step grew its design
# 1.4 to 1.7 times, inside the +50-75% the question asked for, so the size
# of a good design was set by the question. The two moves that stayed good
# (a sub-unit 5 of 5, storage 5 of 7) now ask to double the design
# (Rishab: "do all 6 changes").
DOUBLING_MOVES = ("SUBUNIT", "STORAGE")


def size_target(verilog, move=None):
    """(lines now, fewest, most) new lines: a sub-unit or storage step
    roughly doubles a design, other big steps grow it by half to three
    quarters (Rishab, 17 Oct 2026: "we want to see rapid growth"). Lines
    are counted with copies counted once (honest_lines, 19 Oct 2026), so a
    design swollen by copied slots is not asked for still more."""
    n = honest_lines(verilog)
    if move in DOUBLING_MOVES:
        return n, n, round(n * 1.25)
    return n, round(n * 0.5), round(n * 0.75)


def size_share(move):
    return ("about as much again as it has now, so the design roughly "
            "doubles" if move in DOUBLING_MOVES
            else "half to three quarters of its current size")

# Batch-runner sampling policy, from measured state-bit returns per call:
# STAGE +9, SPLIT +9, PEER +7, BOUND +4, GUARD +1. GUARD is dead weight;
# second-property rarely grows the invariant list (kept for the shared-
# clause coupling case only).
# LEDGER back on (Rishab, 2026-09-06): the bookkeeping family is where
# a single model call struggles but our oracle+fixer loop still wins -
# so these rows are exactly the ones worth making.
# 17 Oct 2026: the big moves (SUBUNIT, STORAGE, PEER) now take most
# draws, so a lineage grows several times its first size in a run.
# 18 Oct 2026, hand review of the 17 Oct trial: the big moves gave 5 real
# designs out of 5, the small ones (STAGE k=1, SPLIT, BOUND, LEDGER's forced
# gating) 4 contrived ones out of 4, so the big moves take 12 draws in 17.
# 18 Oct 2026, both trials read by hand: SPLIT gave 0 good designs in 5,
# LEDGER 0 in 4, BOUND 0 in 1, so they are no longer drawn and their turns
# go to SUBUNIT and STORAGE; LEDGER's weight of 2 (Rishab, 6 Sep) is
# dropped with his "do all 6 changes". STAGE stays, in runs of 4 or 8.
# The moves remain for runs by hand (extend.py --move).
MOVE_WEIGHTS = {"STAGE": 1, "SPLIT": 0, "PEER": 3, "BOUND": 0, "GUARD": 0,
                "LEDGER": 0, "SUBUNIT": 7, "STORAGE": 6}
# 19 Oct 2026: kept 0 times in 8 in the 18 Oct good-growth trial, and a big
# step can now carry an assertion of its own; the type stays for runs by
# hand (extend.py --type second)
SECOND_PROPERTY_RATE = 0

# 19 Oct 2026: it asked for each stage written out, which is copied text
# (rule 10 of SYSTEM_PROMPT); now an array shifted by a loop
STAGE_K_TEXT = """
        Magnitude K={k}: insert a pipeline of {k} registers, held in an
        array of {k} stages that a for loop shifts along. The update that
        completed in one cycle now takes {k}+1, with {k} in-flight values
        each held in its own stage, and the invariant list must pin every
        stage."""

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

# What a structural move's new state must do. LEDGER's counters only watch
# a transition: asked to write or gate something as well, the 17 Oct trial's
# ledger child added a gate that was always true (hand review, 18 Oct 2026).
EFFECT_RULE = """\
The new state must write, or gate the write of, something the property or
the parent invariants read. If your new logic only reads existing state and
never affects it, that is a distractor, not a structural extension."""

LEDGER_EFFECT_RULE = """\
The new counters observe an existing transition.
They do not need to write or gate anything: their difference must equal
existing state, and that relation goes in your invariant list."""

# 18 Oct 2026: with only the rule above, every new part had to change what
# the first property reads, so on a long chain they piled up on one path
# (five buffers in a row in front of a fixed table, all weak). A big move
# may now grow the design sideways instead, the way real blocks gain
# features (Rishab: "do all 6 changes"). grade_step4 holds the new
# assertion to the second-property rules and requires the coupling fact.
SIDEWAYS_EFFECT_RULE = """\
The new state must write, or gate the write of, something the property or
the parent invariants read - OR it may sit beside the existing logic as a
feature of its own, the way real blocks grow: it reads state the existing
facts describe, and it comes with ONE new assertion about the new part.
For that second way:
- put the assertion in the existing (top) module, in its own always block
  in the same style as the existing one;
- it must not prove without facts;
- it must read at least one register no existing assertion reads;
- at least one fact must name both new state and state the existing facts
  describe;
- never change or remove an existing assertion;
- in new_state below, say which existing state the part reads and what its
  assertion says; the induction_gap may break that assertion instead of
  the existing property.
New logic that only reads existing state, with no assertion of its own and
no effect on what the property reads, is a distractor, not a structural
extension."""

# 18 Oct 2026: on a long chain the model could not see what earlier steps
# had added, and added the same kind of part again (a third, fourth and
# fifth buffer in a row). batch.py passes the family's steps in.
# 19 Oct 2026: also what other designs grown from this one added (two
# siblings added the same length guard) and what was tried on this design
# and rejected (the reviewer turned down the same credit pool three times).
HISTORY_TEXT = """\
Earlier steps on this design, oldest first:
{lines}
"""
SIBLINGS_TEXT = """\
Other designs already grown from this one added (do not add the same
feature again):
{lines}
"""
REJECTED_TEXT = """\
Tried on this design before and rejected (do not try these again):
{lines}
"""
NO_REPEAT_TEXT = """\
Give the design something it does not have yet. Do not add a part that
does the job of one of these again: another buffer in a row in front of
the same output, or a second count of something already counted, adds
size and nothing else.

"""


def history_text(history, siblings=None, rejected=None):
    if not (history or siblings or rejected):
        return ""
    out = ""
    if history:
        out += HISTORY_TEXT.format(lines="\n".join(
            f"  {i}. {h}" for i, h in enumerate(history, 1)))
    if siblings:
        out += SIBLINGS_TEXT.format(lines="\n".join(f"  - {s}"
                                                    for s in siblings))
    if rejected:
        out += REJECTED_TEXT.format(lines="\n".join(f"  - {r}"
                                                    for r in rejected))
    return out + NO_REPEAT_TEXT


# 19 Oct 2026: a new sub-module talked only to the top module, so a design
# collected parts that never worked with each other. The question now lists
# the sub-modules already there, so a new one can connect to them.
MODULES_TEXT = """\
Sub-modules already in this design:
{lines}
A new unit may take inputs from these modules' outputs or drive their
inputs (through wires in the top module, by changing their port
connections), so the parts work together as one system rather than each
talking only to the top module.

"""


def module_map(verilog, top):
    """`instance (module): inputs ...; outputs ...` for every sub-module a
    design instantiates."""
    import gates                      # gates imports this module
    out = []
    for mod, inst, _ in gates.instances(verilog, verilog, top):
        dirs = gates.port_directions(verilog, mod)
        ins = ", ".join(p for p, d in dirs.items() if d == "input")
        outs = ", ".join(p for p, d in dirs.items() if d == "output")
        out.append(f"  {inst} ({mod}): inputs {ins}; outputs {outs}")
    return out


def modules_text(verilog, top):
    lines = module_map(verilog, top)
    return MODULES_TEXT.format(lines="\n".join(lines)) if lines else ""


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

{modules}{history}REQUIREMENTS SPECIFIC TO THIS TYPE

{effect_rule}

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

{history}REQUIREMENTS SPECIFIC TO THIS TYPE

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

{plan}REQUIREMENTS

1. The wrapper instantiates A once and B once, and connects a value A
   produces to an input B consumes — NEVER through a plain wire. Connect
   them the way a real design would: what A sends must mean something to B
   (data into a data input, an event into a trigger or enable), never a
   value forced into an input that expects something else. Put a
   register stage, a valid/data handshake pair, or a small buffer between
   them, and the glue must HOLD its contents while its enable is low. A
   plain wire makes B's proof trivial and the extension worthless; glue
   that refills itself every cycle unconditionally is a plain wire with
   extra steps.
2. Expose any per-parent signal your invariants mention as a named
   top-level wire. NEVER write inst.signal — hierarchical references are
   rejected (the toolchain silently turns them into dangling wires).
   Where a parent's facts need an internal register, the parent carries a
   mirror output port for it (for example `assign count_o = count;`):
   connect that port to a wrapper wire and state the fact through it.
3. The wrapper contains exactly ONE new assertion about the composed
   behaviour, guarded by an antecedent where natural.
4. The invariant list must contain: every parent clause renamed to the
   wrapper's wires (one copy each), PLUS at least one clause about the
   glue state itself. The glue clause is the point — it is the state
   neither parent's invariants describe.
5. Same clock and reset names as the parents. Every new register gets a
   reset and an initial value. Glue registers that carry data match the
   parents' widths; counters are sized to their range. Do not add a glue
   register that only copies a register a parent already has.
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
  from reset, e.g. "pool == 4'd6". Prose like "pool reaches zero
  after all credits are held" is rejected before grading.

Reply with JSON: {{"glue_scheme": "...", "induction_gap": "...",
 "why_parents_insufficient": "...", "top_module": "...", "wrapper": "...",
 "antecedents": [...], "sanity_covers": [...], "invariants": ["..."]}}"""


PLAN_TEXT = """\
You picked B to work with A. The combined system: {system}
Planned connection: {connection}
Build the wrapper the way you planned.

"""


def compose_prompt(pa, pb, plan=None):
    """The COMPOSE question for A and B; `plan` is the model's own partner
    choice ({"system", "connection"}), when it chose B (19 Oct 2026)."""
    return COMPOSE_TEMPLATE.format(
        verilog_a=pa["verilog"], verilog_b=pb["verilog"],
        property_a="; ".join(pa["property"]),
        property_b="; ".join(pb["property"]),
        invariants_a="; ".join(pa["invariants"]) or "(none)",
        invariants_b="; ".join(pb["invariants"]) or "(none)",
        plan=PLAN_TEXT.format(system=plan.get("system", ""),
                              connection=plan.get("connection", ""))
        if plan else "")


# --- the partner for a combination (19 Oct 2026) ---
# Partners drawn at random from a fitting hardware group gave 1 borderline
# design in about 20 tries: font pixels fed to an audio filter, power steps
# fed to an image transform. The model now sees a short list of small
# starting designs, with what each is for and its ports, and picks the one
# a real chip would wire to A, or says none fits.

PARTNER_TEMPLATE = """\
Design A, a formally verified SystemVerilog module:

{card_a}

Candidate partners, each a verified module of its own:

{cards}

Pick the ONE candidate that a real chip would wire to design A, so that
together they form one system that does something neither does alone.
What one sends must mean something to the other: data into a data input,
an event into a trigger or enable, a request into a request input. Never
force a value into an input that expects something else: pixels are not
audio samples, and a status word is not random-number seed material. If no
candidate would really be wired to A, answer "none"; that is a good answer
when nothing fits.

Reply with JSON:
- choice: the candidate's id, or "none";
- system: one sentence on what the combined design does (or why none fits);
- connection: which output drives which input, in which direction, and what
  the value means at that input."""

PARTNER_SCHEMA = {"type": "object",
                  "properties": {"choice": {"type": "string"},
                                 "system": {"type": "string"},
                                 "connection": {"type": "string"}},
                  "required": ["choice", "system", "connection"],
                  "additionalProperties": False}

_DECL_WORDS = {"input", "output", "inout", "wire", "reg", "logic", "signed",
               "unsigned"}


def port_summary(verilog, top):
    """(inputs, outputs) of a module as name[width] lists."""
    masked = _mask_comments(verilog)
    span = _port_list_span(masked, top)
    ins, outs = [], []
    if not span:
        return ins, outs
    direction, width = None, ""
    for item in _split_outside_brackets(masked[span[0]:span[1]]):
        d = re.match(r"\s*(input|output|inout)\b", item)
        if d:
            direction = d.group(1)
            w = re.search(r"\[[^\]]*\]", item)
            width = w.group(0).replace(" ", "") if w else ""
        names = [x for x in _ID.findall(re.sub(r"\[[^\]]*\]", " ", item))
                 if x not in _DECL_WORDS]
        if names and direction:
            (ins if direction == "input" else outs).append(names[-1] + width)
    return ins, outs


def design_card(row, by_id=None, grown=None):
    """id, module, size, purpose and ports of one design."""
    root, by_id = row, by_id or {}
    while root.get("parent") and root["parent"] in by_id:
        root = by_id[root["parent"]]
    purpose = " ".join((root.get("construct") or "(not recorded)").split())
    ins, outs = port_summary(row["verilog"], row["top_module"])
    lines = [f"[{row['id']}] module {row['top_module']} "
             f"({honest_lines(row['verilog'])} lines)",
             f"  purpose: {purpose}"]
    if grown:
        lines.append("  grown since: " + "; ".join(grown))
    lines += [f"  inputs: {', '.join(ins)}", f"  outputs: {', '.join(outs)}"]
    return "\n".join(lines)


def partner_prompt(parent, candidates, by_id=None, history=None):
    return PARTNER_TEMPLATE.format(
        card_a=design_card(parent, by_id, history),
        cards="\n\n".join(design_card(c, by_id) for c in candidates))


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
   pool: a shared budget register of TOTAL credits. An
   instance can only gain a unit of its resource when the pool has a free
   credit; a credit returns to the pool when the work it paid for
   finishes. The wrapper may rely only on signals the parent exposes
   through its ports. Every copy's results must be used: route each copy's
   outputs to the wrapper's outputs or into the wrapper's logic (for
   example a mux by grant), and never XOR copies' results together just to
   keep them alive.
2. Expose every per-instance signal your invariants mention as a named
   top-level wire with an instance-numbered name (t0..t{last}, h0..h{last}).
   NEVER write inst.signal — hierarchical references are rejected: the
   toolchain silently turns them into dangling wires. Where the parent's
   facts need an internal register, the parent carries a mirror output
   port for it (for example `assign count_o = count;`): connect that port
   to a wrapper wire per instance and state the fact through it.
3. The wrapper contains exactly ONE new assertion: a property of the pool,
   immediate assertion, guarded by an antecedent where natural.
4. The invariant list must contain, for EVERY parent clause, one renamed
   copy per instance ({n} copies each), PLUS at least one clause that is a
   genuine sum or reduction across all {n} instances (for example
   pool + t0 + ... + t{last} == TOTAL). Per-instance and pairwise clauses
   do not count as that aggregate.
5. Same clock and reset names as the parent. Every new register gets a
   reset in the wrapper's if (rst) branch and an initial value. New
   registers that carry data match the parent's widths; counters are
   sized to their range. Watch summation width:
   zero-extend operands so the sum
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
  from reset, e.g. "pool == 4'd6". Prose like "pool reaches zero
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

# The model's account of each extension, above all the fake state that
# fools the proof (induction_gap, not_inductive_alone): the reasoning a
# second training run learns from. Both runners save it through
# keep_reasoning; until 4 Oct 2026 batch.py did not, and 372 rows lost it.
REASONING_FIELDS = ("new_state", "coupling", "induction_gap",
                    "why_parent_insufficient", "claim", "shared_state",
                    "not_inductive_alone", "shared_clause", "pool_scheme",
                    "why_aggregate_needed", "glue_scheme",
                    "why_parents_insufficient")


def keep_reasoning(out, record):
    for k in REASONING_FIELDS:
        if k in out:
            record[k] = out[k]
    return record

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


def one_new_assert(parent_props, child_asserts):
    """Like split_second_property_asserts, but the new assertion may sit
    anywhere in the file (a part beside the old logic, 18 Oct 2026): every
    parent assertion is still there, in its order, plus exactly one more.
    Returns (new_assert, None) or (None, error)."""
    p = [_norm(x) for x in parent_props]
    rest = list(child_asserts)
    for want in p:
        i = next((j for j, a in enumerate(rest) if _norm(a) == want), None)
        if i is None:
            return None, f"parent assertion not preserved verbatim: {want!r}"
        rest.pop(i)
    kept = [_norm(a) for a in child_asserts if _norm(a) in p]
    if kept != p:
        return None, f"parent assertions reordered: {p} vs {kept}"
    if len(rest) != 1:
        return None, (f"expected exactly one new assertion: parent has "
                      f"{len(p)}, child has {len(child_asserts)}")
    return rest[0], None


def remove_asserts(source, exprs):
    """Blank the assert(...) statements whose expression matches one of
    `exprs` (whitespace-normalized), leaving legal Verilog — used to
    isolate the second property for the proves-unaided check."""
    targets = {_norm(e) for e in exprs}
    out = source
    for start, i, j, concurrent in reversed(assert_spans(
            _mask_comments(source))):
        if _norm(source[i + 1:j]) in targets:
            end = j + 1
            if concurrent:
                # `assert property (e);` stands alone in the module: take
                # its semicolon too (19 Oct 2026)
                k = end
                while k < len(out) and out[k] in " \t":
                    k += 1
                if out.startswith(";", k):
                    end = k + 1
            out = out[:start] + out[end:]
    return out


_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# 4'b0101, 8'sd1, 'hff: numbers, not names. The signed and unsized forms
# were read as names ("sd1") until 17 Oct 2026, which made a fact like
# acc >= -8'sd1 look as if it needed a signal no design declares.
_BASED_LITERAL = r"\d*'[sS]?[bodhBODH][0-9a-fA-FxXzZ_?]+"
_VERILOG_NOISE = {"posedge", "negedge", "always", "assert", "module",
                  "endmodule", "begin", "end", "if", "else", "wire", "reg",
                  "input", "output", "initial", "assign",
                  "localparam", "parameter", "case", "casez", "casex",
                  "endcase", "default", "function", "endfunction",
                  "integer", "genvar", "generate", "endgenerate",
                  "signed", "unsigned", "logic", "assume", "cover"}


def identifiers(text):
    text = _mask_comments(text)   # comment words are not identifiers
    ids = set(_ID.findall(re.sub(_BASED_LITERAL, " ",
                                 text)))
    return ids - _VERILOG_NOISE


_TOP_MODULE_RE = r"\bmodule\s+%s\b(.*?)\bendmodule\b"


def out_of_scope(invariants, verilog, top_module):
    """Identifiers in the invariants that are not signals of the TOP
    module. Yosys fabricates a free wire for an unknown name, so such a
    clause grades as a fake FALSE; catch it as a naming error instead.
    Port names in instantiations (.name(...)) are masked - they belong
    to the sub-module."""
    m = re.search(_TOP_MODULE_RE % re.escape(top_module), verilog, re.S)
    text = m.group(1) if m else verilog
    text = re.sub(r"\.\s*[A-Za-z_]\w*", " ", text)
    return identifiers("\n".join(invariants)) - identifiers(text)


def p2_new_ids(parent_props, new_assert, child_verilog):
    """REGISTERS the second property reads that no parent property does.
    Only declared regs count — a new localparam or wire name is a new
    label, not new state. Empty means P2 rides entirely on the first
    property's support — the same-invariants, no-new-clause pattern —
    and is rejected."""
    new = identifiers(new_assert) - identifiers("\n".join(parent_props))
    return new & wrapper_regs(child_verilog)


def state_carriers(parent, child):
    """The registers a step added and the wires that carry them: a wire on
    an output port of a module the step added, or one assigned from new
    state. 18 Oct 2026: a sub-unit is now a module of its own, so its
    state reaches an assertion in the top module through a wire, and
    counting registers only turned such a step away (found grading a
    hand-made step with the real checker). A wire made from old state
    alone is still not new state."""
    import gates                      # gates imports this module
    carried = wrapper_regs(child) - wrapper_regs(parent["verilog"])
    new_modules = set(gates._modules(child)) - \
        set(gates._modules(parent["verilog"]))
    for mod, _, body in gates.instances(child, child, parent["top_module"]):
        if mod in new_modules:
            outputs = {p for p, d in gates.port_directions(child, mod).items()
                       if d == "output"}
            for port, expr in gates._CONNECTION.findall(body):
                if port in outputs:
                    carried |= identifiers(expr)
    driven = gates._DRIVEN.findall(_mask_comments(child))
    while True:
        more = {lhs for lhs, rhs in driven
                if lhs not in carried and identifiers(rhs) & carried}
        if not more:
            return carried
        carried |= more


def own_property_new_ids(parent, new_assert, child):
    """p2_new_ids for a part's own assertion: wires that carry new state
    count as well as registers (state_carriers)."""
    new = identifiers(new_assert) - identifiers("\n".join(parent["property"]))
    return new & (wrapper_regs(child) | state_carriers(parent, child))


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
    masked = re.sub(_BASED_LITERAL, " ", expr)
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
    """Names declared in the module's port header. Reads past a
    `#( parameter ... )` list (17 Oct 2026): the old one-line pattern
    found no header there, so every signal of a parameterised module,
    ports included, counted as hidden."""
    masked = _mask_comments(verilog)
    span = _port_list_span(masked, top_module)
    return identifiers(masked[span[0]:span[1]]) if span else set()


def _port_list_span(masked, top_module):
    """(start, end) of the text inside the top module's port parentheses,
    in comment-masked source, or None when it has no port list."""
    m = re.search(r"\bmodule\s+%s\b" % re.escape(top_module), masked)
    if not m:
        return None
    i = m.end()
    while i < len(masked) and masked[i].isspace():
        i += 1
    if masked.startswith("#", i):
        j = masked.find("(", i)
        k = _matching(masked, j) if j >= 0 else None
        if k is None:
            return None
        i = k + 1
        while i < len(masked) and masked[i].isspace():
            i += 1
    if not masked.startswith("(", i):
        return None
    close = _matching(masked, i)
    return (i + 1, close) if close is not None else None


def compose_hidden_signals(row):
    """Invariant identifiers a wrapper could never reach: mentioned in the
    parent's invariants but not exposed through its ports, directly or by
    a mirror port the port step added (`exposed`). Non-empty means the
    parent is ineligible for COMPOSE (and REPLICATE)."""
    ports = ports_of(row["verilog"], row["top_module"])
    used = set()
    for clause in row["invariants"]:
        used |= identifiers(clause)
    return used - ports - set(row.get("exposed", {}))


# --- the port step (17 Oct 2026) ---
# 243 of the 252 new designs could not be combined or copied: their facts
# name internal registers, and a wrapper may not reach inside an instance.
# Mirroring each such register on an output port changes no behaviour and
# makes 220 of them eligible; the other 23 need a memory or a function.

def _split_outside_brackets(text):
    """Split on commas that are outside (), [] and {}."""
    out, cur, depth = [], [], 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


_DECL_START = re.compile(r"(?:^|\n)[ \t]*(reg|logic|wire|bit)\b[ \t]*"
                         r"(signed\b)?\s*((?:\[[^\]]*\]\s*)*)", re.S)


def _declaration(body, name):
    """(signed, packed dimensions) of an internal reg/logic/wire/bit
    declared as `name` in a module body with comments masked, or None
    when it is not declared that way or is an array: a port cannot
    mirror a memory."""
    for stmt in body.split(";"):
        for m in _DECL_START.finditer(stmt):
            for part in _split_outside_brackets(stmt[m.end():]):
                d = re.match(r"\s*([A-Za-z_]\w*)\s*(\[)?", part)
                if d and d.group(1) == name:
                    if d.group(2):
                        return None
                    return bool(m.group(2)), " ".join(m.group(3).split())
    return None


def _matching(text, i):
    """Index of the bracket closing the one at text[i]."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for j in range(i, len(text)):
        if text[j] in pairs:
            stack.append(pairs[text[j]])
        elif stack and text[j] == stack[-1]:
            stack.pop()
            if not stack:
                return j
    return None


def expose_for_wrapper(row):
    """`row` with every internal register its facts mention mirrored on a
    new output port (`output wire [..] count_o` and `assign count_o =
    count;`), so a wrapper can state the facts per instance. The design's
    behaviour, property and facts are unchanged. Returns `row` itself when
    nothing is hidden, and None when a hidden name cannot be mirrored (an
    array, a function, a non-ANSI port list)."""
    hidden = compose_hidden_signals(row)
    if not hidden:
        return row
    src, top = row["verilog"], row["top_module"]
    masked = _mask_comments(src)
    m = re.search(_TOP_MODULE_RE % re.escape(top), masked, re.S)
    if not m:
        return None
    body = masked[m.start(1):m.end(1)]
    decls = {}
    for name in sorted(hidden):
        d = _declaration(body, name)
        if d is None:
            return None
        decls[name] = d
    span = _port_list_span(masked, top)
    if span is None:
        return None
    ports_text, close = masked[span[0]:span[1]], span[1]
    if not re.search(r"\b(input|output|inout)\b", ports_text):
        return None                       # names-only (non-ANSI) port list
    taken = identifiers(src)
    mapping, port_lines, assigns = {}, [], []
    for name in sorted(hidden):
        signed, dims = decls[name]
        port = next(p for p in [f"{name}_o", f"{name}_obs"]
                    + [f"{name}_obs{k}" for k in range(2, 99)]
                    if p not in taken)
        taken.add(port)
        mapping[name] = port
        kind = "output wire" + (" signed" if signed else "") \
            + (f" {dims}" if dims else "")
        port_lines.append(f"    {kind} {port}")
        assigns.append(f"    assign {port} = {name};\n")
    lead = ",\n" if ports_text.strip() else "\n"
    insert_ports = lead + ",\n".join(port_lines) + "\n"
    end = masked.rfind("endmodule", m.start(), m.end())
    line_start = src.rfind("\n", 0, end) + 1
    out = (src[:close] + insert_ports + src[close:line_start]
           + "".join(assigns) + src[line_start:])
    return dict(row, verilog=out, exposed=mapping)


_WRAPPABLE = {}


def wrappable(row):
    """Can a wrapper combine or copy this row, after the port step?
    Remembered per design: the sampler asks it of every row, every task."""
    key = (row.get("id"), hash(row["verilog"]), tuple(row["invariants"]))
    if key not in _WRAPPABLE:
        _WRAPPABLE[key] = expose_for_wrapper(row) is not None
    return _WRAPPABLE[key]


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

def build_prompt(parent, ext_type, move=None, n=6, k=1, history=None,
                 siblings=None, rejected=None):
    """`history`: the family's earlier steps, oldest first, as
    promote.history gives them; `siblings`: what other designs grown from
    this one added (promote.children); `rejected`: what was tried on this
    design and rejected (structural and second property only)."""
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
        diff_format=DIFF_FORMAT,
        history=history_text(history, siblings, rejected))
    if ext_type == "structural":
        fields["modules"] = modules_text(parent["verilog"],
                                         parent.get("top_module", ""))
        fields["effect_rule"] = (LEDGER_EFFECT_RULE if move == "LEDGER"
                                 else SIDEWAYS_EFFECT_RULE if move in BIG_MOVES
                                 else EFFECT_RULE)
        fields["move"] = f"{move}   {MOVES[move]}"
        if move == "STAGE" and k > 1:
            fields["move"] += STAGE_K_TEXT.format(k=k)
        if move in BIG_MOVES:
            n, lo, hi = size_target(parent["verilog"], move)
            fields["move"] += SIZE_TEXT.format(n=n, lo=lo, hi=hi,
                                               share=size_share(move))
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
        result = grade_triple_generated(json.dumps(payload),
                                        **grade_kwargs(record))
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
        result = grade_triple_generated(json.dumps(payload),
                                        **grade_kwargs(record))
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

    # a sub-unit is a module of its own since 18 Oct 2026, so a fact can
    # now reach for its registers by instance.signal; say so before grading
    bad = hierarchical_refs(out)
    if bad:
        record["verdict"] = "HIERARCHICAL_REF"
        record["error"] = (
            f"facts contain hierarchical references {bad} - the toolchain "
            "silently turns inst.signal into a dangling wire. Bring the "
            "register out on an output port of the new module, connect it "
            "to a wire in the top module and name that wire.")
        return record

    try:
        child = apply_patch(parent["verilog"], out["patch"])
    except PatchError as exc:
        record["verdict"] = "PATCH_ERROR"
        record["error"] = str(exc)
        return record
    record["child_verilog"] = child

    child_asserts = extract_asserts(child)
    sideways = False
    if ext_type == "structural":
        same = [_norm(a) for a in child_asserts] == \
            [_norm(a) for a in parent["property"]]
        # a big move may add a part beside the old logic with ONE assertion
        # of its own (18 Oct 2026); any other change of the asserts is not
        # allowed for a structural step
        sideways = (not same and record.get("move") in BIG_MOVES
                    and len(child_asserts) == len(parent["property"]) + 1)
        if not same and not sideways:
            record["verdict"] = "PROPERTY_CHANGED"
            record["error"] = (f"structural patches must not touch asserts: "
                               f"parent {parent['property']} vs child "
                               f"{child_asserts}")
            return record
        child_props = parent["property"]
    if ext_type == "second" or sideways:
        if sideways:
            new_assert, err = one_new_assert(parent["property"],
                                             child_asserts)
        else:
            new_assert, err = split_second_property_asserts(
                parent["property"], child_asserts)
        if err:
            record["verdict"] = "PROPERTY_CHANGED"
            record["error"] = err
            return record
        record["own_property" if sideways else "second_property"] = new_assert
        record["p2_new_ids"] = sorted(
            own_property_new_ids(parent, new_assert, child) if sideways
            else p2_new_ids(parent["property"], new_assert, child))
        if not record["p2_new_ids"]:
            record["verdict"] = "P2_SAME_SUPPORT"
            record["error"] = ("the new property reads no register the "
                               "existing ones do not already read — it "
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
    if sideways and not record["has_coupling_clause"]:
        record["verdict"] = "NO_COUPLING"
        record["error"] = ("a part with an assertion of its own needs at "
                           "least one fact naming both its new state and "
                           "existing state; without one it is a second "
                           "design bolted on the side")
        return record

    if ext_type == "second" or sideways:
        # P2 alone, no invariants: PROVEN here means it teaches nothing.
        p2_only = remove_asserts(child, parent["property"])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / f"{parent['top_module']}.sv"
            f.write_text(p2_only)
            with Spinner("checking the new property does not prove unaided"):
                solo = grade(f, PropertyInfo(top_module=parent["top_module"],
                                             clock=parent["clock"]),
                             **grade_kwargs(record, leg="p2_alone"))
        record["p2_unaided_tier"] = solo.tier.name
        record["p2_unaided_result"] = asdict(solo)   # keep everything
        if solo.tier.name == "PROVEN":
            record["verdict"] = "P2_PROVES_UNAIDED"
            record["error"] = ("the new property proves by k-induction with "
                               "no invariants — teaches nothing")
            return record

    payload = {"verilog": child, "top_module": parent["top_module"],
               "clock": parent["clock"], "antecedents": parent["antecedents"],
               "sanity_covers": parent["sanity_covers"],
               "invariants": out["invariants"]}
    t0 = time.monotonic()
    with Spinner(f"oracle grading extended {parent['top_module']}"):
        result = grade_triple_generated(json.dumps(payload),
                                        **grade_kwargs(record))
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
        if record.get("own_property"):
            print(f"its own assertion: {record['own_property']}")
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
            ready = expose_for_wrapper(row)
            mark = "OK " if ready is not None else "no "
            extra = ("" if not hidden else
                     f"  mirrored: {sorted(hidden)}" if ready is not None
                     else f"  cannot mirror: {sorted(hidden)}")
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
    if args.type == "replicate":
        ready = expose_for_wrapper(parent)
        if ready is None:
            p.error(
                f"{parent['id']} ({parent['top_module']}) is not "
                f"replicate-eligible: invariants mention signal(s) "
                f"{sorted(compose_hidden_signals(parent))} that no port can "
                "mirror (a memory or a function). Pick another parent "
                "(see --list-compose).")
        parent = ready
    if args.type == "compose":
        if not args.parent2:
            p.error("--parent2 required for compose")
        parent2 = load_parent(args.parent2)
        if parent2["top_module"] == parent["top_module"]:
            p.error("compose parents must have distinct module names")
        ready = []
        for row in (parent, parent2):
            r = expose_for_wrapper(row)
            if r is None:
                p.error(
                    f"{row['id']} ({row['top_module']}) is not compose-"
                    f"eligible: invariants mention signal(s) "
                    f"{sorted(compose_hidden_signals(row))} that no port can "
                    "mirror (a memory or a function). Pick another parent "
                    "(see --list-compose).")
            ready.append(r)
        parent, parent2 = ready
        print(f"parent2 {parent2['id']} {parent2['top_module']} "
              f"invariants={parent2['invariants']}")
        prompt = compose_prompt(parent, parent2)
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
    with Spinner(f"{llm_client.model_label(MODEL)} writing a {args.type} extension"):
        out, usage, stop = call_model(prompt, schema)
    record["usage"] = usage
    if out is None:
        record["verdict"] = {"refusal": "REFUSED",
                             "length": "TRUNCATED"}.get(stop, "UNPARSEABLE")
        record["raw_text"] = llm_client.LAST_RAW
        dump(record)
        print(f"{record['verdict'].lower()} - logged (raw text kept)")
        return
    keep_reasoning(out, record)
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
