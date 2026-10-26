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
  small storage  an array the step adds may hold at most 16 entries, all
                 its dimensions counted together, or as many as the
                 parent's own biggest array (not one built by a generate
                 loop, 23 Oct 2026); a deeper one is sent back before
                 grading (21 Oct 2026; 22 Oct 2026).

Inputs stay fixed, except the new lane of a PEER step (26 Oct 2026); new
state may get new OUTPUT ports (18 Oct 2026). Big steps and combinations
declare the streams of items they carry, and the checker searches for a
lost item before the proof (nothing_lost.py, 26 Oct 2026).

  venv/bin/python extender/extend.py --plan                      # the 5-seed plan
  venv/bin/python extender/extend.py --parent g0_014 --type structural --move GUARD --dry
  venv/bin/python extender/extend.py --parent g0_014 --type structural --move GUARD
  venv/bin/python extender/extend.py --parent g0_014 --type second
"""
import argparse
import ast
from collections import Counter
import itertools
import json
import math
import operator
import re
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import (_mask_comments, assert_spans,      # noqa: E402
                          extract_asserts, honest_lines, state_bits)
from distractor import (DIFF_FORMAT, MODEL, EFFORT,   # noqa: E402
                        Spinner, dump, grade_kwargs, load_parent,
                        necessity_check)
from patch import PatchError, apply_patch                 # noqa: E402

from fact_format import SYSTEM_FUNCTION                         # noqa: E402
from reset_reads import step_reset_error                        # noqa: E402
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
   and add no inputs. The one exception is a PEER step: its new lane gets
   an input stream of its own (its valid, its data and the like).
   You may add OUTPUT ports for new state, and a ready or busy output that
   tells a sender to wait. New state must be observable through an output,
   directly or through existing logic.
4. Match the module's own reset: every new register resets the same way,
   in the same kind of block, as the module's existing registers, and
   gets an initial value, matching the style already in the module.
5. Same Verilog subset as the module: synthesizable clocked blocks
   written like the module's own (the same clock edge and the same
   reset) and always @(*) logic, immediate assertions only, no concurrent
   SVA, no system tasks.
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
   new lines are mostly copies of each other is rejected.
11. Declare sizes as parameters, the way real designs do: how many entries
   a storage array holds, and how many channels, lanes or stages there
   are, as #(parameter DEPTH = 4) in the module header or as a localparam,
   with arrays, loops and pointer widths sized from them. The design is
   proved with these values, and engineers prove a design at small sizes,
   so keep them small: 4 to 8 entries, never more than 16. All the
   entries of one array count together: an array with two dimensions
   holds both sizes multiplied (4 channels of 8 entries are 32 entries),
   and a packed array of words (reg [3:0][7:0] w) holds one entry per
   word. A step that adds an array of more than 16 entries is rejected
   before it is checked, unless the module already has a bigger array,
   not counting registers built by a generate loop: a new array that
   holds one entry for each of the same things (one per slave or channel
   the module already has) may then hold as many entries as the module's
   biggest array. An existing array may not be made deeper than that
   either, or than it already is when it is bigger, and a register
   declared inside a generate loop counts once for each pass of the loop.
   Only counts are kept small: every entry keeps the module's own data
   width (rule 7).
12. Never read the reset in a new assertion, its guard or a fact, directly
   or through a wire that copies it: they describe the design in normal
   operation, not what holds while reset is held. Checkers disagree about
   what an asynchronous reset does within a cycle, so a step whose new
   assertion, guard or facts read one is rejected before it is checked.
   An assertion that must skip the cycles of a reset can be guarded by a
   register the reset clears."""

# 26 Oct 2026, Rishab approved (change 5): rule 3 allowed no inputs, so
# every PEER step of the 21 and 23 Oct trials split an existing input;
# nothing in code checked the rule. new_input_error now does, with PEER as
# the one exception.

# 23 Oct 2026, review finding F4: rule 12. The generator's prompt has said
# the same since 22 Oct (initiator/prompts.py); a step on a design with an
# asynchronous reset could read it, and step_reads_reset now turns such a
# step away.

# 22 Oct 2026, review finding F12: rules 4 and 5 said "a reset in the
# existing if (rst) branch" and "always @(posedge clk) or always @(*)
# logic", but since 21 Oct one new design in five resets at once
# (always @(posedge clk or negedge rst_n), if (!rst_n)); a question about
# such a design asked for a reset style it does not use. They now ask for
# the module's own reset, whichever it is (so do REPLICATE, COMPOSE and the
# distractor's question).
# 22 Oct 2026, review finding F8: rule 11 now says that all the entries of
# one array count together; a step with 4 channels of 8 entries was sent
# back as too deep while its depth was already 8. And, review finding F1,
# that an array may match the module's own biggest one (new_array_limit).
# 23 Oct 2026, review finding: rule 11 also says that an existing array may
# not be made deeper and that a register in a generate loop counts once per
# pass, as too_deep now judges them. And, Rishab chose the same day, that
# registers built by a generate loop do not raise the limit for a new
# array (new_array_limit).

# 21 Oct 2026: STAGE, PEER, SUBUNIT and STORAGE each say that their sizes
# are parameters with small values for the proof (SYSTEM_PROMPT rule 11;
# Rishab: "make all 1-8 changes"). Real designs declare sizes as
# parameters and are proved at small sizes (ZipCPU proves his FIFOs at
# small depths); in the 21 Oct trial 5 of 10 storage steps ran out of the
# checker's 300 s.
MOVES = {
    "SPLIT": ("Take a quantity that is currently held in one register and "
              "divide it, so part of it lives in a new register some of "
              "the time."),
    "STAGE": ("Insert a pipeline register between a producer and a "
              "consumer. An update that currently completes in one cycle "
              "now takes two, with the in-flight value held in new state. "
              "When there are several stages, declare how many as a "
              "parameter or localparam (localparam STAGES = 4) and size "
              "the stage array from it, never more than 16."),
    "GUARD": ("Add an enable register that gates an existing update. The "
              "existing transition fires only when the new enable is set."),
    # 22 Oct 2026, review finding F8: an array per channel counts every
    # channel's entries together, which this text did not say
    # 26 Oct 2026, Rishab approved (change 5): every PEER step of the 21 and
    # 23 Oct trials split the existing input (by a data bit, an ID range or
    # a command type), and "shares a resource" was met with a counter over
    # two separate queues. The new lane now has its own input stream (the
    # one exception to SYSTEM_PROMPT rule 3), shares one real thing through
    # an arbiter, and the lane that loses waits.
    "PEER": ("Add a second lane beside the existing one: similar logic with "
             "an input stream of its OWN (new input ports for its valid, its "
             "data and the like; this is the one step that may add inputs) "
             "and its own ready output. Never feed the new lane by "
             "splitting an existing input (by a data bit, an ID range or a "
             "command type): every existing input keeps doing exactly what "
             "it did. The two lanes share ONE real thing: the same storage "
             "array, the same output port or the same data path (one adder, "
             "one memory port, one bus), and an arbiter written as real "
             "logic (round robin or fixed priority) decides each cycle "
             "which lane uses it. The lane that loses waits: it keeps its "
             "item, its sender is told to wait through a ready or busy "
             "output, and the item is never dropped or overwritten, counted "
             "or not. If the existing lane has no way to make its sender "
             "wait, it wins every conflict, or you give it a ready output. "
             "Two separate queues joined only by a counter or a credit "
             "count do not share a real thing. Declare how many lanes there "
             "are as a parameter or localparam (localparam CHANNELS = 2), "
             "and give any storage it adds a parameter depth with a small "
             "value for the proof: 4 to 8 entries, never more than 16. All "
             "the entries of one array count together: an array with one "
             "row per channel (reg [7:0] buf_q [0:CHANNELS-1][0:DEPTH-1]) "
             "holds CHANNELS times DEPTH entries, and that product is what "
             "may not pass 16 (or the size of the module's own biggest "
             "array, when the new one holds an entry for each of the same "
             "things)."),
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
    # 26 Oct 2026, Rishab approved (change 1): five kept designs dropped
    # work and counted or flagged it; dropping is refused counted or not.
    "SUBUNIT": ("Add a complete new functional unit that works with the "
                "existing logic, the way the next feature of a real block "
                "is added: for example a request queue in front of an "
                "existing input, a second channel that shares an existing "
                "resource, a control and status register bank that "
                "configures an existing behaviour, or a retry or backoff "
                "controller around an existing operation. It holds real "
                "state of its own (several registers and the logic that "
                "drives them). Its sizes (how many entries, requests or "
                "channels it holds) are parameters of the new module with "
                "small values for the proof: 4 to 8 entries, never more "
                "than 16. Write it as its own module, with its own "
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
                "(see the list below). If the unit can refuse work in "
                "some state (it is full, busy or out of credits, or it "
                "takes a request only when it is idle), it must tell the "
                "sender to wait, through a ready or busy output the sender "
                "reads or by holding only the path that feeds the unit "
                "(see WAITING below), and never drop or overwrite work, "
                "counted or not: counting, flagging or raising an error "
                "bit for lost work does not make losing it acceptable."),
    # 18 Oct 2026: "on the path the property depends on" piled every
    # storage step onto one path (five buffers in a row in front of a
    # fixed table in the 18 Oct trial, all weak)
    # 21 Oct 2026, Gary's suggestion taken as general rules: storage built
    # the way real storage is built, and facts about what it holds (6% of
    # the 1,277 facts of the three growth trials were about stored entries)
    # 26 Oct 2026, Rishab approved (change 1): five kept designs of the 25
    # Oct study dropped data and counted or flagged it; dropping or
    # overwriting is refused counted or not (WAITING_RULE)
    "STORAGE": ("Add storage the design needs: a register file, FIFO or "
                "lookup table (an array with a write side and a read "
                "side), sized with the module's own widths, that holds "
                "values on their way to existing state or serves a new "
                "part with an assertion of its own. Build it the way real "
                "storage is built. Its depth is a parameter "
                "(#(parameter DEPTH = 4) or localparam DEPTH = 8) with a "
                "small value for the proof, 4 to 8 entries and never more "
                "than 16, and its pointers are sized from it. A queue "
                "keeps read and write pointers "
                "that wrap around, with full and empty worked out from the "
                "pointers (one extra wrap bit), not only from a separate "
                "count. When it is full or busy it tells the sender to wait "
                "through a ready output, and it never drops or overwrites "
                "data, counted or not (see WAITING below). "
                "An output register stage is welcome. The invariant list "
                "must relate the stored entries, or how many of them are "
                "in use, to the state around them, and at least one fact "
                "(or the new part's own assertion) must be about the "
                "stored values themselves, not only counts and pointers: "
                "for example an entry does not change while it waits, "
                "every valid entry holds what was written into it, or the "
                "read pointer points at the oldest valid entry."),
}

# the moves that add a whole piece of design; their question says how big
BIG_MOVES = ("SUBUNIT", "STORAGE", "PEER")

SIZE_TEXT = """
        Size: this is a substantial addition. The module now has {n} lines;
        your patch should add roughly {lo} to {hi} new lines ({share}),
        written out as real logic, not padding, comments, copies of
        existing lines or one block repeated per slot. Lines are counted
        with copies counted once."""

# 22 Oct 2026, review finding F1: a structural question on a design that
# already has an array of more than MAX_NEW_DEPTH entries says how big a
# new one may be (new_array_limit), so a table with one entry per existing
# slave is not cut to 16 for a design with 24 slaves.
ARRAY_ALLOWANCE_TEXT = """
        This design already has an array of {n} entries ({array} in
        module {module}). An array the step adds may hold up to {n}
        entries, all its dimensions counted together, when it holds one
        entry for each of the same things (one per slave, channel or slot
        the design already has); any other new array keeps 4 to 8
        entries, never more than {cap}."""

# 18 Oct 2026, hand review of two trials: every good step grew its design
# 1.4 to 1.7 times, inside the +50-75% the question asked for, so the size
# of a good design was set by the question. The two moves that stayed good
# (a sub-unit 5 of 5, storage 5 of 7) now ask to double the design
# (Rishab: "do all 6 changes").
DOUBLING_MOVES = ("SUBUNIT", "STORAGE")

# 21 Oct 2026: answers were cut off. In the 21 Oct trial 2 storage answers
# stopped at the 64,000-token limit (TRUNCATED) after the question asked
# designs of 381 and 397 lines for about as many new lines again, and
# other big answers there used up to 62,500 tokens. Once a design is over
# BIG_STEP_CAP lines, every step asks for BIG_STEP_FEWEST to BIG_STEP_CAP
# new lines, whatever the move (Rishab: "make all 1-8 changes").
# 22 Oct 2026, review finding F7: a sub-unit or storage step on a design of
# 241 to 300 lines was still asked for up to 375 new lines, more than a
# design over 300 lines is asked for. The cap now also holds whenever the
# usual ask goes over BIG_STEP_CAP (size_capped).
BIG_STEP_CAP = 300
BIG_STEP_FEWEST = 200


def _usual_ask(move, n):
    """(fewest, most) new lines for a design of `n` lines before the cap:
    about as much again for a sub-unit or storage step, half to three
    quarters for any other."""
    if move in DOUBLING_MOVES:
        return n, round(n * 1.25)
    return round(n * 0.5), round(n * 0.75)


def size_capped(move, n):
    """Is the ask for a design of `n` lines held at BIG_STEP_FEWEST to
    BIG_STEP_CAP? Yes for a design over BIG_STEP_CAP lines, and whenever
    the usual ask would go over BIG_STEP_CAP (22 Oct 2026)."""
    return n > BIG_STEP_CAP or _usual_ask(move, n)[1] > BIG_STEP_CAP


def size_target(verilog, move=None):
    """(lines now, fewest, most) new lines: a sub-unit or storage step
    roughly doubles a design, other big steps grow it by half to three
    quarters (Rishab, 17 Oct 2026: "we want to see rapid growth"); a
    design over BIG_STEP_CAP lines (21 Oct 2026), or one whose usual ask
    would go over BIG_STEP_CAP (22 Oct 2026), is asked for BIG_STEP_FEWEST
    to BIG_STEP_CAP. Lines are counted with copies counted once
    (honest_lines, 19 Oct 2026), so a design swollen by copied slots is
    not asked for still more."""
    n = honest_lines(verilog)
    if size_capped(move, n):
        return n, BIG_STEP_FEWEST, BIG_STEP_CAP
    return (n,) + _usual_ask(move, n)


def size_share(move, n=0):
    """size_target's request in words, for a design of `n` lines."""
    if n > BIG_STEP_CAP:
        return (f"at most {BIG_STEP_CAP} for a design over {BIG_STEP_CAP} "
                "lines, so that the whole answer fits in one reply")
    if size_capped(move, n):
        return (f"at most {BIG_STEP_CAP} in one step, so that the whole "
                "answer fits in one reply")
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

# 26 Oct 2026, Rishab approved (Gary's suggestion, the hardware expert):
# IBEX_FACT_PATTERNS, the last bullet of INVARIANT_RULES and a paragraph of
# the combining question, gives the fact patterns of the Ibex core's own buffer
# proofs (lowRISC Ibex, formal/icache/formal_tb.sv, its "internal
# (induction) assertions": the older-than order of its fill buffers, the
# requests-in-flight count, oldest first, the oldest buffer's address,
# one-hot selects), written as our same-cycle facts. Ibex is real hardware
# and not in the benchmark.
IBEX_FACT_PATTERNS = """\
- When the design keeps several entries (buffers, queue slots, requests
  in flight), these are the facts real processor engineers write for
  their proofs (the Ibex core's own buffer proofs):
  (1) order among entries: when entries carry an age or an order (an
  older-than table, sequence numbers), no busy entry is older than
  itself, two busy entries are never each older than the other, the
  order is transitive, and any two busy entries are ordered, for example
  `forall k in 0..3, m in 0..3 where k != m: !(busy[k] && busy[m]) || (older[k][m] != older[m][k])`;
  (2) work in flight: what was requested minus what came back, added up
  over the entries, equals the count of requests still outstanding, and
  no entry has more replies than requests;
  (3) oldest first: an entry has started its work only when every older
  busy entry has finished;
  (4) the oldest busy entry's address is the address being sent out, and
  the entry after it holds the next address;
  (5) a select that picks one entry has at most one bit set, written as a
  sum (sel[0] + sel[1] + sel[2] + sel[3] <= 1)."""

# 21 Oct 2026: facts are same-cycle expressions with no system functions,
# which the benchmark's checker cannot read (system_function_facts below);
# and with sizes now parameters (SYSTEM_PROMPT rule 11), a compact fact's
# range is still written in numbers, the only form expand_facts reads
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
  syntactically identical to any assertion is rejected outright.
- Good facts relate whole words to each other (a pointer equals a count
  plus the entries in flight; the credits held add up to the budget) and
  relate values the design moves from one cycle to the next (a stage holds
  what the stage before it held; an entry holds what was written into it).
  These are what provers find hard bit by bit; prefer them to long case
  lists over single bits.
- Every fact is a same-cycle boolean expression over the design's
  signals: it reads them as they are in one cycle. A relation across
  cycles is written between the registers that carry a value from one
  cycle to the next (a pipeline stage and the stage before it, a delay
  line, a stored entry, a captured copy), never with a system function:
  no $past, $stable, $rose, $fell, $changed, $countones, $onehot,
  $onehot0, $isunknown or $clog2 in a fact. The checker that scores the
  facts cannot read them, so a fact that uses one is rejected. Write a
  count of set bits as a sum (v[0] + v[1] + v[2]), and supersede a parent
  fact that uses one with the same relation written without it. The
  design's own assertions are not facts and may still use $past.
- A fact about every entry or stage of an array can be written once with
  an index, `forall k in 0..7: !v[k] || (tag[k] != 4'd15)`, and a fact
  about pairs of entries with two indexes,
  `forall k in 0..7, m in 0..7 where k != m: !(v[k] && v[m]) || (age[k] != age[m])`.
  Use index names that are not signals of the module. Write the range in
  numbers, the value a size parameter has (0..3 when DEPTH = 4), never
  its name. Each such fact is checked once for every index value."""
INVARIANT_RULES += "\n" + IBEX_FACT_PATTERNS

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

# 26 Oct 2026, Rishab approved (changes 1 and 2): the root-cause study of
# 25 Oct found 8 of 14 combined and second-lane designs losing data, 5 of
# them kept because the loss was counted or flagged; and the advice to hold
# the sender's enable low froze whole designs (run c2_004, step t033, lost
# route-table writes). Big moves get this block; COMPOSE and the partner
# question say the same in their own words.
WAITING_RULE = """\
WAITING. Never drop work, counted or not. When a part is full or busy, or
takes an item only in some states, the item waits and its sender is made
to wait; nothing is dropped, overwritten or ignored. Counting, flagging or
raising an error bit for a lost item does not make losing it acceptable.
Make the sender wait on the stream's own path only:
- a sender outside the module waits on a ready or busy OUTPUT (one you add
  or one the module has), low exactly in the cycles its item would not be
  taken;
- logic inside the module waits by holding only the registers that carry
  the stream (its valid, its data, its pointers), never by gating the
  module's enable for everything: that also stops the settings, table
  writes and other inputs that arrive meanwhile;
- whatever pause you add shows on an output, so whoever drives an input
  that is not taken while paused knows to wait.

"""

# 26 Oct 2026, Rishab approved (change 5): with SIDEWAYS_EFFECT_RULE's "must
# write, or gate the write of, something the property reads", the PEER
# steps of the 21 and 23 Oct trials pulled the new lane into the existing
# path and the existing lane stopped working as before. A second lane
# affects the existing logic only through the thing it shares.
PEER_EFFECT_RULE = """\
The second lane affects the existing logic in one way only: through the
one thing the lanes share and the arbiter that decides who uses it. While
the arbiter gives the shared thing to the second lane, the existing lane
waits; that wait is how the new state gates what the property or the
parent invariants read. Beyond the shared thing, leave the existing lane
as it was: do not route the second lane's items through the existing
lane's input, counters or state machine, and do not change anything the
existing lane does while the second lane is idle.
The second lane may also come with ONE new assertion of its own about the
arbiter or the shared thing:
- put the assertion in the existing (top) module, in its own always block
  in the same style as the existing one;
- it must not prove without facts;
- it must read at least one register no existing assertion reads;
- at least one fact must name both new state and state the existing facts
  describe;
- never change or remove an existing assertion;
- in new_state below, say what the assertion says; the induction_gap may
  break that assertion instead of the existing property."""

# 26 Oct 2026, Rishab approved (change 3): the checker, not the model,
# counts what is lost; the model only says where the items go. It counts
# what the SENDER hands over: counting only the items the design chose to
# take missed every drop in the 25 Oct study.
STREAMS_TEXT = """\
STREAMS. The checker counts, by itself, every item handed over to the part
your step adds or changes (an item is one request, data word, command or
event), and searches every input sequence for many cycles from the start
for an item that is neither delivered nor still held. A step it finds
losing an item is sent back. In "streams", declare each stream of items
the step adds or changes:
- name: a few words, for example "requests into the new queue";
- inputs: one entry for each place items enter (two lanes that share one
  thing are one stream with two inputs), each with:
  offer: true in a cycle when the sender has an item for the stream. For a
    sender outside the design (its items arrive on input ports): its valid
    input, and the design's enable if it has one ("en && in_valid"), only
    input ports. For a sender inside the design (logic it already has,
    logic you add, or in a combination the sending design): that sender's
    own valid signal ("st_v", "a_out_valid"), which stays true while it
    waits. Never fold in whether the item will be taken (full, busy, a
    grant, your new state): that belongs in ready.
  ready: what tells the sender its item is taken this cycle. For a sender
    outside the design: the ready or busy output it reads ("in_ready" or
    "!busy_o"). For a sender inside: the condition under which its item is
    taken, with the design's enable if it has one ("en && !q_full").
    Write "" when the sender cannot be made to wait: then every cycle its
    offer is true counts as one item handed over, so the offer must be
    true for exactly one cycle per item;
- delivered: how many items leave the stream this cycle, taken by the part
  they are sent to ("q_pop", "out_valid && out_ready"; a true/false
  condition counts one, a sum counts several). An item thrown away on
  purpose because of what it is (a bad checksum, an address a setting
  filters out) may be counted here; one lost because a part was full or
  busy never may;
- held: how many items the stream holds now, counting every place an item
  can wait (queue entries, holding registers, an output stage): the name
  of a count the design keeps, a sum of 1-bit flags ("q_count",
  "slot_v[0] + slot_v[1]") or "0";
- flush: an input that empties the stream on purpose (a clear, abort or
  flush command), or "" when there is none.
Write every value as a Verilog expression over the top module's signals:
no prose, no instance.signal, no $ functions, and no subtraction in
delivered or held (name a wire of the design that holds a difference of
pointers at their own width). Never name a signal that counts or flags
lost items. {rule}

"""

STREAMS_RULES = {
    "STORAGE": ("This step must declare at least one stream through its "
                "storage. For a register file or lookup table the stream "
                "is its writes: an item is delivered when it is written."),
    "PEER": ("This step must declare the stream through the thing its two "
             "lanes share, with at least two inputs: the new lane's, whose "
             "offer reads its own new inputs, and the existing lane's, "
             "whose offer reads the inputs it had before."),
    "SUBUNIT": ("When the new unit holds or queues work (a queue, a buffer, "
                "holding registers), declare at least one stream through "
                "it, and a unit that adds an array must. Write [] only when "
                "it carries no items and adds no array (a timer, a "
                "controller); a register bank declares its writes."),
    "compose": ("This combination must declare at least one stream from "
                "the sending design through the part between them into the "
                "receiving one: its offer reads the sender's output wires, "
                "its ready, if any, is what drives the sender's ready, busy, "
                "request or valid input, and delivered counts the receiver "
                "taking the item, judged from what the receiver does: if it "
                "takes an item only in some states (only when idle, not "
                "while busy), delivered includes that condition."),
}


def streams_text(kind):
    """The STREAMS block for a big move or "compose" (26 Oct 2026)."""
    return STREAMS_TEXT.format(rule=STREAMS_RULES[kind])

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


# 26 Oct 2026, Rishab approved (changes 1-3): {waiting}, {streams},
# {streams_step} and {streams_reply} carry the waiting rule and the
# STREAMS block of the big moves (build_prompt); they are empty for the
# other moves, whose question reads exactly as before.
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

{waiting}{streams}Your new invariant list must contain at least one clause mentioning both a
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

5. Then write the patch, {streams_step}then the full new invariant list.

{invariant_rules}

{disposition_block}

Diff format:
{diff_format}

Reply with JSON:
{{"new_state": "...", "coupling": "...", "induction_gap": "...",
  "why_parent_insufficient": "...", "patch": "...",{streams_reply}
  "invariants": ["..."], "dispositions": [...]}}"""

# 21 Oct 2026: this said new state is observed "through existing outputs",
# left from when the port list was fixed; SYSTEM_PROMPT rule 3 has allowed
# new OUTPUT ports since 18 Oct 2026, and every move text agrees with it
# (Rishab: "make all 1-8 changes", item 8c)
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
to. If you do, the rules above about resets, initial values, and making new
state observable through an output (an existing one or a new OUTPUT port)
still apply.

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

# 26 Oct 2026, Rishab approved (change 3): the streams a step declares for
# the checker's own nothing-lost search (nothing_lost.py). Strict
# structured output needs fixed keys: a stream with no ready, or no flush,
# writes "", a step with no stream writes [].
_STREAM_ITEMS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "inputs": {"type": "array", "items": {
                "type": "object",
                "properties": {"offer": {"type": "string"},
                               "ready": {"type": "string"}},
                "required": ["offer", "ready"],
                "additionalProperties": False}},
            "delivered": {"type": "string"},
            "held": {"type": "string"},
            "flush": {"type": "string"},
        },
        "required": ["name", "inputs", "delivered", "held", "flush"],
        "additionalProperties": False,
    },
}
# the moves that declare streams: the ones that add a whole piece of design
STREAM_MOVES = BIG_MOVES


def _schema(reasoning_fields, streams=False):
    """`streams`: the answer declares its streams right after the patch,
    before the facts, so the model names the signals of the code it has
    just written (26 Oct 2026, change 3)."""
    props = {f: {"type": "string"} for f in reasoning_fields}
    props["patch"] = {"type": "string"}
    if streams:
        props["streams"] = _STREAM_ITEMS
    props["invariants"] = {"type": "array", "items": {"type": "string"}}
    props["dispositions"] = _DISPOSITION_ITEMS
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


# 21 Oct 2026: in the 21 Oct trial 2 of 10 combinations were kept. The
# reviewer turned down a font bitmap row (8 one-bit pixels) fed in as a
# 32-bit brightness sample, and a source with no way to wait feeding a
# queue that can refuse; the two that failed UNUSED_COPIES at 69% padded a
# 64-bit word with zeros into a 128-bit input, which leaves 1,536 of the
# receiver's flip-flops holding zero for ever (measured with yosys: fed
# from a real source instead, the share is 101%). So requirement 1 and the
# partner question (PARTNER_TEMPLATE) now ask that the parts fit at the
# wires: widths that match or a conversion that keeps the meaning, and a
# sender that waits instead of losing data (Rishab: "make all 1-8
# changes", item 6).
# 26 Oct 2026, Rishab approved (changes 1, 2, 4 and 7): the forced register
# stage produced one-cycle-late copies (proven in 5 designs of the 25 Oct
# study) while the DECORATIVE guard it was for never fired (0 of 36
# combinations); the advice to hold A's enable low froze whole designs
# (t033 and c2_004 lost route-table writes); 8 of 12 partner answers called
# a receiver that refuses "never refuses". The part between the designs now
# does a real job, waiting holds back only the stream's own path, refusal
# is judged from what the receiver does, and either design may send.
COMPOSE_TEMPLATE = """\
Extension type: COMPOSE.

Wire the two verified modules below together into one system: one
design's results feed the other's inputs, in your plan's direction when
you have a plan (A to B, or B to A), through a part that does a real job
between them. You write ONLY the wrapper; both parent modules are
appended verbatim by the harness — do not repeat or modify them.

Module A:

{verilog_a}

A's property: {property_a}
A's invariants: {invariants_a}

Module B:

{verilog_b}

B's property: {property_b}
B's invariants: {invariants_b}

{plan}REQUIREMENTS

1. The wrapper instantiates A once and B once, and connects a value one of
   them produces to an input the other consumes. Below, the sender is the
   design that produces the value and the receiver the one that takes it.
   Connect them the way a real design would: what the sender sends must
   mean something to the receiver (data into a data input, an event into
   a start or trigger input, a request into a request input), never a
   value forced into an input that expects something else. The part
   between them must do a real job: a buffer that holds items and makes
   the sender wait when the receiver cannot take them, a conversion that
   keeps the meaning (a width, format or encoding change), or an arbiter
   that shares one input of the receiver between sources. A plain wire
   does no job, and neither does a register that only passes the value on
   one cycle later: never add a register just to have one. The two must
   also fit at the wires:
   - Widths: connect signals of the same width, or convert the value in a
     way that keeps its meaning and say how in glue_scheme. A number (a
     count, an address, a sample) may be zero- or sign-extended into a
     wider input, since it is still the same number, and a wider value
     may be cut down to the field the input expects. Eight 1-bit font
     pixels packed in a byte are not a 32-bit brightness sample. Do not
     pad a data word with zeros to fill a wider data input: the part of the
     receiver that stores the padding never changes and does no work.
   - Waiting: decide from what the receiver does whether it can refuse
     work, not from its signal names: it can when, in some state, it
     ignores or loses what arrives at an input (it takes a request only
     when it is idle, it is busy for several cycles after a start, it
     overwrites a value not yet used, or its full, busy or ready signal
     goes low). Then the part between them holds the item and makes the
     sender wait on the stream's own path: through the sender's ready or
     busy input, or by holding back the request, start or valid input that
     begins its next item; when the items come from outside the wrapper,
     show the wait on a ready output of the wrapper. Never drop data,
     counted or not: counting, flagging or raising an error bit for dropped
     data does not make dropping it acceptable.
   - Enables: connect each design's enable (the input that, when low,
     leaves all its state unchanged) straight to the wrapper's own enable
     input. Never hold it low to make the design wait, and never use it as
     the stream's valid, while that design's other control inputs come from
     outside: while it is low the design ignores them, so settings, table
     writes and requests that have nothing to do with the stream are lost.
2. Expose any per-parent signal your invariants mention as a named
   top-level wire. NEVER write inst.signal — hierarchical references are
   rejected (the toolchain silently turns them into dangling wires).
   Where a parent's facts need an internal register, the parent carries a
   mirror output port for it (for example `assign count_o = count;`):
   connect that port to a wrapper wire and state the fact through it.
3. The wrapper contains exactly ONE new assertion about the composed
   behaviour, guarded by an antecedent where natural.
4. The invariant list must contain every parent clause renamed to the
   wrapper's wires (one copy each), plus the facts your new assertion
   needs that neither parent's list has: usually a fact linking the two
   designs across the connection (what the sender has produced and what
   the receiver or the part between them holds), or one about the part
   between them when it holds state.
5. Same clock and reset names as the parents. Every new register resets
   the way the parents' registers do, in the same kind of block, and gets
   an initial value. Registers that carry data match the parents' widths;
   counters are sized to their range. Do not add a register that
   only copies a register a parent already has, or one that only passes
   a value on one cycle later.
6. No assume, no cover, no -> implication, no system tasks. Each fact is
   a same-cycle expression with no system function ($past, $stable,
   $countones, $onehot, ...): the checker that scores the facts cannot
   read them.

{ibex_patterns}

{streams}WORK IT OUT BEFORE YOU WRITE THE WRAPPER

1. glue_scheme: which design sends what to which, the job of the part
   between them (a buffer that makes the sender wait, a conversion that
   keeps the meaning, or an arbiter), what it holds if anything, and when
   it moves; two sentences.
2. induction_gap: a state of the combined design that satisfies your
   property, can be held indefinitely with enables low, and steps to a
   property violation; it is unreachable only because of how the two
   designs are connected.
3. why_parents_insufficient: why neither parent's invariants exclude it.
4. Then the wrapper module, then streams (see STREAMS above), then the
   complete invariant list. The wrapper field is PLAIN Verilog module
   source — not a diff, no +/- line prefixes, no ---/+++/@@ headers.

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
 "streams": [...], "antecedents": [...], "sanity_covers": [...],
 "invariants": ["..."]}}"""


# 26 Oct 2026, Rishab approved (change 7): the plan is made from ports and
# purposes alone, and 8 of 12 partner answers of the 25 Oct study said the
# receiver never refuses, at least twice falsely; the plan's direction is
# kept (9 of 21 plans send from B to A) and checked against the code.
PLAN_TEXT = """\
You picked B to work with A. The combined system: {system}
Planned connection: {connection}
The plan was made from the two designs' ports and purposes alone. Build
the wrapper the way you planned, in the plan's direction, but check it
against the code above first: decide from what the receiver does whether
it can refuse work (it ignores or loses an input in some state, for
example it takes a request only when it is idle), whatever the plan says.

"""


def compose_prompt(pa, pb, plan=None):
    """The COMPOSE question for A and B; `plan` is the model's own partner
    choice ({"system", "connection"}), when it chose B (19 Oct 2026). It
    asks for the stream from one design into the other (26 Oct 2026)."""
    return COMPOSE_TEMPLATE.format(
        verilog_a=pa["verilog"], verilog_b=pb["verilog"],
        property_a="; ".join(pa["property"]),
        property_b="; ".join(pb["property"]),
        invariants_a="; ".join(shown_facts(pa)) or "(none)",
        invariants_b="; ".join(shown_facts(pb)) or "(none)",
        plan=PLAN_TEXT.format(system=plan.get("system", ""),
                              connection=plan.get("connection", ""))
        if plan else "",
        streams=streams_text("compose"),
        ibex_patterns=IBEX_FACT_PATTERNS)


# --- the partner for a combination (19 Oct 2026) ---
# Partners drawn at random from a fitting hardware group gave 1 borderline
# design in about 20 tries: font pixels fed to an audio filter, power steps
# fed to an image transform. The model now sees a short list of small
# starting designs, with what each is for and its ports, and picks the one
# a real chip would wire to A, or says none fits. 21 Oct 2026: the pick must
# also fit at the wires, widths and waiting (see COMPOSE_TEMPLATE).

# 26 Oct 2026, Rishab approved (changes 2 and 7): "can refuse" was defined
# by signal names, and 8 of 12 partner answers said the receiver never
# refuses, at least twice falsely; 9 of 21 plans send from B to A; an
# enable held low is not a way to wait.
PARTNER_TEMPLATE = """\
Design A, a formally verified SystemVerilog module:

{card_a}

Candidate partners, each a verified module of its own:

{cards}

Pick the ONE candidate that a real chip would wire to design A, so that
together they form one system that does something neither does alone.
Either may send to the other.
What one sends must mean something to the other: data into a data input,
an event into a trigger or enable, a request into a request input. Never
force a value into an input that expects something else: pixels are not
audio samples, and a status word is not random-number seed material.

The connection must also fit at the wires:
- Widths: an output drives an input of the same width, or the value is
  converted in a way that keeps its meaning, and you say how. A number (a
  count, an address, a sample) may be zero- or sign-extended into a wider
  input, since it is still the same number, and a wider value may be cut
  down to the field the input expects. Eight 1-bit font pixels packed in
  a byte are not a 32-bit brightness sample. A data word padded with
  zeros to fill a wider data input does not fit: the part of the receiver
  that stores the padding never changes and does no work.
- Waiting: decide from what the receiving design does whether it can
  refuse or stall, not from its signal names. It can when, in some state,
  it ignores or loses what arrives at an input: it takes a request only
  when it is idle, it is busy for several cycles after a start, it
  overwrites a value not yet used, or its full, busy or ready signal goes
  low. A design with no full, busy or ready signal can still refuse. If
  it can, the sending design must be able to wait on its own path,
  through a ready or busy input of its own, or a request, start or valid
  input that can be held back while the rest of it keeps running, so that
  no data is ever dropped, counted or not. Holding a design's enable low
  does not count as waiting: it also stops that design's settings, table
  writes and other inputs. A sender that cannot be made to wait cannot
  feed a receiver that can refuse.

If no candidate would really be wired to A this way, answer "none"; that
is a good answer when nothing fits.

Reply with JSON:
- choice: the candidate's id, or "none";
- system: one sentence on what the combined design does (or why none fits);
- connection: which output drives which input, in which direction (A to
  B or B to A), the width on each side and how a difference is
  converted, what the value means at that input, in which states the
  receiver ignores or loses what arrives, and how the sender waits when
  the receiver cannot take more (only if the receiver takes every input
  in every state, say so and why)."""

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
    # the stream from one design into the other, right after the wrapper
    # (26 Oct 2026, Rishab approved, change 3)
    props["streams"] = _STREAM_ITEMS
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
5. Same clock and reset names as the parent, and the parent's own kind
   of reset: every new register resets the same way, in the same kind of
   block, as the parent's registers, and gets an initial value. New
   registers that carry data match the parent's widths; counters are
   sized to their range. Watch summation width:
   zero-extend operands so the sum
   cannot wrap.
6. No assume, no cover, no -> implication, no system tasks. Each fact is
   a same-cycle expression with no system function ($past, $stable,
   $countones, $onehot, ...): the checker that scores the facts cannot
   read them.

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
# 26 Oct 2026, Rishab approved (change 3): the big moves declare their
# streams; STRUCTURAL_SCHEMA stays as it was for the other moves
STREAM_STRUCTURAL_SCHEMA = _schema(
    ["new_state", "coupling", "induction_gap", "why_parent_insufficient"],
    streams=True)


def structural_schema(move):
    """The structural schema for `move`: with streams for the big moves
    (26 Oct 2026), as before for the others."""
    return STREAM_STRUCTURAL_SCHEMA if move in STREAM_MOVES \
        else STRUCTURAL_SCHEMA


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


# --- compact facts (21 Oct 2026) ---
# One fact for every entry of an array written once with an index
# ("forall k in 0..7: !v[k] || ..."), pairs of entries included, expanded
# into one fact per index value before anything is checked (Gary's
# suggestion). With storage now written as arrays (SYSTEM_PROMPT rule 10),
# per-entry facts written out one by one would soon pass the 64-fact limit.
# Rows keep both: "invariants" (expanded, what the checker proves) and
# "invariants_compact" (as written, shown in the next question).

_FORALL = re.compile(r"^\s*forall\s+(.+?)\s*:(.*)$", re.S)
_BINDER = re.compile(r"^\s*([A-Za-z_]\w*)\s+in\s+(\d+)\s*\.\.\s*(\d+)\s*$")
MAX_INDEX_VALUES = 64
MAX_EXPANDED = 256


def _index_condition(cond, names):
    """The `where` condition as Python, or None when it uses anything but
    its indexes, numbers and comparisons."""
    for tok in re.findall(r"[A-Za-z_]\w*|\d+|&&|\|\||[<>!=]=|\S", cond):
        if re.match(r"[A-Za-z_]", tok) and tok not in names:
            return None
        if not re.match(r"[A-Za-z_]|\d|&&|\|\||[<>!=]=|[<>()+\-*%!]$", tok):
            return None
    py = cond.replace("&&", " and ").replace("||", " or ")
    return re.sub(r"!(?!=)", " not ", py)


def expand_facts(facts):
    """(facts with every compact fact expanded, None), or (None, error)
    when a compact fact is malformed or expands too far."""
    out = []
    for fact in facts:
        m = _FORALL.match(fact)
        if not m:
            out.append(fact)
            continue
        binders, body = m.group(1), m.group(2).strip()
        if not body:
            return None, f"compact fact has no fact after the colon: {fact!r}"
        cond = None
        parts = re.split(r"\s+where\s+", binders, maxsplit=1)
        if len(parts) == 2:
            binders, cond = parts
        names, ranges = [], []
        for part in binders.split(","):
            b = _BINDER.match(part)
            if not b:
                return None, ("a compact fact's index must read "
                              f"'k in LO..HI': {fact!r}")
            lo, hi = int(b.group(2)), int(b.group(3))
            if hi < lo or hi - lo + 1 > MAX_INDEX_VALUES:
                return None, (f"index range {lo}..{hi} is empty or longer "
                              f"than {MAX_INDEX_VALUES}: {fact!r}")
            names.append(b.group(1))
            ranges.append(range(lo, hi + 1))
        test = None
        if cond is not None:
            test = _index_condition(cond, names)
            if test is None:
                return None, ("a compact fact's condition may compare only "
                              f"its indexes: {cond!r}")
        for values in itertools.product(*ranges):
            env = dict(zip(names, values))
            try:
                if test is not None and not eval(test, {"__builtins__": {}},
                                                 env):
                    continue
            except Exception as exc:
                return None, f"condition {cond!r} cannot be read: {exc}"
            text = body
            for name, value in env.items():
                text = re.sub(rf"(?<![\w'$]){re.escape(name)}(?!\w)",
                              str(value), text)
            out.append(text)
            if len(out) > MAX_EXPANDED:
                return None, (f"compact facts expand to more than "
                              f"{MAX_EXPANDED} facts")
    return out, None


def shown_facts(row):
    """The facts a question shows: as written when there is a compact
    form, else the plain list."""
    return row.get("invariants_compact") or row["invariants"]


# --- facts the benchmark's checker can read (21 Oct 2026) ---
# EBMC 6.0, the checker that scores the trained model's facts, cannot read
# a system function in a fact: a fact with $past gives "unknown identifier
# $past" (ERROR), and so do $countones, $onehot, $onehot0, $stable, $rose,
# $changed, $signed and $clog2 (each tried on EBMC 6.0 on 21 Oct 2026).
# The benchmark's question also forbids them in facts. So a fact with one
# is a format failure; a relation across cycles is written between the
# registers that carry values (INVARIANT_RULES). The design's own
# assertions are not facts and may still use them (Rishab: "make all 1-8
# changes"). A `$` inside a name (a$b) is part of the name.
# 22 Oct 2026: the pattern is fact_format's, so the extender, the fixer and
# the training builder share one definition of a system function.
_SYSTEM_FUNCTION = SYSTEM_FUNCTION


def system_function_facts(facts):
    """The facts that use a system function ($past, $countones, ...)."""
    return [f for f in facts if _SYSTEM_FUNCTION.search(f)]


def system_function_error(facts):
    """Why facts with a system function are turned away and how to write
    them instead, or None when there are none."""
    bad = system_function_facts(facts)
    if not bad:
        return None
    names = sorted({m.group(0) for f in bad
                    for m in _SYSTEM_FUNCTION.finditer(f)})
    shown = "; ".join(bad[:3]) + (f" (and {len(bad) - 3} more)"
                                  if len(bad) > 3 else "")
    return (f"facts may not use system functions ({', '.join(names)}): the "
            "checker that scores the facts cannot read them. Write each "
            "fact about one cycle of the design. For a relation across "
            "cycles, relate the registers that hold the earlier value "
            "instead: a pipeline stage and the stage before it, a delay "
            "line, a stored entry or a captured copy. Write a count of set "
            "bits as a sum (v[0] + v[1] + v[2]). Facts turned away: "
            + shown)


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


# 26 Oct 2026, Rishab approved (change 4): glue_clause, which required a
# fact about the wrapper's registers, is gone with the forced register (see
# grade_compose).


def has_coupling_clause(new_invs, parent_ids, new_ids):
    for clause in new_invs:
        ids = identifiers(clause)
        if ids & new_ids and ids & parent_ids:
            return True
    return False


# --- storage small enough to prove (21 Oct 2026) ---
# Real designs declare sizes as parameters, and engineers prove them at
# small sizes (ZipCPU proves his FIFOs at small depths). In the 21 Oct
# trial 5 of 10 storage steps ran out of the checker's 300 s, and 6 steps
# added an array of 24 or 32 entries: 3 of them ran out of time. The
# question now asks for sizes as parameters with small values
# (SYSTEM_PROMPT rule 11), and a step that still adds a deeper array is
# sent back before anything is checked, as a format failure the run
# retries (Rishab: "make all 1-8 changes"). Only arrays the step adds are
# judged; a depth that cannot be worked out is let through.
# 22 Oct 2026, review finding F1: of the 6 steps of that trial the rule
# sends back, 5 held one entry for each of the 24 slaves of a design that
# already had a 24-entry queue (t008, t014, t026, t029, t035; t008 and
# t026 were kept as good designs); only t028's 32-byte replay memory was a
# free choice. A new array may now hold as many entries as the parent's
# own biggest array when that is more than MAX_NEW_DEPTH
# (new_array_limit). The parent's flat vectors (reg [23:0] mask, one bit
# per slave) are not counted: by its declaration such a vector cannot be
# told from a data word, and counting words would let any design with a
# 32-bit word add 32-entry arrays.
# 22 Oct 2026, review finding F5: a packed array of words
# (reg [255:0][7:0] dly) escaped the rule; its outer dimensions now count
# as entries. A flat vector (reg [24*8-1:0] dly) still cannot be told from
# a wide word and is left alone.
# 23 Oct 2026, review finding: storage written in other ways still escaped
# the rule, each at 256 or 1024 entries: an array of int, byte, shortint or
# longint; an array whose entries are a type the design declares (typedef,
# struct, enum), or a typedef that is itself an array; a depth set by a
# `define; registers declared inside a generate loop (c0_102 of 9 Oct
# builds 256 slots that way), each one counting once for each pass of the
# loop; and an old module given a deeper size by a new or changed instance,
# never judged because its array was "old". All of these now count, and
# an array the parent already has is judged too, by the size the step
# builds it at: it may not grow past the limit (which is never below the
# parent's own size for it). No trial step made an old array deeper.
MAX_NEW_DEPTH = 16      # most entries a new array may hold: 4 to 8 asked for

_MODULE_TEXT = re.compile(r"\bmodule\s+([A-Za-z_]\w*)(.*?)\bendmodule\b",
                          re.S)
_PARAM_WORD = re.compile(r"\b(parameter|localparam)\b")
# the type between `parameter` and its name: integer, logic [3:0], ...
_PARAM_TYPE = re.compile(r"\s*(?:(?:integer|int|logic|reg|bit|signed|"
                         r"unsigned)\b\s*|\[[^\]]*\]\s*)*")
# the words that declare state; int to longint since 23 Oct 2026
_STORAGE_WORDS = ("reg", "logic", "bit", "integer", "int", "byte",
                  "shortint", "longint")
_CONST_TOKEN = re.compile(
    r"\s*(?:(\d*\s*'[sS]?[bodhBODH]\s*[0-9a-fA-F_xXzZ?]+)|(\d[\d_]*)"
    r"|(\$clog2\b)|([A-Za-z_]\w*)|(\*\*|<<|>>|[-+*/%()]))")
_BASES = {"b": 2, "o": 8, "d": 10, "h": 16}
_CONST_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod, ast.LShift: operator.lshift,
              ast.RShift: operator.rshift, ast.Pow: operator.pow}
_CONST_LIMIT = 1 << 64  # no size is bigger; stops 2**2**99 from running on


def _const_node(node):
    """The value of a parsed constant expression; ValueError for
    anything that is not one."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op,
                                                    (ast.UAdd, ast.USub)):
        value = _const_node(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _CONST_OPS:
        a, b = _const_node(node.left), _const_node(node.right)
        if isinstance(node.op, (ast.Pow, ast.LShift)) and not 0 <= b <= 64:
            raise ValueError("power or shift out of range")
        value = _CONST_OPS[type(node.op)](a, b)
        if abs(value) > _CONST_LIMIT:
            raise ValueError("too big to be a size")
        return value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "clog2" and len(node.args) == 1
            and not node.keywords):
        value = _const_node(node.args[0])
        return (value - 1).bit_length() if value > 1 else 0
    raise ValueError("not a constant")


def const_value(expr, env):
    """The value of a constant expression made of numbers (16, 6'd48,
    'h10), parameters whose values are known (`env`), $clog2, + - * / %
    << >> ** and brackets; None when it uses anything else or a parameter
    whose value is not known."""
    text, i, py = expr.strip(), 0, []
    while i < len(text):
        m = _CONST_TOKEN.match(text, i)
        if not m:
            return None
        based, number, clog2, name, op = m.groups()
        if based:
            b = re.fullmatch(r"\d*\s*'[sS]?([bodhBODH])\s*([0-9a-fA-F_]+)",
                             based)
            try:
                py.append(str(int(b.group(2).replace("_", ""),
                                  _BASES[b.group(1).lower()])))
            except (AttributeError, ValueError):
                return None           # x, z or ? digits, or a bad digit
        elif number:
            py.append(number.replace("_", ""))
        elif clog2:
            py.append("clog2")
        elif name:
            if name not in env:
                return None
            py.append(f"({env[name]})")
        else:
            py.append("//" if op == "/" else op)
        i = m.end()
    try:
        return _const_node(ast.parse(" ".join(py), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError,
            OverflowError, TypeError):
        return None


def module_parameters(text):
    """[(name, value text, settable)] for the parameters and localparams
    of one module's text, in the order written; settable is True for a
    `parameter`, which an instance may set."""
    import gates                      # gates imports this module
    out = []
    for m in _PARAM_WORD.finditer(text):
        for item in _split_outside_brackets(
                gates._declaration_text(text, m.end())):
            if _PARAM_WORD.match(item.strip()):
                break                 # read under its own keyword
            rest = item[_PARAM_TYPE.match(item).end():]
            d = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*", rest, re.S)
            if d:
                out.append((d.group(1), d.group(2),
                            m.group(1) == "parameter"))
    return out


# Declarations that hold no state: a constant table, a type, an input
# port, or what a function returns. The last three matter since a packed
# array counts (22 Oct 2026): `input logic [3:0][7:0] din` is four words
# arriving each cycle, not four entries the checker must follow. Nor does
# a declaration right after a bracket (23 Oct 2026): a function's argument,
# or the variable of a loop (`for (int i = 0; ...)`), which since int
# counts would otherwise be read as a register.
_NOT_STORAGE_BEFORE = re.compile(
    r"(?:\b(?:parameter|localparam|typedef|input|inout|"
    r"function(?:\s+(?:automatic|static))?)|\()\s*$")
_DIMS = re.compile(r"\[([^\]]*)\]")


def _typedefs(text):
    """[(start, end, type name, dimension texts)] for every typedef in
    comment-masked `text`. The dimensions are what one value of the type
    holds as entries, as for a variable: its unpacked dimensions and its
    outer packed ones (every packed one but the last, the word); a struct
    or an enum has none of its own (23 Oct 2026, review finding)."""
    out = []
    for m in re.finditer(r"\btypedef\b", text):
        depth, j = 0, m.end()
        while j < len(text):
            if text[j] in "([{":
                depth += 1
            elif text[j] in ")]}":
                depth -= 1
            elif text[j] == ";" and depth == 0:
                break
            j += 1
        body = text[m.end():j]
        d = re.search(r"([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$", body)
        if not d:
            continue
        dims = _DIMS.findall(d.group(2))
        head = body[:d.start()]
        packed = re.search(r"((?:\[[^\]]*\]\s*)+)$", head)
        if packed and "{" not in head:     # not a struct or an enum
            dims = _DIMS.findall(packed.group(1))[:-1] + dims
        out.append((m.start(), j, d.group(1), dims))
    return out


def declared_types(text):
    """{type name: its dimension texts} for every typedef in comment-masked
    `text` (_typedefs)."""
    return {name: dims for _, _, name, dims in _typedefs(text)}


def _declarations(text, types=None, scalars=False):
    """[(name, [dimension text, ...])] for the state one text declares: a
    reg, logic, bit, integer, int, byte, shortint or longint, or a variable
    of a type in `types` (declared_types), the type's own dimensions first.
    Only arrays, unless `scalars`. Declarations that hold no state, and the
    members inside a typedef, are left out."""
    import gates                      # gates imports this module
    types = types or {}
    words = re.compile(r"\b(?:%s)\b" % "|".join(
        list(_STORAGE_WORDS) + [re.escape(t) for t in types]))
    inside = [(a, b) for a, b, _, _ in _typedefs(text)]
    out = []
    for m in words.finditer(text):
        if any(a <= m.start() < b for a, b in inside) or \
                _NOT_STORAGE_BEFORE.search(text, max(0, m.start() - 64),
                                           m.start()):
            continue
        rest = re.sub(r"^\s*(?:signed|unsigned)\b", "",
                      gates._declaration_text(text, m.end()))
        packed = re.match(r"\s*(?:\[[^\]]*\]\s*)*", rest).end()
        outer = types.get(m.group(0), []) + \
            _DIMS.findall(rest[:packed])[:-1]
        for item in _split_outside_brackets(rest[packed:]):
            s = item.strip()
            if re.match(r"(input|output|inout)\b", s):
                break
            d = re.match(r"([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)", s)
            if d and (scalars or outer or d.group(2)):
                out.append((d.group(1), outer + _DIMS.findall(d.group(2))))
    return out


def module_arrays(text, types=None):
    """[(name, [dimension text, ...])] for every array one module's text
    declares: a reg, logic, bit, integer, int, byte, shortint or longint
    with an entry index, or a variable of a type the design declares
    (`types`, declared_types) with one, the type's own dimensions first
    (int to longint and types since 23 Oct 2026, review finding). A
    constant table (localparam logic [7:0] T [0:31] = ...) holds no state
    and is not one. A packed array of words counts its outer dimensions,
    every packed one but the last (the word): reg [3:0][7:0] w [0:7] gives
    ["3:0", "0:7"] (22 Oct 2026, review finding F5)."""
    return _declarations(text, types)


# --- registers made by a generate loop (23 Oct 2026, review finding) -------
_GENVAR = re.compile(r"\bgenvar\s+([^;]+);")
_FOR = re.compile(r"\bfor\s*\(")
_BEGIN_END = re.compile(r"\b(begin|end)\b")


def _loop_count(header, genvars):
    """How many passes a generate loop makes, as a constant expression over
    the module's parameters, read from its header (`g = A; g < B; g = g +
    S`, `g++`, counting down too); None for a loop not over a genvar (a
    loop inside an always block) or not of that shape."""
    parts = []
    depth, start = 0, 0
    for k, ch in enumerate(header + ";"):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth == 0:
            parts.append(header[start:k].strip())
            start = k + 1
    if len(parts) != 3:
        return None
    init = re.fullmatch(r"(genvar\s+)?([A-Za-z_]\w*)\s*=\s*(.+)", parts[0],
                        re.S)
    if not init or not (init.group(1) or init.group(2) in genvars):
        return None
    v = re.escape(init.group(2))
    cond = re.fullmatch(rf"{v}\s*(<=|<|>=|>|!=)\s*(.+)", parts[1], re.S)
    step = (re.fullmatch(rf"{v}\s*=\s*{v}\s*([+-])\s*(.+)", parts[2], re.S)
            or re.fullmatch(rf"{v}\s*([+-])=\s*(.+)", parts[2], re.S))
    if step:
        sign, size = step.group(1), step.group(2)
    elif re.fullmatch(rf"{v}\s*\+\+|\+\+\s*{v}", parts[2]):
        sign, size = "+", "1"
    elif re.fullmatch(rf"{v}\s*--|--\s*{v}", parts[2]):
        sign, size = "-", "1"
    else:
        return None
    if not cond:
        return None
    op, a, b, s = cond.group(1), f"({init.group(3)})", f"({cond.group(2)})", \
        f"({size})"
    return {("+", "<"): f"({b}-{a}+{s}-1)/{s}",
            ("+", "!="): f"({b}-{a}+{s}-1)/{s}",
            ("+", "<="): f"({b}-{a}+{s})/{s}",
            ("-", ">"): f"({a}-{b}+{s}-1)/{s}",
            ("-", "!="): f"({a}-{b}+{s}-1)/{s}",
            ("-", ">="): f"({a}-{b}+{s})/{s}"}.get((sign, op))


def _generate_loops(text, genvars):
    """[(pass count expression, body start, body end)] for the generate
    loops written directly in `text`, not inside another one, with a
    begin ... end body."""
    out, i = [], 0
    while True:
        m = _FOR.search(text, i)
        if not m:
            return out
        close = _matching(text, m.end() - 1)
        if close is None:
            return out
        count = _loop_count(text[m.end():close], genvars)
        opened = re.match(r"\s*begin\b(?:\s*:\s*[A-Za-z_]\w*)?",
                          text[close + 1:])
        if count is None or not opened:
            i = m.end()               # look inside: it may hold a loop
            continue
        start, depth = close + 1 + opened.end(), 1
        for word in _BEGIN_END.finditer(text, start):
            depth += 1 if word.group(1) == "begin" else -1
            if depth == 0:
                out.append((count, start, word.start()))
                i = word.end()
                break
        else:
            return out


def generate_arrays(text, types=None, genvars=None):
    """[(name, dimensions)] for every register declared inside a generate
    loop of one module's text, a single register included: it is built
    once for each pass of each loop it sits in, so each loop adds a
    ("loop", pass count) dimension in front of its own."""
    if genvars is None:
        genvars = {g.strip() for m in _GENVAR.finditer(text)
                   for g in m.group(1).split(",")}
    out = []
    for count, start, end in _generate_loops(text, genvars):
        body = text[start:end]
        own = body
        for _, s, e in reversed(_generate_loops(body, genvars)):
            own = own[:s] + " " * (e - s) + own[e:]      # inner loops apart
        here = [("loop", count)]
        out += [(name, here + dims) for name, dims in
                _declarations(own, types, scalars=True)]
        out += [(name, here + dims) for name, dims in
                generate_arrays(body, types, genvars)]
    return out


# --- a size set by a `define (23 Oct 2026, review finding) -----------------
_MACRO = re.compile(r"`define[ \t]+([A-Za-z_]\w*)[ \t]+([^\n]*)")


def _expand_macros(text):
    """`text` with each use of a `define that takes no arguments replaced
    by its value in brackets, a few rounds deep, so a size set by one can
    be worked out."""
    macros = {m.group(1): m.group(2).strip() for m in _MACRO.finditer(text)}
    if not macros:
        return text
    use = re.compile(r"`(%s)\b" % "|".join(map(re.escape, macros)))
    for _ in range(5):
        expanded = use.sub(lambda m: f"({macros[m.group(1)]})", text)
        if expanded == text:
            break
        text = expanded
    return text


def _dimension_sizes(dims, env):
    """How many entries each dimension holds ([0:15], [15:0] and [16] hold
    16 each; a ("loop", count) dimension, its loop's passes), or None when
    a bound cannot be worked out."""
    sizes = []
    for dim in dims:
        if isinstance(dim, tuple):              # a generate loop's passes
            n = const_value(dim[1], env)
            if n is None:
                return None
            sizes.append(max(n, 0))
            continue
        ends = [const_value(e, env) for e in dim.split(":")]
        if None in ends or len(ends) > 2:
            return None
        if len(ends) == 2:
            sizes.append(abs(ends[0] - ends[1]) + 1)
        elif ends[0] > 0:
            sizes.append(ends[0])
        else:
            return None
    return sizes


def _entries(dims, env):
    """How many entries the dimensions hold together, or None when a bound
    cannot be worked out."""
    sizes = _dimension_sizes(dims, env)
    return None if sizes is None else math.prod(sizes)


_PARAMS_OPEN = re.compile(r"\s*#\s*\(")
_INSTANCE_NAME = re.compile(r"\s*[A-Za-z_]\w*\s*(?:\[[^\]]*\]\s*)?\(")


def _instances_with_parameters(text, names):
    """(module, text inside its #( ), or None) for every instance, in one
    module's text, of a module named in `names`."""
    out = []
    for mod in names:
        for m in re.finditer(r"\b%s\b" % re.escape(mod), text):
            i, params = m.end(), None
            opened = _PARAMS_OPEN.match(text, i)
            if opened:
                close = _matching(text, opened.end() - 1)
                if close is None:
                    continue
                params, i = text[opened.end():close], close + 1
            if _INSTANCE_NAME.match(text, i):
                out.append((mod, params))
    return out


def instance_parameters(params, module_text, env):
    """{parameter: value} an instance sets, by name (#(.DEPTH(8))) or in
    order (#(8)), worked out with the instantiating module's parameters
    (`env`); a value that cannot be worked out is None."""
    if not params:
        return {}
    order = [p for p, _, settable in module_parameters(module_text)
             if settable]
    items = [x for x in _split_outside_brackets(params) if x.strip()]
    out = {}
    for pos, item in enumerate(items):
        named = re.fullmatch(r"\s*\.\s*([A-Za-z_]\w*)\s*\((.*)\)\s*", item,
                             re.S)
        if named:
            out[named.group(1)] = const_value(named.group(2), env)
        elif pos < len(order):
            out[order[pos]] = const_value(item, env)
    return out


def array_shapes(verilog, top):
    """{(module, array): (entries, [entries of each dimension])} for the
    arrays of the design the checker builds from `top`: a parameter takes
    its default, or the value an instance gives it. An array with several
    dimensions counts every entry; the most any instance gives is kept. An
    array whose size cannot be worked out, or in a module `top` never
    uses, is left out. Since 23 Oct 2026 (review finding) the arrays
    include those of int, byte, shortint and longint and of the types the
    design declares, a register built by a generate loop counts once for
    each pass (its first sizes are the loops' passes), and a `define is
    read for its value."""
    masked = _expand_macros(_mask_comments(verilog))
    types = declared_types(masked)
    texts = {}
    for m in _MODULE_TEXT.finditer(masked):
        texts.setdefault(m.group(1), m.group(2))
    out, seen = {}, set()

    def visit(name, given, depth):
        key = (name, tuple(sorted(given.items())))
        if name not in texts or key in seen or depth > 16:
            return
        seen.add(key)
        text, env = texts[name], {}
        for pname, expr, settable in module_parameters(text):
            value = given[pname] if settable and pname in given \
                else const_value(expr, env)
            if value is None:
                env.pop(pname, None)
            else:
                env[pname] = value
        for array, dims in (module_arrays(text, types)
                            + generate_arrays(text, types)):
            sizes = _dimension_sizes(dims, env)
            if sizes is not None and math.prod(sizes) > \
                    out.get((name, array), (0, []))[0]:
                out[(name, array)] = (math.prod(sizes), sizes)
        for mod, params in _instances_with_parameters(text,
                                                      set(texts) - {name}):
            visit(mod, instance_parameters(params, texts[mod], env),
                  depth + 1)

    visit(top, {}, 0)
    return out


def storage_depths(verilog, top):
    """{(module, array): entries} for the arrays of the design the checker
    builds from `top` (array_shapes)."""
    return {key: n for key, (n, _) in array_shapes(verilog, top).items()}


def declared_arrays(verilog):
    """{(module, array)} for every array declared anywhere in `verilog`,
    whatever its size (module_arrays; registers built by a generate loop
    are not arrays here)."""
    masked = _mask_comments(verilog)
    types = declared_types(masked)
    return {(m.group(1), name)
            for m in _MODULE_TEXT.finditer(masked)
            for name, _ in module_arrays(m.group(2), types)}


def generated_registers(verilog):
    """{(module, register)} for every register declared inside a generate
    loop in `verilog` (generate_arrays)."""
    masked = _expand_macros(_mask_comments(verilog))
    types = declared_types(masked)
    return {(m.group(1), name)
            for m in _MODULE_TEXT.finditer(masked)
            for name, _ in generate_arrays(m.group(2), types)}


def new_array_limit(parent_verilog, parent_tops):
    """(most entries an array a step adds may hold, the parent's biggest
    array as (module, array, entries) or None when it has none): the
    parent's biggest array when it holds more than MAX_NEW_DEPTH entries,
    else MAX_NEW_DEPTH (22 Oct 2026, review finding F1). Sizes are the
    parent design's own, each part built from its own top
    (`parent_tops`: two for a combination), so a step cannot raise its
    limit by making an old array deeper. Registers built by a generate
    loop do not raise it (23 Oct 2026, Rishab chose): c0_102's 256 slots,
    the only such storage among the stored designs, would have let any
    new array hold 256 entries."""
    biggest = None
    generated = generated_registers(parent_verilog)
    for top in parent_tops:
        for (mod, name), n in storage_depths(parent_verilog, top).items():
            if (mod, name) in generated:
                continue
            if biggest is None or n > biggest[2]:
                biggest = (mod, name, n)
    limit = max(MAX_NEW_DEPTH, biggest[2]) if biggest else MAX_NEW_DEPTH
    return limit, biggest


def parent_sizes(parent_verilog, parent_tops):
    """{(module, array): entries} for the parent design's arrays, each
    part built from its own top, the most any part gives kept."""
    sizes = {}
    for top in parent_tops:
        for key, n in storage_depths(parent_verilog, top).items():
            sizes[key] = max(n, sizes.get(key, 0))
    return sizes


def too_deep(parent_verilog, child_verilog, top, parent_tops=None):
    """[(module, array, entries)] for every array of the design built from
    `top` that holds more entries than new_array_limit allows, or than the
    parent already had in it when that is more. `parent_tops` are the
    tops the parent is built from (`top` itself when not given: a patch
    keeps its parent's top). Until 23 Oct 2026 an array the parent
    declares was never judged, so a new copy of an old module with a
    deeper size, or an old instance given one, escaped (review finding).
    Now every array is judged by the size the step builds it at, and what
    the parent already had is never blamed: its own size of an array is
    allowed, which matters since registers built by a generate loop no
    longer raise the limit (23 Oct 2026)."""
    tops = parent_tops or [top]
    limit, _ = new_array_limit(parent_verilog, tops)
    before = parent_sizes(parent_verilog, tops)
    return sorted((mod, name, n) for (mod, name), n in
                  storage_depths(child_verilog, top).items()
                  if n > max(limit, before.get((mod, name), 0)))


def _array_named(mod, name, n, sizes, before=None):
    """`buf_q in module m (32 entries: 4 x 8, every dimension counted)`,
    the dimensions shown when there are several (22 Oct 2026, review
    finding F8); for an array the parent already had, how big it was
    there (`before`, 23 Oct 2026)."""
    if sizes and len(sizes) > 1:
        text = (f"{name} in module {mod} ({n} entries: "
                + " x ".join(str(s) for s in sizes)
                + ", every dimension counted)")
    else:
        text = f"{name} in module {mod} ({n} entries)"
    if before:
        text += f", which held {before} entries before this step"
    return text


def storage_too_deep(record, parent_verilog, child_verilog, top,
                     parent_tops=None):
    """Mark the record STORAGE_TOO_DEEP, with the fix, when the step added
    an array of more entries than new_array_limit allows; True when it
    did. The error names each array's entries and dimensions, and the
    parent's array whose size a new one may match (22 Oct 2026)."""
    deep = too_deep(parent_verilog, child_verilog, top, parent_tops)
    if not deep:
        return False
    limit, biggest = new_array_limit(parent_verilog, parent_tops or [top])
    shapes = array_shapes(child_verilog, top)
    sizes = {(mod, name): shapes.get((mod, name), (0, []))[1]
             for mod, name, _ in deep}
    before = parent_sizes(parent_verilog, parent_tops or [top])
    record["verdict"] = "STORAGE_TOO_DEEP"
    record["too_deep"] = [{"module": mod, "array": name, "entries": n}
                          for mod, name, n in deep]
    error = (f"the step adds storage of more than {limit} entries: "
             + ", ".join(_array_named(mod, name, n, sizes[(mod, name)],
                                      before.get((mod, name)))
                         for mod, name, n in deep)
             + ". The checker has to follow every entry, and storage this "
             "deep can make the proof run out of time.")
    if any((mod, name) in before for mod, name, _ in deep):
        error += (" An array the design already has may not be made deeper "
                  "than that either, or than it already was when it was "
                  "bigger, by a new instance of its module or a changed "
                  "one.")
    if {(mod, name) for mod, name, _ in deep} & \
            generated_registers(child_verilog):
        error += (" A register declared inside a generate loop is built "
                  "once for each pass of the loop, so each pass counts as "
                  "one entry.")
    if any(len(s) > 1 for s in sizes.values()):
        error += (" All the entries of one array count together, so the "
                  "sizes of its dimensions multiply: 4 channels of 8 "
                  "entries are 32. Use fewer channels or a smaller depth.")
    if limit > MAX_NEW_DEPTH:
        bmod, bname, bn = biggest
        error += (" A new array may hold as many entries as the design's "
                  f"own biggest array ({bname} in module {bmod}, {bn} "
                  "entries) only when it holds one entry for each of the "
                  "same things; any other new array keeps 4 to 8 entries.")
    error += (" Declare the depth as a parameter with a small value for the "
              "proof (#(parameter DEPTH = 4) or localparam DEPTH = 8), 4 to "
              f"8 entries and never more than {limit} in the whole array, "
              "and size the pointers and loops from it; the design works "
              "the same way at any depth.")
    record["error"] = error
    return True


# 26 Oct 2026, Rishab approved (change 5): nothing checked SYSTEM_PROMPT
# rule 3 before (no step of the five trials added an input), but the rule
# now has one exception, and the stream rules trust a new input only as a
# PEER lane's own offer. Ports of a new sub-module do not count: only the
# top module's port list is compared.
def new_input_ports(parent_verilog, child, top):
    """Input ports of `top` in the child that the parent's `top` lacks."""
    import gates                      # gates imports this module

    def inputs(text):
        return {p for p, d in gates.port_directions(text, top).items()
                if d == "input"}
    return sorted(inputs(child) - inputs(parent_verilog))


def new_input_error(move, parent_verilog, child, top):
    """Why the step's new input ports break rule 3, or None."""
    new = new_input_ports(parent_verilog, child, top)
    if not new or move == "PEER":
        return None
    return (f"the step adds the input port(s) {', '.join(new)} to module "
            f"{top}. Only a PEER step may add inputs, for its new lane's own "
            "input stream. Keep the module's inputs as they are; to make a "
            "sender wait, add a ready or busy OUTPUT instead.")


# 23 Oct 2026, review finding F4, the extender's side: the generator turns
# away a design whose property, guard or facts read the reset (run.py
# reset_gate), but a step was never asked. With an asynchronous reset in
# 1 design in 5 since 21 Oct, a STORAGE step that added `assert ((rst_n ||
# a == 3'd0) && d <= 4'd8)` to such a design was NECESSARY for our checker
# and REFUTED by EBMC, in the 30-step search and in the 1-step check with
# the facts. Each grader now asks reset_reads.py about what the step adds
# (a new or changed assertion and its guard, the facts, new antecedents),
# before any check runs, as a format failure the run retries with the
# reason. Only asynchronous resets count: the 30 older designs whose
# property's guard reads a synchronous reset (`if (pv && !rst)`) are not
# touched, and their checkers agree. Measured 23 Oct 2026: of the 167
# steps of the five trials that grew a design, it turns away none (one
# stored design has an asynchronous reset, and nothing reads it).
def step_reads_reset(record, child, clock, facts, antecedents, parents):
    """Mark the record READS_RESET, with the rule, when the grown design
    reads an asynchronous reset where the designs it grew from did not
    (`parents`: their rows); True when it does."""
    error = step_reset_error(
        child, clock or "clk", facts, antecedents,
        [(p.get("verilog"), p.get("clock") or "clk",
          p.get("invariants") or [], p.get("antecedents") or [])
         for p in parents])
    if error is None:
        return False
    record["verdict"] = "READS_RESET"
    record["error"] = error
    return True


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
            invariants="\n".join(shown_facts(parent)))
    tmpl = STRUCTURAL_TEMPLATE if ext_type == "structural" else SECOND_TEMPLATE
    fields = dict(
        verilog=parent["verilog"],
        property="\n".join(parent["property"]),
        invariants="\n".join(shown_facts(parent)),
        invariant_rules=INVARIANT_RULES,
        disposition_block=DISPOSITION_BLOCK,
        diff_format=DIFF_FORMAT,
        history=history_text(history, siblings, rejected))
    if ext_type == "structural":
        fields["modules"] = modules_text(parent["verilog"],
                                         parent.get("top_module", ""))
        # a second lane affects the old logic only through what it shares
        # (26 Oct 2026, Rishab approved, change 5)
        fields["effect_rule"] = (LEDGER_EFFECT_RULE if move == "LEDGER"
                                 else PEER_EFFECT_RULE if move == "PEER"
                                 else SIDEWAYS_EFFECT_RULE if move in BIG_MOVES
                                 else EFFECT_RULE)
        # 26 Oct 2026, Rishab approved (changes 1-3): big moves get the
        # waiting rule and declare their streams; the others read as before
        stream_move = move in STREAM_MOVES
        fields["waiting"] = WAITING_RULE if stream_move else ""
        fields["streams"] = streams_text(move) if stream_move else ""
        fields["streams_step"] = ("then streams (see STREAMS above), "
                                  if stream_move else "")
        fields["streams_reply"] = ' "streams": [...],' if stream_move else ""
        fields["move"] = f"{move}   {MOVES[move]}"
        if move == "STAGE" and k > 1:
            fields["move"] += STAGE_K_TEXT.format(k=k)
        if move in BIG_MOVES:
            n, lo, hi = size_target(parent["verilog"], move)
            fields["move"] += SIZE_TEXT.format(n=n, lo=lo, hi=hi,
                                               share=size_share(move, n))
        limit, biggest = new_array_limit(parent["verilog"],
                                         [parent.get("top_module", "")])
        if limit > MAX_NEW_DEPTH:
            fields["move"] += ARRAY_ALLOWANCE_TEXT.format(
                n=limit, array=biggest[1], module=biggest[0],
                cap=MAX_NEW_DEPTH)
    return tmpl.format(**fields)


def call_model(prompt_text, schema):
    text, usage, stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        user=prompt_text, schema=schema, effort=EFFORT)
    return (None if text is None else json.loads(text)), usage, stop


def grade_compose(pa, pb, out, record):
    """assemble (A verbatim + B verbatim + wrapper) -> storage depth ->
    property discipline -> coverage -> frozen enables and stream
    declaration -> yosys -> nothing-lost search -> oracle. A plain-wire
    wrapper is turned away by gates.IDLE_GLUE after the proof; DECORATIVE
    means the combined assertion needs no facts (26 Oct 2026)."""
    wrapper = strip_diff_decoration(out["wrapper"])
    top = out["top_module"]
    record["wrapper"] = wrapper
    # the stream the combination declares, for the nothing-lost search
    # (26 Oct 2026, Rishab approved, change 3)
    if "streams" in out:
        record["streams"] = out["streams"]
    out = _expanded(out, record)          # compact facts (21 Oct 2026)
    if out is None:
        return record
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
    # the glue's own storage small enough to prove (21 Oct 2026), or as
    # big as either part's biggest array (22 Oct 2026)
    if storage_too_deep(record, pa["verilog"] + "\n" + pb["verilog"],
                        child, top,
                        parent_tops=[pa["top_module"], pb["top_module"]]):
        return record

    parent_props = pa["property"] + pb["property"]
    new_assert, err = split_second_property_asserts(
        parent_props, extract_asserts(child))
    if err:
        record["verdict"] = "PROPERTY_CHANGED"
        record["error"] = err
        return record
    record["compose_property"] = new_assert
    # the wrapper's assertion, its guard, facts and antecedents leave an
    # asynchronous reset alone (23 Oct 2026, review finding F4)
    if step_reads_reset(record, child, pa.get("clock"), out["invariants"],
                        out.get("antecedents", []), [pa, pb]):
        return record

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

    # 26 Oct 2026, Rishab approved (change 4): a wrapper with no register,
    # or with no fact about its registers, is no longer sent back. That
    # rule never caught a plain wire (DECORATIVE fired for 0 of 36
    # combinations) and it pushed the model into a register that passes a
    # value on one cycle later (proven in 5 kept designs). The part between
    # the designs must do a real job instead (COMPOSE_TEMPLATE); gates.
    # IDLE_GLUE turns away a plain wire, the reviewer a delayed copy.
    record["glue_identifiers"] = sorted(wrapper_regs(wrapper))
    import nothing_lost                  # it imports this module
    # a design's enable driven by logic while its other inputs come from
    # outside loses those inputs (26 Oct 2026, change 2; t033, c2_004)
    err = nothing_lost.frozen_enable_error(wrapper, child, top,
                                           pa.get("clock") or "clk")
    if err:
        record["verdict"] = "PAUSES_WHOLE_DESIGN"
        record["error"] = err
        record["reason"] = err.split(".")[0]
        return record
    if "streams" in out:
        err = nothing_lost.declaration_error(
            out["streams"], "compose", child, top,
            pa.get("clock") or "clk", wrapper=wrapper)
        if err:
            record["verdict"] = "STREAM_INVALID"
            record["error"] = err
            return record

    pa_bits = state_bits(pa["verilog"], pa["top_module"])
    pb_bits = state_bits(pb["verilog"], pb["top_module"])
    child_bits = state_bits(child, top)
    record["parent_state_bits"] = [pa_bits, pb_bits]
    record["child_state_bits"] = child_bits
    if child_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = "child does not read in yosys"
        return record

    # the checker's own nothing-is-lost search: seconds, before the proof
    # takes minutes (26 Oct 2026, Rishab approved, change 3)
    if record.get("streams") and nothing_lost.check(
            record, child, top, pa.get("clock") or "clk", [pa, pb],
            wrapper=wrapper):
        return record

    payload = {"verilog": child, "top_module": top, "clock": pa["clock"],
               "antecedents": out["antecedents"],
               "sanity_covers": out["sanity_covers"],
               "invariants": out["invariants"]}
    result = necessity_check(grade_triple_generated, payload, record,
                             f"oracle grading composed {top}")
    if result.verdict.name == "DECORATIVE":
        # 26 Oct 2026, Rishab approved (change 4): the part between the
        # designs no longer has to hold registers, so the old "glue not
        # stateful enough" said the wrong thing
        record["error"] = ("the combined assertion proves without any "
                           "facts, so the combination teaches nothing: make "
                           "the assertion about state that crosses the "
                           "connection between the two designs")
    return record


def grade_replicate(parent, out, record, n):
    """assemble (parent verbatim + wrapper) -> storage depth -> property
    discipline -> coverage + aggregate checks -> yosys -> oracle."""
    wrapper = strip_diff_decoration(out["wrapper"])
    top = out["top_module"]
    record["wrapper"] = wrapper
    out = _expanded(out, record)          # compact facts (21 Oct 2026)
    if out is None:
        return record
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
    # the pool's own storage small enough to prove (21 Oct 2026), or as
    # big as the copied design's biggest array (22 Oct 2026)
    if storage_too_deep(record, parent["verilog"], child, top,
                        parent_tops=[parent["top_module"]]):
        return record

    new_assert, err = split_second_property_asserts(
        parent["property"], extract_asserts(child))
    if err:
        record["verdict"] = "PROPERTY_CHANGED"
        record["error"] = err
        return record
    record["pool_property"] = new_assert
    # the wrapper's assertion, its guard, facts and antecedents leave an
    # asynchronous reset alone (23 Oct 2026, review finding F4)
    if step_reads_reset(record, child, parent.get("clock"),
                        out["invariants"], out.get("antecedents", []),
                        [parent]):
        return record

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
    necessity_check(grade_triple_generated, payload, record,
                    f"oracle grading {top} (N={n})")
    return record


def _expanded(out, record):
    """`out` with its compact facts expanded, the record holding both
    forms; None, with the record marked FACT_FORMAT, when one is
    malformed or, once expanded, a fact uses a system function (21 Oct
    2026)."""
    compact = list(out.get("invariants", []))
    expanded, err = expand_facts(compact)
    if err:
        record["invariants"] = compact
        record["verdict"] = "FACT_FORMAT"
        record["error"] = err
        return None
    record["invariants"] = expanded
    if expanded != compact:
        record["invariants_compact"] = compact
    err = system_function_error(expanded)
    if err:
        record["verdict"] = "FACT_FORMAT"
        record["error"] = err
        return None
    return dict(out, invariants=expanded)


def grade_step4(parent, ext_type, out, record):
    """apply -> storage depth -> new inputs -> stream declaration ->
    property discipline -> dispositions -> yosys -> nothing-lost search
    (26 Oct 2026) -> oracle (-> P2-unaided for second property). Fills
    record in place."""
    record["patch"] = out["patch"]
    record["dispositions"] = out["dispositions"]
    # the streams a big step declares, for the nothing-lost search (26 Oct
    # 2026, Rishab approved, change 3)
    if "streams" in out:
        record["streams"] = out["streams"]
    compact = list(out["invariants"])
    out = _expanded(out, record)
    if out is None:
        return record

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
    # storage small enough to prove, before any check runs (21 Oct 2026)
    if storage_too_deep(record, parent["verilog"], child,
                        parent.get("top_module", "")):
        return record
    # only a PEER step may add an input port, for its new lane (26 Oct
    # 2026, Rishab approved, change 5); second-property steps too, since
    # SYSTEM_PROMPT rule 3 covers them
    record["new_inputs"] = new_input_ports(parent["verilog"], child,
                                           parent.get("top_module", ""))
    err = new_input_error(record.get("move"), parent["verilog"], child,
                          parent.get("top_module", ""))
    if err:
        record["verdict"] = "NEW_INPUT"
        record["error"] = err
        return record
    # the streams the step declares, before any tool runs (26 Oct 2026,
    # Rishab approved, change 3); an answer written before the rule has none
    if "streams" in out:
        import nothing_lost              # it imports this module
        err = nothing_lost.declaration_error(
            out["streams"], record.get("move"), child,
            parent.get("top_module", ""), parent.get("clock") or "clk",
            parent=parent)
        if err:
            record["verdict"] = "STREAM_INVALID"
            record["error"] = err
            return record

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

    # a new or changed assertion, its guard and the facts leave an
    # asynchronous reset alone (23 Oct 2026, review finding F4); the
    # antecedents are the parent's, judged only if the step made a signal
    # they read an asynchronous reset
    if step_reads_reset(record, child, parent.get("clock"),
                        out["invariants"], parent.get("antecedents") or [],
                        [parent]):
        return record

    # what was kept or replaced is read in the facts as written: the
    # parent's compact facts against this answer's (21 Oct 2026)
    err = check_dispositions(shown_facts(parent), compact,
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

    # the checker's own nothing-is-lost search: seconds, before the proofs
    # take minutes (26 Oct 2026, Rishab approved, change 3)
    if record.get("streams"):
        import nothing_lost
        if nothing_lost.check(record, child, parent["top_module"],
                              parent.get("clock") or "clk", [parent]):
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
    necessity_check(grade_triple_generated, payload, record,
                    f"oracle grading extended {parent['top_module']}")
    return record


def report(record, ext_type):
    if ext_type in ("replicate", "compose"):
        print(f"\nwrapper:\n{record.get('wrapper', '<none>')}\n")
        if "aggregate_clause" in record:
            print("aggregate:", record["aggregate_clause"])
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
    # what the checker's own nothing-is-lost search found (26 Oct 2026,
    # Rishab approved, change 3); the glue clause it replaced is gone
    if "nothing_lost" in record:
        nl = record["nothing_lost"]
        print(f"nothing lost: {nl['result']} (depth {nl['depth']}, "
              f"{nl['seconds']} s)")
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

    # a big move's schema asks for its streams (26 Oct 2026, Rishab
    # approved, change 3)
    schema = (structural_schema(args.move) if args.type == "structural"
              else {"second": SECOND_SCHEMA, "replicate": REPLICATE_SCHEMA,
                    "compose": COMPOSE_SCHEMA}[args.type])
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
