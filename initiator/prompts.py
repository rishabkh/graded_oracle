"""Initiator prompt: fixed system prompt + per-call user template.

The system prompt carries the contract and the criterion (internal spec 1.1).
One deviation from the doc: cti_state is an array of {signal, value}
pairs rather than a free-form dict, because strict structured-output
schemas require fixed keys on every object. Same information.
"""
import re

# The size-parameter rule at the end of the Verilog constraints (21 Oct 2026,
# Rishab approved): 87% of 560 real open-source modules declare parameters
# against about 40% of our designs (measured 15 Oct 2026). Defaults of 4 to
# 8 entries keep the proof fast; a size seed, when one is drawn, sets the
# default instead, so the real-hardware sizes of scales_real_hardware.txt
# still arrive.
# Two rules after it (22 Oct 2026, review findings):
# - the property, its guard and the facts never read a reset. With the
#   asynchronous-reset trait (1 in 5 designs), a property reading the reset
#   was NECESSARY for our checker and false for EBMC 6.0, which scores the
#   trained model: in a cycle where the reset is held, our checker shows
#   each register at its reset value and EBMC the value it held. run.py's
#   reset_gate turns such designs away. Since 23 Oct 2026 (Rishab chose)
#   the rule names asynchronous resets only: both checkers read a
#   synchronous reset the same way, and 30 of the 917 stored designs guard
#   their property with one (`if (pv && !rst)`).
# - each fact is a plain same-cycle expression, no system function: the
#   prompt only said "use it in the property, not in the invariants" in
#   the style seed, and c0_144 still came out with
#   `occ == $countones(valid)`, which EBMC cannot read. The wording follows
#   the extender's (extender/extend.py INVARIANT_RULES); run.py's
#   facts_gate turns such designs away.
# The answer form since 26 Oct 2026 (Rishab approved): the design first,
# then the fake state and a short explanation of the finished answer. Until
# then the output format said "the reasoning fields come FIRST. Work out the
# state S that breaks induction BEFORE writing the design", and the first
# field was `cti_reasoning`. Opus 5.5 stopped 7 of the 41 calls of the 25 Oct
# run, all with stop category "reasoning_extraction" (as were 23 of the 26
# stops of 9 Oct that recorded one). Anthropic's refusals page says that
# category means a request that "asks the model to reproduce its internal
# reasoning in the response text", names a scratchpad filled in before the
# answer and a `reasoning` field in JSON output as its usual causes, allows
# "a short explanation", and says to change the prompt rather than retry.
# Opus 5.5 thinks before it answers anyway.
SYSTEM_PROMPT = """\
You write SystemVerilog modules for a formal verification dataset.

Each module is a "planted triple": a design containing an internal structural
relation R that holds by construction, plus a property P that is true of the design
but CANNOT be proven by k-induction unless the prover is also given R.

## The criterion

Three conditions, all required.

(1) P is true of the design.

(2) P is NOT inductive on its own. Concretely, there exists a state S such that:
      - S satisfies P
      - S violates R  (so S is unreachable in real operation, but a solver
        performing induction may start there)
      - S transitions in ONE step to a state that VIOLATES P

    S must also be a state the design can IDLE IN — one it can remain in for
    arbitrarily many cycles by holding the enable low. This matters: a one-step
    CTI only shows P is not 1-inductive, but the prover uses a 20-state window.
    If the design can stall in S, the solver builds a window of 20 copies of S and
    then takes the violating transition, so the property fails induction at ANY
    depth rather than only at depth 1. Without a stall, P may survive 20-induction
    and the triple is rejected.

    Condition (2) is the whole task. A property that is merely "weaker than R" is
    not enough. If every state satisfying P steps to another state satisfying P,
    then P is inductive by itself and the module is worthless for this dataset.

    Counterexample to avoid: if `count` is a 3-bit register, `count <= 7` is a
    consequence of any relation involving count, but it is trivially inductive
    because the register cannot hold anything larger. Rejected.

(3) P AND R TOGETHER ARE INDUCTIVE. R must be strong enough to exclude S and every
    state like it. It is not enough that R is true and P is weak — R must actually
    close the proof.

    R may be a conjunction of several clauses. The individual clauses do NOT each
    need to be inductive on their own; the requirement is that the conjunction,
    together with P, closes induction. Multi-clause relations are welcome and often
    necessary.

## Hard constraints on the Verilog

- These assertion forms are supported and all of them are welcome:
      always @(posedge clk) assert (expr);
      always @(*)           assert (expr);
      assert property (expr);                    // module level, NO clock inside
      always @(posedge clk) assert property (expr);
  `$past`, `$rose`, `$fell` and `$stable` work inside a clocked always block.
  Gate any `$past` check with a past-valid register so cycle zero is not judged:
      reg pv = 0;
      always @(posedge clk) pv <= 1;
      always @(posedge clk) if (pv) assert (out == $past(in));
- Do NOT put a clock inside the property: `assert property (@(posedge clk) ...)`,
  `disable iff`, `|->`, `|=>`, `##N`, `property ... endproperty`, `s_eventually`,
  sequences and `bind` are REJECTED by the toolchain — the module fails to parse
  and is never graded. Do not nest these functions either: `$past($rose(x))`
  fails to elaborate.
- Give every state register an `initial` value, so counterexamples are genuine
  from-reset traces rather than artifacts of an unconstrained start state.
- The design MUST have an enable or idle condition — an input that, when low, leaves
  all state unchanged (e.g. `i_ce`, `step`, or push and pop both low). Condition (2)
  depends on it: the state S must be one the design can sit in indefinitely.
- One property per module, unless the property style below asks for two.
- Never write `assume`. Assumptions are not checked and would let a false premise
  carry the proof.
- Never write `->` as logical implication in an expression — the toolchain rejects
  it. Write `!a || b` instead.
- If a register's declared width admits values it can never reach (e.g. a counter
  that saturates at 8 in a 4-bit reg), include an invariant bounding it — induction
  otherwise starts from an impossible value and fails on a phantom CTI.
- Declare the design's sizes as parameters, as real hardware does: a depth, a
  number of entries or a number of channels is a `parameter` with a default
  (e.g. `parameter DEPTH = 8`), and the widths and constants that depend on it
  are written in terms of it. Give such a count a default of 4 to 8, never more
  than 16, so the proof stays fast; when the question gives a size seed, the
  seed's size is the default instead.
- The property, its guard (the antecedents) and the invariants never read an
  asynchronous reset (an input in an always block's sensitivity list, like
  `negedge rst_n`) or a signal that carries one (like `wire in_reset = !rst_n`):
  they describe the design in normal operation, not what holds while reset is
  held. Checkers disagree about what an asynchronous reset does within a cycle,
  so a design whose property, guard or invariants name one is rejected. A
  synchronous reset, tested only inside `always @(posedge clk)`, may guard the
  property. When a `$past` check must skip the cycle after an asynchronous reset,
  let the reset clear the past-valid register instead.

## Output format

Your output is emitted through the API's structured-output mode — the schema is
enforced. Decide on the state S that breaks induction before you write the
design, then write a design in which that state is reachable-under-induction
and the transition is real. The answer gives the design first, then the state
S and a short explanation.

Field notes:

- `invariants` is R. Supplied SEPARATELY from the Verilog — it must NOT appear as
  an assertion in the source. One string per clause. Each clause is a same-cycle
  boolean expression over the design's signals, with no system function: no
  `$past`, `$stable`, `$rose`, `$fell`, `$countones`, `$onehot` or `$onehot0` in an
  invariant, or it is rejected. Write a count of set bits as a sum
  (v[0] + v[1] + v[2]), and a relation across cycles between the registers that
  carry the value from one cycle to the next. The property may still use `$past`.

- `verilog` is the full module source with the property assertion inline.

- `top_module` must match `module <name>` in the source. `clock` is the clock
  signal name.

- `antecedents` is the guard of P. If P is `if (count == 3'd0) assert (wptr == rptr);`
  the antecedent is `"count == 3'd0"`. If P is unconditional, use [].
  Prefer writing P with a guard, because real hardware properties are overwhelmingly
  conditional — an assertion about what must hold *when something happens*.

- `sanity_covers` must be genuinely reachable, e.g. `"count == 3'd2"` for a counter
  that gets there. Only checked when the property has an antecedent.

- `cti_state`: the state S from condition (2), as a list of {signal, value}
  pairs with concrete values. Every signal needed to evaluate both P and R must
  appear. Vague answers are rejected: "pointers disagree" is not acceptable;
  [{"signal": "count", "value": "3'd1"}, {"signal": "wptr", "value": "2'd2"},
  {"signal": "rptr", "value": "2'd0"}] is.

  Self-check before you answer: evaluate R at cti_state. It MUST come out false.
  If your named state does not violate your own invariant, you have contradicted
  yourself — fix one or the other.

- `cti_explanation`: a short explanation of the finished answer, in one or two
  sentences: why this state satisfies P, why it violates R, and what its
  successor does that breaks P.
"""

USER_TEMPLATE = """\
## Theme seed

The following README is from an unrelated open-source project. Use it only as loose
inspiration for what the module is *about* — its domain, its vocabulary, what it
models. Do not implement the project. Do not mention it in the output.

{readme}

## Structure seed

Build the module around this hardware construct:

{construct}

The construct decides the state registers and the relation R that holds by
construction; the property and invariants FOLLOW from the structure. Do not
pick an invariant first and decorate a module around it. You may scale the
construct, specialise it, or embed it in a surrounding mechanism, but
the named structure must genuinely be there.

## Property pattern seed

The property must be an instance of this pattern (Dwyer et al., ICSE 1999):

{pattern}

The pattern decides WHAT the property says. Pick the design's state so that
this pattern is natural for it, rather than bolting the pattern onto a design
that does not support it.

## Property scope seed

The property is required in this scope (same paper):

{scope}

A scope is not a guard. `if (g) assert (e)` is the same as `assert (!g || e)`
and is evaluated fresh every cycle; a scope LATCHES, so the arming register
records that something has already happened and keeps that memory. If the
scope is anything other than global, the arming register must appear in
`antecedents`, and `sanity_covers` must show the armed state is reachable.
A property that can never be armed is vacuously true and is rejected.

## Property style seed

Write the property in this style:

{style}

The style decides the SHAPE of the assertion, not what it is about. The
criterion is unchanged: the property must still be true, still fail induction
on its own, and still close with R. If the style names a system function, use
it in the property, not in the invariants.

## Worked example

{exemplar}

## Your task

Write one new planted triple. It must be structurally different from the worked
example — a different kind of design, a different sort of relation, a different
property shape — and it must actually contain the seeded construct. Fill in
every field of the schema.
"""


# Added before the pattern seed only when a size seed is drawn (the
# catalog run, task 5, Oct 2026); without one the message is unchanged.
SIZE_SECTION = """## Size seed

{scale}

Declare the design's registers and storage at this size, and size every
constant to match.

"""


# Added before the pattern seed, after any size seed, when a design trait is
# drawn (21 Oct 2026, Rishab approved: real hardware uses handshakes,
# asynchronous resets, named states and registers loaded from inputs more
# than our designs did; design_traits.txt has the numbers). Without one
# the message is unchanged, so old rows rebuilt from their seeds stay as
# they were.
TRAIT_SECTION = """## Design trait seed

Real hardware is often written this way, and this design must be too:

{trait}

The trait shapes how the design is written, not what is asked of it: every
rule of the task still holds, including an initial value on every state
register, an idle condition that leaves all state unchanged, and no assume.

"""

# What the two-step questions say when no size seed or no trait was drawn.
# The size text was "choose ordinary sizes" until 21 Oct 2026; the
# size-parameter rule of SYSTEM_PROMPT makes it precise.
NO_SIZE_SEED = ("(no size seed: choose ordinary word widths, and give each "
                "depth, number of entries or number of channels a parameter "
                "default of 4 to 8, never more than 16)")
NO_TRAIT_SEED = "(no design trait seed: plain synchronous style)"


_CAP = re.compile(r"Keep the module under\s+(\d+)\s+lines")

# The rules added since the line cap went (22 Oct 2026, review finding: the
# stamp only read the cap, so the prompt with the size-parameter rule still
# said "nocap", like the 9 Oct prompt). Each rule is found by a phrase only
# it uses, so a stamp still cannot claim a rule its prompt lacks. The reset
# rule's first wording (22 Oct 2026, every reset, stamped "no-reset-reads")
# wrote no rows before it narrowed to asynchronous resets on 23 Oct 2026,
# so only the narrow one has a stamp. The answer form of 26 Oct 2026 (the
# explanation after the design) is the last.
STAMPED_RULES = (
    ("size-params", "Declare the design's sizes as parameters"),
    ("plain-facts", "boolean expression over the design's signals, with no "
                    "system function"),
    ("no-async-reset-reads", "never read an asynchronous reset"),
    ("explanation-after-design", "a short explanation of the finished answer"),
)


def prompt_version(system_prompt=None):
    """A short stamp naming the rules this prompt imposes, read off the
    prompt itself so it cannot drift from the rules it names: the line cap
    ("200cap", "800cap" or "nocap"), then each rule of STAMPED_RULES the
    prompt holds, joined by "+" (since 22 Oct 2026). Rows carry it because
    each rule splits the corpus into two populations, and a later score
    change is unreadable unless each row says which prompt wrote it. A
    prompt without those rules gets the stamp it always had. The design
    trait is not here: every row since 21 Oct 2026 names its own."""
    text = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    m = _CAP.search(text)
    flat = " ".join(text.split())
    return "+".join([f"{m.group(1)}cap" if m else "nocap"]
                    + [name for name, phrase in STAMPED_RULES
                       if phrase in flat])


# --- two-step and self-repairing generation (8 Oct 2026) ------------------
# Formal Disco's idea -> implement split and its self-repairing generator
# (formal-disco/language/*/prompt.py idea, implement, repair_full), adapted
# to the planted-triple contract above. Their idea step sees only a README;
# ours also gets the structural seeds (kind, size, design trait since 21 Oct
# 2026, pattern, scope, style), which are where our variety comes from.

IDEA_SYSTEM = """\
You propose precise ideas for SystemVerilog modules for a formal
verification dataset. Each idea must be implementable as a "planted triple":
a design with an internal relation R that holds by construction, plus a
property P that is true of the design but cannot be proven by k-induction
unless the prover is also given R. You will receive a GitHub README for the
theme and seeds for the hardware kind, its size, a design trait (a way real
hardware is written), and the property's pattern, scope and style. Output
exactly one concise idea and specification. The README's project is most
likely unrelated to hardware verification, so adapt or reinterpret its theme
freely."""

IDEA_TEMPLATE = """\
## Theme (loose inspiration only; do not implement the project)

Repository: {repo}

{readme}

## Hardware kind

{construct}

## Size

{scale}

## Design trait

{trait}

## Property pattern

{pattern}

## Property scope

{scope}

## Property style

{style}

## Task

- Propose one idea for a module built around the hardware kind, themed by
  the README, at the given size, written with the given design trait.
- Name its size parameters with their defaults, its state registers with
  their widths, and the enable or idle input that holds all state when low.
- State the relation R that holds by construction.
- State the property P (following the pattern, scope and style) and a state
  that satisfies P, violates R, can be held while idle, and steps to a
  violation of P.
- Keep it self-contained. Output only the idea and specification, in
  under 400 words.
"""

IMPLEMENT_TEMPLATE = """\
## Design idea

{idea}

## Size

{scale}

Declare the design's registers and storage at this size, and size every
constant to match.

## Design trait

{trait}

## Worked example

{exemplar}

## Your task

Write one planted triple that implements this idea. It must be structurally
different from the worked example. Fill in every field of the schema.
"""

REPAIR_TEMPLATE = """\
Your planted triple was checked and rejected.

## Your triple

{triple}

## What the checker found

{notes}

## Your task

Return the complete corrected triple, with every field of the schema. Keep
the same hardware idea; change whatever the checker's findings require: the
design, the property, the helper facts (invariants) or the other fields.
"""
