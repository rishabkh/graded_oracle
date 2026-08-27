"""Initiator prompt: fixed system prompt + per-call user template.

The system prompt carries the contract and the criterion (internal spec 1.1).
One deviation from the doc: cti_state is an array of {signal, value}
pairs rather than a free-form dict, because strict structured-output
schemas require fixed keys on every object. Same information.
"""

SYSTEM_PROMPT = """\
You write small SystemVerilog modules for a formal verification dataset.

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

- IMMEDIATE assertions only, inside `always` blocks:
      always @(posedge clk) assert (expr);
      always @(*)           assert (expr);
- Do NOT use concurrent SVA. `assert property (@(posedge clk) ...)`, `|->`, `|=>`,
  `##N`, `s_eventually`, sequences, and `bind` are REJECTED by the toolchain — the
  module fails to parse and is never graded.
- For temporal behaviour, hand-roll a register:
      reg past_rst = 0;
      always @(posedge clk) past_rst <= rst;
      always @(posedge clk) if (past_rst) assert (count == 0);
- Give every state register an `initial` value, so counterexamples are genuine
  from-reset traces rather than artifacts of an unconstrained start state.
- The design MUST have an enable or idle condition — an input that, when low, leaves
  all state unchanged (e.g. `i_ce`, `step`, or push and pop both low). Condition (2)
  depends on it: the state S must be one the design can sit in indefinitely.
- Exactly one property per module.
- Never write `assume`. Assumptions are not checked and would let a false premise
  carry the proof.
- Never write `->` as logical implication in an expression — the toolchain rejects
  it. Write `!a || b` instead.
- If a register's declared width admits values it can never reach (e.g. a counter
  that saturates at 8 in a 4-bit reg), include an invariant bounding it — induction
  otherwise starts from an impossible value and fails on a phantom CTI.
- Keep the module under 200 lines.

## Output format

Your output is emitted through the API's structured-output mode — the schema is
enforced. Field order is deliberate: the reasoning fields come FIRST. Work out the
state S that breaks induction BEFORE writing the design, then write a design in
which that state is reachable-under-induction and the transition is real.

Field notes:

- `cti_reasoning`: one or two sentences: why this state satisfies P, why it
  violates R, and what its successor does that breaks P.

- `cti_state`: the state S from condition (2), as a list of {signal, value}
  pairs with concrete values. Every signal needed to evaluate both P and R must
  appear. Vague answers are rejected: "pointers disagree" is not acceptable;
  [{"signal": "count", "value": "3'd1"}, {"signal": "wptr", "value": "2'd2"},
  {"signal": "rptr", "value": "2'd0"}] is.

  Self-check before you answer: evaluate R at cti_state. It MUST come out false.
  If your named state does not violate your own invariant, you have contradicted
  yourself — fix one or the other.

- `invariants` is R. Supplied SEPARATELY from the Verilog — it must NOT appear as
  an assertion in the source. One string per clause.

- `verilog` is the full module source with the property assertion inline.

- `top_module` must match `module <name>` in the source. `clock` is the clock
  signal name.

- `antecedents` is the guard of P. If P is `if (count == 3'd0) assert (wptr == rptr);`
  the antecedent is `"count == 3'd0"`. If P is unconditional, use [].
  Prefer writing P with a guard, because real hardware properties are overwhelmingly
  conditional — an assertion about what must hold *when something happens*.

- `sanity_covers` must be genuinely reachable, e.g. `"count == 3'd2"` for a counter
  that gets there. Only checked when the property has an antecedent.
"""

USER_TEMPLATE = """\
## Theme seed

The following README is from an unrelated open-source project. Use it only as loose
inspiration for what the module is *about* — its domain, its vocabulary, what it
models. Do not implement the project. Do not mention it in the output.

{readme}

## Property-form seed

Property shapes that appear in real hardware assertions. Use them as a guide to how
properties get shaped — guard conditions, signal relationships, the kinds of facts
that get asserted. Do not copy them verbatim.

{shape_1}
{shape_2}

## Worked example

{exemplar}

## Your task

Write one new planted triple. It must be structurally different from the worked
example — a different kind of design, a different sort of relation, a different
property shape. Fill in every field of the schema.
"""
