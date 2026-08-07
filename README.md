# graded_oracle

A deterministic oracle for hardware formal verification. It wraps
SymbiYosys (sby) and sorts an (RTL, property, invariants) triple into
one of seven tiers: ERROR, TIMEOUT, FALSE, VACUOUS,
NOT_INDUCTIVE, BOUNDED (reserved), PROVEN. Every verdict comes with
full evidence: return codes, depth, engine, timings, log excerpts, and
counterexample/CTI traces. Traces are also rendered as plain text
(`trace_text`) so they can be used in an LLM prompt.

Depth guesses never decide verdicts. VACUOUS requires an unbounded
`abc pdr` proof that the antecedent is unreachable. An antecedent that
is merely unreached at depth D gets a PDR reachability check, which can
rescue it with a deeper witness. Prove-mode rc=4 also gets a PDR second
opinion, so a property that is false beyond the base case grades FALSE
instead of being sent to the Fixer as NOT_INDUCTIVE.

## Prerequisites

- OSS CAD Suite on PATH: `hwtools` (sources oss-cad-suite/environment)
- Python 3.10+; `python3 -m venv venv && venv/bin/pip install pytest`

## Usage

```python
from oracle import grade, PropertyInfo

prop = PropertyInfo(top_module="counter",
                    antecedents=["past_rst"],
                    sanity_covers=["count == 4'd15"])
result = grade("design.sv", prop, depth=20, timeout_s=300)
print(result.tier.name, "-", result.reason)
```

The necessity check asks whether the strengthening invariant actually
does any work. It grades twice: once with the invariants injected, once
with them stripped. Only PROVEN with plus NOT_INDUCTIVE without earns
NECESSARY, which is the next stage's acceptance bar:

```python
from oracle import grade_triple, PropertyInfo

prop = PropertyInfo(top_module="kitest", clock="i_clk",
                    invariants=["sa == sb"])
r = grade_triple("kitest_weak.sv", prop)
print(r.verdict.name, "-", r.reason)   # NECESSARY
```

The LLM-facing entry point takes the raw JSON a worker emitted (inline
verilog plus structured metadata). Malformed output grades ERROR and
never raises:

```python
from oracle import grade_generated

r = grade_generated(llm_response_text)   # or grade_triple_generated
print(r.tier.name, "-", r.reason)
```

When a run fails, the evidence says what to repair. Every rc=2 carries
the source lines of the failing assertions (`failed_assert_lines`), and
`grade_triple` uses them to tell a falsified injected invariant (fix or
drop the invariant) apart from a false property (rewrite the property,
leave the invariants alone). The CTI for NOT_INDUCTIVE arrives as text.
For kitest that is `sa = 32'h00000000` and `sb = 32'h40000000` at the
induction start: the unreachable state the next invariant must rule out.

## Demo / tests

- `venv/bin/python demo.py` runs every exemplar triple and prints its tier.
- `venv/bin/python -m pytest tests/ -v` runs the test suite. Unit tests
  run anywhere; sby-backed tests skip unless hwtools is active.
