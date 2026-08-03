# graded_oracle

Stage 3 of the Hardware Formal Disco project: a deterministic graded
oracle that wraps SymbiYosys (sby) and sorts a (RTL, property,
invariants) triple into one of seven categorical tiers —
ERROR, TIMEOUT, FALSE, VACUOUS, NOT_INDUCTIVE, BOUNDED (reserved), PROVEN —
with complete evidence (return codes, depth, engine, timings, log
excerpts, counterexample/CTI traces) attached to every verdict.

Design spec: `design_spec.md`.

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
print(result.tier.name, "—", result.reason)
```

The necessity criterion — is the strengthening invariant load-bearing,
or decorative? Graded by two calls (with the invariants injected, then
with them stripped); only PROVEN-with + NOT_INDUCTIVE-without earns
NECESSARY, the Stage-4 acceptance criterion:

```python
from oracle import grade_triple, PropertyInfo

prop = PropertyInfo(top_module="kitest", clock="i_clk",
                    invariants=["sa == sb"])
r = grade_triple("kitest_weak.sv", prop)
print(r.verdict.name, "—", r.reason)   # NECESSARY — ...
```

The LLM-facing boundary — workers emit one JSON object (inline verilog +
structured metadata); malformed output grades ERROR, never raises:

```python
from oracle import grade_generated

r = grade_generated(llm_response_text)   # or grade_triple_generated
print(r.tier.name, "—", r.reason)
```

## Demo / tests

- `venv/bin/python demo.py` — every exemplar triple → its tier.
- `venv/bin/python -m pytest tests/ -v` — unit tests run anywhere;
  sby-backed tests skip unless hwtools is active.
