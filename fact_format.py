"""Can the benchmark's checker read a helper fact? (22 Oct 2026)

The open model is scored on the Technion benchmark by EBMC 6.0, which reads
a helper fact only as one plain boolean expression about a single cycle.
Measured 21 Oct 2026 on this laptop's EBMC: every system function in a fact
($past, $stable, $rose, $countones, $onehot, $onehot0, $signed, $clog2, ...)
gives "unknown identifier". Two more forms are ours, not Verilog: the
compact `forall k in 0..7: ...` form, which the extender expands before our
own checker sees it, and a dotted name (`u_fifo.count`), which our checker
silently turns into a dangling wire.

A training answer that holds any of these teaches the model to write facts
the benchmark rejects, so every place that accepts or trains on facts asks
this module (review of 21 Oct 2026: they reached training files through the
generator, the fixer and the training builder, not only the extender).

  from fact_format import unreadable_facts
  for fact, problems in unreadable_facts(answer["invariants"]): ...
"""
import re

# `$name`, not a `$` inside a name: `a$b` is one identifier in Verilog
SYSTEM_FUNCTION = re.compile(r"(?<![\w$])\$[A-Za-z_][\w$]*")
COMPACT = re.compile(r"^\s*forall\b")
# the same pattern the generator contract uses for hierarchical references
DOTTED = re.compile(r"[A-Za-z0-9_\]]\s*\.\s*[A-Za-z_]")


def fact_problems(fact):
    """Why EBMC could not read `fact`, as short plain phrases; an empty
    list means it is a plain fact."""
    problems = []
    names = sorted({m.group(0) for m in SYSTEM_FUNCTION.finditer(fact)})
    if names:
        problems.append("system function " + ", ".join(names))
    if COMPACT.match(fact):
        problems.append("compact forall form")
    if DOTTED.search(fact):
        problems.append("dotted name")
    return problems


def unreadable_facts(facts):
    """[(fact, problems)] for every fact EBMC could not read, in order."""
    out = []
    for fact in facts:
        problems = fact_problems(fact)
        if problems:
            out.append((fact, problems))
    return out


def all_plain(facts):
    """True when EBMC can read every fact."""
    return not unreadable_facts(facts)
