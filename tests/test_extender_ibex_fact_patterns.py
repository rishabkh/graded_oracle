"""The fact guidance carries the Ibex core's buffer-proof patterns (26 Oct
2026, Gary's suggestion: use Ibex's buffer-ordering lemmas as templates).
Reads text only: no model, no checker."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import extend                                                   # noqa: E402


def squash(text):
    return " ".join(text.split())


def test_the_five_ibex_patterns_are_in_the_fact_guidance():
    text = squash(extend.INVARIANT_RULES)
    for phrase in ("the Ibex core's own buffer proofs",
                   "no busy entry is older than itself",
                   "the order is transitive",
                   "equals the count of requests still outstanding",
                   "only when every older busy entry has finished",
                   "the oldest busy entry's address is the address being "
                   "sent out",
                   "has at most one bit set, written as a sum"):
        assert phrase in text, phrase


def test_the_example_is_a_compact_fact_the_checker_expands():
    example = re.search(r"`(forall k in 0\.\.3, m in 0\.\.3 where k != m:"
                        r"[^`]*)`", extend.INVARIANT_RULES).group(1)
    facts, error = extend.expand_facts([example])
    assert error is None and len(facts) == 12      # 4 x 4 pairs, k != m
    # spelled out, each fact is plain: no system function
    assert not extend.system_function_facts(facts)


def test_every_question_with_facts_carries_the_patterns():
    """Structural steps get them through the shared fact guidance; the
    combining question, which has fact rules of its own, gets the same
    block (a buffer or an arbiter between two designs keeps entries)."""
    assert "{invariant_rules}" in extend.STRUCTURAL_TEMPLATE
    assert extend.INVARIANT_RULES.endswith(extend.IBEX_FACT_PATTERNS)
    design = {"verilog": "module m(input clk); endmodule", "property": [
        "assert (1);"], "invariants": ["1"]}
    text = squash(extend.compose_prompt(design, dict(design)))
    assert squash(extend.IBEX_FACT_PATTERNS) in text
