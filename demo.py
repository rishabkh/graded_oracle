#!/usr/bin/env python
"""Run every exemplar triple through grade() and print the tier table.

The August-meeting demo: shows each hand-crafted triple landing in its
designed tier, with the one-line reason the oracle gives.
"""
import json
import sys
from pathlib import Path

from oracle import NecessityVerdict, PropertyInfo, grade, grade_triple
from oracle.sby import sby_available

TRIPLES = Path(__file__).parent / "tests" / "triples"

NECESSITY_DEMOS = [
    # (label, sv, PropertyInfo kwargs, expected verdict)
    ("kitest + 'sa == sb'",
     "kitest_weak.sv",
     dict(top_module="kitest", clock="i_clk", invariants=["sa == sb"]),
     NecessityVerdict.NECESSARY),
    ("counter + tautology",
     "counter_proven.sv",
     dict(top_module="counter", invariants=["count <= 4'd15"]),
     NecessityVerdict.DECORATIVE),
]


def main() -> int:
    if not sby_available():
        print("sby not on PATH — activate hwtools first")
        return 1
    mismatches = 0
    for jf in sorted(TRIPLES.glob("*.json")):
        info = json.loads(jf.read_text())
        prop = PropertyInfo(
            top_module=info["top_module"],
            clock=info.get("clock", "clk"),
            antecedents=info.get("antecedents", []),
            sanity_covers=info.get("sanity_covers", []))
        result = grade(TRIPLES / info["sv"], prop,
                       **info.get("grade_kwargs", {}))
        expected = info["expected_tier"]
        ok = result.tier.name == expected
        mismatches += 0 if ok else 1
        mark = "ok      " if ok else "MISMATCH"
        print(f"{mark} {jf.stem:18s} expected={expected:13s} "
              f"got={result.tier.name:13s} {result.reason}")

    print()
    print("necessity criterion (grade twice: with / without invariants):")
    for label, sv, prop_kwargs, expected_verdict in NECESSITY_DEMOS:
        r = grade_triple(TRIPLES / sv, PropertyInfo(**prop_kwargs),
                         timeout_s=120)
        ok = r.verdict is expected_verdict
        mismatches += 0 if ok else 1
        mark = "ok      " if ok else "MISMATCH"
        print(f"{mark} {label:22s} expected={expected_verdict.name:11s} "
              f"got={r.verdict.name:11s} {r.reason}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
