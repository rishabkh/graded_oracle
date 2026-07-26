#!/usr/bin/env python
"""Run every exemplar triple through grade() and print the tier table.

The August-meeting demo: shows each hand-crafted triple landing in its
designed tier, with the one-line reason the oracle gives.
"""
import json
import sys
from pathlib import Path

from oracle import PropertyInfo, grade
from oracle.sby import sby_available

TRIPLES = Path(__file__).parent / "tests" / "triples"


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
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
