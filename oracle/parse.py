"""Parsers for sby logs.

Reachability comes from the per-statement engine log lines, never from
the bare cover-mode rc: rc=2 in cover mode means "some cover unreached",
which is sometimes exactly the wanted signal (a tool convention, not a
failure). Format captured verbatim from real runs:

    Reached cover statement in step 1 at counter: counter.sv:36.5-36.26 (...)
    Unreached cover statement at counter: counter.sv:35.5-35.26 (...)

The summary section repeats both in lowercase; case-sensitive matching
on the capitalized engine lines ignores it.
"""
from __future__ import annotations

import re

_REACHED = re.compile(r"Reached cover statement in step \d+ at .*?\.sv:(\d+)\.")
_UNREACHED = re.compile(r"Unreached cover statement at .*?\.sv:(\d+)\.")
_ASSERT_FAILED = re.compile(r"Assert failed in .*?\.sv:(\d+)\.")


def parse_assert_failures(log_text: str) -> set[int]:
    """Source line numbers of failed assertions (prove-mode rc=2).

    The line identifies WHICH assertion is false — an oracle-injected
    invariant or the property itself — which decides the repair route.
    """
    return {int(m.group(1))
            for m in _ASSERT_FAILED.finditer(log_text)}


def parse_cover_log(log_text: str) -> tuple[set[int], set[int]]:
    reached: set[int] = set()
    unreached: set[int] = set()
    for line in log_text.splitlines():
        m = _REACHED.search(line)
        if m:
            reached.add(int(m.group(1)))
            continue
        m = _UNREACHED.search(line)
        if m:
            unreached.add(int(m.group(1)))
    return reached, unreached


def tail(text: str, n: int = 40) -> str:
    return "\n".join(text.splitlines()[-n:])
