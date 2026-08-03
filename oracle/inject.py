"""Inject oracle-owned blocks into a copy of the design source.

Blocks anchor on the *named* top module: first `endmodule` after
`module <top_module>` — never the file's last endmodule, so multi-module
files where the top is not last stay correct. Comments are masked before
searching so `endmodule` inside a comment cannot false-match.

Two block kinds:
- cover block (vacuity check): clocked covers on antecedent/sanity exprs
- invariant block (necessity/strengthening): combinational asserts,
  `always @(*)` so they constrain every state in the induction window
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADER = "// ORACLE-INJECTED VACUITY CHECK"
INVARIANT_HEADER = "// ORACLE-INJECTED INVARIANTS"


class InjectionError(Exception):
    pass


@dataclass
class Injection:
    text: str
    line_map: dict[int, tuple[str, str]]  # 1-based line -> (kind, expr)


def _mask_comments(source: str) -> str:
    """Blank out comments with spaces, preserving every character offset."""
    def blank(m: re.Match) -> str:
        return "".join(c if c == "\n" else " " for c in m.group())

    masked = re.sub(r"/\*.*?\*/", blank, source, flags=re.S)
    masked = re.sub(r"//[^\n]*", blank, masked)
    return masked


def _insert_point(source: str, top_module: str) -> int:
    """Char offset of the start of the named module's endmodule line."""
    masked = _mask_comments(source)
    mod = re.search(r"\bmodule\s+" + re.escape(top_module) + r"\b", masked)
    if mod is None:
        raise InjectionError(f"top module '{top_module}' not found in source")
    end = re.search(r"\bendmodule\b", masked[mod.end():])
    if end is None:
        raise InjectionError(f"no endmodule found after module '{top_module}'")
    end_pos = mod.end() + end.start()
    return source.rfind("\n", 0, end_pos) + 1


def _inject_block(source: str, top_module: str, header: str,
                  always_head: str, stmt: str,
                  tagged: list[tuple[str, str]]) -> Injection:
    if not tagged:
        return Injection(text=source, line_map={})
    insert_at = _insert_point(source, top_module)
    block = [header, always_head]
    block += [f"    {stmt} ({expr});  // {kind}" for kind, expr in tagged]
    block.append("end")

    header_line = source[:insert_at].count("\n") + 1
    line_map = {header_line + 2 + i: tagged[i] for i in range(len(tagged))}
    text = source[:insert_at] + "\n".join(block) + "\n" + source[insert_at:]
    return Injection(text=text, line_map=line_map)


def inject_covers(source: str, top_module: str, clock: str,
                  antecedents: list[str], sanity_covers: list[str]) -> Injection:
    tagged = [("antecedent", e) for e in antecedents]
    tagged += [("sanity", e) for e in sanity_covers]
    return _inject_block(source, top_module, HEADER,
                         f"always @(posedge {clock}) begin", "cover", tagged)


def inject_invariants(source: str, top_module: str,
                      invariants: list[str]) -> Injection:
    tagged = [("invariant", e) for e in invariants]
    return _inject_block(source, top_module, INVARIANT_HEADER,
                         "always @(*) begin", "assert", tagged)
