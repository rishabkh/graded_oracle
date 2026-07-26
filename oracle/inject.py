"""Inject the oracle-owned cover block into a copy of the design source.

The block anchors on the *named* top module: first `endmodule` after
`module <top_module>` — never the file's last endmodule, so multi-module
files where the top is not last stay correct. Comments are masked before
searching so `endmodule` inside a comment cannot false-match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADER = "// ORACLE-INJECTED VACUITY CHECK"


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


def inject_covers(source: str, top_module: str, clock: str,
                  antecedents: list[str], sanity_covers: list[str]) -> Injection:
    masked = _mask_comments(source)
    mod = re.search(r"\bmodule\s+" + re.escape(top_module) + r"\b", masked)
    if mod is None:
        raise InjectionError(f"top module '{top_module}' not found in source")
    end = re.search(r"\bendmodule\b", masked[mod.end():])
    if end is None:
        raise InjectionError(f"no endmodule found after module '{top_module}'")
    end_pos = mod.end() + end.start()

    insert_at = source.rfind("\n", 0, end_pos) + 1  # start of endmodule's line

    tagged = [("antecedent", e) for e in antecedents]
    tagged += [("sanity", e) for e in sanity_covers]
    block = [HEADER, f"always @(posedge {clock}) begin"]
    block += [f"    cover ({expr});  // {kind}" for kind, expr in tagged]
    block.append("end")

    header_line = source[:insert_at].count("\n") + 1
    line_map = {header_line + 2 + i: tagged[i] for i in range(len(tagged))}
    text = source[:insert_at] + "\n".join(block) + "\n" + source[insert_at:]
    return Injection(text=text, line_map=line_map)
