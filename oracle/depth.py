"""How far the cover check looks for a property's condition (8 Oct 2026).

The proof itself stays at its 20 steps. Only the bounded "can this
condition happen at all" pass may look further: a 64-slot buffer needs 64
pushes before "full" can be true, so at 20 steps its condition is always
unreached and goes to the slow pdr check, which on big storage runs out of
time and loses the design. The deeper look is the storage size, or the
largest count a condition names, whichever is bigger, on top of the proof
depth, capped. A size it cannot read is skipped, never fatal: the pdr
check behind it still decides.
"""
from __future__ import annotations

import ast
import operator
import re

COVER_DEPTH_CAP = 300
# a count bigger than this cannot be reached step by step anyway; pdr decides
MAX_COUNT_HINT = 1024

_PARAM = re.compile(
    r"\b(?:parameter|localparam)\b(?:\s+(?:int|integer|logic|bit|"
    r"unsigned|signed)\b)*(?:\s*\[[^\]]*\])?\s*([A-Za-z_]\w*)\s*=\s*"
    r"([^,;)]+)")
# a declaration's unpacked dimensions: name followed by one or more [..]
_DECL = re.compile(
    r"\b(?:reg|logic|wire|bit|int|integer)\b[^;=]*?\b([A-Za-z_]\w*)\s*"
    r"((?:\[[^\]]+\]\s*)+);")
_DIM = re.compile(r"\[([^\]]+)\]")
_LITERAL = re.compile(r"\d*'[sS]?([bBoOdDhH])([0-9a-fA-F_xXzZ?]+)|\b(\d+)\b")
_BASES = {"b": 2, "o": 8, "d": 10, "h": 16}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
        ast.Div: operator.floordiv, ast.Pow: operator.pow,
        ast.LShift: operator.lshift}


def _eval(expr: str, names: dict[str, int]) -> int | None:
    expr = re.sub(r"\d*'[sS]?[dD](\d+)", r"\1", expr.strip())
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, b = ev(node.left), ev(node.right)
            if a is None or b is None or (isinstance(node.op, ast.Pow)
                                          and b > 64):
                return None
            try:
                return _OPS[type(node.op)](a, b)
            except (ZeroDivisionError, ValueError, OverflowError):
                return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = ev(node.operand)
            return None if v is None else -v
        return None
    return ev(tree)


def _params(source: str) -> dict[str, int]:
    names: dict[str, int] = {}
    for name, value in _PARAM.findall(source):
        v = _eval(value, names)
        if v is not None:
            names[name] = v
    return names


def _dim_size(dim: str, names: dict[str, int]) -> int | None:
    if ":" in dim:
        lo, _, hi = dim.partition(":")
        a, b = _eval(lo, names), _eval(hi, names)
        return None if a is None or b is None else abs(a - b) + 1
    return _eval(dim, names)            # SystemVerilog [N]


def storage_slots(source: str) -> int:
    """The largest unpacked array in the design, in slots (0 if none)."""
    names = _params(source)
    best = 0
    for _, dims in _DECL.findall(source):
        slots = 1
        for dim in _DIM.findall(dims):
            n = _dim_size(dim, names)
            if n is None or n <= 0:
                slots = 0
                break
            slots *= n
        best = max(best, slots)
    return best


def largest_count(exprs: list[str], names: dict[str, int] | None = None) -> int:
    best = 0
    for expr in exprs:
        for name in re.findall(r"[A-Za-z_]\w*", expr):
            v = (names or {}).get(name)
            if v is not None and 0 < v <= MAX_COUNT_HINT:
                best = max(best, v)
        for base, digits, plain in _LITERAL.findall(expr):
            text = digits.replace("_", "") if base else plain
            if not text or re.search(r"[xXzZ?]", text):
                continue
            try:
                v = int(text, _BASES[base.lower()] if base else 10)
            except ValueError:
                continue
            if v <= MAX_COUNT_HINT:
                best = max(best, v)
    return best


def suggest_cover_depth(source: str, exprs: list[str], depth: int,
                        cap: int = COVER_DEPTH_CAP) -> int:
    """`depth` plus the storage size or the largest count in the cover
    expressions, whichever is bigger; never above `cap`."""
    names = _params(source)
    # a named constant of the design counts too: a timer loaded with PERIOD
    # needs PERIOD steps before "remaining == 0", which names no count
    design_counts = [v for v in names.values() if 0 < v <= MAX_COUNT_HINT]
    hint = max([storage_slots(source), largest_count(exprs, names)]
               + design_counts)
    return max(depth, min(depth + hint, cap))
