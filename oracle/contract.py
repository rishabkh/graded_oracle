"""The generator output contract.

The LLM workers emit one JSON object per attempt:

    {"verilog": "...", "top_module": "...", "clock": "clk",
     "antecedents": [...], "sanity_covers": [...], "invariants": [...]}

parse_generator_output() validates that text into a GeneratorOutput
(verilog + PropertyInfo) or raises ContractViolation with a precise,
Fixer-consumable message. The grading layer catches the violation at the
boundary and returns tier ERROR — malformed generator output is a
non-verdict, never an exception in the loop.

Markdown code fences are stripped before parsing: models routinely wrap
JSON in ``` fences, and starving the loop over a formatting tic wastes
attempts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .inject import _mask_comments
from .types import PropertyInfo

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n(.*?)\n?```\s*$", re.S)


class ContractViolation(Exception):
    pass


@dataclass
class GeneratorOutput:
    verilog: str
    prop: PropertyInfo


def _strip_fences(text: str) -> str:
    m = _FENCE.match(text.strip())
    return m.group(1) if m else text


def _string_list(obj: dict, key: str) -> list[str]:
    value = obj.get(key, [])
    if not isinstance(value, list):
        raise ContractViolation(f"'{key}' must be a list of strings, "
                                f"got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ContractViolation(f"'{key}' must contain only non-empty "
                                    f"strings, got {item!r}")
    return value


def parse_generator_output(text: str) -> GeneratorOutput:
    try:
        obj = json.loads(_strip_fences(text))
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"output is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ContractViolation("output must be a JSON object, got "
                                f"{type(obj).__name__}")

    verilog = obj.get("verilog")
    if not isinstance(verilog, str) or not verilog.strip():
        raise ContractViolation("'verilog' must be a non-empty string")

    top_module = obj.get("top_module")
    if not isinstance(top_module, str) or not top_module:
        raise ContractViolation("'top_module' must be a non-empty string")
    if not _IDENTIFIER.match(top_module):
        raise ContractViolation(f"'top_module' is not a legal Verilog "
                                f"identifier: {top_module!r}")

    clock = obj.get("clock", "clk")
    if not isinstance(clock, str) or not clock.strip():
        raise ContractViolation("'clock' must be a non-empty string")

    if not re.search(r"\bmodule\s+" + re.escape(top_module) + r"\b",
                     _mask_comments(verilog)):
        raise ContractViolation(f"module '{top_module}' not found in the "
                                "supplied verilog")

    prop = PropertyInfo(
        top_module=top_module,
        clock=clock,
        antecedents=_string_list(obj, "antecedents"),
        sanity_covers=_string_list(obj, "sanity_covers"),
        invariants=_string_list(obj, "invariants"))
    return GeneratorOutput(verilog=verilog, prop=prop)
