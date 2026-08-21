"""The triple schema, in the Claude API's structured-output form.

Field order matches the prompt: reasoning first, design after — models
generate left to right, so the gap is stated before the design that
contains it. cti_state is an array of {signal, value} pairs (strict
schemas require fixed keys on every object; a free-form dict is not
expressible). The oracle's contract ignores both cti_* fields; they are
logged, not graded.
"""

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "cti_reasoning": _STR,
        "cti_state": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"signal": _STR, "value": _STR},
                "required": ["signal", "value"],
                "additionalProperties": False,
            },
        },
        "invariants": _STR_LIST,
        "verilog": _STR,
        "top_module": _STR,
        "clock": _STR,
        "antecedents": _STR_LIST,
        "sanity_covers": _STR_LIST,
    },
    "required": ["cti_reasoning", "cti_state", "invariants", "verilog",
                 "top_module", "clock", "antecedents", "sanity_covers"],
    "additionalProperties": False,
}
