"""The triple schema, in the Claude API's structured-output form.

Field order matches the prompt: the design first, then the fake state and
a short explanation of the finished answer (26 Oct 2026). Until then
`cti_reasoning` and the fake state came first, so the model stated the gap
before the design that contains it; Opus 5.5 stops such requests as
reasoning written into the answer (prompts.py has the details), and it
thinks before it answers anyway. cti_state is an array of {signal, value}
pairs (strict schemas require fixed keys on every object; a free-form dict
is not expressible). The oracle's contract ignores both cti_* fields; they
are logged, not graded.
"""

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "invariants": _STR_LIST,
        "verilog": _STR,
        "top_module": _STR,
        "clock": _STR,
        "antecedents": _STR_LIST,
        "sanity_covers": _STR_LIST,
        "cti_state": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"signal": _STR, "value": _STR},
                "required": ["signal", "value"],
                "additionalProperties": False,
            },
        },
        "cti_explanation": _STR,
    },
    "required": ["invariants", "verilog", "top_module", "clock",
                 "antecedents", "sanity_covers", "cti_state",
                 "cti_explanation"],
    "additionalProperties": False,
}
