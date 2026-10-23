"""A corpus row keeps the design trait that made it (22 Oct 2026, review
finding F10, corpus part).

Since 21 Oct 2026 the generator draws a design trait for every design
(asynchronous reset, a valid/ready handshake, named states, a register
loaded from an input, or plain style) and its run log records it as
"trait". build_corpus.flatten_record copied the kinds file, size seed,
seed weights and model onto the row, but not the trait, so a corpus could
not say which designs were written with which trait. No tools needed."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

from build_corpus import flatten_record                         # noqa: E402

TRAIT = ("asynchronous reset: reset the state with an asynchronous "
         "active-low input rst_n, written always @(posedge clk or negedge "
         "rst_n) with if (!rst_n) first")


def attempt(**extra):
    triple = {"top_module": "m", "clock": "clk", "verilog": "module m; "
              "endmodule", "invariants": ["a <= 4'd9"]}
    return dict({"raw_json": json.dumps(triple), "run_id": "r",
                 "attempt": 1, "verdict": "NECESSARY"}, **extra)


def test_the_trait_is_kept_on_the_row():
    row = flatten_record(attempt(trait=TRAIT, scale="64 entries"), 3, "c0")
    assert row["trait"] == TRAIT
    assert row["scale"] == "64 entries"          # kept as before


def test_a_run_from_before_the_traits_has_none():
    row = flatten_record(attempt(), 3)
    assert "trait" not in row
