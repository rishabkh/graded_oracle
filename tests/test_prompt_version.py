"""Which prompt wrote a row.

Raising the design-size cap splits the corpus into two populations. If a
later score moves we have to say whether it was more data or bigger
data, and that answer is unrecoverable unless the row says which prompt
made it. Derived from the prompt text rather than typed by hand, so the
stamp cannot drift away from the rule it names."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from prompts import SYSTEM_PROMPT, prompt_version                  # noqa: E402
from extender.build_corpus import flatten_record                   # noqa: E402


def test_the_current_prompt_stamps_its_line_cap():
    assert prompt_version(SYSTEM_PROMPT) == "200cap"


def test_a_prompt_with_no_cap_says_so():
    assert prompt_version("You write SystemVerilog modules.\n") == "nocap"


def test_a_different_cap_is_read_off_the_prompt():
    assert prompt_version("- Keep the module under 800 lines.") == "800cap"


def test_a_row_carries_the_prompt_that_made_it():
    record = {
        "run_id": "r1", "attempt": 1, "prompt_version": "200cap",
        "raw_json": json.dumps({
            "top_module": "m",
            "verilog": "module m (input wire clk);\n"
                       "  always @(*) assert (1);\nendmodule\n",
            "invariants": ["x == 1"]}),
    }
    assert flatten_record(record, 0)["prompt_version"] == "200cap"


def test_a_row_from_before_the_stamp_is_not_guessed():
    """The 350 rows already in the corpus were written under the 200-line
    prompt, but nothing recorded it, so the row says unknown rather than
    claiming a version it cannot know."""
    record = {"run_id": "r0", "attempt": 1,
              "raw_json": json.dumps({
                  "top_module": "m",
                  "verilog": "module m (input wire clk);\n"
                             "  always @(*) assert (1);\nendmodule\n",
                  "invariants": ["x == 1"]})}
    assert flatten_record(record, 0)["prompt_version"] is None
