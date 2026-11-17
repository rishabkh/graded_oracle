"""The initiator is told two things real hardware always does (16 Nov
2026, Rishab: "alright let's do it", change 2 of 7).

Of the 254 starting designs of the 15 Nov pilot, 42 were really wrong
(GPT's labels, each complaint traced in the code). 21 dropped or overwrote
a request that arrived while the design was busy or full, with no output
telling the sender to wait. 4 froze with one pattern: a pipeline that
moves only when a new input arrives or the receiver is already ready
(`advance = in_valid || out_ready`), so a lone item never reaches the
output for a receiver that waits for valid. The checker sees neither, so
the instructions now say:
  - never drop or overwrite a request silently: an input that can be
    refused comes with an output that tells the sender to wait;
  - an accepted item reaches the output without help: no later input to
    push it along, and valid never waits for ready.
Each rule is stamped on every row, like the rules before it."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import prompts as P                                             # noqa: E402
import run as R                                                 # noqa: E402

FLAT = " ".join(P.SYSTEM_PROMPT.split())
HARD = " ".join(P.SYSTEM_PROMPT.split("## Hard constraints on the Verilog")[1]
                .split("## Output format")[0].split())


def test_a_request_is_never_dropped_or_overwritten_silently():
    assert "Never drop or overwrite a request silently" in HARD
    assert "tells the sender to wait" in HARD
    # the ways to tell it: a ready, or a busy or full flag
    for word in ("`ready`", "`busy`", "`full`"):
        assert word in HARD
    # and what a design without such an output owes its sender
    assert "must take every item it is offered" in HARD


def test_an_accepted_item_reaches_the_output_without_help():
    assert "must reach the output without help" in HARD
    assert "no later input to push it along" in HARD
    assert "valid must not wait for the receiver's ready" in HARD
    # the receiver the labels judge by, and the sender that falls silent
    assert "may raise ready only after it sees valid" in HARD
    assert "send one item and then nothing more" in HARD


def test_the_pattern_that_froze_four_pilot_designs_is_named():
    assert "`in_valid || out_ready`" in HARD


def test_the_enable_rule_is_still_there():
    # the two rules sit beside the enable rule, they do not replace it
    assert "The design MUST have an enable or idle condition" in HARD


def test_each_rule_has_its_own_stamp():
    assert P.prompt_version().split("+")[-2:] == [
        "senders-told-to-wait", "items-delivered-unaided"]
    wait = "- Never drop or overwrite a request silently."
    alone = ("- An item the design has accepted must reach the output "
             "without help:")
    assert P.prompt_version(wait) == "nocap+senders-told-to-wait"
    assert P.prompt_version(alone) == "nocap+items-delivered-unaided"


def test_every_earlier_prompt_keeps_the_stamp_it_had():
    assert P.prompt_version("You write SystemVerilog modules.\n") == "nocap"
    old = P.SYSTEM_PROMPT.replace(
        "Never drop or overwrite a request silently", "").replace(
        "must reach the output without help", "")
    assert P.prompt_version(old) == (
        "nocap+size-params+plain-facts+no-async-reset-reads"
        "+explanation-after-design")


def test_rows_carry_the_new_stamps(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    _, _, plans = R.plan_attempts(2, cmd="test", grade=False, model=R.MODEL,
                                  constructs_path=None, scales_path=None,
                                  even_seeds=False)
    for record, *_ in plans:
        assert record["prompt_version"].endswith(
            "+senders-told-to-wait+items-delivered-unaided")


def test_the_rules_do_not_name_any_test_design():
    for name in ("Technion", "benchmark", "lemma miner", "hard set"):
        assert name.lower() not in FLAT.lower()
