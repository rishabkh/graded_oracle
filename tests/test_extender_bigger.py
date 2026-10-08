"""The extender, made ready for bigger designs (8 Oct 2026, change 10).

Its prompts held every extension to 4-8 bit logic ("Widths stay narrow"),
its checks used the old 120 s limit with no deeper look, two of its steps
cut answers at 16,000 tokens, and a timed-out check was sent to the paid
fixer with nothing to fix. Its proofs now go to a folder next to its logs
when the batch runner says where."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import distractor                                                # noqa: E402
import extend                                                    # noqa: E402
import fix                                                       # noqa: E402
from promote import fixable                                      # noqa: E402


def _all_prompt_text():
    return "\n".join([extend.SYSTEM_PROMPT, extend.COMPOSE_TEMPLATE,
                      extend.REPLICATE_TEMPLATE] + list(extend.MOVES.values()))


def test_no_extender_prompt_holds_widths_narrow():
    text = _all_prompt_text().lower()
    for phrase in ("narrow", "four to eight", "at most 8"):
        assert phrase not in text, phrase


def test_new_logic_matches_the_widths_already_there():
    assert "match the widths" in extend.SYSTEM_PROMPT.lower()
    assert "parents' widths" in extend.COMPOSE_TEMPLATE
    assert "parent's widths" in extend.REPLICATE_TEMPLATE


def test_covers_are_no_longer_promised_within_20_cycles():
    text = _all_prompt_text()
    assert "~20 cycles" not in text
    for template in (extend.COMPOSE_TEMPLATE, extend.REPLICATE_TEMPLATE):
        assert "reachable from reset" in " ".join(template.split())


def test_checks_use_the_new_limits_and_the_deeper_look():
    kw = distractor.GRADE_KWARGS
    assert kw["timeout_s"] == 300 and kw["pdr_timeout_s"] == 120
    assert kw["cover_depth"] == "auto"


def test_side_steps_no_longer_cut_answers_at_16000():
    assert distractor.MAX_TOKENS >= 64000 and fix.MAX_TOKENS >= 64000


def test_a_timed_out_check_is_not_sent_to_the_fixer():
    assert not fixable({"reason": "with-invariants grade is TIMEOUT, not PROVEN"})
    assert not fixable({"reason": "with-invariants grade is ERROR, not PROVEN"})
    assert fixable({"reason": "with-invariants grade is NOT_INDUCTIVE, not PROVEN"})


def test_proofs_go_next_to_the_logs_when_a_folder_is_given(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(distractor, "PROOF_ROOT", None)
    assert distractor.grade_kwargs({}) == distractor.GRADE_KWARGS
    monkeypatch.setattr(distractor, "PROOF_ROOT", tmp_path / "proofs")
    rec = {"extension_id": "2026-10-08_10h00m00s", "parent_id": "c0_001"}
    kw = distractor.grade_kwargs(rec)
    assert kw["workdir_root"].parent == tmp_path / "proofs"
    assert rec["proof_dir"].startswith("proofs/2026-10-08_10h00m00s_c0_001_")
    again = distractor.grade_kwargs(rec, leg="p2_alone")
    assert again["workdir_root"] == kw["workdir_root"] / "p2_alone"
    assert kw["timeout_s"] == 300


def test_the_second_property_check_keeps_its_whole_result(monkeypatch):
    from oracle import GradeResult, Tier
    seen = {}

    def fake_grade(f, prop, **kw):
        seen.update(kw)
        return GradeResult(Tier.NOT_INDUCTIVE, "did not close")
    monkeypatch.setattr(extend, "grade", fake_grade)
    monkeypatch.setattr(extend, "state_bits",
                        lambda v, top: 2 if "c <=" in v else 1)
    monkeypatch.setattr(extend, "grade_triple_generated",
                        lambda *a, **k: (_ for _ in ()).throw(StopIteration))
    parent = {"verilog": ("module m (input wire clk, input wire b, "
                          "output reg a);\n"
                          "always @(posedge clk)\n"
                          "    assert (a == 1'b0);\nendmodule"),
              "property": ["a == 1'b0"], "invariants": [], "top_module": "m",
              "clock": "clk", "antecedents": [], "sanity_covers": []}
    out = {"patch": ("@@     assert (a == 1'b0); @@\n"
                     "+ reg c = 0;\n"
                     "+ always @(posedge clk) c <= b;\n"
                     "+ always @(posedge clk)\n"
                     "+     assert (!c || !a);"),
           "invariants": [], "dispositions": []}
    record = {}
    try:
        extend.grade_step4(parent, "second", out, record)
    except StopIteration:
        pass
    assert record["p2_unaided_tier"] == "NOT_INDUCTIVE"
    assert record["p2_unaided_result"]["reason"] == "did not close"
    assert seen["cover_depth"] == "auto"



def test_the_fixer_queue_skips_timed_out_records():
    """8 Oct 2026 review: only promote.py checked fixable(); the fixer
    itself took every queued record."""
    queue = [{"extension_id": "e1", "reason": "with-invariants grade is "
              "TIMEOUT, not PROVEN"},
             {"extension_id": "e2", "reason": "with-invariants grade is "
              "NOT_INDUCTIVE, not PROVEN"}]
    assert [r["extension_id"] for r in fix.pending_tasks(queue, set())] \
        == ["e2"]
