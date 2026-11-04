"""Labeling designs good, weak or wrong with a paid model (4 Nov 2026):
the 217 starting designs in the biggest training set that nobody labeled,
with the 27 Oct reading-only rubric, after a calibration run on designs
that already have labels. The model is faked."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))
sys.path.insert(0, str(ROOT / "training"))

import label_designs as ld                                       # noqa: E402
import pick_training_designs as ptd                              # noqa: E402

DESIGN = {"key": "k1", "verilog": "module q (input clk);\n  reg a = 0;\nendmodule\n",
          "top_module": "q", "property": ["a == 1'b0"],
          "invariants": ["!a"], "description": "What it is: a tiny queue"}


def label_row(key, label=None, steps=0, file="initiator/logs/attempts.jsonl",
              simulated=False):
    return {"key": key, "label": label, "steps_from_start": steps,
            "appears_in": [{"file": file}], "checked_by_simulation": simulated}


def reply(label="wrong"):
    def call(**kw):
        call.kw = kw
        return (json.dumps({"label": label, "reason": "it drops a push when full",
                            "concern": "lines 2-3: push while full",
                            "stored_data_facts": False}),
                {"input": 2000, "output": 1500}, "ok")
    return call


def test_the_designs_to_label_are_the_unlabeled_starting_designs():
    labels = [label_row("a"), label_row("b", "good"), label_row("c", steps=1),
              label_row("d", file=ptd.OLDEST), label_row("e")]
    assert ld.to_label(labels) == ptd.keys_of("unlabeled", labels) == ["a", "e"]


def test_the_question_holds_the_rubric_the_numbered_design_and_its_facts():
    q = ld.question(DESIGN)
    assert "when torn between good and weak, choose weak" in q
    assert "ready" in q and "freeze" in q
    assert "   2| " in q and "reg a = 0;" in q
    assert "a == 1'b0" in q and "!a" in q and "a tiny queue" in q
    assert "parent" not in q.lower()            # starting designs only


def test_a_reply_becomes_a_label_row():
    row = ld.label_one(DESIGN, call=reply("wrong"), effort="medium")
    assert row["key"] == "k1" and row["label"] == "wrong"
    assert row["concern"] == "lines 2-3: push while full"
    assert row["first_pass"].endswith("reading only")
    assert row["usage"] == {"input": 2000, "output": 1500}


def test_an_unreadable_reply_is_kept_but_is_no_label():
    def call(**kw):
        return None, {"input": 10, "output": 5}, "refusal"
    row = ld.label_one(DESIGN, call=call, effort="medium")
    assert row["label"] is None and row["error"]


def test_a_rerun_skips_designs_already_labeled_and_retries_the_rest(tmp_path):
    out = tmp_path / "labels.jsonl"
    out.write_text(json.dumps({"key": "k1", "label": "good"}) + "\n" +
                   json.dumps({"key": "k2", "label": None, "error": "x"}) + "\n")
    done = ld.already_labeled(out)
    assert done == {"k1"}


def test_calibration_takes_labeled_starting_designs_evenly_simulated_first():
    labels = ([label_row(f"g{i}", "good", simulated=i < 2) for i in range(5)] +
              [label_row(f"w{i}", "weak") for i in range(5)] +
              [label_row(f"x{i}", "wrong") for i in range(5)] +
              [label_row("grown", "wrong", steps=2)])
    picked = ld.calibration_keys(labels, 6)
    by = {lab: [k for k in picked if k[0] == lab[0]] for lab in ("good", "weak", "wrong")}
    assert [len(v) for v in by.values()] == [2, 2, 2]
    assert set(by["good"]) == {"g0", "g1"}          # the simulated ones first
    assert "grown" not in picked


def test_the_agreement_report_says_how_well_wrong_is_found():
    pairs = [("wrong", "wrong"), ("wrong", "weak"), ("good", "good"),
             ("weak", "wrong"), ("weak", "weak")]
    r = ld.agreement(pairs)
    assert r["agree"] == 3 and r["total"] == 5
    assert r["wrong_found"] == (1, 2)       # 1 of the 2 known wrong ones
    assert r["wrong_right"] == (1, 2)       # 1 of its 2 "wrong" calls is right


def test_the_dry_run_prices_the_run_and_calls_nothing(capsys):
    def call(**kw):
        raise AssertionError("a dry run must not call the model")
    ld.estimate([DESIGN] * 10, call=call)
    out = capsys.readouterr().out
    assert "10 designs" in out and "$" in out
