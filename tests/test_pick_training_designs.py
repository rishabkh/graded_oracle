"""Picking a training file's designs by group. The two groups of 5 Nov
2026 take GPT-6.1 Sol's labels (extender/label_designs.py) and keep only
the designs it called good or weak."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import pick_training_designs as ptd                              # noqa: E402


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def gpt_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(ptd, "GPT_LABELS_217", write(tmp_path / "l217.jsonl", [
        {"key": "a", "label": "good"}, {"key": "b", "label": "wrong"},
        {"key": "c", "label": "weak"}, {"key": "d", "label": None}]))
    monkeypatch.setattr(ptd, "GPT_LABELS_540", write(tmp_path / "l540.jsonl", [
        {"key": "e", "label": "wrong"}, {"key": "f", "label": "weak"}]))


def test_the_217_kept_leave_out_what_gpt_called_wrong(monkeypatch, tmp_path):
    gpt_labels(monkeypatch, tmp_path)
    assert ptd.keys_of("unlabeled_good_weak", []) == ["a", "c"]


def test_the_540_kept_are_its_good_and_weak_ones(monkeypatch, tmp_path):
    gpt_labels(monkeypatch, tmp_path)
    assert ptd.keys_of("remaining_good_weak", []) == ["f"]


def test_a_kept_design_keeps_the_id_it_had_so_its_1_step_check_is_reused(
        monkeypatch, tmp_path):
    gpt_labels(monkeypatch, tmp_path)
    designs = [{"key": k, "verilog": "module m; endmodule", "top_module": "m",
                "invariants": [], "sources": []} for k in ("a", "c")]
    monkeypatch.setattr(ptd, "DESIGNS", write(tmp_path / "d.jsonl", designs))
    monkeypatch.setattr(ptd, "LABELS", write(tmp_path / "l.jsonl", []))
    rows = ptd.rows(["unlabeled_good_weak"])
    assert [r["id"] for r in rows] == ["unlabeled_a", "unlabeled_c"]  # as in R7
    assert {r["group"] for r in rows} == {"unlabeled_good_weak"}


def test_the_old_unlabeled_group_is_unchanged():
    labels = [{"key": "x", "label": None, "steps_from_start": 0,
               "appears_in": [{"file": "initiator/logs/attempts.jsonl"}]}]
    assert ptd.keys_of("unlabeled", labels) == ["x"]


def test_both_new_groups_can_be_asked_for():
    assert {"unlabeled_good_weak", "remaining_good_weak"} <= set(ptd.GROUPS)


def test_the_storage_group_takes_the_programs_designs_but_never_the_held_out(
        monkeypatch, tmp_path):
    rows = [{"id": f"storage_queue_{i:03d}", "verilog": "module m; endmodule",
             "top_module": "m", "invariants": ["1"], "generation": 0,
             "held_out": i == 10, "verdict": "NECESSARY"} for i in (0, 1, 10)]
    monkeypatch.setattr(ptd, "STORAGE", write(tmp_path / "s.jsonl", rows))
    monkeypatch.setattr(ptd, "DESIGNS", write(tmp_path / "d.jsonl", []))
    monkeypatch.setattr(ptd, "LABELS", write(tmp_path / "l.jsonl", []))
    picked = ptd.rows(["storage"])
    assert [r["id"] for r in picked] == ["storage_queue_000", "storage_queue_001"]
    assert {r["group"] for r in picked} == {"storage"}
