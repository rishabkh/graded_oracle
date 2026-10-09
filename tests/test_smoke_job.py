"""The short practice job on the cluster (9 Oct 2026): it can now practise
on any training file, so the new chat records (system message, question,
answer) are tried on a small model before 32B time is spent."""
from pathlib import Path

JOB = (Path(__file__).resolve().parent.parent / "training"
       / "smoke.sbatch").read_text()


def test_the_practice_file_can_be_chosen_and_defaults_to_the_old_one():
    assert 'PAIRS="${PAIRS:-extender/sft_smoke.jsonl}"' in JOB
    assert JOB.count('--pairs "$PAIRS"') == 2       # training and the check
    assert "--pairs extender/sft_smoke.jsonl" not in JOB


def test_the_practice_sample_mixes_both_kinds_of_record():
    import json
    rows = [json.loads(l) for l in (Path(__file__).resolve().parent.parent
            / "extender" / "sft_smoke_with_design_writing.jsonl")
            .read_text().splitlines()]
    assert isinstance(rows[0]["prompt"], list)       # checked: a chat record
    assert rows[0]["prompt"][0]["role"] == "system"
    assert any(isinstance(r["prompt"], str) for r in rows)    # solving rows
    assert all(r["holdout"] is False for r in rows)
