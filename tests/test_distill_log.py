"""Every model call saved as a training example in Formal Disco's shape
(formal-disco/distill_common.py DistillExample): prompt (the call type),
arguments (its inputs), response, outcome, metadata. Ours also keeps the
exact messages sent, so a later change to a prompt template cannot
silently change what an example says."""
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import distill_log as dl                                         # noqa: E402

MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


def test_the_call_types_cover_formal_discos_and_ours():
    assert set(dl.TYPES) == {"idea", "implement", "initiate", "generate",
                             "repair", "extend", "solve"}


@pytest.mark.parametrize("verdict, outcome", [
    ("NECESSARY", "success"), ("PROVEN", "success"),
    ("NOT_PROVEN", "fail"), ("DECORATIVE", "fail"), ("SCOPE_UNARMED", "fail"),
    ("NO_NEW_SHAPE", "fail"), ("FALSE", "fail"), ("NOT_INDUCTIVE", "fail"),
    ("WRAPPER_ERROR", "fail"), ("HIERARCHICAL_REF", "fail"),
    ("REFUSED", "error"), ("TRUNCATED", "error"), ("UNPARSEABLE", "error"),
    ("ERROR", "error"), ("TIMEOUT", "error"), (None, "error"),
])
def test_our_verdicts_map_to_formal_discos_outcomes(verdict, outcome):
    assert dl.outcome_of(verdict) == outcome


def test_a_record_has_formal_discos_fields_plus_the_exact_messages(tmp_path):
    log = tmp_path / "distill.jsonl"
    dl.record(log, "initiate", {"repo": "a/b", "readme": "R", "kind": "FIFO"},
              '{"verilog": "..."}', "NECESSARY", MESSAGES,
              model="claude-opus-5-5", run_id="r1")
    [line] = [json.loads(l) for l in log.read_text().splitlines()]
    assert set(line) == {"prompt", "arguments", "response", "outcome",
                         "metadata"}
    assert line["prompt"] == "initiate" and line["outcome"] == "success"
    assert line["arguments"]["readme"] == "R"
    assert line["metadata"]["messages"] == MESSAGES
    assert line["metadata"]["verdict"] == "NECESSARY"
    assert line["metadata"]["model"] == "claude-opus-5-5"


def test_an_unknown_call_type_is_refused(tmp_path):
    with pytest.raises(ValueError):
        dl.record(tmp_path / "d.jsonl", "summarize", {}, "x", "NECESSARY",
                  MESSAGES)


def test_many_threads_never_mix_their_lines(tmp_path):
    log = tmp_path / "distill.jsonl"

    def write(i):
        dl.record(log, "extend", {"program": "p" * 5000}, f"r{i}", "NECESSARY",
                  MESSAGES, n=i)
    ts = [threading.Thread(target=write, args=(i,)) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert sorted(l["metadata"]["n"] for l in lines) == list(range(40))


def test_reading_back_skips_a_line_cut_short_by_a_killed_run(tmp_path):
    log = tmp_path / "distill.jsonl"
    dl.record(log, "solve", {"design": "d"}, "r", "PROVEN", MESSAGES)
    with log.open("a") as f:
        f.write('{"prompt": "solve", "argum')
    assert len(dl.read(log)) == 1
