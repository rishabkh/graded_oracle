"""One status line for the two-step and self-repair runs (9 Oct 2026):
they went silent while a batch was out, sometimes for many minutes. The
line says what every design is doing, how long batches have been out and
what the run has spent. The model and the checker are faked."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

from tests.test_initiator_pipeline import Fake                   # noqa: E402


def _status(tmp_path, n=3, batch=False):
    import run as R
    return R.PipelineStatus(n, "claude-opus-5-5", batch, tmp_path / "batches")


def test_the_line_counts_what_every_design_is_doing(tmp_path):
    s = _status(tmp_path)
    s.set(0, "writing")
    s.set(1, "checking")
    s.set(2, "done")
    line = s.label()
    assert "3 designs: 1 waiting for the model, 1 being checked, 1 done" in line


def test_the_line_says_how_long_a_batch_has_been_out(tmp_path):
    s = _status(tmp_path, batch=True)
    d = tmp_path / "batches"
    d.mkdir()
    sent = datetime.now(timezone.utc) - timedelta(minutes=4, seconds=10)
    (d / "batch_x_r00001.json").write_text(json.dumps(
        {"created": sent.isoformat(), "batch_id": "msgbatch_1"}))
    (d / "batch_y_r00002.json").write_text(json.dumps(       # came back
        {"created": sent.isoformat(), "batch_id": "msgbatch_2",
         "ended": datetime.now(timezone.utc).isoformat()}))
    old = datetime.now(timezone.utc) - timedelta(days=1)       # another run
    (d / "batch_z_r00003.json").write_text(json.dumps(
        {"created": old.isoformat(), "batch_id": "msgbatch_0"}))
    s.started = sent - timedelta(seconds=5)
    line = s.label(force=True)
    assert "1 batch out for 4:1" in line


def test_the_line_shows_what_the_run_has_spent(tmp_path):
    s = _status(tmp_path, batch=True)
    s.records = [{"usage": {"input": 1_000_000, "output": 100_000}},
                 {"usage": {"input": 0, "output": 0}}, {}]
    # Opus 5.5 at half price: $2 per million in, $10 per million out
    assert "$3.00 so far" in s.label()


def test_a_pipeline_run_shows_one_status_line_and_no_check_spinners(
        monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN", "NECESSARY"])
    made = []

    class Recorder:
        def __init__(self, label):
            made.append(label)
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def say(self, text):
            print(text)
    monkeypatch.setattr(f.R, "Spinner", Recorder)
    [row], _ = f.run(n=1, repairs=2)
    assert row["verdict"] == "NECESSARY"
    assert len(made) == 1 and callable(made[0])      # the status line only
    assert "1 done" in made[0]()


def test_the_spinner_takes_a_label_that_changes(capsys):
    import run as R
    values = iter(["first", "second"])
    s = R.Spinner(lambda: next(values))
    assert s.text() == "first" and s.text() == "second"
    s.say("a result line")                       # no terminal: plain print
    assert "a result line" in capsys.readouterr().out


def test_a_status_that_cannot_be_built_never_stops_the_run():
    import run as R

    def broken():
        raise ValueError("no price for this model")
    assert R.Spinner(broken).text() == "working (status unavailable)"
