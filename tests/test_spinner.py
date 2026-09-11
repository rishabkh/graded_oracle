"""Spinner switch: with many batch workers, per-step spinners must stay
silent (they draw over each other) while one pool-wide spinner runs."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

from distractor import Spinner


class TTY(io.StringIO):
    def isatty(self):
        return True


def test_quiet_spinner_draws_nothing(monkeypatch):
    monkeypatch.setattr(sys, "stderr", TTY())
    monkeypatch.setattr(Spinner, "quiet", True)
    with Spinner("oracle grading x") as s:
        assert s._thread is None


def test_always_spinner_ignores_quiet(monkeypatch):
    monkeypatch.setattr(sys, "stderr", TTY())
    monkeypatch.setattr(Spinner, "quiet", True)
    with Spinner("batch 0/30 done", always=True) as s:
        assert s._thread is not None
