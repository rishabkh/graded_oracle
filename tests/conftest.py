import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_client                                                # noqa: E402


@pytest.fixture(autouse=True)
def no_old_provider_name(monkeypatch):
    """CLAUDE_PROVIDER was renamed LLM_PROVIDER on 3 Nov 2026, and the old
    name alone now stops every call. A terminal that still exports it must
    not make tests fail that never ask for a company."""
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)


@pytest.fixture(autouse=True)
def a_fresh_spending_count(monkeypatch):
    """Each test counts its own fake calls, and the test run itself never
    prints a spending line."""
    monkeypatch.setattr(llm_client, "_spending",
                        {"spent": 0.0, "calls": 0, "unpriced": 0},
                        raising=False)
