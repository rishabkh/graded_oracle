"""Data vocabulary of the graded oracle. No logic lives here.

Tier is ordered worst-to-best. ERROR/TIMEOUT are non-verdicts (the judge
did not rule); they must never be conflated with FALSE/VACUOUS/etc.
BOUNDED is reserved for deliberate bmc-only grading and is never emitted
by the v1 orchestration (grade() always runs prove mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path


class Tier(IntEnum):
    ERROR = 0
    TIMEOUT = 1
    FALSE = 2
    VACUOUS = 3
    NOT_INDUCTIVE = 4
    BOUNDED = 5
    PROVEN = 6


@dataclass
class PropertyInfo:
    top_module: str
    clock: str = "clk"
    antecedents: list[str] = field(default_factory=list)
    sanity_covers: list[str] = field(default_factory=list)
    # Candidate strengthening invariants (Verilog expressions), supplied
    # separately from the design so grade_triple() can test necessity by
    # grading with and without them.
    invariants: list[str] = field(default_factory=list)


@dataclass
class RunEvidence:
    mode: str                 # "prove" | "cover"
    rc: int | None            # None => outer guard killed sby before it exited
    depth: int
    engine: str
    duration_s: float
    workdir: Path
    log_excerpt: str
    trace_paths: list[Path] = field(default_factory=list)
    reached_covers: list[str] = field(default_factory=list)
    unreached_covers: list[str] = field(default_factory=list)
    timeout_source: str | None = None   # "sby" | "outer_guard" | None
    notes: list[str] = field(default_factory=list)
    # Source lines of failed assertions (rc=2) — decides whether the
    # false thing was the property or an injected invariant.
    failed_assert_lines: list[int] = field(default_factory=list)
    # Plain-text rendering of the CEX/CTI trace — the Fixer-readable
    # form of trace_paths[0]; a .vcd cannot go in a prompt.
    trace_text: str | None = None


@dataclass
class GradeResult:
    tier: Tier
    reason: str
    runs: list[RunEvidence] = field(default_factory=list)


class NecessityVerdict(Enum):
    """Outcome of the two-call necessity check (grade_triple).

    A triple is Stage-4-worthy only when the strengthening invariants are
    load-bearing: PROVEN with them, NOT_INDUCTIVE without them.
    """
    NECESSARY = "necessary"          # with: PROVEN, without: NOT_INDUCTIVE
    DECORATIVE = "decorative"        # with: PROVEN, without: PROVEN
    NOT_PROVEN = "not_proven"        # with-invariants grade != PROVEN
    INCONCLUSIVE = "inconclusive"    # without-run gave a non-verdict/other
    NO_INVARIANTS = "no_invariants"  # nothing to test necessity of


@dataclass
class TripleResult:
    verdict: NecessityVerdict
    reason: str
    with_invariants: GradeResult | None = None
    without_invariants: GradeResult | None = None
