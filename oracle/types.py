"""Data vocabulary of the graded oracle. No logic lives here.

Tier is ordered worst-to-best. ERROR/TIMEOUT are non-verdicts (the judge
did not rule); they must never be conflated with FALSE/VACUOUS/etc.
BOUNDED is reserved for deliberate bmc-only grading and is never emitted
by the v1 orchestration (grade() always runs prove mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
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


@dataclass
class GradeResult:
    tier: Tier
    reason: str
    runs: list[RunEvidence] = field(default_factory=list)
