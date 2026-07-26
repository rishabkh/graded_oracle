"""One sby invocation. Never uses the GNU `timeout` binary (absent on
macOS; relying on it once turned every verification into a silent crash
masquerading as a verdict). Timeout layering: sby-native `timeout`
option (rc=8) is primary; a Python subprocess timeout is the outer guard
for a hung sby itself, reported as rc=None.
"""
from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENGINE = "smtbmc yices"
OUTER_GUARD_FACTOR = 1.5
OUTER_GUARD_GRACE_S = 10


@dataclass
class SbyOutcome:
    rc: int | None
    duration_s: float
    workdir: Path
    log_text: str
    trace_paths: list[Path] = field(default_factory=list)


def sby_available() -> bool:
    return shutil.which("sby") is not None


def make_sby_text(sv_name: str, top_module: str, mode: str, depth: int,
                  timeout_s: int, engine: str = DEFAULT_ENGINE) -> str:
    return "\n".join([
        "[options]",
        f"mode {mode}",
        f"depth {depth}",
        f"timeout {timeout_s}",
        "",
        "[engines]",
        engine,
        "",
        "[script]",
        f"read -formal {sv_name}",
        f"prep -top {top_module}",
        "",
        "[files]",
        sv_name,
        "",
    ])


def run_sby(name: str, sv_path: Path, top_module: str, mode: str, depth: int,
            timeout_s: int, workdir_root: Path,
            engine: str = DEFAULT_ENGINE) -> SbyOutcome:
    rundir = Path(workdir_root) / f"{name}_{mode}_{uuid.uuid4().hex[:8]}"
    rundir.mkdir(parents=True)
    shutil.copy(sv_path, rundir / sv_path.name)
    sby_file = rundir / "job.sby"
    sby_file.write_text(
        make_sby_text(sv_path.name, top_module, mode, depth, timeout_s, engine))
    workdir = rundir / "job"

    start = time.monotonic()
    stderr = ""
    try:
        # cwd=rundir: sby resolves [files] entries relative to the process
        # CWD, not the .sby location.
        proc = subprocess.run(
            ["sby", "-f", "job.sby"],
            cwd=rundir,
            capture_output=True, text=True,
            timeout=OUTER_GUARD_FACTOR * timeout_s + OUTER_GUARD_GRACE_S)
        rc: int | None = proc.returncode
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        rc = None
    duration = time.monotonic() - start

    logfile = workdir / "logfile.txt"
    if logfile.exists():
        log_text = logfile.read_text()
    else:
        log_text = stderr
    traces = sorted(workdir.rglob("*.vcd")) if workdir.exists() else []
    return SbyOutcome(rc=rc, duration_s=duration, workdir=workdir,
                      log_text=log_text, trace_paths=traces)
