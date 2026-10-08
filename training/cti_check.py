"""Is a claimed fake state a real counterexample to induction?

The reasoning written for each training row names a "fake state": values
for some signals where the invariants R are false, the property still
holds, and one clock step breaks the property. A model can describe such
a state convincingly and still be wrong, so before a reasoning trace goes
into training we ask the proof tool.

Accept iff some choice of every unnamed register and input makes, at the
first step: every named signal equal its value, R false, every assertion
true; and after one clock step some assertion false.

How it is asked (one bounded sby run; two for a mixed design):
  - the named values and !R are assumed under $initstate, in a block
    placed before the top module's endmodule;
  - `attrmap -remove init` frees every register's start value so the
    pins can choose it. `memory_map` runs first because a memory's start
    value from an initial loop is not an init attribute and would stick;
  - `skip K` with smtbmc `--assume-skipped 0` assumes every assertion in
    the skipped steps and checks only the last step.
An assertion inside always @(posedge ...) is checked one step late, one
in always @(*) (or a bare module-level assert property) at the same
step. So all-clocked designs use depth 3 skip 2, all-same-cycle designs
depth 2 skip 1. A design with both kinds needs both timings. At depth 3
skip 2 alone, a same-cycle assertion breaking right after the step falls
in a skipped step and is assumed away, so a real fake state for a
same-cycle property would always be refused. So first a depth 2 skip 1
run with the clocked asserts turned into assumes, kept one step late
with `assume_early off` (they still hold of the first state, nothing
else changes); only if that finds no break, a depth
3 skip 2 run where only a clocked failure counts. A same-cycle failure
there is two steps away: NOT_ONE_STEP.

When the assumptions cannot all hold, one cover run on the design with
its assertions removed says which part of the claim was impossible.

Unknown names are refused before running anything: yosys quietly makes
a free wire for a name it does not know, which would let any claim pass.

"Real" means real to the proof tool, with its timing (decided 4 Oct
2026). 40 corpus rows assert only about past values (`$past(x)` under a
past-valid register); there the tool's step is one later than a reader
would count, and the tool's count is the one the proof uses.

  check_state(row, state) -> {ok, verdict, detail, style, wall_s}
"""
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

from oracle.inject import (InjectionError, _insert_point,   # noqa: E402
                           _mask_comments, strip_assertions)
from extend import out_of_scope                           # noqa: E402

VERDICTS = ("REAL", "NO_BREAK", "STATE_IMPOSSIBLE", "R_TRUE", "P_FALSE",
            "NOT_ONE_STEP", "UNKNOWN_SIGNAL", "BAD_STATE", "ERROR",
            "TIMEOUT")

STEPS = {"clocked": (3, 2), "comb": (2, 1)}     # depth, skip
CHECK_ENGINE = "smtbmc yices -- --assume-skipped 0"
COVER_ENGINE = "smtbmc yices"
OUTER_GUARD_FACTOR = 1.5     # same layering as oracle/sby.py
OUTER_GUARD_GRACE_S = 10

_SIGNAL = re.compile(r"[A-Za-z_]\w*(\s*\[[\w\s:+\-']+\])*")
_PATH = re.compile(r"[A-Za-z_]\w*(\s*\[[^\]]*\])*(\s*\.\s*[A-Za-z_]\w*"
                   r"(\s*\[[^\]]*\])*)+")
_NUMBER = re.compile(r"(\d[\d_]*\s*)?'[sS]?([bB]\s*[01_]+|[oO]\s*[0-7_]+"
                     r"|[dD]\s*[\d_]+|[hH]\s*[\da-fA-F_]+)|\d[\d_]*")
_ALWAYS = re.compile(r"\balways(_ff|_comb|_latch)?\b")
_WORD = re.compile(r"[A-Za-z_]\w*")
_FAILED = re.compile(r"Assert failed in \S+: [^\s:]+:(\d+)(?:\.(\d+))?"
                     r"(?:-(\d+)(?:\.(\d+))?)?")
_COVER = re.compile(r"\b(un)?reached cover statement\b[^\n]*?\.sv:(\d+)",
                    re.I)


# --- which assertions are clocked ---

def _skip_ws(text, i):
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _close_paren(text, i):
    """Index of the ')' matching the '(' at i."""
    if i >= len(text) or text[i] != "(":
        raise ValueError("expected (")
    depth = 0
    for k in range(i, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError("unbalanced (")


def _match_close(text, i, opener, closer):
    """Index just past the closer word matching the opener word at i."""
    depth = 0
    for m in re.compile(r"\b(%s|%s)\b" % (opener, closer)).finditer(text, i):
        depth += -1 if m.group() == closer else 1
        if depth == 0:
            return m.end()
    raise ValueError(f"no matching {closer}")


def _semicolon(text, i):
    depth = 0
    for k in range(i, len(text)):
        c = text[k]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return k + 1
    raise ValueError("no ;")


def _stmt_end(text, i):
    """Index just past the one statement starting at i (an always body)."""
    i = _skip_ws(text, i)
    m = _WORD.match(text, i)
    word = m.group() if m else ""
    if word == "begin":
        return _match_close(text, i, "begin", "end")
    if word in ("case", "casez", "casex"):
        return _match_close(text, i, "case[zx]?", "endcase")
    if word in ("if", "for", "while", "repeat"):
        end = _stmt_end(text, _close_paren(text, _skip_ws(text, m.end())) + 1)
        if word == "if":
            k = _skip_ws(text, end)
            if re.match(r"else\b", text[k:k + 5]):
                return _stmt_end(text, k + 4)
        return end
    if word == "forever":
        return _stmt_end(text, m.end())
    return _semicolon(text, i)


def _always_spans(masked):
    """(start, end, clocked) for every always block."""
    heads = list(_ALWAYS.finditer(masked))
    spans = []
    for n, m in enumerate(heads):
        i = _skip_ws(masked, m.end())
        sens = ""
        try:
            if masked.startswith("@", i):
                i = _skip_ws(masked, i + 1)
                if masked.startswith("(", i):
                    j = _close_paren(masked, i)
                    sens, i = masked[i:j + 1], j + 1
                elif masked.startswith("*", i):
                    i += 1
            end = _stmt_end(masked, i)
        except ValueError:
            # cannot parse the body: it runs to the next always or endmodule
            nxt = re.compile(r"\bendmodule\b").search(masked, i)
            end = min([h.start() for h in heads[n + 1:n + 2]]
                      + [nxt.start() if nxt else len(masked)])
        clocked = (m.group() == "always_ff"
                   or bool(re.search(r"\b(posedge|negedge)\b", sens)))
        spans.append((m.start(), end, clocked))
    return spans


def _assert_sites(source):
    """(line, column, style, offset) of every assert keyword; line and
    column are 1-based like yosys's source locations."""
    masked = _mask_comments(source)
    spans = _always_spans(masked)
    sites = []
    for m in re.finditer(r"\bassert\b", masked):
        pos = m.start()
        inside = [s for s in spans if s[0] < pos < s[1]]
        # a bare module-level assert property is checked every step
        style = "clocked" if inside and inside[-1][2] else "comb"
        line = masked.count("\n", 0, pos) + 1
        col = pos - (masked.rfind("\n", 0, pos) + 1) + 1
        sites.append((line, col, style, pos))
    return sites


def assert_styles(source):
    """{line of each assert keyword: "clocked" or "comb"}."""
    return {line: style for line, _, style, _ in _assert_sites(source)}


def design_style(source):
    styles = set(assert_styles(source).values())
    if not styles:
        return "none"
    return "mixed" if len(styles) == 2 else styles.pop()


def _failed_spans(log):
    out = []
    for l1, c1, l2, c2 in _FAILED.findall(log):
        l1, c1 = int(l1), int(c1 or 0)
        out.append(((l1, c1), (int(l2 or l1), int(c2 or 10**6))))
    return out


def failed_lines(log):
    """(first line, last line) of every failing assertion in an sby log."""
    return [(a[0], b[0]) for a, b in _failed_spans(log)]


def _failing_sites(text, log):
    """Assert sites that the log's failures point at. A module-level
    assert property's span starts where the previous item ended, so
    sites are matched by position, not line alone."""
    sites = _assert_sites(text)
    hits = []
    for a, b in _failed_spans(log):
        found = [s for s in sites if a <= (s[0], s[1]) <= b]
        hits.extend(found or [s for s in sites if a[0] <= s[0] <= b[0]])
    return hits


def _assert_text(text, pos, limit=120):
    end = text.find(";", pos)
    snippet = " ".join(text[pos:end + 1 if end != -1 else pos + limit].split())
    return snippet if len(snippet) <= limit else snippet[:limit - 3] + "..."


# --- checking the claimed state before running ---

def _top_text(verilog, top_module):
    m = re.search(r"\bmodule\s+%s\b(.*?)\bendmodule\b"
                  % re.escape(top_module), _mask_comments(verilog), re.S)
    return m.group(1) if m else ""


def _param_names(top_text):
    names = set()
    for m in re.finditer(r"\b(?:localparam|parameter)\b([^;]*)", top_text):
        names |= set(re.findall(r"([A-Za-z_]\w*)\s*=(?!=)", m.group(1)))
    return names


def _too_big(val):
    """The stated width when a sized constant's digits do not fit in it.
    Verilog quietly drops the extra high bits, so 4'd20 would be checked
    as 4'd4, a different number from the one in the reasoning."""
    m = re.fullmatch(r"(\d[\d_]*)\s*'[sS]?([bBoOdDhH])\s*([\da-fA-F_]+)", val)
    if not m or not m.group(3).strip("_"):
        return None
    width = int(m.group(1).replace("_", ""))
    base = {"b": 2, "o": 8, "d": 10, "h": 16}[m.group(2).lower()]
    return width if int(m.group(3).replace("_", ""), base) >> width else None


def _refuse(state, verilog, top_module):
    """(verdict, detail) when the state cannot be run, else None."""
    if not isinstance(state, list) or not state:
        return ("BAD_STATE", "The fake state must be a non-empty list of "
                "{signal, value} pairs.")
    for n, e in enumerate(state, 1):
        if not (isinstance(e, dict)
                and all(isinstance(e.get(k), str) and e[k].strip()
                        for k in ("signal", "value"))):
            return ("BAD_STATE", f"Entry {n} of the fake state needs a "
                    "non-empty text signal and a non-empty text value.")
    params = _param_names(_top_text(verilog, top_module))
    names = []
    for e in state:
        sig, val = e["signal"].strip(), e["value"].strip()
        if _PATH.fullmatch(sig):
            return ("UNKNOWN_SIGNAL", f"'{sig}' is inside a sub-module; "
                    f"only signals of the top module {top_module} can be "
                    "named.")
        if not _SIGNAL.fullmatch(sig):
            return ("BAD_STATE", f"'{sig}' is not a plain signal name. Give "
                    f"only a name declared in {top_module}, like count or "
                    "mem[1], with no notes and no $past.")
        if not (_NUMBER.fullmatch(val) or val in params):
            return ("BAD_STATE", f"The value of {sig} must be a plain "
                    f"constant like 4'd3, with no notes (got '{val}').")
        if _too_big(val):
            return ("BAD_STATE", f"The value of {sig}, {val}, does not fit "
                    f"in {_too_big(val)} bits. Give the number with a size "
                    "large enough for it.")
        names.append(sig)
    unknown = sorted(out_of_scope(names, verilog, top_module))
    if unknown:
        return ("UNKNOWN_SIGNAL", f"Not signals of the top module "
                f"{top_module}: {', '.join(unknown)}. Name registers, wires "
                "or inputs exactly as declared there.")
    return None


# --- building and running the jobs ---

def _conj(invariants):
    return " && ".join(f"({r})" for r in invariants)


def pin_block(state, invariants):
    """The claimed fake state and !R, assumed at the first step only."""
    lines = ["// cti_check: the claimed fake state, pinned at the first step",
             "always @(*) begin",
             "    if ($initstate) begin"]
    lines += [f"        assume ({e['signal'].strip()} == "
              f"{e['value'].strip()});" for e in state]
    lines += [f"        assume (!({_conj(invariants)}));", "    end", "end"]
    return "\n".join(lines) + "\n"


def _cover_block(state, invariants):
    pins = " && ".join(f"({e['signal'].strip()} == {e['value'].strip()})"
                       for e in state)
    return "\n".join([
        "// cti_check: which part of the claim cannot hold",
        "always @(*) begin",
        "    if ($initstate) begin",
        f"        cover ({pins});",
        f"        cover ({pins} && !({_conj(invariants)}));",
        "    end",
        "end"]) + "\n"


def _inject(source, top_module, block):
    """(new text, line number of the block's first line)."""
    at = _insert_point(source, top_module)
    indented = "".join("    " + ln + "\n" for ln in block.splitlines())
    return (source[:at] + indented + source[at:],
            source[:at].count("\n") + 1)


def _sby_text(top_module, mode, depth, skip, timeout_s, engine,
              late_assumes=False):
    return "\n".join(
        ["[options]", f"mode {mode}", f"depth {depth}"]
        + ([f"skip {skip}"] if skip else [])
        + (["assume_early off"] if late_assumes else [])
        + [f"timeout {timeout_s}", "",
           "[engines]", engine, "",
           "[script]", "read -formal design.sv", f"prep -top {top_module}",
           "memory_map", "attrmap -remove init", "",
           "[files]", "design.sv", ""])


def _run_sby(rundir, text, sby_text, timeout_s):
    """(rc, log). rc is None when the outer guard killed a hung sby."""
    rundir.mkdir(parents=True)
    (rundir / "design.sv").write_text(text)
    (rundir / "job.sby").write_text(sby_text)
    try:
        proc = subprocess.run(
            ["sby", "-f", "job.sby"], cwd=rundir,
            capture_output=True, text=True,
            timeout=OUTER_GUARD_FACTOR * timeout_s + OUTER_GUARD_GRACE_S)
    except subprocess.TimeoutExpired as e:
        # keep what a hung run printed (8 Oct 2026)
        for name, x in (("sby_stdout.txt", e.stdout), ("sby_stderr.txt", e.stderr)):
            (rundir / name).write_text(
                x.decode(errors="replace") if isinstance(x, bytes) else x or "")
        return None, ""
    (rundir / "sby_stdout.txt").write_text(proc.stdout or "")
    (rundir / "sby_stderr.txt").write_text(proc.stderr or "")
    logfile = rundir / "job" / "logfile.txt"
    log = logfile.read_text() if logfile.exists() else ""
    return proc.returncode, log or (proc.stdout or "") + (proc.stderr or "")


def _first_error(log, rc):
    for line in log.splitlines():
        if "ERROR" in line:
            return re.sub(r"^SBY \S+ \[[^\]]*\] ", "", line).strip()
    return f"sby exited with code {rc} and no ERROR line in its log"


def _timed_out(rc, log):
    return rc is None or "DONE (TIMEOUT" in log


def _diagnose(rundir, row, state, timeout_s):
    """The pinned run had no solution at all: find out which part."""
    stripped = strip_assertions(row["verilog"])
    text, first = _inject(stripped, row["top_module"],
                          _cover_block(state, row["invariants"]))
    pins_line, not_r_line = first + 3, first + 4
    rc, log = _run_sby(rundir, text,
                       _sby_text(row["top_module"], "cover", 1, 0, timeout_s,
                                 COVER_ENGINE), timeout_s)
    if _timed_out(rc, log):
        return ("TIMEOUT", f"The follow-up check did not finish within "
                f"{timeout_s} seconds.")
    reached, unreached = set(), set()
    for un, line in _COVER.findall(log):
        (unreached if un else reached).add(int(line))
    if pins_line in unreached:
        return ("STATE_IMPOSSIBLE", "These values cannot all hold at once: "
                "they contradict each other or the design's own wiring "
                "(for example a wire given a value its inputs cannot "
                "produce, or two values for one signal).")
    if pins_line in reached and not_r_line in unreached:
        return ("R_TRUE", "Every state with these values makes all the "
                "invariants true, so it does not show why they are needed. "
                "A fake state must make at least one invariant false.")
    if pins_line in reached and not_r_line in reached:
        return ("P_FALSE", "The property is already false in this state, "
                "so the induction step never starts from it. A fake state "
                "must satisfy every assertion and break one only after a "
                "clock step.")
    return "ERROR", "Follow-up check failed: " + _first_error(log, rc)


def _clocked_as_assumes(text):
    """The design with every clocked assert turned into an assume. Both
    words are six letters, so every line and column stays put."""
    for _, _, style, pos in _assert_sites(text):
        if style == "clocked":
            text = text[:pos] + "assume" + text[pos + 6:]
    return text


def _bmc(rundir, text, top_module, style, timeout_s, late_assumes=False):
    depth, skip = STEPS[style]
    return _run_sby(rundir, text,
                    _sby_text(top_module, "bmc", depth, skip, timeout_s,
                              CHECK_ENGINE, late_assumes), timeout_s)


def _decide(rundir, row, state, style, timeout_s):
    top = row["top_module"]
    text, _ = _inject(row["verilog"], top,
                      pin_block(state, row["invariants"]))
    same_cycle_late = False
    if style == "mixed":
        # same-cycle asserts first, checked right after the step; the
        # clocked ones become assumptions so they still hold of the first
        # state. sby makes a clocked assume act at once by default, which
        # would also demand it of the state after the step, so it is kept
        # one step late like the assert it was. Only if nothing breaks,
        # the clocked ones, one step late.
        assumed = _clocked_as_assumes(text)
        rc, log = _bmc(rundir / "same_cycle", assumed, top, "comb",
                       timeout_s, late_assumes=True)
        if "DONE (PASS" in log:
            rc, log = _bmc(rundir / "check", text, top, "clocked", timeout_s)
            same_cycle_late = True
        else:
            text = assumed
    else:
        rc, log = _bmc(rundir / "check", text, top, style, timeout_s)

    if _timed_out(rc, log):
        return ("TIMEOUT", f"The proof tool did not finish within "
                f"{timeout_s} seconds.")
    if "Assumptions are unsatisfiable" in log:
        return _diagnose(rundir / "cover", row, state, timeout_s)
    if "DONE (PASS" in log:
        return ("NO_BREAK", "With these values no single clock step can "
                "break the property, whatever the other signals and inputs "
                "are. A fake state must let one step reach a failing "
                "assertion. Any input you named is held at that value on "
                "the step, so an enable named at its idle value makes the "
                "step do nothing.")
    if "DONE (FAIL" in log and _failed_spans(log):
        hits = _failing_sites(text, log)
        if same_cycle_late:
            if not hits:
                return ("ERROR", "Could not tell which assertion failed: "
                        f"lines {failed_lines(log)}.")
            hits = [h for h in hits if h[2] == "clocked"]
            if not hits:
                return ("NOT_ONE_STEP", "The property breaks only after "
                        "two clock steps, not after one.")
        which = f": {_assert_text(text, hits[0][3])}" if hits else ""
        return ("REAL", "Confirmed: with these values the invariants are "
                "false and the property holds, and one clock step breaks "
                f"the assertion{which}")
    return "ERROR", "The proof tool failed: " + _first_error(log, rc)


def check_state(row, state, timeout_s=120, keep_dir=None):
    """Is `state` (a list of {signal, value}) a real fake state for `row`
    (verilog, top_module, invariants)? A fresh directory per call, so it
    is safe from several threads; removed afterwards unless keep_dir."""
    start = time.monotonic()
    style = design_style(row["verilog"])

    folder = f"{row.get('id', 'row')}_cti_{uuid.uuid4().hex[:8]}"

    def done(verdict, detail):
        out = {"ok": verdict == "REAL", "verdict": verdict,
               "detail": " ".join(detail.split()), "style": style,
               "wall_s": round(time.monotonic() - start, 3)}
        if keep_dir and (Path(keep_dir) / folder).exists():
            out["proof_dir"] = folder     # under keep_dir, kept
        return out

    refused = _refuse(state, row["verilog"], row["top_module"])
    if refused:
        return done(*refused)
    if not row.get("invariants"):
        return done("ERROR", "This row has no invariants, so no state can "
                    "make them false.")
    if style == "none":
        return done("ERROR", "The design has no assertions, so nothing can "
                    "break.")
    if shutil.which("sby") is None:
        return done("ERROR", "sby is not on PATH; source "
                    "oss-cad-suite/environment first.")
    if keep_dir:
        root = Path(keep_dir)
    else:
        root = Path(tempfile.mkdtemp(prefix="cti_check_"))
    try:
        return done(*_decide(root / folder, row, state, style, timeout_s))
    except InjectionError as e:
        return done("ERROR", str(e))
    finally:
        if not keep_dir:
            shutil.rmtree(root, ignore_errors=True)
