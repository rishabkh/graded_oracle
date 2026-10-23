"""Extender, distractor extension: one seed, one call.

The distractor is the control condition. It is the only extension type
whose expected outcome is fully known in advance: the added logic is
irrelevant to the property, so the parent's invariants stay verbatim and
the verdict must still be NECESSARY. Anything else means the machinery
(patch applier, liveness check, re-grade plumbing) is at fault, not the
model.

Liveness is measured, not trusted: yosys `opt` sweeps unconsumed logic,
so the added state must survive into the child's flop count —
state_bits(child) > state_bits(parent), or the logic was dead.

  venv/bin/python extender/distractor.py --dry       # print the prompt, no calls
  venv/bin/python extender/distractor.py --selftest  # hand patch through the full
                                                     # apply->liveness->grade path,
                                                     # no API (needs hwtools)
  venv/bin/python extender/distractor.py             # ONE API call
                                                     # (needs ANTHROPIC_API_KEY + hwtools)
"""
import argparse
import itertools
import json
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import extract_asserts, state_bits   # noqa: E402
from patch import PatchError, apply_patch              # noqa: E402

import llm_client                            # noqa: E402
from oracle import grade_triple_generated    # noqa: E402
from reset_reads import step_reset_error     # noqa: E402

CORPUS = HERE / "corpus.jsonl"
OUT_LOG = HERE / "logs" / "extensions.jsonl"
MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 64000       # was 16000; bigger designs (8 Oct 2026) need room
# Bigger designs (8 Oct 2026): 300 s per proof run (was 120); the pdr
# second opinion, the sanity check and the deeper cover look keep 120 s
# (none of them can make a design pass); conditions unreached in 20 steps
# get a deeper look first (oracle/depth.py).
GRADE_KWARGS = dict(timeout_s=300, pdr_timeout_s=120, cover_depth="auto")
# 21 Oct 2026 (Rishab: "make all 1-8 changes"): 9 of the 40 steps of the
# real-systems trial were lost to the 300 s limit. Checked again by hand
# with 1200 s, 2 of the 9 proved, and their slowest proof (the one without
# facts) took 539 s and 625 s. A step whose verdict came from the time
# limit is now checked once more with this limit before it is routed
# (necessity_check). Only the main limit grows: the pdr second opinion,
# the sanity check and the deeper cover look cannot make a design pass.
# 0 turns the second check off (batch.py --recheck-timeout-s).
RECHECK_TIMEOUT_S = 900
# Where proofs go. None keeps the shared runs/ folder; the batch runner
# points it at <log-dir>/proofs so a run's proofs travel with its logs.
PROOF_ROOT = None


def grade_kwargs(record, leg="", timeout_s=None):
    """GRADE_KWARGS, plus this record's own proof folder when PROOF_ROOT is
    set. The record keeps the folder as "proof_dir", relative to the log
    folder; every check for one record lands under it (`leg` names extra
    checks, e.g. "p2_alone" or "recheck"; the fixer keeps its own folder
    next to its own log since 22 Oct 2026, fix.proof_folder). `timeout_s`
    sets the time limit of this one check (the second check of a step that
    ran out of time) and leaves GRADE_KWARGS, which every worker reads,
    alone."""
    kw = dict(GRADE_KWARGS)
    if timeout_s:
        kw["timeout_s"] = timeout_s
    if PROOF_ROOT is None:
        return kw
    if "proof_dir" not in record:
        import uuid
        record["proof_dir"] = (f"proofs/{record.get('extension_id', 'x')}_"
                               f"{record.get('parent_id', '')}_"
                               f"{uuid.uuid4().hex[:6]}")
    folder = Path(PROOF_ROOT) / Path(record["proof_dir"]).name
    kw["workdir_root"] = folder / leg if leg else folder
    return kw

DIFF_FORMAT = """\
One directive per line:
  @@ line @@   anchor: consecutive anchor lines must match consecutive lines in
               the file; the edit cursor moves just past the last matched line.
  = line       keep: find this line forward of the cursor, move cursor past it.
  - line       delete: find this line forward of the cursor and delete it.
  + line       add: insert this line at the cursor.
Matching ignores leading/trailing whitespace but is otherwise exact. Every
anchor, = and - line MUST exist in the file (forward of the cursor) or the
patch is rejected whole — there are no fuzzy matches.
An anchor must be a line ABOVE the edit and must not itself be modified:
after an anchor matches, the cursor sits past it, so a - or = directive can
only reach lines below it. To replace a line, anchor on the line above it,
then - the target and + its replacement.

Example. Applying
@@ a @@
+ inserted-after-a
@@ b cd @@
- line2
= line3
+ after3
to "hello / world / a / b cd / line2 / line3" yields
"hello / world / a / inserted-after-a / b cd / line3 / after3"."""

DISTRACTOR_PROMPT = """\
You are extending a formally verified SystemVerilog module. The module's \
property is proven by k-induction with the strengthening invariant(s) listed \
below. Your task is the DISTRACTOR extension: add a realistic block of new \
sequential logic that is completely irrelevant to the property, so the proof \
still closes with the SAME invariants, unchanged. Make it the kind of feature \
real modules carry that this property never looks at, for example status and \
statistics counters, an error logger, a watchdog timer or a debug register bank.

Size: the module now has {n} lines; add roughly {lo} to {hi} new lines \
({share}), written out as real logic, not padding, comments or copies of \
existing lines.

The module:

{verilog}

The property (assertion), which you must not touch:
{property}

The strengthening invariants, which are kept verbatim and re-verified after \
your patch (your logic must not write any signal they or the property read):
{invariants}

Rules:
1. The new logic must be LIVE: it must drive a NEW OUTPUT PORT. Dead logic is
   optimised away by yosys and counts as failure. Adding a port is a two-step
   edit — the previous last port line needs a trailing comma, so replace it
   and insert the new port after it, anchored on the line above it:

   @@ <second-to-last port line, copied exactly> @@
   - <last port line, copied exactly>
   + <last port line with a comma appended>
   + output reg [3:0] your_new_port
2. Add at least one new register. Match the module's own reset: it resets the
   same way, in the same kind of block, as the module's existing registers,
   and gets an `initial` like the existing ones.
3. Do not modify, move, or delete the assertion; add no assert/assume/cover.
4. Do not change how any existing register or wire is computed. You may READ
   existing signals; you must not write them.
5. Same Verilog subset as the module: synthesizable clocked logic written like
   the module's own (the same clock edge and the same reset), no new always
   blocks containing assertions, no system tasks.
6. Emit a PATCH in the diff format below — never a rewritten module.

Diff format:
{diff_format}

Reply with JSON: {{"note": "<why this logic is irrelevant to the \
property and how it stays live>", "patch": "<the diff>"}}"""

# 22 Oct 2026, review finding F12: rules 2 and 5 asked for "a reset in the
# existing if (rst) branch" and "always @(posedge clk) logic"; since 21 Oct
# one new design in five resets at once (always @(posedge clk or negedge
# rst_n), if (!rst_n)), so they now ask for the module's own reset.

# The rationale field is called "note", not "reasoning": a strict schema
# demanding a field of that exact name is refused before it reaches the
# model (probed 21 Sep 2026 - blocked with "reasoning", accepted with
# "note", same prompt, same route, thinking on or off). The initiator's
# "cti_reasoning" has never been refused in 105 calls, so it is the exact
# name that matters.
EXT_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string"},
        "patch": {"type": "string"},
    },
    "required": ["note", "patch"],
    "additionalProperties": False,
}

# The selftest patch: a hand-written distractor for token_bucket — a spend
# counter driving a new output port (live by construction). Everything the
# model is asked to do, done by hand, so the machinery is proven before any
# API money is spent. Uses the same anchored port-edit shape the prompt
# teaches, so control and instruction agree.
SELFTEST_PATCH = """\
@@ output reg [2:0] tokens, @@
- output reg tokens_hi
+ output reg tokens_hi,
+ output reg [3:0] spent_total
@@ initial tokens_hi = 1'b0; @@
+     initial spent_total = 4'd0;
@@ if (rst) begin @@
@@ tokens    <= 3'd0; @@
@@ tokens_hi <= 1'b0; @@
+             spent_total <= 4'd0;
@@ end else begin @@
+             if (dec) spent_total <= spent_total + 4'd1;"""


class Spinner:
    """Braille-dot spinner with colour and a mm:ss clock. Silent when
    stderr is not a TTY (logs and pipes stay clean). Copied rather than
    imported so each worker stays standalone."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CYAN, DIM, RESET = "\033[36m", "\033[2m", "\033[0m"

    quiet = False   # batch sets this with --workers > 1

    def __init__(self, label, always=False):
        self.label = label
        self.always = always   # draw even when quiet (the pool spinner)
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        start = time.monotonic()
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            elapsed = int(time.monotonic() - start)
            clock = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
            import shutil as _sh
            width = _sh.get_terminal_size().columns
            label = self.label[:max(10, width - 12)]
            sys.stderr.write(f"\r{self.CYAN}{frame}{self.RESET} {label} "
                             f"{self.DIM}{clock}{self.RESET} ")
            sys.stderr.flush()
            self._stop.wait(0.08)
        sys.stderr.write("\r" + " " * (len(self.label) + 12) + "\r")
        sys.stderr.flush()

    def __enter__(self):
        if sys.stderr.isatty() and (self.always or not Spinner.quiet):
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()


def load_parent(parent_id):
    for line in CORPUS.read_text().splitlines():
        row = json.loads(line)
        if row["id"] == parent_id:
            return row
    sys.exit(f"parent {parent_id!r} not in {CORPUS.name}")


def build_prompt(parent):
    # the same size rule as the big moves: half to three quarters, and 200
    # to 300 new lines once a design is over 300 lines (21 Oct 2026). Read
    # from extend.py, so the two cannot drift apart; imported here, not at
    # the top, because extend.py imports this module.
    from extend import size_share, size_target
    n, lo, hi = size_target(parent["verilog"])  # copies counted once (19 Oct)
    return DISTRACTOR_PROMPT.format(
        verilog=parent["verilog"],
        property="\n".join(parent["property"]),
        # as written, when compact facts were used (21 Oct 2026)
        invariants="\n".join(parent.get("invariants_compact")
                              or parent["invariants"]),
        diff_format=DIFF_FORMAT, n=n, lo=lo, hi=hi, share=size_share(None, n))


def call_model(prompt_text):
    text, usage, stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, user=prompt_text,
        schema=EXT_SCHEMA, effort=EFFORT)
    return (None if text is None else json.loads(text)), usage, stop


def ran_out_of_time(result):
    """Did this verdict of the checker come from its time limit? The proof
    with the facts ran out of time (NOT_PROVEN), or the proof without them
    did (INCONCLUSIVE). A false property, a proof that did not close, a
    tool failure or a format error did not. Nor did a proof whose
    induction step had already failed when the time ran out (the checker's
    induction_failed_before_timeout note): more time cannot make it close."""
    leg = {"NOT_PROVEN": result.with_invariants,
           "INCONCLUSIVE": result.without_invariants}.get(result.verdict.name)
    if leg is None or leg.tier.name != "TIMEOUT":
        return False
    return not any(str(note).startswith("induction_failed_before_timeout")
                   for run in leg.runs for note in run.notes)


def _timed_check(check, payload, label, kwargs):
    t0 = time.monotonic()
    with Spinner(label):
        result = check(json.dumps(payload), **kwargs)
    return result, round(time.monotonic() - t0, 2)


def _put_verdict(record, result, wall_s):
    record["grade_wall_s"] = wall_s
    record["verdict"] = result.verdict.name
    record["reason"] = result.reason
    record["result"] = asdict(result)


def necessity_check(check, payload, record, label):
    """The checker's verdict on a step, put on its record (grade_wall_s,
    verdict, reason, result). `check` is the caller's own
    grade_triple_generated, so a test that fakes it there fakes both checks.

    A verdict that came from the time limit (ran_out_of_time) is checked
    once more, never twice, with RECHECK_TIMEOUT_S, its proofs in the
    "recheck" folder of the record's own proof folder (21 Oct 2026). The
    second verdict replaces the first; the first check is kept whole as
    "first_result", and the record says "rechecked", the limit used
    ("recheck_timeout_s") and how long the second check took
    ("recheck_wall_s"). Returns the result the record now holds."""
    result, wall_s = _timed_check(check, payload, label, grade_kwargs(record))
    _put_verdict(record, result, wall_s)
    if RECHECK_TIMEOUT_S and not record.get("rechecked") \
            and ran_out_of_time(result):
        record["first_result"] = {k: record[k] for k in
                                  ("verdict", "reason", "grade_wall_s",
                                   "result")}
        record["rechecked"] = True
        record["recheck_timeout_s"] = RECHECK_TIMEOUT_S
        result, wall_s = _timed_check(
            check, payload,
            f"{label}: checking again with a {RECHECK_TIMEOUT_S} s limit",
            grade_kwargs(record, leg="recheck", timeout_s=RECHECK_TIMEOUT_S))
        record["recheck_wall_s"] = wall_s
        _put_verdict(record, result, wall_s)
    return result


def grade_extension(parent, patch_text, record):
    """Apply -> liveness -> re-grade with the parent's invariants verbatim.
    Fills `record` in place; returns the record."""
    record["patch"] = patch_text
    try:
        child_verilog = apply_patch(parent["verilog"], patch_text)
    except PatchError as exc:
        record["verdict"] = "PATCH_ERROR"
        record["error"] = str(exc)
        return record
    record["child_verilog"] = child_verilog

    # The property must survive verbatim: a patch that deletes the assert
    # and writes a different one would re-grade NECESSARY on the wrong
    # property and log a false success.
    norm = lambda xs: [" ".join(x.split()) for x in xs]  # noqa: E731
    child_asserts = norm(extract_asserts(child_verilog))
    parent_asserts = norm(parent["property"])
    if child_asserts != parent_asserts:
        record["verdict"] = "PROPERTY_CHANGED"
        record["error"] = (f"assertions changed: parent {parent_asserts} "
                           f"vs child {child_asserts}")
        return record

    # The check above reads only what is inside each assert, so a patch
    # could still put a guard that reads an asynchronous reset around the
    # property, or make a signal the facts read one; our checker and EBMC
    # can then disagree (23 Oct 2026, review finding F4: every step of the
    # extender is now asked, extend.py step_reads_reset).
    clock = parent.get("clock") or "clk"
    error = step_reset_error(
        child_verilog, clock, parent.get("invariants") or [],
        parent.get("antecedents") or [],
        [(parent["verilog"], clock, parent.get("invariants") or [],
          parent.get("antecedents") or [])])
    if error:
        record["verdict"] = "READS_RESET"
        record["error"] = error
        return record

    parent_bits = state_bits(parent["verilog"], parent["top_module"])
    record["parent_state_bits"] = parent_bits
    if parent_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = ("parent does not read in yosys - machinery "
                           "fault, not the model's patch")
        return record
    child_bits = state_bits(child_verilog, parent["top_module"])
    record["child_state_bits"] = child_bits
    if child_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = "child does not read in yosys"
        return record
    record["live"] = child_bits > parent_bits

    payload = {"verilog": child_verilog,
               "top_module": parent["top_module"], "clock": parent["clock"],
               "antecedents": parent["antecedents"],
               "sanity_covers": parent["sanity_covers"],
               "invariants": parent["invariants"]}   # verbatim — the control
    necessity_check(grade_triple_generated, payload, record,
                    f"oracle re-grading {parent['top_module']}")
    return record


def dump(record):
    OUT_LOG.parent.mkdir(exist_ok=True)
    with OUT_LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def report(record):
    print(f"\npatch:\n{record.get('patch', '<none>')}\n")
    print(f"live: {record.get('live')} "
          f"(state_bits {record.get('parent_state_bits')} -> "
          f"{record.get('child_state_bits')})")
    print(f"verdict: {record['verdict']} - {record.get('reason', record.get('error'))}")
    expected = (record["verdict"] == "NECESSARY" and record.get("live"))
    print("CONTROL " + ("PASSED: invariants unchanged, still NECESSARY, logic live"
                        if expected else
                        "FAILED: expected live + NECESSARY with unchanged invariants"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", default="g0_014",
                   help="corpus id of the seed (default: token_bucket)")
    p.add_argument("--dry", action="store_true", help="print the prompt, no calls")
    p.add_argument("--selftest", action="store_true",
                   help="run the hand-written patch through the full path, no API")
    args = p.parse_args()

    parent = load_parent(args.parent)
    print(f"parent {parent['id']} {parent['top_module']} "
          f"invariants={parent['invariants']}")

    if args.dry:
        print("\n" + build_prompt(parent))
        return

    record = {"extension_id": datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
              "ext_type": "distractor", "parent_id": parent["id"],
              "invariants": parent["invariants"]}

    if args.selftest:
        record["model"] = "selftest (hand-written patch)"
        grade_extension(parent, SELFTEST_PATCH, record)
    else:
        record["model"] = llm_client.model_label(MODEL)
        record["effort"] = EFFORT
        with Spinner(f"{llm_client.model_label(MODEL)} writing a distractor patch"):
            out, usage, stop = call_model(build_prompt(parent))
        record["usage"] = usage
        if out is None:
            record["verdict"] = {"refusal": "REFUSED",
                                 "length": "TRUNCATED"}.get(stop, "UNPARSEABLE")
            record["raw_text"] = llm_client.LAST_RAW
            dump(record)
            print(f"{record['verdict'].lower()} - logged (raw text kept)")
            return
        record["reasoning"] = out["note"]      # kept under the old key
        grade_extension(parent, out["patch"], record)

    dump(record)
    report(record)


if __name__ == "__main__":
    main()
