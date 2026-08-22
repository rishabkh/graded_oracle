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

  venv/bin/python extender/extend_one.py --dry       # print the prompt, no calls
  venv/bin/python extender/extend_one.py --selftest  # hand patch through the full
                                                     # apply->liveness->grade path,
                                                     # no API (needs hwtools)
  venv/bin/python extender/extend_one.py             # ONE API call
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

from oracle import grade_triple_generated    # noqa: E402

CORPUS = HERE / "corpus.jsonl"
OUT_LOG = HERE / "logs" / "extensions.jsonl"
MODEL = "claude-opus-5"
EFFORT = "high"
MAX_TOKENS = 16000
GRADE_KWARGS = dict(timeout_s=120)

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
below. Your task is the DISTRACTOR extension: add a small piece of new \
sequential logic that is completely irrelevant to the property, so the proof \
still closes with the SAME invariants, unchanged.

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
2. Add at least one new register, with a reset in the existing if (rst) branch
   and an `initial` like the existing ones.
3. Do not modify, move, or delete the assertion; add no assert/assume/cover.
4. Do not change how any existing register or wire is computed. You may READ
   existing signals; you must not write them.
5. Same Verilog subset as the module: synthesizable always @(posedge clk)
   logic, no new always blocks containing assertions, no system tasks.
6. Emit a PATCH in the diff format below — never a rewritten module.

Diff format:
{diff_format}

Reply with JSON: {{"reasoning": "<why this logic is irrelevant to the \
property and how it stays live>", "patch": "<the diff>"}}"""

EXT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "patch": {"type": "string"},
    },
    "required": ["reasoning", "patch"],
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
    """Same spinner as initiator/run.py, copied to keep workers standalone."""
    FRAMES = "|/-\\"

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        start = time.monotonic()
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            elapsed = time.monotonic() - start
            sys.stderr.write(f"\r{frame} {self.label} ({elapsed:.0f}s) ")
            sys.stderr.flush()
            self._stop.wait(0.2)
        sys.stderr.write("\r" + " " * (len(self.label) + 12) + "\r")
        sys.stderr.flush()

    def __enter__(self):
        if sys.stderr.isatty():
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
    return DISTRACTOR_PROMPT.format(
        verilog=parent["verilog"],
        property="\n".join(parent["property"]),
        invariants="\n".join(parent["invariants"]),
        diff_format=DIFF_FORMAT)


def call_model(prompt_text):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT,
                       "format": {"type": "json_schema", "schema": EXT_SCHEMA}},
        messages=[{"role": "user", "content": prompt_text}],
    )
    if resp.stop_reason == "refusal":
        return None, resp
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp


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

    parent_bits = state_bits(parent["verilog"], parent["top_module"])
    record["parent_state_bits"] = parent_bits
    if parent_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = ("parent does not read in yosys — machinery "
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
    t0 = time.monotonic()
    with Spinner(f"oracle re-grading {parent['top_module']}"):
        result = grade_triple_generated(json.dumps(payload), **GRADE_KWARGS)
    record["grade_wall_s"] = round(time.monotonic() - t0, 2)
    record["verdict"] = result.verdict.name
    record["reason"] = result.reason
    record["result"] = asdict(result)
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
    print(f"verdict: {record['verdict']} — {record.get('reason', record.get('error'))}")
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
        record["model"] = MODEL
        record["effort"] = EFFORT
        with Spinner(f"{MODEL} writing a distractor patch"):
            out, resp = call_model(build_prompt(parent))
        record["usage"] = {"input": resp.usage.input_tokens,
                           "output": resp.usage.output_tokens}
        if out is None:
            record["verdict"] = "REFUSED"
            dump(record)
            print("refusal — logged")
            return
        record["reasoning"] = out["reasoning"]
        grade_extension(parent, out["patch"], record)

    dump(record)
    report(record)


if __name__ == "__main__":
    main()
