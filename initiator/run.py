"""Stage 4 Initiator loop: sample seeds -> call the model -> grade -> log.

Standalone by design: no Formal Disco, no Fixer, no agenda. Run the
stages in order, cheapest first:

  venv/bin/python initiator/run.py check-contract    # no API, no solver
  venv/bin/python initiator/run.py check-exemplars   # solver, no API (needs hwtools)
  venv/bin/python initiator/run.py one               # ONE API call, prints raw output, no grading
  venv/bin/python initiator/run.py grade-one         # one API call + grade (needs both)
  venv/bin/python initiator/run.py pilot --n 10      # the loop (asserts exemplar pool first)

Model: claude-opus-5. Temperature is not a parameter on this model
(the API rejects it); diversity comes from seed rotation + adaptive
thinking, and the log records model + effort instead. Server-side
refusal fallbacks are deliberately NOT enabled: a corpus row must record
which model wrote it, so a refusal is logged and discarded, never
silently rerouted to another model.
"""
import argparse
import itertools
import json
import random
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # graded_oracle root -> `oracle` package
sys.path.insert(0, str(HERE))

from prompts import SYSTEM_PROMPT, USER_TEMPLATE   # noqa: E402
from schema import TRIPLE_SCHEMA                   # noqa: E402

import llm_client                                            # noqa: E402
from oracle import NecessityVerdict, grade_triple_generated  # noqa: E402
from oracle.contract import parse_generator_output           # noqa: E402

MODEL = "claude-opus-5"
EFFORT = "high"
MAX_TOKENS = 16000
GRADE_KWARGS = dict(timeout_s=120)
LOG_PATH = HERE / "logs" / "attempts.jsonl"


class Spinner:
    """Terminal spinner with a label and elapsed seconds; silent when
    stderr is not a TTY (logs and pipes stay clean)."""
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


def load_pools():
    exemplars = json.loads((HERE / "exemplars.json").read_text())
    shapes = [s.strip() for s in (HERE / "shapes.txt").read_text().splitlines()
              if s.strip()]
    readmes = [json.loads(line) for line in
               (HERE / "readmes.jsonl").read_text().splitlines() if line.strip()]
    return exemplars, shapes, readmes


def check_contract():
    """Preflight 0.1: unknown cti_* fields must not trip a violation."""
    out = parse_generator_output(
        '{"verilog": "module m (input wire clk); always @(*) assert (1); endmodule",'
        ' "top_module": "m", "cti_state": [{"signal": "x", "value": "1"}],'
        ' "cti_reasoning": "t"}')
    print(f"contract ok: parsed top_module={out.prop.top_module!r}, "
          "cti_* fields ignored without violation")


def assert_exemplar_pool(exemplars):
    """Preflight 0.2: every exemplar must grade NECESSARY, or the run
    would teach the model to imitate a broken example."""
    for name, ex in exemplars.items():
        with Spinner(f"grading exemplar {name}"):
            r = grade_triple_generated(json.dumps(ex), **GRADE_KWARGS)
        assert r.verdict is NecessityVerdict.NECESSARY, \
            f"exemplar {name}: {r.verdict.name} - {r.reason}"
        print(f"exemplar {name}: NECESSARY")
    print(f"exemplar pool ok ({len(exemplars)})")


def build_user_msg(readme, shapes2, exemplar):
    return USER_TEMPLATE.format(
        readme=readme["readme"], shape_1=shapes2[0], shape_2=shapes2[1],
        exemplar=json.dumps(exemplar, indent=2))


def call_model(user_msg):
    """One API call (Anthropic direct or OpenRouter, per CLAUDE_PROVIDER).
    Returns (raw_json_text or None, usage, stop)."""
    return llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        user=user_msg, schema=TRIPLE_SCHEMA, effort=EFFORT)


def dump(record):
    LOG_PATH.parent.mkdir(exist_ok=True)
    try:
        line = json.dumps(record, default=str)
        with LOG_PATH.open("a") as f:
            f.write(line + "\n")
    except Exception as exc:
        # Never lose the artifact to a serialisation bug.
        print(f"LOG WRITE FAILED ({exc}); raw_json follows:", file=sys.stderr)
        print(record.get("raw_json"), file=sys.stderr)


class BalancedSampler:
    """Deal from a shuffled bag without replacement, reshuffling when the
    bag empties. Over N draws every item appears floor(N/len) or
    ceil(N/len) times — no zero-hit seeds, no triple-hits, so seed
    coverage is not a noise source in the diversity measurement."""

    def __init__(self, items):
        self.items = list(items)
        self._bag = []

    def draw(self):
        if not self._bag:
            self._bag = random.sample(self.items, len(self.items))
        return self._bag.pop()

    def draw2(self):
        a = self.draw()
        b = self.draw()
        while b == a:   # only possible across a reshuffle boundary
            b = self.draw()
        return [a, b]


def make_samplers(exemplars, shapes, readmes):
    return (BalancedSampler(readmes), BalancedSampler(shapes),
            BalancedSampler(list(exemplars.items())))


def sample_seeds(readme_s, shape_s, exemplar_s):
    readme = readme_s.draw()
    shapes2 = shape_s.draw2()
    ex_id, exemplar = exemplar_s.draw()
    return readme, shapes2, ex_id, exemplar


def run_attempts(n, grade=True, show_raw=False, cmd=""):
    exemplars, shapes, readmes = load_pools()
    if grade:
        assert_exemplar_pool(exemplars)

    # One id per invocation: attempts from different runs (one, grade-one,
    # pilot, the 200) all append to the same file and stay separable.
    # e.g. "2026-08-14_15h30m42s" (local wall clock; cmd is its own field).
    run_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    print(f"run_id: {run_id}")

    samplers = make_samplers(exemplars, shapes, readmes)
    tally = {}
    for i in range(n):
        readme, shapes2, ex_id, exemplar = sample_seeds(*samplers)
        record = {
            "run_id": run_id,
            "cmd": cmd,
            "attempt": i,
            "graded": grade,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": llm_client.model_label(MODEL), "effort": EFFORT,
            "temperature": "n/a: removed from the API on this model; "
                           "effort + seed rotation are the diversity knobs",
            "readme_id": readme["repo"], "shape_ids": shapes2, "exemplar_id": ex_id,
        }
        try:
            with Spinner(f"[{i}] {MODEL} writing a triple"):
                raw_json, usage, stop = call_model(
                    build_user_msg(readme, shapes2, exemplar))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            dump(record)
            print(f"[{i}] API error: {type(exc).__name__} — logged, continuing")
            time.sleep(5)
            continue

        record["usage"] = usage
        if raw_json is None:
            record["verdict"] = "REFUSED" if stop == "refusal" else "UNPARSEABLE"
            dump(record)
            print(f"[{i}] refusal — logged, continuing")
            continue
        record["raw_json"] = raw_json
        if show_raw:
            print(json.dumps(json.loads(raw_json), indent=2))

        if grade:
            t0 = time.monotonic()
            with Spinner(f"[{i}] oracle grading"):
                result = grade_triple_generated(raw_json, **GRADE_KWARGS)
            record["grade_wall_s"] = round(time.monotonic() - t0, 2)
            record["verdict"] = result.verdict.name
            record["result"] = asdict(result)
            tally[result.verdict.name] = tally.get(result.verdict.name, 0) + 1
            print(f"[{i}] {result.verdict.name:12s} ({record['grade_wall_s']}s) "
                  f"— {result.reason[:100]}")
        dump(record)

    if grade and tally:
        print("\nverdict distribution:", dict(sorted(tally.items())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-contract")
    sub.add_parser("check-exemplars")
    sub.add_parser("one")
    sub.add_parser("grade-one")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--n", type=int, default=10)

    args = p.parse_args()
    if args.cmd == "check-contract":
        check_contract()
    elif args.cmd == "check-exemplars":
        assert_exemplar_pool(load_pools()[0])
    elif args.cmd == "one":
        run_attempts(1, grade=False, show_raw=True, cmd="one")
    elif args.cmd == "grade-one":
        run_attempts(1, grade=True, show_raw=True, cmd="grade-one")
    elif args.cmd == "pilot":
        run_attempts(args.n, grade=True, cmd="pilot")


if __name__ == "__main__":
    main()
