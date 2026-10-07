"""Stage 4 Initiator loop: sample seeds -> call the model -> grade -> log.

Standalone by design: no Formal Disco, no Fixer, no agenda. Run the
stages in order, cheapest first:

  venv/bin/python initiator/run.py check-contract    # no API, no solver
  venv/bin/python initiator/run.py check-exemplars   # solver, no API (needs hwtools)
  venv/bin/python initiator/run.py one               # ONE API call, prints raw output, no grading
  venv/bin/python initiator/run.py grade-one         # one API call + grade (needs both)
  venv/bin/python initiator/run.py pilot --n 10      # the loop (asserts exemplar pool first)

Model: claude-opus-5-5 (claude-opus-5 before 8 Oct 2026; every row
records which). Temperature is not a parameter on this model
(the API rejects it); diversity comes from seed rotation + adaptive
thinking, and the log records model + effort instead. Server-side
refusal fallbacks are deliberately NOT enabled: a corpus row must record
which model wrote it, so a refusal is logged and discarded, never
silently rerouted to another model.
"""
import argparse
import itertools
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # graded_oracle root -> `oracle` package
sys.path.insert(0, str(HERE))

from prompts import (SIZE_SECTION, SYSTEM_PROMPT,   # noqa: E402
                     USER_TEMPLATE, prompt_version)
from schema import TRIPLE_SCHEMA                   # noqa: E402

import llm_client                                            # noqa: E402
from oracle import NecessityVerdict, grade_triple_generated  # noqa: E402
from oracle.contract import parse_generator_output           # noqa: E402

MODEL = "claude-opus-5-5"
EFFORT = "high"
# 32000 suits a frontier model with a huge window. A locally served 32k
# model has no room for it: prompt plus budget is then over the limit and
# every call is refused, so lower it when generating on the cluster.
MAX_TOKENS = int(os.getenv("INITIATOR_MAX_TOKENS", "32000"))
GRADE_KWARGS = dict(timeout_s=120)
LOG_PATH = HERE / "logs" / "attempts.jsonl"


class Spinner:
    """Braille-dot spinner with colour and a mm:ss clock. Silent when
    stderr is not a TTY (logs and pipes stay clean). Copied rather than
    imported so each worker stays standalone."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CYAN, DIM, RESET = "\033[36m", "\033[2m", "\033[0m"

    def __init__(self, label):
        self.label = label
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
        if sys.stderr.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()


def _pool(path):
    """One seed per line. A repeated line is how a pool is weighted, and
    lines starting with # carry the provenance of the weights."""
    return [s.strip() for s in path.read_text().splitlines()
            if s.strip() and not s.startswith("#")]


def load_pools(constructs_path=None, even=False):
    """`constructs_path` swaps in another kinds file (the catalog run's
    picked kinds); the default is our 32. `even` keeps each distinct
    style, pattern and scope line once: the files weight lines by
    repeating them, and their own notes say the weights came from
    counting the evaluation sets (written 20 Sep 2026, before the rule
    against shaping training data on the tests)."""
    exemplars = json.loads((HERE / "exemplars.json").read_text())
    constructs = [s.strip() for s in
                  Path(constructs_path or HERE / "constructs.txt")
                  .read_text().splitlines()
                  if s.strip()]
    readmes = [json.loads(line) for line in
               (HERE / "readmes.jsonl").read_text().splitlines() if line.strip()]
    styles = _pool(HERE / "styles.txt")
    patterns = _pool(HERE / "patterns.txt")
    scopes = _pool(HERE / "scopes.txt")
    if even:
        styles, patterns, scopes = (list(dict.fromkeys(x))
                                    for x in (styles, patterns, scopes))
    return exemplars, constructs, readmes, styles, patterns, scopes


def _property_is_conditional(verilog):
    """Does any assertion actually depend on something? Either guarded by
    an `if` within the three lines above it, or written as an implication
    (`!armed || ...`, or a ternary)."""
    lines = verilog.splitlines()
    for i, line in enumerate(lines):
        if "assert" not in line:
            continue
        window = " ".join(lines[max(0, i - 3):i + 1])
        if re.search(r"if\s*\(", window) or "||" in line or "?" in line:
            return True
    return False


def scope_gate(scope, triple):
    """An armed property that is never armed is vacuously true, and an arm
    register the property ignores is decoration. Two conditions for a
    scoped seed: the arm register is declared as an antecedent, so the
    oracle's cover run proves the armed state reachable, and the property
    is actually conditional on something. Returns a reason to reject, or
    None."""
    if scope.lower().startswith("globally"):
        return None
    if not (triple.get("antecedents") or []):
        return ("scope is armed but antecedents is empty - the arm "
                "register must be listed so the cover run can prove the "
                "property is ever required")
    verilog = triple.get("verilog")
    if verilog and not _property_is_conditional(verilog):
        return ("scope is armed but every assertion is unconditional - the "
                "arm register is decoration, not a scope")
    return None


def endpoint_model(list_models=None):
    """What the server is really serving. Both the base model and the
    fine-tune answer to the alias "llm", so a yield number is unreadable
    without this. Returns None if the endpoint cannot be asked."""
    try:
        if list_models is None:
            from openai import OpenAI
            client = OpenAI(
                base_url=os.environ["OPENROUTER_BASE_URL"],
                api_key=os.environ.get("OPENROUTER_API_KEY", "none"))
            listing = client.models.list()
        else:
            listing = list_models()
        first = (getattr(listing, "data", None) or [None])[0]
        if first is None:
            return None
        return getattr(first, "root", None) or getattr(first, "id", None)
    except Exception:
        return None


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


def build_user_msg(readme, construct, style, pattern, scope, exemplar,
                   scale=None):
    msg = USER_TEMPLATE.format(
        readme=readme["readme"], construct=construct, style=style,
        pattern=pattern, scope=scope, exemplar=json.dumps(exemplar, indent=2))
    if scale:
        marker = "## Property pattern seed"
        msg = msg.replace(marker, SIZE_SECTION.format(scale=scale) + marker, 1)
    return msg


def call_model(user_msg, model=None):
    """One API call (Anthropic direct or OpenRouter, per CLAUDE_PROVIDER).
    Returns (raw_json_text or None, usage, stop)."""
    return llm_client.call_claude(
        model=model or MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        user=user_msg, schema=TRIPLE_SCHEMA, effort=EFFORT)


def dump(record, path=None):
    path = Path(path or LOG_PATH)
    path.parent.mkdir(exist_ok=True)
    try:
        line = json.dumps(record, default=str)
        with path.open("a") as f:
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


def make_samplers(exemplars, constructs, readmes, styles, patterns, scopes):
    return (BalancedSampler(readmes), BalancedSampler(constructs),
            BalancedSampler(styles), BalancedSampler(patterns),
            BalancedSampler(scopes), BalancedSampler(list(exemplars.items())))


def sample_seeds(readme_s, construct_s, style_s, pattern_s, scope_s,
                 exemplar_s):
    readme = readme_s.draw()
    construct = construct_s.draw()
    style = style_s.draw()
    pattern = pattern_s.draw()
    scope = scope_s.draw()
    ex_id, exemplar = exemplar_s.draw()
    return readme, construct, style, pattern, scope, ex_id, exemplar


def plan_attempts(n, *, cmd, grade, model, constructs_path, scales_path,
                  even_seeds):
    """Draw every attempt's seeds and build its question, before any call.
    Returns (exemplars, run_id, [(record, user_msg, scope)])."""
    exemplars, constructs, readmes, styles, patterns, scopes = load_pools(
        constructs_path, even=even_seeds)
    scale_s = BalancedSampler(_pool(Path(scales_path))) if scales_path else None
    # One id per invocation: attempts from different runs (one, grade-one,
    # pilot, the 200) all append to the same file and stay separable.
    # e.g. "2026-08-14_15h30m42s" (local wall clock; cmd is its own field).
    run_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    serving = (endpoint_model()
               if os.environ.get("OPENROUTER_BASE_URL") else None)
    print(f"run_id: {run_id}"
          + (f"   serving: {serving}" if serving else ""))
    samplers = make_samplers(exemplars, constructs, readmes, styles,
                             patterns, scopes)
    plans = []
    for i in range(n):
        (readme, construct, style, pattern, scope, ex_id,
         exemplar) = sample_seeds(*samplers)
        scale = scale_s.draw() if scale_s else None
        record = {
            "run_id": run_id,
            "cmd": cmd,
            "attempt": i,
            "graded": grade,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": llm_client.model_label(model), "effort": EFFORT,
            "served_model": serving,
            "temperature": "n/a: removed from the API on this model; "
                           "effort + seed rotation are the diversity knobs",
            "prompt_version": prompt_version(),
            "readme_id": readme["repo"], "construct": construct,
            "style": style, "pattern": pattern, "scope": scope,
            "exemplar_id": ex_id,
        }
        if constructs_path:
            record["constructs_file"] = Path(constructs_path).name
        if scale:
            record["scale"] = scale
        if even_seeds:
            record["seed_weights"] = "even"
        plans.append((record, build_user_msg(readme, construct, style,
                                             pattern, scope, exemplar, scale),
                      scope))
    return exemplars, run_id, plans


def finish_attempt(i, record, raw_json, usage, stop, raw_text, *, grade,
                   show_raw, log_path, tally):
    """Everything after the model answered: the same for a direct call and
    a batch entry, so a batch design is judged exactly like any other."""
    scope = record["scope"]
    record["usage"] = usage
    if raw_json is None:
        record["verdict"] = {"refusal": "REFUSED",
                             "length": "TRUNCATED"}.get(stop, "UNPARSEABLE")
        record["raw_text"] = raw_text
        dump(record, log_path)
        print(f"[{i}] {record['verdict']} - logged, continuing")
        return
    record["raw_json"] = raw_json
    if show_raw:
        print(json.dumps(json.loads(raw_json), indent=2))

    reason = scope_gate(scope, json.loads(raw_json))
    if reason:
        record["verdict"] = "SCOPE_UNARMED"
        record["reason"] = reason
        tally["SCOPE_UNARMED"] = tally.get("SCOPE_UNARMED", 0) + 1
        dump(record, log_path)
        print(f"[{i}] SCOPE_UNARMED - {reason[:70]}")
        return

    if grade:
        t0 = time.monotonic()
        with Spinner(f"[{i}] oracle grading"):
            result = grade_triple_generated(raw_json, **GRADE_KWARGS)
        record["grade_wall_s"] = round(time.monotonic() - t0, 2)
        record["verdict"] = result.verdict.name
        record["result"] = asdict(result)
        tally[result.verdict.name] = tally.get(result.verdict.name, 0) + 1
        print(f"[{i}] {result.verdict.name:12s} ({record['grade_wall_s']}s) "
              f"- {result.reason[:100]}")
    dump(record, log_path)


def run_attempts(n, grade=True, show_raw=False, cmd="", constructs_path=None,
                 scales_path=None, log_path=None, model=None,
                 even_seeds=False):
    """`constructs_path`, `scales_path` and `log_path` are the catalog run's
    (task 5): its own kinds, a size seed per attempt, and its own log, so
    its designs never enter the main corpus. `model` swaps the writer
    (Opus 5.5 is $4/$20 per million tokens against Opus 5's $5/$25);
    `even_seeds` draws each style, pattern and scope equally often.
    Unset, a run is as before."""
    model = model or MODEL
    exemplars, run_id, plans = plan_attempts(
        n, cmd=cmd, grade=grade, model=model, constructs_path=constructs_path,
        scales_path=scales_path, even_seeds=even_seeds)
    if grade:
        assert_exemplar_pool(exemplars)
    tally = {}
    for i, (record, user_msg, scope) in enumerate(plans):
        try:
            with Spinner(f"[{i}] {llm_client.model_label(model)} writing a triple"):
                raw_json, usage, stop = call_model(user_msg, model=model)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            dump(record, log_path)
            print(f"[{i}] API error: {type(exc).__name__} - logged, continuing")
            time.sleep(5)
            continue
        finish_attempt(i, record, raw_json, usage, stop, llm_client.LAST_RAW,
                       grade=grade, show_raw=show_raw, log_path=log_path,
                       tally=tally)

    if grade and tally:
        print("\nverdict distribution:", dict(sorted(tally.items())))


# --- batch mode (8 Oct 2026): the same questions through the Message
# Batches API at half price. Every question and its seeds are saved in a
# manifest BEFORE the batch is sent, so a crash can never lose paid work;
# answers are judged by finish_attempt, exactly like direct ones, and
# `collect` can be re-run safely: answers already logged are skipped.

def run_batch(n, grade=True, cmd="", constructs_path=None, scales_path=None,
              log_path=None, model=None, even_seeds=False, poll_s=60):
    model = model or MODEL
    exemplars, run_id, plans = plan_attempts(
        n, cmd=cmd, grade=grade, model=model, constructs_path=constructs_path,
        scales_path=scales_path, even_seeds=even_seeds)
    if grade:
        assert_exemplar_pool(exemplars)
    log = Path(log_path or LOG_PATH)
    manifest = log.parent / f"batch_{run_id}.json"
    attempts = [{"custom_id": f"a{i:04d}", "record": record,
                 "user_msg": user_msg}
                for i, (record, user_msg, _) in enumerate(plans)]
    m = {"run_id": run_id, "log": str(log), "model": model, "grade": grade,
         "max_tokens": MAX_TOKENS, "effort": EFFORT, "batch_id": None,
         "attempts": attempts}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(m, indent=1))
    print(f"{len(attempts)} questions saved -> {manifest}")
    items = [(a["custom_id"], llm_client.request_kwargs(
        model=model, max_tokens=MAX_TOKENS, user=a["user_msg"],
        system=SYSTEM_PROMPT, schema=TRIPLE_SCHEMA, effort=EFFORT))
        for a in attempts]
    m["batch_id"] = llm_client.batch_submit(items)
    m["submitted"] = datetime.now(timezone.utc).isoformat()
    manifest.write_text(json.dumps(m, indent=1))
    print(f"batch {m['batch_id']} sent; if this stops, resume with:\n"
          f"  venv/bin/python initiator/run.py collect --manifest {manifest}")
    collect(manifest, poll_s=poll_s)


def collect(manifest, poll_s=60):
    """Wait for the batch to end, then judge and log every answer not
    already in the log."""
    manifest = Path(manifest)
    m = json.loads(manifest.read_text())
    if not m.get("batch_id"):
        sys.exit(f"{manifest.name} was never sent: no batch id, nothing paid")
    while True:
        status, counts = llm_client.batch_status(m["batch_id"])
        print(f"batch {m['batch_id']}: {status} {counts}", flush=True)
        if status == "ended":
            break
        time.sleep(poll_s)
    log = Path(m["log"])
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("batch_id") == m["batch_id"]:
                done.add(r.get("custom_id"))
    results = dict(llm_client.batch_results(m["batch_id"]))
    tally, spent = {}, {"input": 0, "output": 0}
    for i, a in enumerate(m["attempts"]):
        cid = a["custom_id"]
        if cid in done:
            continue
        record = dict(a["record"], batch=True, batch_id=m["batch_id"],
                      custom_id=cid)
        stop, text, usage, raw = results.get(
            cid, ("error:no_result", None, {"input": 0, "output": 0}, None))
        spent["input"] += usage.get("input", 0)
        spent["output"] += usage.get("output", 0)
        if stop.startswith("error"):
            record["usage"] = usage
            record["error"] = f"batch {stop}"
            dump(record, log)
            print(f"[{i}] {record['error']} - logged, continuing")
            continue
        finish_attempt(i, record, text, usage, stop, raw, grade=m["grade"],
                       show_raw=False, log_path=log, tally=tally)
    if tally:
        print("\nverdict distribution:", dict(sorted(tally.items())))
    print(f"this collect: {spent['input']:,} in, {spent['output']:,} out = "
          f"${llm_client.dollars(m['model'], spent['input'], spent['output'], batch=True):.2f}"
          " at batch price")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-contract")
    sub.add_parser("check-exemplars")
    sub.add_parser("one")
    sub.add_parser("grade-one")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--n", type=int, default=10)
    pilot.add_argument("--constructs", default=None,
                       help="kinds file to draw from instead of constructs.txt")
    pilot.add_argument("--scales", default=None,
                       help="size seeds, one drawn per attempt")
    pilot.add_argument("--log", default=None,
                       help="attempts log to write instead of logs/attempts.jsonl")
    pilot.add_argument("--batch", action="store_true",
                       help="send every question as one batch at half price")
    pilot.add_argument("--even-seeds", action="store_true",
                       help="draw each style, pattern and scope equally often")
    pilot.add_argument("--model", default=None,
                       help=f"model that writes the designs (default {MODEL})")

    col = sub.add_parser("collect")
    col.add_argument("--manifest", required=True,
                     help="the batch_<run_id>.json a batch run wrote")

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
        run = run_batch if args.batch else run_attempts
        run(args.n, grade=True, cmd="pilot",
            constructs_path=args.constructs, scales_path=args.scales,
            log_path=args.log, model=args.model, even_seeds=args.even_seeds)
    elif args.cmd == "collect":
        collect(args.manifest)


if __name__ == "__main__":
    main()
