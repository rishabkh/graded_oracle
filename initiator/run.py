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
import contextlib
import itertools
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # graded_oracle root -> `oracle` package
sys.path.insert(0, str(HERE))

from prompts import (IDEA_SYSTEM, IDEA_TEMPLATE,     # noqa: E402
                     IMPLEMENT_TEMPLATE, REPAIR_TEMPLATE, SIZE_SECTION,
                     SYSTEM_PROMPT, USER_TEMPLATE, prompt_version)
from schema import TRIPLE_SCHEMA                   # noqa: E402

import llm_client                                            # noqa: E402
import distill_log                                           # noqa: E402
from oracle import NecessityVerdict, grade_triple_generated  # noqa: E402
from oracle.contract import parse_generator_output           # noqa: E402

MODEL = "claude-opus-5-5"
EFFORT = "high"
# 128000 is the model's own limit (9 Oct 2026; it was 32000): bigger
# designs need long answers, thinking counts towards the limit, and an
# answer cut off is paid for and lost. Only what an answer uses is paid
# for. A locally served 32k model has no room for it: prompt plus budget
# is then over the limit and every call is refused, so lower it with
# INITIATOR_MAX_TOKENS when generating on the cluster.
MAX_TOKENS = int(os.getenv("INITIATOR_MAX_TOKENS", "128000"))
# Bigger designs (9 Oct 2026): 300 s per proof run (was 120); the pdr
# second opinion and the sanity check keep 120 s; conditions unreached in
# 20 steps get a deeper look first (oracle/depth.py).
GRADE_KWARGS = dict(timeout_s=300, pdr_timeout_s=120, cover_depth="auto")
LOG_PATH = HERE / "logs" / "attempts.jsonl"


class Spinner:
    """Braille-dot spinner with colour and a mm:ss clock. Silent when
    stderr is not a TTY (logs and pipes stay clean). Copied rather than
    imported so each worker stays standalone."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CYAN, DIM, RESET = "\033[36m", "\033[2m", "\033[0m"

    def __init__(self, label):
        """`label` is text, or a function giving the current text (9 Oct
        2026: a status line that changes while it spins)."""
        self.label = label
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def text(self):
        if not callable(self.label):
            return self.label
        try:
            return self.label()
        except Exception:                 # a display must never stop a run
            return "working (status unavailable)"

    def say(self, text):
        """Print a line without the spinner running through it."""
        with self._lock:
            if self._thread:
                sys.stderr.write("\r\033[K")
                sys.stderr.flush()
            print(text, flush=True)

    def _spin(self):
        start = time.monotonic()
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            elapsed = int(time.monotonic() - start)
            clock = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
            import shutil as _sh
            width = _sh.get_terminal_size().columns
            label = self.text()[:max(10, width - 12)]
            with self._lock:
                sys.stderr.write(f"\r{self.CYAN}{frame}{self.RESET} {label} "
                                 f"{self.DIM}{clock}{self.RESET}\033[K")
                sys.stderr.flush()
            self._stop.wait(0.08)
        with self._lock:
            sys.stderr.write("\r\033[K")
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


_BLOCK_START = re.compile(r"\b(always\w*|initial|assign|module)\b")


def _property_is_conditional(verilog):
    """Does any assertion actually depend on something? Either guarded by
    an `if` in the block it sits in, or written as an implication
    (`!armed || ...`, or a ternary) anywhere in the assert statement. 9 Oct
    2026: it looked only 3 lines up and at the assert's first line, which
    bigger designs, with longer guarded blocks and wrapped asserts, beat."""
    lines = verilog.splitlines()
    for i, line in enumerate(lines):
        if not re.search(r"\bassert\b", line):
            continue
        stmt, j = line, i
        while ";" not in stmt and j + 1 < len(lines):
            j += 1
            stmt += " " + lines[j]
        if "||" in stmt or "?" in stmt:
            return True
        k = i
        while k > 0 and not _BLOCK_START.search(lines[k]):
            k -= 1
        if re.search(r"\bif\s*\(", " ".join(lines[k:i + 1])):
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


def assert_exemplar_pool(exemplars, proof_root=None):
    """Preflight 0.2: every exemplar must grade NECESSARY, or the run
    would teach the model to imitate a broken example. With `proof_root`,
    each exemplar's proofs are kept in <proof_root>/<name>."""
    for name, ex in exemplars.items():
        kw = dict(GRADE_KWARGS)
        if proof_root is not None:
            kw["workdir_root"] = Path(proof_root) / name
        with Spinner(f"grading exemplar {name}"):
            r = grade_triple_generated(json.dumps(ex), **kw)
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


IDEA_SCHEMA = {"type": "object", "properties": {"idea": {"type": "string"}},
               "required": ["idea"], "additionalProperties": False}


def call_model(user_msg, model=None, system=None, schema=None):
    """One API call (Anthropic direct or OpenRouter, per CLAUDE_PROVIDER).
    Returns (raw_json_text or None, usage, stop). The default is the
    generator's own question: SYSTEM_PROMPT and the triple schema."""
    return llm_client.call_claude(
        model=model or MODEL, max_tokens=MAX_TOKENS,
        system=system or SYSTEM_PROMPT, user=user_msg,
        schema=schema or TRIPLE_SCHEMA, effort=EFFORT)


def _messages(system, user):
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def default_distill(log_path):
    """Training examples sit next to the run log they came from."""
    log = Path(log_path or LOG_PATH)
    return log.with_name("distill_" + log.name)


def seed_arguments(record, extras):
    """The inputs of one generator question, Formal Disco style: the README
    by name and full text, and every seed that shaped the question."""
    return {"repo": extras["readme"]["repo"],
            "readme": extras["readme"]["readme"],
            "construct": record["construct"], "scale": record.get("scale"),
            "pattern": record["pattern"], "scope": record["scope"],
            "style": record["style"], "exemplar_id": record["exemplar_id"]}


def _meta(record):
    keys = ("run_id", "attempt", "model", "effort", "prompt_version",
            "constructs_file", "seed_weights", "batch")
    return {k: record[k] for k in keys if k in record}


def calls_path(log_path):
    """Every API call of a run, next to its run log (llm_client.call_log)."""
    log = Path(log_path or LOG_PATH)
    return log.with_name("calls_" + log.name)


def proofs_dir(log_path):
    """Kept proof folders, next to the run log (9 Oct 2026, keep
    everything): <log stem>_proofs/. They travel with the log."""
    log = Path(log_path or LOG_PATH)
    return log.with_name(f"{log.stem}_proofs")


def design_proofs(log_path, record):
    """One design's proof folder: its path relative to the log's folder
    (what rows record, so it works on any machine) and its real path. Each
    check of the design gets a round inside it: r0, then r1, r2 per
    repair."""
    log = Path(log_path or LOG_PATH)
    rel = f"{proofs_dir(log).name}/{record['run_id']}_a{record['attempt']:03d}"
    return rel, log.parent / rel


def timed_out(j):
    """The checker ran out of time: a repair cannot fix that."""
    return j.get("verdict") == "INCONCLUSIVE" or \
        "TIMEOUT" in (j.get("reason") or "")


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
                      scope, {"readme": readme, "exemplar": exemplar}))
    return exemplars, run_id, plans


def judge(raw_json, stop, scope, grade=True, label="", proof_root=None,
          spin=True):
    """The checker on one answer, the same on every path: no design
    (refused, cut off, unreadable), then the scope gate, then the oracle.
    Returns verdict, reason, result and grading time (verdict None when
    grading is off). `proof_root` is where this check's proofs are kept."""
    out = {"verdict": None, "reason": None, "result": None,
           "grade_wall_s": None}
    if raw_json is None:
        out["verdict"] = {"refusal": "REFUSED",
                          "length": "TRUNCATED"}.get(stop, "UNPARSEABLE")
        return out
    reason = scope_gate(scope, json.loads(raw_json))
    if reason:
        return dict(out, verdict="SCOPE_UNARMED", reason=reason)
    if not grade:
        return out
    t0 = time.monotonic()
    kw = dict(GRADE_KWARGS)
    if proof_root is not None:
        kw["workdir_root"] = proof_root
    # a run with one status line for everything turns this spinner off
    with (Spinner(f"{label} oracle grading") if spin
          else contextlib.nullcontext()):
        result = grade_triple_generated(raw_json, **kw)
    return {"verdict": result.verdict.name, "reason": result.reason,
            "result": asdict(result),
            "grade_wall_s": round(time.monotonic() - t0, 2)}


def finish_attempt(i, record, raw_json, usage, stop, raw_text, *, grade,
                   show_raw, log_path, tally, distill_path=None, extras=None,
                   user_msg=None):
    """Everything after the model answered: the same for a direct call and
    a batch entry, so a batch design is judged exactly like any other.
    With a distill path, the answer is also saved as an `initiate`
    training example (distill_log.py)."""
    record["usage"] = usage
    if raw_json is not None:
        record["raw_json"] = raw_json
        if show_raw:
            print(json.dumps(json.loads(raw_json), indent=2))
    rel, root = design_proofs(log_path, record)
    record["proof_dir"] = rel
    j = judge(raw_json, stop, record["scope"], grade, label=f"[{i}]",
              proof_root=root / "r0")
    if raw_json is None:
        record["verdict"] = j["verdict"]
        record["raw_text"] = raw_text
        print(f"[{i}] {record['verdict']} - logged, continuing")
    elif j["verdict"] == "SCOPE_UNARMED":
        record["verdict"] = "SCOPE_UNARMED"
        record["reason"] = j["reason"]
        tally["SCOPE_UNARMED"] = tally.get("SCOPE_UNARMED", 0) + 1
        print(f"[{i}] SCOPE_UNARMED - {j['reason'][:70]}")
    elif grade:
        record["grade_wall_s"] = j["grade_wall_s"]
        record["verdict"] = j["verdict"]
        record["reason"] = j["reason"]
        record["result"] = j["result"]
        tally[j["verdict"]] = tally.get(j["verdict"], 0) + 1
        print(f"[{i}] {j['verdict']:12s} ({record['grade_wall_s']}s) "
              f"- {j['reason'][:100]}")
    dump(record, log_path)
    if distill_path and extras is not None and user_msg is not None:
        distill_log.record(distill_path, "initiate",
                           seed_arguments(record, extras),
                           raw_json if raw_json is not None else raw_text,
                           record.get("verdict"),
                           _messages(SYSTEM_PROMPT, user_msg),
                           reason=record.get("reason"),
                           proof_dir=f"{rel}/r0",
                           call={"usage": usage, "stop": stop,
                                 "batch_id": record.get("batch_id"),
                                 "custom_id": record.get("custom_id")},
                           **_meta(record))


def run_attempts(n, grade=True, show_raw=False, cmd="", constructs_path=None,
                 scales_path=None, log_path=None, model=None,
                 even_seeds=False, distill_path=None):
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
        assert_exemplar_pool(exemplars, proof_root=proofs_dir(log_path)
                             / f"{run_id}_exemplars")
    tally = {}
    distill_path = distill_path or default_distill(log_path)
    with llm_client.call_log(calls_path(log_path)):
        _attempt_loop(plans, model, grade, show_raw, log_path, tally,
                      distill_path)
    if grade and tally:
        print("\nverdict distribution:", dict(sorted(tally.items())))


def _attempt_loop(plans, model, grade, show_raw, log_path, tally,
                  distill_path):
    for i, (record, user_msg, scope, extras) in enumerate(plans):
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
                       tally=tally, distill_path=distill_path, extras=extras,
                       user_msg=user_msg)


# --- batch mode (8 Oct 2026): the same questions through the Message
# Batches API at half price. Every question and its seeds are saved in a
# manifest BEFORE the batch is sent, so a crash can never lose paid work;
# answers are judged by finish_attempt, exactly like direct ones, and
# `collect` can be re-run safely: answers already logged are skipped.

def run_batch(n, grade=True, cmd="", constructs_path=None, scales_path=None,
              log_path=None, model=None, even_seeds=False, poll_s=60,
              distill_path=None):
    model = model or MODEL
    exemplars, run_id, plans = plan_attempts(
        n, cmd=cmd, grade=grade, model=model, constructs_path=constructs_path,
        scales_path=scales_path, even_seeds=even_seeds)
    log = Path(log_path or LOG_PATH)
    if grade:
        assert_exemplar_pool(exemplars, proof_root=proofs_dir(log)
                             / f"{run_id}_exemplars")
    manifest = log.parent / f"batch_{run_id}.json"
    attempts = [{"custom_id": f"a{i:04d}", "record": record,
                 "user_msg": user_msg, "extras": extras}
                for i, (record, user_msg, _, extras) in enumerate(plans)]
    m = {"run_id": run_id, "log": str(log), "model": model, "grade": grade,
         "distill": str(distill_path or default_distill(log)),
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
    spinner = Spinner(f"waiting for batch {m['batch_id']}")
    with spinner:
        while True:
            status, counts = llm_client.batch_status(m["batch_id"])
            done = counts.get("succeeded", 0) + counts.get("errored", 0) + \
                counts.get("canceled", 0) + counts.get("expired", 0)
            spinner.label = (f"batch {m['batch_id']}: {status}, {done} of "
                             f"{len(m['attempts'])} answered (checks every "
                             f"{poll_s}s)")
            if status == "ended":
                break
            time.sleep(poll_s)
    print(f"batch {m['batch_id']} ended: {counts}", flush=True)
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
        llm_client.append_call(calls_path(log), {
            "started": m.get("submitted"), "provider": "anthropic",
            "model": m["model"], "batch": True, "batch_id": m["batch_id"],
            "custom_id": cid, "seconds": None,
            "request": llm_client.request_kwargs(
                model=m["model"], max_tokens=m["max_tokens"],
                user=a["user_msg"], system=SYSTEM_PROMPT,
                schema=TRIPLE_SCHEMA, effort=m["effort"]),
            "raw": raw, "stop": stop, "usage": usage})
        if stop.startswith("error"):
            record["usage"] = usage
            record["error"] = f"batch {stop}"
            dump(record, log)
            print(f"[{i}] {record['error']} - logged, continuing")
            continue
        finish_attempt(i, record, text, usage, stop, raw, grade=m["grade"],
                       distill_path=m.get("distill"), extras=a.get("extras"),
                       user_msg=a["user_msg"],
                       show_raw=False, log_path=log, tally=tally)
    if tally:
        print("\nverdict distribution:", dict(sorted(tally.items())))
    print(f"this collect: {spent['input']:,} in, {spent['output']:,} out = "
          f"${llm_client.dollars(m['model'], spent['input'], spent['output'], batch=True):.2f}"
          " at batch price")


# --- two-step and self-repairing generation (8 Oct 2026) --------------------
# Formal Disco's idea -> implement split and its self-repairing generator,
# on our contract. Per design, in order: (two-step only) the idea call; the
# design call; the checker; then up to `repairs` rounds that hand back what
# the checker found and ask for the complete corrected design. Each call is
# also saved as a training example in Formal Disco's shape (distill_log):
#   one-step, no repairs   initiate   the answer as given
#   one-step, repairs      generate   the ORIGINAL question -> the final design
#   two-step               idea       the idea as the model sent it, counted a
#                                     success only when its implementation
#                                     passed (Formal Disco's rule)
#                          implement  the idea -> the FIRST design, judged on
#                                     its own; fixes are repair examples
#                                     (Formal Disco's rule, 9 Oct 2026)
#   every repair round     repair     design + checker findings -> new design
# A check that ran out of time is never sent for repair (a repair cannot
# fix the checker's clock) and its example is saved as an error. Every
# check keeps its proof folder, one round per check (proofs_dir).
# Designs run in parallel threads, so with batch=True every call goes
# through llm_client's batch gate at half price; checking is limited to a
# few at a time so a returning batch cannot overload the machine and push
# checks past their time limit.

# 12000 since 9 Oct 2026 (was 4000): a 32-bit, 8-entry design already gave a
# 15,096-character counterexample, so bigger designs lost what broke
NOTES_CAP = 12000


class PipelineStatus:
    """What every design of a two-step or self-repair run is doing, for one
    status line (9 Oct 2026: these runs went silent while a batch was out,
    sometimes for many minutes). Batches out are read from the manifests
    the batch gate writes; spend from the records' usage."""
    def __init__(self, n, model, batch, manifest_dir):
        self.n, self.model, self.batch = n, model, batch
        self.manifest_dir = Path(manifest_dir) if manifest_dir else None
        self.started = datetime.now(timezone.utc)
        self.records = []
        self.spinner = None
        self._stage = {}
        self._lock = threading.Lock()
        self._batches = (float("-inf"), "")

    def set(self, i, stage):
        """stage: writing (waiting for the model), checking or done."""
        with self._lock:
            self._stage[i] = stage

    def say(self, text):
        (self.spinner.say if self.spinner else print)(text)

    def _batch_text(self, force=False):
        if not self.batch or self.manifest_dir is None:
            return ""
        now = time.monotonic()
        if not force and now - self._batches[0] < 2:
            return self._batches[1]
        out, oldest = 0, 0.0
        for f in self.manifest_dir.glob("batch_*.json"):
            try:
                m = json.loads(f.read_text())
                created = datetime.fromisoformat(m["created"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if created < self.started or m.get("ended") or m.get("error"):
                continue
            out += 1
            oldest = max(oldest, (datetime.now(timezone.utc)
                                  - created).total_seconds())
        text = (f"{out} batch{'es' if out > 1 else ''} out for "
                f"{int(oldest) // 60}:{int(oldest) % 60:02d}" if out else "")
        self._batches = (now, text)
        return text

    def label(self, force=False):
        with self._lock:
            stages = [self._stage.get(i, "queued") for i in range(self.n)]
        count = lambda s: sum(x == s for x in stages)            # noqa: E731
        line = (f"{self.n} designs: {count('writing')} waiting for the "
                f"model, {count('checking')} being checked, {count('done')} "
                "done")
        if count("queued"):
            line += f", {count('queued')} not started"
        parts = [line]
        batches = self._batch_text(force)
        if batches:
            parts.append(batches)
        spent_in = sum((r.get("usage") or {}).get("input", 0)
                       for r in self.records)
        spent_out = sum((r.get("usage") or {}).get("output", 0)
                        for r in self.records)
        parts.append(f"${llm_client.dollars(self.model, spent_in, spent_out, batch=self.batch):.2f} so far")
        return " | ".join(parts)


def _full_log(run):
    """The whole proof log when it was kept, else the 40-line excerpt."""
    path = run.get("log_path")
    try:
        if path and Path(path).exists():
            return Path(path).read_text(errors="replace")
    except OSError:
        pass
    return run.get("log_excerpt") or ""


def checker_notes(verdict, reason, result):
    """What the checker found, for a repair question: the verdict and its
    reason, each leg's reason, the counterexample trace, and any error
    lines from the proof logs. A cover run's trace shows a condition being
    reached, not a failure, so it is left out (9 Oct 2026)."""
    parts = [f"Verdict: {verdict}", f"Reason: {reason or '(none given)'}"]
    for leg in ("with_invariants", "without_invariants"):
        info = (result or {}).get(leg) or {}
        if info.get("reason"):
            parts.append(f"{leg.replace('_', ' ')}: {info['reason']}")
        for run in info.get("runs") or []:
            if run.get("trace_text") and run.get("mode") != "cover":
                parts.append(f"Counterexample ({leg.replace('_', ' ')}):\n"
                             f"{run['trace_text']}")
            errors = [l.strip() for l in _full_log(run).splitlines()
                      if "ERROR" in l or "error:" in l]
            if errors:
                parts.append("Tool errors:\n" + "\n".join(errors[:6]))
    return "\n\n".join(parts)[:NOTES_CAP]


def _add(usage, more):
    usage["input"] += (more or {}).get("input", 0)
    usage["output"] += (more or {}).get("output", 0)


def _step(user_msg, model, system=None, schema=None):
    """One call plus what the client recorded about it: (answer, usage,
    stop, call info, raw text). The raw text is kept even when there is no
    usable answer (a refusal, a cut-off reply)."""
    before = llm_client.last_call()
    t0 = time.monotonic()
    raw, usage, stop = call_model(user_msg, model=model, system=system,
                                  schema=schema)
    after = llm_client.last_call()
    entry = after if after is not None and after is not before else None
    info = {"usage": usage, "stop": stop,
            "seconds": entry["seconds"] if entry and entry.get("seconds")
            is not None else round(time.monotonic() - t0, 2),
            "batch_id": (entry or {}).get("batch_id"),
            "custom_id": (entry or {}).get("custom_id")}
    return raw, usage, stop, info, (entry or {}).get("raw", raw)


def design_pipeline(i, record, user_msg, extras, *, mode, repairs, model,
                    grade, log_path, distill_path, slots, tally, lock,
                    status=None):
    stage = status.set if status else (lambda i, s: None)
    say = status.say if status else print
    usage = {"input": 0, "output": 0}
    record.update(mode=mode, usage=usage)
    stage(i, "writing")
    rel, root = design_proofs(log_path, record)
    record["proof_dir"] = rel
    # filled in as the design goes, so a crash keeps the rounds already done
    history = record["history"] = []
    seeds = seed_arguments(record, extras)
    meta = lambda: _meta(record)                               # noqa: E731
    if mode == "two-step":
        idea_user = IDEA_TEMPLATE.format(
            repo=extras["readme"]["repo"], readme=extras["readme"]["readme"],
            construct=record["construct"], scale=record.get("scale") or
            "(no size seed: choose ordinary sizes)", pattern=record["pattern"],
            scope=record["scope"], style=record["style"])
        raw, u, stop, idea_call, idea_raw = _step(
            idea_user, model, system=IDEA_SYSTEM, schema=IDEA_SCHEMA)
        _add(usage, u)
        record["idea_call"] = idea_call
        idea = json.loads(raw).get("idea") if raw else None
        if not idea:
            record["verdict"] = {"refusal": "REFUSED", "length": "TRUNCATED"
                                 }.get(stop, "UNPARSEABLE")
            record["failed_step"] = "idea"
            record["raw_text"] = idea_raw
            stage(i, "done")
            dump(record, log_path)
            distill_log.record(distill_path, "idea", seeds, idea_raw,
                               record["verdict"],
                               _messages(IDEA_SYSTEM, idea_user),
                               call=idea_call, chain_usage=usage, **meta())
            return record
        record["idea"] = idea
        first_kind, first_args = "implement", {
            "idea": idea, "exemplar_id": record["exemplar_id"]}
        first_user = IMPLEMENT_TEMPLATE.format(
            idea=idea, scale=record.get("scale") or
            "(no size seed: choose ordinary sizes)",
            exemplar=json.dumps(extras["exemplar"], indent=2))
    else:
        first_kind = "generate" if repairs else "initiate"
        first_args, first_user = seeds, user_msg

    current, u, stop, last_call, raw_text = _step(first_user, model)
    _add(usage, u)
    with slots:
        stage(i, "checking")
        j = judge(current, stop, record["scope"], grade, label=f"[{i}]",
                  proof_root=root / "r0", spin=status is None)
    stage(i, "writing")
    history.append(dict(j, design=current if current is not None else raw_text,
                        call=last_call, proof_dir=f"{rel}/r0"))
    # the first answer and its own result: Formal Disco's implement example
    first = {"answer": current if current is not None else raw_text,
             "verdict": j["verdict"], "reason": j["reason"], "call": last_call}
    attempt = 0
    while j["verdict"] != "NECESSARY" and current is not None \
            and attempt < repairs:
        if timed_out(j):
            record["repair_skipped"] = ("the checker ran out of time; a "
                                        "repair cannot fix that")
            break
        notes = checker_notes(j["verdict"], j["reason"], j["result"])
        pretty = json.dumps(json.loads(current), indent=2)
        repair_user = REPAIR_TEMPLATE.format(triple=pretty, notes=notes)
        new, u, stop, call, new_raw = _step(repair_user, model)
        _add(usage, u)
        attempt += 1
        with slots:
            stage(i, "checking")
            jn = judge(new, stop, record["scope"], grade, label=f"[{i}]",
                       proof_root=root / f"r{attempt}", spin=status is None)
        stage(i, "writing")
        distill_log.record(distill_path, "repair",
                           {"triple": current, "notes": notes},
                           new if new is not None else new_raw, jn["verdict"],
                           _messages(SYSTEM_PROMPT, repair_user),
                           reason=jn["reason"], repair_round=attempt,
                           proof_dir=f"{rel}/r{attempt}", call=call, **meta())
        history.append(dict(jn, design=new if new is not None else new_raw,
                            call=call, proof_dir=f"{rel}/r{attempt}"))
        if new is None:
            break                       # nothing new to judge or repair
        current, j, last_call = new, jn, call

    record.update(verdict=j["verdict"], reason=j["reason"], result=j["result"],
                  grade_wall_s=j["grade_wall_s"], repair_attempts=attempt)
    if current is not None:
        record["raw_json"] = current
    else:
        record["raw_text"] = raw_text
    dump(record, log_path)
    if first_kind == "generate":
        # the self-repairing generator: the question -> its final design
        distill_log.record(distill_path, "generate", first_args,
                           current if current is not None else raw_text,
                           j["verdict"], _messages(SYSTEM_PROMPT, first_user),
                           reason=j["reason"], repair_attempts=attempt,
                           proof_dir=rel, call=last_call, chain_usage=usage,
                           **meta())
    else:
        distill_log.record(distill_path, first_kind, first_args,
                           first["answer"], first["verdict"],
                           _messages(SYSTEM_PROMPT, first_user),
                           reason=first["reason"], repair_attempts=attempt,
                           proof_dir=f"{rel}/r0", call=first["call"],
                           chain_usage=usage, **meta())
    if mode == "two-step":
        # as the model sent it (JSON), counted good only if its first design
        # passed
        distill_log.record(distill_path, "idea", seeds, idea_raw,
                           first["verdict"], _messages(IDEA_SYSTEM, idea_user),
                           reason=first["reason"], proof_dir=f"{rel}/r0",
                           call=record["idea_call"], chain_usage=usage,
                           **meta())
    with lock:
        tally[j["verdict"]] = tally.get(j["verdict"], 0) + 1
    stage(i, "done")
    say(f"[{i}] {mode:9s} {str(j['verdict']):12s} after {attempt} "
        f"repair(s)" + (" (checker ran out of time)"
                        if record.get("repair_skipped") else ""))
    return record


def run_pipeline(n, *, mode="initiate", repairs=0, workers=8, batch=False,
                 grade_slots=None, flush_after=30.0, grade=True, cmd="",
                 constructs_path=None, scales_path=None, log_path=None,
                 model=None, even_seeds=False, distill_path=None):
    from concurrent.futures import ThreadPoolExecutor
    if mode not in ("initiate", "two-step"):
        raise ValueError(f"mode must be initiate or two-step, not {mode!r}")
    model = model or MODEL
    exemplars, run_id, plans = plan_attempts(
        n, cmd=cmd, grade=grade, model=model, constructs_path=constructs_path,
        scales_path=scales_path, even_seeds=even_seeds)
    log = Path(log_path or LOG_PATH)
    if grade:
        assert_exemplar_pool(exemplars, proof_root=proofs_dir(log)
                             / f"{run_id}_exemplars")
    distill_path = distill_path or default_distill(log)
    slots = threading.BoundedSemaphore(
        grade_slots or max(1, (os.cpu_count() or 4) - 2))
    tally, lock = {}, threading.Lock()
    gate = contextlib.nullcontext()
    if batch:
        gate = llm_client.batch_gate(flush_after_s=flush_after, poll_s=60,
                                     manifest_dir=log.parent / "batches")
    for record, *_ in plans:
        record["repairs_allowed"] = repairs
        if batch:
            record["batch"] = True

    status = PipelineStatus(len(plans), model, batch,
                            log.parent / "batches" if batch else None)
    status.records = [record for record, *_ in plans]

    def one(item):
        i, (record, user_msg, scope, extras) = item
        try:
            return design_pipeline(
                i, record, user_msg, extras, mode=mode, repairs=repairs,
                model=model, grade=grade, log_path=log,
                distill_path=distill_path, slots=slots, tally=tally, lock=lock,
                status=status)
        except Exception as exc:
            # the rounds already done stay in record["history"]
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            dump(record, log)
            status.set(i, "done")
            status.say(f"[{i}] error: {record['error'][:120]} - logged, "
                       "continuing")
            return record

    with llm_client.call_log(calls_path(log)), gate:
        with Spinner(status.label) as spinner:
            status.spinner = spinner
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                out = list(ex.map(one, list(enumerate(plans))))
        status.spinner = None
    if tally:
        print("\nverdict distribution:", dict(sorted(tally.items(),
                                                       key=lambda kv: str(kv[0]))))
    spent = {"input": sum(r["usage"]["input"] for r in out if r.get("usage")),
             "output": sum(r["usage"]["output"] for r in out if r.get("usage"))}
    print(f"tokens: {spent['input']:,} in, {spent['output']:,} out = $"
          f"{llm_client.dollars(model, spent['input'], spent['output'], batch=batch):.2f}"
          + (" at batch price" if batch else ""))
    return out

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
                       help="send the questions as batches at half price")
    pilot.add_argument("--mode", choices=["initiate", "two-step"],
                       default="initiate",
                       help="initiate: one step (as before); two-step: an "
                            "idea call, then the design")
    pilot.add_argument("--repairs", type=int, default=0,
                       help="self-repair rounds after a rejected design")
    pilot.add_argument("--workers", type=int, default=8,
                       help="designs worked on at once (two-step/repairs)")
    pilot.add_argument("--grade-slots", type=int, default=None,
                       help="designs checked at once (default: cores - 2)")
    pilot.add_argument("--flush-after", type=float, default=30.0,
                       help="with --batch in the pipeline: send waiting "
                            "questions after this many quiet seconds")
    pilot.add_argument("--distill-log", default=None,
                       help="training examples file (default: next to --log)")
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
        common = dict(grade=True, cmd="pilot", constructs_path=args.constructs,
                      scales_path=args.scales, log_path=args.log,
                      model=args.model, even_seeds=args.even_seeds,
                      distill_path=args.distill_log)
        if args.mode == "two-step" or args.repairs > 0:
            run_pipeline(args.n, mode=args.mode, repairs=args.repairs,
                         workers=args.workers, batch=args.batch,
                         grade_slots=args.grade_slots,
                         flush_after=args.flush_after, **common)
        else:
            run = run_batch if args.batch else run_attempts
            run(args.n, **common)
    elif args.cmd == "collect":
        collect(args.manifest)


if __name__ == "__main__":
    main()
