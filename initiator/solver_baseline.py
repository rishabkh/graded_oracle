"""Solver-direction baseline: strip a triple's invariants, hand the bare
(design, property) to a cold model, ask it to FIND the invariant, and
grade whatever comes back. Solved = NECESSARY.

Standalone by design — imports nothing from run.py and writes its own
log, so it cannot interfere with a generation run.

  venv/bin/python initiator/solver_baseline.py --solver qwen --per-gen 20 --dry
  venv/bin/python initiator/solver_baseline.py --solver qwen --per-gen 20   # QWEN_* env + hwtools
  venv/bin/python initiator/solver_baseline.py --solver opus --per-gen 8    # OpenRouter key + hwtools

Samples are stratified from extender/corpus.jsonl: --per-gen rows from
each generation in --gens (default 0,1,2), spread round-robin across
extension types so one move family cannot dominate a stratum. The
report prints TWO rates per generation: raw NECESSARY, and NECESSARY
conditional on well-formed output (truncations and yosys/contract
rejects excluded) - the first run diverged badly (12 of Qwen's 16
failures were format), and only the conditional rate measures reasoning.

Qwen runs through any OpenAI-compatible endpoint (OpenRouter, Together,
Fireworks, ollama). Set all three explicitly - there are no defaults:
  QWEN_BASE_URL   e.g. https://openrouter.ai/api/v1
  QWEN_API_KEY
  QWEN_MODEL      e.g. qwen/qwen-2.5-coder-32b-instruct
"""
import argparse
import itertools
import json
import re
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from oracle import grade_triple_generated  # noqa: E402

SOURCE_LOG = HERE / "logs" / "attempts.jsonl"
CORPUS = HERE.parent / "extender" / "corpus.jsonl"
OUT_LOG = HERE / "logs" / "solver_baseline.jsonl"
GRADE_KWARGS = dict(timeout_s=120)
OPUS_MODEL = "claude-opus-5"

# The bare prompt — the hard direction, no planting scaffold, no
# exemplars. Byte-identical for both solver models.
SOLVER_PROMPT = """\
Here is a SystemVerilog module. It contains exactly one property, written as an
immediate assertion. The property is true of the design but is NOT provable by
k-induction on its own: a prover starting from an arbitrary state can violate it.

{verilog}

Find strengthening invariant(s): facts about the design's reachable states that
(a) hold in every reachable state, and (b) together with the property make the
set inductive, closing the proof by k-induction.

Reply with JSON: {{"invariants": ["<expr>", ...]}} where each entry is a single
Verilog boolean expression over the module's signals. No other output.

Format rules for each expression - violations make the answer ungradeable:
- plain synthesizable Verilog only: no `->` (write `!a || b` for implication),
  no `$past` or other system functions, no SVA operators, no prose
- complete expressions only, with sized constants (`4'd8`, not `8`)
- signals of this module only, no hierarchical references
"""

INV_SCHEMA = {
    "type": "object",
    "properties": {"invariants": {"type": "array", "items": {"type": "string"}}},
    "required": ["invariants"],
    "additionalProperties": False,
}


class Spinner:
    """Braille-dot spinner with colour and a mm:ss clock, same as
    run.py, copied rather than imported so this file stays fully
    standalone. Silent when stderr is not a TTY."""
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
            sys.stderr.write(f"\r{self.CYAN}{frame}{self.RESET} {self.label} "
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


def strip_comments(verilog):
    """Remove comments so a generated design cannot leak its own
    invariant to the solver through narration."""
    out = re.sub(r"/\*.*?\*/", "", verilog, flags=re.S)
    out = re.sub(r"//[^\n]*", "", out)
    return "\n".join(line.rstrip() for line in out.splitlines() if line.strip())


def stratify(corpus_rows, per_gen, gens, rng):
    """per_gen rows from each generation, drawn round-robin across
    ext_type buckets so no single move family dominates a stratum.
    Deterministic under a seeded rng."""
    out = []
    for gen in gens:
        buckets = {}
        for r in corpus_rows:
            if r.get("generation", 0) == gen:
                buckets.setdefault(r.get("ext_type"), []).append(r)
        for b in buckets.values():
            rng.shuffle(b)
        order = sorted(buckets)
        take = []
        while len(take) < per_gen and any(buckets[k] for k in order):
            for k in order:
                if buckets[k] and len(take) < per_gen:
                    take.append(buckets[k].pop())
        out.extend(take)
    return out


def rates(records):
    """(raw, conditional, n, n_well_formed). Raw counts NECESSARY over
    everything; conditional excludes MALFORMED (truncation, unparseable,
    yosys/contract reject) - the formatting-vs-reasoning split."""
    n = len(records)
    if not n:
        return None, None, 0, 0
    wf = [r for r in records if r.get("verdict") != "MALFORMED"]
    hits = sum(r.get("verdict") == "NECESSARY" for r in records)
    raw = hits / n
    cond = (hits / len(wf)) if wf else None
    return raw, cond, n, len(wf)


def replay_slate(old_records, triples, baseline_id):
    """The exact inputs of a previous baseline run, in its order: the
    corrected before-number must be measured on the same 20 designs the
    polluted number was. Missing pairs abort - a partial replay would
    silently compare different slates."""
    pairs, seen = [], set()
    for r in old_records:
        if r.get("baseline_id") != baseline_id:
            continue
        key = (r.get("source_run_id"), r.get("source_attempt"))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    by_key = {(t["source_run_id"], t["source_attempt"]): t for t in triples}
    missing = [k for k in pairs if k not in by_key]
    if missing:
        sys.exit(f"replay: {len(missing)} source pair(s) not found in "
                 f"{SOURCE_LOG.name}: {missing[:3]}")
    return [by_key[k] for k in pairs]


def load_corpus_rows():
    return [json.loads(l) for l in CORPUS.read_text().splitlines()]


def load_triples(run_id=None, limit=None):
    triples = []
    for line in SOURCE_LOG.read_text().splitlines():
        r = json.loads(line)
        if r.get("verdict") != "NECESSARY":
            continue
        if run_id and r.get("run_id") != run_id:
            continue
        t = json.loads(r["raw_json"])
        triples.append({
            "source_run_id": r.get("run_id"), "source_attempt": r.get("attempt"),
            "top_module": t["top_module"], "clock": t.get("clock", "clk"),
            "antecedents": t.get("antecedents", []),
            "sanity_covers": t.get("sanity_covers", []),
            "verilog": t["verilog"],
            "planted_invariants": t.get("invariants", []),
        })
    return triples[-limit:] if limit else triples


def parse_invariants(text):
    """Lenient JSON extraction (fences, prose around the object).
    Repairs the one illegal escape qwen habitually emits - \' inside
    JSON strings (as if single-quoted) - which valid JSON never
    contains, so the substitution is safe and model-neutral."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0).replace("\\'", "'"))
    except json.JSONDecodeError:
        return None
    inv = obj.get("invariants")
    if isinstance(inv, list) and all(isinstance(x, str) and x.strip() for x in inv):
        return inv
    return None


def solve_opus(prompt_text):
    """Routed through llm_client (Anthropic direct or OpenRouter per
    CLAUDE_PROVIDER), same as every other Opus call in the pipeline."""
    import llm_client
    text, usage, stop = llm_client.call_claude(
        model=OPUS_MODEL, max_tokens=16000, user=prompt_text,
        schema=INV_SCHEMA, effort="high")
    if text is None:
        return None, {"refusal": "refusal",
                      "length": "truncated"}.get(stop, "unparseable")
    return json.loads(text)["invariants"], None


def solve_qwen(prompt_text):
    import os
    from openai import OpenAI
    client = solve_qwen.client or OpenAI(
        base_url=os.environ["QWEN_BASE_URL"], api_key=os.environ["QWEN_API_KEY"])
    solve_qwen.client = client
    for attempt in range(3):
        resp = client.chat.completions.create(
            model=os.environ["QWEN_MODEL"],
            max_tokens=16000,     # 2000 truncated 7 of 20 in the first run
            messages=[{"role": "user", "content": prompt_text}],
        )
        text = resp.choices[0].message.content or ""
        fin = resp.choices[0].finish_reason
        if fin != "error":
            break
        # provider died mid-generation (finish_reason=error is OpenRouter's
        # upstream-failure marker, cuts answers at ~30 chars) - a transport
        # failure, not an answer; retry like a timeout
        time.sleep(2)
    solve_qwen.last_raw = text
    if fin == "length":
        return None, f"truncated at 16000 tokens: {text[-120:]!r}"
    inv = parse_invariants(text)
    if inv is None:
        return None, (f"unparseable output (finish_reason={fin}, "
                      f"{len(text)} chars): {text[:200]!r}")
    return inv, None


solve_qwen.client = None


def classify(record):
    """MALFORMED = the answer never really met the oracle: no parse,
    truncation, refusal, or a with-leg yosys/contract ERROR (the reason
    string names it). Everything else keeps its oracle verdict."""
    if record.get("verdict") == "NO_ANSWER":
        record["verdict"] = "MALFORMED"
    elif "grade is ERROR" in str(record.get("reason", "")):
        record["malformed_reason"] = record["verdict"]
        record["verdict"] = "MALFORMED"
    return record


def report(records):
    print("\n=== per-generation report ===")
    print(f"{'gen':>4} {'n':>3} {'wf':>3} {'raw':>6} {'cond':>6}")
    gens = sorted({r.get("generation") for r in records})
    for g in gens:
        rs = [r for r in records if r.get("generation") == g]
        raw, cond, n, wf = rates(rs)
        fmt = lambda x: "  -  " if x is None else f"{100*x:4.0f}%"
        print(f"{g!s:>4} {n:>3} {wf:>3} {fmt(raw):>6} {fmt(cond):>6}")
    raw, cond, n, wf = rates(records)
    print(f"{'all':>4} {n:>3} {wf:>3} "
          f"{100*(raw or 0):4.0f}% {100*(cond or 0):4.0f}%")
    print("raw = NECESSARY / all;  cond = NECESSARY / well-formed "
          "(MALFORMED excluded: truncation, unparseable, yosys reject)")


def main():
    import random
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["opus", "qwen"], required=True)
    p.add_argument("--per-gen", type=int, default=20,
                   help="stratified sample size per generation")
    p.add_argument("--gens", default="0,1,2",
                   help="comma-separated generations to test")
    p.add_argument("--seed", type=int, default=0,
                   help="sampling seed (same seed = same rows, so both "
                        "solvers see an identical slate)")
    p.add_argument("--ids",
                   help="comma-separated corpus ids: probe exactly these "
                        "rows instead of sampling")
    p.add_argument("--repeat", type=int, default=1,
                   help="run each row this many times (probing whether a "
                        "miss is systematic or stochastic)")
    p.add_argument("--replay",
                   help="baseline_id of a previous run: rerun its exact "
                        "triples instead of sampling the corpus")
    p.add_argument("--dry", action="store_true", help="list inputs, call nothing")
    args = p.parse_args()

    if args.ids:
        wanted = args.ids.split(",")
        by_id = {r["id"]: r for r in load_corpus_rows()}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            sys.exit(f"not in corpus: {missing}")
        rows = [by_id[i] for i in wanted for _ in range(args.repeat)]
        print(f"{len(rows)} probe row(s): {wanted} x{args.repeat}")
    elif args.replay:
        old = [json.loads(l) for l in OUT_LOG.read_text().splitlines()]
        slate = replay_slate(old, load_triples(), args.replay)
        rows = [{"id": f"replay_{t['source_run_id']}_{t['source_attempt']}",
                 "generation": 0, "ext_type": None, **t} for t in slate]
        print(f"{len(rows)} triples replayed from baseline {args.replay}")
    else:
        gens = tuple(int(g) for g in args.gens.split(","))
        rows = stratify(load_corpus_rows(), args.per_gen, gens,
                        random.Random(args.seed))
        print(f"{len(rows)} rows stratified from {CORPUS.name} "
              f"(per-gen {args.per_gen}, gens {gens}, seed {args.seed})")
    if args.dry:
        for r in rows:
            invs = r.get("invariants", r.get("planted_invariants", []))
            print(f"  g{r['generation']} {r['id']:8s} {r['top_module']:26s} "
                  f"{(r.get('ext_type') or 'g0'):10s} "
                  f"clauses={len(invs)} "
                  f"lines={len(r['verilog'].splitlines())}")
        return

    import shutil
    if shutil.which("sby") is None:
        sys.exit("sby not on PATH - activate hwtools in this shell first, "
                 "nothing was called")
    if args.solver == "qwen":
        import os
        missing = [v for v in ("QWEN_BASE_URL", "QWEN_API_KEY", "QWEN_MODEL")
                   if not os.environ.get(v)]
        if missing:
            sys.exit(f"missing env: {', '.join(missing)} - export them "
                     "in this shell first, nothing was called")
    solve = solve_opus if args.solver == "opus" else solve_qwen
    baseline_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    records = []
    for i, t in enumerate(rows):
        prompt_text = SOLVER_PROMPT.format(verilog=strip_comments(t["verilog"]))
        record = {"baseline_id": baseline_id, "solver": args.solver, "i": i,
                  "triple_id": t["id"], "generation": t["generation"],
                  "ext_type": t.get("ext_type"), "seed": args.seed,
                  "top_module": t["top_module"],
                  "planted_invariants": t.get("invariants",
                                               t.get("planted_invariants"))}
        t0 = time.monotonic()
        try:
            with Spinner(f"[{i}] {args.solver} solving g{t['generation']} "
                         f"{t['top_module']}"):
                invariants, err = solve(prompt_text)
        except Exception as exc:
            invariants, err = None, f"{type(exc).__name__}: {exc}"
        record["solve_wall_s"] = round(time.monotonic() - t0, 2)

        if invariants is None:
            record["error"] = err
            record["raw_text"] = getattr(solve, "last_raw", None)
            record["verdict"] = "NO_ANSWER"
        else:
            record["solver_invariants"] = invariants
            payload = {"verilog": t["verilog"], "top_module": t["top_module"],
                       "clock": t["clock"], "antecedents": t["antecedents"],
                       "sanity_covers": t["sanity_covers"],
                       "invariants": invariants}
            with Spinner(f"[{i}] oracle grading ({t['top_module']})"):
                result = grade_triple_generated(json.dumps(payload),
                                                **GRADE_KWARGS)
            record["verdict"] = result.verdict.name
            record["reason"] = result.reason
            record["result"] = asdict(result)
        classify(record)
        records.append(record)
        note = str(record.get("error") or record.get("reason") or "")[:60]
        print(f"[{i}] g{t['generation']} {t['top_module']:24s} "
              f"{record['verdict']:12s} ({record['solve_wall_s']}s) {note}")

        OUT_LOG.parent.mkdir(exist_ok=True)
        with OUT_LOG.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    report(records)


if __name__ == "__main__":
    main()
