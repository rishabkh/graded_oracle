"""Solver-direction baseline: strip a triple's invariants, hand the bare
(design, property) to a cold model, ask it to FIND the invariant, and
grade whatever comes back. Solved = NECESSARY.

Standalone by design — imports nothing from run.py and writes its own
log, so it cannot interfere with a generation run.

  venv/bin/python initiator/solver_baseline.py --solver opus --dry     # list inputs, no calls
  venv/bin/python initiator/solver_baseline.py --solver opus           # needs ANTHROPIC_API_KEY + hwtools
  venv/bin/python initiator/solver_baseline.py --solver qwen           # needs QWEN_* env vars + hwtools

Qwen runs through any OpenAI-compatible endpoint (OpenRouter, Together,
Fireworks, ollama). Set:
  QWEN_BASE_URL   e.g. https://openrouter.ai/api/v1
  QWEN_API_KEY
  QWEN_MODEL      e.g. qwen/qwen-2.5-coder-32b-instruct
and `pip install openai` (do this AFTER any generation run finishes).
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
"""

INV_SCHEMA = {
    "type": "object",
    "properties": {"invariants": {"type": "array", "items": {"type": "string"}}},
    "required": ["invariants"],
    "additionalProperties": False,
}


class Spinner:
    """Same spinner as run.py, copied rather than imported so this file
    stays fully standalone. Silent when stderr is not a TTY."""
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


def strip_comments(verilog):
    """Remove comments so a generated design cannot leak its own
    invariant to the solver through narration."""
    out = re.sub(r"/\*.*?\*/", "", verilog, flags=re.S)
    out = re.sub(r"//[^\n]*", "", out)
    return "\n".join(line.rstrip() for line in out.splitlines() if line.strip())


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
    """Lenient JSON extraction (fences, prose around the object)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    inv = obj.get("invariants")
    if isinstance(inv, list) and all(isinstance(x, str) and x.strip() for x in inv):
        return inv
    return None


def solve_opus(prompt_text):
    import anthropic
    client = solve_opus.client or anthropic.Anthropic()
    solve_opus.client = client
    resp = client.messages.create(
        model=OPUS_MODEL, max_tokens=8000,
        output_config={"format": {"type": "json_schema", "schema": INV_SCHEMA}},
        messages=[{"role": "user", "content": prompt_text}],
    )
    if resp.stop_reason == "refusal":
        return None, "refusal"
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)["invariants"], None


solve_opus.client = None


def solve_qwen(prompt_text):
    import os
    from openai import OpenAI
    client = solve_qwen.client or OpenAI(
        base_url=os.environ["QWEN_BASE_URL"], api_key=os.environ["QWEN_API_KEY"])
    solve_qwen.client = client
    resp = client.chat.completions.create(
        model=os.environ["QWEN_MODEL"],
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt_text}],
    )
    text = resp.choices[0].message.content or ""
    inv = parse_invariants(text)
    if inv is None:
        return None, f"unparseable output: {text[:200]!r}"
    return inv, None


solve_qwen.client = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["opus", "qwen"], required=True)
    p.add_argument("--run-id", help="only triples from this generation run")
    p.add_argument("--n", type=int,
                   help="use only the LAST n triples from the log")
    p.add_argument("--dry", action="store_true", help="list inputs, call nothing")
    args = p.parse_args()

    triples = load_triples(args.run_id, args.n)
    print(f"{len(triples)} NECESSARY triples loaded from {SOURCE_LOG.name}")
    if args.dry:
        for t in triples:
            print(f"  {t['source_run_id']}/{t['source_attempt']:>3} "
                  f"{t['top_module']:24s} planted_clauses={len(t['planted_invariants'])}")
        return

    solve = solve_opus if args.solver == "opus" else solve_qwen
    baseline_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    solved = 0
    tally = {}
    for i, t in enumerate(triples):
        prompt_text = SOLVER_PROMPT.format(verilog=strip_comments(t["verilog"]))
        record = {"baseline_id": baseline_id, "solver": args.solver, "i": i,
                  "source_run_id": t["source_run_id"],
                  "source_attempt": t["source_attempt"],
                  "top_module": t["top_module"],
                  "planted_invariants": t["planted_invariants"]}
        t0 = time.monotonic()
        try:
            with Spinner(f"[{i}] {args.solver} searching for invariant "
                         f"({t['top_module']})"):
                invariants, err = solve(prompt_text)
        except Exception as exc:
            invariants, err = None, f"{type(exc).__name__}: {exc}"
        record["solve_wall_s"] = round(time.monotonic() - t0, 2)

        if invariants is None:
            record["error"] = err
            record["verdict"] = "NO_ANSWER"
        else:
            record["solver_invariants"] = invariants
            payload = {"verilog": t["verilog"], "top_module": t["top_module"],
                       "clock": t["clock"], "antecedents": t["antecedents"],
                       "sanity_covers": t["sanity_covers"], "invariants": invariants}
            with Spinner(f"[{i}] oracle grading ({t['top_module']})"):
                result = grade_triple_generated(json.dumps(payload),
                                                **GRADE_KWARGS)
            record["verdict"] = result.verdict.name
            record["reason"] = result.reason
            record["result"] = asdict(result)
            solved += result.verdict.name == "NECESSARY"
        tally[record["verdict"]] = tally.get(record["verdict"], 0) + 1
        print(f"[{i}] {t['top_module']:24s} {record['verdict']:12s} "
              f"({record['solve_wall_s']}s) inv={invariants}")

        OUT_LOG.parent.mkdir(exist_ok=True)
        with OUT_LOG.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    print(f"\nsolver={args.solver}: solved {solved}/{len(triples)} "
          f"({100 * solved / max(len(triples), 1):.0f}%)")
    print("verdicts:", dict(sorted(tally.items())))


if __name__ == "__main__":
    main()
