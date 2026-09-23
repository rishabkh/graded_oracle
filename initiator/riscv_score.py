"""Score a model on the riscv-formal problem set.

The survey found 144 checks across nerv, serv and picorv32 whose proofs
do not close by k-induction alone. This asks a model for strengthening
invariants on each, injects them into the core (riscv_adapter), runs the
check, and tallies the verdicts. That number, before and after training,
is the result the project exists to produce.

Conditions, carried into every record and never mixed silently:

  native   the whole design, every assertion        the real task
  focused  the whole design, one assertion          is the target buried?
  cut      the cone-sliced design (slice_check)     does length hurt?

The model's invariants are injected INSIDE the core module, so the
prompt asks for facts about the core's own registers. That differs from
the corpus prompt, which asks for top-level names, because here the top
level is a testbench wrapper the model should not reason about.

  export QWEN_BASE_URL=http://holygpu8a22406.rc.fas.harvard.edu:8000/v1
  export QWEN_API_KEY=none QWEN_MODEL=llm
  venv/bin/python initiator/riscv_score.py --core nerv --n 5
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from riscv_adapter import CORES, RISCV, run_check            # noqa: E402
from solver_baseline import parse_invariants                 # noqa: E402

SURVEY_LOG = HERE / "logs" / "riscv_survey.jsonl"
OUT_LOG = HERE / "logs" / "riscv_score.jsonl"

PROMPT = """\
Here is a RISC-V core and one of the properties riscv-formal checks against it.

--- the core ---
{core}

--- the property being checked ---
{property}

This property is TRUE of the core but is NOT provable by k-induction on its
own: a prover starting from an arbitrary state can violate it.

Find strengthening invariants: facts about the core's reachable states that
(a) hold in every reachable state, and (b) together with the property close
the proof by k-induction.

Reply with JSON: {{"invariants": ["<expr>", ...]}} where each entry is a single
Verilog boolean expression over signals declared INSIDE module {core_module}.
No other output.

Format rules - violations make the answer ungradeable:
- plain synthesizable Verilog only. There is NO implication operator: `->`,
  `==>`, `|->` and `|=>` are all rejected. Write `!a || b` instead.
- no `$past` or other system functions, no SVA operators, no prose
- complete expressions with sized constants (`4'd8`, not `8`)
- names declared inside module {core_module} only: not the testbench, not the
  wrapper, no hierarchical dotted paths
"""


def malformed(exprs):
    """Anything the checker cannot take, caught before a proof is spent.

    Measured 22 Sep 2026: one serv check drew 9 invariants and then 15,
    every one written with `==>`. Handing those to the prover produced
    ERROR, which reads like a wrong answer instead of an unusable one."""
    for e in exprs:
        if "==>" in e or re.search(r"(?<![<>=!-])->", e):
            return True
        if e.count("(") != e.count(")") or e.count("[") != e.count("]"):
            return True
        if re.search(r"\$\w+\s*\(", e):        # $past and friends
            return True
        if "|->" in e or "|=>" in e or "##" in e:
            return True
    return False


def usable(exprs):
    """The expressions worth proving: drop the malformed ones rather than
    failing the whole answer, and if nothing survives the answer is no
    answer at all."""
    return [e for e in exprs if not malformed([e])]


def served_model(listing):
    """What the endpoint is really serving. Both the base model and the
    fine-tuned one get served under the alias 'llm', so the alias alone
    cannot tell a before-run from an after-run. vLLM reports the real
    path or repo id in `root`."""
    data = getattr(listing, "data", None) or []
    if not data:
        return None
    first = data[0]
    return getattr(first, "root", None) or getattr(first, "id", None)


def credentials_ready():
    """A remote endpoint needs a real key. OpenRouter serves its model
    list to anyone, so the readiness probe passes and then every question
    comes back 401 - ten rows of NO_ANSWER that read exactly like a model
    scoring zero. A local server needs no key at all."""
    url = os.environ.get("QWEN_BASE_URL", "")
    key = os.environ.get("QWEN_API_KEY", "")
    local = any(h in url for h in ("localhost", "127.0.0.1", ".rc.fas."))
    if not local and not key.strip():
        return False, ("QWEN_API_KEY is empty and the endpoint is remote; "
                       "every request would be refused")
    return True, "ok"


def endpoint_ready(probe):
    """Is the model actually answering? A run started against a server
    that is still loading writes NO_ANSWER rows indistinguishable from
    real failures, which is how a results file gets poisoned."""
    try:
        probe()
        return True, "ready"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def summarise(rows, run_id):
    """One run's result, in the form a paper table wants: what was
    solved, out of how many, and how the rest failed."""
    R = [r for r in rows if r.get("run_id") == run_id]
    if not R:
        raise ValueError(f"no rows for run {run_id!r}")
    counts = Counter(r["verdict"] for r in R)
    return {"run_id": run_id, "core": R[0].get("core"),
            "condition": R[0].get("condition"), "model": R[0].get("model"),
            "total": len(R), "proved": counts.get("PROVED", 0),
            "counts": dict(counts)}


def core_of_run(rows, run_id, dirs):
    """Which core produced a survey run. Check names repeat across cores,
    so a run belongs to the core whose directory holds ALL of its checks.
    Returns None rather than guessing."""
    names = {r["check"] for r in rows if r.get("run_id") == run_id}
    if not names:
        return None
    for core, have in dirs.items():
        if names <= have:
            return core
    return None


def problem_set(rows, core, dirs):
    """The checks of one core that needed a strengthening invariant."""
    out = set()
    for run_id in {r.get("run_id") for r in rows}:
        if core_of_run(rows, run_id, dirs) != core:
            continue
        out |= {r["check"] for r in rows
                if r.get("run_id") == run_id
                and r.get("verdict") == "NEEDS_INVARIANT"}
    return sorted(out)


def check_dirs():
    """{core: set of check names on disk}, from the generated checks."""
    out = {}
    for core in CORES:
        d = RISCV / f"cores/{core}/checks"
        if d.exists():
            out[core] = {p.stem for p in d.glob("*.sby")
                         if "_prove" not in p.stem}
    return out


def build_prompt(core_text, property_text, core_module):
    return PROMPT.format(core=core_text, property=property_text,
                         core_module=core_module)


def check_property_text(core, check):
    """The check's own source, which is where the assertions live."""
    src = RISCV / f"cores/{core}/checks/{check}_prove10/src"
    for name in (f"{check}.sv",):
        p = src / name
        if p.exists():
            return p.read_text()
    return "(check source not unpacked; run the survey first)"


def reply_meta(reply):
    """Why the answer looks the way it does. An empty answer whose
    finish_reason is 'length' and whose reasoning_tokens equal the whole
    budget is a budget problem, not a refusal and not ignorance."""
    choice = reply.choices[0]
    usage = getattr(reply, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    message = choice.message
    return {
        "finish_reason": choice.finish_reason,
        "provider": getattr(reply, "provider", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "had_reasoning_text": bool(getattr(message, "reasoning", None)),
        "refusal": str(getattr(message, "refusal", None) or "")[:200] or None,
    }


def solve(prompt, max_tokens=4000, reasoning=None):
    """One call to whatever OpenAI-compatible endpoint is configured.

    A thinking model spends output tokens on thought before it writes
    anything: measured 23 Sep 2026, Opus used all 4000 on reasoning and
    returned an empty answer, logged as NO_ANSWER and read as failure."""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["QWEN_BASE_URL"],
                    api_key=os.environ.get("QWEN_API_KEY", "none"))
    kwargs = dict(model=os.environ["QWEN_MODEL"], max_tokens=max_tokens,
                  messages=[{"role": "user", "content": prompt}])
    if reasoning:
        kwargs["extra_body"] = {"reasoning": reasoning}
    r = client.chat.completions.create(**kwargs)
    text = r.choices[0].message.content or ""
    return parse_invariants(text), text, reply_meta(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core", choices=sorted(CORES), required=True)
    p.add_argument("--n", type=int, default=0, help="0 means every problem")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--condition", default="native",
                   choices=["native", "focused", "cut"])
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="output budget; a thinking model needs far more "
                        "than the answer itself, 16000 for Opus")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["none", "low", "medium", "high"],
                   help="bound the thinking so some budget is left for "
                        "the answer; 'none' turns it off entirely")
    p.add_argument("--dry", action="store_true",
                   help="print the problem set and one prompt, call nothing")
    p.add_argument("--report", action="store_true",
                   help="summarise the runs already logged, call nothing")
    args = p.parse_args()

    if args.report:
        rows = [json.loads(l) for l in
                OUT_LOG.read_text().splitlines() if l.strip()]
        seen = []
        for r in rows:
            if r.get("run_id") not in seen:
                seen.append(r.get("run_id"))
        print(f"{'run':22s} {'core':9s} {'condition':10s} {'model':6s} "
              f"{'solved':>7s}   failures")
        for run_id in seen:
            s = summarise(rows, run_id)
            fails = ", ".join(f"{k} {v}" for k, v in
                              sorted(s["counts"].items()) if k != "PROVED")
            print(f"{str(s['run_id']):22s} {str(s['core']):9s} "
                  f"{str(s['condition']):10s} {str(s['model']):6s} "
                  f"{s['proved']:3d}/{s['total']:<3d}   {fails}")
        return

    for var in ("QWEN_BASE_URL", "QWEN_MODEL"):
        if not args.dry and var not in os.environ:
            sys.exit(f"{var} is not set - point it at the served model")

    rows = [json.loads(l) for l in
            SURVEY_LOG.read_text().splitlines() if l.strip()]
    problems = problem_set(rows, args.core, check_dirs())
    if args.n:
        problems = problems[:args.n]
    core_text = (RISCV / CORES[args.core]["file"]).read_text()
    module = CORES[args.core]["module"]
    print(f"{args.core}: {len(problems)} problems, condition "
          f"{args.condition}, core is {len(core_text.splitlines())} lines")

    if args.dry:
        print("  " + ", ".join(problems[:8]) + (" ..." if len(problems) > 8
                                                else ""))
        if problems:
            prompt = build_prompt(core_text,
                                  check_property_text(args.core, problems[0]),
                                  module)
            print(f"\n  prompt for {problems[0]}: {len(prompt)} chars "
                  f"(~{len(prompt) * 10 // 36} tokens)")
        return

    listing = {}

    def probe():
        from openai import OpenAI
        listing["models"] = OpenAI(
            base_url=os.environ["QWEN_BASE_URL"],
            api_key=os.environ.get("QWEN_API_KEY", "none")).models.list()

    ok, why = credentials_ready()
    if not ok:
        sys.exit(f"{why}.\nNothing was run and nothing was logged.")

    ok, why = endpoint_ready(probe)
    if not ok:
        sys.exit(f"the model endpoint is not answering ({why}).\n"
                 "Nothing was run and nothing was logged. Check the server "
                 "has printed 'Application startup complete', and that the "
                 "tunnel points at the right node.")

    reasoning = None
    if args.reasoning_effort == "none":
        reasoning = {"enabled": False}
    elif args.reasoning_effort:
        reasoning = {"effort": args.reasoning_effort}

    serving = served_model(listing.get("models")) if listing else None
    print(f"  endpoint is serving: {serving or 'unknown'}")

    run_id = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
    tally = Counter()
    OUT_LOG.parent.mkdir(exist_ok=True)
    for i, check in enumerate(problems):
        prompt = build_prompt(core_text,
                              check_property_text(args.core, check), module)
        t0 = time.monotonic()
        try:
            invariants, raw, meta = solve(prompt, args.max_tokens,
                                          reasoning)
        except Exception as exc:
            invariants, raw = None, f"{type(exc).__name__}: {exc}"
            meta = {"error": type(exc).__name__}
        good = usable(invariants or [])
        if not good:
            rec = {"core": args.core, "check": check, "verdict": "NO_ANSWER",
                   "condition": args.condition, "raw": raw[:400],
                   "n_invariants": 0,
                   "dropped_malformed": len(invariants or [])}
        else:
            rec = run_check(args.core, check, good, args.k,
                            args.timeout, args.condition)
            rec["dropped_malformed"] = len(invariants or []) - len(good)
        rec["reply"] = meta
        rec.update(run_id=run_id, model=os.environ.get("QWEN_MODEL"),
                   served_model=serving,
                   solve_wall_s=round(time.monotonic() - t0, 1))
        tally[rec["verdict"]] += 1
        with OUT_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [{i}] {check:28s} {rec['verdict']:16s} "
              f"{rec.get('n_invariants', 0)} invariants")
    solved = tally["PROVED"]
    print(f"\n{args.core} {args.condition}: {solved}/{len(problems)} proved")
    print(dict(tally))


if __name__ == "__main__":
    main()
