"""The untrained model writes its own fake state, reasoning and answer
for corpus designs; a try is kept only when the proof tool backs both.

Why: v3 was trained on reasoning Opus wrote while it held the answer. It
learned to act out knowing the answer: on new designs it talks about an
"R" the question never mentions, makes the answer up inside its story,
and scored far worse. So here the model that will be trained writes the
reasoning itself, with no answer in front of it, and a try is kept only if
  (a) the proof closes with its own invariants (NECESSARY), and
  (b) its fake state is a real counterexample to induction against those
      same invariants (training/cti_check.py says REAL).
That makes the kept reasoning checked and correct. It does not show that
the reasoning is what produced the answer.

The question is the solver baseline's (initiator/solver_baseline.py),
design comments stripped, so the model sees what it is scored on. Only
the reply format changes: fake state first, then reasoning, then
invariants. The prompt never calls the answer R or a clause.

The model is served by vLLM on the cluster and reached through a tunnel
at QWEN_BASE_URL; grading runs here. One request per design asks for all
--tries replies at once. A served path with runs/ in it is a fine-tune
trained on these very designs, which would recall answers rather than
search for them, so it is refused unless --allow-trained.

Every try, kept or not, is one line in extender/logs/self_reason.jsonl.
A design with --tries lines already is skipped next time; one with fewer
is asked only for the rest.

  source ~/Desktop/HarvardResearch/oss-cad-suite/environment
  export QWEN_BASE_URL=http://localhost:8000/v1
  venv/bin/python training/self_reason.py --dry          # no requests
  venv/bin/python training/self_reason.py                # 7 per generation
  venv/bin/python training/self_reason.py --report       # no requests
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extender"))

import llm_client                                             # noqa: E402
from distractor import Spinner                                # noqa: E402
from solver_baseline import (GRADE_KWARGS, SOLVER_PROMPT,     # noqa: E402
                             out_of_scope, strip_comments, stratify)

CORPUS = HERE.parent / "extender" / "corpus.jsonl"
LOG = HERE.parent / "extender" / "logs" / "self_reason.jsonl"

GENS = (0, 1, 2, 3, 4, 5, 6)
# eight replies of up to 8000 tokens can outlast the client's default ten
# minutes, and a timed-out request is sent again from the start
REQUEST_TIMEOUT_S = 1800
MAX_FAILED_REQUESTS = 3       # in a row; the endpoint or tunnel is gone

# Replaces the solver prompt's reply paragraph. The order is the point:
# the fake state is found first and the invariants come last, written to
# rule it out.
REPLY = """\
Reply with JSON: {{"fake_state": [{{"signal": "...", "value": "..."}}, ...],
"reasoning": "...", "invariants": ["<expr>", ...]}}
No other output. Write the three fields in this order:

1. fake_state: first find a fake state, a state of the design in which every
   assertion holds but which the design can never reach in real operation,
   and from which ONE clock step breaks an assertion. A prover starting from
   an arbitrary state can start there, which is why the assertions cannot be
   proven on their own.
2. reasoning: then one or two sentences: why the assertions hold in that
   state, why the design can never reach it, and what the next step does
   that breaks an assertion.
3. invariants: last, facts true in every reachable state that rule that
   state out, so at least one of them is false in your fake state. Each
   entry is a single Verilog boolean expression over the module's signals."""

# The two-moments rule is regen_reasoning.py's, reworded for a prompt
# that has no S, P or R.
STATE_RULES = """
Rules for fake_state:
- top-level signal names only, as for the invariants: never a name inside a
  sub-module instance
- every value is a sized constant, e.g. 4'd3 or 1'b0
- give each signal whole, never a bit slice such as `c[1]`; a memory word is
  named like `mem[1]`
- every signal needed to evaluate the assertions and your invariants must
  appear, including any register that guards an assertion
- Two moments, kept apart. The fake state is where the design sits: it can
  stay there for many cycles with its enable held low. The step that breaks
  an assertion is the next clock step, taken with the enable ON. If you list
  an input in fake_state, give its value on that breaking step, never its
  idle value (that step would change nothing). You may leave inputs out;
  every input value is then tried.
"""


def _prompt():
    start = SOLVER_PROMPT.index('Reply with JSON: {{"invariants"')
    end = SOLVER_PROMPT.index("No other output.", start) + len(
        "No other output.")
    return SOLVER_PROMPT[:start] + REPLY + SOLVER_PROMPT[end:] + STATE_RULES


PROMPT = _prompt()
FIELDS = ("fake_state", "reasoning", "invariants")

_log_lock = threading.Lock()


def build_prompt(row):
    return PROMPT.format(verilog=strip_comments(row["verilog"]))


def grade(row, invariants):
    """The proof verdict, graded exactly as the solver baseline grades.
    Tests replace this function."""
    from oracle import grade_triple_generated
    payload = {"verilog": row["verilog"], "top_module": row["top_module"],
               "clock": row.get("clock", "clk"),
               "antecedents": row.get("antecedents", []),
               "sanity_covers": row.get("sanity_covers", []),
               "invariants": invariants}
    result = grade_triple_generated(json.dumps(payload), **GRADE_KWARGS)
    # the whole result, proof folders included (8 Oct 2026: only the
    # verdict was kept)
    out = {"verdict": result.verdict.name, "reason": result.reason,
           "result": json.loads(json.dumps(asdict(result), default=str))}
    # on a big design a too-weak answer fails induction quickly, then the
    # base case runs out of time; that is a "no", not a tool failure
    runs = result.with_invariants.runs if result.with_invariants else []
    if any(n.startswith("induction_failed_before_timeout")
           for ev in runs for n in ev.notes):
        out["induction_failed"] = True
    return out


def check(row, state):
    """The fake-state checker. Imported late so tests can replace this
    function without training/cti_check.py existing. Its proofs are kept
    next to the log (8 Oct 2026; they used to be made and deleted in a
    temporary folder)."""
    import cti_check
    keep = Path(LOG).parent / f"{Path(LOG).stem}_proofs"
    out = cti_check.check_state(row, state, keep_dir=keep)
    if out.get("proof_dir"):     # from the log's folder, as one_step gives it
        out["proof_dir"] = f"{keep.name}/{out['proof_dir']}"
    return out


def make_client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ["QWEN_BASE_URL"],
                  api_key=os.environ.get("QWEN_API_KEY", "none"),
                  timeout=REQUEST_TIMEOUT_S)


def preflight():
    """Why grading cannot run, or None. Asked before any request, so a
    missing prover does not turn every try into a tool failure."""
    if shutil.which("sby") is None:
        return ("sby is not on PATH. Run first: "
                "source ~/Desktop/HarvardResearch/oss-cad-suite/environment")
    try:
        import cti_check                                      # noqa: F401
    except ImportError as e:
        return f"training/cti_check.py cannot be imported ({e})"
    return None


def model_name():
    return os.environ.get("QWEN_MODEL", "llm")


def served_root(listing, name):
    """What the endpoint serves under the name we ask for, or None. vLLM
    lists the base model and each adapter as their own entries, so the
    entry we will ask is the one to look at; trusting the name alone
    would pass an adapter from runs/ served as 'llm'."""
    for m in getattr(listing, "data", None) or []:
        if getattr(m, "id", None) == name:
            return getattr(m, "root", None) or name
    return None


def refusal(served, allow_trained):
    """Why this endpoint must not be used, or None."""
    if allow_trained:
        return None
    if not served:
        return (f"the endpoint lists no model named {model_name()} (from "
                "QWEN_MODEL, default llm), so cannot tell what it is serving "
                "or rule out a fine-tune. Check QWEN_MODEL, or pass "
                "--allow-trained to run anyway.")
    if "runs/" in served:
        return (f"the endpoint is serving {served}, a fine-tune (its path "
                "has runs/ in it). It was trained on these very designs, "
                "so it would recall answers instead of searching for them. "
                "Serve the base model, or pass --allow-trained.")
    return None


def parse_reply(text):
    """({fake_state, reasoning, invariants}, problem). A field that is
    missing or malformed is None, and problem names it; problem is None
    when all three are well formed. Fields are judged one by one so a
    right answer with a broken fake state is still graded."""
    fields = dict.fromkeys(FIELDS)
    # qwen writes \' inside JSON strings, which valid JSON never has
    text = (text or "").replace("\\'", "'")
    m = re.search(r"\{.*\}", text, flags=re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = llm_client.extract_json(text)
    if not isinstance(obj, dict):
        return fields, "no JSON object in the reply"
    state = obj.get("fake_state")
    if isinstance(state, list) and state and all(
            isinstance(p, dict) and isinstance(p.get("signal"), str)
            and isinstance(p.get("value"), str) for p in state):
        fields["fake_state"] = state
    reasoning = obj.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        fields["reasoning"] = reasoning
    inv = obj.get("invariants")
    if isinstance(inv, list) and inv and all(
            isinstance(x, str) and x.strip() for x in inv):
        fields["invariants"] = inv
    bad = [k for k in FIELDS if fields[k] is None]
    return fields, ("bad or missing: " + ", ".join(bad)) if bad else None


def safe_grade(row, invariants):
    """A crash in the grader is a tool failure, not a verdict."""
    try:
        return grade(row, invariants)
    except Exception as e:
        return {"verdict": "grade_error", "reason": f"{type(e).__name__}: {e}"}


def safe_check(row, state):
    try:
        return check(row, state)
    except Exception as e:
        return {"ok": False, "verdict": "check_error",
                "detail": f"{type(e).__name__}: {e}"}


def judge(row, text, finish):
    """Both checks for one try. The fake state is checked against the
    try's OWN invariants, never the corpus answer, and is checked even
    when the answer is wrong, so "right answer, wrong reasoning" and
    "real fake state, wrong answer" can both be counted."""
    fields, problem = parse_reply(text)
    inv, state = fields["invariants"], fields["fake_state"]
    fake = None
    if inv is None:
        cut = finish == "length"
        proof = {"verdict": "TRUNCATED" if cut else "UNPARSEABLE",
                 "reason": problem + (", cut off at the token limit"
                                      if cut else "")}
    else:
        outside = sorted(out_of_scope(inv, row["verilog"], row["top_module"]))
        if outside:
            why = f"not signals of the top module: {', '.join(outside)}"
            proof = {"verdict": "OUT_OF_SCOPE", "reason": why}
            if state:
                # the checker would read an unknown name as a free wire,
                # which can make any state look real, so it is not asked
                fake = {"ok": False, "verdict": "OUT_OF_SCOPE",
                        "detail": "not checked: the invariants name " + why}
        else:
            proof = safe_grade(row, inv)
            if state:
                own = {"id": row["id"], "verilog": row["verilog"],
                       "top_module": row["top_module"], "invariants": inv}
                fake = safe_check(own, state)
    kept = (problem is None and proof["verdict"] == "NECESSARY"
            and fake is not None and fake.get("verdict") == "REAL")
    return dict(fields, parse_problem=problem, proof=proof,
                fake_check=fake, kept=kept)


def append(path, record):
    with _log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as f:
            # a killed run can leave a cut line with no newline; start a
            # fresh line so this record is not glued onto it and lost
            if f.seek(0, 2):
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write((json.dumps(record, default=str) + "\n").encode())


def read_log(path):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a line cut short by a killed run
    return out


def now():
    return datetime.now(timezone.utc).isoformat()


def say(line):
    """A progress line, printed clear of the spinner: wipe the spinner's
    line first, and the spinner redraws below on its next frame."""
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
    print(line, flush=True)


def progress_label(asked, designs, graded, tries, kept):
    return (f"self-reasoning: {asked}/{designs} designs asked, "
            f"{graded}/{tries} tries graded, {kept} kept")


def select(rows, per_gen=7, gens=GENS, seed=0, ids=None):
    """The designs to ask about: --ids in the order given, else per_gen
    from each generation, spread across extension types, fixed by seed."""
    if ids:
        wanted = list(dict.fromkeys(i.strip() for i in ids.split(",")
                                    if i.strip()))
        by_id = {r["id"]: r for r in rows}
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            print(f"unknown ids: {', '.join(unknown)}", file=sys.stderr)
            sys.exit(2)
        return [by_id[i] for i in wanted]
    return stratify(rows, per_gen, tuple(gens), random.Random(seed))


def logged_tries(lines):
    """How many tries each design has, and its highest try number."""
    count, last = Counter(), Counter()
    for r in lines:
        count[r["id"]] += 1
        last[r["id"]] = max(last[r["id"]], r.get("try") or 0)
    return count, last


def ask(client, row, n, max_tokens, temperature):
    kwargs = dict(model=model_name(), n=n,
                  max_tokens=max_tokens,
                  messages=[{"role": "user", "content": build_prompt(row)}])
    if temperature is not None:
        kwargs["temperature"] = temperature
    return client.chat.completions.create(**kwargs)


def usage_of(reply, tries):
    """Token counts for the whole request: the completion count covers
    every try in it, since the API gives no per-try split."""
    u = getattr(reply, "usage", None)
    return {"prompt_tokens": getattr(u, "prompt_tokens", None),
            "request_completion_tokens": getattr(u, "completion_tokens", None),
            "request_tries": tries}


def run(rows, client, served, args):
    count, last = logged_tries(read_log(LOG))
    todo = [(r, args.tries - count[r["id"]]) for r in rows
            if count[r["id"]] < args.tries]
    total_tries = sum(want for _, want in todo)
    print(f"{len(todo)} designs to ask, {total_tries} tries "
          f"({len(rows) - len(todo)} already fully tried, skipped)")
    running = {"asked": 0, "graded": 0, "kept": 0, "done": 0}
    left, kept_by_id = {}, Counter()
    spinner = Spinner(progress_label(0, len(todo), 0, total_tries, 0),
                      always=True)

    def refresh():
        spinner.label = progress_label(running["asked"], len(todo),
                                       running["graded"], total_tries,
                                       running["kept"])

    def one(row, text, finish, try_no, usage):
        out = judge(row, text, finish)
        append(LOG, {
            "id": row["id"], "generation": row.get("generation"),
            "ext_type": row.get("ext_type"), "try": try_no,
            "served_model": served, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "finish": finish, "raw": text,
            "fake_state": out["fake_state"], "reasoning": out["reasoning"],
            "invariants": out["invariants"],
            "parse_problem": out["parse_problem"], "proof": out["proof"],
            "fake_check": out["fake_check"], "kept": out["kept"],
            "timestamp": now(), "usage": usage})
        with _log_lock:
            running["graded"] += 1
            running["kept"] += out["kept"]
            kept_by_id[row["id"]] += out["kept"]
            left[row["id"]] -= 1
            if not left[row["id"]]:
                running["done"] += 1
                say(f"[{running['done']}/{len(todo)}] {row['id']} "
                    f"g{row.get('generation')}  "
                    f"{kept_by_id[row['id']]} kept")
            refresh()

    failed = 0
    futures = []
    with spinner, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, want in todo:
            try:
                reply = ask(client, row, want, args.max_tokens,
                            args.temperature)
            except Exception as e:
                failed += 1
                say(f"{row['id']}: request failed ({type(e).__name__}: {e}); "
                    "nothing logged for it, a rerun asks again")
                if failed >= MAX_FAILED_REQUESTS:
                    say(f"{failed} requests failed in a row; stopping. "
                        "Check the server and the tunnel, then rerun.")
                    break
                continue
            failed = 0
            choices = list(reply.choices or [])[:want]
            if len(choices) < want:
                say(f"{row['id']}: the endpoint returned {len(choices)} of "
                    f"{want} replies; a rerun asks for the rest")
            usage = usage_of(reply, len(choices))
            with _log_lock:
                running["asked"] += 1
                left[row["id"]] = len(choices)
                refresh()
            for k, choice in enumerate(choices):
                futures.append(pool.submit(
                    one, row, choice.message.content or "",
                    choice.finish_reason, last[row["id"]] + k + 1, usage))
        for f in futures:
            f.result()        # a bug in a worker should not pass silently


def proof_tool_failure(r):
    """The grader timed out, crashed, or could not finish the run without
    the invariants: no verdict on the try's answer. A with-run ERROR is
    left out; that is mostly yosys refusing a malformed answer. So is a
    timeout after induction already failed."""
    p = r.get("proof") or {}
    return (p.get("verdict") in ("grade_error", "INCONCLUSIVE")
            or ("grade is TIMEOUT" in (p.get("reason") or "")
                and not p.get("induction_failed")))


def fake_tool_failure(r):
    return (r.get("fake_check") or {}).get("verdict") in ("check_error",
                                                          "TIMEOUT")


def report(log_path, ids=None):
    """Yield by generation, both verdict counts, and the two-by-two of
    proof against fake state, from the log alone; makes no requests."""
    lines = read_log(log_path)
    if ids is not None:
        wanted = set(ids)
        lines = [r for r in lines if r.get("id") in wanted]
    gens = defaultdict(lambda: {"designs": set(), "kept_designs": set(),
                                "kept": 0, "tries": 0})
    for r in lines:
        g = gens[r.get("generation")]
        g["designs"].add(r["id"])
        g["tries"] += 1
        if r.get("kept"):
            g["kept"] += 1
            g["kept_designs"].add(r["id"])
    by_gen = {k: {"designs": len(v["designs"]),
                  "kept_designs": len(v["kept_designs"]),
                  "kept": v["kept"], "tries": v["tries"]}
              for k, v in gens.items()}
    proof = Counter((r.get("proof") or {}).get("verdict") for r in lines)
    fake = Counter((r.get("fake_check") or {}).get("verdict", "not run")
                   for r in lines)

    def nec(r):
        return (r.get("proof") or {}).get("verdict") == "NECESSARY"

    def real(r):
        return (r.get("fake_check") or {}).get("verdict") == "REAL"

    grid = {"necessary_real": sum(nec(r) and real(r) for r in lines),
            "necessary_not_real": sum(nec(r) and not real(r) for r in lines),
            "not_necessary_real": sum(real(r) and not nec(r) for r in lines),
            "neither": sum(not nec(r) and not real(r) for r in lines)}
    kept = sum(bool(r.get("kept")) for r in lines)
    served = Counter(r.get("served_model") for r in lines)
    tool = {"proof": sum(map(proof_tool_failure, lines)),
            "fake": sum(map(fake_tool_failure, lines)),
            "tries": sum(proof_tool_failure(r) or fake_tool_failure(r)
                         for r in lines)}
    rep = {"by_gen": by_gen, "proof": dict(proof), "fake": dict(fake),
           "grid": grid, "kept": kept, "tries": len(lines),
           "served": dict(served), "tool_failures": tool}

    print(f"{'gen':>4} {'designs':>8} {'with a kept try':>16} "
          f"{'kept tries':>11}")
    order = sorted(by_gen, key=lambda g: (g is None, g if g is not None
                                          else 0))
    for g in order:
        v = by_gen[g]
        print(f"{g!s:>4} {v['designs']:>8} {v['kept_designs']:>16} "
              f"{v['kept']:>5}/{v['tries']:<5}")
    designs = sum(v["designs"] for v in by_gen.values())
    with_kept = sum(v["kept_designs"] for v in by_gen.values())
    print(f"{'all':>4} {designs:>8} {with_kept:>16} "
          f"{kept:>5}/{len(lines):<5}")
    print("proof verdicts: " + (", ".join(
        f"{v} {c}" for v, c in proof.most_common()) or "none"))
    print("fake-state verdicts: " + (", ".join(
        f"{v} {c}" for v, c in fake.most_common()) or "none"))
    print(f"{'':22s}{'fake state REAL':>17}{'not REAL':>10}")
    print(f"{'proof NECESSARY':22s}{grid['necessary_real']:>17}"
          f"{grid['necessary_not_real']:>10}")
    print(f"{'proof not NECESSARY':22s}{grid['not_necessary_real']:>17}"
          f"{grid['neither']:>10}")
    print(f"right answer, wrong reasoning: {grid['necessary_not_real']}   "
          f"real fake state, wrong answer: {grid['not_necessary_real']}")
    # the grid counts these as failures, but the tools gave no verdict
    print(f"timeouts and crashes, not verdicts on the try: {tool['tries']} "
          f"tries (proof {tool['proof']}, fake state {tool['fake']}); "
          "counted as not kept above, and a rerun does not retry them")
    print(f"kept: {kept} of {len(lines)} tries (both checks passed and all "
          "three fields well formed)")
    print("served model: " + (", ".join(
        f"{m} ({c})" for m, c in served.most_common()) or "none"))
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(
        description="The untrained model writes and checks its own "
                    "fake-state reasoning.")
    p.add_argument("--per-gen", type=int, default=7,
                   help="designs per generation")
    p.add_argument("--gens", default=",".join(map(str, GENS)),
                   help="comma-separated generations")
    p.add_argument("--seed", type=int, default=0,
                   help="same seed, same designs")
    p.add_argument("--ids", help="comma-separated corpus ids instead")
    p.add_argument("--tries", type=int, default=8,
                   help="replies asked for per design, in one request")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=None,
                   help="sent only when given; else the server default")
    p.add_argument("--workers", type=int, default=4,
                   help="tries graded at once")
    p.add_argument("--allow-trained", action="store_true",
                   help="run even if the served model is a fine-tune")
    p.add_argument("--dry", action="store_true",
                   help="print the selection and the first prompt; no "
                        "requests, no endpoint needed")
    p.add_argument("--report", action="store_true",
                   help="counts from the log; no requests")
    args = p.parse_args(argv)

    if args.report:
        report(LOG)
        return
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()
            if l.strip()]
    gens = tuple(int(g) for g in args.gens.split(",") if g.strip())
    selected = select(rows, args.per_gen, gens, args.seed, args.ids)
    count, _ = logged_tries(read_log(LOG))

    if args.dry:
        for r in selected:
            print(f"  g{r.get('generation')} {r['id']:8s} "
                  f"{(r.get('ext_type') or 'g0'):10s} "
                  f"lines={len(r['verilog'].splitlines()):<4d} "
                  f"logged={count[r['id']]}/{args.tries}")
        if selected:
            print(f"\nprompt for {selected[0]['id']}:\n")
            print(build_prompt(selected[0]))
        todo = [r for r in selected if count[r["id"]] < args.tries]
        print(f"\n{len(todo)} designs would be asked "
              f"({len(selected) - len(todo)} already fully tried, skipped)")
        return

    if not os.environ.get("QWEN_BASE_URL"):
        print("QWEN_BASE_URL is not set; point it at the served model "
              "(through the tunnel). Nothing was asked.", file=sys.stderr)
        sys.exit(2)
    problem = preflight()
    if problem:
        print(problem, file=sys.stderr)
        sys.exit(2)
    client = make_client()
    try:
        served = served_root(client.models.list(), model_name())
    except Exception as e:
        print(f"the model endpoint is not answering ({type(e).__name__}: "
              f"{e}). Nothing was asked and nothing was logged. Check the "
              "server has printed 'Application startup complete' and that "
              "the tunnel points at the right node.", file=sys.stderr)
        sys.exit(2)
    problem = refusal(served, args.allow_trained)
    if problem:
        print(problem, file=sys.stderr)
        sys.exit(2)
    print(f"endpoint is serving: {served}")
    run(selected, client, served, args)
    print()
    report(LOG, [r["id"] for r in selected])


if __name__ == "__main__":
    main()
