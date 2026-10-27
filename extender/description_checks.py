"""Does the design do what its description says? A model that never sees
the code writes the checks; our checker runs them.

27 Oct 2026, Rishab: "Only pay a model for one question: does the design do
what its description says? Have a model that hasn't seen the design read
the description and write a few 'this should be possible' and 'this should
never happen' checks. If the design fails one, look at it. Don't auto-mark
it wrong, because these checks are often mistaken."

Two steps, so the paid one runs once and the free one can run again:

  ask (paid, one call per design): the model reads the description (the
      initiator's kind, required feature and plan, plus what each extender
      step said it added) and the top module's ports, never the code, and
      writes 3 to 6 checks over the ports:
        possible  a working design can do this: the checker must reach it
                  (cover) within DEPTH steps
        never     the description rules this out: the checker must find no
                  run of DEPTH steps where it happens
  run (free): the design's own assertions are taken out, each check is put
      in and run with sby. A check that will not compile is "bad" and says
      nothing about the design.

A failed check only puts the design on a list to look at (flagged); it is
never a label by itself.

  hwtools
  export CLAUDE_PROVIDER=openai OPENAI_API_KEY=...
  venv/bin/python extender/description_checks.py ask --dry          # prompt + cost, no call
  venv/bin/python extender/description_checks.py ask --limit 5      # 5 paid calls
  venv/bin/python extender/description_checks.py run                # free
"""
import argparse
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import llm_client                                              # noqa: E402
from mutation_check import DESIGNS, top_ports                  # noqa: E402
from oracle.inject import _inject_block, strip_assertions      # noqa: E402
from oracle.parse import parse_assert_failures, parse_cover_log  # noqa: E402
from oracle.sby import run_sby                                 # noqa: E402

LOG_DIR = HERE / "logs" / "description_checks_2026-10-27"
CHECKS = LOG_DIR / "checks.jsonl"      # ask writes, run reads
RESULTS = LOG_DIR / "results.jsonl"    # run writes
MODEL = "claude-opus-5-5"              # CLAUDE_PROVIDER=openai calls OPENAI_MODEL instead
EFFORT = "medium"
MAX_TOKENS = 16000                     # counts GPT-6.1 Sol's hidden thinking too
TRIES = 2
DEPTH = 25
TIMEOUT_S = 120
HEADER = "// DESCRIPTION CHECKS"

SYSTEM = """\
You write checks for a hardware design you cannot see. You get its \
description and the names of its ports, and write checks that a design \
matching the description must pass."""

SCHEMA = {
    "type": "object",
    "properties": {"checks": {"type": "array", "items": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["possible", "never"]},
                       "expr": {"type": "string"},
                       "why": {"type": "string"}},
        "required": ["kind", "expr", "why"],
        "additionalProperties": False}}},
    "required": ["checks"],
    "additionalProperties": False,
}


def ports_text(ports):
    return "\n".join(f"  {d:6s} {f'[{w - 1}:0] ' if w > 1 else ''}{n}"
                     for n, d, w in ports)


def build(design, ports):
    """The question: the description and the ports, never the code."""
    return f"""A hardware block was built from the description below. You will not see its code. \
From the description and its ports alone, write checks our formal checker will run on it.

DESCRIPTION
{design["description"]}

PORTS of the top module {design["top_module"]} (clock: {design.get("clock") or "clk"})
{ports_text(ports)}

Write 3 to 6 checks. Each is one Verilog boolean expression over these port names only:
- "possible": something a design matching the description can do, so the checker must find a run \
where it is true (for example a result coming out, a request being accepted, a full or error output \
turning on).
- "never": something the description rules out, so the checker must find no run where it is true \
(for example two grants at once, a result coming out with no request before it, an item taken while \
the block says it is full).

Rules:
- Use only the port names above, Verilog operators, sized constants, and $past(port) for a port's \
value one cycle earlier. No internal names, no SVA (no |-> or ##), no new wires.
- The checker starts from the design's power-on values and then gives every input any value in \
every cycle. A "never" check must hold in every cycle of normal operation, so guard it with the reset \
input if there is one (for example !rst && ... or rst_n && ...) and with any condition the \
description puts on the inputs.
- Write only checks the description clearly supports; skip what it leaves open (exact timing, \
behaviour on bad inputs, details it does not mention).
- Give each check a short "why" that quotes the part of the description it comes from."""


def _call(prompt):
    text, usage, stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, user=prompt,
        schema=SCHEMA, effort=EFFORT)
    if text is None:
        raise RuntimeError(f"no answer ({stop})")
    return json.loads(text), usage


def ask_one(design, ports, call=None):
    """One design's checks: {"key", "checks", "usage"} or {"key", "error"}."""
    prompt = build(design, ports)
    err = None
    for _ in range(TRIES):
        try:
            answer, usage = (call or _call)(prompt)
        except Exception as exc:        # network, refusal, cut off, parse
            err = f"{type(exc).__name__}: {exc}"
            continue
        checks = [c for c in (answer or {}).get("checks", [])
                  if isinstance(c, dict) and c.get("kind") in ("possible", "never")
                  and str(c.get("expr", "")).strip()]
        if checks:
            return {"key": design["key"], "checks": checks, "usage": usage}
        err = "the answer has no usable check"
    return {"key": design["key"], "error": err}


_CONST = re.compile(r"\d*\s*'[sS]?[bBoOdDhH]\s*[0-9a-fA-FxXzZ_?]+")
_NAME = re.compile(r"\$?[A-Za-z_][A-Za-z0-9_$]*")


def unknown_names(expr, ports):
    """Names in `expr` that are not ports. Verilog quietly makes an unknown
    name a new wire, and the checker lets an undriven wire take any value,
    so a misspelled port would pass every check (found in the first test,
    27 Oct 2026)."""
    names = {n for n, _, _ in ports}
    return sorted({m for m in _NAME.findall(_CONST.sub(" ", expr))
                   if not m.startswith("$") and m not in names})


def with_checks(source, top, clock, kind, exprs):
    """The design with `exprs` put in as covers (possible) or as assertions
    that they never happen (never), checked on the clock edge; the line of
    each check is in the returned Injection's line_map."""
    stmt = "cover" if kind == "possible" else "assert"
    tagged = [(kind, e if kind == "possible" else f"!({e})") for e in exprs]
    return _inject_block(source, top, HEADER, f"always @(posedge {clock}) begin",
                         stmt, tagged)


def run_one(design, checks, workroot, depth=DEPTH, timeout_s=TIMEOUT_S, sby=run_sby,
            ports=None):
    """Run one design's checks; each check gets a result:
      possible: reached | unreached        never: held | happened
      bad (names something that is not a port, or did not compile) |
      timeout | error"""
    top, clock = design["top_module"], design.get("clock") or "clk"
    if ports is None:
        ports = top_ports(design["verilog"], top, Path(tempfile.mkdtemp(dir=workroot)))
    base = strip_assertions(design["verilog"])
    out = [dict(c) for c in checks]
    work = Path(tempfile.mkdtemp(prefix=design["key"] + "_", dir=workroot))

    def one(name, kind, exprs, mode):
        inj = with_checks(base, top, clock, kind, exprs)
        sv = work / f"{name}.sv"
        sv.write_text(inj.text)
        return inj, sby(name, sv, top, mode, depth, timeout_s, work)

    # each check alone first: one that does not compile is "bad", and a
    # never check stops at its own failure, so each runs on its own anyway
    for i, c in enumerate(out):
        bad = unknown_names(c["expr"], ports)
        if bad:
            c.update(result="bad", not_ports=bad)
        elif c["kind"] == "never":
            _, res = one(f"never{i}", "never", [c["expr"]], "bmc")
            c["result"] = {0: "held", 2: "happened", 8: "timeout", None: "timeout"}.get(
                res.rc, "bad" if "ERROR" in res.log_text or res.rc == 1 else "error")
        else:
            inj, res = one(f"possible{i}", "possible", [c["expr"]], "cover")
            reached, unreached = parse_cover_log(res.log_text)
            lines = set(inj.line_map)
            c["result"] = ("reached" if lines & reached else
                           "unreached" if lines & unreached else
                           "timeout" if res.rc in (8, None) else "bad")
    flags = [c for c in out if c.get("result") in ("unreached", "happened")]
    return {"key": design["key"], "checks": out, "flagged": bool(flags),
            "flags": [f"{c['kind']} check {c['result']}: {c['expr']}" for c in flags]}


def _rows(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def cmd_ask(args):
    designs = [d for d in _rows(args.designs) if d.get("description")]
    if args.keys:
        designs = [d for d in designs if d["key"] in set(args.keys.split(","))]
    done = {r["key"] for r in _rows(args.checks) if "checks" in r}
    todo = [d for d in designs if d["key"] not in done][:args.limit or None]
    work = Path(tempfile.mkdtemp(prefix="ports_"))
    if args.dry:
        d = todo[0]
        prompt = build(d, top_ports(d["verilog"], d["top_module"], work))
        est_in = len(SYSTEM + prompt) // 4 * len(todo)
        est_out = 3000 * len(todo)      # checks plus hidden thinking, a guess
        print(prompt)
        print(f"\n{len(todo)} designs to ask; about {est_in:,} tokens in and "
              f"{est_out:,} out: about ${llm_client.dollars(llm_client.model_label(MODEL), est_in, est_out):.2f}")
        return
    Path(args.checks).parent.mkdir(parents=True, exist_ok=True)
    print(f"asking {len(todo)} designs with {llm_client.model_label(MODEL)}")
    with llm_client.call_log(Path(args.checks).parent / "calls.jsonl"), \
            ThreadPoolExecutor(args.workers) as pool, open(args.checks, "a") as fh:
        def job(d):
            try:
                ports = top_ports(d["verilog"], d["top_module"],
                                  Path(tempfile.mkdtemp(dir=work)))
            except Exception as exc:
                return {"key": d["key"], "error": f"ports: {exc}"}
            return ask_one(d, ports)
        for i, fut in enumerate(as_completed([pool.submit(job, d) for d in todo]), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {row['key']} "
                  f"{len(row.get('checks', []))} checks {row.get('error', '')}", flush=True)


def cmd_run(args):
    designs = {d["key"]: d for d in _rows(args.designs)}
    asked = [r for r in _rows(args.checks) if r.get("checks")]
    if args.keys:
        asked = [r for r in asked if r["key"] in set(args.keys.split(","))]
    done = {r["key"] for r in _rows(args.results)}
    todo = [r for r in asked if r["key"] not in done][:args.limit or None]
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    workroot = Path(tempfile.mkdtemp(prefix="desc_checks_", dir=Path(args.results).parent))
    print(f"running the checks of {len(todo)} designs, {args.workers} at a time")
    with ThreadPoolExecutor(args.workers) as pool, open(args.results, "a") as fh:
        futs = [pool.submit(run_one, designs[r["key"]], r["checks"], workroot)
                for r in todo if r["key"] in designs]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{i}/{len(futs)}] {row['key']} "
                  f"{'FLAGGED ' + '; '.join(row['flags']) if row['flagged'] else 'ok'}",
                  flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "run"):
        p = sub.add_parser(name)
        p.add_argument("--designs", default=str(DESIGNS))
        p.add_argument("--checks", default=str(CHECKS))
        p.add_argument("--results", default=str(RESULTS))
        p.add_argument("--keys", help="comma-separated design keys (default: all)")
        p.add_argument("--limit", type=int)
        p.add_argument("--workers", type=int, default=5 if name == "ask" else 4)
        if name == "ask":
            p.add_argument("--dry", action="store_true", help="print one prompt and the cost, call nothing")
    args = ap.parse_args(argv)
    {"ask": cmd_ask, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
