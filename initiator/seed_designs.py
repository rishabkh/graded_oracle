"""Real open-source modules made into training designs (28 Oct 2026).

Rishab chose option 1: real designs only as starting points; our pipeline
writes every property and helper fact. For each seed in SEEDS (real modules
with their own formal sections cut out, open_source_seeds.py), the model
gets the module and the initiator's criterion (a property that is true, not
inductive alone, and made inductive by helper facts) and answers with edits
only, never the file:

  parameters      smaller parameter defaults, so the proof stays fast
  initial_values  `initial` statements giving registers their reset values
  property_code   the assertion (and a past-valid register) for the top

We put the edits into the real module ourselves and yosys checks that the
logic is unchanged: with every assertion and initial value taken out, the
module with the additions is equivalent to the module without them
(CHANGED_DESIGN otherwise). Then the initiator's own gates and checker
(run.judge), with REPAIRS repairs from the checker's notes. Proved designs
go to CORPUS as starting designs like the initiator's.

  hwtools
  export CLAUDE_PROVIDER=openai OPENAI_API_KEY=...
  venv/bin/python initiator/seed_designs.py --dry          # one question + cost, no call
  venv/bin/python initiator/seed_designs.py --limit 3      # 3 seeds (paid)
  venv/bin/python initiator/seed_designs.py                # all seeds (paid), resumable
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extender"))

import llm_client                                              # noqa: E402
from build_corpus import flatten_record                        # noqa: E402
from oracle.inject import _insert_point                        # noqa: E402
from prompts import SYSTEM_PROMPT                              # noqa: E402
from run import EFFORT, MAX_TOKENS, MODEL, checker_notes, judge, timed_out  # noqa: E402

SEEDS = HERE / "open_source_seeds_2026-10-28.jsonl"
LOG = HERE / "logs" / "seeded_open_source_2026-10-28.jsonl"
CORPUS = HERE.parent / "extender" / "corpus_open_source_seeded_2026-10-28.jsonl"
REPAIRS = 1

_S = {"type": "string"}
SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "parameters": {"type": "array", "items": {
            "type": "object", "properties": {"module": _S, "name": _S, "value": _S},
            "required": ["module", "name", "value"], "additionalProperties": False}},
        "initial_values": {"type": "array", "items": {
            "type": "object", "properties": {"module": _S, "code": _S},
            "required": ["module", "code"], "additionalProperties": False}},
        "property_code": _S,
        "clock": _S,
        "invariants": {"type": "array", "items": _S},
        "antecedents": {"type": "array", "items": _S},
        "sanity_covers": {"type": "array", "items": _S},
        "cti_state": {"type": "array", "items": {
            "type": "object", "properties": {"signal": _S, "value": _S},
            "required": ["signal", "value"], "additionalProperties": False}},
        "cti_explanation": _S,
    },
    "required": ["parameters", "initial_values", "property_code", "clock", "invariants",
                 "antecedents", "sanity_covers", "cti_state", "cti_explanation"],
    "additionalProperties": False,
}

# 28 Oct 2026, the 5-seed trial (4 proved, $0.25): axis_adapter's property
# named `upsize.s_axis_tvalid_reg`, a register inside a generate block, twice;
# yosys cannot see it from the top module, so the last rule below was added.
SEED_SYSTEM = SYSTEM_PROMPT[:SYSTEM_PROMPT.index("## Output format")] + """\
## Output format: a real module

Here you do not write the design. It is a real open-source module, given in
the question, and it stays as it is. Your answer lists only your edits and the
triple's other fields. The checker puts the edits into the module and confirms,
by an equivalence check, that the module still does exactly what it did; an
answer whose edits change the module's logic is rejected.

- `parameters`: new defaults, as {module, name, value}: only `parameter`s
  (never a `localparam`), only to make the proof small, keeping every relation
  the module needs between its parameters.
- `initial_values`: for a register with no initial value, Verilog `initial`
  statements giving it its reset value, as {module, code}, put at the end of
  that module. [] when every register already has one.
- `property_code`: the property, as Verilog put at the end of the top module:
  the assertion, and a past-valid register if it uses `$past`.
- `clock`, `invariants` (R, never inside the Verilog), `antecedents`,
  `sanity_covers`, `cti_state`, `cti_explanation`: as the rules above say.
- The property, `invariants`, `antecedents` and `sanity_covers` name only
  signals the top module declares itself, outside any generate block: never a
  dotted name (`block.signal`, `instance.signal`). When the register you want
  sits inside a generate block or a sub-module, choose a property about
  signals the top module declares.
"""

SEED_TEMPLATE = """\
## A real module

From the open-source repository {repo} ({licence}), file(s): {files}.
Its own formal checks were removed. Top module: `{top}`.

```verilog
{verilog}
```

## Your task

Make this module a planted triple without changing what it does. Choose a
property P about what the module really does (its handshake, its storage,
its counters, its state machine): true, not inductive on its own, and made
inductive by helper facts R, as the system prompt explains. Its idle
condition is its own: with its valid or enable inputs held low, nothing moves.
"""

REPAIR_TEMPLATE = """\
Your answer for this real module was checked and rejected.

## The module (`{top}`)

```verilog
{verilog}
```

## Your answer

{answer}

## What the checker found

{notes}

## Your task

Return the complete corrected answer, with every field of the schema. The
module itself stays as it is: change your edits, the property, the helper
facts or the other fields.
"""


class EditError(Exception):
    pass


def _module_span(text, module):
    m = re.search(r"\bmodule\s+" + re.escape(module) + r"\b", text)
    if not m:
        raise EditError(f"no module {module!r} in the file")
    e = re.search(r"\bendmodule\b", text[m.end():])
    if not e:
        raise EditError(f"no endmodule after module {module!r}")
    return m.start(), m.end() + e.end()


def set_param(text, module, name, value):
    """`text` with the default of `parameter name` in `module` set to
    `value`. Only a `parameter`: a name declared by `localparam` is refused."""
    start, end = _module_span(text, module)
    body = text[start:end]
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*=(?!=)", body):
        decl = body[:m.start()]
        kw = max(decl.rfind("parameter"), decl.rfind("localparam"))
        if kw < 0 or ";" in decl[kw:] or decl[kw:].startswith("localparam"):
            continue
        # the old default runs to the next ',' ')' or ';' outside brackets
        i, depth = m.end(), 0
        while i < len(body):
            c = body[i]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c in ",;" and depth == 0:
                break
            i += 1
        new = body[:m.end()] + f" {value.strip()}" + body[i:]
        return text[:start] + new + text[end:]
    raise EditError(f"no parameter {name!r} in module {module!r}")


def _insert_end(text, module, code):
    at = _insert_point(text, module)
    return text[:at] + code.rstrip() + "\n" + text[at:]


def apply_edits(seed, answer):
    """(module with the parameter edits only, module with every edit)."""
    base = seed["verilog"]
    for p in answer.get("parameters") or []:
        base = set_param(base, p["module"], p["name"], p["value"])
    full = base
    for iv in answer.get("initial_values") or []:
        if iv.get("code", "").strip():
            full = _insert_end(full, iv["module"], iv["code"])
    if not answer.get("property_code", "").strip():
        raise EditError("property_code is empty")
    full = _insert_end(full, seed["top_module"], answer["property_code"])
    return base, full


def _flat(text, top, name, workdir):
    """The module as yosys builds it, with every assertion and initial value
    taken out, flattened and renamed `name`."""
    Path(workdir, f"{name}.v").write_text(text)
    # memory and async2sync (28 Oct 2026): yosys's equivalence check has no
    # model for a memory or a flip-flop with an asynchronous reset, and its
    # error was reported as a changed design (32 answers in three runs)
    script = (f"read -formal {name}.v; prep -top {top}; chformal -remove; "
              "delete t:$meminit*; memory; setattr -unset init; async2sync; flatten; "
              f"hierarchy -top {top}; opt_clean; rename {top} {name}; "
              f"write_rtlil {name}.il")
    p = subprocess.run(["yosys", "-q", "-p", script], cwd=workdir,
                       capture_output=True, text=True, timeout=300)
    if p.returncode:
        raise EditError(f"yosys could not read the edited module: {(p.stderr or p.stdout)[-400:]}")


def same_logic(base, full, top):
    """True when the additions change nothing but assertions and initial
    values; otherwise (False, what yosys said)."""
    with tempfile.TemporaryDirectory() as d:
        _flat(base, top, "gold", d)
        _flat(full, top, "gate", d)
        p = subprocess.run(
            ["yosys", "-q", "-p", "read_rtlil gold.il; read_rtlil gate.il; "
             "equiv_make gold gate eq; hierarchy -top eq; equiv_simple -seq 2; "
             "equiv_induct -seq 2; equiv_status -assert"],
            cwd=d, capture_output=True, text=True, timeout=300)
        return (p.returncode == 0, (p.stderr or p.stdout)[-400:])


def check(seed, answer, proof_root, grade=True):
    """One answer through the edits, the logic check and the checker; the
    same dict judge() returns (verdict, reason, result, grade_wall_s)."""
    out = {"verdict": None, "reason": None, "result": None, "grade_wall_s": None}
    try:
        base, full = apply_edits(seed, answer)
        same, said = same_logic(base, full, seed["top_module"])
    except EditError as exc:
        return dict(out, verdict="BAD_EDIT", reason=str(exc))
    if not same:
        return dict(out, verdict="CHANGED_DESIGN",
                    reason="the edits change the module's logic, not only its assertions "
                           "and initial values; yosys: " + said)
    triple = {k: answer[k] for k in ("clock", "invariants", "antecedents", "sanity_covers",
                                     "cti_state", "cti_explanation")}
    triple.update(verilog=full, top_module=seed["top_module"])
    j = judge(json.dumps(triple), "ok", "globally", grade, proof_root=proof_root, spin=False)
    j["triple"] = triple
    return j


def question(seed):
    return SEED_TEMPLATE.format(repo=seed["repo"], licence=seed["licence"],
                                files=", ".join(seed["files"]), top=seed["top_module"],
                                verilog=seed["verilog"].strip())


def _call(user, effort):
    raw, usage, stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SEED_SYSTEM, user=user,
        schema=SEED_SCHEMA, effort=effort)
    return (json.loads(raw) if raw else None), usage, stop


def run_seed(seed, proof_dir, repairs=REPAIRS, effort=EFFORT, call=None, grade=True):
    """One seed: question, check, up to `repairs` repairs. One log record."""
    call = call or _call
    record = {"seed": seed["name"], "repo": seed["repo"], "licence": seed["licence"],
              "commit": seed["commit"], "files": seed["files"], "top_module": seed["top_module"],
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "model": llm_client.model_label(MODEL), "effort": effort,
              "usage": {"input": 0, "output": 0}, "history": []}
    user = question(seed)
    for attempt in range(repairs + 1):
        answer, usage, stop = call(user, effort)
        for k in ("input", "output"):
            record["usage"][k] += (usage or {}).get(k, 0)
        if answer is None:
            j = {"verdict": {"refusal": "REFUSED", "length": "TRUNCATED"}.get(stop, "UNPARSEABLE"),
                 "reason": stop, "result": None}
        else:
            j = check(seed, answer, Path(proof_dir) / f"r{attempt}", grade)
        record["history"].append({"answer": answer, "verdict": j["verdict"],
                                  "reason": j["reason"]})
        record.update(verdict=j["verdict"], reason=j["reason"], repair_attempts=attempt)
        if "triple" in j:
            record["raw_json"] = json.dumps(j["triple"])
        if j["verdict"] == "NECESSARY" or answer is None or attempt == repairs \
                or (j.get("result") and timed_out(j)):
            break
        notes = (checker_notes(j["verdict"], j["reason"], j["result"])
                 if j.get("result") else f"{j['verdict']}: {j['reason']}")
        user = REPAIR_TEMPLATE.format(top=seed["top_module"], verilog=seed["verilog"].strip(),
                                      answer=json.dumps(answer, indent=2), notes=notes)
    return record


# the first list's 4 modules that are copies or run on two clocks (28 Oct
# 2026); 2 of them proved. Only these are left out by name: the later rule
# that skips modules with no stored state reads the default settings, and
# axis_adapter, addrdecode and axi3reorder gained state from the settings
# the model chose, and proved
DROPPED = {"verilog-axi/arbiter", "verilog-axi/priority_encoder",
           "verilog-axi/axil_dp_ram", "wb2axip/afifo"}


def write_corpus(log_path=LOG, corpus_path=CORPUS, dropped=DROPPED):
    """Proved seeds as corpus rows, ids os0_NNN in the order they proved,
    minus the DROPPED modules and near-copies."""
    rows = [json.loads(line) for line in open(log_path)] if Path(log_path).exists() else []
    good = [r for r in rows if r.get("verdict") == "NECESSARY" and r.get("raw_json")
            and r["seed"] not in dropped]
    # near-copies (28 Oct 2026): Ultra-Embedded's two RISC-V repos share their
    # icache and tcm_mem_pmem (96% and 99% alike), and verilog-pcie writes one
    # MSI block for two vendors (91%); the first one proved is kept
    kept, toks = [], []
    for r in good:
        t = re.findall(r"[A-Za-z_]\w*|\d+|\S", re.sub(
            r"//[^\n]*|/\*.*?\*/", "", json.loads(r["raw_json"])["verilog"], flags=re.S))
        if not any(abs(len(t) - len(u)) <= 0.2 * max(len(t), len(u))
                   and difflib.SequenceMatcher(None, t, u, autojunk=False).quick_ratio() > 0.9
                   and difflib.SequenceMatcher(None, t, u, autojunk=False).ratio() > 0.9
                   for u in toks):
            kept.append(r)
            toks.append(t)
    good = kept
    with open(corpus_path, "w") as fh:
        for i, r in enumerate(good):
            row = flatten_record(r, i, prefix="os0")
            row.update(open_source={k: r[k] for k in ("seed", "repo", "licence", "commit", "files")})
            fh.write(json.dumps(row) + "\n")
    return len(good)


def recheck(log_path, seeds, workers=4, verdicts=("CHANGED_DESIGN",)):
    """Check again, with no model call, the last saved answer of every seed
    the logic check turned away with a yosys error (28 Oct 2026: memories
    and asynchronous resets, fixed in _flat). The log is rewritten with the
    new verdicts; the old one is kept as verdict_before_recheck."""
    rows = [json.loads(line) for line in open(log_path)]
    by_name = {s["name"]: s for s in seeds}
    todo = [i for i, r in enumerate(rows) if r.get("verdict") in verdicts
            and r.get("seed") in by_name and (r.get("history") or [{}])[-1].get("answer")]
    proofs = Path(log_path).with_name(Path(log_path).stem + "_proofs")

    def one(i):
        r = rows[i]
        return i, check(by_name[r["seed"]], r["history"][-1]["answer"],
                        proofs / (r["seed"].replace("/", "_") + "_recheck"))
    with ThreadPoolExecutor(workers) as pool:
        for i, j in pool.map(one, todo):
            r = rows[i]
            r.update(verdict_before_recheck=r["verdict"], verdict=j["verdict"],
                     reason=j["reason"], rechecked="2026-10-28")
            if "triple" in j:
                r["raw_json"] = json.dumps(j["triple"])
            print(f"{r['seed']}: {r['verdict_before_recheck']} -> {r['verdict']}", flush=True)
    with open(log_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recheck", action="store_true",
                    help="check the saved answers turned away as CHANGED_DESIGN again; no model call")
    ap.add_argument("--seeds", default=str(SEEDS))
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--names", help="comma-separated seed names (default: all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repairs", type=int, default=REPAIRS)
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--dry", action="store_true", help="print one question and the cost; call nothing")
    args = ap.parse_args(argv)

    seeds = [json.loads(line) for line in open(args.seeds)]
    if args.recheck:
        n = recheck(args.log, seeds, args.workers)
        print(f"rechecked {n}; {write_corpus(args.log)} proved designs -> "
              f"{CORPUS.relative_to(HERE.parent)}")
        return
    if args.names:
        seeds = [s for s in seeds if s["name"] in set(args.names.split(","))]
    log = Path(args.log)
    done = {json.loads(line)["seed"] for line in open(log)} if log.exists() else set()
    todo = [s for s in seeds if s["name"] not in done][:args.limit or None]
    if args.dry:
        q = question(todo[0])
        tin = (len(SEED_SYSTEM) + sum(len(question(s)) for s in todo)) // 4
        tout = 12000 * len(todo)          # edits, facts and hidden thinking: a guess
        calls = 1 + 0.6 * args.repairs     # about 6 in 10 need a repair (a guess)
        print(SEED_SYSTEM[-1200:], "\n----\n", q[:1500], "\n...")
        print(f"\n{len(todo)} seeds; about ${calls * llm_client.dollars(llm_client.model_label(MODEL), tin, tout):.2f} "
              f"with up to {args.repairs} repair(s) each")
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    proofs = log.with_name(log.stem + "_proofs")
    lock = threading.Lock()
    print(f"{len(todo)} seeds with {llm_client.model_label(MODEL)}, effort {args.effort}, "
          f"{args.repairs} repair(s) each")
    with llm_client.call_log(log.with_name("calls_" + log.name)), \
            ThreadPoolExecutor(args.workers) as pool, open(log, "a") as fh:
        futs = {pool.submit(run_seed, s, proofs / s["name"].replace("/", "_"),
                            args.repairs, args.effort): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:            # one seed never stops the run
                rec = {"seed": futs[fut]["name"], "verdict": "ERROR", "reason": repr(exc)}
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            print(f"[{i}/{len(todo)}] {rec['seed']}: {rec['verdict']} "
                  f"(repairs {rec.get('repair_attempts', 0)})", flush=True)
    n = write_corpus(log)
    print(f"{n} proved designs -> {CORPUS.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
