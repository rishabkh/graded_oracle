"""Extender step 4: structural and second-property extensions — the two
types where the invariant list is allowed to change.

New machinery beyond the distractor:
  dispositions   every parent invariant clause must be declared kept or
                 superseded; a silent deletion rejects the extension
                 BEFORE grading.
  P2-unaided     a second property that proves by k-induction without any
                 invariants teaches nothing; checked by stripping the
                 first property and grading the second alone.
  coupling       a structural extension must produce at least one clause
                 mentioning both new and existing state; its absence is
                 reported (the human reads it), not auto-rejected.

Port list is FIXED for these types (unlike the distractor): new state
must be observable through existing outputs.

  venv/bin/python extender/extend.py --plan                      # the 5-seed plan
  venv/bin/python extender/extend.py --parent g0_014 --type structural --move GUARD --dry
  venv/bin/python extender/extend.py --parent g0_014 --type structural --move GUARD
  venv/bin/python extender/extend.py --parent g0_014 --type second
"""
import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import (_mask_comments, extract_asserts,   # noqa: E402
                          state_bits)
from extend_one import (DIFF_FORMAT, GRADE_KWARGS, MODEL, EFFORT,   # noqa: E402
                        Spinner, dump, load_parent)
from patch import PatchError, apply_patch                 # noqa: E402

from oracle import PropertyInfo, grade, grade_triple_generated  # noqa: E402

MAX_TOKENS = 20000

SYSTEM_PROMPT = """\
You are extending a formally verified SystemVerilog module.

The module has a property proven by k-induction with the strengthening
invariants listed in the user message. Your job is to grow the design so
that it remains provable, but is harder.

The result is training data for teaching a model to discover invariants.
So the logic you add should read as though it exists for its own functional
reason. A reader should not be able to guess the invariant by noticing which
part of the module looks like scaffolding. Write the extension the way a
hardware engineer would write the next feature, not the way a proof author
would write a hint.

HARD RULES

1. The existing property text survives verbatim. Do not modify, move, or
   delete the existing assertion. Do not add assume or cover statements.
2. Emit a PATCH in the diff format given below. Never a rewritten module.
3. The port list is fixed. Do not add, remove, or rename ports. New state
   must be observable through an existing output, directly or through
   existing logic.
4. Every new register gets a reset in the existing if (rst) branch and an
   initial value, matching the style already in the module.
5. Same Verilog subset as the module: synthesizable always @(posedge clk)
   or always @(*) logic, immediate assertions only, no concurrent SVA, no
   system tasks.
6. Do not use the -> implication operator. Yosys rejects it. Write !a || b.
7. Keep datapath widths narrow, four to eight bits. Wide arithmetic over
   free inputs makes the proof intractable and teaches nothing.
8. If a new register's declared width admits values outside its reachable
   range, add an invariant clause bounding it."""

MOVES = {
    "SPLIT": ("Take a quantity that is currently held in one register and "
              "divide it, so part of it lives in a new register some of "
              "the time."),
    "STAGE": ("Insert a pipeline register between a producer and a "
              "consumer. An update that currently completes in one cycle "
              "now takes two, with the in-flight value held in new state."),
    "GUARD": ("Add an enable register that gates an existing update. The "
              "existing transition fires only when the new enable is set."),
    "PEER": ("Add a second instance of similar logic that shares a "
             "resource with the existing logic."),
    "BOUND": ("Replace a constant limit with a register that can change, "
              "so the existing comparison is against state rather than a "
              "literal."),
}

INVARIANT_RULES = """\
- Emit the COMPLETE new invariant list, not just what changed.
  Inductiveness is a property of the whole conjunction, so satisfy
  yourself that the full list closes the proof.
- The list must be sufficient: the property must be provable by
  k-induction from the design plus these invariants.
- Keep clauses as strong as the parent's. Generalising a clause to cover
  new state is correct. Weakening it until it happens to still hold is not.
- Add no clause you cannot justify. A clause that could be deleted without
  breaking the proof is noise in the corpus."""

DISPOSITION_BLOCK = """\
For every invariant clause in the parent list, record what happened to it:

  kept        the clause appears verbatim in your new list
  superseded  the clause is no longer true, or is subsumed by a stronger
              clause. Name the replacing clause and give a one-line reason.
              If several new clauses jointly replace it, list them all in
              replaced_by — do not paste them into one string.

A parent clause that is neither kept nor declared superseded is a silent
deletion, and the extension is rejected before it is graded. Deleting a
clause is allowed. Deleting it quietly is not.

Example: the parent has `fault == (timer == 4'd0)`. You add a register that
latches an external error signal, so the fault output now has a second
source and the parent clause is false whenever that latch is set. The
correct disposition is superseded by `fault == (timer == 4'd0 || err_latch)`,
reason "the fault condition now has a second source"."""

STRUCTURAL_TEMPLATE = """\
Extension type: STRUCTURAL.

Add new state that INTERACTS with the state the property depends on, so the
parent's invariants are no longer sufficient and the invariant list must
grow to describe the coupling.

The move you must make:
{move}

The module:

{verilog}

The property, which you must not touch:
{property}

The parent's strengthening invariants:
{invariants}

REQUIREMENTS SPECIFIC TO THIS TYPE

The new state must write, or gate the write of, something the property or
the parent invariants read. If your new logic only reads existing state and
never affects it, that is a distractor, not a structural extension.

Your new invariant list must contain at least one clause mentioning both a
new identifier and an existing one. That coupling clause is the point of
the extension. A clause about only new state is a fact bolted on the side.

WORK IT OUT BEFORE YOU WRITE THE PATCH

Answer these in order. The patch comes last, after you know what breaks.

1. new_state: which register or registers you are adding, and which existing
   signal each one writes or gates.

2. coupling: in one sentence, the relation that must hold between the new
   state and the existing state for the design to be correct.

3. induction_gap: a concrete state of the extended design that an induction
   engine will treat as a legal starting point, that the design can never
   actually reach, and from which the property fails within a cycle. The
   design must be able to sit in this state indefinitely with its enables
   held low, so that a counterexample starting there survives the full
   twenty-step induction window.

4. why_parent_insufficient: which parent invariant clauses fail to exclude
   that state, and why.

5. Then write the patch, then the full new invariant list.

{invariant_rules}

{disposition_block}

Diff format:
{diff_format}

Reply with JSON:
{{"new_state": "...", "coupling": "...", "induction_gap": "...",
  "why_parent_insufficient": "...", "patch": "...",
  "invariants": ["..."], "dispositions": [...]}}"""

SECOND_TEMPLATE = """\
Extension type: SECOND PROPERTY.

Add a SECOND assertion about this module, proven from the SAME invariant
list as the first.

The module:

{verilog}

The existing property, which you must not touch:
{property}

The parent's strengthening invariants:
{invariants}

REQUIREMENTS SPECIFIC TO THIS TYPE

The second property must be true of the design and must NOT be provable by
k-induction on its own without any invariants. If it proves unaided it
teaches nothing and the extension is rejected.

The second property must depend on state the first property already
depends on, so that at least one invariant clause is required by both. Two
properties needing disjoint clauses is two problems in one file, not an
extension.

You may add new state if the second property needs it, but you do not have
to. If you do, the rules above about resets, initial values, and observing
through existing outputs still apply.

Add the assertion as its own block, in the same style as the existing one:

    always @(posedge clk)
        <optional guard> assert (<expression>);

WORK IT OUT BEFORE YOU WRITE THE PATCH

1. claim: what the second property asserts, in one sentence.

2. shared_state: which registers both properties depend on.

3. not_inductive_alone: a concrete state that satisfies the second property,
   violates the invariant list, and steps to a violation of the second
   property. The design must be able to hold this state indefinitely with
   enables low.

4. shared_clause: which invariant clause or clauses are needed by both
   properties, and why each is needed by each.

5. Then write the patch and the full invariant list.

{invariant_rules}

{disposition_block}

Diff format:
{diff_format}

Reply with JSON:
{{"claim": "...", "shared_state": "...", "not_inductive_alone": "...",
  "shared_clause": "...", "patch": "...", "invariants": ["..."],
  "dispositions": [...]}}"""

_DISPOSITION_ITEMS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "clause": {"type": "string"},
            "status": {"type": "string", "enum": ["kept", "superseded"]},
            "replaced_by": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["clause", "status", "replaced_by", "reason"],
        "additionalProperties": False,
    },
}


def _schema(reasoning_fields):
    props = {f: {"type": "string"} for f in reasoning_fields}
    props["patch"] = {"type": "string"}
    props["invariants"] = {"type": "array", "items": {"type": "string"}}
    props["dispositions"] = _DISPOSITION_ITEMS
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


STRUCTURAL_SCHEMA = _schema(
    ["new_state", "coupling", "induction_gap", "why_parent_insufficient"])
SECOND_SCHEMA = _schema(
    ["claim", "shared_state", "not_inductive_alone", "shared_clause"])

# One structural move per seed, five diverse seeds. Distractor and second
# property run on the same five, so the by-hand reading covers 15 results.
PLAN = [
    ("g0_014", "token_bucket", "GUARD"),
    ("g0_035", "pwm_dimmer", "STAGE"),
    ("g0_036", "watchdog", "BOUND"),
    ("g0_043", "stack_hwm", "SPLIT"),
    ("g0_046", "rr_arbiter_pri", "PEER"),
]


# --- pure checks (unit-tested) ---

def _norm(s):
    return " ".join(s.split())


def check_dispositions(parent_invs, new_invs, dispositions):
    """Every parent clause must be declared kept (and present verbatim) or
    superseded (absent, with a named present replacement and a reason).
    Returns an error string, or None if the audit passes."""
    new_normed = {_norm(x) for x in new_invs}
    parent_normed = {_norm(x): x for x in parent_invs}
    seen = set()
    for disp in dispositions:
        clause = _norm(disp.get("clause", ""))
        if clause not in parent_normed:
            return f"disposition names a non-parent clause: {disp.get('clause')!r}"
        if clause in seen:
            return f"duplicate disposition for clause: {disp.get('clause')!r}"
        seen.add(clause)
        status = disp.get("status")
        if status == "kept":
            if clause not in new_normed:
                return (f"clause declared kept but absent from the new "
                        f"list: {disp.get('clause')!r}")
        elif status == "superseded":
            if clause in new_normed:
                return (f"clause declared superseded but still present: "
                        f"{disp.get('clause')!r}")
            replacements = disp.get("replaced_by") or []
            if isinstance(replacements, str):   # tolerate the singular form
                replacements = [replacements]
            if not replacements:
                return (f"superseded with no replacing clause named: "
                        f"{disp.get('clause')!r}")
            missing = [r for r in replacements if _norm(r) not in new_normed]
            if missing:
                return (f"superseding clause not in the new list for: "
                        f"{disp.get('clause')!r} (missing: {missing})")
            if not disp.get("reason", "").strip():
                return f"superseded without a reason: {disp.get('clause')!r}"
        else:
            return f"unknown disposition status: {status!r}"
    missing = [orig for n, orig in parent_normed.items() if n not in seen]
    if missing:
        return ("silent deletion — no disposition for parent clause(s): "
                + "; ".join(repr(m) for m in missing))
    return None


def split_second_property_asserts(parent_props, child_asserts):
    """Child must carry the parent's asserts verbatim plus EXACTLY one new
    one. Returns (new_assert, None) or (None, error)."""
    p = [_norm(x) for x in parent_props]
    c = [_norm(x) for x in child_asserts]
    if len(c) != len(p) + 1:
        return None, (f"expected exactly one new assertion: parent has "
                      f"{len(p)}, child has {len(c)}")
    if c[:len(p)] != p:
        return None, (f"parent assertion(s) not preserved verbatim: "
                      f"{p} vs {c[:len(p)]}")
    return child_asserts[-1], None


def remove_asserts(source, exprs):
    """Blank the assert(...) statements whose expression matches one of
    `exprs` (whitespace-normalized), leaving legal Verilog — used to
    isolate the second property for the proves-unaided check."""
    targets = {_norm(e) for e in exprs}
    out = source
    for m in reversed(list(re.finditer(r"\bassert\b", source))):
        i = m.end()
        while i < len(source) and source[i].isspace():
            i += 1
        if i >= len(source) or source[i] != "(":
            continue
        depth, j = 0, i
        while j < len(source):
            if source[j] == "(":
                depth += 1
            elif source[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if _norm(source[i + 1:j]) in targets:
            out = out[:m.start()] + out[j + 1:]
    return out


_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VERILOG_NOISE = {"posedge", "negedge", "always", "assert", "module",
                  "endmodule", "begin", "end", "if", "else", "wire", "reg",
                  "input", "output", "initial", "assign"}


def identifiers(text):
    text = _mask_comments(text)   # comment words are not identifiers
    ids = set(_ID.findall(re.sub(r"\d+'[bodhBODH][0-9a-fA-FxXzZ_?]+", " ",
                                 text)))
    return ids - _VERILOG_NOISE


def has_coupling_clause(new_invs, parent_ids, new_ids):
    for clause in new_invs:
        ids = identifiers(clause)
        if ids & new_ids and ids & parent_ids:
            return True
    return False


# --- prompt / call / grade ---

def build_prompt(parent, ext_type, move=None):
    tmpl = STRUCTURAL_TEMPLATE if ext_type == "structural" else SECOND_TEMPLATE
    fields = dict(
        verilog=parent["verilog"],
        property="\n".join(parent["property"]),
        invariants="\n".join(parent["invariants"]),
        invariant_rules=INVARIANT_RULES,
        disposition_block=DISPOSITION_BLOCK,
        diff_format=DIFF_FORMAT)
    if ext_type == "structural":
        fields["move"] = f"{move}   {MOVES[move]}"
    return tmpl.format(**fields)


def call_model(prompt_text, schema):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT,
                       "format": {"type": "json_schema", "schema": schema}},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_text}],
    )
    if resp.stop_reason == "refusal":
        return None, resp
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp


def grade_step4(parent, ext_type, out, record):
    """apply -> property discipline -> dispositions -> yosys -> oracle
    (-> P2-unaided for second property). Fills record in place."""
    record["patch"] = out["patch"]
    record["invariants"] = out["invariants"]
    record["dispositions"] = out["dispositions"]

    try:
        child = apply_patch(parent["verilog"], out["patch"])
    except PatchError as exc:
        record["verdict"] = "PATCH_ERROR"
        record["error"] = str(exc)
        return record
    record["child_verilog"] = child

    child_asserts = extract_asserts(child)
    if ext_type == "structural":
        if [_norm(a) for a in child_asserts] != \
                [_norm(a) for a in parent["property"]]:
            record["verdict"] = "PROPERTY_CHANGED"
            record["error"] = (f"structural patches must not touch asserts: "
                               f"parent {parent['property']} vs child "
                               f"{child_asserts}")
            return record
        child_props = parent["property"]
    else:
        new_assert, err = split_second_property_asserts(
            parent["property"], child_asserts)
        if err:
            record["verdict"] = "PROPERTY_CHANGED"
            record["error"] = err
            return record
        record["second_property"] = new_assert
        child_props = child_asserts

    err = check_dispositions(parent["invariants"], out["invariants"],
                             out["dispositions"])
    if err:
        record["verdict"] = "DISPOSITION_ERROR"
        record["error"] = err
        return record

    parent_bits = state_bits(parent["verilog"], parent["top_module"])
    child_bits = state_bits(child, parent["top_module"])
    record["parent_state_bits"] = parent_bits
    record["child_state_bits"] = child_bits
    if parent_bits is None or child_bits is None:
        record["verdict"] = "YOSYS_ERROR"
        record["error"] = ("parent" if parent_bits is None else "child") + \
            " does not read in yosys"
        return record
    record["state_grew"] = child_bits > parent_bits

    parent_ids = identifiers(parent["verilog"])
    new_ids = identifiers(child) - parent_ids
    record["new_identifiers"] = sorted(new_ids)
    record["has_coupling_clause"] = has_coupling_clause(
        out["invariants"], parent_ids, new_ids)

    if ext_type == "second":
        # P2 alone, no invariants: PROVEN here means it teaches nothing.
        p2_only = remove_asserts(child, parent["property"])
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / f"{parent['top_module']}.sv"
            f.write_text(p2_only)
            with Spinner("checking the second property does not prove unaided"):
                solo = grade(f, PropertyInfo(top_module=parent["top_module"],
                                             clock=parent["clock"]),
                             **GRADE_KWARGS)
        record["p2_unaided_tier"] = solo.tier.name
        if solo.tier.name == "PROVEN":
            record["verdict"] = "P2_PROVES_UNAIDED"
            record["error"] = ("second property proves by k-induction with "
                               "no invariants — teaches nothing")
            return record

    payload = {"verilog": child, "top_module": parent["top_module"],
               "clock": parent["clock"], "antecedents": parent["antecedents"],
               "sanity_covers": parent["sanity_covers"],
               "invariants": out["invariants"]}
    t0 = time.monotonic()
    with Spinner(f"oracle grading extended {parent['top_module']}"):
        result = grade_triple_generated(json.dumps(payload), **GRADE_KWARGS)
    record["grade_wall_s"] = round(time.monotonic() - t0, 2)
    record["verdict"] = result.verdict.name
    record["reason"] = result.reason
    record["result"] = asdict(result)
    return record


def report(record, ext_type):
    print(f"\npatch:\n{record.get('patch', '<none>')}\n")
    print("invariants:", json.dumps(record.get("invariants", []), indent=1))
    print("dispositions:", json.dumps(record.get("dispositions", []), indent=1))
    if "state_grew" in record:
        print(f"state: {record['parent_state_bits']} -> "
              f"{record['child_state_bits']} bits "
              f"(new ids: {', '.join(record['new_identifiers']) or 'none'})")
    if ext_type == "structural":
        print(f"coupling clause present: {record.get('has_coupling_clause')}")
    if ext_type == "second":
        print(f"second property: {record.get('second_property')}")
        print(f"P2 unaided tier: {record.get('p2_unaided_tier')}")
    print(f"verdict: {record['verdict']} — "
          f"{record.get('reason', record.get('error'))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent")
    p.add_argument("--type", choices=["structural", "second"])
    p.add_argument("--move", choices=sorted(MOVES))
    p.add_argument("--dry", action="store_true")
    p.add_argument("--plan", action="store_true")
    args = p.parse_args()

    if args.plan:
        print("five seeds, one structural move each; distractor and second "
              "property run on the same five (15 reads total):\n")
        for pid, name, move in PLAN:
            print(f"  venv/bin/python extender/extend_one.py --parent {pid}"
                  f"            # distractor ({name})")
            print(f"  venv/bin/python extender/extend.py --parent {pid} "
                  f"--type structural --move {move}")
            print(f"  venv/bin/python extender/extend.py --parent {pid} "
                  f"--type second")
        return

    if not args.parent or not args.type:
        p.error("--parent and --type required (or use --plan)")
    if args.type == "structural" and not args.move:
        p.error("--move required for structural extensions")

    parent = load_parent(args.parent)
    print(f"parent {parent['id']} {parent['top_module']} "
          f"invariants={parent['invariants']}")
    prompt = build_prompt(parent, args.type, args.move)
    if args.dry:
        print("\n=== SYSTEM ===\n" + SYSTEM_PROMPT)
        print("\n=== USER ===\n" + prompt)
        return

    schema = STRUCTURAL_SCHEMA if args.type == "structural" else SECOND_SCHEMA
    record = {"extension_id": datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
              "ext_type": args.type, "move": args.move,
              "parent_id": parent["id"], "model": MODEL, "effort": EFFORT}
    with Spinner(f"{MODEL} writing a {args.type} extension"):
        out, resp = call_model(prompt, schema)
    record["usage"] = {"input": resp.usage.input_tokens,
                       "output": resp.usage.output_tokens}
    if out is None:
        record["verdict"] = "REFUSED"
        dump(record)
        print("refusal — logged")
        return
    for k in ("new_state", "coupling", "induction_gap",
              "why_parent_insufficient", "claim", "shared_state",
              "not_inductive_alone", "shared_clause"):
        if k in out:
            record[k] = out[k]
    grade_step4(parent, args.type, out, record)
    dump(record)
    report(record, args.type)


if __name__ == "__main__":
    main()
