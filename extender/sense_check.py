"""A second model reads each proved step and says whether it makes sense
as hardware (18 Oct 2026).

The checker proves what a design says; gates.py rejects structural signs of
a weak design. What neither can see is a design that is consistent and
meaningless at once: in the 17 Oct trial, CRC report frames sent with zeros
where the check bits go, under a property proving exactly that. This review
asks an experienced-engineer question set about the whole design and
rejects it (verdict NONSENSE) when the parts are not connected sensibly, a
value is used against its meaning, the new part has no job of its own, the
change breaks what the design did, or no engineer would build it.

It read only combinations and copies until 18 Oct; it now reads every
proved step, because in the 18 Oct trial good steps ran out after about two
and a step built on a weak design was good 3 times in 18: only what it
keeps is grown. It sees the change, the whole design and the family's
earlier steps. A review that cannot be made in two tries (a refusal, a
cut-off answer) rejects the design (UNREVIEWED): the one design kept that
way in the 18 Oct trial was weak.

It runs only when batch.py is given --sense-check, and should not be
trusted before it has been tried on the hand-labelled trials:

  venv/bin/python extender/sense_check.py --validate \\
      extender/logs/growth_trial_2026-10-17 extender/logs/growth_trial_2026-10-18
"""
import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import llm_client                                               # noqa: E402
from distractor import EFFORT, MODEL                            # noqa: E402
from promote import history as family_history                   # noqa: E402

WRAPPERS = ("compose", "replicate")
PATCHED = ("structural", "second", "distractor")
MAX_TOKENS = 16000
TRIES = 2

SYSTEM = """\
You review hardware designs for a dataset of realistic SystemVerilog. The
designs are formally verified, so they are internally consistent; your job
is to judge whether they make sense as hardware an engineer would build.
Answer plainly and strictly."""

PROMPT = """\
This design was built by {how}. In the builder's words:

{scheme}
{history}{change}
The whole design:

{verilog}

Answer as an experienced hardware engineer:

1. parts_connected_sensibly: are the parts connected the way a real design
   connects them? Does what one part sends mean something to the part that
   receives it?
2. value_used_against_its_meaning: is any value used against its meaning?
   For example zeros sent where check bits belong, predictable status data
   used as seed material for a random-number generator, a counter's parity
   fed to a checker that expects a pseudo-random stream, a status count used
   as an arithmetic operand, unrelated counters XOR-ed together into one
   status word.
3. new_logic_only_copies_existing_state: does the new logic only copy or
   shadow state the parts already hold?
4. results_of_parts_used: are the main results of every part used or
   brought out?
5. new_part_has_its_own_job: does the new part do a job the design did not
   already do? A further buffer in a row in front of the same output, a
   second bank that shares the same single input and output, or a second
   count of something already counted does not.
6. breaks_what_it_did: does the change stop the design doing what it did
   before, or does a new part drop data silently instead of making the
   sender wait? For example data that used to get through is now dropped,
   or a channel can now get stuck.
7. engineer_would_build_this: would an engineer build this design, as a
   whole, for a real purpose?
8. why: one or two sentences naming the main reason for your answers.

Reply with JSON."""

HISTORY = """
Earlier steps on this design, oldest first:
{lines}
"""

CHANGE = """
The change, as a patch to the earlier design (+ adds a line, - removes one;
@@ and = lines only mark where):

{patch}
"""

_FLAGS = ("parts_connected_sensibly", "value_used_against_its_meaning",
          "new_logic_only_copies_existing_state", "results_of_parts_used",
          "new_part_has_its_own_job", "breaks_what_it_did",
          "engineer_would_build_this")
# the answer field is "why", not "reasoning": a strict schema with a field
# of that name is refused before it reaches the model (distractor.py)
SCHEMA = {"type": "object",
          "properties": {**{f: {"type": "boolean"} for f in _FLAGS},
                         "why": {"type": "string"}},
          "required": list(_FLAGS) + ["why"],
          "additionalProperties": False}

_MOVE_WORDS = {
    "SUBUNIT": "a new functional unit",
    "STORAGE": "new storage (a queue, register file or table)",
    "PEER": "a second unit that shares a resource with the first",
    "STAGE": "pipeline stages",
    "SPLIT": "a quantity split across two registers",
    "BOUND": "a limit held in a register",
    "GUARD": "an enable on an existing update",
    "LEDGER": "a pair of event counters",
}
# the builder's own account of each kind of step
_SCHEME_FIELD = {"compose": "glue_scheme", "replicate": "pool_scheme",
                 "structural": "new_state", "second": "claim",
                 "distractor": "reasoning"}


def _how(record):
    kind = record.get("ext_type")
    if kind == "replicate":
        n = record.get("n_instances") or "several"
        return (f"instantiating one verified design {n} times behind a "
                "shared pool")
    if kind == "compose":
        return "combining two verified designs through new glue logic"
    if kind == "second":
        return ("adding a second property, and any state it needs, to a "
                "verified design")
    if kind == "distractor":
        return ("adding logic the design's property never looks at "
                "(statistics, status or debug) to a verified design")
    return (f"adding {_MOVE_WORDS.get(record.get('move'), 'new logic')} "
            "to a verified design")


def build(record, history=None):
    """The question for one record; `history` is the family's earlier
    steps, oldest first (promote.history of the parent)."""
    lines = "\n".join(f"  {i}. {h}" for i, h in enumerate(history or [], 1))
    patched = record.get("ext_type") in PATCHED and record.get("patch")
    return PROMPT.format(
        how=_how(record),
        scheme=record.get(_SCHEME_FIELD.get(record.get("ext_type"), ""))
        or "(not given)",
        history=HISTORY.format(lines=lines) if history else "",
        change=CHANGE.format(patch=record["patch"]) if patched else "",
        verilog=record.get("child_verilog", ""))


def _call(prompt):
    text, _usage, stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, user=prompt,
        schema=SCHEMA, effort=EFFORT)
    if text is None:
        raise RuntimeError(f"no answer ({stop})")
    return json.loads(text)


def review(record, call=None, history=None):
    """Review a proved step, once. A design judged not to make sense gets
    the verdict NONSENSE, one that could not be reviewed in TRIES tries
    UNREVIEWED; both are rejections."""
    if record.get("verdict") != "NECESSARY" or "sense_check" in record:
        return record
    prompt = build(record, history)
    answer, err = None, None
    for _ in range(TRIES):
        try:
            answer = (call or _call)(prompt)
        except Exception as exc:        # network, refusal, cut off, parse
            err = f"{type(exc).__name__}: {exc}"
            answer = None
        if answer is not None:
            break
    if answer is None:
        why = err or "no answer (refused, cut off or unparseable)"
        record["sense_check"] = {"keep": False, "error": why, "tries": TRIES}
        record["verdict"] = "UNREVIEWED"
        record["reason"] = f"sense check not made in {TRIES} tries: {why}"
        return record
    keep = bool(answer["parts_connected_sensibly"]
                and not answer["value_used_against_its_meaning"]
                and answer["new_part_has_its_own_job"]
                and not answer["breaks_what_it_did"]
                and answer["engineer_would_build_this"])
    record["sense_check"] = dict(answer, keep=keep)
    if not keep:
        record["verdict"] = "NONSENSE"
        record["reason"] = "sense check: " + answer.get("why", "")
    return record


def agreement(labels, kept):
    """{hand label: {"kept": n, "rejected": n}} for the reviewed designs."""
    table = {}
    for cid, lab in labels.items():
        if cid in kept:
            row = table.setdefault(lab["label"], {"kept": 0, "rejected": 0})
            row["kept" if kept[cid] else "rejected"] += 1
    return table


def _jsonl(path):
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def trial_corpus(log_dir):
    """A trial's rows by id, from extender/corpus_<trial folder>.jsonl
    (the way the trials are laid out); {} when there is none."""
    path = HERE / f"corpus_{Path(log_dir).name}.jsonl"
    return {r["id"]: r for r in _jsonl(path)} if path.exists() else {}


def validate(log_dirs, workers=5, call=None):
    """Review every hand-labelled design of the given trial folders, each
    with its family's earlier steps, and count agreement with the hand
    labels. Writes sense_check_every_step.jsonl into each folder and logs
    the calls in the first. Returns the agreement over all of them."""
    labels, todo, hist, folder = {}, {}, {}, {}
    for d in map(Path, log_dirs):
        recs = {r["extension_id"]: r for r in _jsonl(d / "extensions.jsonl")}
        rows = trial_corpus(d)
        for cid, entry in json.loads(
                (d / "hand_labels.json").read_text()).items():
            key = f"{d.name}/{cid}"
            labels[key], folder[key] = entry, d
            rec = copy.deepcopy(recs[entry["extension_id"]])
            rec["verdict"] = "NECESSARY"
            rec.pop("sense_check", None)
            todo[key] = rec
            row = rows.get(cid)
            hist[key] = (family_history(rows.get(row.get("parent")), rows)
                         if row else [])
    print(f"reviewing {len(todo)} designs with {MODEL} "
          f"(paid; about $0.05-0.15 each)")
    with llm_client.call_log(Path(log_dirs[0])
                             / "sense_check_every_step_calls.jsonl"):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            done = dict(zip(todo, ex.map(
                lambda k: review(todo[k], call=call, history=hist[k]),
                todo)))
    kept = {k: rec.get("verdict") == "NECESSARY" for k, rec in done.items()}
    for d in map(Path, log_dirs):
        with (d / "sense_check_every_step.jsonl").open("w") as out:
            for key, rec in done.items():
                if folder[key] == d:
                    out.write(json.dumps({
                        "id": key.split("/", 1)[1],
                        "label": labels[key]["label"],
                        "verdict": rec.get("verdict"),
                        "sense_check": rec.get("sense_check", {})}) + "\n")
    for key, rec in done.items():
        sc = rec.get("sense_check", {})
        print(f"{labels[key]['label']:5s} {key} "
              f"{'kept    ' if kept[key] else 'REJECTED'} "
              f"{sc.get('why') or sc.get('error', '')}")
    table = agreement(labels, kept)
    print("\nagreement with the hand labels:")
    for lab, row in sorted(table.items()):
        print(f"  {lab:5s} kept {row['kept']}, rejected {row['rejected']}")
    print("results: " + ", ".join(
        str(Path(d) / "sense_check_every_step.jsonl") for d in log_dirs))
    return table


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--validate", required=True, nargs="+", metavar="LOG_DIR",
                   help="trial log folders holding hand_labels.json; every "
                        "labelled design in them is reviewed")
    args = p.parse_args(argv)
    validate(args.validate)


if __name__ == "__main__":
    main()
