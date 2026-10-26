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

Since 26 Oct 2026 (Rishab approved) it reads the design in two calls: the
first sees the code, the change, the earlier steps and what the checker's
tools found (the nothing-lost search, new input ports), never the
builder's words; only a design it keeps goes to a second call that checks
the builder's claims against the code. The first call also rejects a value
cut or padded out of its meaning, a part that is only a delayed copy of a
signal, and an item that can be lost, counted or not. For now the second
call only records what it finds (record["claims_check"], shown by
--validate) and never changes the verdict.

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

# 26 Oct 2026, Rishab approved (change 6): in the 25 Oct study 8 of 14
# combined and second-lane designs lost data and 5 of them were kept,
# because the loss was counted or flagged; the reviewer read the builder's
# words before the code and repeated them ("never refuses"). The first call
# now judges the code alone, with what the checker's own tools found, and
# is also asked about any lost item counted or not, delayed copies, and
# values cut or padded.
PROMPT = """\
This design was built by {how}.
{history}{change}
The whole design:

{verilog}
{facts}
Judge the design and the change from the code above, as an experienced
hardware engineer. What the builder says about it is checked in a separate
question later; do not assume anything the code does not show.

1. parts_connected_sensibly: are the parts connected the way a real design
   connects them? Does what one part sends mean something to the part that
   receives it?
2. value_used_against_its_meaning: is any value used against its meaning?
   For example zeros sent where check bits belong, predictable status data
   used as seed material for a random-number generator, a counter's parity
   fed to a checker that expects a pseudo-random stream, a status count used
   as an arithmetic operand, unrelated counters XOR-ed together into one
   status word.
3. value_cut_or_padded: is any value cut down or padded so that it no longer
   means what it meant? For example an address or data word cut to fewer
   bits than it carries, a count whose high bits are dropped, a data word
   padded with zeros into a wider data input, a wide input that only ever
   receives a few real bits, or a flag widened into a number. Cutting a
   wider value down to the field the input expects is fine when the dropped
   bits are a different field.
4. new_logic_only_copies_existing_state: does the new logic only copy or
   shadow state the parts already hold?
5. only_a_delayed_copy: is the new part, or the part between two designs,
   only a delayed copy of a signal (the same value one or more cycles
   later) with no job of its own?
6. results_of_parts_used: are the main results of every part used or
   brought out?
7. new_part_has_its_own_job: does the new part do a job the design did not
   already do? A further buffer in a row in front of the same output, a
   second bank that shares the same single input and output, or a second
   count of something already counted does not. For a combination, the
   part between the two designs does a job when it is a buffer that makes
   the sender wait, a conversion that keeps the meaning, or an arbiter; a
   plain wire or a delayed copy does not.
8. item_can_be_dropped: can an item offered at any input of the design, or
   handed from one part to another, be lost in some state: dropped when a
   part is full or busy, overwritten before it is used, or ignored while a
   part is busy, not idle or paused, or while a lane loses an arbitration?
   Answer true if it can, even when the loss is counted, flagged or raised
   as an error bit: counting a lost item does not make losing it
   acceptable. Making the sender wait is the right answer; losing the item
   is not. The checker's search above covers only the streams it lists,
   and only so many cycles.
9. breaks_what_it_did: does the change stop the old part doing anything it
   did before? For example a request, setting or table write it used to
   take is now ignored or lost (for instance because the change holds its
   enable low), a result it used to produce is gone, an existing input is
   now split between two lanes, or a channel can now get stuck.
10. engineer_would_build_this: would an engineer build this design, as a
   whole, for a real purpose?
11. why: one or two sentences naming the main reason for your answers.

Reply with JSON."""

# 26 Oct 2026 (change 6): what the checker's own tools found, one line each
# (checker_notes); left out when they found nothing to say
FACTS = """
What the checker's own tools found (not the builder's words):
{lines}
"""

# 26 Oct 2026 (change 6): the earlier steps are their builders' own
# descriptions, and the question now says so
HISTORY = """
Earlier steps on this design, oldest first, as their builders described
them:
{lines}
"""

CHANGE = """
The change, as a patch to the earlier design (+ adds a line, - removes one;
@@ and = lines only mark where):

{patch}
"""

# 26 Oct 2026, Rishab approved (change 6): the builder's words are read in a
# second call, only after the code has been judged, and checked against the
# code, which is shown again because a claim cannot be checked without it
# (in the 25 Oct study 8 of 12 partner answers said the receiver never
# refuses, at least twice falsely). For now the call only records what it
# finds, as record["claims_check"], and rejects nothing (decision 2 of the
# 26 Oct spec: it has not been tried on the hand-labelled trials).
CLAIMS_PROMPT = """\
You reviewed a hardware design, built by {how}, from its code alone.
{change}
The whole design:

{verilog}
{facts}
Your answers about it:

{answers}

Now read what the builder says about the design and the change:

{claims}

Does the builder claim anything about what the design does that its code
does not do? Only claims about behaviour count: for example "it never
refuses" for a part that ignores requests while busy, "nothing is lost"
when an item can be dropped, a wait the code does not have, or a width or
a meaning the code does not give a value. Judge each claim against the
code, not against how sure the builder sounds.

Reply with JSON: claims_match_design (true when no claim about behaviour
is false) and false_claims (the false claims, quoted, or "")."""

CLAIMS_SCHEMA = {"type": "object",
                 "properties": {"claims_match_design": {"type": "boolean"},
                                "false_claims": {"type": "string"}},
                 "required": ["claims_match_design", "false_claims"],
                 "additionalProperties": False}

# 26 Oct 2026, Rishab approved (change 6): value_cut_or_padded,
# only_a_delayed_copy and item_can_be_dropped are new (the forced register
# made one-cycle-late copies in 5 designs of the 25 Oct study, and 5 kept
# designs lost data that was counted or flagged); breaks_what_it_did no
# longer asks about drops, which have their own question
_FLAGS = ("parts_connected_sensibly", "value_used_against_its_meaning",
          "value_cut_or_padded", "new_logic_only_copies_existing_state",
          "only_a_delayed_copy", "results_of_parts_used",
          "new_part_has_its_own_job", "item_can_be_dropped",
          "breaks_what_it_did", "engineer_would_build_this")
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
    # 26 Oct 2026, Rishab approved (change 5): every PEER step of the 21
    # and 23 Oct trials split an existing input; the new lane now has an
    # input of its own and shares one real thing through an arbiter
    "PEER": ("a second lane with its own input that shares one storage, "
             "output or data path with the first through an arbiter"),
    "STAGE": "pipeline stages",
    "SPLIT": "a quantity split across two registers",
    "BOUND": "a limit held in a register",
    "GUARD": "an enable on an existing update",
    "LEDGER": "a pair of event counters",
}
# the builder's own account of each kind of step; since 26 Oct 2026 only
# the second call, which checks the builder's claims, reads it
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


# 26 Oct 2026, Rishab approved (change 6): the reviewer is shown what the
# checker's own tools found. nothing_lost.py words the lines from the
# record (the streams a step declared and the search's result, new input
# ports, a part between two designs with no registers). It is loaded only
# when a review is made: it brings in extend and the proof tools, which
# this module does not otherwise need.
def checker_notes(record):
    """What the checker's own tools found about the step, one plain line
    each (nothing_lost.reviewer_notes)."""
    import nothing_lost
    return nothing_lost.reviewer_notes(record)


# 26 Oct 2026 (change 6): the checker's notes and the change, shown by both
# questions
def _facts(notes):
    return (FACTS.format(lines="\n".join(f"  - {x}" for x in notes))
            if notes else "")


def _change(record):
    patched = record.get("ext_type") in PATCHED and record.get("patch")
    return CHANGE.format(patch=record["patch"]) if patched else ""


def build(record, history=None, notes=None):
    """The design question for one record: the design, the change, the
    family's earlier steps (`history`, oldest first: promote.history of the
    parent) and what the checker's tools found (`notes`, by default
    checker_notes of the record). Since 26 Oct 2026 it holds none of the
    builder's words about the step."""
    lines = "\n".join(f"  {i}. {h}" for i, h in enumerate(history or [], 1))
    return PROMPT.format(
        how=_how(record),
        history=HISTORY.format(lines=lines) if history else "",
        change=_change(record),
        verilog=record.get("child_verilog", ""),
        facts=_facts(checker_notes(record) if notes is None else notes))


def builder_words(record):
    """What the builder said about its step: its own account
    (_SCHEME_FIELD) and, for a combination, the plan it chose its partner
    by; "" when it said nothing (26 Oct 2026)."""
    said = [record.get(_SCHEME_FIELD.get(record.get("ext_type"), "")) or ""]
    if record.get("ext_type") == "compose":
        plan = record.get("partner_choice") or {}
        if plan.get("system"):
            said.append(f"Planned system: {plan['system']}")
        if plan.get("connection"):
            said.append(f"Planned connection: {plan['connection']}")
    return "\n".join(s for s in said if s.strip())


def build_claims(record, answer, notes=None):
    """The second question (26 Oct 2026): the same design and change, the
    design call's `answer`, the checker's notes, and then the builder's
    words, to check against the code."""
    return CLAIMS_PROMPT.format(
        how=_how(record),
        change=_change(record),
        verilog=record.get("child_verilog", ""),
        facts=_facts(checker_notes(record) if notes is None else notes),
        answers=json.dumps({k: answer[k] for k in _FLAGS + ("why",)},
                           indent=1),
        claims=builder_words(record))


# 26 Oct 2026 (change 6): each question has its own schema (SCHEMA or
# CLAIMS_SCHEMA); both keep SYSTEM, so the realism report counts both as
# reviews
def _call(prompt, schema=SCHEMA):
    text, _usage, stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, user=prompt,
        schema=schema, effort=EFFORT)
    if text is None:
        raise RuntimeError(f"no answer ({stop})")
    return json.loads(text)


# 26 Oct 2026, Rishab approved (change 6): both calls are tried the same
# way, and an answer that leaves out a question (the design call now has
# 11) is a failed try, never a crash on the missing field.
def _ask(prompt, schema, required, call):
    """One answer with every required field, in TRIES tries; (answer, err).
    A reply missing a field counts as a failed try, never a crash."""
    err = None
    for _ in range(TRIES):
        try:
            answer = call(prompt) if call else _call(prompt, schema)
        except Exception as exc:        # network, refusal, cut off, parse
            err = f"{type(exc).__name__}: {exc}"
            continue
        if answer is None:              # refused or cut off (a fake's way)
            continue
        if not isinstance(answer, dict):
            err = "the answer is not a JSON object"
            continue
        missing = [f for f in required if f not in answer]
        if missing:
            err = f"the answer leaves out {', '.join(missing)}"
            continue
        return answer, None
    return None, err or "no answer (refused, cut off or unparseable)"


def review(record, call=None, history=None):
    """Review a proved step, once. The design call judges the code: a
    design judged not to make sense gets the verdict NONSENSE, one that
    could not be reviewed in TRIES tries UNREVIEWED; both are rejections.
    A design it keeps whose builder said something about it then goes to
    the claims call, whose finding is only recorded, as
    record["claims_check"], and changes no verdict (26 Oct 2026).
    `call(prompt)` stands in for the model in tests and answers both."""
    if record.get("verdict") != "NECESSARY" or "sense_check" in record:
        return record
    notes = checker_notes(record)
    answer, err = _ask(build(record, history, notes), SCHEMA,
                       SCHEMA["required"], call)
    if answer is None:
        record["sense_check"] = {"keep": False, "error": err, "tries": TRIES}
        record["verdict"] = "UNREVIEWED"
        record["reason"] = f"sense check not made in {TRIES} tries: {err}"
        return record
    # 26 Oct 2026, Rishab approved (change 6): five kept designs were kept
    # because their drops were "flagged", and the reviewer repeated the
    # builder's "never refuses"; it now judges the code before it reads the
    # builder's words and is asked about drops counted or not, delayed
    # copies and cut or padded values.
    keep = bool(answer["parts_connected_sensibly"]
                and not answer["value_used_against_its_meaning"]
                and not answer["value_cut_or_padded"]
                and not answer["only_a_delayed_copy"]
                and answer["new_part_has_its_own_job"]
                and not answer["item_can_be_dropped"]
                and not answer["breaks_what_it_did"]
                and answer["engineer_would_build_this"])
    record["sense_check"] = dict(answer, keep=keep)
    if not keep:
        record["verdict"] = "NONSENSE"
        record["reason"] = "sense check: " + answer.get("why", "")
        return record
    if not builder_words(record):       # the builder said nothing to check
        return record
    # 26 Oct 2026 (change 6, the spec's decision 2): recorded only. A false
    # claim, or a claims call that fails twice, leaves the verdict and keep
    # as the design call made them.
    found, err = _ask(build_claims(record, answer, notes), CLAIMS_SCHEMA,
                      CLAIMS_SCHEMA["required"], call)
    record["claims_check"] = (dict(found) if found is not None
                              else {"error": err, "tries": TRIES})
    return record


def agreement(labels, kept):
    """{hand label: {"kept": n, "rejected": n}} for the reviewed designs."""
    table = {}
    for cid, lab in labels.items():
        if cid in kept:
            row = table.setdefault(lab["label"], {"kept": 0, "rejected": 0})
            row["kept" if kept[cid] else "rejected"] += 1
    return table


# 26 Oct 2026, Rishab approved (change 6): the claims call rejects nothing
# for now (the spec's decision 2), so the report shows what it found, per
# design and per hand label: what making it reject would change.
def claims_line(found):
    """One line on what the claims call found (record["claims_check"])."""
    if "error" in found:
        return f"the builder's claims could not be checked: {found['error']}"
    if found.get("claims_match_design"):
        return "the builder's claims match the code"
    return ("the builder claims what the code does not do (recorded, not a "
            f"rejection): {found.get('false_claims', '')}")


def claims_agreement(labels, records):
    """{hand label: {"match": n, "false": n, "unchecked": n}} over the
    designs whose builder's claims went to the claims call."""
    table = {}
    for cid, lab in labels.items():
        found = (records.get(cid) or {}).get("claims_check")
        if found is None:
            continue
        row = table.setdefault(lab["label"],
                               {"match": 0, "false": 0, "unchecked": 0})
        row["unchecked" if "error" in found else
            "match" if found.get("claims_match_design") else "false"] += 1
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
    the calls in the first. Returns the agreement over all of them. Since
    26 Oct 2026 each design's line also says what the claims call found,
    which decides nothing yet."""
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
            # the claims call's finding from the run is made again here
            # (26 Oct 2026), like the review itself
            rec.pop("claims_check", None)
            todo[key] = rec
            row = rows.get(cid)
            hist[key] = (family_history(rows.get(row.get("parent")), rows)
                         if row else [])
    # 26 Oct 2026 (change 6): a kept design whose builder said something
    # gets a second call, about as long as the first
    print(f"reviewing {len(todo)} designs with {MODEL} "
          f"(paid; about $0.05-0.15 each, up to twice that for a kept "
          f"design, which gets a second call)")
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
                        "sense_check": rec.get("sense_check", {}),
                        # 26 Oct 2026 (change 6): what the claims call
                        # found; null when none was made
                        "claims_check": rec.get("claims_check")}) + "\n")
    for key, rec in done.items():
        sc = rec.get("sense_check", {})
        print(f"{labels[key]['label']:5s} {key} "
              f"{'kept    ' if kept[key] else 'REJECTED'} "
              f"{sc.get('why') or sc.get('error', '')}")
        if "claims_check" in rec:
            print(f"      {claims_line(rec['claims_check'])}")
    table = agreement(labels, kept)
    print("\nagreement with the hand labels:")
    for lab, row in sorted(table.items()):
        print(f"  {lab:5s} kept {row['kept']}, rejected {row['rejected']}")
    claims = claims_agreement(labels, done)
    if claims:
        print("\nthe builder's claims, checked on kept designs (recorded, "
              "not a rejection):")
        for lab, row in sorted(claims.items()):
            print(f"  {lab:5s} match {row['match']}, false {row['false']}, "
                  f"not checked {row['unchecked']}")
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
