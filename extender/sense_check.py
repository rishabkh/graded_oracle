"""A second model reads each combination or copy and says whether it makes
sense as hardware (18 Oct 2026).

The checker proves what a design says; gates.py rejects structural signs of
a weak design. What neither can see is a design that is consistent and
meaningless at once: in the 17 Oct trial, CRC report frames sent with zeros
where the check bits go, under a property proving exactly that. This review
asks an experienced-engineer question set about the whole design and
rejects it (verdict NONSENSE) when the parts are not connected sensibly, a
value is used against its meaning, or no engineer would build it.

It runs only when batch.py is given --sense-check, and should not be trusted
before it has been tried on the hand-labelled designs of the 17 Oct trial:

  venv/bin/python extender/sense_check.py --validate extender/logs/growth_trial_2026-10-17
  ... --validate <dir> --all      # all 28 designs, not only the 15 wrapped
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

WRAPPERS = ("compose", "replicate")
MAX_TOKENS = 16000

SYSTEM = """\
You review hardware designs for a dataset of realistic SystemVerilog. The
designs are formally verified, so they are internally consistent; your job
is to judge whether they make sense as hardware an engineer would build.
Answer plainly and strictly."""

PROMPT = """\
This design was built by {how}. In the builder's words:

{scheme}

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
   as an arithmetic operand.
3. new_logic_only_copies_existing_state: does the new logic only copy or
   shadow state the parts already hold?
4. results_of_parts_used: are the main results of every part used or
   brought out?
5. engineer_would_build_this: would an engineer build this design, as a
   whole, for a real purpose?
6. why: one or two sentences naming the main reason for your answers.

Reply with JSON."""

_FLAGS = ("parts_connected_sensibly", "value_used_against_its_meaning",
          "new_logic_only_copies_existing_state", "results_of_parts_used",
          "engineer_would_build_this")
# the answer field is "why", not "reasoning": a strict schema with a field
# of that name is refused before it reaches the model (distractor.py)
SCHEMA = {"type": "object",
          "properties": {**{f: {"type": "boolean"} for f in _FLAGS},
                         "why": {"type": "string"}},
          "required": list(_FLAGS) + ["why"],
          "additionalProperties": False}


def _how(record):
    if record.get("ext_type") == "replicate":
        n = record.get("n_instances") or "several"
        return (f"instantiating one verified design {n} times behind a "
                "shared pool")
    return "combining two verified designs through new glue logic"


def _call(prompt):
    text, _usage, _stop = llm_client.call_claude(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, user=prompt,
        schema=SCHEMA, effort=EFFORT)
    return None if text is None else json.loads(text)


def review(record, call=None):
    """Review a proved combination or copy, once. A design judged not to
    make sense gets the verdict NONSENSE (a rejection). A review that could
    not be made keeps the design and records why."""
    if record.get("verdict") != "NECESSARY" or \
            record.get("ext_type") not in WRAPPERS or "sense_check" in record:
        return record
    prompt = PROMPT.format(
        how=_how(record),
        scheme=record.get("glue_scheme") or record.get("pool_scheme")
        or "(not given)",
        verilog=record.get("child_verilog", ""))
    err, answer = None, None
    try:
        answer = (call or _call)(prompt)
    except Exception as exc:                    # network, refusal, parse
        err = f"{type(exc).__name__}: {exc}"
    if answer is None:
        record["sense_check"] = {"keep": True, "error": err or
                                 "no answer (refused, cut off or "
                                 "unparseable)"}
        return record
    keep = bool(answer["parts_connected_sensibly"]
                and not answer["value_used_against_its_meaning"]
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


def validate(log_dir, all_types=False, workers=5):
    log_dir = Path(log_dir)
    labels = json.loads((log_dir / "hand_labels.json").read_text())
    recs = {r["extension_id"]: r for r in
            map(json.loads, (log_dir / "extensions.jsonl").open())}
    todo = {}
    for cid, lab in labels.items():
        rec = copy.deepcopy(recs[lab["extension_id"]])
        if all_types or rec["ext_type"] in WRAPPERS:
            rec["verdict"] = "NECESSARY"
            rec.pop("sense_check", None)
            rec["ext_type"] = rec["ext_type"] if rec["ext_type"] in WRAPPERS \
                else "compose"                  # --all: review every kind
            todo[cid] = rec
    print(f"reviewing {len(todo)} designs with {MODEL} "
          f"(paid; about $0.03-0.10 each)")
    with llm_client.call_log(log_dir / "sense_check_calls.jsonl"):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            done = dict(zip(todo, ex.map(review, todo.values())))
    kept = {}
    with (log_dir / "sense_check_validation.jsonl").open("w") as out:
        for cid, rec in sorted(done.items()):
            sc = rec.get("sense_check", {})
            kept[cid] = sc.get("keep", True)
            out.write(json.dumps({"id": cid, "label": labels[cid]["label"],
                                  "sense_check": sc}) + "\n")
            print(f"{labels[cid]['label']:5s} {cid} "
                  f"{'kept    ' if kept[cid] else 'REJECTED'} "
                  f"{sc.get('why') or sc.get('error', '')}")
    print("\nagreement with the hand labels:")
    for lab, row in sorted(agreement(labels, kept).items()):
        print(f"  {lab:5s} kept {row['kept']}, rejected {row['rejected']}")
    print(f"results: {log_dir / 'sense_check_validation.jsonl'}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--validate", required=True, metavar="LOG_DIR",
                   help="a trial's log folder holding hand_labels.json")
    p.add_argument("--all", action="store_true",
                   help="review all labelled designs, not only the "
                        "combined and copied ones")
    args = p.parse_args(argv)
    validate(args.validate, all_types=args.all)


if __name__ == "__main__":
    main()
