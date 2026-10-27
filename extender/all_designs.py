"""Every unique design that ever passed our checker, in one file.

27 Oct 2026, Rishab: "check each and every unique design we ever created".
Reads every corpus file and every initiator log, keeps the designs with
verdict NECESSARY, and folds together copies of the same design text
(spaces ignored). Each row says where the design appears, which design it
grew from, and what it was meant to be: the initiator's plan and kind for a
starting design, and for a grown design its starting design's description
followed by what each step added. September's and the oldest designs were
never saved with a description, so theirs is None.

  venv/bin/python extender/all_designs.py      # writes OUT, prints counts
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from build_corpus import extract_asserts                       # noqa: E402

OUT = HERE / "all_unique_designs_2026-10-27.jsonl"

CORPORA = [
    ("extender/corpus.jsonl", "September designs (initiator + old extender)"),
    ("extender/corpus_old.jsonl", "oldest corpus (before the September rebuild)"),
    ("extender/corpus_2026-10-09.jsonl", "9 Oct initiator designs"),
    ("extender/corpus_initiator_real_hardware_all_2026-10-26.jsonl",
     "26 Oct initiator designs (the 110)"),
    ("extender/corpus_extended_trial_2026-10-15.jsonl", "extender trial 15 Oct (Claude)"),
    ("extender/corpus_growth_trial_2026-10-17.jsonl", "extender trial 17 Oct (Claude)"),
    ("extender/corpus_growth_trial_2026-10-18.jsonl", "extender trial 18 Oct stricter (Claude)"),
    ("extender/corpus_good_growth_trial_2026-10-18.jsonl",
     "extender trial 18 Oct start-from-biggest (Claude)"),
    ("extender/corpus_real_systems_trial_2026-10-21.jsonl", "extender trial 21 Oct systems (Claude)"),
    ("extender/corpus_engine_stuck_trial_2026-10-23.jsonl", "extender run 23 Oct engine-stuck (Claude)"),
    ("extender/corpus_initiator_110_grown_with_gpt_sol_2026-10-26.jsonl",
     "extender run 27 Oct (GPT-6.1 Sol)"),
    ("extender/corpus_extra_for_training_2026-10-26.jsonl", "26 Oct training extras"),
]
LOGS = [
    ("initiator/logs/attempts.jsonl", "September initiator log"),
    ("initiator/logs/attempts_2026-10-09.jsonl", "9 Oct initiator log"),
    ("initiator/logs/attempts_catalog.jsonl", "catalog trial (old settings)"),
    ("initiator/logs/trial_v5.jsonl", "initiator trial 8 Oct"),
    ("initiator/logs/trial_big.jsonl", "initiator trial 9 Oct biggest sizes"),
    ("initiator/logs/generation_test_with_design_writing_300_2026-10-12.jsonl",
     "designs written by our trained model (design-writing test, 12 Oct)"),
    ("initiator/logs/generation_test_untrained_same_test_as_2026-10-12.jsonl",
     "designs written by untrained Qwen (12 Oct test)"),
    ("initiator/logs/initiator_real_hardware_features_40_2026-10-25.jsonl", "25 Oct initiator log"),
    ("initiator/logs/initiator_explanation_after_design_40_2026-10-26.jsonl", "26 Oct initiator log"),
    ("initiator/logs/initiator_real_hardware_runs_combined_2026-10-26.jsonl",
     "26 Oct initiator log, combined"),
]


def key_of(verilog):
    return hashlib.sha1(re.sub(r"\s+", " ", verilog).strip().encode()).hexdigest()[:12]


def plan_text(row):
    """What the initiator was asked to build and planned, written before the
    design: the kind, the real-hardware feature it had to have, its plan."""
    parts = []
    if row.get("construct"):
        parts.append("What it is: " + row["construct"])
    if row.get("trait"):
        parts.append("A feature it must have: " + row["trait"])
    if row.get("idea"):
        parts.append("The designer's plan, written before the code:\n" + row["idea"])
    return "\n".join(parts) or None


def collect(repo=REPO):
    designs, by_file_id = {}, {}

    def add(rec, src):
        k = key_of(rec["verilog"])
        if k not in designs:
            designs[k] = dict(rec, key=k, sources=[])
        d = designs[k]
        for f in ("property", "invariants", "top_module", "clock", "added", "move",
                  "ext_type", "plan"):
            if not d.get(f) and rec.get(f):
                d[f] = rec[f]
        if rec.get("generation", 0) > 0 and not d.get("parent_ref"):
            d.update(parent_ref=rec.get("parent_ref"), parent2_ref=rec.get("parent2_ref"),
                     generation=rec["generation"])
        d["sources"].append(src)
        return k

    for f, what in CORPORA:
        for line in open(repo / f):
            r = json.loads(line)
            rec = {"verilog": r["verilog"], "property": r.get("property"),
                   "invariants": r.get("invariants") or [], "top_module": r.get("top_module"),
                   "clock": r.get("clock"), "generation": r.get("generation", 0),
                   "added": r.get("added"), "move": r.get("move"), "ext_type": r.get("ext_type"),
                   "parent_ref": (f, r["parent"]) if r.get("parent") else None,
                   "parent2_ref": (f, r["parent2"]) if r.get("parent2") else None}
            by_file_id[(f, r["id"])] = add(rec, {"file": f, "id": r["id"], "what": what})

    for f, what in LOGS:
        for i, line in enumerate(open(repo / f)):
            try:
                r = json.loads(line)
                t = json.loads(r["raw_json"]) if r.get("verdict") == "NECESSARY" else None
            except (ValueError, KeyError, TypeError):
                continue
            if not t or not t.get("verilog"):
                continue
            try:
                prop = extract_asserts(t["verilog"])
            except Exception:
                prop = None
            rec = {"verilog": t["verilog"], "property": prop,
                   "invariants": t.get("invariants") or [], "top_module": t.get("top_module"),
                   "clock": t.get("clock"), "generation": 0, "plan": plan_text(r)}
            add(rec, {"file": f, "line": i, "what": what,
                      "model": r.get("served_model") or r.get("model")})

    for d in designs.values():
        for fld in ("parent_ref", "parent2_ref"):
            ref = d.pop(fld, None)
            d[fld.replace("_ref", "_key")] = by_file_id.get(tuple(ref)) if ref else None

    # a grown design's description: its starting design's, then each step
    for d in designs.values():
        chain, cur, seen = [], d, set()
        while cur.get("parent_key") and cur["key"] not in seen:
            seen.add(cur["key"])
            chain.append(cur)
            cur = designs.get(cur["parent_key"]) or {}
        start = cur.get("plan")
        if d.get("generation", 0) == 0 or not chain:
            d["description"] = start
        elif start:
            steps = [s.get("added") or "(no description saved)" for s in reversed(chain)]
            d["description"] = (start + "\nThen grown in steps; each step added:\n"
                                + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps)))
        else:
            d["description"] = None
    return designs


def main(argv=None):
    designs = collect()
    with open(OUT, "w") as fh:
        for d in designs.values():
            fh.write(json.dumps({k: v for k, v in d.items() if k != "plan"}) + "\n")
    gen = collections.Counter(d.get("generation", 0) for d in designs.values())
    print(f"{len(designs)} unique designs -> {OUT.relative_to(REPO)}")
    print("by steps from start:", dict(sorted(gen.items())))
    print("with a description:", sum(1 for d in designs.values() if d["description"]))


if __name__ == "__main__":
    main()
