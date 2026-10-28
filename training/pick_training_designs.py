"""The designs of a training file, in one corpus (28 Oct 2026).

Rishab: train on "our good designs ... plus some designs from open source
... let's add 30 more to them if they are mildly weak (preferably extended
designs)". Then, after that file scored hard 2.67 and easy 5.0 per pass on
144 pairs: "take all weak designs too and only remove the wrong ones", and
"the 216 works" (unlabeled starting designs). The groups:

  good       labeled good (labels_good_weak_wrong_partial_2026-10-27.jsonl)
  mild       the 30 mildly weak ones (mildly_weak_30_for_training_2026-10-28.json)
  weak       every design labeled weak (the 30 mild ones included)
  wrong      every design labeled wrong (for the "do wrong designs hurt" run)
  os         real open-source modules our pipeline made into designs, whose
             property and facts our model wrote (corpus_open_source_seeded_2026-10-28.jsonl)
  unlabeled  the 216 never-labeled starting designs: September's starting
             designs, 9 Oct's, 26 Oct's and a few others. Left out: the oldest
             corpus (no good one in 32 labeled) and every grown design never
             labeled (September's grown ones were 63% wrong, Claude's October
             ones 48%).

Rows keep the fields the 1-step check and build_sft.py read (id, verilog,
top_module, invariants, generation); ids say which group a row is from.

  venv/bin/python training/pick_training_designs.py                       # good, mild, os
  venv/bin/python training/pick_training_designs.py --groups good,weak,os
"""
import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / "extender"
DESIGNS = EXT / "all_unique_designs_2026-10-27.jsonl"
LABELS = EXT / "labels_good_weak_wrong_partial_2026-10-27.jsonl"
MILD = EXT / "mildly_weak_30_for_training_2026-10-28.json"
OPEN_SOURCE = EXT / "corpus_open_source_seeded_2026-10-28.jsonl"
GROUPS = ("good", "mild", "weak", "wrong", "os", "unlabeled")   # and weak:N
WEAK_SEED = 20261029
# the first two files' names are kept as they were made
NAMES = {("good", "mild", "os"): "corpus_good_mild_open_source_2026-10-28.jsonl",
         ("good", "weak", "os"): "corpus_good_weak_open_source_2026-10-28.jsonl"}
OLDEST = "extender/corpus_old.jsonl"


def out_path(groups):
    name = NAMES.get(tuple(groups)) or \
        "corpus_" + "_".join("open_source" if g == "os" else g for g in groups) + "_2026-10-28.jsonl"
    return EXT / name


def keys_of(group, labels):
    if group.startswith("weak:"):
        # the first N of one fixed shuffle (29 Oct 2026), so a smaller share
        # always sits inside a larger one: 67 in 69 in 148 in 244
        weak = sorted(r["key"] for r in labels if r["label"] == "weak")
        random.Random(WEAK_SEED).shuffle(weak)
        return weak[:int(group.split(":")[1])]
    if group in ("good", "weak", "wrong"):
        return [r["key"] for r in labels if r["label"] == group]
    if group == "mild":
        return [d["key"] for d in json.load(open(MILD))["designs"]]
    if group == "unlabeled":
        return [r["key"] for r in labels if not r["label"] and r["steps_from_start"] == 0
                and r["appears_in"][0]["file"] != OLDEST]
    return []


def rows(groups=("good", "mild", "os")):
    designs = {json.loads(l)["key"]: json.loads(l) for l in open(DESIGNS)}
    labels = [json.loads(l) for l in open(LABELS)]
    out = []
    for group in groups:
        if group == "os":
            # an id from the module (29 Oct 2026): os0_NNN follows the order the
            # seeds proved in, which changed when the list was rebuilt, and the
            # 1-step log is keyed by id
            for l in open(OPEN_SOURCE):
                r = json.loads(l)
                out.append(dict(r, id="os_" + re.sub(r"\W", "_", r["open_source"]["seed"]),
                                group="os"))
            continue
        for k in keys_of(group, labels):
            d = designs[k]
            g = group.split(":")[0]
            out.append({"id": f"{g}_{k}", "group": group, "key": k,
                        "generation": d.get("generation", 0), "verilog": d["verilog"],
                        "top_module": d["top_module"], "clock": d.get("clock") or "clk",
                        "property": d.get("property"), "invariants": d["invariants"],
                        "appears_in": d["sources"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="good,mild,os",
                    help="comma-separated, from " + ", ".join(GROUPS) + ", weak:N")
    ap.add_argument("--name", help="output name, e.g. good_weak148_2026-10-29 "
                    "(writes corpus_<name>.jsonl)")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",")]
    bad = [g for g in groups if g not in GROUPS and not re.fullmatch(r"weak:\d+", g)]
    if bad:
        raise SystemExit(f"unknown group(s) {bad}; choose from {GROUPS}")
    out = rows(groups)
    path = EXT / f"corpus_{args.name}.jsonl" if args.name else out_path(groups)
    with open(path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    counts = {g: sum(1 for r in out if r["group"] == g) for g in groups}
    print(f"{len(out)} designs -> {path.relative_to(HERE.parent)}: {counts}")


if __name__ == "__main__":
    main()
