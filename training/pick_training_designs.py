"""The designs of the next training file, in one corpus (28 Oct 2026).

Rishab: train on "our good designs ... plus some designs from open source
... let's add 30 more to them if they are mildly weak (preferably extended
designs)". So:

  good   the designs labeled good (labels_good_weak_wrong_partial_2026-10-27.jsonl)
  mild   the 30 mildly weak ones (mildly_weak_30_for_training_2026-10-28.json)
  os     real open-source modules our pipeline made into designs, whose
         property and facts our model wrote (corpus_open_source_seeded_2026-10-28.jsonl)

Rows keep the fields the 1-step check and build_sft.py read (id, verilog,
top_module, invariants, generation); ids say which group a row is from.

  venv/bin/python training/pick_training_designs.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / "extender"
DESIGNS = EXT / "all_unique_designs_2026-10-27.jsonl"
LABELS = EXT / "labels_good_weak_wrong_partial_2026-10-27.jsonl"
MILD = EXT / "mildly_weak_30_for_training_2026-10-28.json"
OPEN_SOURCE = EXT / "corpus_open_source_seeded_2026-10-28.jsonl"
OUT = EXT / "corpus_good_mild_open_source_2026-10-28.jsonl"


def rows():
    designs = {json.loads(l)["key"]: json.loads(l) for l in open(DESIGNS)}
    good = [json.loads(l)["key"] for l in open(LABELS) if json.loads(l)["label"] == "good"]
    mild = [d["key"] for d in json.load(open(MILD))["designs"]]
    out = []
    for group, keys in (("good", good), ("mild", mild)):
        for k in keys:
            d = designs[k]
            out.append({"id": f"{group}_{k}", "group": group, "key": k,
                        "generation": d.get("generation", 0), "verilog": d["verilog"],
                        "top_module": d["top_module"], "clock": d.get("clock") or "clk",
                        "property": d.get("property"), "invariants": d["invariants"],
                        "appears_in": d["sources"]})
    for line in open(OPEN_SOURCE):
        r = json.loads(line)
        out.append(dict(r, group="os"))
    return out


def main():
    out = rows()
    with open(OUT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    groups = {g: sum(1 for r in out if r["group"] == g) for g in ("good", "mild", "os")}
    print(f"{len(out)} designs -> {OUT.relative_to(HERE.parent)}: {groups}")


if __name__ == "__main__":
    main()
