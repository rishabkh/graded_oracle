"""Verdict router and corpus promotion — verdict to action, nothing more:

  NECESSARY      accept: write the child at generation parent+1
  DECORATIVE     reject, no repair — the added structure closed the gap,
                 the invariant is no longer load-bearing
  NOT_PROVEN /   queue for the Fixer (evidence rides with the record)
  NOT_INDUCTIVE
  FALSE          drop the branch — broken hardware, no invariant helps
  ERROR/TIMEOUT/ retire — non-verdicts
  INCONCLUSIVE
  (machinery)    reformat — PATCH_ERROR, NO_AGGREGATE, TRUNCATED, ... are
                 format failures; the cheap fix is a fresh call with the
                 lesson, never a repair model

Rates are logged because they are the health signal: FALSE at 5% means
dropping is right; FALSE at 40% means the extension prompt is generating
broken hardware and that must be known before a Fixer papers over it.

Idempotent: children are deduped by content hash, so re-running promote
over the same log never double-writes.

  venv/bin/python extender/promote.py          # rates + promote (needs yosys)
  venv/bin/python extender/promote.py --dry    # rates only, write nothing
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import (extract_asserts, pdr_wall_s, state_bits,   # noqa: E402
                          template)

CORPUS = HERE / "corpus.jsonl"
EXT_LOG = HERE / "logs" / "extensions.jsonl"
FIXER_QUEUE = HERE / "logs" / "fixer_queue.jsonl"

_ACTIONS = {
    "NECESSARY": "accept",
    "DECORATIVE": "reject",
    "NOT_PROVEN": "fixer",
    "NOT_INDUCTIVE": "fixer",
    "FALSE": "drop",
    "ERROR": "retire",
    "TIMEOUT": "retire",
    "INCONCLUSIVE": "retire",
}


def route(verdict):
    """Everything not in the oracle's vocabulary is a machinery/format
    verdict: retry with the lesson, not Fixer food."""
    return _ACTIONS.get(verdict, "reformat")


_MODULES = re.compile(r"\bmodule\s+([A-Za-z_]\w*)")


def child_top(record, parent):
    """Wrapper types append their new top module last; patch types keep
    the parent's."""
    if record.get("ext_type") in ("replicate", "compose"):
        names = _MODULES.findall(record["child_verilog"])
        if names:
            return names[-1]
    return parent["top_module"]


def _content_hash(verilog, invariants):
    return hashlib.sha256(
        (verilog + "\n" + json.dumps(sorted(invariants))).encode()).hexdigest()


def promote(corpus_rows, ext_records, compute_metrics=True):
    """Route every extension record; build g+1 corpus rows for the
    accepted ones. Returns (buckets, new_rows)."""
    by_id = {r["id"]: r for r in corpus_rows}
    seen_hashes = {r["content_hash"] for r in corpus_rows
                   if "content_hash" in r}
    next_seq = Counter()
    for r in corpus_rows:
        m = re.fullmatch(r"g(\d+)_(\d+)", r["id"])
        if m:
            gen, seq = int(m.group(1)), int(m.group(2))
            next_seq[gen] = max(next_seq[gen], seq + 1)

    buckets = {a: [] for a in
               ("accept", "reject", "fixer", "drop", "retire", "reformat")}
    new_rows = []
    for rec in ext_records:
        action = route(rec.get("verdict", "ERROR"))
        buckets[action].append(rec)
        if action != "accept":
            continue
        parent = by_id.get(rec.get("parent_id"))
        child_v = rec.get("child_verilog")
        if parent is None or not child_v:
            continue
        h = _content_hash(child_v, rec.get("invariants", []))
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        gen = parent.get("generation", 0) + 1
        seq = next_seq[gen]
        next_seq[gen] += 1
        wrapper_type = rec.get("ext_type") in ("replicate", "compose")
        top = child_top(rec, parent)
        invariants = rec.get("invariants", [])
        row = {
            "id": f"g{gen}_{seq:03d}",
            "generation": gen,
            "parent": rec.get("parent_id"),
            "parent2": rec.get("parent2_id"),
            "source_extension_id": rec.get("extension_id"),
            "ext_type": rec.get("ext_type"),
            "move": rec.get("move"),
            "content_hash": h,
            "top_module": top,
            "clock": parent.get("clock", "clk"),
            "antecedents": (rec.get("antecedents") if wrapper_type
                            else parent.get("antecedents", [])) or [],
            "sanity_covers": (rec.get("sanity_covers") if wrapper_type
                              else parent.get("sanity_covers", [])) or [],
            "verilog": child_v,
            "property": extract_asserts(child_v),
            "invariants": invariants,
            "metrics": {
                "state_bits": (state_bits(child_v, top)
                               if compute_metrics else None),
                "pdr_wall_s": pdr_wall_s(rec),
                "clause_count": len(invariants),
                "invariant_templates": [template(x) for x in invariants],
            },
        }
        new_rows.append(row)
    return buckets, new_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true",
                   help="print rates, write nothing")
    args = p.parse_args()

    corpus_rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    ext_records = [json.loads(l) for l in EXT_LOG.read_text().splitlines()]

    buckets, new_rows = promote(corpus_rows, ext_records,
                                compute_metrics=not args.dry)

    total = sum(len(v) for v in buckets.values())
    print(f"{total} extension records routed:")
    for action, recs in buckets.items():
        if not recs:
            continue
        verdicts = Counter(r.get("verdict") for r in recs)
        pct = 100 * len(recs) / total
        print(f"  {action:9s} {len(recs):3d} ({pct:4.1f}%)  {dict(verdicts)}")

    if args.dry:
        print(f"\n--dry: {len(new_rows)} row(s) would be promoted")
        return

    if new_rows:
        with CORPUS.open("a") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")
        for row in new_rows:
            print(f"promoted {row['id']}  {row['top_module']:24s} "
                  f"gen={row['generation']} parent={row['parent']}"
                  + (f"+{row['parent2']}" if row.get("parent2") else "")
                  + f"  state_bits={row['metrics']['state_bits']}")
    else:
        print("nothing new to promote")

    queued_ids = set()
    if FIXER_QUEUE.exists():
        queued_ids = {json.loads(l).get("extension_id")
                      for l in FIXER_QUEUE.read_text().splitlines()}
    added = 0
    with FIXER_QUEUE.open("a") as f:
        for rec in buckets["fixer"]:
            if rec.get("extension_id") in queued_ids:
                continue
            f.write(json.dumps(rec, default=str) + "\n")
            added += 1
    print(f"fixer queue: +{added} (total "
          f"{len(queued_ids) + added})")


if __name__ == "__main__":
    main()
