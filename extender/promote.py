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

  venv/bin/python extender/promote.py --write  # rates + promote (needs yosys)
  venv/bin/python extender/promote.py          # rates only, write nothing
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

from build_corpus import (extract_asserts, honest_lines,             # noqa: E402
                          pdr_wall_s, state_bits, template,
                          without_main_s)
from coi import coi_ratio                                            # noqa: E402
import gates                                                         # noqa: E402
from extend import BIG_MOVES                                         # noqa: E402

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
    # proved, but no harder than its parent (see ratchet): correct, so
    # nothing for the Fixer, and it counts against the parent's branch
    "NO_NEW_SHAPE": "reject",
    # proved, but weak or wrong as a design (gates.py, 18 Oct 2026): the
    # facts are not what is wrong, so nothing for the Fixer either
    **{v: "reject" for v in gates.VERDICTS},
    # judged meaningless as hardware by the second-model review
    # (sense_check.py, run only with batch.py --sense-check)
    "NONSENSE": "reject",
    # the review could not be made, twice: not known to be good, so not
    # kept or grown (18 Oct 2026; a refused review had kept a design)
    "UNREVIEWED": "reject",
    # no candidate partner fits the design (19 Oct 2026): not a failure of
    # the design, so it goes back to the frontier and its branch is not
    # charged
    "NO_PARTNER": "pass",
}


def fixable(rec):
    """A proof failure the Fixer can act on. A with-leg that ended in
    ERROR (toolchain missing, cover run crashed) carries no CTI - there
    is nothing to repair, so it never enters the queue. Nor does one that
    ran out of time (8 Oct 2026): a slow proof is not a wrong fact, and
    the fixer would be paid to change facts that may be fine."""
    reason = str(rec.get("reason", ""))
    return "grade is ERROR" not in reason and "grade is TIMEOUT" not in reason


def route(verdict):
    """Everything not in the oracle's vocabulary is a machinery/format
    verdict: retry with the lesson, not Fixer food."""
    return _ACTIONS.get(verdict, "reformat")


# The distractor copies its invariants on purpose, as the control for
# irrelevant logic, so no shape rule can apply to it.
RATCHET_EXEMPT = {"distractor"}


def new_shapes(child_invariants, parent_invariants):
    """Clause shapes (identifiers and constants blanked) the child has
    and no parent clause has. Renamed copies of an old clause are the
    same idea again and do not count."""
    return sorted({template(c) for c in child_invariants}
                  - {template(c) for c in parent_invariants})


def growth(record, parents):
    """How much bigger the child is than its biggest parent: the larger
    of the line growth and the flip-flop growth, as a fraction (0.5 is
    half again as big). None when nothing can be compared. Recorded for
    reports only: from 17 to 18 Oct 2026 a child that grew by half was
    kept without a new kind of claim, and the only children that rule
    kept were three meaningless combinations and a contrived 64-bit
    drain window (hand review of the 17 Oct trial), so it was dropped."""
    rows = [p for p in parents if p and p.get("verilog")]
    child_v = record.get("child_verilog")
    out = []
    if rows and child_v:
        # copies counted once (19 Oct 2026): a block written once per slot
        # is not growth
        before = max(honest_lines(p["verilog"]) for p in rows)
        if before:
            out.append(honest_lines(child_v) / before - 1)
    pb, cb = record.get("parent_state_bits"), record.get("child_state_bits")
    if isinstance(pb, list):
        pb = max([b for b in pb if b] or [0])
    if pb and cb:
        out.append(cb / pb - 1)
    return round(max(out), 3) if out else None


# 19 Oct 2026: three sub-units of the 18 Oct trial that doubled their
# designs with real features (a result retire queue, frame pause and
# resume, a playback controller) were turned away only because their facts
# had familiar shapes. A big step that grew its design by half or more may
# skip the rule when the second reviewer reads it (review_waiver); the
# reviewer then decides. From 17 to 18 Oct a similar pass, with no
# reviewer, let only weak and wrong designs in.
WAIVER_GROWTH = 0.5


def ratchet(record, parents, review_waiver=False):
    """Keep a proved child only if it needs a kind of claim its
    parent(s) did not, and it passes the design checks in gates.py.
    With `review_waiver` (the run's reviewer is on), a big structural step
    that grew by WAIVER_GROWTH or more is passed to the reviewer instead.

    Measured 26 Sep 2026 over 142 parent-to-child pairs: with NECESSARY
    as the only rule, 56% of children left the signals-per-clause count
    unchanged and 27 proved faster than their parents, so generation
    number counted edits, not progress. Under this rule 16 of the 142
    would have been turned away. It asks for a new shape rather than a
    higher shape count, so a child that swaps an old idea for a new one
    still passes. Idempotent; mutates and returns the record."""
    if record.get("verdict") != "NECESSARY":
        return record
    if record.get("ext_type") not in RATCHET_EXEMPT:
        old = [c for row in parents if row
               for c in row.get("invariants", [])]
        fresh = new_shapes(record.get("invariants", []), old)
        record["new_shapes"] = fresh
        record["grew_by"] = growth(record, parents)
        if not fresh and not record.get("shape_waiver"):
            if (review_waiver and record.get("ext_type") == "structural"
                    and record.get("move") in BIG_MOVES
                    and (record["grew_by"] or 0) >= WAIVER_GROWTH):
                record["shape_waiver"] = ("no new kind of claim; kept only "
                                          "if the reviewer keeps it")
            else:
                record["verdict"] = "NO_NEW_SHAPE"
                record["reason"] = ("proved, but needs no kind of claim its "
                                    "parent did not already have")
                return record
    # once per record: the result is kept on it (gates.apply)
    gates.apply(record, parents)
    return record


# --- what a family already has (18 Oct 2026) ---
# On a long chain the model could not see what earlier steps had added and
# added the same kind of part again (a third, fourth and fifth buffer in a
# row). Each row now says what its step added; batch.py lists a family's
# steps in the next question and limits how often one kind is drawn.

# the builder's own account of each kind of step
_ADDED_FIELD = {"structural": "new_state", "second": "claim",
                "distractor": "reasoning", "compose": "glue_scheme",
                "replicate": "pool_scheme"}


def what_was_added(rec, limit=300):
    """The first two sentences of the builder's account of the step, or
    the first alone when two run long (19 Oct 2026: a first sentence such
    as "A new sub-module fm_pkt_guard is added." says nothing of its
    job)."""
    text = " ".join(str(rec.get(_ADDED_FIELD.get(rec.get("ext_type")), "")
                        or "").split())
    ends = [m.start() + 1 for m in re.finditer(r"\.(\s|$)", text)]
    first = text[:ends[0]] if ends else text
    two = text[:ends[1]] if len(ends) > 1 else text
    for cut in (two, first):
        if len(cut) <= limit:
            return cut
    return first[:limit].rstrip() + "..."


def step_kind(row):
    """The move for a structural step, the type for the others; None for
    a starting design."""
    if not row.get("ext_type"):
        return None
    return row.get("move") or row["ext_type"]


def lineage(row, by_id, _seen=None):
    """The steps that made `row`, oldest first, through both parents of a
    combination; `row` itself last when it is a step."""
    seen = _seen if _seen is not None else set()
    if row is None or row["id"] in seen:
        return []
    seen.add(row["id"])
    out = []
    for pid in (row.get("parent"), row.get("parent2")):
        if pid:
            out += lineage(by_id.get(pid), by_id, seen)
    if step_kind(row):
        out.append(row)
    return out


def _said(r):
    return f"{step_kind(r)}: {r['added']}" if r.get("added") \
        else step_kind(r)


def history(row, by_id):
    """`kind: what it added` for each step that made `row`, oldest first
    (rows from before 18 Oct 2026 say only their kind)."""
    return [_said(r) for r in lineage(row, by_id)]


def children(row, by_id):
    """`kind: what it added` for each design already grown from `row`
    (19 Oct 2026: two siblings added the same length guard)."""
    if row is None:
        return []
    return [_said(r) for r in by_id.values()
            if r.get("parent") == row["id"] and step_kind(r)]


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
    # names count per (letter, generation): a child keeps its parent's
    # letter, so the catalog corpus's c0_000 has children c1_..., never
    # v2's g1_... names (8 Oct 2026)
    next_seq = Counter()
    for r in corpus_rows:
        m = re.fullmatch(r"([a-z])(\d+)_(\d+)", r["id"])
        if m:
            key = (m.group(1), int(m.group(2)))
            next_seq[key] = max(next_seq[key], int(m.group(3)) + 1)

    buckets = {a: [] for a in
               ("accept", "reject", "fixer", "drop", "retire", "reformat",
                "pass")}
    new_rows = []
    for rec in ext_records:
        ratchet(rec, [by_id.get(rec.get("parent_id")),
                      by_id.get(rec.get("parent2_id"))])
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
        letter = parent["id"][0] if re.match(r"[a-z]\d", parent["id"]) else "g"
        seq = next_seq[(letter, gen)]
        next_seq[(letter, gen)] += 1
        wrapper_type = rec.get("ext_type") in ("replicate", "compose")
        top = child_top(rec, parent)
        invariants = rec.get("invariants", [])
        row = {
            "id": f"{letter}{gen}_{seq:03d}",
            "generation": gen,
            "parent": rec.get("parent_id"),
            "parent2": rec.get("parent2_id"),
            "source_extension_id": rec.get("extension_id"),
            "ext_type": rec.get("ext_type"),
            "move": rec.get("move"),
            "added": what_was_added(rec),
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
                "coi_ratio": (coi_ratio(child_v, top)
                              if compute_metrics else None),
                "pdr_wall_s": pdr_wall_s(rec),
                # the branch stop reads this when a later run starts
                "without_main_s": without_main_s(rec),
                "clause_count": len(invariants),
                "invariant_templates": [template(x) for x in invariants],
            },
        }
        if rec.get("invariants_compact"):
            # the facts as written, one compact fact for every entry
            # (21 Oct 2026); "invariants" holds them expanded
            row["invariants_compact"] = rec["invariants_compact"]
        new_rows.append(row)
    return buckets, new_rows


def main(argv=None):
    """Reports by default. Writing needs --write: found 8 Oct 2026, a plain
    run would have added 115 old rows to extender/corpus.jsonl, because
    batch.py already promotes as it goes and the log holds records whose
    rows were since rebuilt."""
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true",
                   help="print rates, write nothing (the default)")
    p.add_argument("--write", action="store_true",
                   help="append the promoted rows to the corpus")
    p.add_argument("--no-metrics", action="store_true",
                   help="skip yosys metrics on promoted rows")
    args = p.parse_args(argv)
    args.dry = args.dry or not args.write

    corpus_rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    ext_records = [json.loads(l) for l in EXT_LOG.read_text().splitlines()]

    buckets, new_rows = promote(corpus_rows, ext_records,
                                compute_metrics=not (args.dry or args.no_metrics))

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
            if rec.get("extension_id") in queued_ids or not fixable(rec):
                continue
            f.write(json.dumps(rec, default=str) + "\n")
            added += 1
    print(f"fixer queue: +{added} (total "
          f"{len(queued_ids) + added})")


if __name__ == "__main__":
    main()
