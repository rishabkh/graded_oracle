"""Choose which parent to extend next.

The old order was a plain shuffle, which made depth an accident of the
population: with 100 generation-zero rows and 60 calls, a run never
reached a child at all, so generation 2 only appeared in later runs.

Two forces decide the order here:

  depth   deeper rows first, because depth is what the corpus lacks and
          what the solver baseline shows is hard (Qwen: 55% on gen 0,
          0% on gen 2)
  rarity  the surprisal score from select_rows, so a deep row made of
          shapes the corpus already has ranks below an unusual one

and one guard: a cap per FAMILY, a family being everything descended
from one generation-zero design. Without it, depth-first feeds the
deepest lineage forever. That is not hypothetical: every maximum-size
row in the corpus from generation 1 onward is the same REPLICATE
descendant, so sorting by depth alone would keep drawing it.

Families are dealt round-robin, so every lineage gets a turn before any
lineage gets a second.
"""
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from select_rows import pooled_stats, surprisals                # noqa: E402


def family_of(row, by_id, _limit=64):
    """The generation-zero ancestor's id. A row with no parent in the
    corpus is its own family."""
    seen = set()
    while row.get("parent") and row["parent"] in by_id:
        if row["id"] in seen or len(seen) > _limit:
            break            # a cycle would mean a corrupt corpus
        seen.add(row["id"])
        row = by_id[row["parent"]]
    return row["id"]


def _rarity(row, pooled):
    scores = surprisals(row, pooled)
    return max(scores.values()) if scores else 0.0


def order_frontier(rows, by_id, per_family=3):
    """Deepest and most unusual first, dealt round-robin across families
    and capped so no lineage can absorb a run."""
    if not rows:
        return []
    pooled = pooled_stats(rows)

    families = defaultdict(list)
    for row in rows:
        families[family_of(row, by_id)].append(row)
    for members in families.values():
        members.sort(key=lambda r: (-r.get("generation", 0),
                                    -_rarity(r, pooled), r["id"]))
        del members[per_family:]

    # families themselves compete on their best member, then deal one
    # row from each in turn
    ranked = sorted(families.values(),
                    key=lambda m: (-m[0].get("generation", 0),
                                   -_rarity(m[0], pooled), m[0]["id"]))
    out, depth = [], 0
    while any(len(m) > depth for m in ranked):
        for members in ranked:
            if len(members) > depth:
                out.append(members[depth])
        depth += 1
    return out
