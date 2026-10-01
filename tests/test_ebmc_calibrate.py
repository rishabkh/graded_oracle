"""Proving a rebuilt EBMC judges exactly like the one the old scores came from.

The first two models were scored on the laptop's EBMC (6.0, 0308d417)
with a 300 s limit. Scoring moves to the cluster so a run needs no
laptop, which means a rebuilt binary. Before any score from it counts,
stored answers are replayed through it and every verdict must match.
Only answers judged well under the limit are replayed, so a slower or
faster machine cannot flip a verdict by timing alone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

import ebmc_calibrate as C                                     # noqa: E402


def row(bench, verdict, t=0.1, set_="hard", judgeable=True, lemmas=("a",)):
    return {"run_id": "r1", "bench": bench, "set": set_, "verdict": verdict,
            "ebmc_time": t, "file_judgeable": judgeable,
            "lemmas": list(lemmas), "served_model": "x"}


LOG = ([row(f"p{i}.sv", "PROVEN") for i in range(6)]
       + [row(f"c{i}.sv", "CEX") for i in range(6)]
       + [row(f"i{i}.sv", "INCONCLUSIVE") for i in range(6)]
       + [row("slow.sv", "PROVEN", t=250.0),
          row("t.sv", "TIMEOUT", t=300.0),
          row("nj.sv", "CEX", judgeable=False),
          dict(row("na.sv", "NO_ANSWER"), lemmas=None)])


def test_only_fast_judgeable_answered_rows_are_chosen():
    chosen = C.select(LOG, per_verdict=10, max_time=20)
    names = {r["bench"] for r in chosen}
    assert not names & {"slow.sv", "t.sv", "nj.sv", "na.sv"}


def test_every_verdict_type_is_represented_and_capped():
    chosen = C.select(LOG, per_verdict=4, max_time=20)
    by = {}
    for r in chosen:
        by[r["verdict"]] = by.get(r["verdict"], 0) + 1
    assert by == {"PROVEN": 4, "CEX": 4, "INCONCLUSIVE": 4}


def test_the_choice_is_the_same_every_time():
    assert C.select(LOG, 4, 20) == C.select(list(reversed(LOG)), 4, 20)


def test_matching_verdicts_pass_and_a_mismatch_is_named():
    rows = C.select(LOG, per_verdict=2, max_time=20)
    same = C.check(rows, lambda r: r["verdict"])
    assert all(m for _, _, m in same)
    flip = C.check(rows, lambda r: "CEX" if r["verdict"] == "PROVEN" else r["verdict"])
    bad = [(r["bench"], got) for r, got, m in flip if not m]
    assert bad and all(got == "CEX" for _, got in bad)
