"""Two knobs the depth experiment needs.

The corpus stops at generation 3 and every lineage is capped at three
frontier entries, both hard-coded. That is right for building a broad
corpus and wrong for the one question we cannot answer from the rows we
have: does a single lineage keep growing if you let it? Answering it
needs a run aimed at one family, writing somewhere other than the real
corpus."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extender"))

import batch as B                                              # noqa: E402
from batch import run_loop                                     # noqa: E402


def row(rid, gen=0, parent=None, top="m"):
    return {"id": rid, "generation": gen, "parent": parent,
            "top_module": top, "clock": "clk", "antecedents": [],
            "sanity_covers": [],
            "verilog": f"module {top} (input wire clk, output reg [1:0] a);\n"
                       "endmodule",
            "invariants": ["a <= 2"], "metrics": {}}


# one deep lineage plus one lone row, so a per-family cap is visible
DEEP = [row("g0_a"), row("g1_a", 1, "g0_a"), row("g2_a", 2, "g1_a"),
        row("g0_b")]


def _parents_tried(per_family, max_gen=3):
    """Every parent the loop ever reaches, with an executor that always
    rejects so each branch dies after BRANCH_FAILS and the run ends when
    the frontier is exhausted."""
    seen = []
    run_loop(DEEP, lambda task, rows: {"verdict": "DECORATIVE"},
             max_calls=500, max_gen=max_gen, order="depth",
             per_family=per_family,
             on_task=lambda t: seen.append(t["parent_id"]))
    return set(seen)


def test_per_family_cap_is_settable_from_the_caller():
    """With the cap at one, the deep lineage contributes a single row,
    so only two of the four parents are ever tried."""
    assert _parents_tried(per_family=1) == {"g2_a", "g0_b"}
    assert _parents_tried(per_family=3) == {"g2_a", "g1_a", "g0_a", "g0_b"}


def test_a_run_can_read_a_corpus_other_than_the_real_one(tmp_path, capsys):
    """The depth probe must not append its throwaway rows to the corpus
    the training file is built from."""
    scratch = tmp_path / "lineage.jsonl"
    scratch.write_text("\n".join(json.dumps(r) for r in DEEP[:2]))

    B.main(["--dry", "--max-calls", "1", "--corpus", str(scratch)])
    assert "frontier: 2 extendable rows" in capsys.readouterr().out
