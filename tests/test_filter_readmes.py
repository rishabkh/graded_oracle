"""The theme README pool, strictly digital hardware (8 Oct 2026). Every
README was judged by two independent judges against a written rule, a
third settling disagreements; the pool is rebuilt from those verdicts by
a fixed rule, never by hand."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "initiator"))

import filter_readmes as fr                                      # noqa: E402

TEXT = "A real project README. " * 20


def cand(repo, topic, stars):
    return {"repo": repo, "readme": TEXT, "topic": topic, "stars": stars}


def verdicts(**cats):
    return {r: {"category": c, "decided_by": "both judges", "reasons": ["x"]}
            for r, c in cats.items()}


def test_only_digital_hardware_is_kept_and_placeholders_never():
    old = [{"repo": "example/stack-lib", "readme": TEXT},
           {"repo": "verilator/verilator", "readme": TEXT},
           {"repo": "simdjson/simdjson", "readme": TEXT}]
    new = [cand("a/cpu", "verilog", 300), cand("b/os", "riscv", 200)]
    v = verdicts(**{"verilator/verilator": "digital-hardware",
                    "simdjson/simdjson": "not-hardware",
                    "a/cpu": "digital-hardware", "b/os": "other-hardware"})
    pool = fr.rebuild(old, new, v, target=10)
    assert [r["repo"] for r in pool] == ["verilator/verilator", "a/cpu"]
    assert pool[0] == old[1]                      # an old row is kept as it was


def test_riscv_spelled_two_ways_takes_one_turn():
    new = ([cand(f"v{i}", "verilog", 100 - i) for i in range(3)]
           + [cand("r1", "risc-v", 90), cand("r2", "riscv", 95),
              cand("r3", "risc-v", 80)])
    v = verdicts(**{c["repo"]: "digital-hardware" for c in new})
    pool = fr.rebuild([], new, v, target=4)
    # verilog and RISC-V alternate; within RISC-V the most starred first
    assert [r["repo"] for r in pool] == ["v0", "r2", "v1", "r1"]


def test_the_higher_star_tier_comes_first():
    new = [cand("low", "verilog", 20), cand("high", "fpga", 60)]
    v = verdicts(low="digital-hardware", high="digital-hardware")
    assert [r["repo"] for r in fr.rebuild([], new, v, target=1)] == ["high"]


def test_every_candidate_must_have_a_verdict():
    with pytest.raises(ValueError):
        fr.rebuild([], [cand("x/y", "fpga", 99)], {}, target=5)


def test_the_command_writes_the_pool_and_the_record(tmp_path):
    pool = tmp_path / "readmes.jsonl"
    pool.write_text(json.dumps({"repo": "old/hw", "readme": TEXT}) + "\n")
    c = tmp_path / "cands.jsonl"
    c.write_text("".join(json.dumps(cand(f"n{i}", "fpga", 99 - i)) + "\n"
                         for i in range(3)))
    v = tmp_path / "verdicts.json"
    v.write_text(json.dumps(dict(verdicts(**{"old/hw": "digital-hardware"}),
                                 **verdicts(n0="digital-hardware",
                                            n1="not-hardware",
                                            n2="digital-hardware"))))
    record = tmp_path / "readme_record.json"
    fr.main(["--pool", str(pool), "--candidates", str(c), "--verdicts", str(v),
             "--target", "5", "--record", str(record)])
    assert [json.loads(l)["repo"] for l in pool.read_text().splitlines()] == [
        "old/hw", "n0", "n2"]
    rec = json.loads(record.read_text())
    assert rec["rule"] and rec["rubric"] and rec["kept"] == 3
    assert rec["verdicts"]["n1"]["category"] == "not-hardware"
