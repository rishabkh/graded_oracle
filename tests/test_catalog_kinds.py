"""Task 5's new hardware kinds are chosen mechanically: one call to a
fresh model with a fixed prompt that names neither the benchmark nor
anything we saw in the variety map, then a draw fixed by a seed written
down beforehand. The model is faked here: no calls, no money."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import catalog_kinds as ck                                       # noqa: E402

OURS = ["a credit counter: a producer spends credits",
        "a one-hot FSM; exactly one state bit is ever set"]

# The benchmark, its sources, and every variety the map showed we lack.
# Our own 32 kinds go into the prompt (so the model skips them), so only
# the fixed wording around them is checked.
MAP_WORDS = ["benchmark", "lemma miners", "ebmc", "technion", "ijcai",
             "hwmcc", "gulwani", "itc99", "vcegar", "v2c", "neuralmc",
             "processor", "cpu", "serial", "ethernet", "usb", "lookup",
             "content-addressable", "cam", "protocol", "arbiter",
             "loop program", "c program", "64", "wide", "width",
             "verification", "induction", "lemma"]


def reply(n, dup=False):
    kinds = [{"name": f"block {i}", "description": f"does job {i}; keeps r{i}"}
             for i in range(n)]
    if dup:
        kinds.append({"name": "Block 3", "description": "the same again"})
    return json.dumps({"kinds": kinds})


def test_the_prompt_names_neither_the_benchmark_nor_anything_from_the_map():
    text = ck.PROMPT.lower()
    for word in MAP_WORDS:
        assert not re.search(rf"\b{re.escape(word)}\b", text), word


def test_our_kinds_are_listed_so_the_model_skips_them():
    p = ck.build_prompt(200, OURS)
    assert all(k in p for k in OURS) and "200" in p


def test_the_draw_is_fixed_by_the_seed():
    kinds = json.loads(reply(200))["kinds"]
    a = ck.pick(kinds, 50, seed=20261007)
    assert a == ck.pick(kinds, 50, seed=20261007)
    assert a != ck.pick(kinds, 50, seed=1)
    assert len(a) == 50 and all(k in kinds for k in a)


def test_a_repeated_name_is_dropped_before_the_draw():
    kinds = ck.parse(reply(5, dup=True))
    assert [k["name"] for k in kinds] == [f"block {i}" for i in range(5)]


@pytest.mark.parametrize("text", [None, "not json", '{"kinds": []}',
                                  '{"kinds": [{"name": "x"}]}'])
def test_an_unusable_reply_gives_no_kinds(text):
    assert ck.parse(text) is None


def test_a_kind_becomes_one_generator_line():
    line = ck.construct_line({"name": "Barrel shifter",
                              "description": "Shifts a word by any amount.\n"})
    assert line == "Barrel shifter: Shifts a word by any amount."
    assert "\n" not in line


class FakeModel:
    def __init__(self, monkeypatch, text):
        self.calls = []
        self.text = text
        monkeypatch.setattr(ck.llm_client, "call_claude", self.call)

    def call(self, **kw):
        self.calls.append(kw)
        return self.text, {"input": 2000, "output": 9000}, "ok"


def run(tmp_path, *extra):
    ck.main(["--out-dir", str(tmp_path), "--ours", str(tmp_path / "ours.txt"),
             *extra])


def test_a_run_records_everything_needed_to_repeat_it(monkeypatch, tmp_path):
    (tmp_path / "ours.txt").write_text("\n".join(OURS) + "\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fake = FakeModel(monkeypatch, reply(200))
    run(tmp_path, "--ask", "200", "--pick", "50", "--seed", "7")
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call.get("system") is None                 # a fresh model, no context
    [record] = list(tmp_path.glob("kinds_*.json"))
    rec = json.loads(record.read_text())
    assert rec["prompt"] == call["user"]
    assert rec["raw_reply"] == reply(200)
    assert rec["model"] and rec["effort"] and rec["timestamp"]
    assert rec["seed"] == 7 and rec["asked"] == 200 and len(rec["kinds"]) == 200
    assert rec["usage"] == {"input": 2000, "output": 9000}
    assert rec["picked"] == ck.pick(rec["kinds"], 50, seed=7)
    [lines] = list(tmp_path.glob("constructs_*.txt"))
    assert lines.read_text().splitlines() == [ck.construct_line(k)
                                              for k in rec["picked"]]


def test_too_few_kinds_keeps_the_paid_reply_but_writes_no_kind_list(
        monkeypatch, tmp_path):
    (tmp_path / "ours.txt").write_text("\n".join(OURS) + "\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    FakeModel(monkeypatch, reply(30))
    with pytest.raises(SystemExit):
        run(tmp_path, "--ask", "200", "--pick", "50")
    [record] = list(tmp_path.glob("kinds_*.json"))
    assert json.loads(record.read_text())["raw_reply"] == reply(30)
    assert not list(tmp_path.glob("constructs_*.txt"))


def test_dry_prints_the_prompt_and_calls_nothing(monkeypatch, tmp_path, capsys):
    (tmp_path / "ours.txt").write_text("\n".join(OURS) + "\n")
    fake = FakeModel(monkeypatch, reply(200))
    run(tmp_path, "--dry")
    assert fake.calls == [] and OURS[0] in capsys.readouterr().out
    assert not list(tmp_path.glob("kinds_*.json"))


def test_using_every_kind_needs_no_call_and_keeps_the_seeded_ones_first(
        monkeypatch, tmp_path):
    """7 Oct 2026: 150 kinds were drawn from the 250 returned, then all 250
    were wanted. The other 100 come from the saved reply: no new call,
    and still no hand choice, since every kind is used."""
    kinds = json.loads(reply(250))["kinds"]
    rec = {"run_id": "r1", "seed": 7, "kinds": kinds,
           "picked": ck.pick(kinds, 150, seed=7)}
    path = tmp_path / "kinds_r1.json"
    path.write_text(json.dumps(rec))
    fake = FakeModel(monkeypatch, reply(250))
    ck.main(["--out-dir", str(tmp_path), "--use-all", str(path)])
    assert fake.calls == []
    lines = (tmp_path / "constructs_r1_all.txt").read_text().splitlines()
    assert len(lines) == 250 == len(set(lines))
    assert lines[:150] == [ck.construct_line(k) for k in rec["picked"]]
    note = json.loads((tmp_path / "kinds_r1_all.json").read_text())
    assert note["from_record"] == "kinds_r1.json" and note["count"] == 250
    assert note["rule"] and note["timestamp"]


# --- category by category, toward 1,000 kinds (8 Oct 2026) ---------------
# One reply cannot hold 700+ kinds without being cut off or repeating, so
# a fresh model first lists categories, then kinds within each category,
# leaving out every kind we already have. Exact repeats are dropped by
# name; near repeats are judged separately (two judges) and applied by
# --finish, which then makes the seeded draw.

def test_the_category_prompts_name_nothing_from_the_map():
    for template in (ck.CATEGORY_PROMPT, ck.KINDS_IN_CATEGORY_PROMPT):
        text = template.lower()
        for word in MAP_WORDS:
            assert not re.search(rf"\b{re.escape(word)}\b", text), word


def test_a_category_prompt_asks_for_kinds_we_do_not_have():
    p = ck.build_kinds_prompt(25, {"name": "Arithmetic", "description": "math"},
                              ["Barrel shifter", "CRC generator"])
    assert "Arithmetic" in p and "Barrel shifter" in p and "25" in p


def _cat_reply(n):
    return json.dumps({"categories": [{"name": f"cat {i}", "description": f"d{i}"}
                                      for i in range(n)]})


def _kinds_reply(cat, k, clash=None):
    kinds = [{"name": f"{cat} block {i}", "description": f"does {i}; keeps r{i}"}
             for i in range(k)]
    if clash:
        kinds.append({"name": clash, "description": "already have it"})
    return json.dumps({"kinds": kinds})


def test_by_category_collects_kinds_and_drops_exact_repeats(monkeypatch):
    calls = []

    def call(prompt, schema):
        calls.append(prompt)
        if "categories" in schema["properties"]:
            return _cat_reply(3), {"input": 10, "output": 20}, "ok"
        cat = re.search(r'category "([^"]+)"', prompt).group(1)
        return _kinds_reply(cat, 4, clash="Barrel Shifter"), {"input": 10, "output": 20}, "ok"
    rec = ck.by_category(3, 4, known=["Barrel shifter"], call=call, workers=2)
    assert len(calls) == 4                       # 1 category call + 3
    names = [k["name"] for k in rec["candidates"]]
    assert len(names) == 12 and "Barrel Shifter" not in names
    assert all(k["category"].startswith("cat ") for k in rec["candidates"])
    assert rec["dropped_repeats"] == ["Barrel Shifter"] * 3
    assert rec["prompts"]["categories"] and len(rec["replies"]) == 4


def test_finish_drops_judged_repeats_then_draws_by_seed(tmp_path):
    cands = [{"name": f"k{i}", "description": f"d{i}", "category": "c"}
             for i in range(10)]
    rec = tmp_path / "kinds_bycat_r2.json"
    rec.write_text(json.dumps({"run_id": "r2", "candidates": cands}))
    verdicts = tmp_path / "v.json"
    verdicts.write_text(json.dumps({f"k{i}": {"repeat_of": "k0" if i in (3, 4) else None}
                                    for i in range(10)}))
    ck.main(["--out-dir", str(tmp_path), "--finish", str(rec), "--verdicts",
             str(verdicts), "--pick", "5", "--seed", "11"])
    lines = (tmp_path / "constructs_r2_bycat.txt").read_text().splitlines()
    kept = [c for c in cands if c["name"] not in ("k3", "k4")]
    assert lines == [ck.construct_line(k) for k in ck.pick(kept, 5, seed=11)]
    note = json.loads((tmp_path / "kinds_r2_bycat_finish.json").read_text())
    assert note["repeats_dropped"] == ["k3", "k4"] and note["picked"] == 5


def test_finish_refuses_a_candidate_with_no_verdict(tmp_path):
    rec = tmp_path / "kinds_bycat_r3.json"
    rec.write_text(json.dumps({"run_id": "r3", "candidates": [
        {"name": "k0", "description": "d", "category": "c"}]}))
    verdicts = tmp_path / "v.json"
    verdicts.write_text("{}")
    with pytest.raises(SystemExit) as stop:
        ck.main(["--out-dir", str(tmp_path), "--finish", str(rec),
                 "--verdicts", str(verdicts), "--pick", "1"])
    assert "no verdict" in str(stop.value.code)    # our refusal, not argparse
