"""A list of kinds for a big run may use a kind again, but never a kind
held out for a test (16 Nov 2026, Rishab: "alright let's do it", change 6
of 7).

Every draw so far took kinds nothing had used (catalog_kinds.unused_kinds),
and 324 were left after the 15 Nov pilot. A run of thousands of starting
designs has to come back to kinds that already have designs; the size and
design-trait seeds still differ from try to try. What must not happen is a
kind of the generation test (initiator/logs/kinds_unseen_*.txt) turning up
in training designs. So the reusable pool is every checked kind of the
catalog, used or not, minus the 10 adders with no state, minus the kinds
the two judges put in different areas, minus every held-out kind; and
with no held-out list to read, nothing is drawn.
No model is called here."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import catalog_kinds as ck                                       # noqa: E402


def _kind(name, cat="A1", source="oct8_drawn", j2=None):
    return {"id": name, "name": name, "description": f"{name} does things",
            "source": source, "category": cat,
            "judge1": {"primary": cat, "also": []},
            "judge2": {"primary": j2 or cat, "also": []}}


def _catalog(tmp_path, kinds):
    path = tmp_path / "kind_categories.json"
    path.write_text(json.dumps({"kinds": kinds}))
    return path


def _held_out(tmp_path, *names):
    path = tmp_path / "kinds_unseen_test.txt"
    path.write_text("".join(f"{n}: {n} does things\n" for n in names))
    return path


def names(pairs):
    return [k["name"] for _, k in pairs]


def test_a_kind_that_already_has_designs_can_be_drawn_again(tmp_path):
    catalog = _catalog(tmp_path, [_kind("Fresh one"), _kind("Used one"),
                                  _kind("Old one", source="original_32")])
    pool = ck.reusable_kinds(catalog, keep_out=[_held_out(tmp_path, "Other")])
    assert names(pool) == ["Fresh one", "Used one", "Old one"]


def test_a_kind_held_out_for_a_test_never_comes_back(tmp_path):
    catalog = _catalog(tmp_path, [_kind("Fresh one"), _kind("Test one"),
                                  _kind("Test two")])
    held = _held_out(tmp_path, "Test one", "Test two")
    assert names(ck.reusable_kinds(catalog, keep_out=[held])) == ["Fresh one"]


def test_stateless_adders_and_disputed_kinds_stay_out(tmp_path):
    catalog = _catalog(tmp_path, [_kind("Fresh one"),
                                  _kind("Disputed one", j2="A2"),
                                  _kind("Ripple-carry adder")])
    pool = ck.reusable_kinds(catalog, keep_out=[_held_out(tmp_path, "Other")])
    assert names(pool) == ["Fresh one"]


def test_with_no_held_out_list_nothing_is_drawn(tmp_path):
    catalog = _catalog(tmp_path, [_kind("Fresh one")])
    with pytest.raises(ValueError, match="held-out"):
        ck.reusable_kinds(catalog, keep_out=[])
    with pytest.raises(ValueError, match="held-out"):
        ck.draw_reusable(1, seed=7, name="x", out_dir=tmp_path,
                         catalog=catalog, keep_out=[])
    assert not (tmp_path / "x.txt").exists()


def test_a_draw_of_reusable_kinds_is_written_with_its_rule(tmp_path):
    kinds = [_kind(f"K{i}", cat=f"A{i % 3}") for i in range(9)] + \
        [_kind("Test one")]
    catalog = _catalog(tmp_path, kinds)
    held = _held_out(tmp_path, "Test one")
    out = ck.draw_reusable(4, seed=7, name="kinds_reusable_4_even_test",
                           out_dir=tmp_path, catalog=catalog, keep_out=[held])
    lines = out.read_text().splitlines()
    assert len(lines) == 4 and all(": " in l for l in lines)
    assert not any(l.startswith("Test one") for l in lines)
    note = json.loads((tmp_path / "kinds_reusable_4_even_test_note.json")
                      .read_text())
    assert note["seed"] == 7 and note["left"] == 9
    assert note["held_out"] == {"kinds_unseen_test.txt": 1}
    assert "even_draw" in note["rule"]
    assert "may already have designs" in note["what"]
    again = ck.draw_reusable(4, seed=7, name="again", out_dir=tmp_path,
                             catalog=catalog, keep_out=[held])
    assert again.read_text() == out.read_text()


def test_every_reusable_kind_can_be_drawn_at_once(tmp_path):
    kinds = [_kind(f"K{i}", cat=f"A{i % 3}") for i in range(9)] + \
        [_kind("Test one")]
    catalog = _catalog(tmp_path, kinds)
    out = ck.draw_reusable(None, seed=7, name="all", out_dir=tmp_path,
                           catalog=catalog,
                           keep_out=[_held_out(tmp_path, "Test one")])
    assert len(out.read_text().splitlines()) == 9


def test_asking_for_more_than_there_are_says_how_many_there_are(tmp_path):
    catalog = _catalog(tmp_path, [_kind("K1"), _kind("K2")])
    with pytest.raises(ValueError, match="2 kinds can be reused"):
        ck.draw_reusable(5, seed=7, name="x", out_dir=tmp_path,
                         catalog=catalog,
                         keep_out=[_held_out(tmp_path, "Other")])


def test_the_real_pool_keeps_the_generation_tests_kinds_out():
    held = ck._names((ck.LOGS / "kinds_unseen_100_even_2026-10-12.txt")
                     .read_text().splitlines())
    assert len(held) == 100
    pool = set(names(ck.reusable_kinds()))
    assert not pool & held
    # and it does reach kinds that already have designs: the pilot's
    used = ck._names((ck.LOGS / "kinds_unused_150_even_2026-11-14.txt")
                     .read_text().splitlines())
    assert len(pool & used) == 150


def test_a_reusable_list_is_kept_with_the_code():
    """initiator/logs is not in git, so a list written there never reaches
    the cluster; the catalog folder is, like the 9 Oct lists."""
    import inspect
    default = inspect.signature(ck.draw_reusable).parameters["out_dir"].default
    assert default == ck.OUT_DIR and default.name == "catalog"
    tracked = (ROOT / ".gitignore").read_text().splitlines()
    assert "initiator/logs/" in tracked
    assert not any(line.strip().startswith("initiator/catalog")
                   for line in tracked)


def test_the_command_line_draws_them(monkeypatch, capsys):
    seen = {}

    def draw(n, seed, name):
        seen.update(n=n, seed=seed, name=name)
        return Path("logs") / f"{name}.txt"
    monkeypatch.setattr(ck, "draw_reusable", draw)
    ck.main(["--draw-reusable", "0", "--seed", "3",
             "--name", "kinds_reusable_all_even_test"])
    # 0 asks for every reusable kind
    assert seen == {"n": None, "seed": 3,
                    "name": "kinds_reusable_all_even_test"}
    assert "kinds_reusable_all_even_test.txt" in capsys.readouterr().out
    ck.main(["--draw-reusable", "40", "--seed", "3", "--name", "forty"])
    assert seen["n"] == 40


def test_the_command_line_needs_a_name(monkeypatch):
    monkeypatch.setattr(ck, "draw_reusable", lambda *a, **kw: pytest.fail(
        "nothing may be drawn"))
    with pytest.raises(SystemExit) as stop:
        ck.main(["--draw-reusable", "40"])
    assert "--name" in str(stop.value)
