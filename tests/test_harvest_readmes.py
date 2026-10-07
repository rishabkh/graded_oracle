"""More theme READMEs (Oct 2026: 133 -> about 1,133), collected the way
the first ones were: GitHub hardware topics, more than 50 stars, most
starred first. A target run takes the topics in turn, so the first two
do not fill the whole quota. GitHub is faked here: no network."""
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "initiator"))

import harvest_readmes as hr                                     # noqa: E402

LONG = "A real project README. " * 20


def fake(monkeypatch, tmp_path, topics, readmes=None, existing=()):
    calls = []

    def gh_json(path):
        calls.append(path)
        if path.startswith("search/"):
            topic = re.search(r"topic:([\w-]+)", path).group(1)
            per = int(re.search(r"per_page=(\d+)", path).group(1))
            # "&page=": a bare "page=" would match inside "per_page="
            page = int(re.search(r"&page=(\d+)", path).group(1)) \
                if "&page=" in path else 1
            names = topics.get(topic, [])
            return {"total_count": len(names), "items": [
                {"full_name": n, "stargazers_count": 1000 - i}
                for i, n in enumerate(names)][(page - 1) * per:page * per]}
        name = path[len("repos/"):-len("/readme")]
        text = (readmes or {}).get(name, LONG)
        return {"content": base64.b64encode(text.encode()).decode()}

    out = tmp_path / "readmes.jsonl"
    out.write_text("".join(json.dumps({"repo": r, "readme": "old"}) + "\n"
                           for r in existing))
    monkeypatch.setattr(hr, "gh_json", gh_json)
    monkeypatch.setattr(hr, "OUT", out)
    monkeypatch.setattr(hr, "TOPICS", list(topics))
    return calls, out


def rows(out):
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_a_target_run_takes_topics_in_turn_and_stops_at_the_target(
        monkeypatch, tmp_path):
    calls, out = fake(monkeypatch, tmp_path,
                      {"verilog": ["v1", "v2", "v3"], "fpga": ["f1", "f2"]},
                      existing=["old/one"])
    hr.main(["--target", "4"])
    got = [r["repo"] for r in rows(out)]
    assert got == ["old/one", "v1", "f1", "v2", "f2"]
    assert rows(out)[0]["readme"] == "old"           # existing rows kept


def test_a_repo_in_two_topics_or_already_held_is_taken_once(monkeypatch,
                                                            tmp_path):
    calls, out = fake(monkeypatch, tmp_path,
                      {"verilog": ["a", "b"], "fpga": ["a", "c"]},
                      existing=["b"])
    hr.main(["--target", "10"])
    assert [r["repo"] for r in rows(out)] == ["b", "a", "c"]


def test_a_thin_readme_is_skipped_and_does_not_count(monkeypatch, tmp_path):
    calls, out = fake(monkeypatch, tmp_path, {"verilog": ["thin", "x", "y"]},
                      readmes={"thin": "too short"})
    hr.main(["--target", "2"])
    assert [r["repo"] for r in rows(out)] == ["x", "y"]


def test_new_rows_record_topic_stars_and_date(monkeypatch, tmp_path):
    calls, out = fake(monkeypatch, tmp_path, {"fpga": ["f1"]})
    hr.main(["--target", "1"])
    [row] = rows(out)
    assert row["topic"] == "fpga" and row["stars"] == 1000
    assert row["harvested"] and row["readme"].startswith("A real project")


def test_a_topic_is_read_page_by_page_at_the_star_limit(monkeypatch,
                                                         tmp_path):
    names = [f"r{i}" for i in range(250)]
    calls, out = fake(monkeypatch, tmp_path, {"verilog": names})
    hr.main(["--target", "250"])
    searches = [c for c in calls if c.startswith("search/")]
    assert len(searches) == 3
    assert all("stars:%3E50" in c and "sort=stars" in c and "per_page=100" in c
               for c in searches)
    assert len(rows(out)) == 250


def test_without_a_target_the_search_is_the_old_one(monkeypatch, tmp_path):
    calls, out = fake(monkeypatch, tmp_path, {"verilog": ["v1"]})
    hr.main(["--per-topic", "14"])
    assert calls[0] == ("search/repositories?q=topic:verilog+stars:%3E50"
                        "&sort=stars&per_page=14")
