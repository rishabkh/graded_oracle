"""Our earlier design-writing calls as training examples (8 Oct 2026).
The earlier log kept each design's seeds but not the messages sent, so the
question is rebuilt
from the seeds with today's prompt, as Formal Disco rebuilds every example
from its stored arguments. The examples say so in their metadata."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "initiator"))

import rebuild_old_examples as cv                                 # noqa: E402
import build_sft as bs                                            # noqa: E402
from prompts import SYSTEM_PROMPT                                 # noqa: E402

DESIGN = json.dumps({"top_module": "m", "verilog": "module m; endmodule",
                     "invariants": ["a == b"]})
ROW = {"run_id": "2026-09-16_10h00m00s", "attempt": 3,
       "model": "claude-opus-5", "readme_id": "someone/thing",
       "exemplar_id": "A_fifo", "construct": "Ring counter: rotates",
       "style": "STYLE TEXT", "pattern": "PATTERN TEXT",
       "scope": "SCOPE TEXT", "raw_json": DESIGN, "verdict": "NECESSARY",
       "prompt_version": "200cap"}
README = {"someone/thing": "README BODY"}
EXEMPLARS = {"A_fifo": {"verilog": "module fifo; endmodule"}}


def test_a_full_row_becomes_an_initiate_example_with_its_question_rebuilt():
    [ex], skipped = cv.convert([ROW], README, EXEMPLARS)
    assert ex["prompt"] == "initiate" and ex["outcome"] == "success"
    assert ex["response"] == DESIGN
    sys_msg, user = ex["metadata"]["messages"]
    assert sys_msg == {"role": "system", "content": SYSTEM_PROMPT}
    for part in ("README BODY", "Ring counter: rotates", "STYLE TEXT",
                 "PATTERN TEXT", "SCOPE TEXT", "module fifo; endmodule"):
        assert part in user["content"], part
    assert "## Size seed" not in user["content"]   # the old runs had none
    meta = ex["metadata"]
    assert meta["rebuilt"] is True and meta["source_attempt"] == 3
    assert meta["original_prompt_version"] == "200cap"
    assert ex["arguments"]["readme"] == "README BODY"
    assert skipped == {}


def test_a_failed_design_is_kept_as_a_failure():
    [ex], _ = cv.convert([dict(ROW, verdict="DECORATIVE")], README, EXEMPLARS)
    assert ex["outcome"] == "fail"


def test_rows_that_cannot_be_rebuilt_are_counted_not_guessed():
    rows = [dict(ROW, model="openrouter:llm"),          # Qwen, not Opus
            dict(ROW, construct=None),                   # older seeds
            dict(ROW, readme_id="gone/repo"),
            dict(ROW, raw_json=None),
            dict(ROW, exemplar_id="Z_old")]
    out, skipped = cv.convert(rows, README, EXEMPLARS)
    assert out == []
    assert skipped == {"not an Opus row": 1, "seeds missing": 1,
                       "README not found": 1, "no design": 1,
                       "worked example not found": 1}


def test_opus_through_openrouter_counts_as_opus():
    out, _ = cv.convert([dict(ROW, model="openrouter:anthropic/claude-opus-5")],
                        README, EXEMPLARS)
    assert len(out) == 1


def test_later_readme_versions_win():
    old = json.dumps({"repo": "a/b", "readme": "OLD"}) + "\n"
    new = (json.dumps({"repo": "a/b", "readme": "NEW"}) + "\n"
           + json.dumps({"repo": "c/d", "readme": "CD"}) + "\n")
    assert cv.readme_texts([old, new]) == {"a/b": "NEW", "c/d": "CD"}
    assert cv.readme_texts([new, old])["a/b"] == "OLD"


def test_the_builder_takes_them_and_holds_back_held_designs():
    [ex], _ = cv.convert([ROW], README, EXEMPLARS)
    recs, counts = bs.generation_records([ex], {"module m; endmodule"})
    assert counts["initiate"] == {"kept": 1, "skipped": 0}
    assert recs[0]["holdout"] is True


def test_every_committed_readme_version_is_read(tmp_path):
    import subprocess
    run = lambda *a: subprocess.run(a, cwd=tmp_path, check=True,
                                    capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    f = tmp_path / "initiator" / "readmes.jsonl"
    f.parent.mkdir()
    for text in ("V1\n", "V2\n"):
        f.write_text(text)
        run("git", "add", "-A")
        run("git", "commit", "-q", "-m", text.strip())
    f.write_text("NOW\n")
    assert cv.git_versions(f) == ["V1\n", "V2\n", "NOW\n"]
