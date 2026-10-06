"""Turning the corpus into training pairs.

The rule that matters: the training question must be byte-identical in
shape to the evaluation question, or we train the model on one task and
score it on another."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from build_sft import build_pair, build_pairs, repair_pairs

ROW = {
    "id": "g0_000",
    "generation": 0,
    "top_module": "m",
    "verilog": "module m (input wire clk);\n  reg [3:0] c;\nendmodule\n",
    "property": ["c <= 4'd9"],
    "invariants": ["c <= 4'd9", "c != 4'd15"],
}


def test_the_training_question_is_the_evaluation_question():
    from solver_baseline import SOLVER_PROMPT
    pair = build_pair(ROW)
    assert pair["prompt"] == SOLVER_PROMPT.format(verilog=ROW["verilog"])


def test_the_answer_is_the_json_the_grader_parses():
    pair = build_pair(ROW)
    assert json.loads(pair["completion"]) == {"invariants": ROW["invariants"]}


def test_a_row_with_no_invariants_is_refused():
    with pytest.raises(ValueError):
        build_pair({**ROW, "invariants": []})


def test_identical_designs_are_not_trained_on_twice():
    pairs = build_pairs([ROW, dict(ROW, id="g0_001")])
    assert len(pairs) == 1


def test_pairs_carry_the_row_they_came_from():
    pairs = build_pairs([ROW])
    assert pairs[0]["id"] == "g0_000" and pairs[0]["generation"] == 0


def test_repairs_pair_the_failed_attempt_with_the_fix():
    """Run 2 data: what the model first said, why it was wrong, and what
    finally closed the proof."""
    attempts = [
        {"task_id": 7, "attempt": 1, "verdict": "NOT_PROVEN",
         "invariants": ["c <= 4'd15"], "diagnosis": "too weak",
         "cti_leg": "c = 4'd12"},
        {"task_id": 7, "attempt": 2, "verdict": "NECESSARY",
         "invariants": ["c <= 4'd9"]},
    ]
    out = repair_pairs(attempts, {7: ROW})
    assert len(out) == 1
    assert "c <= 4'd15" in out[0]["prompt"]        # the failed attempt
    assert "c = 4'd12" in out[0]["prompt"]         # the counterexample
    assert json.loads(out[0]["completion"]) == {"invariants": ["c <= 4'd9"]}


def test_a_task_that_never_got_fixed_yields_nothing():
    attempts = [{"task_id": 9, "attempt": 1, "verdict": "NOT_PROVEN",
                 "invariants": ["x"], "diagnosis": "d", "cti_leg": "c"}]
    assert repair_pairs(attempts, {9: ROW}) == []


# --- run v3: the same question, with reasoning inside the answer ---------

from solver_baseline import parse_invariants                    # noqa: E402

REAL_CORPUS = (Path(__file__).resolve().parent.parent
               / "extender" / "corpus.jsonl")

# Everything that could break the scorer's reader if the reasoning were
# written as prose around the JSON: quotes, backslashes, sized constants,
# newlines and Verilog concatenation braces.
TRICKY = ('Say "c" can reach 3\'d1 \\ the prover picks {2\'b00, x}.\n'
          'Then {c, 1\'b0} } { wraps and the property breaks.')


def _regen(id_, attempt, accepted, reasoning="c starts at 12.",
           state=None, kind="regen"):
    return {"id": id_, "attempt": attempt, "prompt_kind": kind,
            "cti_reasoning": reasoning, "accepted": accepted,
            "cti_state": state if state is not None
            else [{"signal": "c", "value": "4'd12"}]}


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_reasoning_text_names_the_fake_state_then_the_reasoning():
    from build_sft import reasoning_text
    rec = _regen("g0_000", 1, True, reasoning="c jumps past 9.",
                 state=[{"signal": "c", "value": "4'd12"},
                        {"signal": "en", "value": 1}])
    assert reasoning_text(rec) == ("Fake state: c = 4'd12, en = 1. "
                                   "c jumps past 9.")


def test_reasoning_goes_inside_the_answer_object_first():
    from build_sft import build_reasoning_pairs
    pairs, missing = build_reasoning_pairs(
        [ROW], {"g0_000": _regen("g0_000", 1, True)})
    assert missing == []
    assert pairs[0]["completion"] == json.dumps(
        {"reasoning": "Fake state: c = 4'd12. c starts at 12.",
         "invariants": ROW["invariants"]})


def test_reasoning_pairs_ask_exactly_the_v2_question():
    from build_sft import build_reasoning_pairs
    pairs, _ = build_reasoning_pairs(
        [ROW], {"g0_000": _regen("g0_000", 1, True)})
    assert pairs[0]["prompt"] == build_pair(ROW)["prompt"]


@pytest.mark.skipif(not REAL_CORPUS.exists(), reason="no corpus here")
def test_every_real_row_reads_back_through_the_scorer():
    """The scorer reads from the first { to the last } and takes
    obj["invariants"]. Reasoning full of braces and quotes must not
    change what it reads, for any row we will train on."""
    from build_sft import build_reasoning_pairs, check_parity
    rows = [json.loads(l) for l in REAL_CORPUS.read_text().splitlines()
            if l.strip()]
    by_id = {r["id"]: _regen(r["id"], 1, True, reasoning=TRICKY)
             for r in rows}
    pairs, missing = build_reasoning_pairs(rows, by_id)
    assert missing == [] and len(pairs) == len(build_pairs(rows))
    invs = {r["id"]: r["invariants"] for r in rows}
    for pair in pairs:
        assert parse_invariants(pair["completion"]) == invs[pair["id"]]
        assert json.loads(pair["completion"])["reasoning"].endswith(TRICKY)
    assert check_parity(pairs, rows) == []


def test_the_original_generator_reasoning_is_refused(tmp_path):
    from build_sft import load_reasoning
    path = _write_jsonl(tmp_path / "r.jsonl", [
        _regen("g0_000", 1, True),
        _regen("g0_001", 1, True, kind="original")])
    with pytest.raises(ValueError, match="line 2.*original"):
        load_reasoning(path)


def test_a_line_with_no_prompt_kind_is_refused(tmp_path):
    from build_sft import load_reasoning
    rec = _regen("g0_000", 1, True)
    del rec["prompt_kind"]
    with pytest.raises(ValueError, match="line 1"):
        load_reasoning(_write_jsonl(tmp_path / "r.jsonl", [rec]))


def test_an_accepted_line_with_no_fake_state_is_refused(tmp_path):
    from build_sft import load_reasoning
    path = _write_jsonl(tmp_path / "r.jsonl",
                        [_regen("g0_000", 1, True, state=[])])
    with pytest.raises(ValueError, match="line 1"):
        load_reasoning(path)


def test_the_last_accepted_attempt_wins(tmp_path):
    from build_sft import load_reasoning
    path = _write_jsonl(tmp_path / "r.jsonl", [
        _regen("g0_000", 1, True, reasoning="first"),
        _regen("g0_000", 2, False, reasoning="rejected"),
        _regen("g0_001", 1, False, reasoning="never accepted"),
        _regen("g0_000", 3, True, reasoning="last"),
        _regen("g0_000", 4, False, reasoning="rejected again")])
    got = load_reasoning(path)
    assert set(got) == {"g0_000"}
    assert got["g0_000"]["cti_reasoning"] == "last"


ROW2 = dict(ROW, id="g0_001",
            verilog="module m (input wire clk);\n  reg [2:0] d;\nendmodule\n",
            invariants=["d != 3'd7"])


def _run_main(tmp_path, missing=None, records=None):
    from build_sft import main
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [ROW, ROW2])
    reasoning = _write_jsonl(tmp_path / "r.jsonl",
                             records or [_regen("g0_000", 1, True)])
    out = tmp_path / "v3.jsonl"
    argv = ["--corpus", str(corpus), "--reasoning", str(reasoning),
            "--out", str(out)]
    if missing:
        argv += ["--missing", missing]
    main(argv)
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_missing_reasoning_fails_by_default_and_names_the_rows(tmp_path,
                                                              capsys):
    with pytest.raises(SystemExit) as exc:
        _run_main(tmp_path)
    assert exc.value.code not in (0, None)
    assert "g0_001" in str(exc.value.code) + capsys.readouterr().err
    assert not (tmp_path / "v3.jsonl").exists()


def test_missing_drop_leaves_those_rows_out(tmp_path, capsys):
    out = _run_main(tmp_path, missing="drop")
    assert len(out) == 1
    assert json.loads(out[0]["completion"])["invariants"] == ROW["invariants"]
    said = capsys.readouterr().out
    assert "drop" in said and str(tmp_path / "v3.jsonl") in said


def test_missing_answers_keeps_those_rows_answers_only(tmp_path):
    out = _run_main(tmp_path, missing="answers")
    assert len(out) == 2
    # v2's shape plus v2's own held-back choice
    assert set(out[0]) == {"prompt", "completion", "holdout"}
    assert "reasoning" in json.loads(out[0]["completion"])
    assert {k: out[1][k] for k in ("prompt", "completion")} == {
        "prompt": build_pair(ROW2)["prompt"],
        "completion": build_pair(ROW2)["completion"]}


def test_the_parity_check_catches_reasoning_the_scorer_cannot_read(
        tmp_path):
    """The scorer turns \\' into ' before parsing, so a backslash right
    before a single quote in the reasoning makes the whole answer
    unreadable. Such a file must never be written."""
    from build_sft import build_reasoning_pairs, check_parity
    bad = _regen("g0_000", 1, True, reasoning="a \\' b")
    pairs, _ = build_reasoning_pairs([ROW], {"g0_000": bad})
    assert check_parity(pairs, [ROW]) != []
    with pytest.raises(SystemExit) as exc:
        _run_main(tmp_path, missing="drop", records=[bad])
    assert exc.value.code not in (0, None)
    assert not (tmp_path / "v3.jsonl").exists()


def test_the_parity_check_catches_a_changed_prompt():
    from build_sft import build_reasoning_pairs, check_parity
    pairs, _ = build_reasoning_pairs(
        [ROW], {"g0_000": _regen("g0_000", 1, True)})
    pairs[0]["prompt"] += " "
    assert check_parity(pairs, [ROW]) != []


def test_without_reasoning_the_answers_file_is_unchanged(tmp_path):
    from build_sft import main
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [ROW, ROW2])
    out = tmp_path / "v2.jsonl"
    main(["--corpus", str(corpus), "--out", str(out)])
    assert out.read_text() == "".join(
        json.dumps({"prompt": p["prompt"], "completion": p["completion"]})
        + "\n" for p in build_pairs([ROW, ROW2]))


def test_a_broken_line_names_its_line_number(tmp_path):
    from build_sft import load_reasoning
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(_regen("g0_000", 1, True)) + "\n{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        load_reasoning(path)


def test_a_missing_reasoning_file_stops_with_a_clear_message(tmp_path):
    from build_sft import main
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [ROW])
    with pytest.raises(SystemExit) as exc:
        main(["--corpus", str(corpus),
              "--reasoning", str(tmp_path / "nope.jsonl")])
    assert "nope.jsonl" in str(exc.value.code)


def test_reasoning_pairs_hold_back_exactly_v2s_rows_even_after_a_drop():
    """Dropping rows must not change which rows are held back: each pair
    carries v2's own held-back choice, so v3 trains on v2's training rows
    minus the dropped ones and nothing else."""
    from build_sft import build_reasoning_pairs
    from sft_data import split
    rows = [dict(ROW, id=f"g0_{i:03d}",
                 verilog=ROW["verilog"].replace("module m", f"module m{i}"))
            for i in range(30)]
    held_v2 = {p["prompt"] for p in split(build_pairs(rows))[1]}
    reasoning = {r["id"]: _regen(r["id"], 1, True) for r in rows
                 if r["id"] not in ("g0_004", "g0_017")}
    pairs, absent = build_reasoning_pairs(rows, reasoning, missing="drop")
    assert absent == ["g0_004", "g0_017"] and len(pairs) == 28
    for p in pairs:
        assert p["holdout"] is (p["prompt"] in held_v2)
    tr, ev = split(pairs, holdout=0.1, seed=0)
    assert {p["prompt"] for p in ev} == held_v2 - {
        build_pair(r)["prompt"] for r in rows
        if r["id"] in ("g0_004", "g0_017")}


def test_the_written_file_carries_the_holdout_mark(tmp_path):
    out = _run_main(tmp_path, missing="drop")
    assert all(set(line) == {"prompt", "completion", "holdout"}
               for line in out)


# --- run v4: answers that pass the 1-step rule, and preference pairs ----

def _one_step_log(tmp_path, passing=("g0_000",), needed=None):
    """A fake one_step.jsonl: ROW passes with clause 0 needed, ROW2 fails
    the 1-step rule."""
    needed = needed if needed is not None else {"g0_000": [0]}
    lines = []
    for r in (ROW, ROW2):
        ok = r["id"] in passing
        lines.append({"id": r["id"], "dropped": -1, "tier": "INDUCTIVE"})
        lines.append({"id": r["id"], "dropped": -2,
                      "tier": "INDUCTIVE" if ok else "NOT_INDUCTIVE"})
        for i in range(len(r["invariants"])):
            hit = i in needed.get(r["id"], [])
            lines.append({"id": r["id"], "dropped": i,
                          "clause": r["invariants"][i],
                          "tier": "NOT_INDUCTIVE" if hit else "INDUCTIVE",
                          "trace": f"cex for {r['id']} {i}" if hit else None})
    path = tmp_path / "one_step.jsonl"
    _write_jsonl(path, lines)
    return path


def test_clean_pairs_keep_only_answers_that_pass_one_step(tmp_path):
    from build_sft import build_clean_pairs
    import one_step
    s = one_step.summary(_one_step_log(tmp_path))
    pairs = build_clean_pairs([ROW, ROW2], s["passes_one_step"])
    assert [p["id"] for p in pairs] == ["g0_000"]
    assert pairs[0]["prompt"] == build_pair(ROW)["prompt"]
    assert pairs[0]["completion"] == build_pair(ROW)["completion"]
    assert isinstance(pairs[0]["holdout"], bool)


def test_preference_pairs_drop_one_needed_fact_each(tmp_path):
    from build_sft import build_preference_pairs
    import one_step
    row = dict(ROW, invariants=["c <= 4'd9", "c != 4'd12", "c >= 4'd0"])
    log = _one_step_log(tmp_path, needed={"g0_000": [0, 1]})
    log.write_text(log.read_text())                   # same fixture, wider row
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    lines = [l for l in lines if l["id"] != "g0_000" or l["dropped"] < 0] + [
        {"id": "g0_000", "dropped": i, "clause": row["invariants"][i],
         "tier": "NOT_INDUCTIVE" if i < 2 else "INDUCTIVE",
         "trace": f"cex {i}" if i < 2 else None} for i in range(3)]
    _write_jsonl(log, lines)
    s = one_step.summary(log)
    pairs = build_preference_pairs([row, ROW2], s)
    assert len(pairs) == 2                            # one per needed fact
    for p, i in zip(pairs, (0, 1)):
        assert p["prompt"] == build_pair(row)["prompt"]
        assert p["chosen"] == build_pair(row)["completion"]
        rest = [c for j, c in enumerate(row["invariants"]) if j != i]
        assert parse_invariants(p["rejected"]) == rest
        assert parse_invariants(p["chosen"]) == row["invariants"]
        assert p["dropped"] == row["invariants"][i]
        assert p["counterexample"] == f"cex {i}"
        assert isinstance(p["holdout"], bool)


def test_clean_and_preference_files_are_written(tmp_path):
    from build_sft import main
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [ROW, ROW2])
    log = _one_step_log(tmp_path)
    clean, pref = tmp_path / "v4a.jsonl", tmp_path / "v4b.jsonl"
    main(["--corpus", str(corpus), "--one-step", str(log),
          "--clean", str(clean), "--preference", str(pref)])
    a = [json.loads(l) for l in clean.read_text().splitlines()]
    b = [json.loads(l) for l in pref.read_text().splitlines()]
    assert len(a) == 1 and set(a[0]) == {"prompt", "completion", "holdout"}
    assert len(b) == 1 and {"prompt", "chosen", "rejected", "holdout",
                            "id", "dropped", "counterexample"} <= set(b[0])
