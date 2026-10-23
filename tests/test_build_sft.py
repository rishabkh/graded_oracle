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


def refuted(trace):
    """A fixer log line's "result" for a check whose proof was refused with
    this counterexample, the way fix.py logs it."""
    return {"verdict": "NOT_PROVEN",
            "reason": "with-invariants grade is NOT_INDUCTIVE",
            "with_invariants": {"runs": [{"mode": "cover", "trace_text": "x"},
                                         {"mode": "prove",
                                          "trace_text": trace}]}}


def test_repairs_pair_the_failed_attempt_with_the_fix():
    """Run 2 data: what the model first said, why it was wrong, and what
    finally closed the proof. The counterexample is the one that refuted
    that attempt; until 23 Oct 2026 the prompt showed its cti_leg, the
    name of a check (review finding)."""
    attempts = [
        {"task_id": 7, "attempt": 1, "verdict": "NOT_PROVEN",
         "reason": "with-invariants grade is NOT_INDUCTIVE",
         "invariants": ["c <= 4'd15"], "diagnosis": "too weak",
         "cti_leg": "with_invariants", "result": refuted("c = 4'd12")},
        {"task_id": 7, "attempt": 2, "verdict": "NECESSARY",
         "invariants": ["c <= 4'd9"]},
    ]
    out = repair_pairs(attempts, {7: ROW})
    assert len(out) == 1
    assert "c <= 4'd15" in out[0]["prompt"]        # the failed attempt
    assert "Counterexample state: c = 4'd12" in out[0]["prompt"]
    assert "with_invariants" not in out[0]["prompt"]
    assert json.loads(out[0]["completion"]) == {"invariants": ["c <= 4'd9"]}


def test_only_attempts_the_prover_refused_become_repairs():
    """23 Oct 2026, review finding: "The prover refused them" was said of
    attempts that never reached it."""
    never_checked = [
        {"verdict": v, "invariants": ["c <= $past(c)"], "error": "e"}
        for v in ("FACT_FORMAT", "READS_RESET", "PRE_GATE")] + [
        {"verdict": v} for v in ("REFUSED", "UNPARSEABLE")]
    not_about_facts = [
        {"verdict": "NOT_PROVEN", "invariants": ["c <= 4'd15"],
         "reason": f"with-invariants grade is {g}", "result": {}}
        for g in ("TIMEOUT", "ERROR")]
    refused = {"verdict": "NOT_PROVEN", "invariants": ["c <= 4'd14"],
               "reason": "with-invariants grade is NOT_INDUCTIVE",
               "result": refuted("c = 4'd13")}
    winner = {"verdict": "NECESSARY", "invariants": ["c <= 4'd9"]}
    attempts = [dict(a, task_id=7, attempt=n) for n, a in enumerate(
        never_checked + not_about_facts + [refused, winner], 1)]
    [pair] = repair_pairs(attempts, {7: ROW})
    assert "c <= 4'd14" in pair["prompt"]
    assert "Counterexample state: c = 4'd13" in pair["prompt"]
    from build_sft import refused_by_the_prover
    assert [refused_by_the_prover(a) for a in attempts] == \
        [False] * 7 + [True, False]


def test_a_refused_attempt_with_no_trace_kept_says_so():
    attempts = [{"task_id": 7, "attempt": 1, "verdict": "NOT_PROVEN",
                 "reason": "with-invariants grade is NOT_INDUCTIVE",
                 "invariants": ["c <= 4'd15"], "result": {}},
                {"task_id": 7, "attempt": 2, "verdict": "NECESSARY",
                 "invariants": ["c <= 4'd9"]}]
    [pair] = repair_pairs(attempts, {7: ROW})
    assert "Counterexample state: (none recorded)" in pair["prompt"]


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


# --- v2's rows plus a separate corpus (the catalog run, 8 Oct 2026) ------
# Appending new rows to v2's file would silently re-pick the held-back
# set (measured: 60 of v2's 66 held-back rows would be trained on). The
# combined file marks every row instead: v2's keep v2's marks, new rows
# are all trained on.

import build_sft as bs                                           # noqa: E402


def _rows(prefix, n):
    return [dict(ROW, id=f"{prefix}_{i:03d}",
                 verilog=ROW["verilog"].replace("endmodule",
                                                f"// {prefix} {i}\nendmodule"))
            for i in range(n)]


def test_the_combined_file_keeps_v2s_marks_and_trains_every_new_row():
    v2, new = _rows("g0", 30), _rows("c0", 5)
    pairs = bs.build_combined_pairs(v2, new)
    held = bs.held_back(v2)
    v2_part = pairs[:30]
    assert [p["holdout"] for p in v2_part] == [p["prompt"] in held
                                               for p in bs.build_pairs(v2)]
    assert [p["prompt"] for p in v2_part] == [p["prompt"] for p in
                                              bs.build_pairs(v2)]
    assert all(p["holdout"] is False for p in pairs[30:]) and len(pairs) == 35


def test_a_new_question_equal_to_a_v2_question_is_refused():
    import pytest
    v2 = _rows("g0", 3)
    with pytest.raises(ValueError):
        bs.build_combined_pairs(v2, [dict(v2[1], id="c0_000")])


def test_the_command_writes_a_fully_marked_file(tmp_path):
    v2, new = tmp_path / "v2.jsonl", tmp_path / "new.jsonl"
    v2.write_text("".join(json.dumps(r) + "\n" for r in _rows("g0", 20)))
    new.write_text("".join(json.dumps(r) + "\n" for r in _rows("c0", 4)))
    out = tmp_path / "combined.jsonl"
    bs.main(["--corpus", str(v2), "--extra-corpus", str(new), "--out", str(out)])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 24 and all(set(l) == {"prompt", "completion", "holdout"}
                                    for l in lines)
    assert sum(not l["holdout"] for l in lines[20:]) == 4


# --- the training-file builder for generation too (8 Oct 2026) -------------
# Formal Disco's rules (formal-disco/distill.py build_sft_records): only
# successes; an idea counts only if a design built from it passed; the
# question is the exact chat messages, the answer one assistant turn.

SYS = {"role": "system", "content": "S"}


def _ex(kind, response, outcome, user="U", **args):
    return {"prompt": kind, "arguments": args, "response": response,
            "outcome": outcome,
            "metadata": {"verdict": "x",
                         "messages": [SYS, {"role": "user", "content": user}]}}


def _design(verilog):
    return json.dumps({"top_module": "m", "verilog": verilog,
                       "invariants": ["a == b"]})


def test_only_successes_are_kept():
    recs, counts = bs.generation_records([
        _ex("generate", _design("module m; endmodule"), "success"),
        _ex("generate", _design("module m2; endmodule"), "fail"),
        _ex("repair", "cut off", "error")])
    assert [r["kind"] for r in recs] == ["generate"]
    assert counts["generate"] == {"kept": 1, "skipped": 1}
    assert counts["repair"] == {"kept": 0, "skipped": 1}


def test_an_idea_counts_only_if_a_design_built_from_it_passed():
    recs, _ = bs.generation_records([
        _ex("idea", "GOOD IDEA", "success"),
        _ex("idea", "BAD IDEA", "success"),        # its design failed
        _ex("implement", _design("module a; endmodule"), "success",
            idea="GOOD IDEA"),
        _ex("implement", _design("module b; endmodule"), "fail",
            idea="BAD IDEA")])
    kept = {(r["kind"], r["completion"][0]["content"]) for r in recs}
    assert ("idea", "GOOD IDEA") in kept and ("idea", "BAD IDEA") not in kept


def test_a_record_is_the_exact_messages_and_one_answer_turn():
    [rec], _ = bs.generation_records(
        [_ex("initiate", _design("module m; endmodule"), "success",
             user="the exact question")])
    assert rec["prompt"] == [SYS, {"role": "user",
                                   "content": "the exact question"}]
    assert rec["completion"] == [{"role": "assistant",
                                  "content": _design("module m; endmodule")}]
    assert rec["holdout"] is False


def test_a_held_back_design_stays_held_back_in_every_record_that_shows_it():
    held = "module held; endmodule"
    recs, _ = bs.generation_records([
        _ex("idea", "IDEA H", "success"),
        _ex("implement", _design(held), "success", idea="IDEA H"),
        _ex("repair", _design(held), "success"),
        _ex("generate", _design("module free; endmodule"), "success")],
        held_designs={held})
    by = {r["kind"]: r["holdout"] for r in recs}
    assert by == {"idea": True, "implement": True, "repair": True,
                  "generate": False}


def test_solving_examples_are_not_taken_from_the_call_log():
    # solving pairs come from the corpus, asked exactly as the test asks
    recs, counts = bs.generation_records([_ex("solve", "x", "success")])
    assert recs == [] and "solve" not in counts


def test_the_command_adds_generation_records_after_the_solving_pairs(tmp_path):
    v2, new = tmp_path / "v2.jsonl", tmp_path / "new.jsonl"
    rows = _rows("g0", 20)
    v2.write_text("".join(json.dumps(r) + "\n" for r in rows))
    new.write_text("".join(json.dumps(r) + "\n" for r in _rows("c0", 4)))
    held = bs.held_back(rows)
    held_row = next(r for r in rows if bs.build_pair(r)["prompt"] in held)
    distill = tmp_path / "distill.jsonl"
    distill.write_text("".join(json.dumps(e) + "\n" for e in [
        _ex("generate", _design("module new1; endmodule"), "success"),
        _ex("generate", _design(held_row["verilog"]), "success"),
        _ex("generate", _design("module bad; endmodule"), "fail")]))
    out = tmp_path / "both.jsonl"
    bs.main(["--corpus", str(v2), "--extra-corpus", str(new),
             "--distill", str(distill), "--out", str(out)])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 24 + 2
    gen = lines[24:]
    assert all(isinstance(l["prompt"], list) for l in gen)
    assert [l["holdout"] for l in gen] == [False, True]
    assert all(l["kind"] == "generate" for l in gen)
    assert all(isinstance(l["prompt"], str) for l in lines[:24])


def test_long_records_are_reported_before_training(tmp_path, capsys):
    v2 = tmp_path / "v2.jsonl"
    v2.write_text("".join(json.dumps(r) + "\n" for r in _rows("g0", 5)))
    distill = tmp_path / "distill.jsonl"
    distill.write_text(json.dumps(
        _ex("generate", _design("x" * 80000), "success")) + "\n")
    bs.main(["--corpus", str(v2), "--distill", str(distill),
             "--out", str(tmp_path / "o.jsonl"), "--max-len", "16384"])
    out = capsys.readouterr().out
    assert "1 record(s) over 16384 tokens" in out and "generate" in out


def test_held_designs_are_v2s_held_back_rows_and_skip_rows_without_facts():
    rows = _rows("g0", 20) + [dict(ROW, id="g0_bare", invariants=[],
                                   verilog="module bare; endmodule")]
    held = bs.held_designs(rows)
    prompts = bs.held_back(rows)
    assert held == {r["verilog"] for r in rows[:20]
                    if bs.build_pair(r)["prompt"] in prompts}
    assert held



# --- review fixes (8 Oct 2026) ---------------------------------------------

def test_generation_records_cannot_be_combined_with_reasoning_or_one_step(
        tmp_path):
    import pytest
    for extra in (["--reasoning", "r.jsonl"], ["--one-step", "o.jsonl"]):
        with pytest.raises(SystemExit) as e:
            bs.main(["--distill", "d.jsonl", "--out", str(tmp_path / "o"),
                     *extra])
        assert "cannot be combined" in str(e.value)


def test_a_held_back_design_in_the_question_holds_the_record_back():
    held = "module held;\n  reg a;\nendmodule"
    # a repair question shows the broken design as pretty JSON: escaped
    question = json.dumps({"verilog": held}, indent=2)
    ex = _ex("repair", _design("module fixed; endmodule"), "success",
             user=f"Your planted triple was checked and rejected.\n{question}")
    [rec], _ = bs.generation_records([ex], {held})
    assert rec["holdout"] is True


def test_an_extend_record_is_held_back_by_the_design_it_made():
    held = "module child; endmodule"
    ex = _ex("extend", json.dumps({"patch": "+ x", "invariants": ["a"]}),
             "success")
    ex["metadata"]["design"] = held
    [rec], _ = bs.generation_records([ex], {held})
    assert rec["holdout"] is True


def test_an_idea_answered_as_json_is_matched_to_its_design():
    recs, _ = bs.generation_records([
        _ex("idea", json.dumps({"idea": "GOOD IDEA"}), "success"),
        _ex("implement", _design("module a; endmodule"), "success",
            idea="GOOD IDEA")])
    assert [r["kind"] for r in recs] == ["idea", "implement"]


# --- designs the automatic engine cannot prove alone count for more ----------
# (21 Oct 2026, Rishab approved.) Rows whose metrics say the engine ran out
# of time on the property alone are written K times; held-back rows never.

from sft_data import split                                       # noqa: E402


def _engine(rows, which):
    """Copies of `rows` with metrics["engine_alone"] set by index."""
    return [dict(r, metrics={"engine_alone": which[i]}) if i in which
            else r for i, r in enumerate(rows)]


def _hard_rows(n=20):
    """n rows: three trained and one held-back row timed out alone, two
    proved alone, the rest built before the metric existed."""
    rows = _rows("g0", n)
    held = bs.held_back(rows)
    is_held = [bs.build_pair(r)["prompt"] in held for r in rows]
    trained = [i for i, h in enumerate(is_held) if not h]
    held_i = [i for i, h in enumerate(is_held) if h]
    which = {i: "timed_out" for i in trained[:3]}
    which[held_i[0]] = "timed_out"
    which.update({trained[3]: "proved", trained[4]: "proved"})
    return _engine(rows, which), trained[:3], held_i[0]


def _counts(lines):
    out = {}
    for l in lines:
        out[l["prompt"]] = out.get(l["prompt"], 0) + 1
    return out


def test_engine_hard_pairs_repeat_right_after_themselves_held_back_never():
    pairs = [{"prompt": q, "completion": "c", "holdout": h}
             for q, h in (("a", False), ("b", False), ("c", True))]
    out, repeated, held = bs.repeat_engine_hard(pairs, {"b", "c"}, 3)
    assert [p["prompt"] for p in out] == ["a", "b", "b", "b", "c"]
    assert (repeated, held) == (1, 1)
    assert out[2] is not out[1] and out[2] == out[1]


def test_one_copy_changes_nothing():
    pairs = [{"prompt": "b", "completion": "c", "holdout": False}]
    assert bs.repeat_engine_hard(pairs, {"b"}, 1)[0] == pairs


def test_repeating_unmarked_pairs_is_refused():
    """An unmarked file lets the trainer pick its own held-back rows, so a
    copy could land on the other side."""
    with pytest.raises(ValueError, match="mark every pair"):
        bs.repeat_engine_hard([{"prompt": "b", "completion": "c"}], {"b"}, 2)


def test_only_rows_that_timed_out_alone_are_engine_hard():
    rows = _engine(_rows("c0", 5), {0: "timed_out", 1: "proved", 2: "error",
                                    3: "not_tried"})
    assert bs.engine_hard_questions(rows) == {bs.build_pair(rows[0])["prompt"]}
    assert bs.engine_alone(rows[4]) is None


def test_the_answers_file_with_repeats_marks_every_line(tmp_path, capsys):
    rows, hard, held_hard = _hard_rows()
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", rows)
    out = tmp_path / "out.jsonl"
    bs.main(["--corpus", str(corpus), "--out", str(out),
             "--engine-hard-copies", "3"])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    held = bs.held_back(rows)
    assert all(set(l) == {"prompt", "completion", "holdout"} for l in lines)
    assert all(l["holdout"] is (l["prompt"] in held) for l in lines)
    counts = _counts(lines)
    prompt = lambda i: bs.build_pair(rows[i])["prompt"]           # noqa: E731
    assert [counts[prompt(i)] for i in hard] == [3, 3, 3]
    assert counts[prompt(held_hard)] == 1                # held back: once
    assert sum(counts.values()) == len(rows) + 2 * 3     # everyone else once
    # the trainer keeps exactly v2's held-back rows, no copy crosses over
    train, ev = split(lines)
    assert {l["prompt"] for l in ev} == held
    assert not {l["prompt"] for l in train} & held
    said = capsys.readouterr().out
    assert "3 repeated, each written 3 times (+6 lines)" in said
    assert "1 held back, never repeated" in said
    assert "proved 2" in said and "timed_out 4" in said


def test_with_one_copy_the_answers_file_is_byte_for_byte_unchanged(tmp_path):
    rows, _, _ = _hard_rows()
    plain = _write_jsonl(tmp_path / "plain.jsonl",
                         [{k: v for k, v in r.items() if k != "metrics"}
                          for r in rows])
    marked = _write_jsonl(tmp_path / "marked.jsonl", rows)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    bs.main(["--corpus", str(plain), "--out", str(a)])
    bs.main(["--corpus", str(marked), "--out", str(b)])
    assert a.read_text() == b.read_text()
    assert "holdout" not in a.read_text()


def test_the_combined_file_repeats_engine_hard_rows_of_both_corpora(
        tmp_path, capsys):
    rows, hard, held_hard = _hard_rows()
    new = _engine(_rows("c0", 4), {0: "timed_out", 1: "proved"})
    v2 = _write_jsonl(tmp_path / "v2.jsonl", rows)
    extra = _write_jsonl(tmp_path / "new.jsonl", new)
    out = tmp_path / "combined.jsonl"
    bs.main(["--corpus", str(v2), "--extra-corpus", str(extra),
             "--out", str(out), "--engine-hard-copies", "2"])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    counts = _counts(lines)
    assert counts[bs.build_pair(new[0])["prompt"]] == 2
    assert counts[bs.build_pair(new[1])["prompt"]] == 1
    assert [counts[bs.build_pair(rows[i])["prompt"]] for i in hard] == [2] * 3
    assert counts[bs.build_pair(rows[held_hard])["prompt"]] == 1
    assert len(lines) == 24 + 4
    train, ev = split(lines)
    assert {l["prompt"] for l in ev} == bs.held_back(rows)
    assert "4 repeated, each written 2 times (+4 lines)" in \
        capsys.readouterr().out


def test_the_reasoning_file_repeats_engine_hard_rows_too(tmp_path):
    rows = _engine([ROW, ROW2], {1: "timed_out"})
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", rows)
    reasoning = _write_jsonl(tmp_path / "r.jsonl",
                             [_regen("g0_000", 1, True),
                              _regen("g0_001", 1, True)])
    out = tmp_path / "v3.jsonl"
    bs.main(["--corpus", str(corpus), "--reasoning", str(reasoning),
             "--out", str(out), "--engine-hard-copies", "2"])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    expect = 1 if bs.build_pair(ROW2)["prompt"] in bs.held_back(rows) else 2
    assert _counts(lines)[bs.build_pair(ROW2)["prompt"]] == expect
    assert _counts(lines)[bs.build_pair(ROW)["prompt"]] == 1


def test_the_clean_file_repeats_engine_hard_rows_too(tmp_path):
    rows = _engine([ROW, ROW2], {0: "timed_out"})
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", rows)
    clean = tmp_path / "v4a.jsonl"
    bs.main(["--corpus", str(corpus), "--one-step",
             str(_one_step_log(tmp_path)), "--clean", str(clean),
             "--engine-hard-copies", "2"])
    lines = [json.loads(l) for l in clean.read_text().splitlines()]
    expect = 1 if bs.build_pair(ROW)["prompt"] in bs.held_back(rows) else 2
    assert len(lines) == expect


def test_fewer_than_one_copy_is_refused(tmp_path):
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [ROW])
    with pytest.raises(SystemExit) as e:
        bs.main(["--corpus", str(corpus), "--out", str(tmp_path / "o"),
                 "--engine-hard-copies", "0"])
    assert "1 or more" in str(e.value)
    assert not (tmp_path / "o").exists()


def test_copies_without_a_solving_file_are_refused(tmp_path):
    """Preference pairs are not solving pairs and never repeat, so the
    option alone with them would quietly do nothing."""
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [ROW, ROW2])
    pref = tmp_path / "v4b.jsonl"
    with pytest.raises(SystemExit) as e:
        bs.main(["--corpus", str(corpus), "--one-step",
                 str(_one_step_log(tmp_path)), "--preference", str(pref),
                 "--engine-hard-copies", "2"])
    assert "needs a solving file" in str(e.value)
    assert not pref.exists()


def test_a_log_from_before_the_clock_edge_fix_is_flagged(tmp_path, capsys):
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [ROW, ROW2])
    bs.main(["--corpus", str(corpus), "--one-step",
             str(_one_step_log(tmp_path)),
             "--clean", str(tmp_path / "v4a.jsonl")])
    assert "before the clock-edge fix" in capsys.readouterr().out


# --- extend records and held-back rows (10 Oct review) ----------------------
# An extend answer is a patch and a fact list, never a whole design, so a
# grown design that IS a held-back row, or that grew FROM one, has to be
# caught by its metadata: child_verilog (the whole grown design, or null)
# and parent_ids (the corpus ids it grew from), as distill_log.record keeps
# them.

import distill_log                                               # noqa: E402

PATCH = json.dumps({"patch": "@@ -1 +1 @@\n-x\n+y",
                    "invariants": ["c <= 4'd9", "c != 4'd15"]})


def _extend(tmp_path, child=None, parents=(), question="grow this design"):
    log = tmp_path / "distill.jsonl"
    distill_log.record(log, "extend", {"move": "double"}, PATCH, "NECESSARY",
                       [SYS, {"role": "user", "content": question}],
                       child_verilog=child, parent_ids=list(parents),
                       extension_id="2026-10-21_x")
    return distill_log.read(log)[-1]


def _held_row():
    rows = _rows("g0", 20)
    held = bs.held_back(rows)
    return rows, next(r for r in rows if bs.build_pair(r)["prompt"] in held)


def test_an_extend_record_whose_grown_design_is_held_back_is_held_back(
        tmp_path):
    rows, h = _held_row()
    respaced = "\n\n".join("    " + l for l in h["verilog"].splitlines())
    assert respaced != h["verilog"]
    [rec], _ = bs.generation_records([_extend(tmp_path, child=respaced)],
                                     bs.held_designs(rows), bs.held_ids(rows))
    assert rec["holdout"] is True


def test_an_extend_record_grown_from_a_held_back_row_is_held_back(tmp_path):
    rows, h = _held_row()
    ex = _extend(tmp_path, child="module grown; endmodule",
                 parents=["c0_001", h["id"]])
    [rec], _ = bs.generation_records([ex], bs.held_designs(rows),
                                     bs.held_ids(rows))
    assert rec["holdout"] is True


def test_a_held_back_design_in_the_question_holds_it_back_in_any_spacing(
        tmp_path):
    rows, h = _held_row()
    question = "Grow this design:\n" + " ".join(h["verilog"].split())
    [rec], _ = bs.generation_records([_extend(tmp_path, question=question)],
                                     bs.held_designs(rows), bs.held_ids(rows))
    assert rec["holdout"] is True


def test_an_extend_record_from_trained_rows_is_trained(tmp_path):
    rows, h = _held_row()
    trained = next(r for r in rows if r is not h
                   and r["id"] not in bs.held_ids(rows))
    exs = [_extend(tmp_path, child="module grown; endmodule",
                   parents=[trained["id"]],
                   question="Grow:\n" + trained["verilog"]),
           _extend(tmp_path, child=None, parents=[])]
    recs, counts = bs.generation_records(exs, bs.held_designs(rows),
                                         bs.held_ids(rows))
    assert [r["holdout"] for r in recs] == [False, False]
    assert counts["extend"] == {"kept": 2, "skipped": 0}


def test_held_ids_are_the_ids_of_v2s_held_back_rows():
    rows = _rows("g0", 20)
    assert bs.held_ids(rows) == {r["id"] for r in rows
                                 if r["verilog"] in bs.held_designs(rows)}
    assert bs.held_ids(rows)


def test_a_blank_held_design_holds_nothing_back(tmp_path):
    [rec], _ = bs.generation_records([_extend(tmp_path, child="module a;")],
                                     {"", "  \n"})
    assert rec["holdout"] is False


def test_the_command_holds_back_extend_records_by_parent_and_design(
        tmp_path, capsys):
    rows = _rows("g0", 20)
    held = bs.held_back(rows)
    held_rows = [r for r in rows if bs.build_pair(r)["prompt"] in held]
    v2 = _write_jsonl(tmp_path / "v2.jsonl", rows)
    log = tmp_path / "distill.jsonl"
    for child, parents in ((None, [held_rows[0]["id"]]),
                           (held_rows[1]["verilog"], ["c0_000"]),
                           ("module free; endmodule", ["c0_000"])):
        distill_log.record(log, "extend", {}, PATCH, "NECESSARY",
                           [SYS, {"role": "user", "content": "grow it"}],
                           child_verilog=child, parent_ids=parents,
                           extension_id="e")
    out = tmp_path / "both.jsonl"
    bs.main(["--corpus", str(v2), "--distill", str(log), "--out", str(out)])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["holdout"] for l in lines if l.get("kind") == "extend"] == \
        [True, True, False]
    assert "2 generation record(s) held back" in capsys.readouterr().out


# --- only facts the benchmark's checker can read (22 Oct 2026) -------------
# Review findings F2, F3 and F9: EBMC 6.0, the benchmark's checker, cannot
# read a fact with a system function, the compact `forall` form or a dotted
# name (fact_format.py). Every answer written is checked: an extend answer
# gets the plain list the checker proved, any other unreadable answer is
# left out and named, never rewritten, and leaving one out never moves
# another row across the held-back line.

from fact_format import unreadable_facts                         # noqa: E402

ONEHOT = ["$onehot(c)", "c <= 4'd9"]
COUNTONES = ["occ == $countones(c)", "c != 4'd15"]
COMPACT = ["forall k in 0..1: !v[k] || (tag[k] != 4'd15)", "c <= 4'd9"]
EXPANDED = ["!v[0] || (tag[0] != 4'd15)", "!v[1] || (tag[1] != 4'd15)",
            "c <= 4'd9"]


def _bad(rows, i, facts):
    """`rows` with row i's answer holding facts EBMC cannot read."""
    return [dict(r, invariants=list(facts)) if k == i else r
            for k, r in enumerate(rows)]


def _held_and_trained(rows):
    """Indexes of v2's held-back rows and of its trained rows."""
    held = bs.held_back(rows)
    is_held = [bs.build_pair(r)["prompt"] in held for r in rows]
    return ([i for i, h in enumerate(is_held) if h],
            [i for i, h in enumerate(is_held) if not h])


def _facts_of(line):
    comp = line["completion"]
    if isinstance(comp, list):
        comp = comp[0]["content"]
    return bs.answer_facts(comp) or []


def _lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


@pytest.mark.parametrize("text, facts", [
    (json.dumps({"invariants": ["a == b"]}), ["a == b"]),
    ('```json\n{"invariants": ["$onehot(s)"]}\n```', ["$onehot(s)"]),
    ('Here: {"reasoning": "{x}", "invariants": ["a"]} done', ["a"]),
    (json.dumps([{"invariants": ["a"]}]), ["a"]),
    ("an idea in plain words", None),
    (json.dumps({"note": "why", "patch": "+x"}), None),
    (None, None),
])
def test_answer_facts_reads_an_answer_the_way_the_scorer_does(text, facts):
    assert bs.answer_facts(text) == facts


def test_unreadable_answers_are_left_out_named_and_never_rewritten():
    pairs = [{"id": "a", "completion": json.dumps({"invariants": EXPANDED})},
             {"id": "b", "completion": json.dumps({"invariants": ONEHOT})},
             {"id": "c", "completion": json.dumps({"invariants": COMPACT})}]
    kept, left_out = bs.leave_out_unreadable(pairs)
    assert kept == [pairs[0]] and kept[0] is pairs[0]
    assert left_out == [
        ("b", [("$onehot(c)", ["system function $onehot"])]),
        ("c", [(COMPACT[0], ["compact forall form"])])]


def test_a_left_out_answer_is_never_silent(capsys):
    bs.say_left_out("solving answers", [])
    assert "solving answers: EBMC can read every fact" in \
        capsys.readouterr().out
    onehot = [("$onehot(st)", ["system function $onehot"])]
    bs.say_left_out("preference pairs", [
        ("g0_051", onehot), ("g0_051", onehot),
        ("c0_144", [("occ == $countones(valid)",
                     ["system function $countones"])])])
    said = capsys.readouterr().out
    assert ("left out 3 preference pairs with facts EBMC cannot read (not "
            "rewritten): g0_051, c0_144") in said
    assert said.count("g0_051: $onehot(st)  (system function $onehot)") == 1
    assert "c0_144: occ == $countones(valid)  (system function " \
        "$countones)" in said


def test_the_answers_file_leaves_out_unreadable_answers_and_keeps_v2s_choice(
        tmp_path, capsys):
    rows = _rows("g0", 30)
    h, t = _held_and_trained(rows)
    rows = _bad(_bad(rows, h[0], ONEHOT), t[0], COUNTONES)
    held = bs.held_back(rows)                  # v2's choice on every row
    gone = {bs.build_pair(rows[i])["prompt"] for i in (h[0], t[0])}
    out = tmp_path / "out.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "c.jsonl", rows)),
             "--out", str(out)])
    lines = _lines(out)
    assert len(lines) == 28
    assert not any(unreadable_facts(_facts_of(l)) for l in lines)
    assert not gone & {l["prompt"] for l in lines}
    said = capsys.readouterr().out
    assert "left out 2 solving answers" in said
    assert rows[h[0]]["id"] in said and rows[t[0]]["id"] in said
    # every line carries v2's own held-back choice, and the trainer keeps
    # it: a fresh shuffle of the shorter list would pick other rows
    assert all(set(l) == {"prompt", "completion", "holdout"} for l in lines)
    assert all(l["holdout"] is (l["prompt"] in held) for l in lines)
    train, ev = split(lines)
    assert {l["prompt"] for l in ev} == held - gone
    assert {l["prompt"] for l in train} == \
        {bs.build_pair(r)["prompt"] for r in rows} - held - gone


def test_with_every_fact_readable_the_answers_file_is_as_before(tmp_path,
                                                               capsys):
    out = tmp_path / "v2.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "c.jsonl", [ROW, ROW2])),
             "--out", str(out)])
    assert out.read_text() == "".join(
        json.dumps({"prompt": p["prompt"], "completion": p["completion"]})
        + "\n" for p in build_pairs([ROW, ROW2]))
    assert "solving answers: EBMC can read every fact" in \
        capsys.readouterr().out


def test_the_combined_file_leaves_out_unreadable_answers_of_both_corpora(
        tmp_path, capsys):
    rows = _rows("g0", 20)
    h, _ = _held_and_trained(rows)
    rows = _bad(rows, h[0], ONEHOT)
    new = _bad(_rows("c0", 4), 1, COUNTONES)
    held = bs.held_back(rows)
    out = tmp_path / "combined.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "v2.jsonl", rows)),
             "--extra-corpus", str(_write_jsonl(tmp_path / "n.jsonl", new)),
             "--out", str(out)])
    lines = _lines(out)
    gone = {bs.build_pair(rows[h[0]])["prompt"],
            bs.build_pair(new[1])["prompt"]}
    assert len(lines) == 24 - 2 and not gone & {l["prompt"] for l in lines}
    assert not any(unreadable_facts(_facts_of(l)) for l in lines)
    assert all(l["holdout"] is (l["prompt"] in held) for l in lines)
    said = capsys.readouterr().out
    assert "left out 2 solving answers" in said
    assert rows[h[0]]["id"] in said and "c0_001" in said


def test_the_reasoning_file_leaves_out_unreadable_answers(tmp_path, capsys):
    rows = _rows("g0", 20)
    h, t = _held_and_trained(rows)
    rows = _bad(_bad(rows, h[0], ONEHOT), t[0], ONEHOT)
    held = bs.held_back(rows)
    reasoning = _write_jsonl(tmp_path / "r.jsonl",
                             [_regen(r["id"], 1, True) for r in rows])
    out = tmp_path / "v3.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "c.jsonl", rows)),
             "--reasoning", str(reasoning), "--out", str(out)])
    lines = _lines(out)
    assert len(lines) == 18
    assert not any(unreadable_facts(_facts_of(l)) for l in lines)
    assert all(l["holdout"] is (l["prompt"] in held) for l in lines)
    assert sum(l["holdout"] for l in lines) == len(held) - 1
    said = capsys.readouterr().out
    assert "left out 2 solving answers" in said and "(18 lines)" in said


def _all_pass_log(tmp_path, rows):
    """A fixed-check 1-step log: every answer passes, its first fact
    needed."""
    lines = []
    for r in rows:
        lines += [{"id": r["id"], "dropped": d, "tier": "INDUCTIVE",
                   "clock_edge_fix": True} for d in (-1, -2)]
        lines += [{"id": r["id"], "dropped": i, "clause": c,
                   "clock_edge_fix": True,
                   "tier": "NOT_INDUCTIVE" if i == 0 else "INDUCTIVE",
                   "trace": "cex" if i == 0 else None}
                  for i, c in enumerate(r["invariants"])]
    return _write_jsonl(tmp_path / "one_step_fixed.jsonl", lines)


def test_the_clean_and_preference_files_leave_out_unreadable_answers(
        tmp_path, capsys):
    rows = _rows("g0", 20)
    h, t = _held_and_trained(rows)
    rows = _bad(_bad(rows, h[0], ONEHOT), t[0], COUNTONES)
    held = bs.held_back(rows)
    clean, pref = tmp_path / "v4a.jsonl", tmp_path / "v4b.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "c.jsonl", rows)),
             "--one-step", str(_all_pass_log(tmp_path, rows)),
             "--clean", str(clean), "--preference", str(pref)])
    a, b = _lines(clean), _lines(pref)
    assert len(a) == 18 and len(b) == 18
    assert not {rows[h[0]]["id"], rows[t[0]]["id"]} & {p["id"] for p in b}
    assert not any(unreadable_facts(_facts_of(l)) for l in a)
    assert not any(unreadable_facts(bs.answer_facts(p["chosen"]))
                   for p in b)
    assert all(l["holdout"] is (l["prompt"] in held) for l in a + b)
    said = capsys.readouterr().out
    assert "left out 2 clean answers" in said
    assert "left out 2 preference pairs" in said
    assert "before the clock-edge fix" not in said


def _repair_setup(tmp_path, monkeypatch, bad_task=None):
    """20 fixed tasks, each a failed attempt then a winner; the winner of
    `bad_task` holds a fact EBMC cannot read."""
    rows = [dict(r, source_task=k) for k, r in enumerate(_rows("g0", 20))]
    attempts = []
    for k in range(20):
        attempts += [
            {"task_id": k, "attempt": 1, "verdict": "NOT_PROVEN",
             "invariants": ["c <= 4'd15"], "diagnosis": "too weak",
             "cti_leg": "c = 4'd12"},
            {"task_id": k, "attempt": 2, "verdict": "NECESSARY",
             "invariants": ["c <= 4'd9"] + (["!b || (a == $past(b))"]
                                           if k == bad_task else [])}]
    monkeypatch.setattr(bs, "FIXER_LOG",
                        _write_jsonl(tmp_path / "fixer.jsonl", attempts))
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", rows)
    return bs.repair_pairs(attempts, {r["source_task"]: r for r in rows}), \
        corpus


def test_the_repairs_file_leaves_out_unreadable_answers_and_keeps_the_choice(
        tmp_path, monkeypatch, capsys):
    full, _ = _repair_setup(tmp_path, monkeypatch)
    held = {p["prompt"] for p in split(full)[1]}   # the trainer's choice
    bad = next(p["task_id"] for p in full if p["prompt"] not in held)
    _, corpus = _repair_setup(tmp_path, monkeypatch, bad_task=bad)
    out = tmp_path / "repairs.jsonl"
    bs.main(["--corpus", str(corpus), "--repairs", str(out)])
    lines = _lines(out)
    assert len(lines) == 19 and bad not in {l["task_id"] for l in lines}
    assert not any(unreadable_facts(_facts_of(l)) for l in lines)
    assert all(l["holdout"] is (l["prompt"] in held) for l in lines)
    assert {l["prompt"] for l in split(lines)[1]} == held
    said = capsys.readouterr().out
    assert f"left out 1 repair answers with facts EBMC cannot read (not " \
        f"rewritten): {bad}" in said


def test_with_every_fact_readable_the_repairs_file_is_as_before(
        tmp_path, monkeypatch):
    full, corpus = _repair_setup(tmp_path, monkeypatch)
    out = tmp_path / "repairs.jsonl"
    bs.main(["--corpus", str(corpus), "--repairs", str(out)])
    assert out.read_text() == "".join(json.dumps(o) + "\n" for o in full)


# extend answers: the plain list the checker proved (review finding F2)

def _extend_answer(facts):
    return json.dumps({"claim": "two entries", "patch": "@@ -1 +1 @@\n-x\n+y",
                       "invariants": list(facts), "dispositions": []})


def _saved_extend(tmp_path, answer, expanded=None, ext_id="2026-10-22_t001"):
    """A saved extension example as the extender writes it; `expanded`
    is its metadata "invariants_expanded"."""
    log = tmp_path / f"distill_{ext_id}.jsonl"
    meta = {} if expanded is None else {"invariants_expanded": expanded}
    distill_log.record(log, "extend", {"move": "STORAGE"}, answer,
                       "NECESSARY", [SYS, {"role": "user",
                                           "content": "grow it"}],
                       child_verilog="module grown; endmodule",
                       parent_ids=["c0_000"], extension_id=ext_id, **meta)
    return distill_log.read(log)[-1]


def test_an_extend_answer_gets_the_plain_list_the_checker_proved(tmp_path):
    answer = _extend_answer(COMPACT)
    report = {}
    [rec], counts = bs.generation_records(
        [_saved_extend(tmp_path, answer, EXPANDED)], fact_report=report)
    got = json.loads(rec["completion"][0]["content"])
    assert got == dict(json.loads(answer), invariants=EXPANDED)
    assert list(got) == list(json.loads(answer))    # every field, in place
    assert report == {"rebuilt": ["2026-10-22_t001"], "left_out": [],
                      "lists": []}
    assert counts["extend"] == {"kept": 1, "skipped": 0}


def test_compact_facts_with_no_plain_list_leave_the_answer_out(tmp_path):
    report = {}
    recs, counts = bs.generation_records(
        [_saved_extend(tmp_path, _extend_answer(COMPACT))],
        fact_report=report)
    assert recs == [] and counts["extend"] == {"kept": 0, "skipped": 1}
    assert report == {"rebuilt": [], "left_out": [
        ("extend 2026-10-22_t001", [(COMPACT[0], ["compact forall form"])])],
        "lists": []}


def test_an_extend_answer_already_plain_stays_exactly_as_written(tmp_path):
    answer = json.dumps(json.loads(_extend_answer(EXPANDED)), indent=2)
    report = {}
    recs, _ = bs.generation_records(
        [_saved_extend(tmp_path, answer, EXPANDED),
         _saved_extend(tmp_path, answer, ext_id="e2")], fact_report=report)
    assert [r["completion"][0]["content"] for r in recs] == [answer, answer]
    assert report == {"rebuilt": [], "left_out": [], "lists": []}


def test_a_proved_list_ebmc_still_cannot_read_is_left_out(tmp_path):
    report = {}
    recs, counts = bs.generation_records(
        [_saved_extend(tmp_path, _extend_answer(COMPACT),
                       EXPANDED + ["a == $past(b)"])], fact_report=report)
    assert recs == [] and counts["extend"]["skipped"] == 1
    assert report["left_out"][0][1] == [("a == $past(b)",
                                         ["system function $past"])]


def test_a_distractor_answer_has_no_facts_and_is_kept_as_written(tmp_path):
    answer = json.dumps({"note": "irrelevant logic", "patch": "+x"})
    report = {}
    [rec], _ = bs.generation_records(
        [_saved_extend(tmp_path, answer, EXPANDED)], fact_report=report)
    assert rec["completion"][0]["content"] == answer
    assert report == {"rebuilt": [], "left_out": [], "lists": []}


# the kept/replaced list of an extend answer (23 Oct 2026, Rishab chose to
# leave out an answer whose list names a compact fact)

def _kept(fact):
    return {"clause": fact, "status": "kept", "replaced_by": [],
            "reason": "unchanged"}


def _listed_answer(facts, dispositions):
    return json.dumps({"claim": "two entries", "patch": "@@ -1 +1 @@\n-x\n+y",
                       "invariants": list(facts),
                       "dispositions": dispositions})


def test_an_answer_that_kept_a_compact_parent_fact_is_left_out(tmp_path):
    """The parent's compact fact was kept; spelled out, the answer's list
    no longer holds it, and the list still writes the compact form."""
    answer = _listed_answer(COMPACT, [_kept(COMPACT[0]), _kept("c <= 4'd9")])
    report = {}
    recs, counts = bs.generation_records(
        [_saved_extend(tmp_path, answer, EXPANDED)], fact_report=report)
    assert recs == [] and counts["extend"] == {"kept": 0, "skipped": 1}
    assert report == {"rebuilt": [], "lists": ["2026-10-22_t001"],
                      "left_out": [("extend 2026-10-22_t001", [
                          (COMPACT[0], ["compact forall form, in the "
                                        "kept/replaced list"])])]}


def test_an_answer_that_replaced_a_fact_by_a_compact_one_is_left_out(
        tmp_path):
    answer = _listed_answer(COMPACT, [
        {"clause": "c <= 4'd15", "status": "superseded",
         "replaced_by": COMPACT[0], "reason": "tighter"}])
    report = {}
    recs, _ = bs.generation_records(
        [_saved_extend(tmp_path, answer, EXPANDED)], fact_report=report)
    assert recs == [] and report["lists"] == ["2026-10-22_t001"]
    assert report["left_out"][0][1] == [
        (COMPACT[0], ["compact forall form, in the kept/replaced list"])]


def test_a_list_of_plain_facts_keeps_the_rebuilt_answer(tmp_path):
    """The answer's own new fact is compact, but its list names only
    plain facts, which are the same once the compact one is spelled
    out."""
    disps = [_kept("c <= 4'd9"),
             {"clause": "c <= 4'd15", "status": "superseded",
              "replaced_by": ["c <= 4'd9"], "reason": "tighter"}]
    answer = _listed_answer(COMPACT, disps)
    report = {}
    [rec], _ = bs.generation_records(
        [_saved_extend(tmp_path, answer, EXPANDED)], fact_report=report)
    got = json.loads(rec["completion"][0]["content"])
    assert got["invariants"] == EXPANDED and got["dispositions"] == disps
    assert report == {"rebuilt": ["2026-10-22_t001"], "left_out": [],
                      "lists": []}


@pytest.mark.parametrize("answer, facts", [
    (_listed_answer([], [{"clause": "a", "status": "superseded",
                          "replaced_by": ["b", "c"], "reason": "r"},
                         _kept("d")]), ["a", "b", "c", "d"]),
    (_listed_answer([], [{"clause": "a", "replaced_by": "b"}]), ["a", "b"]),
    (_listed_answer([], ["a", None, {"clause": " "}, {}]), []),
    (_extend_answer(["x"]), []),
    (json.dumps({"patch": "+x"}), []),
    ("not json", []),
])
def test_listed_facts_reads_every_fact_the_list_names(answer, facts):
    assert bs.listed_facts(answer) == facts


def test_the_command_says_how_many_answers_were_left_out_for_their_list(
        tmp_path, capsys):
    log = tmp_path / "distill.jsonl"
    for ext_id, disps in (("kept_compact", [_kept(COMPACT[0])]),
                          ("plain_list", [_kept("c <= 4'd9")])):
        distill_log.record(log, "extend", {}, _listed_answer(COMPACT, disps),
                           "NECESSARY", [SYS, {"role": "user",
                                               "content": "grow it"}],
                           extension_id=ext_id, invariants_expanded=EXPANDED)
    out = tmp_path / "o.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "v2.jsonl",
                                          _rows("g0", 5))),
             "--distill", str(log), "--out", str(out)])
    said = capsys.readouterr().out
    assert ("1 extend answer(s) left out because their kept/replaced list "
            "names a fact EBMC cannot read") in said
    assert "extend kept_compact" in said
    assert "extend 1 kept / 1 skipped" in said
    assert [_facts_of(l) for l in _lines(out) if l.get("kind") == "extend"] \
        == [EXPANDED]


def test_a_design_answer_ebmc_cannot_read_is_left_out_and_its_idea_kept():
    """The idea still led to a design that passed; only the answer that
    writes a fact EBMC cannot read is left out."""
    bad = json.dumps({"top_module": "m", "verilog": "module m; endmodule",
                      "invariants": ["occ == $countones(valid)"]})
    implement = _ex("implement", bad, "success", idea="GOOD IDEA")
    implement["metadata"].update(run_id="run9", attempt=4)
    report = {}
    recs, counts = bs.generation_records(
        [_ex("idea", "GOOD IDEA", "success"), implement,
         _ex("generate", _design("module ok; endmodule"), "success")],
        fact_report=report)
    assert [r["kind"] for r in recs] == ["idea", "generate"]
    assert counts["implement"] == {"kept": 0, "skipped": 1}
    assert report["left_out"] == [("implement run9 attempt 4", [
        ("occ == $countones(valid)", ["system function $countones"])])]


@pytest.mark.parametrize("meta, name", [
    ({"extension_id": "e7", "run_id": "r"}, "e7"),
    ({"run_id": "r1", "attempt": 3}, "r1 attempt 3"),
    ({"source_run_id": "old", "source_attempt": 2}, "old attempt 2"),
    ({}, "example 5 of the input"),
])
def test_a_left_out_example_is_named_so_it_can_be_found(meta, name):
    assert bs.example_name({"metadata": meta}, 4) == name


def test_leaving_answers_out_moves_no_record_across_the_held_back_line(
        tmp_path, capsys):
    rows = _rows("g0", 20)
    h, t = _held_and_trained(rows)
    held_row = rows[h[0]]
    rows = _bad(rows, h[0], ONEHOT)        # its solving answer is left out
    log = tmp_path / "distill.jsonl"
    for child, parents, facts, ext_id in (
            # a design showing the left-out held-back row: still held back
            (held_row["verilog"], ["c0_000"], EXPANDED, "shows_held"),
            (None, [held_row["id"]], EXPANDED, "grew_from_held"),
            # compact facts and no plain list: left out
            ("module free; endmodule", ["c0_000"], COMPACT, "unreadable"),
            ("module free2; endmodule", [rows[t[0]]["id"]], EXPANDED,
             "trained")):
        distill_log.record(log, "extend", {}, _extend_answer(facts),
                           "NECESSARY", [SYS, {"role": "user",
                                               "content": "grow it"}],
                           child_verilog=child, parent_ids=parents,
                           extension_id=ext_id)
    out = tmp_path / "both.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "v2.jsonl", rows)),
             "--distill", str(log), "--out", str(out)])
    lines = _lines(out)
    assert [l["holdout"] for l in lines if l.get("kind") == "extend"] == \
        [True, True, False]
    solving = [l for l in lines if isinstance(l["prompt"], str)]
    held = bs.held_back(rows)
    assert len(solving) == 19
    assert all(l["holdout"] is (l["prompt"] in held) for l in solving)
    said = capsys.readouterr().out
    assert "extend 3 kept / 1 skipped" in said
    assert "0 extend answer(s) rebuilt" in said
    assert ("left out 1 generation answers (counted as skipped) with facts "
            "EBMC cannot read (not rewritten): extend unreadable") in said


def test_the_command_says_how_many_extend_answers_were_rebuilt(tmp_path,
                                                              capsys):
    log = tmp_path / "distill.jsonl"
    for ext_id, meta in (("rebuilt_one", {"invariants_expanded": EXPANDED}),
                         ("left_out_one", {})):
        distill_log.record(log, "extend", {}, _extend_answer(COMPACT),
                           "NECESSARY", [SYS, {"role": "user",
                                               "content": "grow it"}],
                           extension_id=ext_id, **meta)
    out = tmp_path / "o.jsonl"
    bs.main(["--corpus", str(_write_jsonl(tmp_path / "v2.jsonl",
                                          _rows("g0", 5))),
             "--distill", str(log), "--out", str(out)])
    said = capsys.readouterr().out
    assert ("1 extend answer(s) rebuilt with the plain fact list the checker "
            "proved") in said
    assert "extend left_out_one" in said
    assert "extend 1 kept / 1 skipped" in said
    gen = [l for l in _lines(out) if l.get("kind") == "extend"]
    assert [_facts_of(g) for g in gen] == [EXPANDED]


# the real data, read only (22 Oct 2026: five answers with $onehot(st) in
# corpus.jsonl, c0_144 with $countones in corpus_2026-10-09.jsonl, and a
# kept step of the 21 Oct trial with a compact fact)

REAL_NEW_CORPUS = REAL_CORPUS.parent / "corpus_2026-10-09.jsonl"
REAL_TRIAL = REAL_CORPUS.parent / "logs" / "real_systems_trial_2026-10-21"


def _real_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()
            if l.strip()]


def _unreadable_ids(rows):
    return [r["id"] for r in rows
            if r.get("invariants") and unreadable_facts(r["invariants"])]


@pytest.mark.skipif(not REAL_CORPUS.exists(), reason="no corpus here")
def test_no_real_answer_ebmc_cannot_read_reaches_the_answers_file(
        tmp_path, capsys):
    rows = _real_rows(REAL_CORPUS)
    bad = _unreadable_ids(rows)
    out = tmp_path / "v2.jsonl"
    bs.main(["--corpus", str(REAL_CORPUS), "--out", str(out)])
    lines = _lines(out)
    assert not any(unreadable_facts(_facts_of(l)) for l in lines)
    assert len(lines) == len(bs.build_pairs(rows)) - len(bad)
    held = bs.held_back(rows)
    assert all(l["holdout"] is (l["prompt"] in held) for l in lines)
    if bad:
        assert (f"left out {len(bad)} solving answers with facts EBMC cannot "
                f"read (not rewritten): {', '.join(bad)}") in \
            capsys.readouterr().out


@pytest.mark.skipif(not (REAL_CORPUS.exists() and REAL_NEW_CORPUS.exists()),
                    reason="no corpora here")
def test_no_real_answer_ebmc_cannot_read_reaches_the_combined_file(
        tmp_path, capsys):
    rows, new = _real_rows(REAL_CORPUS), _real_rows(REAL_NEW_CORPUS)
    bad = _unreadable_ids(rows) + _unreadable_ids(new)
    out = tmp_path / "combined.jsonl"
    bs.main(["--corpus", str(REAL_CORPUS), "--extra-corpus",
             str(REAL_NEW_CORPUS), "--out", str(out)])
    lines = _lines(out)
    assert not any(unreadable_facts(_facts_of(l)) for l in lines)
    assert len(lines) == (len(bs.build_pairs(rows)) + len(bs.build_pairs(new))
                          - len(bad))
    held = bs.held_back(rows)
    assert all(l["holdout"] is (l["prompt"] in held) for l in lines)
    if bad:
        assert f"left out {len(bad)} solving answers" in \
            capsys.readouterr().out


@pytest.mark.skipif(not (REAL_TRIAL / "extensions.jsonl").exists(),
                    reason="no 21 Oct trial here")
def test_the_real_kept_step_with_a_compact_fact_trains_its_plain_list(
        tmp_path):
    rec = next(json.loads(l) for l in
               (REAL_TRIAL / "extensions.jsonl").read_text().splitlines()
               if '"2026-10-21_07h28m39s_t001"' in l)

    def facts_of(call):
        try:
            return json.loads(call["raw"]).get("invariants")
        except (TypeError, ValueError, AttributeError):
            return None
    raw = next(c["raw"] for c in map(
        json.loads, (REAL_TRIAL / "calls.jsonl").read_text().splitlines())
        if facts_of(c) == rec["invariants_compact"])
    assert unreadable_facts(rec["invariants_compact"])     # as written
    assert not unreadable_facts(rec["invariants"])         # as proved
    [kept], _ = bs.generation_records(
        [_saved_extend(tmp_path, raw, rec["invariants"],
                       ext_id=rec["extension_id"])])
    assert _facts_of(kept) == rec["invariants"]
    assert bs.generation_records(
        [_saved_extend(tmp_path, raw, ext_id="no_plain_list")])[0] == []


@pytest.mark.skipif(not (REAL_TRIAL / "extensions.jsonl").exists(),
                    reason="no 21 Oct trial here")
def test_the_real_kept_step_that_kept_a_compact_parent_fact_is_left_out(
        tmp_path):
    """2026-10-21_08h58m35s_t030 kept its parent's `forall k in 0..7: ...`
    fact; of the trial's 19 kept steps, it is the only one whose
    kept/replaced list names a fact EBMC cannot read."""
    recs = [json.loads(l) for l in
            (REAL_TRIAL / "extensions.jsonl").read_text().splitlines()
            if l.strip()]
    kept = [r for r in recs if r.get("verdict") == "NECESSARY"]
    assert len(kept) == 19

    def listed(r):
        return bs.listed_facts(json.dumps(
            {"dispositions": r.get("dispositions") or []}))
    assert [r["extension_id"] for r in kept
            if unreadable_facts(listed(r))] == ["2026-10-21_08h58m35s_t030"]
    rec = next(r for r in kept
               if r["extension_id"] == "2026-10-21_08h58m35s_t030")
    answer = json.dumps({"patch": "+x", "invariants": rec["invariants_compact"],
                         "dispositions": rec["dispositions"]})
    report = {}
    assert bs.generation_records(
        [_saved_extend(tmp_path, answer, rec["invariants"],
                       ext_id=rec["extension_id"])],
        fact_report=report)[0] == []
    assert report["lists"] == ["2026-10-21_08h58m35s_t030"]


# --- the combined build and a v2 file rebuilt in place (23 Oct 2026) --------
# Review finding: rebuilt with this builder, sft_train_v2.jsonl leaves out
# the answers EBMC cannot read and marks every line, and the combined
# build, which compared line for line with the file as first written,
# stopped with "the v2 part differs". Either form is now accepted.

def _v2_setup(tmp_path, monkeypatch):
    """A v2 corpus of 20 rows, one of them with a fact EBMC cannot read,
    made the builder's own corpus, and a small new corpus."""
    rows = _bad(_rows("g0", 20), 3, ONEHOT)
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", rows)
    monkeypatch.setattr(bs, "CORPUS", corpus)
    monkeypatch.setattr(bs, "V2_FILE", tmp_path / "sft_train_v2.jsonl")
    new = _write_jsonl(tmp_path / "new.jsonl", _rows("c0", 4))
    return rows, corpus, new


def _combined(corpus, new, out):
    bs.main(["--corpus", str(corpus), "--extra-corpus", str(new),
             "--out", str(out)])
    return _lines(out)


def test_a_v2_file_rebuilt_in_place_lets_the_combined_build_run(
        tmp_path, monkeypatch, capsys):
    rows, corpus, new = _v2_setup(tmp_path, monkeypatch)
    bs.main(["--out", str(bs.V2_FILE)])             # rebuilt in place
    assert len(_lines(bs.V2_FILE)) == 19
    lines = _combined(corpus, new, tmp_path / "combined.jsonl")
    assert len(lines) == 19 + 4
    assert "v2 part matches sft_train_v2.jsonl line for line (rebuilt " \
        "with unreadable answers left out)" in capsys.readouterr().out


def test_the_v2_file_as_first_written_still_matches(tmp_path, monkeypatch,
                                                   capsys):
    rows, corpus, new = _v2_setup(tmp_path, monkeypatch)
    bs.V2_FILE.write_text("".join(
        json.dumps({"prompt": p["prompt"], "completion": p["completion"]})
        + "\n" for p in bs.build_pairs(rows)))       # all 20, unmarked
    assert len(_combined(corpus, new, tmp_path / "combined.jsonl")) == 23
    assert "(as first written)" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["answer", "mark", "order", "a line more"])
def test_any_other_v2_file_still_stops_the_combined_build(tmp_path,
                                                          monkeypatch,
                                                          change):
    rows, corpus, new = _v2_setup(tmp_path, monkeypatch)
    bs.main(["--out", str(bs.V2_FILE)])
    lines = _lines(bs.V2_FILE)
    if change == "answer":
        lines[0]["completion"] = json.dumps({"invariants": ["c <= 4'd8"]})
    elif change == "mark":
        lines[0]["holdout"] = not lines[0]["holdout"]
    elif change == "order":
        lines[0], lines[1] = lines[1], lines[0]
    else:
        lines.append(lines[-1])
    _write_jsonl(bs.V2_FILE, lines)
    out = tmp_path / "combined.jsonl"
    with pytest.raises(SystemExit, match="the v2 part differs"):
        _combined(corpus, new, out)
    assert not out.exists()
