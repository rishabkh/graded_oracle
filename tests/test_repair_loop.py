"""The benchmark authors' repair loop (Large Lemma Miners, src/agentic.py
and src/evaluation.py) on our scorer. EBMC and the model are faked, so
these pin the protocol: how one fact is judged, the feedback wording,
the reminder cadence, kept facts, trying kept facts together, the limit
of five answers, and the context window."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import repair_loop as rl                                         # noqa: E402

AUTHORS = ROOT.parent / "large_lemma_miners"


# --- reading EBMC's output the authors' way ---------------------------

def test_a_bounded_check_with_no_counterexample_counts_as_proven():
    # their run_ebmc tests for "PROVED" before "PROVED up to bound", so a
    # 30-step check that finds nothing is PROVEN, not a timeout
    out = "** Results:\n[main.p] always lemma0: PROVED up to bound 30\n"
    assert rl.ebmc_result(out, "", "correctness_bounded")["verdict"] == "PROVEN"


def test_a_refutation_carries_its_trace():
    out = ("** Results:\n[main.p] always lemma0: REFUTED\n"
           "Counterexample:\n\n  c = 0\n  c = 1\n")
    r = rl.ebmc_result(out, "", "correctness_bounded")
    assert r["verdict"] == "CEX" and r["info"] == "c = 0\n  c = 1"


def test_inconclusive_failure_stderr_and_silence():
    assert rl.ebmc_result("INCONCLUSIVE", "", "one_induction")["verdict"] \
        == "INCONCLUSIVE"
    r = rl.ebmc_result("FAILURE: x\nline 12: no such signal foo", "",
                       "one_induction")
    assert r["verdict"] == "ERROR" and r["info"] == "no such signal foo"
    assert rl.ebmc_result("", "boom", "one_induction")["verdict"] == "ERROR"
    assert rl.ebmc_result("nothing", "", "one_induction")["verdict"] \
        == "TIMEOUT"


# --- one fact: the three checks and the booleans -----------------------

def res(v, info=None):
    d = {"verdict": v}
    if info is not None:
        d["info"] = info
    return d


def entry(correct="PROVEN", one="INCONCLUSIVE", with_prop="INCONCLUSIVE",
          lemma=("a <= 3",)):
    return {"lemma": list(lemma), "correct": res(correct),
            "1-inductive": res(one), "1-inductive with property": res(with_prop)}


@pytest.mark.parametrize("correct,one,expected", [
    ("PROVEN", "INCONCLUSIVE", True),     # no cex in 30 steps, not refuted
    ("PROVEN", "CEX", False),             # refuted in one step
    ("CEX", "PROVEN", True),              # inductive alone wins
    ("TIMEOUT", "INCONCLUSIVE", False),
    ("ERROR", "ERROR", False),
])
def test_correct_follows_translate_entry_to_booleans(correct, one, expected):
    assert rl.booleans(entry(correct, one))["correct"] is expected


def test_solves_means_inductive_with_the_property():
    assert rl.solves(entry(with_prop="PROVEN"))
    assert not rl.solves(entry(with_prop="INCONCLUSIVE"))


def test_judge_runs_their_three_checks_and_falls_back_to_buechi(tmp_path):
    calls = []

    def run(bench, lemmas, mode, workdir, buechi=False):
        calls.append((mode, buechi))
        if mode == "one_inductive_with_prop" and not buechi:
            return res("ERROR", "needs buechi")
        return res("PROVEN")
    e = rl.judge("f.sv", ["a <= 3"], tmp_path, run=run)
    assert ("correctness_bounded", False) in calls
    assert ("one_induction", False) in calls
    assert ("one_inductive_with_prop", True) in calls      # the fallback
    assert e["1-inductive with property"]["verdict"] == "PROVEN"


# --- trying the facts: singletons, then kept facts in their order ------

def fake_run(table):
    """table: frozenset(lemmas) -> dict mode -> verdict."""
    def run(bench, lemmas, mode, workdir, buechi=False):
        return res(table.get(frozenset(lemmas), {}).get(mode, "INCONCLUSIVE"))
    return run


def test_a_single_fact_that_closes_the_proof_solves_at_once(tmp_path):
    table = {frozenset(["a"]): {"correctness_bounded": "PROVEN",
                                "one_inductive_with_prop": "PROVEN"}}
    out = rl.evaluate_subsets("f.sv", ["a", "b"], tmp_path, run=fake_run(table))
    assert any(rl.solves(e) for e in out)
    assert all(len(e["lemma"]) == 1 for e in out)          # no prefixes tried


def test_correct_facts_are_tried_together_inductive_first(tmp_path):
    table = {
        frozenset(["z"]): {"correctness_bounded": "PROVEN",
                           "one_induction": "PROVEN"},
        frozenset(["b"]): {"correctness_bounded": "PROVEN"},
        frozenset(["a"]): {"correctness_bounded": "PROVEN"},
        frozenset(["bad"]): {"correctness_bounded": "CEX"},
        frozenset(["z", "a"]): {"one_inductive_with_prop": "PROVEN"},
    }
    out = rl.evaluate_subsets("f.sv", ["b", "bad", "a", "z"], tmp_path,
                              run=fake_run(table))
    prefixes = [e["lemma"] for e in out if len(e["lemma"]) > 1]
    # their sort: 1-inductive first, then by text; stop once solved
    assert prefixes == [["z", "a"]]
    assert any(rl.solves(e) for e in out)


# --- the feedback, verbatim ---------------------------------------------

def single(lemma, **kw):
    return entry(lemma=(lemma,), **kw)


def test_feedback_sentences_are_the_authors(tmp_path):
    entries = [
        single("x", with_prop="PROVEN"),
        single("y", correct="PROVEN", one="INCONCLUSIVE"),
        single("z", correct="ERROR"),
        single("w", correct="TIMEOUT"),
    ]
    entries[2]["correct"]["info"] = "line 3: unknown identifier q"
    msg = rl.repair_message(entries, counter=1, show_cex=False)
    assert "// Feedback for x: \nThe conjunction of the lemma and the " \
           "property is 1-inductive, so the lemma helps prove the property. " \
           "Good job!" in msg
    assert "// Feedback for y: \nThe lemma is correct.\nHowever, it does not " \
           "form a 1-inductive argument when conjoined with the property. " \
           "Please give a better lemma." in msg
    assert "There's an error with the lemma: \nline 3: unknown identifier q" \
        in msg
    assert "The lemma timed out and its correctness could not be " \
           "determined." in msg


def test_an_incorrect_fact_gets_its_counterexample_only_when_asked():
    e = single("v", correct="CEX")
    e["correct"]["info"] = "c = 7"
    with_cex = rl.repair_message([e], counter=1, show_cex=True)
    assert "The lemma is incorrect. Here is a counterexample: \nc = 7" \
        in with_cex
    plain = rl.repair_message([e], counter=1, show_cex=False)
    assert "The lemma is incorrect. Repair or replace it." in plain
    assert "c = 7" not in plain


def test_counterexamples_are_capped():
    e = single("v", correct="CEX")
    e["correct"]["info"] = "x" * (rl.TRACE_CAP + 5000)
    msg = rl.repair_message([e], counter=1, show_cex=True)
    assert len(msg) < rl.TRACE_CAP + 2000 and "trace cut" in msg


def test_the_reminder_comes_on_their_cadence():
    # their counter is the number of answers before this one: the reminder
    # rides on the first, third and fifth repair messages
    e = [single("y")]
    assert rl.REMINDER_HEAD in rl.repair_message(e, counter=0, show_cex=False)
    assert rl.REMINDER_HEAD not in rl.repair_message(e, counter=1,
                                                     show_cex=False)
    assert rl.REMINDER_HEAD in rl.repair_message(e, counter=2, show_cex=False)


def test_reminder_copies_their_template_but_asks_for_our_format():
    if not (AUTHORS / "templates" / "repair_template.txt").exists():
        pytest.skip("authors' checkout absent")
    theirs = (AUTHORS / "templates" / "repair_template.txt").read_text()
    head = theirs.split("## Output Format")[0]
    assert head.strip() in rl.REMINDER
    assert '{"invariants": ["<expr>", ...]}' in rl.REMINDER
    assert "<json>" not in rl.REMINDER


def test_no_json_message_names_the_error_and_our_format():
    msg = rl.no_json_message("no JSON object found")
    assert msg.startswith("Your last response did not follow the expected "
                          "format. It yielded the following error: "
                          "no JSON object found")
    assert '{"invariants": ["<expr>", ...]}' in msg


# --- the conversation ----------------------------------------------------

def test_max_tokens_shrinks_with_the_conversation():
    short = [{"role": "user", "content": "x" * 300}]
    assert rl.max_tokens_for(short) == 16000
    long = [{"role": "user", "content": "x" * 3 * 32000}]
    assert rl.max_tokens_for(long) is None          # under 1000 left


def scripted(replies):
    """A fake model: each call pops the next reply; records the messages."""
    seen = []

    def ask(messages, max_tokens):
        seen.append([dict(m) for m in messages])
        return replies.pop(0), "stop"
    ask.seen = seen
    return ask


def run_loop(tmp_path, replies, table, first="{\"invariants\": [\"a\"]}",
             rounds=5, show_cex=False):
    rows = []
    out = rl.repair("f.sv", "QUESTION", first, ["a"] if "a" in first else None,
                    scripted(replies), rounds=rounds, show_cex=show_cex,
                    workdir=tmp_path, run=fake_run(table), log=rows.append)
    return out, rows


def test_five_answers_in_all_then_stop(tmp_path):
    replies = ['{"invariants": ["b"]}'] * 10
    out, rows = run_loop(tmp_path, replies, table={})
    assert [r["round"] for r in rows] == [0, 1, 2, 3, 4]
    assert out["stop_reason"] == "max_iterations" and not out["solved"]
    assert rows[-1]["final"] is True


def test_kept_facts_closing_together_stop_the_loop(tmp_path):
    table = {frozenset(["a"]): {"correctness_bounded": "PROVEN"},
             frozenset(["b"]): {"correctness_bounded": "PROVEN"},
             frozenset(["a", "b"]): {"one_inductive_with_prop": "PROVEN"}}
    replies = ['{"invariants": ["b"]}', '{"invariants": ["c"]}']
    out, rows = run_loop(tmp_path, replies, table)
    # round 0 keeps a; round 1 keeps b; a and b together close the proof
    assert out["solved"] and out["solved_round"] == 1
    assert out["kept"] == ["a", "b"]
    assert len(rows) == 2


def test_the_conversation_carries_answers_and_feedback(tmp_path):
    replies = ['{"invariants": ["b"]}', '{"invariants": ["c"]}']
    ask = scripted(list(replies))
    rows = []
    rl.repair("f.sv", "QUESTION", '{"invariants": ["a"]}', ["a"], ask,
              rounds=3, show_cex=False, workdir=tmp_path,
              run=fake_run({}), log=rows.append)
    first, second = ask.seen
    assert [m["role"] for m in first] == ["user", "assistant", "user"]
    assert first[0]["content"] == "QUESTION"
    assert first[1]["content"] == '{"invariants": ["a"]}'
    assert "// Feedback for a: " in first[2]["content"]
    assert [m["role"] for m in second] == ["user", "assistant", "user",
                                           "assistant", "user"]


def test_an_unreadable_answer_gets_the_format_message(tmp_path):
    ask = scripted(["no json here", '{"invariants": ["b"]}'])
    rows = []
    rl.repair("f.sv", "QUESTION", '{"invariants": ["a"]}', ["a"], ask,
              rounds=3, show_cex=False, workdir=tmp_path,
              run=fake_run({}), log=rows.append)
    assert ask.seen[1][-1]["content"].startswith(
        "Your last response did not follow the expected format.")
    assert rows[1]["lemmas"] is None


def test_a_full_window_stops_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "max_tokens_for", lambda messages: None)
    out, rows = run_loop(tmp_path, ['{"invariants": ["b"]}'], table={})
    assert out["stop_reason"] == "context_full" and len(rows) == 1


def test_rows_carry_what_analysis_needs(tmp_path):
    out, rows = run_loop(tmp_path, ['{"invariants": ["b"]}'] * 4, table={},
                         rounds=2)
    r = rows[1]
    for field in ("round", "raw", "finish", "lemmas", "per_lemma", "kept",
                  "solved", "solve_wall_s"):
        assert field in r
    assert r["per_lemma"][0]["lemma"] == "b"
    assert set(r["per_lemma"][0]) >= {"correct", "inductive_with_prop",
                                      "correctness", "one_induction",
                                      "with_prop"}
    assert rows[-1]["stop_reason"] == "max_iterations"


def test_a_check_already_made_is_not_run_again(tmp_path):
    """Their evaluator caches every EBMC query; re-trying the kept lemmas
    each round must not rerun checks already made."""
    calls = []

    def run(bench, lemmas, mode, workdir, buechi=False):
        calls.append((tuple(lemmas), mode, buechi))
        if lemmas == ["a"] and mode == "correctness_bounded":
            return res("PROVEN")
        return res("INCONCLUSIVE")
    rows = []
    rl.repair("f.sv", "QUESTION", '{"invariants": ["a"]}', ["a"],
              scripted(['{"invariants": ["a"]}'] * 3), rounds=3,
              show_cex=False, workdir=tmp_path, run=run, log=rows.append)
    assert len(calls) == len(set(calls))


def test_each_row_keeps_the_feedback_the_model_was_sent(tmp_path):
    ask = scripted(['{"invariants": ["b"]}'])
    rows = []
    rl.repair("f.sv", "QUESTION", '{"invariants": ["a"]}', ["a"], ask,
              rounds=2, show_cex=False, workdir=tmp_path, run=fake_run({}),
              log=rows.append)
    assert rows[0]["feedback"] == ask.seen[0][-1]["content"]
