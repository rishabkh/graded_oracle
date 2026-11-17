"""The generator's training file (14 Nov 2026, Rishab's "let's do it").

Formal Disco teaches its open model the workers' jobs from the stronger
model's own calls before the open model runs as a worker: every call that
ended in a program the checker passed becomes one example, the question as
it was asked and the answer as it was given (formal-disco/distill.py,
build_sft_records). build_sft.generation_records has applied those rules
here since 8 Oct 2026. This file adds the one rule they have no need for:
our checker cannot see a design that does the wrong thing, so a call is
kept only when its design is labeled good or weak. The facts file the
solver was trained on goes in first, line for line.

Why not train on every passed call, as October's design-writing model
did: 129 of the 701 saved calls that passed made a design labeled wrong,
and 15 of that model's own 24 passing designs were wrong."""
import io
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "initiator"))
sys.path.insert(0, str(ROOT / "extender"))

import build_generator_set as G                                  # noqa: E402
import sft_data                                                  # noqa: E402
from all_designs import key_of                                   # noqa: E402
from solver_baseline import SOLVER_PROMPT                        # noqa: E402

COUNTER = "Up counter: counts up by one each cycle."
DEMUX = "Demultiplexer: routes one input word to one of several outputs."


def design(name):
    return (f"module {name}(input clk);\n  reg [1:0] c = 0;\n"
            "  always @(posedge clk) c <= c + 1;\nendmodule\n")


def call(kind="generate", name="a", outcome="success", construct=COUNTER,
         model="openai:gpt-6.1-sol"):
    """One saved call, in the shape distill_log.record writes."""
    meta = {"messages": [{"role": "system", "content": "You write modules."},
                         {"role": "user", "content": "Write: " + construct}],
            "verdict": "NECESSARY" if outcome == "success" else "NOT_PROVEN",
            "model": model, "run_id": "run1", "attempt": name}
    if kind == "idea":
        return {"prompt": kind, "arguments": {"construct": construct},
                "response": json.dumps({"idea": "an idea for " + name}),
                "outcome": outcome, "metadata": meta}
    if kind == "extend":
        meta["child_verilog"] = design(name)
        return {"prompt": kind,
                "arguments": {"parent_id": "p0", "ext_type": "structural"},
                "response": json.dumps({"note": "adds a stage",
                                        "patch": "@@ a @@\n+ b",
                                        "invariants": ["c <= 2'd3"]}),
                "outcome": outcome, "metadata": meta}
    arguments = ({"idea": "an idea for " + name} if kind == "implement"
                 else {"construct": construct})
    return {"prompt": kind, "arguments": arguments,
            "response": json.dumps({"invariants": ["c <= 2'd3"],
                                    "verilog": design(name)}),
            "outcome": outcome, "metadata": meta}


def labels(**by_name):
    return {key_of(design(name)): label for name, label in by_name.items()}


def fact_line(name, held=False):
    return json.dumps({"prompt": SOLVER_PROMPT.format(verilog=design(name)),
                       "completion": json.dumps({"invariants": ["c <= 2'd3"]}),
                       "holdout": held})


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def designs_of(calls):
    return [G.design_of_call(c) for c in calls]


# --- which calls are kept ------------------------------------------------------

def test_a_passed_call_whose_design_is_good_or_weak_is_kept():
    calls = [call(name="a"), call(name="b")]
    kept, report = G.choose_calls(calls, labels(a="good", b="weak"))
    assert kept == calls
    assert report["kept"] == {"generate": 2}
    assert report["labels"] == {"good": 1, "weak": 1}


def test_a_call_whose_design_is_labeled_wrong_is_left_out():
    kept, report = G.choose_calls([call(name="a"), call(name="bad")],
                                  labels(a="good", bad="wrong"))
    assert designs_of(kept) == [design("a")]
    assert report["left_out"] == {"wrong": 1}


def test_a_call_whose_design_nobody_labeled_is_left_out_and_counted():
    """Not known to be right is not kept: every design gets a label
    before its call can teach."""
    kept, report = G.choose_calls([call(name="a")], {})
    assert kept == []
    assert report["left_out"] == {"no label": 1}


def test_a_call_the_checker_turned_away_is_left_out():
    kept, report = G.choose_calls([call(name="a", outcome="fail"),
                                   call(name="b", outcome="error")],
                                  labels(a="good", b="good"))
    assert kept == []
    assert report["left_out"] == {"failed": 2}


def test_an_idea_call_is_left_out():
    """Today's initiator writes the design in one step: there is no idea
    job for the model to learn."""
    kept, report = G.choose_calls([call(kind="idea", name="a")],
                                  labels(a="good"))
    assert kept == []
    assert report["left_out"] == {"idea": 1}


@pytest.mark.parametrize("kind", ["initiate", "generate", "implement",
                                  "repair", "extend"])
def test_every_design_job_is_kept(kind):
    c = call(kind=kind, name="a")
    kept, report = G.choose_calls([c], labels(a="weak"))
    assert kept == [c]
    assert report["kept"] == {kind: 1}


def test_a_grown_design_is_looked_up_by_the_design_the_step_made():
    """An extend answer is a patch: the design it made is kept beside it."""
    c = call(kind="extend", name="grown")
    assert G.design_of_call(c) == design("grown")
    kept, _ = G.choose_calls([c], labels(grown="wrong"))
    assert kept == []


def test_a_design_written_with_other_spacing_still_finds_its_label():
    c = call(name="a")
    answer = json.loads(c["response"])
    answer["verilog"] = answer["verilog"].replace("\n", "\n\n    ")
    c["response"] = json.dumps(answer)
    kept, _ = G.choose_calls([c], labels(a="good"))
    assert kept == [c]


def test_the_report_says_which_model_wrote_the_kept_calls():
    calls = [call(name="a"), call(name="b", model="claude-opus-5-5")]
    _, report = G.choose_calls(calls, labels(a="good", b="good"))
    assert report["writers"] == {"openai:gpt-6.1-sol": 1,
                                 "claude-opus-5-5": 1}


# --- kinds set aside for the generation test ------------------------------------

def test_kind_names_are_read_from_a_kinds_list(tmp_path):
    kinds = tmp_path / "kinds.txt"
    kinds.write_text(DEMUX + "\n\nBus keeper: holds the last value.\n")
    assert G.kind_names(kinds) == {"demultiplexer", "bus keeper"}


def test_a_call_for_a_kind_set_aside_for_the_generation_test_is_left_out():
    calls = [call(name="a"), call(name="d", construct=DEMUX)]
    kept, report = G.choose_calls(calls, labels(a="good", d="good"),
                                  {"demultiplexer"})
    assert designs_of(kept) == [design("a")]
    assert report["left_out"] == {"test kind": 1}


def test_a_two_step_design_takes_its_kind_from_its_idea_call():
    idea = call(kind="idea", name="d", construct=DEMUX)
    made = call(kind="implement", name="d")
    kept, report = G.choose_calls([idea, made], labels(d="good"),
                                  {"demultiplexer"})
    assert kept == []
    assert report["left_out"] == {"idea": 1, "test kind": 1}


# --- the labels ----------------------------------------------------------------

def test_labels_are_read_by_design_and_a_later_file_wins(tmp_path):
    k1, k2, k3 = (key_of(design(n)) for n in "abc")
    first = write(tmp_path / "first.jsonl", [
        {"key": k1, "label": "wrong"}, {"key": k2, "label": "weak"},
        {"key": k3, "label": None}])
    second = write(tmp_path / "second.jsonl", [{"key": k1, "label": "good"}])
    assert G.load_labels([str(first), str(second)]) == {k1: "good",
                                                        k2: "weak"}


def test_a_label_file_of_ids_given_without_its_corpus_stops_the_run(tmp_path):
    """Its labels would match no design, and every design would only show
    up as one "nobody labeled"."""
    labs = write(tmp_path / "labels.jsonl", [{"key": "t0_001", "label": "good"}])
    with pytest.raises(SystemExit, match="t0_001"):
        G.load_labels([str(labs)])


def test_labels_given_by_corpus_id_find_their_design(tmp_path):
    corpus = write(tmp_path / "corpus.jsonl", [
        {"id": "t0_001", "verilog": design("a")},
        {"id": "t0_002", "verilog": design("b")}])
    labs = write(tmp_path / "labels.jsonl", [
        {"key": "t0_001", "label": "good"}, {"key": "t0_002", "label": "wrong"}])
    assert G.load_labels([f"{labs}={corpus}"]) == {
        key_of(design("a")): "good", key_of(design("b")): "wrong"}


def test_a_label_for_an_id_its_corpus_does_not_have_stops_the_run(tmp_path):
    corpus = write(tmp_path / "corpus.jsonl",
                   [{"id": "t0_001", "verilog": design("a")}])
    labs = write(tmp_path / "labels.jsonl", [
        {"key": "t0_001", "label": "good"}, {"key": "t0_009", "label": "weak"}])
    with pytest.raises(SystemExit, match="t0_009"):
        G.load_labels([f"{labs}={corpus}"])


# --- the file ------------------------------------------------------------------

def test_the_facts_lines_come_first_exactly_as_they_were():
    facts = [fact_line("f1"), fact_line("f2", held=True)]
    c = call(name="a")
    lines, _ = G.build(facts, [c], labels(a="good"))
    assert lines[:2] == facts
    assert [json.loads(l) for l in lines[2:]] == [{
        "prompt": c["metadata"]["messages"],
        "completion": [{"role": "assistant", "content": c["response"]}],
        "holdout": False, "kind": "generate"}]


def test_a_call_that_shows_a_held_back_design_is_held_back_too():
    """The facts file holds some designs back to watch the loss. A call
    that wrote one of them must not put it back into training."""
    facts = [fact_line("f1"), fact_line("held", held=True)]
    lines, report = G.build(facts, [call(name="held"), call(name="a")],
                            labels(held="good", a="good"))
    assert [json.loads(l)["holdout"] for l in lines[2:]] == [True, False]
    assert report["held_back"] == 1


def test_a_call_with_a_fact_the_scoring_checker_cannot_read_is_left_out():
    c = call(name="a")
    answer = json.loads(c["response"])
    answer["invariants"] = ["$onehot(c)"]
    c["response"] = json.dumps(answer)
    lines, report = G.build([fact_line("f1")], [c], labels(a="good"))
    assert lines == [fact_line("f1")]
    assert report["left_out"] == {"unreadable fact": 1}
    assert report["kept"] == {}


def test_a_facts_file_without_held_back_marks_stops_the_run():
    """The trainer wants a mark on every line or on none, and the calls
    carry one."""
    line = json.dumps({"prompt": SOLVER_PROMPT.format(verilog=design("f")),
                       "completion": "{}"})
    with pytest.raises(SystemExit, match="held-back mark"):
        G.build([line], [], {})


def test_a_held_back_line_that_is_not_a_solver_question_stops_the_run():
    line = json.dumps({"prompt": "something else", "completion": "x",
                       "holdout": True})
    with pytest.raises(SystemExit, match="solver question"):
        G.build([line], [], {})


# --- the command ---------------------------------------------------------------

def files(tmp_path):
    facts = tmp_path / "facts.jsonl"
    facts.write_text(fact_line("f1") + "\n" + fact_line("f2", held=True) + "\n")
    calls = write(tmp_path / "distill_calls.jsonl", [
        call(name="a"), call(name="bad"), call(name="n", outcome="fail"),
        call(kind="extend", name="g")])
    labs = write(tmp_path / "labels.jsonl", [
        {"key": key_of(design("a")), "label": "good"},
        {"key": key_of(design("bad")), "label": "wrong"},
        {"key": key_of(design("g")), "label": "weak"}])
    out = tmp_path / "generator.jsonl"
    return facts, out, ["--facts", str(facts), "--calls", str(calls),
                        "--labels", str(labs), "--out", str(out)]


def test_the_command_writes_the_file_and_says_what_it_kept(tmp_path, capsys):
    facts, out, argv = files(tmp_path)
    G.main(argv)
    lines = out.read_text().splitlines()
    assert lines[:2] == facts.read_text().splitlines()
    assert [json.loads(l)["kind"] for l in lines[2:]] == ["generate", "extend"]
    said = capsys.readouterr().out
    for words in ("2 lines kept as they are", "1 held back",
                  "4 saved calls", "kept 2", "generate 1", "extend 1",
                  "good 1", "weak 1", "wrong 1", "failed 1",
                  "4 lines ->"):
        assert words in said, (words, said)


def test_the_trainer_reads_the_file_the_command_writes(tmp_path):
    _, out, argv = files(tmp_path)
    G.main(argv)
    rows = sft_data.load_pairs(out)
    trained, held = sft_data.split(rows)
    assert (len(trained), len(held)) == (3, 1)


def test_the_command_never_writes_over_a_file(tmp_path):
    _, out, argv = files(tmp_path)
    out.write_text("a file a model was trained on\n")
    with pytest.raises(SystemExit, match="exists"):
        G.main(argv)
    assert out.read_text() == "a file a model was trained on\n"


def test_the_command_leaves_out_the_test_kinds_it_is_given(tmp_path, capsys):
    _, out, argv = files(tmp_path)
    kinds = tmp_path / "kinds.txt"
    kinds.write_text(COUNTER + "\n")
    G.main(argv + ["--test-kinds", str(kinds)])
    lines = out.read_text().splitlines()
    assert [json.loads(l)["kind"] for l in lines[2:]] == ["extend"]
    assert "test kind 2" in capsys.readouterr().out


# --- one spinner for the run -----------------------------------------------------

class TTY(io.StringIO):
    def isatty(self):
        return True


def test_the_command_shows_a_spinner_while_it_works(tmp_path, monkeypatch):
    """Every script he runs shows the shared spinner (10 Nov 2026)."""
    _, out, argv = files(tmp_path)
    err = TTY()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(G.Spinner, "quiet", False)
    real = G.build

    def slow(*args, **kwargs):
        deadline = time.monotonic() + 5
        while "building the generator's training file" not in err.getvalue():
            assert time.monotonic() < deadline, "no spinner was drawn"
            time.sleep(0.01)
        return real(*args, **kwargs)

    monkeypatch.setattr(G, "build", slow)
    G.main(argv)
    assert "building the generator's training file" in err.getvalue()
    assert out.exists()
