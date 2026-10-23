"""A realism report after every run (21 Oct 2026, Rishab: "make all 1-8
changes", item 8d).

realism_report.py reads a run's log folder and the corpus it grew, and
sets the designs the run kept beside real open-source hardware (measured
15 Oct 2026 over 560 modules): steps tried and kept, growth per lineage,
size, how the designs are written, facts per design, what the automatic
engine proved alone, and what the model calls cost. batch.py writes it at
the end of every run with --log-dir; a report that fails never fails the
run. Everything here is a small made-up run in a temporary folder: no
model, no checker, nothing under extender/logs."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))

import batch                                                    # noqa: E402
import distractor                                               # noqa: E402
import extend                                                   # noqa: E402
import llm_client                                               # noqa: E402
import realism_report as R                                      # noqa: E402
import sense_check                                              # noqa: E402
from build_corpus import honest_lines                           # noqa: E402


# --- how a design is written -------------------------------------------------

def test_a_handshake_is_a_valid_and_a_ready_in_one_module():
    for names in ("in_valid, in_ready", "s_axis_tvalid, s_axis_tready",
                  "push_vld, push_rdy", "valid, ready"):
        assert R.has_handshake(f"module m;\n wire {names};\nendmodule")
    for names in ("in_valid, full", "busy, in_ready",
                  "invalidate, in_ready", "in_valid, already_seen"):
        assert not R.has_handshake(f"module m;\n wire {names};\nendmodule")
    # one in each of two modules is not a handshake
    assert not R.has_handshake("module a;\n wire in_valid;\nendmodule\n"
                               "module b;\n wire in_ready;\nendmodule")
    # nor are names in comments
    assert not R.has_handshake("module m;\n // in_valid in_ready\nendmodule")


def test_an_asynchronous_reset_has_two_edges_in_one_event_list():
    for text in ("always @(posedge clk or negedge rst_n)",
                 "always @(posedge clk or posedge reset)",
                 "always_ff @(posedge i_clk, negedge i_aresetn)",
                 "always @(negedge rst_n or posedge clk)"):
        assert R.traits(f"module m;\n{text} q <= 1'b0;\nendmodule")[
            "asynchronous reset"], text
    for text in ("always @(posedge clk)", "always @(*)",
                 "// always @(posedge clk or negedge rst_n)"):
        assert not R.traits(f"module m;\n{text} q <= 1'b0;\nendmodule")[
            "asynchronous reset"], text


def test_each_way_of_writing_is_found():
    plain = "module m (input wire clk);\n  reg [7:0] q;\nendmodule\n"
    assert not any(R.traits(plain).values())
    found = {
        "parameters": "  parameter DEPTH = 4;",
        "generate loops": "  genvar g;",
        "enum": "  typedef enum logic [1:0] {IDLE, BUSY} state_t;",
        "arrays": "  reg [7:0] mem [0:3];",
    }
    for trait, line in found.items():
        text = plain.replace("endmodule", line + "\nendmodule")
        assert R.traits(text)[trait], trait
    # a localparam counts as a parameter, as it did on 15 Oct 2026
    assert R.traits("module m;\n localparam N = 2;\nendmodule")["parameters"]
    assert R.traits("module m;\n generate\n endgenerate\nendmodule")[
        "generate loops"]
    # a constant table holds no state: not an array
    assert not R.traits("module m;\n localparam logic [7:0] T [0:3] = "
                        "'{8'd1, 8'd2, 8'd3, 8'd4};\nendmodule")["arrays"]
    assert R.traits(plain + plain.replace("module m", "module n"))[
        "more than one module"]
    assert not R.traits(plain)["more than one module"]


def test_lines_are_counted_without_comments_or_blank_lines():
    text = ("// header comment\nmodule m;\n\n  /* a\n  block */\n"
            "  wire a; // why\nendmodule\n")
    assert R.code_lines(text) == 3
    assert [R.code_lines(t) for t in R.module_texts(
        text + "module n;\n wire b;\n wire c;\nendmodule\n")] == [3, 4]


def test_percentiles_read_between_neighbouring_values():
    values = [10, 20, 30, 40, 50]
    assert R.percentile(values, 50) == 30
    assert R.percentile(values, 75) == 40
    assert R.percentile(values, 90) == pytest.approx(46)
    assert R.percentile([7], 90) == 7
    assert R.percentile([], 50) is None


# --- a small run -------------------------------------------------------------

def module(name, n, body=()):
    """One module with exactly `n` honest lines: its header, the `body`
    lines (one honest line each) and plain wires for the rest."""
    lines = [f"module {name} (input wire clk, output wire [7:0] q);"]
    lines += list(body)
    lines += [f"    wire [7:0] {name}_w{i} = 8'd{i};"
              for i in range(n - 1 - len(body))]
    return "\n".join(lines + ["endmodule"]) + "\n"


START = module("ctr", 10)
CHILD = (module("ctr", 12) +
         module("sub", 8, ["    parameter DEPTH = 4;",
                           "    reg [7:0] mem [0:3];",
                           "    wire in_valid, in_ready;"]))
GRANDCHILD = (module("ctr", 20, [
    "    always @(posedge clk or negedge rst_n) r <= 8'd0;",
    "    typedef enum logic [1:0] {IDLE, BUSY} state_t;",
    "    genvar g;"]) + module("sub", 10))
SOURCE, PARTNER = module("src", 20), module("snk", 10)
JOINED = SOURCE + PARTNER + module("top", 10)
EARLIER, LATER = module("ctr", 15), module("ctr", 45)


def row(id, verilog, gen, parent=None, ext=None, engine=None, parent2=None,
        invariants=("a",), compact=None):
    r = {"id": id, "generation": gen, "parent": parent, "parent2": parent2,
         "source_extension_id": ext, "top_module": "ctr",
         "verilog": verilog, "invariants": list(invariants),
         "metrics": {"engine_alone": engine} if engine else {}}
    if compact:
        r["invariants_compact"] = list(compact)
    return r


CORPUS = [
    row("c0_001", START, 0, engine="timed_out"),
    row("c0_002", PARTNER, 0, engine="proved"),
    row("c0_003", SOURCE, 0),
    # grown by this run: start c0_001 -> c1_000 -> c2_000
    row("c1_000", CHILD, 1, "c0_001", "e1", "proved", invariants="ab"),
    row("c2_000", GRANDCHILD, 2, "c1_000", "e2",
        invariants=[f"v[{k}] <= 1'b1" for k in range(9)],
        compact=["forall k in 0..8: v[k] <= 1'b1", "a"]),
    # a combination: its lineage is the design extended, not the partner
    row("c1_001", JOINED, 1, "c0_003", "e4", "error", parent2="c0_002",
        invariants="abcd"),
    # grown by an earlier run, then extended by this one
    row("c1_002", EARLIER, 1, "c0_001", "earlier_run", "proved"),
    row("c2_001", LATER, 2, "c1_002", "e5"),
]

TIMED_OUT = {"without_invariants": {"runs": [
    {"engine": "abc pdr", "mode": "prove", "rc": 8}]}}
RECORDS = [
    {"extension_id": "e1", "ext_type": "structural", "move": "SUBUNIT",
     "verdict": "NECESSARY"},
    {"extension_id": "e2", "ext_type": "structural", "move": "STORAGE",
     "verdict": "NECESSARY", "result": TIMED_OUT},
    {"extension_id": "e3", "ext_type": "structural", "move": "STORAGE",
     "verdict": "NOT_PROVEN"},
    {"extension_id": "e4", "ext_type": "compose", "move": None,
     "verdict": "NECESSARY"},
    {"extension_id": "e5", "ext_type": "structural", "move": "SUBUNIT",
     "verdict": "NECESSARY"},
    {"extension_id": "e6", "ext_type": "compose", "move": None,
     "verdict": "NO_PARTNER"},
    # proved, but the same design as one already kept: not a new row
    {"extension_id": "e7", "ext_type": "distractor", "move": None,
     "verdict": "NECESSARY"},
]


def call(system, tokens_in, tokens_out, batch=True,
         model="claude-opus-5-5"):
    return {"model": model, "batch": batch, "stop": "ok",
            "request": {"system": system, "messages": []},
            "usage": {"input": tokens_in, "output": tokens_out}}


CALLS = [
    call(extend.SYSTEM_PROMPT, 1_000_000, 100_000),
    call(extend.SYSTEM_PROMPT, 1_000_000, 100_000),
    call(sense_check.SYSTEM, 100_000, 10_000, batch=False),
    # a distractor question sends no system text; this request failed
    {"model": "claude-opus-5-5", "batch": True, "request": {"user": "q"},
     "usage": None, "stop": "exception: RuntimeError: x"},
    call("Some other script.", 10, 10, model="model-with-no-price"),
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def run(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_jsonl(logs / "extensions.jsonl", RECORDS)
    write_jsonl(logs / "calls.jsonl", CALLS)
    corpus = tmp_path / "corpus_trial.jsonl"
    write_jsonl(corpus, CORPUS)
    return logs, corpus


def test_the_fixture_designs_have_the_sizes_they_say():
    assert [honest_lines(v) for v in (START, CHILD, GRANDCHILD, SOURCE,
                                      JOINED, EARLIER, LATER)] == \
        [10, 20, 30, 20, 40, 15, 45]


def test_steps_tried_and_kept_by_verdict_and_by_kind(run):
    m = R.measure(*run)
    assert (m["tried"], m["kept"]) == (7, 4)
    assert m["verdicts"] == {"NECESSARY": 5, "NOT_PROVEN": 1,
                             "NO_PARTNER": 1}
    assert m["kinds"] == {"SUBUNIT": [2, 2], "STORAGE": [2, 1],
                          "compose": [2, 1], "distractor": [1, 0]}


def test_growth_is_measured_from_where_the_run_started(run):
    lineages = {x["start"]: x for x in R.measure(*run)["lineages"]}
    assert set(lineages) == {"c0_001", "c0_003", "c1_002"}
    a = lineages["c0_001"]
    assert (a["deepest"], a["generation"], a["start_lines"], a["lines"],
            a["kept"]) == ("c2_000", 2, 10, 30, 2)
    assert a["factor"] == pytest.approx(3.0)
    # a combination grows the design it extended, not its partner
    assert lineages["c0_003"]["factor"] == pytest.approx(2.0)
    # a row an earlier run grew is where this run started
    assert lineages["c1_002"]["factor"] == pytest.approx(3.0)


def test_a_start_missing_from_the_corpus_has_no_factor():
    lost = row("c1_009", CHILD, 1, "c0_404", "e1")
    (x,) = R.lineage_growth([lost], {"c1_009": lost})
    assert x["start"] == "c0_404" and x["start_lines"] is None
    assert x["factor"] is None
    assert "c0_404" in "\n".join(R._growth_text({"lineages": [x]}))


def test_how_the_kept_designs_are_written(run):
    m = R.measure(*run)
    assert m["traits"] == {"more than one module": 3,
                           "valid/ready handshake": 1, "parameters": 1,
                           "generate loops": 1, "asynchronous reset": 1,
                           "enum": 1, "arrays": 1}
    assert sorted(m["module_lines"]) == sorted(
        R.code_lines(t) for v in (CHILD, GRANDCHILD, JOINED, LATER)
        for t in R.module_texts(v))


def test_facts_are_counted_expanded_and_as_written(run):
    assert sorted(R.measure(*run)["facts"]) == [(1, 1), (2, 2), (4, 4),
                                                (9, 2)]


def test_what_the_engine_found_alone(run):
    m = R.measure(*run)
    # the row's own metric first; else its step's check in the log
    # (c2_000 ran out of time there); else not known
    assert sorted(m["engine"]) == ["error", "proved", "timed_out",
                                   "unknown"]
    assert sorted(m["engine_starts"]) == ["proved", "timed_out", "unknown"]
    assert R.engine_word({"metrics": {}}, {"result": TIMED_OUT}) == \
        "timed_out"
    assert R.engine_word({"metrics": {"engine_alone": "proved"}},
                         {"result": TIMED_OUT}) == "proved"
    assert R.engine_word({}, {"verdict": "NECESSARY"}) == "unknown"


def test_calls_are_priced_at_list_price_and_half_for_a_batch(run):
    total, kinds = R.measure(*run)["calls"]
    p_in, p_out = llm_client.PRICES["claude-opus-5-5"]
    extension = (1.0 * p_in + 0.1 * p_out) / 2      # one batch call
    review = 0.1 * p_in + 0.01 * p_out              # not a batch call
    assert total["dollars"] == pytest.approx(2 * extension + review)
    assert (total["calls"], total["batch"]) == (5, 4)
    assert (total["in"], total["out"]) == (2_100_010, 210_010)
    assert kinds["extension questions and partner choices"]["calls"] == 2
    assert kinds["extension questions and partner choices"]["dollars"] == \
        pytest.approx(2 * extension)
    assert kinds["reviews"]["dollars"] == pytest.approx(review)
    assert kinds["distractor questions"] == {"calls": 1, "dollars": 0.0}
    # a model with no price is named, never priced as something else
    assert total["unpriced"] == {"model-with-no-price": 1}
    assert kinds["other"]["calls"] == 1


def test_the_report_sets_ours_beside_the_real_numbers(run):
    m = R.measure(*run)
    text = R.render(m)
    assert "tried 7 (every answer checked, retries included), kept 4 (57%)" \
        in text
    assert "NECESSARY 5, NOT_PROVEN 1, NO_PARTNER 1" in text
    assert "median growth x3.0 over 3 lineage(s); most x3.0" in text
    assert "real modules" in text
    assert "    76    223    504" in text
    for trait, real in R.REAL_SHARES.items():
        assert trait in text and f"{100 * real:.0f}%" in text
    assert "more than one module                          3 (75%)" in text
    assert "median of 18 facts per module" in text
    assert R.engine_share(m["engine"]) == ("timed_out 1 (25%), error 1 "
                                           "(25%), proved 1 (25%), unknown "
                                           "1 (25%)")
    # wrapped between items, never inside one
    assert "  kept designs: timed_out 1 (25%), error 1 (25%), proved 1 " \
        "(25%),\n    unknown 1 (25%)" in text
    assert "5 calls, 4 of them at half price as a batch" in text
    assert "1 call(s) of model-with-no-price" in text
    assert "per kept design: $" in text


def test_the_report_is_saved_in_the_log_folder_and_printed(run, capsys):
    logs, corpus = run
    path = R.write_report(logs, corpus)
    assert path == logs / "realism_report.txt"
    out = capsys.readouterr().out
    assert path.read_text() in out
    assert out.startswith("REALISM REPORT for the run logged in")


def test_the_report_can_be_saved_elsewhere(run, tmp_path):
    logs, corpus = run
    out = tmp_path / "elsewhere" / "report.txt"
    out.parent.mkdir()
    R.main(["--log-dir", str(logs), "--corpus", str(corpus),
            "--out", str(out)])
    assert out.exists()
    assert not (logs / "realism_report.txt").exists()


def test_the_command_needs_a_log_folder_and_a_corpus(run, tmp_path):
    logs, corpus = run
    with pytest.raises(SystemExit):
        R.main(["--log-dir", str(tmp_path / "missing"), "--corpus",
                str(corpus)])
    with pytest.raises(SystemExit):
        R.main(["--log-dir", str(logs), "--corpus",
                str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit):
        R.main(["--log-dir", str(logs)])


def test_a_run_that_kept_nothing_and_logged_nothing_still_has_a_report(
        tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    corpus = tmp_path / "corpus_trial.jsonl"
    write_jsonl(corpus, [{"id": "c0_000", "generation": 0}])
    text = R.write_report(logs, corpus).read_text()
    assert "tried 0" in text and "kept 0" in text
    assert "no designs kept" in text
    assert "no call log (calls.jsonl) in the log folder" in text


def test_a_line_cut_off_by_a_stopped_run_is_left_out(tmp_path):
    path = tmp_path / "extensions.jsonl"
    path.write_text(json.dumps({"extension_id": "e1"}) + "\n"
                    + '{"extension_id": "e2", "verd')
    assert R.read_jsonl(path) == [{"extension_id": "e1"}]
    assert R.read_jsonl(tmp_path / "none.jsonl") == []


# --- at the end of a batch run -----------------------------------------------

@pytest.fixture
def command(monkeypatch, tmp_path):
    """batch.main() with the loop replaced; the logs go to a temporary
    folder, and main()'s moved log paths are put back after."""
    monkeypatch.setattr(distractor, "OUT_LOG", tmp_path / "extensions.jsonl")
    monkeypatch.setattr(batch, "QUEUE_LOG", tmp_path / "batch_queue.jsonl")
    monkeypatch.setattr(batch, "FIXER_QUEUE", tmp_path / "fixer_queue.jsonl")
    corpus = tmp_path / "corpus_trial.jsonl"
    write_jsonl(corpus, CORPUS[:3])

    def fake_loop(corpus_rows, executor, **kw):
        # the run's step and its kept row, as the real loop leaves them
        write_jsonl(distractor.OUT_LOG, RECORDS[:1])
        with corpus.open("a") as f:
            f.write(json.dumps(CORPUS[3]) + "\n")
        return [CORPUS[3]]
    monkeypatch.setattr(batch, "run_loop", fake_loop)

    def run(*extra):
        batch.main(["--max-calls", "1", "--corpus", str(corpus)]
                   + list(extra))
    return run, corpus


def test_a_run_with_a_log_folder_ends_with_its_report(command, tmp_path,
                                                       capsys):
    run, corpus = command
    logs = tmp_path / "logs"
    run("--log-dir", str(logs))
    text = (logs / "realism_report.txt").read_text()
    assert "tried 1 (every answer checked, retries included), kept 1" in text
    assert "c0_001" in text and "c1_000" in text
    out = capsys.readouterr().out
    assert out.index("run complete") < out.index("REALISM REPORT")


def test_a_run_without_a_log_folder_writes_no_report(command, monkeypatch,
                                                     tmp_path, capsys):
    run, corpus = command
    monkeypatch.setattr(batch, "CORPUS", corpus)      # the default corpus
    run()
    assert "REALISM REPORT" not in capsys.readouterr().out
    assert not list(tmp_path.rglob("realism_report.txt"))


def test_a_report_that_fails_never_fails_the_run(command, monkeypatch,
                                                 tmp_path, capsys):
    run, corpus = command

    def broken(log_dir, corpus):
        raise RuntimeError("no disk")
    monkeypatch.setattr(R, "write_report", broken)
    run("--log-dir", str(tmp_path / "logs"))
    out = capsys.readouterr().out
    assert "run complete: 1 promoted" in out
    assert "realism report not written: RuntimeError: no disk" in out
