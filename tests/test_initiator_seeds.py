"""Seeds for a generation-zero design: readme, construct and now the
property style. Every corpus row must record which three it drew, or we
cannot tell later which seed pool produced the variety."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "initiator"))

from run import build_user_msg, load_pools, make_samplers, sample_seeds
from extender.build_corpus import flatten_record


def test_load_pools_includes_styles_patterns_and_scopes():
    exemplars, constructs, readmes, styles, patterns, scopes = load_pools()
    assert len(styles) >= 6 and len(patterns) >= 6 and len(scopes) >= 5
    assert all(isinstance(s, str) and s.strip()
               for s in styles + patterns + scopes)


def test_pool_files_skip_comment_lines():
    styles, patterns, scopes = load_pools()[3:]
    assert not any(s.startswith("#") for s in styles + patterns + scopes)


def test_sample_seeds_draws_style_pattern_and_scope():
    pools = load_pools()
    (readme, construct, style, pattern, scope, ex_id,
     exemplar) = sample_seeds(*make_samplers(*pools))
    assert style in pools[3] and pattern in pools[4] and scope in pools[5]
    assert readme["repo"] and construct


def test_user_message_carries_style_pattern_and_scope():
    readme = {"repo": "x/y", "readme": "a readme"}
    msg = build_user_msg(readme, "a credit counter",
                         "check it one cycle later with $past",
                         "Universality: something always holds",
                         "after Q: a sticky arm register", {"a": 1})
    assert "check it one cycle later with $past" in msg
    assert "Universality: something always holds" in msg
    assert "after Q: a sticky arm register" in msg
    assert "a credit counter" in msg


def test_flatten_record_keeps_the_seeds_that_made_the_design():
    record = {
        "run_id": "r1", "attempt": 3,
        "readme_id": "some/repo", "construct": "a credit counter",
        "style": "check it one cycle later with $past",
        "pattern": "Universality: something always holds",
        "raw_json": json.dumps({
            "top_module": "m",
            "verilog": "module m (input wire clk);\n"
                       "  always @(*) assert (1);\nendmodule\n",
            "invariants": ["x == 1"]}),
    }
    row = flatten_record(record, 0)
    assert row["readme_id"] == "some/repo"
    assert row["construct"] == "a credit counter"
    assert row["style"] == "check it one cycle later with $past"
    assert row["pattern"] == "Universality: something always holds"


def test_scopes_pool_loads_with_a_global_majority():
    scopes = load_pools()[5]
    assert len(scopes) >= 5
    globals_ = [s for s in scopes if s.lower().startswith("globally")]
    assert len(globals_) * 2 >= len(scopes)   # half the draws, by weight


def test_scope_gate_demands_an_antecedent_when_the_property_is_armed():
    from run import scope_gate
    armed = "after Q: the property only applies once an arm bit has latched"
    assert scope_gate(armed, {"antecedents": []}) is not None
    assert scope_gate(armed, {"antecedents": ["armed"]}) is None


def test_scope_gate_lets_a_global_property_through_without_one():
    from run import scope_gate
    assert scope_gate("globally: the property applies in every state",
                      {"antecedents": []}) is None


ARMED_VERILOG = ("module m (input wire clk);\n"
                 "  reg armed = 0;\n"
                 "  always @(posedge clk) if (armed) assert (occ <= 3'd2);\n"
                 "endmodule\n")
UNARMED_VERILOG = ("module m (input wire clk);\n"
                   "  reg armed = 0;\n"
                   "  always @(posedge clk) assert (occ <= 3'd2);\n"
                   "endmodule\n")
IMPLIES_VERILOG = ("module m (input wire clk);\n"
                   "  reg armed = 0;\n"
                   "  always @(*) assert (!armed || occ <= 3'd2);\n"
                   "endmodule\n")


def test_scope_gate_rejects_an_arm_register_the_property_ignores():
    from run import scope_gate
    armed = "after Q: a sticky arm register"
    assert scope_gate(armed, {"antecedents": ["armed"],
                              "verilog": UNARMED_VERILOG}) is not None
    assert scope_gate(armed, {"antecedents": ["armed"],
                              "verilog": ARMED_VERILOG}) is None
    assert scope_gate(armed, {"antecedents": ["armed"],
                              "verilog": IMPLIES_VERILOG}) is None


def test_output_budget_can_be_lowered_for_a_local_server(monkeypatch):
    """The initiator asks for 32000 output tokens, which is the whole
    window of a locally served 32k model: prompt plus budget is then one
    token over and every call is refused."""
    import importlib
    import run as R
    monkeypatch.setenv("INITIATOR_MAX_TOKENS", "8000")
    importlib.reload(R)
    try:
        assert R.MAX_TOKENS == 8000
    finally:
        monkeypatch.delenv("INITIATOR_MAX_TOKENS")
        importlib.reload(R)
    assert R.MAX_TOKENS == 128000      # the model's own limit (9 Oct 2026)


def test_a_generation_run_records_which_model_answered(monkeypatch):
    """Both models are served under the alias 'llm', so the attempts log
    could not say whether a yield number came from the base model or the
    fine-tune. Ask the endpoint what it is really serving."""
    import types
    import run as R
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "none")

    listing = types.SimpleNamespace(data=[
        types.SimpleNamespace(id="llm", root="/home/x/runs/v1/merged")])
    assert R.endpoint_model(lambda: listing).endswith("runs/v1/merged")

    def boom():
        raise ConnectionError("nope")

    assert R.endpoint_model(boom) is None


# --- the catalog run (task 5, Oct 2026): new kinds and a size seed -------
# New kinds come from initiator/catalog_kinds.py; a size seed lifts the
# accidental smallness of our designs (95% of corpus rows have nothing
# wider than 8 bits, though no rule asked for that). Without either, a
# run must be exactly what it was before.

import types                                                     # noqa: E402
from dataclasses import dataclass                                # noqa: E402
from pathlib import Path as _P                                   # noqa: E402

from prompts import USER_TEMPLATE                                # noqa: E402

README = {"repo": "x/y", "readme": "a readme"}
SEEDS = ("a credit counter", "style s", "pattern p", "scope q", {"a": 1})


def test_without_a_size_seed_the_message_is_exactly_as_before():
    c, s, p, q, ex = SEEDS
    assert build_user_msg(README, *SEEDS) == USER_TEMPLATE.format(
        readme=README["readme"], construct=c, style=s, pattern=p, scope=q,
        exemplar=json.dumps(ex, indent=2))


def test_a_size_seed_gets_its_own_section_before_the_pattern():
    plain = build_user_msg(README, *SEEDS)
    sized = build_user_msg(README, *SEEDS, scale="Use 64-bit counters.")
    assert "## Size seed" in sized and "Use 64-bit counters." in sized
    assert sized.index("## Size seed") < sized.index("## Property pattern seed")
    start = sized.index("## Size seed")
    end = sized.index("## Property pattern seed")
    assert sized[:start] + sized[end:] == plain


def test_the_size_pool_crosses_four_widths_with_four_storage_sizes():
    import run as R
    scales = R._pool(_P(R.HERE) / "scales.txt")
    assert len(scales) == 16 and len(set(scales)) == 16
    for w in ("8-bit", "16-bit", "32-bit", "64-bit"):
        assert sum(w in s for s in scales) == 4, w
    for n in ("4 entries", "8 entries", "16 entries", "32 entries"):
        assert sum(n in s for s in scales) == 4, n


def test_load_pools_reads_a_given_kinds_file(tmp_path):
    kinds = tmp_path / "kinds.txt"
    kinds.write_text("Barrel shifter: shifts a word\nDebouncer: filters\n")
    assert load_pools(kinds)[1] == ["Barrel shifter: shifts a word",
                                    "Debouncer: filters"]
    assert len(load_pools()[1]) == 32                 # today's default


@dataclass
class _Graded:
    verdict: object
    reason: str


def _fake_run(monkeypatch, tmp_path):
    import run as R
    from oracle import NecessityVerdict
    triple = {"top_module": "m", "antecedents": ["armed"],
              "verilog": "module m(input clk);\n reg armed = 0;\n"
                         " always @(posedge clk) if (armed) assert (1);\n"
                         "endmodule\n", "invariants": ["1"]}
    monkeypatch.setattr(R, "assert_exemplar_pool", lambda ex, **kw: None)
    asked = []

    def call_model(msg, model=None):
        asked.append(model)
        return json.dumps(triple), {"input": 1, "output": 1}, "ok"
    monkeypatch.setattr(R, "call_model", call_model)
    R._asked = asked
    monkeypatch.setattr(R, "grade_triple_generated", lambda raw, **kw:
                        _Graded(NecessityVerdict.NECESSARY, "ok"))
    monkeypatch.setattr(R, "LOG_PATH", tmp_path / "main_log.jsonl")
    return R


def test_a_catalog_run_logs_its_kinds_and_sizes_to_its_own_log(monkeypatch,
                                                                tmp_path):
    R = _fake_run(monkeypatch, tmp_path)
    kinds = tmp_path / "constructs_x.txt"
    kinds.write_text("Barrel shifter: shifts a word\nDebouncer: filters\n")
    log = tmp_path / "catalog_log.jsonl"
    R.run_attempts(4, cmd="pilot", constructs_path=kinds,
                   scales_path=_P(R.HERE) / "scales.txt", log_path=log)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    scales = R._pool(_P(R.HERE) / "scales.txt")
    assert len(rows) == 4 and not (tmp_path / "main_log.jsonl").exists()
    assert all(r["constructs_file"] == "constructs_x.txt" for r in rows)
    assert all(r["scale"] in scales for r in rows)
    # balanced: two kinds over four draws, each exactly twice
    assert sorted(r["construct"] for r in rows) == sorted(
        ["Barrel shifter: shifts a word", "Debouncer: filters"] * 2)


def test_a_plain_run_records_no_size_and_no_kinds_file(monkeypatch, tmp_path):
    R = _fake_run(monkeypatch, tmp_path)
    R.run_attempts(1, cmd="pilot")
    [row] = [json.loads(l) for l in
             (tmp_path / "main_log.jsonl").read_text().splitlines()]
    assert "scale" not in row and "constructs_file" not in row


def test_pilot_takes_the_kinds_sizes_and_log_options(monkeypatch, tmp_path):
    import run as R
    seen = {}
    monkeypatch.setattr(R, "run_attempts", lambda n, **kw: seen.update(n=n, **kw))
    monkeypatch.setattr(sys, "argv", ["run.py", "pilot", "--n", "5",
                                      "--constructs", "k.txt",
                                      "--scales", "s.txt", "--log", "l.jsonl"])
    R.main()
    assert seen["n"] == 5 and seen["constructs_path"] == "k.txt"
    assert seen["scales_path"] == "s.txt" and seen["log_path"] == "l.jsonl"


def test_a_run_calls_and_records_the_model_it_was_given(monkeypatch, tmp_path):
    """Opus 5.5 (Oct 2026) is 20% cheaper than Opus 5: $4/$20 per million
    tokens against $5/$25. The catalog run may use it; every row must
    say which model wrote it."""
    R = _fake_run(monkeypatch, tmp_path)
    R.run_attempts(2, cmd="pilot", model="claude-opus-5-5")
    rows = [json.loads(l) for l in
            (tmp_path / "main_log.jsonl").read_text().splitlines()]
    assert R._asked == ["claude-opus-5-5"] * 2
    assert all(r["model"] == "claude-opus-5-5" for r in rows)


def test_without_a_model_the_run_uses_the_default_opus_5_5(monkeypatch,
                                                         tmp_path):
    R = _fake_run(monkeypatch, tmp_path)
    R.run_attempts(1, cmd="pilot")
    [row] = [json.loads(l) for l in
             (tmp_path / "main_log.jsonl").read_text().splitlines()]
    assert R._asked == ["claude-opus-5-5"] and row["model"] == "claude-opus-5-5"


def test_pilot_takes_a_model_option(monkeypatch):
    import run as R
    seen = {}
    monkeypatch.setattr(R, "run_attempts", lambda n, **kw: seen.update(n=n, **kw))
    monkeypatch.setattr(sys, "argv", ["run.py", "pilot", "--n", "2",
                                      "--model", "claude-opus-5-5"])
    R.main()
    assert seen["model"] == "claude-opus-5-5"


# --- even seed weights (8 Oct 2026) ---------------------------------------
# styles.txt, patterns.txt and scopes.txt repeat lines to weight them, and
# their own notes say the weights came from counting the evaluation sets
# (written 20 Sep, before the rule against shaping data on the tests). The
# catalog run draws every distinct line equally often instead.

def test_even_weights_keep_each_distinct_line_once_in_first_seen_order():
    import run as R
    pools = R.load_pools(even=True)
    styles, patterns, scopes = pools[3], pools[4], pools[5]
    weighted = R.load_pools()
    for even, full in ((styles, weighted[3]), (patterns, weighted[4]),
                       (scopes, weighted[5])):
        assert even == list(dict.fromkeys(full))
        assert len(even) < len(full)          # the files really repeat lines


def test_a_run_with_even_weights_says_so_in_every_row(monkeypatch, tmp_path):
    R = _fake_run(monkeypatch, tmp_path)
    R.run_attempts(2, cmd="pilot", even_seeds=True)
    rows = [json.loads(l) for l in
            (tmp_path / "main_log.jsonl").read_text().splitlines()]
    assert all(r["seed_weights"] == "even" for r in rows)
    R.run_attempts(1, cmd="pilot")
    last = json.loads((tmp_path / "main_log.jsonl").read_text()
                      .splitlines()[-1])
    assert "seed_weights" not in last                 # a plain run as before


def test_pilot_takes_an_even_seeds_option(monkeypatch):
    import run as R
    seen = {}
    monkeypatch.setattr(R, "run_attempts", lambda n, **kw: seen.update(n=n, **kw))
    monkeypatch.setattr(sys, "argv", ["run.py", "pilot", "--n", "2",
                                      "--even-seeds"])
    R.main()
    assert seen["even_seeds"] is True


# --- batch mode for the generator (8 Oct 2026): half price ---------------

def _fake_batch(monkeypatch, R, outcome=None, fail_submit=False):
    triple = json.dumps({"top_module": "m", "antecedents": ["armed"],
                         "verilog": "module m(input clk);\n reg armed = 0;\n"
                                    " always @(posedge clk) if (armed) assert (1);\n"
                                    "endmodule\n", "invariants": ["1"]})
    state = {"submitted": None}

    def submit(items):
        if fail_submit:
            raise RuntimeError("network down")
        state["submitted"] = items
        return "batch_1"

    def results(bid):
        for cid, kw in state["submitted"]:
            yield cid, (outcome or {}).get(cid, ("ok", triple, {"input": 1, "output": 1}, triple))
    monkeypatch.setattr(R.llm_client, "batch_submit", submit)
    monkeypatch.setattr(R.llm_client, "batch_status", lambda bid: ("ended", {"succeeded": 3}))
    monkeypatch.setattr(R.llm_client, "batch_results", results)
    return state


DROP = {"timestamp", "run_id", "cmd", "batch_id", "custom_id", "batch",
        "grade_wall_s", "proof_dir"}      # proof_dir holds the run id


def test_a_batch_run_judges_every_answer_exactly_like_a_direct_run(monkeypatch,
                                                                   tmp_path):
    import random
    R = _fake_run(monkeypatch, tmp_path)
    direct, batched = tmp_path / "direct.jsonl", tmp_path / "batched.jsonl"
    random.seed(7)
    R.run_attempts(3, cmd="pilot", log_path=direct)
    _fake_batch(monkeypatch, R)
    random.seed(7)
    R.run_batch(3, cmd="pilot", log_path=batched, poll_s=0)
    a = [json.loads(l) for l in direct.read_text().splitlines()]
    b = [json.loads(l) for l in batched.read_text().splitlines()]
    strip = lambda r: {k: v for k, v in r.items() if k not in DROP}
    assert [strip(r) for r in a] == [strip(r) for r in b]
    assert all(r["batch"] is True and r["batch_id"] == "batch_1" for r in b)


def test_the_question_list_is_saved_before_anything_is_paid(monkeypatch,
                                                            tmp_path):
    import pytest
    R = _fake_run(monkeypatch, tmp_path)
    _fake_batch(monkeypatch, R, fail_submit=True)
    with pytest.raises(RuntimeError):
        R.run_batch(3, cmd="pilot", log_path=tmp_path / "log.jsonl", poll_s=0)
    [manifest] = list(tmp_path.glob("batch_*.json"))
    m = json.loads(manifest.read_text())
    assert m["batch_id"] is None and len(m["attempts"]) == 3
    assert all(a["user_msg"] and a["record"]["construct"] for a in m["attempts"])
    assert not (tmp_path / "log.jsonl").exists()


def test_collecting_again_never_logs_an_answer_twice(monkeypatch, tmp_path):
    R = _fake_run(monkeypatch, tmp_path)
    _fake_batch(monkeypatch, R)
    log = tmp_path / "log.jsonl"
    R.run_batch(3, cmd="pilot", log_path=log, poll_s=0)
    [manifest] = list(tmp_path.glob("batch_*.json"))
    R.collect(manifest, poll_s=0)
    assert len(log.read_text().splitlines()) == 3


def test_a_failed_request_is_logged_as_an_error_not_dropped(monkeypatch,
                                                            tmp_path):
    R = _fake_run(monkeypatch, tmp_path)
    _fake_batch(monkeypatch, R, outcome={
        "a0001": ("error:overloaded_error", None, {"input": 0, "output": 0}, None),
        "a0002": ("refusal", None, {"input": 5, "output": 0}, "")})
    log = tmp_path / "log.jsonl"
    R.run_batch(3, cmd="pilot", log_path=log, poll_s=0)
    rows = {r["custom_id"]: r for r in map(json.loads, log.read_text().splitlines())}
    assert rows["a0000"]["verdict"] == "NECESSARY"
    assert rows["a0001"]["error"] == "batch error:overloaded_error"
    assert rows["a0002"]["verdict"] == "REFUSED"


def test_pilot_batch_and_collect_options(monkeypatch):
    import run as R
    seen = {}
    monkeypatch.setattr(R, "run_batch", lambda n, **kw: seen.update(n=n, **kw))
    monkeypatch.setattr(R, "collect", lambda m, **kw: seen.update(manifest=m))
    monkeypatch.setattr(sys, "argv", ["run.py", "pilot", "--n", "4", "--batch",
                                      "--even-seeds"])
    R.main()
    assert seen["n"] == 4 and seen["even_seeds"] is True
    monkeypatch.setattr(sys, "argv", ["run.py", "collect", "--manifest", "m.json"])
    R.main()
    assert seen["manifest"] == "m.json"


def test_collecting_a_batch_logs_every_answer_as_a_call(monkeypatch, tmp_path):
    """The one-call batch path reads answers straight from the batch, not
    through the client's call function, so collect writes the call lines."""
    R = _fake_run(monkeypatch, tmp_path)
    _fake_batch(monkeypatch, R)
    log = tmp_path / "log.jsonl"
    R.run_batch(3, cmd="pilot", log_path=log, poll_s=0)
    calls = [json.loads(l) for l in (tmp_path / "calls_log.jsonl").read_text().splitlines()]
    assert len(calls) == 3 and all(c["batch_id"] == "batch_1" for c in calls)
    assert all(c["request"]["messages"][0]["content"] for c in calls)
    assert {c["custom_id"] for c in calls} == {"a0000", "a0001", "a0002"}
