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
    assert R.MAX_TOKENS == 32000


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
