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
