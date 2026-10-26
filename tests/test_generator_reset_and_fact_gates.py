"""Two checks before the generator's checker runs (22 Oct 2026, review
findings), and the prompt stamp that names the new rules.

The reset: with the asynchronous-reset design trait (1 in 5 designs), a
design whose property read the reset was NECESSARY for our checker and
REFUTED by EBMC 6.0, which scores the trained model. In a cycle where the
reset is held, our checker shows every register at its reset value and
EBMC shows the value the register held. The prompt now says the property,
its guard and the facts never read an asynchronous reset, and reset_gate
turns away a design that does. Until 23 Oct 2026 the rule covered every
reset; since then (Rishab chose) a synchronous reset may be read, as both
checkers read it the same way.

The facts: the prompt allowed system functions in facts, and c0_144 came
out with `occ == $countones(valid)`, which EBMC cannot read. The prompt now
asks for plain same-cycle facts, and facts_gate turns away any other.

The model and the checker are faked except in the last section, which runs
sby and EBMC on tiny designs and skips when they are missing."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import distill_log as dl                                         # noqa: E402
import prompts as P                                              # noqa: E402
import run as R                                                  # noqa: E402
from oracle.sby import sby_available                             # noqa: E402
from tests.test_initiator_pipeline import _Graded                # noqa: E402

GLOBAL = "globally: the property applies in every state"

# The reviewer's probe: written the way the asynchronous-reset trait asks,
# with a property that also says what holds while the reset is held.
PAIR = """module rst_pair (input wire clk, input wire rst_n, input wire en);
    reg [2:0] a = 3'd0;
    reg [2:0] b = 3'd0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a <= 3'd0;
            b <= 3'd0;
        end else if (en) begin
            a <= (a == 3'd4) ? 3'd0 : a + 3'd1;
            b <= (a == 3'd4) ? 3'd0 : a + 3'd1;
        end
    end
    always @(posedge clk) assert (PROP);
endmodule
"""
READS_RESET = "(rst_n || a == 3'd0) && b <= 3'd4"
LEAVES_RESET = "b <= 3'd4"


def pair(prop=LEAVES_RESET, **fields):
    triple = {"cti_explanation": "a=5, b=4 holds P, breaks a == b, steps to "
                                 "b=6",
              "cti_state": [{"signal": "a", "value": "3'd5"},
                            {"signal": "b", "value": "3'd4"}],
              "invariants": ["a == b", "a <= 3'd4"],
              "verilog": PAIR.replace("PROP", prop), "top_module": "rst_pair",
              "clock": "clk", "antecedents": [], "sanity_covers": []}
    return dict(triple, **fields)


def design(body, ports="input wire clk, input wire rst, input wire en",
           **fields):
    """A small synchronous design with `body` as its logic."""
    verilog = (f"module m ({ports});\n    reg [2:0] x = 3'd0;\n"
               f"    reg pv = 1'b0;\n{body}\nendmodule\n")
    return dict({"verilog": verilog, "top_module": "m", "clock": "clk",
                 "antecedents": [], "sanity_covers": [],
                 "invariants": ["x <= 3'd4"]}, **fields)


# makes design()'s rst an asynchronous reset: it clears x on its own edge
ASYNC = ("    always @(posedge clk or posedge rst)\n"
         "        if (rst) x <= 3'd0; else if (en) x <= x + 3'd1;\n")


# --- reset_gate: what counts as reading the reset -------------------------

def test_the_probe_is_turned_away_and_the_reason_says_where():
    reason = R.reset_gate(pair(READS_RESET))
    assert reason.startswith("the reset input rst_n is read by the property "
                             "`assert ((rst_n || a == 3'd0) && b <= 3'd4);`")
    assert "never name an asynchronous reset" in reason


def test_the_same_design_with_a_property_about_normal_operation_passes():
    assert R.reset_gate(pair(LEAVES_RESET)) is None


def test_a_guard_that_tests_the_reset_counts():
    t = design(ASYNC + "    always @(posedge clk) if (pv && !rst) "
                       "assert (x <= 3'd4);")
    reason = R.reset_gate(t)
    assert "the property's guard `(pv && !rst)`" in reason
    assert "the property `" not in reason       # the assert itself is clean


def test_a_synchronous_reset_may_be_read():
    """23 Oct 2026, Rishab chose: both checkers read a synchronous reset
    the same way, and 30 of the 917 stored designs guard their property
    with one, as here; the old rule turned all 30 away."""
    t = design("    always @(posedge clk) if (rst) x <= 3'd0;\n"
               "        else if (en) x <= x + 3'd1;\n"
               "    always @(posedge clk) if (pv && !rst) assert (x <= 3'd4);",
               antecedents=["pv && !rst"],
               invariants=["x <= 3'd4", "!rst || x <= 3'd4"])
    assert R.reset_gate(t) is None


def test_a_reset_read_through_past_counts():
    t = design(ASYNC + "    always @(posedge clk) if (pv && $past(rst)) "
                       "assert (x == 3'd0);")
    assert "$past(rst)" in R.reset_gate(t)


def test_an_assert_in_the_else_branch_of_the_reset_counts():
    t = design("    always @(posedge clk or posedge rst) begin\n"
               "        if (rst) x <= 3'd0;\n"
               "        else begin\n"
               "            x <= x + 3'd1;\n"
               "            assert (x <= 3'd4);\n"
               "        end\n"
               "    end")
    assert "the property's guard `(rst)`" in R.reset_gate(t)


def test_an_assert_inside_a_case_under_the_reset_counts():
    t = design(ASYNC + "    always @(posedge clk) begin\n"
                       "        case (rst)\n"
                       "            1'b0: assert (x <= 3'd4);\n"
                       "            default: ;\n"
                       "        endcase\n"
                       "    end")
    assert "guard `(rst)`" in R.reset_gate(t)


def test_an_assert_in_a_block_the_reset_triggers_counts():
    t = pair(LEAVES_RESET)
    t["verilog"] = t["verilog"].replace(
        "        end\n    end\n    always @(posedge clk) assert (b <= 3'd4);",
        "        end\n        assert (b <= 3'd4);\n    end")
    assert "assert (b <= 3'd4)" in t["verilog"]
    reason = R.reset_gate(t)
    assert "a block the reset triggers, `@(posedge clk or negedge rst_n)`" \
        in reason


@pytest.mark.parametrize("name", sorted(R.RESET_NAMES))
def test_every_usual_reset_name_counts_in_a_fact_when_asynchronous(name):
    edge, held = (("negedge", f"!{name}") if name.endswith("_n")
                  else ("posedge", name))
    t = design(f"    always @(posedge clk or {edge} {name})\n"
               f"        if ({held}) x <= 3'd0;\n"
               "        else x <= (x == 3'd4) ? 3'd0 : x + 3'd1;\n"
               "    always @(posedge clk) assert (x <= 3'd4);",
               ports=f"input wire clk, input wire {name}",
               invariants=[f"{name} || x == 3'd0"])
    reason = R.reset_gate(t)
    assert reason.startswith(f"the reset input {name} is read by fact")


@pytest.mark.parametrize("name", sorted(R.RESET_NAMES))
def test_a_usual_reset_name_with_no_edge_may_be_read(name):
    t = design("    always @(posedge clk) assert (x <= 3'd4);",
               ports=f"input wire clk, input wire {name}",
               invariants=[f"{name} || x == 3'd0"])
    assert R.reset_gate(t) is None


def test_an_antecedent_that_tests_the_reset_counts():
    t = design(ASYNC + "    always @(posedge clk) if (!rst) "
                       "assert (x <= 3'd4);",
               antecedents=["!rst"])
    reason = R.reset_gate(t)
    assert "antecedent `!rst`" in reason and "guard `(!rst)`" in reason


def test_a_reset_with_any_name_is_found_from_the_sensitivity_list():
    t = design("    always @(posedge clk or negedge clr_n)\n"
               "        if (!clr_n) x <= 3'd0; else if (en) x <= x + 3'd1;\n"
               "    always @(posedge clk) assert (clr_n || x == 3'd0);",
               ports="input wire clk, input wire clr_n, input wire en")
    assert R.reset_gate(t).startswith("the reset input clr_n is read by")


def test_the_clock_edge_is_not_a_reset():
    t = design("    always @(posedge i_clk) if (en) x <= x + 3'd1;\n"
               "    always @(posedge i_clk) assert (x <= 3'd4);",
               ports="input wire i_clk, input wire en", clock="i_clk",
               invariants=["x <= 3'd4", "i_clk || !i_clk"])
    assert R.reset_gate(t) is None


def test_reset_logic_elsewhere_in_the_design_is_not_the_property():
    """A module-level assertion after an asynchronous block, and an
    assertion after (not under) a test of the reset in the same block."""
    module_level = pair(LEAVES_RESET)
    module_level["verilog"] = module_level["verilog"].replace(
        "always @(posedge clk) assert (b <= 3'd4);",
        "assert property (b <= 3'd4);")
    assert R.reset_gate(module_level) is None
    sibling = design(ASYNC + "    reg seen = 1'b0;\n"
                             "    always @(posedge clk) begin\n"
                             "        if (rst) seen <= 1'b0;\n"
                             "        else if (en) seen <= 1'b1;\n"
                             "        if (pv) assert (x <= 3'd4);\n"
                             "    end")
    assert R.reset_gate(sibling) is None


def test_names_comments_and_numbers_that_only_look_like_a_reset_pass():
    t = design(ASYNC + "    reg [2:0] rst_count = 3'd0;\n"
               "    reg [2:0] burst = 3'd0;\n"
               "    // the property never reads rst: \"rst\" stays out\n"
               "    always @(posedge clk) /* not rst */ "
               "if (pv && rst_count == 3'd0) assert (burst <= 3'd4);",
               antecedents=["pv && rst_count == 3'd0"],
               invariants=["burst <= 3'd4", "rst_count <= 3'd7",
                           "x != 3'h7"])
    assert R.reset_gate(t) is None


def test_a_block_the_reader_cannot_follow_errs_towards_turning_away():
    """A block that never closes cannot be read statement by statement;
    everything from its start counts as what the assertion sits under."""
    t = design(ASYNC + "    always @(posedge clk) begin\n"
                       "        if (rst) x <= 3'd0;\n"
                       "        assert (x <= 3'd4);\n")
    assert "the property's block, which names the reset" in R.reset_gate(t)


def test_odd_fields_never_crash_the_gates():
    for odd in ({}, {"verilog": None, "invariants": "x == 1"},
                {"verilog": "", "antecedents": [3, None]},
                {"verilog": "module m; assert (", "invariants": [None]}):
        assert R.reset_gate(odd) is None
        assert R.facts_gate(odd) is None


# --- facts_gate --------------------------------------------------------------

def test_a_fact_with_a_system_function_is_turned_away_by_name():
    t = pair(invariants=["a == b", "occ == $countones(valid)",
                         "!pv || b == $past(a)"])
    reason = R.facts_gate(t)
    assert reason.startswith("facts that are not plain same-cycle expressions:"
                             " `occ == $countones(valid)` (system function "
                             "$countones); `!pv || b == $past(a)` (system "
                             "function $past)")
    assert "`a == b`" not in reason
    assert "a count of set bits is a sum (v[0] + v[1] + v[2])" in reason


def test_every_form_the_scoring_checker_cannot_read_is_named():
    reason = R.facts_gate(pair(invariants=[
        "forall k in 0..3: !v[k]", "u_fifo.count <= 4'd8"]))
    assert "(compact forall form)" in reason and "(dotted name)" in reason


def test_plain_facts_pass():
    assert R.facts_gate(pair()) is None
    assert R.facts_gate(pair(invariants=[
        "occ == valid[0] + valid[1] + valid[2]", "a$b == 1'b0"])) is None


# --- judge: before the checker, in order, never calling it ----------------

def _no_checker(monkeypatch):
    def grade(raw, **kw):
        raise AssertionError("a design turned away must never be graded")
    monkeypatch.setattr(R, "grade_triple_generated", grade)


def test_judge_turns_both_away_before_the_checker(monkeypatch, tmp_path):
    _no_checker(monkeypatch)
    for triple, verdict in ((pair(READS_RESET), "READS_RESET"),
                            (pair(invariants=["occ == $countones(v)"]),
                             "FACTS_NOT_PLAIN")):
        j = R.judge(json.dumps(triple), "end_turn", GLOBAL, grade=True,
                    proof_root=tmp_path, spin=False)
        assert j["verdict"] == verdict and j["reason"] and j["result"] is None


def test_the_gates_run_in_order_scope_reset_facts(monkeypatch):
    _no_checker(monkeypatch)
    both = pair(READS_RESET, invariants=["rst_n || $onehot(a)"])
    armed = "after Q: the property is required once Q has happened"
    assert R.judge(json.dumps(both), "end_turn", armed, grade=False)[
        "verdict"] == "SCOPE_UNARMED"
    assert R.judge(json.dumps(both), "end_turn", GLOBAL, grade=False)[
        "verdict"] == "READS_RESET"
    assert R.GATE_VERDICTS == ("SCOPE_UNARMED", "READS_RESET",
                               "FACTS_NOT_PLAIN")


def test_a_design_that_passes_every_gate_reaches_the_checker(monkeypatch,
                                                             tmp_path):
    from oracle import NecessityVerdict
    seen = []

    def grade(raw, **kw):
        seen.append(json.loads(raw)["top_module"])
        return _Graded(NecessityVerdict.NECESSARY, "fine")
    monkeypatch.setattr(R, "grade_triple_generated", grade)
    j = R.judge(json.dumps(pair()), "end_turn", GLOBAL, grade=True,
                proof_root=tmp_path, spin=False)
    assert j["verdict"] == "NECESSARY" and seen == ["rst_pair"]


def test_a_turned_away_design_is_a_failure_to_learn_from():
    for verdict in R.GATE_VERDICTS:
        assert dl.outcome_of(verdict, "some reason") == "fail"


def _answers(monkeypatch, triples):
    """The model answers with these triples in turn."""
    left = list(triples)
    questions = []

    def call_model(user_msg, model=None, system=None, schema=None):
        questions.append(user_msg)
        return json.dumps(left.pop(0)), {"input": 1, "output": 1}, "end_turn"
    monkeypatch.setattr(R, "call_model", call_model)
    monkeypatch.setattr(R, "assert_exemplar_pool", lambda ex, **kw: None)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    return questions


def test_a_run_logs_the_verdict_and_reason_like_the_scope_gate(monkeypatch,
                                                               tmp_path,
                                                               capsys):
    _no_checker(monkeypatch)
    _answers(monkeypatch, [pair(READS_RESET),
                           pair(invariants=["occ == $countones(v)"])])
    monkeypatch.setattr(R, "scope_gate", lambda scope, triple: None)
    log, distill = tmp_path / "log.jsonl", tmp_path / "distill.jsonl"
    R.run_attempts(2, cmd="pilot", log_path=log, distill_path=distill)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r["verdict"] for r in rows] == ["READS_RESET", "FACTS_NOT_PLAIN"]
    assert "rst_n" in rows[0]["reason"] and "$countones" in rows[1]["reason"]
    out = capsys.readouterr().out
    assert "[0] READS_RESET - the reset input rst_n" in out
    assert "'FACTS_NOT_PLAIN': 1, 'READS_RESET': 1" in out
    assert [e["outcome"] for e in dl.read(distill)] == ["fail", "fail"]


def test_a_repair_round_is_told_what_to_fix(monkeypatch, tmp_path):
    from oracle import NecessityVerdict
    questions = _answers(monkeypatch, [
        pair(invariants=["a == b", "$countones(a) <= 3"]), pair()])
    monkeypatch.setattr(R, "scope_gate", lambda scope, triple: None)
    monkeypatch.setattr(R, "grade_triple_generated", lambda raw, **kw:
                        _Graded(NecessityVerdict.NECESSARY, "fine"))
    log = tmp_path / "log.jsonl"
    [row] = R.run_pipeline(1, repairs=1, workers=1, cmd="pilot", log_path=log,
                           distill_path=tmp_path / "distill.jsonl")
    assert [h["verdict"] for h in row["history"]] == ["FACTS_NOT_PLAIN",
                                                      "NECESSARY"]
    assert "Verdict: FACTS_NOT_PLAIN" in questions[1]
    assert "`$countones(a) <= 3` (system function $countones)" in questions[1]


# --- the prompt and the stamp -------------------------------------------------

def squash(text):
    return " ".join(text.split())


def test_the_prompt_states_both_rules():
    text = squash(P.SYSTEM_PROMPT)
    assert ("The property, its guard (the antecedents) and the invariants "
            "never read an asynchronous reset") in text
    assert ("A synchronous reset, tested only inside `always @(posedge "
            "clk)`, may guard the property.") in text
    assert ("Each clause is a same-cycle boolean expression over the design's "
            "signals, with no system function") in text
    assert "Write a count of set bits as a sum (v[0] + v[1] + v[2])" in text
    # the style seed's rule still points a named system function at the
    # property
    assert "use it in the property, not in the invariants" in \
        squash(P.USER_TEMPLATE)


def test_every_asynchronous_reset_trait_line_says_the_same():
    lines = [l for l in R.load_traits() if "negedge rst_n" in l]
    assert len(lines) == 6
    for line in lines:
        assert line.endswith("The property, its guard and the invariants "
                             "never read rst_n: they say what holds in "
                             "normal operation, not while reset is held.")


def test_the_stamp_names_each_rule_the_prompt_holds():
    assert P.prompt_version() == \
        "nocap+size-params+plain-facts+no-async-reset-reads" \
        "+explanation-after-design"
    size = "- Declare the design's sizes as parameters, as real hardware does"
    plain = ("Each clause is a same-cycle\n  boolean expression over the "
             "design's signals, with no system function")
    reset = "- The property ... never read an asynchronous reset ..."
    assert P.prompt_version(size) == "nocap+size-params"
    assert P.prompt_version("- Keep the module under 200 lines.\n" + plain) \
        == "200cap+plain-facts"
    assert P.prompt_version(reset + size) == \
        "nocap+size-params+no-async-reset-reads"
    # the first wording, every reset (22 Oct 2026), wrote no rows and has no
    # stamp
    assert P.prompt_version("- The property ... never read a reset input") \
        == "nocap"
    # every earlier prompt keeps the stamp it had
    assert P.prompt_version("You write SystemVerilog modules.\n") == "nocap"
    assert P.prompt_version("- Keep the module under 800 lines.") == "800cap"


def test_rows_carry_the_new_stamp(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    _, _, plans = R.plan_attempts(2, cmd="test", grade=False, model=R.MODEL,
                                  constructs_path=None, scales_path=None,
                                  even_seeds=False)
    assert {r["prompt_version"] for r, *_ in plans} == \
        {"nocap+size-params+plain-facts+no-async-reset-reads"
         "+explanation-after-design"}


# --- the two checkers on asynchronous-reset designs (sby and EBMC) ----------
# Part c of the finding: the probe is NECESSARY for our checker and REFUTED
# by EBMC, which is why the gate exists; designs whose property and facts
# leave the reset alone get the same verdict from both. 22 Oct 2026, six
# such designs agreed (the probe without the reset, a shadow register
# without a reset, a reset value other than the initial one, a limit loaded
# from an input at reset, and the burst sender of
# test_generator_design_traits.py with both resets); two are kept here.

EBMC_BIN = os.environ.get("EBMC_PATH")
requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH - activate hwtools first")
requires_ebmc = pytest.mark.skipif(
    not (EBMC_BIN and Path(EBMC_BIN).is_file()),
    reason="needs EBMC_PATH pointing at the ebmc binary")

# A limit loaded from an input, at the asynchronous reset and on a load
# command: the facts must hold for every limit.
LOADED = """module loaded_limit (input wire clk, input wire rst_n, input wire en,
                     input wire load, input wire [3:0] cfg_limit);
    reg [3:0] limit = 4'd5;
    reg [3:0] cnt   = 4'd0;
    reg [3:0] left  = 4'd5;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            limit <= cfg_limit;
            cnt   <= 4'd0;
            left  <= cfg_limit;
        end else if (load) begin
            limit <= cfg_limit;
            cnt   <= 4'd0;
            left  <= cfg_limit;
        end else if (en && left != 4'd0) begin
            cnt  <= cnt + 4'd1;
            left <= left - 4'd1;
        end
    always @(posedge clk) assert (cnt <= limit);
endmodule
"""
LOADED_TRIPLE = {"cti_explanation": "cnt 3, limit 3, left 2 holds P, breaks "
                                    "the sum, and one step makes cnt 4",
                 "cti_state": [{"signal": "cnt", "value": "4'd3"},
                               {"signal": "limit", "value": "4'd3"},
                               {"signal": "left", "value": "4'd2"}],
                 "invariants": ["cnt + left == limit"], "verilog": LOADED,
                 "top_module": "loaded_limit", "clock": "clk",
                 "antecedents": [], "sanity_covers": []}
_ASSERT = re.compile(r"^\s*always @\(posedge (\w+)\) assert \((.*)\);\s*$",
                     re.M)


def ebmc(triple, lemmas, mode, workdir):
    """EBMC on the design in the scoring file's form: the assertion becomes
    `property prop` and the facts are added as lemmas by ebmc_eval, with
    its own commands. Returns (verdict, output)."""
    import ebmc_eval
    m = _ASSERT.search(triple["verilog"])
    source = triple["verilog"][:m.start()] + triple["verilog"][m.end():]
    at = source.rindex("endmodule")
    source = (source[:at] + f"  property prop;\n    @(posedge {m.group(1)}) "
              f"({m.group(2)});\n  endproperty\n" + source[at:])
    path = Path(workdir) / f"{triple['top_module']}_{mode}_{len(lemmas)}.sv"
    path.write_text(ebmc_eval.build_variant(source, lemmas, mode))
    cmd = ebmc_eval.ebmc_command(path, mode, rst=False, ebmc=EBMC_BIN,
                                 source=source)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=300)
    return ebmc_eval.parse_verdict(r.stdout, r.stderr, mode), r.stdout


def ours(triple, workdir):
    from oracle import grade_triple_generated
    return grade_triple_generated(json.dumps(triple), workdir_root=workdir,
                                  **R.GRADE_KWARGS).verdict.name


@requires_sby
@requires_ebmc
def test_the_probe_splits_the_checkers_and_never_reaches_ours(tmp_path):
    probe = pair(READS_RESET)
    assert ours(probe, tmp_path / "ours") == "NECESSARY"
    verdict, out = ebmc(probe, [], "timing_without_lemmas", tmp_path)
    assert verdict == "CEX" and "REFUTED" in out
    # the generator turns it away before our checker ever sees it
    j = R.judge(json.dumps(probe), "end_turn", GLOBAL, grade=True,
                proof_root=tmp_path / "judge", spin=False)
    assert j["verdict"] == "READS_RESET"
    assert not (tmp_path / "judge").exists()


@requires_sby
@requires_ebmc
@pytest.mark.parametrize("name,triple", [
    ("the probe about normal operation", pair(LEAVES_RESET)),
    ("a limit loaded from an input at the reset", LOADED_TRIPLE)])
def test_designs_that_leave_the_reset_alone_get_one_verdict(tmp_path, name,
                                                            triple):
    assert R.reset_gate(triple) is None and R.facts_gate(triple) is None
    assert ours(triple, tmp_path / "ours") == "NECESSARY"
    # EBMC: true over 30 steps, closed by the facts in one step (the
    # scoring rule), and not closed by the property alone
    assert ebmc(triple, [], "timing_without_lemmas", tmp_path)[0] != "CEX"
    assert ebmc(triple, triple["invariants"], "one_inductive_with_prop",
                tmp_path)[0] == "PROVEN"
    assert ebmc(triple, [], "one_inductive_with_prop", tmp_path)[0] != \
        "PROVEN"
