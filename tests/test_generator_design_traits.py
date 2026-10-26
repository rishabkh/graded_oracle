"""Design traits and size parameters for the generator (21 Oct 2026).

Our designs were written unlike real hardware. Measured 15 Oct 2026 over
560 modules of open-source hardware (PULP common_cells, lowRISC Ibex,
OpenTitan prims and peripherals, alexforencich verilog-axis and
verilog-axi, ZipCPU wb2axip) against our 252 newest designs: parameters
87% against about 40%; valid/ready handshake 49% against 6%; asynchronous
reset 21% (Ibex 77%) against 0%; enum for named states 5% (Ibex 20%)
against 0%; and no design of ours ever loaded a register from an input.
Rishab approved the fix: a weighted design-trait seed, one per design,
and a rule in the prompt to declare sizes as parameters with defaults of
4 to 8 entries (at most 16) so the proof stays fast. The model is faked
everywhere; one test runs the real checker on a tiny design that has
every new trait, and skips when sby is not on PATH."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import distill_log as dl                                         # noqa: E402
import prompts as P                                              # noqa: E402
import run as R                                                  # noqa: E402
from oracle import NecessityVerdict, Tier, grade_triple_generated  # noqa: E402
from oracle.sby import sby_available                             # noqa: E402
from tests.test_initiator_pipeline import Fake                   # noqa: E402

FILE = ROOT / "initiator" / "design_traits.txt"
LINES = R.load_traits()

requires_sby = pytest.mark.skipif(
    not sby_available(), reason="sby not on PATH - activate hwtools first")


def squash(text):
    return " ".join(text.split())


def _share(marker):
    return sum(marker in line for line in LINES)


# --- the seed file -------------------------------------------------------------

def test_the_trait_pool_loads_twenty_six_weighted_lines_without_comments():
    assert len(LINES) == 26
    assert not any(line.startswith("#") for line in LINES)
    assert len(set(LINES)) == 8              # repeated lines carry the weights


def test_the_traits_are_weighted_like_the_decision():
    """Handshake on half the designs; independently, an asynchronous reset
    on 3 in 13, named states on 3 in 13, a register loaded from an input on
    1 in 13 (the measured 7.7%, Rishab's choice on 23 Oct 2026), and plain
    synchronous style on the rest."""
    assert _share("back-pressure") == 13
    assert _share("negedge rst_n") == 6
    assert _share("typedef enum") == 6
    assert _share("load command") == 2
    others = ("negedge rst_n", "typedef enum", "load command")
    plain = [l for l in LINES if not any(m in l for m in others)]
    assert len(plain) == 12                  # 6 in 13: the rest
    # the two choices are crossed: with or without a handshake, the same
    # mix of the other traits
    for marker in others:
        with_hs = sum(marker in l and "back-pressure" in l for l in LINES)
        assert with_hs * 2 == _share(marker), marker
    # at most one trait besides the handshake on any line
    assert all(sum(m in l for m in others) <= 1 for l in LINES)


def test_every_trait_keeps_the_rules_that_make_proofs_sound():
    for line in set(LINES):
        assert "assume" not in line and "->" not in line, line
        if "negedge rst_n" in line or "typedef enum" in line \
                or "load command" in line:
            assert "initial value" in line, line
        if "back-pressure" in line or "negedge rst_n" in line:
            assert "idle condition" in line, line
        if "load command" in line:
            assert "every value it can take" in line, line


def test_the_file_records_where_the_weights_came_from():
    """22 Oct 2026, review finding: the header said every weight came from
    the real-design numbers, but the loaded-register weight had none behind
    it and the named-states weight follows Ibex, not all 560 modules. It
    now says so, with the loaded-register share measured on those modules."""
    head = squash(" ".join(l.lstrip("#") for l in
                           FILE.read_text().splitlines() if l.startswith("#")))
    for word in ("560 modules", "252 newest", "Ibex", "OpenTitan",
                 "common_cells", "verilog-axis", "wb2axip", "15 Oct 2026",
                 "21 Oct 2026", "22 Oct 2026", "20 of 560 modules (3.6%)",
                 "43 (7.7%)", "182 (32.5%)",
                 "follows Ibex (20%) rather than the 5%",
                 "23 Oct 2026, Rishab chose the measured shares",
                 "1 in 13 (7.7%"):
        assert word in head, word
    assert "never from the test sets" not in head


def test_the_file_never_cites_the_test_sets():
    """Training data is justified from real hardware or our own
    measurements, never from the benchmark (project rule)."""
    text = FILE.read_text().lower()
    for name in ("riscv-formal", "large_lemma", "lemma_miners", "technion",
                 "hwmcc", "benchmark", "evaluation set"):
        assert name not in text, name


# --- the prompts ---------------------------------------------------------------

README = {"repo": "x/y", "readme": "a readme"}
SEEDS = ("a credit counter", "style s", "pattern p", "scope q", {"a": 1})


def test_a_trait_gets_its_own_section_before_the_pattern():
    plain = R.build_user_msg(README, *SEEDS)
    trait = LINES[0]
    msg = R.build_user_msg(README, *SEEDS, trait=trait)
    assert "## Design trait seed" in msg and trait in msg
    start = msg.index("## Design trait seed")
    end = msg.index("## Property pattern seed")
    assert start < end
    assert msg[:start] + msg[end:] == plain    # nothing else changed


def test_the_trait_follows_the_size_seed_when_both_are_drawn():
    msg = R.build_user_msg(README, *SEEDS, scale="Use 64-bit counters.",
                           trait="valid/ready handshake: T")
    assert msg.index("## Size seed") < msg.index("## Design trait seed") \
        < msg.index("## Property pattern seed")


def test_without_a_trait_the_question_is_exactly_as_before():
    """Old rows rebuilt from their seeds (training/rebuild_old_examples.py)
    had no trait, so their question must not grow one."""
    c, s, p, q, ex = SEEDS
    assert R.build_user_msg(README, *SEEDS, trait=None) == \
        P.USER_TEMPLATE.format(readme=README["readme"], construct=c, style=s,
                               pattern=p, scope=q,
                               exemplar=json.dumps(ex, indent=2))


def test_the_trait_section_restates_the_rules_it_must_not_break():
    text = squash(P.TRAIT_SECTION)
    assert "initial value on every state register" in text
    assert "idle condition" in text and "no assume" in text


@pytest.mark.parametrize("trait", sorted(set(LINES)),
                         ids=lambda t: t.split(":")[0])
def test_every_trait_reaches_every_question_word_for_word(trait):
    """Some traits hold braces ({IDLE, BUSY, DONE}); filling a template
    must not trip on them."""
    assert trait in R.build_user_msg(README, *SEEDS, trait=trait)
    idea = P.IDEA_TEMPLATE.format(repo="r", readme="t", construct="c",
                                  scale="s", pattern="p", scope="q",
                                  style="y", trait=trait)
    implement = P.IMPLEMENT_TEMPLATE.format(idea="i", scale="s", trait=trait,
                                            exemplar="{}")
    assert trait in idea and trait in implement


def test_the_prompt_asks_for_sizes_as_parameters_with_modest_defaults():
    text = squash(P.SYSTEM_PROMPT)
    assert "`parameter` with a default" in text
    assert "4 to 8, never more than 16, so the proof stays fast" in text
    # a size seed (scales_real_hardware.txt, up to 256 entries) still wins
    assert "the seed's size is the default instead" in text
    assert "small" not in text.lower()       # 9 Oct 2026: no "small" designs


def test_every_rule_that_keeps_proofs_sound_is_still_in_the_prompt():
    text = squash(P.SYSTEM_PROMPT)
    for rule in ("Give every state register an `initial` value",
                 "The design MUST have an enable or idle condition",
                 "One property per module",
                 "Never write `assume`",
                 "Never write `->` as logical implication"):
        assert rule in text, rule
    assert "use it in the property, not in the invariants" in \
        squash(P.USER_TEMPLATE)


def test_two_step_without_a_size_seed_asks_for_the_same_defaults():
    assert "4 to 8, never more than 16" in P.NO_SIZE_SEED
    assert "small" not in P.NO_SIZE_SEED.lower()


# --- a run ---------------------------------------------------------------------

def _plan(n, monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    def no_call(*a, **kw):
        raise AssertionError("planning a run must not call the model")
    monkeypatch.setattr(R, "call_model", no_call)
    _, _, plans = R.plan_attempts(n, cmd="test", grade=False, model=R.MODEL,
                                  constructs_path=None, scales_path=None,
                                  even_seeds=False)
    return plans


def test_a_run_deals_every_trait_line_evenly(monkeypatch):
    """Balanced turns, like the other seeds: over 52 designs every one of
    the 26 lines is drawn exactly twice, so the shares are exact."""
    plans = _plan(52, monkeypatch)
    drawn = Counter(record["trait"] for record, *_ in plans)
    assert drawn == Counter({line: 2 * k for line, k in Counter(LINES).items()})
    assert sum("back-pressure" in t for t in drawn.elements()) == 26


def test_planning_a_run_needs_no_model_call_and_puts_the_trait_in_each_question(
        monkeypatch):
    for record, user_msg, _, _ in _plan(5, monkeypatch):
        assert record["trait"] in LINES
        assert record["trait"] in user_msg
        assert "## Design trait seed" in user_msg


def test_even_seeds_leave_the_trait_weights_alone(monkeypatch):
    """--even-seeds flattens style, pattern and scope, whose weights were
    counted on the evaluation sets; the trait weights were not, so they
    stay (the file's header says where each came from)."""
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    _, _, plans = R.plan_attempts(26, cmd="test", grade=False, model=R.MODEL,
                                  constructs_path=None, scales_path=None,
                                  even_seeds=True)
    assert Counter(r["trait"] for r, *_ in plans) == Counter(LINES)


def test_the_check_contract_command_still_runs_without_the_model(monkeypatch,
                                                                 capsys):
    """run.py has no --dry; its no-model command is check-contract."""
    def no_call(*a, **kw):
        raise AssertionError("check-contract must not call the model")
    monkeypatch.setattr(R, "call_model", no_call)
    monkeypatch.setattr(sys, "argv", ["run.py", "check-contract"])
    R.main()
    assert "contract ok" in capsys.readouterr().out


def test_every_attempt_record_carries_the_trait_its_question_had(monkeypatch,
                                                                 tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"] * 3)
    f.R.run_attempts(3, cmd="pilot", log_path=f.log, distill_path=f.distill)
    rows = [json.loads(l) for l in f.log.read_text().splitlines()]
    assert len(rows) == 3
    for row, (_, question) in zip(rows, f.prompts):
        assert row["trait"] in LINES and row["trait"] in question
    # the training example keeps it among the seeds that made the question
    examples = dl.read(f.distill)
    assert [e["arguments"]["trait"] for e in examples] == \
        [r["trait"] for r in rows]


def test_a_batch_question_list_keeps_the_trait(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, [])

    def submit(items):
        raise RuntimeError("network down")      # nothing is sent or paid
    monkeypatch.setattr(f.R.llm_client, "batch_submit", submit)
    with pytest.raises(RuntimeError):
        f.R.run_batch(2, cmd="pilot", log_path=tmp_path / "log.jsonl",
                      poll_s=0)
    [manifest] = list(tmp_path.glob("batch_*.json"))
    for a in json.loads(manifest.read_text())["attempts"]:
        assert a["record"]["trait"] in a["user_msg"]


def test_two_step_gives_the_trait_to_the_idea_and_to_the_design(monkeypatch,
                                                                tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"])
    [row], ex = f.run(mode="two-step")
    (idea_system, idea_q), (_, design_q) = f.prompts[0], f.prompts[1]
    assert "design trait" in idea_system
    assert row["trait"] in idea_q and row["trait"] in design_q
    # no size seed drawn: both steps get the 4-to-8 default, not "ordinary"
    assert P.NO_SIZE_SEED in idea_q and P.NO_SIZE_SEED in design_q
    idea = next(e for e in ex if e["prompt"] == "idea")
    assert idea["arguments"]["trait"] == row["trait"]


# --- the checker on every new trait at once (real sby) ------------------------
# A burst sender: a request is accepted on a valid/ready handshake, then the
# burst goes out one beat per output handshake, with back-pressure. The burst
# length is loaded from an input, so the proof must hold for every length.
# The fact the property needs ties the two beat counters to that length.

BURST_VERILOG = r"""module burst_ctl #(
    parameter W = 3
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         load,
    input  wire [W-1:0] cfg_len,
    input  wire         in_valid,
    output wire         in_ready,
    output wire         out_valid,
    input  wire         out_ready
);
    typedef enum logic [1:0] {IDLE = 2'd0, BUSY = 2'd1} state_t;
    state_t     state      = IDLE;
    reg [W-1:0] burst_len  = {W{1'b0}};
    reg [W-1:0] beats_left = {W{1'b0}};
    reg [W-1:0] beats_sent = {W{1'b0}};

    assign in_ready  = (state == IDLE) && (burst_len != 0) && !load;
    assign out_valid = (state == BUSY);
    wire start = in_valid && in_ready;
    wire beat  = out_valid && out_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            burst_len  <= RESET_LEN;
            beats_left <= {W{1'b0}};
            beats_sent <= {W{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    if (load)
                        burst_len <= cfg_len;
                    else if (start) begin
                        state      <= BUSY;
                        beats_left <= burst_len;
                        beats_sent <= {W{1'b0}};
                    end
                end
                BUSY: if (beat) begin
                    beats_left <= beats_left - 1'b1;
                    beats_sent <= beats_sent + 1'b1;
                    if (beats_left == 1) state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    always @(posedge clk)
        if (state == BUSY)
            assert (beats_sent < burst_len);
endmodule
"""


@requires_sby
@pytest.mark.parametrize("reset_len", [
    "{W{1'b0}}",          # loaded on a load command only
    "cfg_len",            # also loaded from the input at the asynchronous reset
], ids=["loaded_on_a_load_command", "also_loaded_at_the_async_reset"])
def test_the_checker_handles_a_design_with_every_new_trait(tmp_path,
                                                           reset_len):
    triple = {
        "cti_explanation": "BUSY with burst_len 1, beats_sent 0 and "
                           "beats_left 2 holds the property but breaks the "
                           "fact; held with out_ready low, one beat makes "
                           "beats_sent 1.",
        "cti_state": [{"signal": "state", "value": "BUSY"},
                      {"signal": "burst_len", "value": "3'd1"},
                      {"signal": "beats_left", "value": "3'd2"},
                      {"signal": "beats_sent", "value": "3'd0"}],
        "invariants": ["state != BUSY || (beats_left != 0 && "
                       "beats_left <= burst_len && "
                       "beats_sent == burst_len - beats_left)"],
        "verilog": BURST_VERILOG.replace("RESET_LEN", reset_len),
        "top_module": "burst_ctl", "clock": "clk",
        "antecedents": ["state == BUSY"],
        "sanity_covers": ["beats_sent == 3'd2"],
    }
    r = grade_triple_generated(json.dumps(triple), workdir_root=tmp_path,
                               **R.GRADE_KWARGS)
    assert r.verdict is NecessityVerdict.NECESSARY, r.reason
    cover = next(ev for ev in r.with_invariants.runs if ev.mode == "cover")
    assert {"state == BUSY", "beats_sent == 3'd2"} <= set(cover.reached_covers)
    assert r.without_invariants.tier is Tier.NOT_INDUCTIVE
    # the reset really reached the proof as an asynchronous one (sby turns
    # it into a clocked model with async2sync), not dropped on the way
    prove = r.with_invariants.runs[0]
    logs = "\n".join(p.read_text(errors="replace")
                     for p in prove.workdir.rglob("*.log"))
    assert re.search(r"\(\$al?dff\): A(RST|LOAD)=\\rst_n", logs)
    # everything the checker kept is under the test's own folder
    assert all(str(ev.workdir).startswith(str(tmp_path))
               for leg in (r.with_invariants, r.without_invariants)
               for ev in leg.runs)
