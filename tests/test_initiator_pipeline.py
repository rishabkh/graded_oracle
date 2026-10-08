"""Two-step and self-repairing generation (8 Oct 2026), Formal Disco's
idea -> implement and iterative generate, on our planted-triple contract.
The model and the checker are faked: no calls, no sby, no money."""
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "initiator"))

import distill_log as dl                                         # noqa: E402

# armed and conditional, so the scope gate passes whichever scope is drawn
TRIPLE = {"top_module": "m", "antecedents": ["armed"], "sanity_covers": [],
          "verilog": "module m(input clk);\n reg armed = 0;\n"
                     " always @(posedge clk) if (armed) assert (1);\nendmodule\n",
          "invariants": ["1"]}


@dataclass
class _Graded:
    verdict: object
    reason: str


class Fake:
    """The model answers by schema; the checker returns verdicts in order."""
    def __init__(self, monkeypatch, tmp_path, verdicts, idea="IDEA TEXT"):
        import run as R
        from oracle import NecessityVerdict
        self.R, self.prompts, self.calls = R, [], []
        self.verdicts = list(verdicts)
        self.n = 0
        self.lock = threading.Lock()

        def call_model(user_msg, model=None, system=None, schema=None):
            with self.lock:
                self.prompts.append((system, user_msg))
                if schema is R.IDEA_SCHEMA:
                    return json.dumps({"idea": idea}), {"input": 1, "output": 1}, "ok"
                self.n += 1
                return (json.dumps(dict(TRIPLE, invariants=[f"v{self.n}"])),
                        {"input": 2, "output": 3}, "ok")

        def grade(raw, **kw):
            with self.lock:
                v = self.verdicts.pop(0) if self.verdicts else "NECESSARY"
            return _Graded(NecessityVerdict[v], f"because {v}")
        self.real_call_model = R.call_model
        monkeypatch.setattr(R, "call_model", call_model)
        monkeypatch.setattr(R, "grade_triple_generated", grade)
        monkeypatch.setattr(R, "assert_exemplar_pool", lambda ex: None)
        self.log = tmp_path / "log.jsonl"
        self.distill = tmp_path / "distill.jsonl"

    def run(self, n=1, **kw):
        self.R.run_pipeline(n, cmd="pilot", log_path=self.log,
                            distill_path=self.distill, workers=kw.pop("workers", 1),
                            **kw)
        rows = [json.loads(l) for l in self.log.read_text().splitlines()]
        return rows, dl.read(self.distill)


def test_self_repair_hands_back_the_findings_and_keeps_the_final_design(
        monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN", "NECESSARY"])
    [row], ex = f.run(repairs=2)
    assert row["verdict"] == "NECESSARY" and row["repair_attempts"] == 1
    assert [h["verdict"] for h in row["history"]] == ["NOT_PROVEN", "NECESSARY"]
    assert json.loads(row["raw_json"])["invariants"] == ["v2"]   # the repaired one
    repair_prompt = f.prompts[1][1]
    assert "because NOT_PROVEN" in repair_prompt and '"v1"' in repair_prompt
    kinds = [e["prompt"] for e in ex]
    assert sorted(kinds) == ["generate", "repair"]
    gen = next(e for e in ex if e["prompt"] == "generate")
    assert gen["outcome"] == "success" and json.loads(gen["response"])["invariants"] == ["v2"]
    # the generate example is asked with the ORIGINAL question, as Formal Disco does
    assert gen["metadata"]["messages"][1]["content"] == f.prompts[0][1]
    assert gen["metadata"]["repair_attempts"] == 1


def test_repairs_stop_at_the_limit(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN"] * 5)
    [row], ex = f.run(repairs=2)
    assert row["verdict"] == "NOT_PROVEN" and row["repair_attempts"] == 2
    assert len(row["history"]) == 3
    assert [e["outcome"] for e in ex if e["prompt"] == "repair"] == ["fail", "fail"]
    assert next(e for e in ex if e["prompt"] == "generate")["outcome"] == "fail"


def test_a_first_try_success_asks_for_no_repair(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"])
    [row], ex = f.run(repairs=2)
    assert row["repair_attempts"] == 0 and len(f.prompts) == 1
    assert [e["prompt"] for e in ex] == ["generate"]


def test_two_step_saves_the_idea_and_its_implementation(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"])
    [row], ex = f.run(mode="two-step")
    assert row["mode"] == "two-step" and row["idea"] == "IDEA TEXT"
    assert f.prompts[1][1].startswith("## Design idea\n\nIDEA TEXT")
    by = {e["prompt"]: e for e in ex}
    assert set(by) == {"idea", "implement"}
    assert by["idea"]["response"] == "IDEA TEXT" and by["idea"]["outcome"] == "success"
    assert by["idea"]["arguments"]["construct"] == row["construct"]
    assert by["idea"]["arguments"]["readme"]                     # the full README text
    assert by["implement"]["arguments"]["idea"] == "IDEA TEXT"


def test_an_idea_whose_implementation_fails_counts_as_failed(monkeypatch, tmp_path):
    """Formal Disco keeps an idea only when its implementation verified."""
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN"])
    [row], ex = f.run(mode="two-step")
    assert {e["prompt"]: e["outcome"] for e in ex} == {"idea": "fail",
                                                       "implement": "fail"}


def test_one_step_without_repairs_is_saved_as_initiate(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"])
    [row], ex = f.run()
    assert [e["prompt"] for e in ex] == ["initiate"]
    assert row["mode"] == "initiate" and row["verdict"] == "NECESSARY"


def test_checking_runs_only_a_few_at_a_time(monkeypatch, tmp_path):
    import time
    f = Fake(monkeypatch, tmp_path, [])
    live, most = [0], [0]
    lock = threading.Lock()
    from oracle import NecessityVerdict

    def slow(raw, **kw):
        with lock:
            live[0] += 1
            most[0] = max(most[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return _Graded(NecessityVerdict.NECESSARY, "ok")
    monkeypatch.setattr(f.R, "grade_triple_generated", slow)
    rows, _ = f.run(n=6, workers=6, grade_slots=2)
    assert len(rows) == 6 and most[0] == 2


def test_the_pipeline_goes_through_the_batch_gate(monkeypatch, tmp_path):
    import llm_client
    f = Fake(monkeypatch, tmp_path, [])
    seen = {}

    def gated(user_msg, model=None, system=None, schema=None):
        seen["gate"] = llm_client._gate is not None
        return json.dumps(TRIPLE), {"input": 1, "output": 1}, "ok"
    monkeypatch.setattr(f.R, "call_model", gated)
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    [row], _ = f.run(batch=True, flush_after=0.1)
    assert seen["gate"] is True and row["batch"] is True
    assert llm_client._gate is None


def test_pilot_routes_the_new_modes_to_the_pipeline(monkeypatch):
    import run as R
    seen = {}
    monkeypatch.setattr(R, "run_pipeline", lambda n, **kw: seen.update(n=n, **kw))
    monkeypatch.setattr(sys, "argv", ["run.py", "pilot", "--n", "4", "--mode",
                                      "two-step", "--repairs", "2", "--workers",
                                      "8", "--batch"])
    R.main()
    assert seen["n"] == 4 and seen["mode"] == "two-step"
    assert seen["repairs"] == 2 and seen["workers"] == 8 and seen["batch"] is True


def test_the_old_one_step_paths_also_save_initiate_examples(monkeypatch, tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NECESSARY"])
    f.R.run_attempts(1, cmd="pilot", log_path=f.log, distill_path=f.distill)
    [e] = dl.read(f.distill)
    assert e["prompt"] == "initiate" and e["outcome"] == "success"


# --- logging every single thing (8 Oct 2026) --------------------------------

def test_a_refused_answer_keeps_its_raw_text(monkeypatch, tmp_path):
    import llm_client
    f = Fake(monkeypatch, tmp_path, [])

    def refusing(user_msg, model=None, system=None, schema=None):
        llm_client._thread.call = {"raw": "I won't write that", "stop": "refusal",
                                   "usage": {"input": 5, "output": 1},
                                   "seconds": 0.1, "batch_id": None,
                                   "custom_id": None}
        return None, {"input": 5, "output": 1}, "refusal"
    monkeypatch.setattr(f.R, "call_model", refusing)
    [row], ex = f.run(repairs=2)
    assert row["verdict"] == "REFUSED" and row["raw_text"] == "I won't write that"
    assert ex[0]["response"] == "I won't write that" and ex[0]["outcome"] == "error"


def test_every_judged_design_in_a_repair_chain_keeps_its_evidence(monkeypatch,
                                                                  tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN", "NOT_PROVEN", "NECESSARY"])
    [row], ex = f.run(repairs=2)
    assert [h["verdict"] for h in row["history"]] == ["NOT_PROVEN", "NOT_PROVEN",
                                                      "NECESSARY"]
    assert all(h["result"] and "grade_wall_s" in h for h in row["history"])
    assert all(json.loads(h["design"])["invariants"] for h in row["history"])


def test_each_example_records_its_own_call_and_the_chain_total(monkeypatch,
                                                               tmp_path):
    f = Fake(monkeypatch, tmp_path, ["NOT_PROVEN", "NECESSARY"])
    [row], ex = f.run(repairs=2)
    rep = next(e for e in ex if e["prompt"] == "repair")
    gen = next(e for e in ex if e["prompt"] == "generate")
    assert rep["metadata"]["call"]["usage"] == {"input": 2, "output": 3}
    assert gen["metadata"]["chain_usage"] == {"input": 4, "output": 6}
    assert row["usage"] == {"input": 4, "output": 6}


def test_a_pipeline_run_logs_every_call_with_its_batch(monkeypatch, tmp_path):
    """End to end through the real client: batch functions faked, so no
    network; every call lands in the call log next to the run log."""
    import llm_client
    f = Fake(monkeypatch, tmp_path, [])
    R = f.R
    monkeypatch.setattr(R, "call_model", f.real_call_model)   # the real one
    sent = []

    def submit(items):
        sent.append(items)
        return f"b{len(sent)}"

    def results(bid):
        for cid, kw in sent[int(bid[1:]) - 1]:
            yield cid, ("ok", json.dumps(TRIPLE), {"input": 7, "output": 8}, json.dumps(TRIPLE))
    monkeypatch.setattr(llm_client, "batch_submit", submit)
    monkeypatch.setattr(llm_client, "batch_status", lambda b: ("ended", {}))
    monkeypatch.setattr(llm_client, "batch_results", results)
    monkeypatch.delenv("CLAUDE_PROVIDER", raising=False)
    rows, ex = f.run(n=2, workers=2, batch=True, flush_after=0.1)
    calls = [json.loads(l) for l in (tmp_path / "calls_log.jsonl").read_text().splitlines()]
    assert len(calls) == 2 and all(c["batch_id"] and c["custom_id"] for c in calls)
    assert all(c["request"]["system"] == R.SYSTEM_PROMPT for c in calls)
    assert all(e["metadata"]["call"]["batch_id"] for e in ex)
