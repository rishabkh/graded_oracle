"""Storage designs written by a program (5 Nov 2026): kinds built the way
real blocks are, each with a property and a helper fact for each slot, so
training finally sees the per-slot answers storage needs (713 of 824
answers in the biggest training set named no slot at all). These pin the
text of the queue kind; the judge checks the designs for real."""
import random
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extender"))
sys.path.insert(0, str(ROOT / "initiator"))

import storage_designs as sd                                     # noqa: E402


def queue(**fixed):
    # the count-tracking queue unless a test asks for another way
    fixed.setdefault("track", "count")
    return sd.queue_design(sd.queue_settings(random.Random(0), **fixed))


def test_a_queue_has_a_fact_for_each_slot_plus_its_pointers_and_count():
    d = queue(depth=8, names=0)
    assert len(d["invariants"]) == 8 + 2
    for i in range(8):
        assert any(f.startswith(f"slot_full[{i}] == ") for f in d["invariants"])


def test_a_slot_fact_cannot_wrap_around():
    # the window from the read pointer is tested on each side of it
    d = queue(depth=4, names=0)
    assert ("slot_full[1] == ((rd_ptr <= 2'd1) ? ((3'd1 - rd_ptr) < count)"
            " : ((3'd5 - rd_ptr) < count))") in d["invariants"]
    assert "wr_ptr == rd_ptr + count[1:0]" in d["invariants"]
    assert "count <= 3'd4" in d["invariants"]


def test_depth_two_uses_one_bit_pointers():
    d = queue(depth=2, names=0)
    assert "reg [0:0] wr_ptr;" in d["verilog"]
    assert "wr_ptr == rd_ptr + count[0:0]" in d["invariants"]
    assert "slot_full[1] == ((rd_ptr <= 1'd1) ? ((2'd1 - rd_ptr) < count)" \
        " : ((2'd3 - rd_ptr) < count))" in d["invariants"]


def test_the_facts_use_nothing_ebmc_cannot_read():
    import run
    for depth in (2, 16, 64):
        for prop in sd.QUEUE_PROPERTIES:
            d = queue(depth=depth, prop=prop)
            assert run.facts_gate(d) is None, (depth, prop)
            assert not any("$" in f or "." in f for f in d["invariants"])


@pytest.mark.parametrize("prop,expected", [
    ("from_filled_slot", "!in_valid_out || slot_full[rd_ptr]"),
    ("never_overwrite", "!in_ready || !slot_full[wr_ptr]"),
])
def test_each_property_is_one_engineers_check(prop, expected):
    d = queue(depth=4, names=0, prop=prop, style="ready_valid")
    text = expected.replace("in_valid_out", "out_valid")
    assert d["property"] == [text]
    assert f"assert ({text});" in d["verilog"]


def test_the_count_property_adds_up_every_slot():
    d = queue(depth=4, names=0, prop="count_matches")
    assert d["property"] == ["count == slot_full[0] + slot_full[1] + "
                             "slot_full[2] + slot_full[3]"]


def test_a_queue_with_enable_and_flag_ports_reads_them_as_a_real_one():
    d = queue(depth=4, names=0, style="enable_flag", prop="from_filled_slot")
    v = d["verilog"]
    assert "output wire             full," in v and "input  wire             rd_en," in v
    assert "wire do_push = wr_en && !full;" in v
    assert d["property"] == ["empty || slot_full[rd_ptr]"]


def test_the_sender_waits_when_full_and_nothing_is_overwritten():
    v = queue(depth=4, names=0, style="ready_valid")["verilog"]
    assert "assign in_ready  = (count != 3'd4);" in v
    assert "wire do_push = in_valid && in_ready;" in v


def test_a_flush_empties_every_slot():
    v = queue(depth=4, names=0, flush=True)["verilog"]
    assert "input  wire             flush," in v
    assert "if (rst || flush) begin" in v
    assert "slot_full <= 4'd0;" in v


def test_settings_are_the_same_for_the_same_seed_and_vary_across_seeds():
    one = sd.queue_settings(random.Random(5))
    assert one == sd.queue_settings(random.Random(5))
    seen = {tuple(sorted(sd.queue_settings(random.Random(s)).items()))
            for s in range(40)}
    assert len(seen) > 30


def test_every_register_starts_at_a_known_value():
    v = queue(depth=8, names=0)["verilog"]
    for reg in ("slot_full", "wr_ptr", "rd_ptr", "count"):
        assert re.search(rf"initial {reg}\s*=", v), reg


def test_the_design_is_described_plainly_at_the_top():
    d = queue(depth=16, width=32, names=0)
    assert d["verilog"].startswith("// A 16-slot queue of 32-bit items")


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_every_style_reads_in_yosys(tmp_path):
    import subprocess
    for style in sd.QUEUE_STYLES:
        for track in sd.QUEUE_TRACKS:
          for depth in (2, 64):
            d = queue(depth=depth, style=style, flush=True, width=1, track=track)
            f = tmp_path / f"{style}_{track}_{depth}.sv"
            f.write_text(d["verilog"])
            r = subprocess.run(["yosys", "-q", "-p",
                                f"read_verilog -sv {f}; hierarchy -top {d['top_module']}; proc"],
                               capture_output=True, text=True)
            assert r.returncode == 0, (style, depth, r.stderr[-400:])


@pytest.mark.skipif(shutil.which("sby") is None, reason="needs sby")
@pytest.mark.parametrize("prop", ["from_filled_slot", "never_overwrite",
                                  "count_matches"])
def test_a_small_queue_is_judged_necessary_like_a_model_written_design(
        prop, tmp_path):
    verdict = sd.judge(queue(depth=4, width=4, prop=prop, flush=True),
                       proof_root=tmp_path)
    assert verdict["verdict"] == "NECESSARY", verdict["reason"]


# --- building a batch (task 3) --------------------------------------------

def fake_judge(verdicts):
    """Judges by the design's property: NECESSARY unless named."""
    calls = []

    def judge(design, proof_root=None):
        calls.append(design["top_module"])
        v = verdicts.get(design["property"][0][:12], "NECESSARY")
        return {"verdict": v, "reason": None if v == "NECESSARY" else "why",
                "grade_wall_s": 1.0}
    judge.calls = calls
    return judge


def read(path):
    import json
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_a_batch_saves_rows_the_training_pipeline_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "judge", fake_judge({}))
    out = tmp_path / "corpus.jsonl"
    sd.build(4, seed=7, workers=2, out=out, proof_root=tmp_path / "p")
    rows = read(out)
    assert len(rows) == 4 and len({r["id"] for r in rows}) == 4
    for i in range(4):
        kind = sd.design_settings(7, i)[0]
        assert any(r["id"] == f"storage_{kind}_{i:03d}" and r["kind"] == kind
                   for r in rows), i
    for r in rows:
        for key in ("verilog", "top_module", "clock", "property", "invariants",
                    "sanity_covers", "settings", "source"):
            assert r[key], key
        # a property that holds always (the count one) has no condition
        assert isinstance(r["antecedents"], list)
        assert r["generation"] == 0 and r["kind"] in sd.KINDS
        assert r["verdict"] == "NECESSARY"


def test_a_design_the_checker_turns_away_is_logged_with_why(monkeypatch, tmp_path):
    first = sd.make_design(*sd.design_settings(7, 0))["property"][0][:12]
    monkeypatch.setattr(sd, "judge", fake_judge({first: "DECORATIVE"}))
    out = tmp_path / "corpus.jsonl"
    sd.build(3, seed=7, workers=1, out=out, proof_root=tmp_path / "p")
    turned_away = read(sd.rejected_path(out))
    assert any(r["verdict"] == "DECORATIVE" and r["reason"] == "why"
               for r in turned_away)
    assert all(r["verdict"] == "NECESSARY" for r in read(out))


def test_a_rerun_carries_on_without_judging_again(monkeypatch, tmp_path):
    judge = fake_judge({})
    monkeypatch.setattr(sd, "judge", judge)
    out = tmp_path / "corpus.jsonl"
    sd.build(3, seed=7, workers=1, out=out, proof_root=tmp_path / "p")
    sd.build(5, seed=7, workers=1, out=out, proof_root=tmp_path / "p")
    assert len(judge.calls) == 5 and len(read(out)) == 5


def test_the_same_seed_builds_the_same_designs():
    assert sd.design_settings(7, 3) == sd.design_settings(7, 3)
    assert sd.design_settings(7, 3) != sd.design_settings(8, 3)


def test_the_rejected_log_sits_beside_its_own_corpus(tmp_path):
    # 5 Nov 2026: it went to extender/logs whatever the corpus, so one
    # test's rejected design leaked into the next run
    assert sd.rejected_path(tmp_path / "c.jsonl") == tmp_path / "logs" / "rejected_c.jsonl"
    assert sd.rejected_path(sd.OUT).parent == sd.HERE / "logs"


# --- other ways real queues track how full they are (6 Nov 2026) -----------

def test_a_wrap_bit_queue_tells_full_from_empty_by_its_extra_pointer_bit():
    # verilog-axis axis_fifo: no count; full when the top bits differ
    d = queue(depth=4, names=0, track="wrap_bit", style="ready_valid")
    v = d["verilog"]
    assert "reg [2:0] wr_ptr;" in v and "reg [2:0] count;" not in v
    assert "assign in_ready  = (wr_ptr != {~rd_ptr[2], rd_ptr[1:0]});" in v
    assert "assign out_valid = (wr_ptr != rd_ptr);" in v
    assert len(d["invariants"]) == 4 + 1
    assert ("slot_full[1] == ((rd_ptr[1:0] <= 2'd1) ? ((3'd1 - rd_ptr[1:0]) < "
            "(wr_ptr - rd_ptr)) : ((3'd5 - rd_ptr[1:0]) < (wr_ptr - rd_ptr)))"
            ) in d["invariants"]
    assert "(wr_ptr - rd_ptr) <= 3'd4" in d["invariants"]


def test_a_wrap_bit_queue_counts_by_its_pointers():
    d = queue(depth=4, names=0, track="wrap_bit", prop="count_matches")
    assert d["property"] == ["(wr_ptr - rd_ptr) == slot_full[0] + slot_full[1]"
                             " + slot_full[2] + slot_full[3]"]


def test_a_queue_with_flag_registers_ties_them_to_its_count():
    # ZipCPU sfifo keeps full and empty in registers of their own
    d = queue(depth=4, names=0, track="flags", style="enable_flag")
    assert "full_q == (count == 3'd4)" in d["invariants"]
    assert "empty_q == (count == 3'd0)" in d["invariants"]
    assert len(d["invariants"]) == 4 + 4
    assert "assign full  = full_q;" in d["verilog"]


@pytest.mark.skipif(shutil.which("sby") is None, reason="needs sby")
@pytest.mark.parametrize("track", ["wrap_bit", "flags"])
def test_each_way_of_tracking_is_judged_necessary(track, tmp_path):
    for prop in sd.QUEUE_PROPERTIES:
        verdict = sd.judge(queue(depth=4, width=4, prop=prop, track=track,
                                 flush=True), proof_root=tmp_path / prop)
        assert verdict["verdict"] == "NECESSARY", (prop, verdict["reason"])


# --- the lookup table kind (6 Nov 2026) -----------------------------------
# biriscv's branch target table: keys in an array, a search loop, an update
# in place when the key is there; so a key is never stored twice.

def table(**fixed):
    return sd.table_design(sd.table_settings(random.Random(0), **fixed))


def test_a_table_has_a_fact_for_each_pair_of_entries():
    d = table(entries=4, names=0)
    assert len(d["invariants"]) == 6
    assert "!(valid[0] && valid[1] && (keys[0] == keys[1]))" in d["invariants"]
    assert "!(valid[2] && valid[3] && (keys[2] == keys[3]))" in d["invariants"]


def test_the_property_is_that_a_lookup_matches_at_most_one_entry():
    d = table(entries=4, names=0)
    assert d["property"] == ["lk_match[0] + lk_match[1] + lk_match[2] + "
                             "lk_match[3] <= 3'd1"]


def test_an_insert_updates_the_entry_that_holds_its_key():
    v = table(entries=4, names=0, policy="wait")["verilog"]
    assert "ins_hit ? ins_slot : free_slot" in v
    assert "assign ins_ready = (ins_hit || any_free)" in v


def test_a_full_table_can_replace_entries_in_turn_instead():
    v = table(entries=4, names=0, policy="evict")["verilog"]
    assert "ins_hit ? ins_slot : (any_free ? free_slot : victim)" in v
    assert "victim <= victim + 2'd1;" in v


def test_a_remove_by_key_comes_before_an_insert():
    wait = table(entries=4, names=0, remove=True, policy="wait")["verilog"]
    assert "input  wire             inv_valid," in wait
    assert "assign ins_ready = (ins_hit || any_free) && !inv_valid;" in wait
    evict = table(entries=4, names=0, remove=True, policy="evict")["verilog"]
    assert "assign ins_ready = !inv_valid;" in evict
    assert "end else if (do_inv) begin" in evict


def test_table_facts_use_nothing_ebmc_cannot_read():
    import run
    for n in (2, 8, 16):
        d = table(entries=n)
        assert run.facts_gate(d) is None, n
        assert len(d["invariants"]) == n * (n - 1) // 2


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_every_table_style_reads_in_yosys(tmp_path):
    import subprocess
    for policy in sd.TABLE_POLICIES:
        for n in (2, 16):
            d = table(entries=n, policy=policy, remove=True, flush=True)
            f = tmp_path / f"{policy}_{n}.sv"
            f.write_text(d["verilog"])
            r = subprocess.run(["yosys", "-q", "-p",
                                f"read_verilog -sv {f}; hierarchy -top {d['top_module']}; proc"],
                               capture_output=True, text=True)
            assert r.returncode == 0, (policy, n, r.stderr[-400:])


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_every_name_set_of_every_kind_reads_in_yosys(tmp_path):
    # table name set 11 once called its match vector and its lookup port
    # both lookup_hit: 12 of the 330-design batch never read (6 Nov 2026)
    import subprocess
    extras = {"queue": [{"style": st, "track": "flags", "flush": True}
                        for st in sd.QUEUE_STYLES],
              "table": [{"policy": p, "remove": True, "flush": True}
                        for p in sd.TABLE_POLICIES],
              "arbiter": [{"flush": True}],
              "pool": [{"flush": True}]}
    for kind, (settings, design) in sd.KINDS.items():
        for names in range(12):
            for extra in extras[kind]:
                d = design(settings(random.Random(names), names=names, **extra))
                f = tmp_path / f"{kind}_{names}.sv"
                f.write_text(d["verilog"])
                r = subprocess.run(["yosys", "-q", "-p",
                                    f"read_verilog -sv {f}; hierarchy -top {d['top_module']}; proc"],
                                   capture_output=True, text=True)
                assert r.returncode == 0, (kind, names, extra, r.stderr[-400:])


@pytest.mark.skipif(shutil.which("sby") is None, reason="needs sby")
@pytest.mark.parametrize("policy", ["wait", "evict"])
def test_a_small_table_is_judged_necessary(policy, tmp_path):
    for remove in (False, True):
        verdict = sd.judge(table(entries=4, key_width=4, policy=policy,
                                 remove=remove, flush=True),
                           proof_root=tmp_path / str(remove))
        assert verdict["verdict"] == "NECESSARY", (remove, verdict["reason"])


def test_a_batch_mixes_the_kinds():
    kinds = [sd.design_settings(sd.SEED, i)[0] for i in range(200)]
    assert set(kinds) == set(sd.KINDS)
    assert kinds.count("queue") > kinds.count("table")


# --- the oldest-first arbiter kind (6 Nov 2026) ----------------------------
# Ibex's instruction cache fill buffers keep an age matrix and serve the
# oldest first; its formal checks say the order among busy buffers is
# irreflexive, antisymmetric, transitive and total.

def arbiter(**fixed):
    return sd.arbiter_design(sd.arbiter_settings(random.Random(0), **fixed))


def test_an_arbiter_has_a_fact_for_each_pair_and_each_way_round_three_slots():
    d = arbiter(slots=4, names=0)
    pairs = [f for f in d["invariants"] if f.count("busy[") == 2]
    threes = [f for f in d["invariants"] if f.count("busy[") == 3]
    assert len(pairs) == 6 and len(threes) == 4 * 2      # 4 groups of three
    assert "!(busy[0] && busy[1]) || (older[0][1] != older[1][0])" in pairs
    assert ("!(busy[0] && busy[1] && busy[2] && older[0][1] && older[1][2])"
            " || older[0][2]") in threes
    assert ("!(busy[0] && busy[1] && busy[2] && older[0][2] && older[2][1])"
            " || older[0][1]") in threes


def test_no_arbiter_fact_is_another_turned_around():
    # 0>1>2 then 0>2 rules out the loop 0>1>2>0, and so do 1>2>0 then 1>0
    # and 2>0>1 then 2>1 once each pair is ordered: the 1-step check found
    # none of the three copies needed (0 of 2,028, 6 Nov 2026)
    for n in sd.ARBITER_SLOTS:
        loops = []
        for f in arbiter(slots=n, names=0)["invariants"]:
            m = re.search(r"older\[(\d)\]\[(\d)\] && older\[\d\]\[(\d)\]\) \|\| ", f)
            if m:
                loop = [int(x) for x in m.groups()]
                start = loop.index(min(loop))
                loops.append(tuple(loop[start:] + loop[:start]))
        assert len(loops) == len(set(loops)) == n * (n - 1) * (n - 2) // 3, n


def test_the_property_is_one_oldest_whenever_something_waits():
    d = arbiter(slots=3, names=0)
    assert d["property"] == ["!(|busy) || (oldest[0] + oldest[1] + oldest[2]"
                             " == 2'd1)"]


def test_a_new_request_is_younger_than_every_waiting_one():
    v = arbiter(slots=4, names=0)["verilog"]
    assert "older[j1][free_slot] <= busy[j1];" in v
    assert "older[free_slot][j1] <= 1'b0;" in v


def test_arbiter_facts_use_nothing_ebmc_cannot_read():
    import run
    for n in (3, 5):
        d = arbiter(slots=n)
        assert run.facts_gate(d) is None, n
        assert len(d["invariants"]) == n * (n - 1) // 2 + n * (n - 1) * (n - 2) // 3


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_every_arbiter_reads_in_yosys(tmp_path):
    import subprocess
    for n in sd.ARBITER_SLOTS:
        d = arbiter(slots=n, flush=True)
        f = tmp_path / f"arb_{n}.sv"
        f.write_text(d["verilog"])
        r = subprocess.run(["yosys", "-q", "-p",
                            f"read_verilog -sv {f}; hierarchy -top {d['top_module']}; proc"],
                           capture_output=True, text=True)
        assert r.returncode == 0, (n, r.stderr[-400:])


@pytest.mark.skipif(shutil.which("sby") is None, reason="needs sby")
def test_a_small_arbiter_is_judged_necessary(tmp_path):
    for n in (3, 4):
        verdict = sd.judge(arbiter(slots=n, width=8, flush=True),
                           proof_root=tmp_path / str(n))
        assert verdict["verdict"] == "NECESSARY", (n, verdict["reason"])


def test_two_slot_arbiters_are_left_out_because_the_property_proves_alone():
    assert min(sd.ARBITER_SLOTS) == 3


# --- the ID pool kind (6 Nov 2026) -----------------------------------------
# Free IDs wait in a ring list; allocating hands out the first and marks it
# in use, releasing an ID in use puts it back; no ID is ever out twice.

def pool(**fixed):
    return sd.pool_design(sd.pool_settings(random.Random(0), **fixed))


def test_a_pool_has_a_fact_for_each_list_place_and_each_pair_of_places():
    d = pool(ids=4, names=0)
    assert len(d["invariants"]) == 4 + 6 + 2
    live0 = "((fl_head <= 2'd0) ? ((3'd0 - fl_head) < fl_count) : ((3'd4 - fl_head) < fl_count))"
    live1 = "((fl_head <= 2'd1) ? ((3'd1 - fl_head) < fl_count) : ((3'd5 - fl_head) < fl_count))"
    assert f"!{live0} || !busy[free_list[0]]" in d["invariants"]
    assert f"!({live0} && {live1}) || (free_list[0] != free_list[1])" in d["invariants"]
    assert "fl_tail == fl_head + fl_count[1:0]" in d["invariants"]
    assert "fl_count <= 3'd4" in d["invariants"]


@pytest.mark.parametrize("prop,expected", [
    ("offer_is_free", "!alloc_valid || !busy[alloc_id]"),
    ("counts_add_up", "fl_count + busy[0] + busy[1] + busy[2] + busy[3] == 3'd4"),
])
def test_each_pool_property(prop, expected):
    assert pool(ids=4, names=0, prop=prop)["property"] == [expected]


def test_reset_fills_the_free_list_with_every_id():
    v = pool(ids=4, names=0)["verilog"]
    assert "initial free_list[3] = 2'd3;" in v
    assert "free_list[3] <= 2'd3;" in v


def test_only_an_id_in_use_can_be_released():
    v = pool(ids=4, names=0)["verilog"]
    assert "wire do_rel = rel_valid && busy[rel_id];" in v


def test_pool_facts_use_nothing_ebmc_cannot_read():
    import run
    for n in sd.POOL_IDS:
        for prop in sd.POOL_PROPERTIES:
            d = pool(ids=n, prop=prop)
            assert run.facts_gate(d) is None, (n, prop)


@pytest.mark.skipif(shutil.which("yosys") is None, reason="needs yosys")
def test_every_pool_reads_in_yosys(tmp_path):
    import subprocess
    for n in sd.POOL_IDS:
        d = pool(ids=n, flush=True)
        f = tmp_path / f"pool_{n}.sv"
        f.write_text(d["verilog"])
        r = subprocess.run(["yosys", "-q", "-p",
                            f"read_verilog -sv {f}; hierarchy -top {d['top_module']}; proc"],
                           capture_output=True, text=True)
        assert r.returncode == 0, (n, r.stderr[-400:])


@pytest.mark.skipif(shutil.which("sby") is None, reason="needs sby")
@pytest.mark.parametrize("prop", ["offer_is_free", "counts_add_up"])
def test_a_small_pool_is_judged_necessary(prop, tmp_path):
    verdict = sd.judge(pool(ids=4, prop=prop, flush=True), proof_root=tmp_path)
    assert verdict["verdict"] == "NECESSARY", verdict["reason"]


def test_tables_stop_at_eight_entries_because_sixteen_time_out():
    assert max(sd.TABLE_ENTRIES) == 8


def test_every_eleventh_design_is_set_aside_for_our_own_storage_test(
        monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "judge", fake_judge({}))
    out = tmp_path / "corpus.jsonl"
    sd.build(22, seed=7, workers=2, out=out, proof_root=tmp_path / "p")
    rows = read(out)
    aside = sorted(int(r["id"].rsplit("_", 1)[1]) for r in rows if r["held_out"])
    assert aside == [10, 21]
    assert all(r["held_out"] is False for r in rows
               if int(r["id"].rsplit("_", 1)[1]) not in (10, 21))
