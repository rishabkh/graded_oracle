"""Storage designs written by a program (5 Nov 2026).

Training data so far holds almost no per-slot answers: in the biggest
training set 713 of 824 answers name no slot at all and only 26 name eight
or more, while the storage blocks real hardware is full of need a fact for
each slot. This writes storage blocks the way real ones are built, each
with a property engineers check on it and the facts that make it
provable, and keeps a design only if the checker that judges
model-written designs says those facts are needed and enough (NECESSARY).

Kinds, each from a real open-source block (the benchmark's files are
never opened, under the rule against shaping data like the test):
  queue   a ring buffer with a full bit per slot, wrap-around pointers and a
          sender told to wait when it is full, knowing how full it is by
          a count, by an extra pointer bit, or by full and empty flag
          registers (Ibex's prefetch FIFO keeps a valid bit per entry;
          verilog-axis axis_fifo, ZipCPU's sfifo)

  table   a lookup table whose insert updates the entry holding its key or
          takes a free one, so a key is stored at most once; when full the
          sender waits or entries are replaced in turn (biriscv's branch
          target table)

  arbiter request slots kept in age order by a matrix, the oldest granted
          first (Ibex's instruction cache fill buffers)
  pool    IDs handed out from a ring list of free IDs and put back when
          released (free lists in out-of-order cores)

  venv/bin/python extender/storage_designs.py --n 10 --dry
  venv/bin/python extender/storage_designs.py --n 10
"""
import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_storage_designs_2026-11-05.jsonl"
PROOFS = HERE / "logs" / "storage_designs_proofs_2026-11-05"
SEED = 20261105
SOURCES = {"queue": "Ibex prefetch FIFO (a valid bit per entry); verilog-axis "
                    "axis_fifo; ZipCPU sfifo",
           "table": "biriscv branch target table (biriscv_npc.v): search "
                    "loop, update in place on a hit",
           "arbiter": "Ibex instruction cache fill buffers (ibex_icache.sv): "
                      "age matrix, oldest first",
           "pool": "free lists of tags and renamed registers in out-of-order "
                   "cores (BOOM's rename free list); a ring list like the queue"}

QUEUE_PROPERTIES = ("from_filled_slot", "never_overwrite", "count_matches")
QUEUE_STYLES = ("ready_valid", "enable_flag")
DEPTHS = (2, 4, 8, 16, 32, 64)
WIDTHS = (1, 4, 8, 16, 32, 64)
# port names as real blocks spell them: for ready_valid (in valid, in
# ready, in data, out valid, out ready, out data); for enable_flag
# (write, full, write data, read, empty, read data)
PORTS = {"ready_valid": [
             ("in_valid", "in_ready", "in_data", "out_valid", "out_ready", "out_data"),
             ("s_axis_tvalid", "s_axis_tready", "s_axis_tdata",
              "m_axis_tvalid", "m_axis_tready", "m_axis_tdata"),
             ("enq_valid", "enq_ready", "enq_data", "deq_valid", "deq_ready", "deq_data")],
         "enable_flag": [
             ("wr_en", "full", "wr_data", "rd_en", "empty", "rd_data"),
             ("push", "full", "push_data", "pop", "empty", "pop_data")]}
# inside names: (slot bits, write pointer, read pointer, count, storage,
# level output)
INSIDE = [("slot_full", "wr_ptr", "rd_ptr", "count", "mem", "level"),
          ("occupied", "tail", "head", "fill", "storage", "fill_level"),
          ("entry_valid", "waddr", "raddr", "used", "entries", "occupancy"),
          ("valid_q", "wptr_q", "rptr_q", "cnt_q", "ram", "level")]
# how a queue knows how full it is (6 Nov 2026): a count register; or
# pointers with one extra bit that flips at each wrap and no count
# (verilog-axis axis_fifo: full when the top bits differ and the rest
# match); or a count plus full and empty flags in registers of their own
# (ZipCPU sfifo)
QUEUE_TRACKS = ("count", "wrap_bit", "flags")
FLAGS = [("full_q", "empty_q"), ("is_full", "is_empty"), ("r_full", "r_empty"),
         ("full_r", "empty_r")]
WHAT = ("cmd", "pkt", "req", "evt", "desc", "rx", "tx", "msg", "job", "tag")
NOUN = ("fifo", "queue", "buffer")


def queue_settings(rng, **fixed):
    """One queue's choices, from `rng`; `fixed` overrides any of them.
    `names` picks the inside names and the port names together."""
    s = {"depth": rng.choice(DEPTHS), "width": rng.choice(WIDTHS),
         "style": rng.choice(QUEUE_STYLES), "names": rng.randrange(12),
         "prop": rng.choice(QUEUE_PROPERTIES), "flush": rng.random() < 0.3,
         "top": f"{rng.choice(WHAT)}_{rng.choice(NOUN)}",
         "track": rng.choice(QUEUE_TRACKS)}
    s.update(fixed)
    return s


def _port(direction, width, name):
    return f"    {direction:<6} wire {width:<11} {name},"


def queue_design(s):
    """The design, its property and its facts. Slot i is full exactly when
    it sits in the window of filled slots starting at the read pointer;
    the test is written on each side of the read pointer, so no subtraction
    ever wraps."""
    D, W = s["depth"], s["width"]
    AW = max(1, (D - 1).bit_length())          # bits that name a slot
    CW = AW + 1                                # bits of an item count: 0 to D
    track = s.get("track", "count")
    wrap, flags = track == "wrap_bit", track == "flags"
    slot, wr, rd, count, mem, level = INSIDE[s["names"] % len(INSIDE)]
    f_full, f_empty = FLAGS[s["names"] % len(FLAGS)]
    ports = PORTS[s["style"]]
    p_in, p_in2, p_data, p_out, p_out2, p_odata = ports[s["names"] % len(ports)]
    rv = s["style"] == "ready_valid"

    PW = CW if wrap else AW                    # pointer bits
    wr_i = f"{wr}[{AW - 1}:0]" if wrap else wr     # the slot a pointer names
    rd_i = f"{rd}[{AW - 1}:0]" if wrap else rd
    occ = f"({wr} - {rd})" if wrap else count      # items held
    if wrap:
        is_full = f"({wr} == {{~{rd}[{AW}], {rd}[{AW - 1}:0]}})"
        not_full = f"({wr} != {{~{rd}[{AW}], {rd}[{AW - 1}:0]}})"
        is_empty, not_empty = f"({wr} == {rd})", f"({wr} != {rd})"
    elif flags:
        is_full, not_full, is_empty, not_empty = f_full, f"!{f_full}", f_empty, f"!{f_empty}"
    else:
        is_full, not_full = f"({count} == {CW}'d{D})", f"({count} != {CW}'d{D})"
        is_empty, not_empty = f"({count} == {CW}'d0)", f"({count} != {CW}'d0)"

    port_lines = [_port("input", "", "clk"), _port("input", "", "rst")]
    if s["flush"]:
        port_lines.append(_port("input", "", "flush"))
    port_lines += [_port("input", "", p_in), _port("output", "", p_in2),
                   _port("input", "[WIDTH-1:0]", p_data),
                   _port("output" if rv else "input", "", p_out),
                   _port("input" if rv else "output", "", p_out2),
                   _port("output", "[WIDTH-1:0]", p_odata),
                   _port("output", f"[{AW}:0]", level)]
    port_lines[-1] = port_lines[-1].rstrip(",")

    if rv:          # p_in2 = ready, p_out = valid, p_out2 = the reader's ready
        handshake = [f"    assign {p_in2}  = {not_full};",
                     f"    assign {p_out} = {not_empty};",
                     f"    wire do_push = {p_in} && {p_in2};",
                     f"    wire do_pop  = {p_out} && {p_out2};"]
        has_item, can_push = p_out, p_in2
    else:           # p_in2 = full, p_out = read, p_out2 = empty
        handshake = [f"    assign {p_in2}  = {is_full};",
                     f"    assign {p_out2} = {is_empty};",
                     f"    wire do_push = {p_in} && !{p_in2};",
                     f"    wire do_pop  = {p_out} && !{p_out2};"]
        has_item, can_push = f"!{p_out2}", f"!{p_in2}"

    if s["prop"] == "from_filled_slot":
        prop = f"{'!' + p_out if rv else p_out2} || {slot}[{rd_i}]"
        antecedents, said = [has_item], "what is read always comes from a filled slot"
    elif s["prop"] == "never_overwrite":
        prop = f"{'!' + p_in2 if rv else p_in2} || !{slot}[{wr_i}]"
        antecedents, said = [can_push], "a write never lands on a filled slot"
    else:
        prop = f"{occ} == " + " + ".join(f"{slot}[{i}]" for i in range(D))
        antecedents, said = [], "the count equals the number of filled slots"

    facts = [f"{slot}[{i}] == (({rd_i} <= {AW}'d{i}) ? (({CW}'d{i} - {rd_i}) < {occ})"
             f" : (({CW}'d{i + D} - {rd_i}) < {occ}))" for i in range(D)]
    if wrap:
        facts.append(f"{occ} <= {CW}'d{D}")
    else:
        facts += [f"{wr} == {rd} + {count}[{AW - 1}:0]", f"{count} <= {CW}'d{D}"]
    if flags:
        facts += [f"{f_full} == ({count} == {CW}'d{D})",
                  f"{f_empty} == ({count} == {CW}'d0)"]

    how = {"count": "", "wrap_bit": " Full and empty come from an extra pointer bit.",
           "flags": " Full and empty are kept in registers."}[track]
    regs = [f"    reg [WIDTH-1:0] {mem} [0:DEPTH-1];",
            f"    reg [{D - 1}:0] {slot};",
            f"    reg [{PW - 1}:0] {wr};",
            f"    reg [{PW - 1}:0] {rd};"]
    inits = [f"    initial {slot} = {D}'d0;",
             f"    initial {wr} = {PW}'d0;",
             f"    initial {rd} = {PW}'d0;"]
    resets = [f"            {slot} <= {D}'d0;",
              f"            {wr} <= {PW}'d0;",
              f"            {rd} <= {PW}'d0;"]
    if not wrap:
        regs.append(f"    reg [{AW}:0] {count};")
        inits.append(f"    initial {count} = {CW}'d0;")
        resets.append(f"            {count} <= {CW}'d0;")
    if flags:
        regs.append(f"    reg {f_full}, {f_empty};")
        inits += [f"    initial {f_full} = 1'b0;", f"    initial {f_empty} = 1'b1;"]
        resets += [f"            {f_full} <= 1'b0;", f"            {f_empty} <= 1'b1;"]
    if wrap:
        counting = []
    elif flags:
        counting = [
            "            if (do_push && !do_pop) begin",
            f"                {count} <= {count} + {CW}'d1;",
            f"                {f_full} <= ({count} == {CW}'d{D - 1});",
            f"                {f_empty} <= 1'b0;",
            "            end else if (do_pop && !do_push) begin",
            f"                {count} <= {count} - {CW}'d1;",
            f"                {f_full} <= 1'b0;",
            f"                {f_empty} <= ({count} == {CW}'d1);",
            "            end"]
    else:
        counting = [
            "            if (do_push && !do_pop)",
            f"                {count} <= {count} + {CW}'d1;",
            "            else if (do_pop && !do_push)",
            f"                {count} <= {count} - {CW}'d1;"]

    v = [f"// A {D}-slot queue of {W}-bit items with a full bit per slot; the "
         f"sender waits when it is full.{how}",
         f"module {s['top']} #(",
         f"    parameter WIDTH = {W}",
         ") ("] + port_lines + [
        ");",
        f"    localparam DEPTH = {D};",
        ""] + regs + [""] + inits + [""] + handshake + [
        f"    assign {p_odata} = {mem}[{rd_i}];",
        f"    assign {level} = {wr} - {rd};" if wrap else f"    assign {level} = {count};",
        "",
        "    always @(posedge clk) begin",
        f"        if (do_push) {mem}[{wr_i}] <= {p_data};",
        "    end",
        "",
        "    always @(posedge clk) begin",
        f"        if (rst{' || flush' if s['flush'] else ''}) begin"] + resets + [
        "        end else begin",
        "            if (do_push) begin",
        f"                {slot}[{wr_i}] <= 1'b1;",
        f"                {wr} <= {wr} + {PW}'d1;",
        "            end",
        "            if (do_pop) begin",
        f"                {slot}[{rd_i}] <= 1'b0;",
        f"                {rd} <= {rd} + {PW}'d1;",
        "            end"] + counting + [
        "        end",
        "    end",
        "",
        f"    // {said}",
        f"    always @(posedge clk) assert ({prop});",
        "endmodule",
        ""]
    return {"verilog": "\n".join(v), "top_module": s["top"], "clock": "clk",
            "property": [prop], "invariants": facts,
            "antecedents": antecedents,
            "sanity_covers": [f"{occ} == {CW}'d{min(D, 2)}"]}


# --- the lookup table kind (6 Nov 2026) ------------------------------------
# biriscv's branch target table (biriscv_npc.v): keys in an array, a search
# loop over every entry, an update in place when the key is already there
# and a new entry otherwise, so a key is never stored twice. When no entry
# is free the sender waits, or entries are replaced in turn (the branch
# table replaces by a pseudo-random pick).
TABLE_POLICIES = ("wait", "evict")
# up to 8: with 16 entries (120 pair facts) the checker ran out of its
# 300 s before proving the facts (6 Nov 2026)
TABLE_ENTRIES = (2, 4, 8)
KEY_WIDTHS = (4, 8, 12, 16, 32)
VAL_WIDTHS = (4, 8, 16, 32)
# inside names: (filled bits, keys, values, lookup match vector); the last
# match vector was lookup_hit, the same name as the "lookup" ports' hit
# output, so name set 11 never read (6 Nov 2026)
TABLE_INSIDE = [("valid", "keys", "vals", "lk_match"),
                ("entry_valid", "tag_q", "data_q", "hit_vec"),
                ("v_q", "key_q", "val_q", "match_q"),
                ("in_use", "addr_tag", "payload", "entry_hit")]
# port prefixes: (insert, remove, lookup)
TABLE_PORTS = [("ins", "inv", "lk"), ("wr", "del", "rd"), ("fill", "drop", "lookup")]
TABLE_WHAT = ("tag", "addr", "btb", "flow", "route", "tlb", "session", "id")
TABLE_NOUN = ("table", "cam", "cache", "map")


def table_settings(rng, **fixed):
    """One lookup table's choices, from `rng`; `fixed` overrides any."""
    s = {"entries": rng.choice(TABLE_ENTRIES), "key_width": rng.choice(KEY_WIDTHS),
         "val_width": rng.choice(VAL_WIDTHS), "policy": rng.choice(TABLE_POLICIES),
         "remove": rng.random() < 0.5, "flush": rng.random() < 0.3,
         "names": rng.randrange(12),
         "top": f"{rng.choice(TABLE_WHAT)}_{rng.choice(TABLE_NOUN)}"}
    s.update(fixed)
    return s


def table_design(s):
    """The table, the property that a lookup matches at most one entry,
    and a fact for each pair of entries: two filled entries never hold
    the same key."""
    N, KW, VW = s["entries"], s["key_width"], s["val_width"]
    IW = max(1, (N - 1).bit_length())          # bits that name an entry
    CW = N.bit_length()                        # bits of a count: 0 to N
    valid, keys, vals, match = TABLE_INSIDE[s["names"] % len(TABLE_INSIDE)]
    ins, inv, lk = TABLE_PORTS[s["names"] % len(TABLE_PORTS)]
    evict, remove = s["policy"] == "evict", s["remove"]

    ports = [_port("input", "", "clk"), _port("input", "", "rst")]
    if s["flush"]:
        ports.append(_port("input", "", "flush"))
    ports += [_port("input", "", f"{ins}_valid"), _port("output", "", f"{ins}_ready"),
              _port("input", "[KEY_W-1:0]", f"{ins}_key"),
              _port("input", "[VAL_W-1:0]", f"{ins}_val")]
    if remove:
        ports += [_port("input", "", f"{inv}_valid"),
                  _port("input", "[KEY_W-1:0]", f"{inv}_key")]
    ports += [_port("input", "[KEY_W-1:0]", f"{lk}_key"),
              _port("output", "", f"{lk}_hit"),
              _port("output", "[VAL_W-1:0]", f"{lk}_val")]
    ports[-1] = ports[-1].rstrip(",")

    guard = f" && !{inv}_valid" if remove else ""
    if evict:
        ready = (f"    assign {ins}_ready = !{inv}_valid;" if remove
                 else f"    assign {ins}_ready = 1'b1;")
        target = f"{ins}_hit ? {ins}_slot : (any_free ? free_slot : victim)"
    else:
        ready = f"    assign {ins}_ready = ({ins}_hit || any_free){guard};"
        target = f"{ins}_hit ? {ins}_slot : free_slot"
    prop = " + ".join(f"{match}[{i}]" for i in range(N)) + f" <= {CW}'d1"
    facts = [f"!({valid}[{i}] && {valid}[{j}] && ({keys}[{i}] == {keys}[{j}]))"
             for i in range(N) for j in range(i + 1, N)]
    full = (" When no entry is free, entries are replaced in turn." if evict
            else " When no entry is free the sender waits.")

    v = [f"// A {N}-entry lookup table of {KW}-bit keys and {VW}-bit values; an "
         "insert updates the entry that holds its key or takes a free one, so "
         f"a key is stored at most once.{full}",
         f"module {s['top']} #(",
         f"    parameter KEY_W = {KW},",
         f"    parameter VAL_W = {VW}",
         ") ("] + ports + [
        ");",
        f"    localparam N = {N};",
        "",
        f"    reg [N-1:0] {valid};",
        f"    reg [KEY_W-1:0] {keys} [0:N-1];",
        f"    reg [VAL_W-1:0] {vals} [0:N-1];"]
    if evict:
        v.append(f"    reg [{IW - 1}:0] victim;")
    v += ["", f"    initial {valid} = {N}'d0;"]
    if evict:
        v.append(f"    initial victim = {IW}'d0;")
    v += ["",
          "    // lookup: every entry compared at once",
          f"    reg [N-1:0] {match};",
          "    reg [VAL_W-1:0] lk_val_r;",
          "    integer i0;",
          "    always @(*) begin",
          "        lk_val_r = {VAL_W{1'b0}};",
          "        for (i0 = 0; i0 < N; i0 = i0 + 1) begin",
          f"            {match}[i0] = {valid}[i0] && ({keys}[i0] == {lk}_key);",
          f"            if ({match}[i0]) lk_val_r = {vals}[i0];",
          "        end",
          "    end",
          f"    assign {lk}_hit = |{match};",
          f"    assign {lk}_val = lk_val_r;",
          "",
          "    // insert: the entry holding the key, else the lowest free one",
          f"    reg {ins}_hit, any_free;",
          f"    reg [{IW - 1}:0] {ins}_slot, free_slot;",
          "    integer i1;",
          "    always @(*) begin",
          f"        {ins}_hit = 1'b0;",
          f"        {ins}_slot = {IW}'d0;",
          "        any_free = 1'b0;",
          f"        free_slot = {IW}'d0;",
          "        for (i1 = N - 1; i1 >= 0; i1 = i1 - 1) begin",
          f"            if ({valid}[i1] && ({keys}[i1] == {ins}_key)) begin",
          f"                {ins}_hit = 1'b1;",
          f"                {ins}_slot = i1;",
          "            end",
          f"            if (!{valid}[i1]) begin",
          "                any_free = 1'b1;",
          "                free_slot = i1;",
          "            end",
          "        end",
          "    end"]
    if remove:
        v += ["",
              "    // remove: the entry holding the key, if any",
              f"    reg {inv}_hit;",
              f"    reg [{IW - 1}:0] {inv}_slot;",
              "    integer i2;",
              "    always @(*) begin",
              f"        {inv}_hit = 1'b0;",
              f"        {inv}_slot = {IW}'d0;",
              "        for (i2 = 0; i2 < N; i2 = i2 + 1)",
              f"            if ({valid}[i2] && ({keys}[i2] == {inv}_key)) begin",
              f"                {inv}_hit = 1'b1;",
              f"                {inv}_slot = i2;",
              "            end",
              "    end"]
    v += ["",
          ready,
          f"    wire [{IW - 1}:0] target = {target};",
          f"    wire do_ins = {ins}_valid && {ins}_ready;"]
    if remove:
        v.append(f"    wire do_inv = {inv}_valid && {inv}_hit;")
    v += ["",
          "    always @(posedge clk) begin",
          f"        if (rst{' || flush' if s['flush'] else ''}) begin",
          f"            {valid} <= {N}'d0;"]
    if evict:
        v.append(f"            victim <= {IW}'d0;")
    if remove:
        v += ["        end else if (do_inv) begin",
              f"            {valid}[{inv}_slot] <= 1'b0;"]
    v += ["        end else if (do_ins) begin",
          f"            {valid}[target] <= 1'b1;",
          f"            {keys}[target] <= {ins}_key;",
          f"            {vals}[target] <= {ins}_val;"]
    if evict:
        v.append(f"            if (!{ins}_hit && !any_free) victim <= victim + {IW}'d1;")
    v += ["        end",
          "    end",
          "",
          "    // a lookup matches at most one entry",
          f"    always @(posedge clk) assert ({prop});",
          "endmodule",
          ""]
    return {"verilog": "\n".join(v), "top_module": s["top"], "clock": "clk",
            "property": [prop], "invariants": facts, "antecedents": [],
            "sanity_covers": [f"{valid}[0] && {valid}[1]"]}


# --- the oldest-first arbiter kind (6 Nov 2026) ----------------------------
# Ibex's instruction cache (ibex_icache.sv) keeps its fill buffers in an age
# matrix and serves the oldest first; its formal testbench states that the
# order among busy buffers is irreflexive, antisymmetric, transitive and
# total. Here requests wait in slots, older[i][j] says slot i arrived before
# slot j, and the oldest waiting request is granted first.
# from 3 slots: with 2, "exactly one is oldest" already says the pair is
# ordered, so the property proves alone (judged DECORATIVE, 6 Nov 2026)
ARBITER_SLOTS = (3, 4, 5)
ARBITER_WIDTHS = (8, 16, 32)
# inside names: (waiting bits, age matrix, stored requests, oldest bits)
ARBITER_INSIDE = [("busy", "older", "req_q", "oldest"),
                  ("pending", "age", "addr_q", "is_oldest"),
                  ("in_use", "before", "payload_q", "head_sel"),
                  ("waiting", "arrived_first", "slot_data", "first")]
# port prefixes: (requests in, grants out)
ARBITER_PORTS = [("req", "gnt"), ("alloc", "issue"), ("in", "out")]
ARBITER_WHAT = ("fill", "miss", "dma", "mem", "bus", "irq", "load")
ARBITER_NOUN = ("arbiter", "tracker", "scheduler", "buffers")


def arbiter_settings(rng, **fixed):
    """One arbiter's choices, from `rng`; `fixed` overrides any."""
    s = {"slots": rng.choice(ARBITER_SLOTS), "width": rng.choice(ARBITER_WIDTHS),
         "flush": rng.random() < 0.3, "names": rng.randrange(12),
         "top": f"{rng.choice(ARBITER_WHAT)}_{rng.choice(ARBITER_NOUN)}"}
    s.update(fixed)
    return s


def arbiter_design(s):
    """The arbiter, the property that exactly one waiting request is the
    oldest whenever something waits, and the order facts: for each pair of
    waiting slots exactly one arrived first, and for each three the order
    carries over (a before b and b before c means a before c), once each
    way round. Ibex states the carry-over for every ordering of the three;
    written out, that is each loop three times over (b, c, a and c, a, b
    say the same as a, b, c once each pair is ordered), and the 1-step
    check found none of the three copies needed (6 Nov 2026)."""
    N, W = s["slots"], s["width"]
    IW = max(1, (N - 1).bit_length())          # bits that name a slot
    CW = N.bit_length()                        # bits of a count: 0 to N
    busy, older, data, oldest = ARBITER_INSIDE[s["names"] % len(ARBITER_INSIDE)]
    a, g = ARBITER_PORTS[s["names"] % len(ARBITER_PORTS)]

    ports = [_port("input", "", "clk"), _port("input", "", "rst")]
    if s["flush"]:
        ports.append(_port("input", "", "flush"))
    ports += [_port("input", "", f"{a}_valid"), _port("output", "", f"{a}_ready"),
              _port("input", "[WIDTH-1:0]", f"{a}_data"),
              _port("output", "", f"{g}_valid"), _port("input", "", f"{g}_ready"),
              _port("output", "[WIDTH-1:0]", f"{g}_data"),
              _port("output", f"[{IW - 1}:0]", f"{g}_slot")]
    ports[-1] = ports[-1].rstrip(",")

    prop = (f"!(|{busy}) || (" + " + ".join(f"{oldest}[{i}]" for i in range(N))
            + f" == {CW}'d1)")
    facts = [f"!({busy}[{i}] && {busy}[{j}]) || ({older}[{i}][{j}] != {older}[{j}][{i}])"
             for i in range(N) for j in range(i + 1, N)]
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                three = f"{busy}[{i}] && {busy}[{j}] && {busy}[{k}]"
                facts += [f"!({three} && {older}[{i}][{j}] && {older}[{j}][{k}])"
                          f" || {older}[{i}][{k}]",
                          f"!({three} && {older}[{i}][{k}] && {older}[{k}][{j}])"
                          f" || {older}[{i}][{j}]"]

    v = [f"// {N} request slots, each holding a {W}-bit request; {older}[i][j] "
         "says slot i arrived before slot j, and the oldest waiting request is "
         "granted first.",
         f"module {s['top']} #(",
         f"    parameter WIDTH = {W}",
         ") ("] + ports + [
        ");",
        f"    localparam N = {N};",
        "",
        f"    reg [N-1:0] {busy};",
        f"    reg [N-1:0] {older} [0:N-1];",
        f"    reg [WIDTH-1:0] {data} [0:N-1];",
        "",
        f"    initial {busy} = {N}'d0;",
        "",
        "    // the lowest free slot takes a new request",
        "    reg any_free;",
        f"    reg [{IW - 1}:0] free_slot;",
        "    integer i1;",
        "    always @(*) begin",
        "        any_free = 1'b0;",
        f"        free_slot = {IW}'d0;",
        "        for (i1 = N - 1; i1 >= 0; i1 = i1 - 1)",
        f"            if (!{busy}[i1]) begin",
        "                any_free = 1'b1;",
        "                free_slot = i1;",
        "            end",
        "    end",
        "",
        "    // a waiting slot is the oldest when no other waiting slot arrived",
        "    // before it; the lowest such slot is granted",
        f"    reg [N-1:0] {oldest};",
        f"    reg [{IW - 1}:0] grant_slot;",
        "    integer i0, j0;",
        "    always @(*) begin",
        f"        grant_slot = {IW}'d0;",
        "        for (i0 = N - 1; i0 >= 0; i0 = i0 - 1) begin",
        f"            {oldest}[i0] = {busy}[i0];",
        "            for (j0 = 0; j0 < N; j0 = j0 + 1)",
        f"                if (j0 != i0 && {busy}[j0] && {older}[j0][i0])",
        f"                    {oldest}[i0] = 1'b0;",
        f"            if ({oldest}[i0]) grant_slot = i0;",
        "        end",
        "    end",
        "",
        f"    assign {a}_ready = any_free;",
        f"    assign {g}_valid = |{busy};",
        f"    assign {g}_data = {data}[grant_slot];",
        f"    assign {g}_slot = grant_slot;",
        f"    wire do_req = {a}_valid && {a}_ready;",
        f"    wire do_gnt = {g}_valid && {g}_ready;",
        "",
        "    integer j1;",
        "    always @(posedge clk) begin",
        f"        if (rst{' || flush' if s['flush'] else ''}) begin",
        f"            {busy} <= {N}'d0;",
        "        end else begin",
        "            if (do_gnt)",
        f"                {busy}[grant_slot] <= 1'b0;",
        "            if (do_req) begin",
        f"                {busy}[free_slot] <= 1'b1;",
        f"                {data}[free_slot] <= {a}_data;",
        "                // every waiting request is older than the new one",
        "                for (j1 = 0; j1 < N; j1 = j1 + 1) begin",
        f"                    {older}[j1][free_slot] <= {busy}[j1];",
        f"                    {older}[free_slot][j1] <= 1'b0;",
        "                end",
        "            end",
        "        end",
        "    end",
        "",
        "    // whenever something waits, exactly one waiting request is the oldest",
        f"    always @(posedge clk) assert ({prop});",
        "endmodule",
        ""]
    return {"verilog": "\n".join(v), "top_module": s["top"], "clock": "clk",
            "property": [prop], "invariants": facts,
            "antecedents": [f"|{busy}"],
            "sanity_covers": [f"{busy}[0] && {busy}[1]"]}


# --- the ID pool kind (6 Nov 2026) -----------------------------------------
# A free list as out-of-order cores keep for tags and renamed registers (for
# example BOOM's rename free list): free IDs wait in a ring list, an
# allocation hands out the first and marks it in use, and releasing an ID
# in use puts it back, so no ID is ever out twice.
POOL_IDS = (4, 8)
POOL_PROPERTIES = ("offer_is_free", "counts_add_up")
# inside names: (in-use bits, free list, its head, its tail, free count)
POOL_INSIDE = [("busy", "free_list", "fl_head", "fl_tail", "fl_count"),
               ("in_use", "free_ids", "rd_ptr", "wr_ptr", "num_free"),
               ("taken", "pool_q", "head", "tail", "avail"),
               ("out_q", "spare", "get_ptr", "put_ptr", "n_free")]
# port prefixes: (allocate, release)
POOL_PORTS = [("alloc", "rel"), ("get", "put"), ("take", "give")]
POOL_WHAT = ("tag", "txn", "rob", "phys_reg", "mshr", "credit", "slot")
POOL_NOUN = ("pool", "allocator", "free_list")


def pool_settings(rng, **fixed):
    """One ID pool's choices, from `rng`; `fixed` overrides any."""
    s = {"ids": rng.choice(POOL_IDS), "prop": rng.choice(POOL_PROPERTIES),
         "flush": rng.random() < 0.3, "names": rng.randrange(12),
         "top": f"{rng.choice(POOL_WHAT)}_{rng.choice(POOL_NOUN)}"}
    s.update(fixed)
    return s


def pool_design(s):
    """The pool, its property, and the facts: every live place in the free
    list holds an ID not in use, no two live places hold the same ID, and
    the tail sits `count` places after the head."""
    N = s["ids"]
    IW = max(1, (N - 1).bit_length())          # bits of an ID
    CW = IW + 1                                # bits of a count: 0 to N
    busy, fl, head, tail, cnt = POOL_INSIDE[s["names"] % len(POOL_INSIDE)]
    a, r = POOL_PORTS[s["names"] % len(POOL_PORTS)]

    ports = [_port("input", "", "clk"), _port("input", "", "rst")]
    if s["flush"]:
        ports.append(_port("input", "", "flush"))
    ports += [_port("output", "", f"{a}_valid"), _port("input", "", f"{a}_ready"),
              _port("output", f"[{IW - 1}:0]", f"{a}_id"),
              _port("input", "", f"{r}_valid"),
              _port("input", f"[{IW - 1}:0]", f"{r}_id"),
              _port("output", f"[{IW}:0]", "free_count")]
    ports[-1] = ports[-1].rstrip(",")

    def live(k):
        return (f"(({head} <= {IW}'d{k}) ? (({CW}'d{k} - {head}) < {cnt}) : "
                f"(({CW}'d{k + N} - {head}) < {cnt}))")

    if s["prop"] == "offer_is_free":
        prop = f"!{a}_valid || !{busy}[{a}_id]"
        antecedents, said = [f"{a}_valid"], "an ID on offer is never in use"
    else:
        prop = (f"{cnt} + " + " + ".join(f"{busy}[{i}]" for i in range(N))
                + f" == {CW}'d{N}")
        antecedents, said = [], "free IDs and IDs in use always add up to all of them"
    facts = [f"!{live(k)} || !{busy}[{fl}[{k}]]" for k in range(N)]
    facts += [f"!({live(k)} && {live(m)}) || ({fl}[{k}] != {fl}[{m}])"
              for k in range(N) for m in range(k + 1, N)]
    facts += [f"{tail} == {head} + {cnt}[{IW - 1}:0]", f"{cnt} <= {CW}'d{N}"]

    v = [f"// A pool of {N} IDs: free IDs wait in a ring list; an allocation "
         "hands out the first one and marks it in use, and an ID in use goes "
         "back on the list when it is released, so no ID is ever out twice.",
         f"module {s['top']} ("] + ports + [
        ");",
        f"    localparam N = {N};",
        "",
        f"    reg [N-1:0] {busy};",
        f"    reg [{IW - 1}:0] {fl} [0:N-1];",
        f"    reg [{IW - 1}:0] {head};",
        f"    reg [{IW - 1}:0] {tail};",
        f"    reg [{IW}:0] {cnt};",
        "",
        f"    initial {busy} = {N}'d0;",
        f"    initial {head} = {IW}'d0;",
        f"    initial {tail} = {IW}'d0;",
        f"    initial {cnt} = {CW}'d{N};"] + [
        f"    initial {fl}[{k}] = {IW}'d{k};" for k in range(N)] + [
        "",
        f"    assign {a}_valid = ({cnt} != {CW}'d0);",
        f"    assign {a}_id = {fl}[{head}];",
        f"    assign free_count = {cnt};",
        f"    wire do_alloc = {a}_valid && {a}_ready;",
        f"    wire do_rel = {r}_valid && {busy}[{r}_id];",
        "",
        "    always @(posedge clk) begin",
        f"        if (rst{' || flush' if s['flush'] else ''}) begin",
        f"            {busy} <= {N}'d0;",
        f"            {head} <= {IW}'d0;",
        f"            {tail} <= {IW}'d0;",
        f"            {cnt} <= {CW}'d{N};"] + [
        f"            {fl}[{k}] <= {IW}'d{k};" for k in range(N)] + [
        "        end else begin",
        "            if (do_alloc) begin",
        f"                {busy}[{fl}[{head}]] <= 1'b1;",
        f"                {head} <= {head} + {IW}'d1;",
        "            end",
        "            if (do_rel) begin",
        f"                {busy}[{r}_id] <= 1'b0;",
        f"                {fl}[{tail}] <= {r}_id;",
        f"                {tail} <= {tail} + {IW}'d1;",
        "            end",
        "            if (do_alloc && !do_rel)",
        f"                {cnt} <= {cnt} - {CW}'d1;",
        "            else if (do_rel && !do_alloc)",
        f"                {cnt} <= {cnt} + {CW}'d1;",
        "        end",
        "    end",
        "",
        f"    // {said}",
        f"    always @(posedge clk) assert ({prop});",
        "endmodule",
        ""]
    return {"verilog": "\n".join(v), "top_module": s["top"], "clock": "clk",
            "property": [prop], "invariants": facts, "antecedents": antecedents,
            "sanity_covers": [f"{cnt} == {CW}'d{N - 2}"]}

KINDS = {"queue": (queue_settings, queue_design),
         "table": (table_settings, table_design),
         "arbiter": (arbiter_settings, arbiter_design),
         "pool": (pool_settings, pool_design)}

def judge(design, proof_root=None):
    """The checker that judges model-written designs, on this one: the
    gates, then the proof with the facts and without them. NECESSARY means
    the facts prove the property and the property alone does not."""
    sys.path.insert(0, str(HERE.parent / "initiator"))
    import run
    return run.judge(json.dumps(design), "ok", "globally", grade=True,
                     proof_root=proof_root, spin=False)


# how often each kind is drawn
KIND_WEIGHTS = {"queue": 3, "table": 2, "arbiter": 2, "pool": 2}


def design_settings(seed, i):
    """Design i of a batch, (kind, settings): its own seed, so a rerun or
    a bigger batch rebuilds exactly the designs it already has."""
    rng = random.Random(f"{seed}-{i}")
    kind = rng.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]
    return kind, KINDS[kind][0](rng)


def make_design(kind, settings):
    return KINDS[kind][1](settings)


def held_out(i):
    """Every 11th design is set aside, never trained on: our own storage
    test, to tell whether training taught the skill or only our style."""
    return i % 11 == 10


def rejected_path(out):
    """Where designs the checker turned away are kept, with its reason: in
    the logs folder beside `out` (extender/logs for the real corpus)."""
    out = Path(out)
    return out.parent / "logs" / f"rejected_{out.name}"


def _ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}


def build(n, seed=SEED, workers=4, out=OUT, proof_root=PROOFS):
    """Designs 0 to n-1 of the batch `seed`, each judged; a design the
    checker accepts goes to `out` as a corpus row, one it turns away to
    the rejected log with its reason. A design already in either is not
    built again, and two settings that write the same text count once."""
    out, rejected = Path(out), rejected_path(out)
    done = _ids(out) | _ids(rejected)
    todo, seen = [], set()
    for i in range(n):
        kind, s = design_settings(seed, i)
        d = make_design(kind, s)
        if d["verilog"] in seen:
            continue
        seen.add(d["verilog"])
        rid = f"storage_{kind}_{i:03d}"
        if rid not in done:
            todo.append((rid, kind, s, d))
    out.parent.mkdir(parents=True, exist_ok=True)
    rejected.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def one(item):
        rid, kind, s, d = item
        v = judge(d, proof_root=Path(proof_root) / rid)
        number = int(rid.rsplit("_", 1)[1])
        row = dict(d, id=rid, kind=kind, generation=0, settings=s,
                   held_out=held_out(number),
                   seed=f"{seed}-{int(rid.rsplit('_', 1)[1])}", source=SOURCES[kind],
                   verdict=v["verdict"], reason=v.get("reason"),
                   grade_wall_s=v.get("grade_wall_s"),
                   made_by="extender/storage_designs.py")
        with lock:
            with open(out if v["verdict"] == "NECESSARY" else rejected, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        size = s.get("depth") or s.get("entries")
        print(f"{rid} ({size}): {v['verdict']} ({v.get('grade_wall_s')} s)",
              flush=True)
        return row

    with ThreadPoolExecutor(max(1, workers)) as pool:
        rows = [f.result() for f in as_completed(
            [pool.submit(one, item) for item in todo])]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=10, help="designs 0 to n-1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry", action="store_true",
                    help="show the designs' settings and the first one; judge nothing")
    args = ap.parse_args(argv)
    if args.dry:
        for i in range(args.n):
            kind, s = design_settings(args.seed, i)
            print(f"storage_{kind}_{i:03d}: {s}")
        print(make_design(*design_settings(args.seed, 0))["verilog"])
        return
    rows = build(args.n, args.seed, args.workers, args.out)
    kept = sum(r["verdict"] == "NECESSARY" for r in rows)
    print(f"{kept} of {len(rows)} new designs kept -> {args.out}")


if __name__ == "__main__":
    main()
