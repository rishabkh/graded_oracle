"""Label designs good, weak or wrong with a paid model (4 Nov 2026).

The biggest training set holds 217 starting designs nobody labeled; by
the share of wrong ones in the groups that were labeled, about 20 of them
are wrong, and wrong designs hurt training (seven-set experiment, 30 Oct
2026). This labels them with the rubric of the 27 Oct reading-only pass
(the label-every-design-first-pass workflow), one design a call, so a new
label means what an old one means. Calibrate first: --calibrate N labels
N designs that already have labels and prints how often the model agrees,
above all on the wrong ones.

--remaining (5 Nov 2026) labels every design still without a label after
that: September's grown designs, the oldest corpus and October's grown
ones. A grown design is judged against the design it grew from, with the
27 Oct rubric's parts for grown designs, in rounds: designs with no parent
first, then one step from a start, and so on, so each question carries
its parent's label even when this run made it.

  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --calibrate 30 --dry
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --calibrate 30
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --calibrate 30 --grown
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --calibrate 30 --grown --fresh
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --remaining
"""
import argparse
import json
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "training"))
import llm_client                                                # noqa: E402
import pick_training_designs as ptd                              # noqa: E402

MODEL = "claude-opus-5-5"     # LLM_PROVIDER=openai calls OPENAI_MODEL instead
EFFORT = "medium"
MAX_TOKENS = 16000
LABELS = ("good", "weak", "wrong")
OUT = HERE / "labels_unlabeled_starting_designs_2026-11-04.jsonl"
OUT_REMAINING = HERE / "labels_remaining_540_designs_2026-11-05.jsonl"
CALIBRATION_OUT = HERE / "logs" / "label_calibration_2026-11-04.jsonl"
CALIBRATION_GROWN_OUT = HERE / "logs" / "label_calibration_grown_2026-11-05.jsonl"
CALIBRATION_GROWN_FRESH_OUT = HERE / "logs" / \
    "label_calibration_grown_fresh_designs_clarified_rules_2026-11-05.jsonl"
CALIBRATION_SEED = 20261104
# tokens a reply, thinking included: measured on the 217 starting designs
# (4 Nov 2026: mean 178, max 567), rounded up for the longer grown ones
OUTPUT_GUESS = 300

# The 27 Oct first-pass rubric for one starting design: its context,
# failure checks and labels, word for word, without the parts about
# batches, files and grown designs.
RUBRIC = """\
You are labeling hardware designs for a formal-verification research project.

Context: our pipeline makes SystemVerilog designs, each with a property (what our checker proves) and helper facts that make the property provable. This is a starting design, written from scratch by the "initiator", and it passed our checker. Designs like it are training examples: a model sees the design and property and learns to write the helper facts. We want to know which designs are GOOD examples of realistic hardware to train on. Judge the design itself.

Failures to check for by reading the code (the property usually cannot see them):
1. When a buffer, queue or table is full or busy, is the sender told to wait, or is an item dropped or overwritten?
2. Can a handshake lose an item or deliver it twice (valid without ready, ready ignored, an item taken while paused, a push and pop in the same cycle miscounted)?
3. Can any part get stuck forever or starve? A receiver is allowed to wait until it sees valid before it raises ready; a design that freezes with such a receiver is wrong (for example a valid that depends on ready, or a side queue that only empties on another output's ready). Also check that credits or tokens come back after an abort, flush, error or reset.

Labels:
- good: a real hardware feature or block a real chip would have, built sensibly, with facts that state real relations about its state (counts against pointers, stored entries against what was written, who holds a token, a state machine's phase against its counters).
- weak: little real function: a trivial or decorative design, counters that only watch, debug words built by XOR, a duplicate of an existing counter or register that the property then compares with the original, one block copied per slot instead of an array, another buffer piled on the same path, or facts that are only trivial bounds or copies.
- wrong: loses or duplicates data, can freeze or starve, even though the property cannot see it. Only say wrong when you can point to exact lines and a short input scenario that shows it.

Be strict and concrete; when torn between good and weak, choose weak. Reasons are plain English, one sentence, for a reader who is not a hardware specialist. Keep write-ups short.

Reply with JSON: {"label": "good" or "weak" or "wrong", "reason": "...", "concern": "for wrong: exact lines and scenario; for good or weak: the closest call you checked", "stored_data_facts": true or false (does any fact relate stored entries' contents to something, not just counts or bounds?)}
"""

# The same rubric's parts for a grown design, word for word: the extender's
# steps, judging the step against its parent, a bug the parent passes on,
# the fourth check, and the grown-design clauses of each label. Plus two
# points after checks 3 and 4 (5 Nov 2026, Rishab's yes): in the grown
# calibration only 10 of GPT's 20 "wrong" calls held, and all 10 misses had
# labels checked by simulation. GPT read a clock enable as a local pause,
# and called normal design choices "breaking what the design used to do";
# the simulating reviewers had judged both fine.
RUBRIC_GROWN = """\
You are labeling hardware designs for a formal-verification research project.

Context: our pipeline makes SystemVerilog designs, each with a property (what our checker proves) and helper facts that make the property provable. A starting design is written from scratch by the "initiator"; a grown design comes from an earlier design through one step of the "extender" (add a sub-unit, storage, a peer lane, combine two designs, a small tweak such as splitting a register, adding a bound, a pipeline stage, a counter or a copy of a block, or a "distractor" step that adds logic the property does not need). This is a grown design, and it passed our checker. Designs like it are training examples: a model sees the design and property and learns to write the helper facts. We want to know which designs are GOOD examples of realistic hardware to train on.

Judge the step against the design it grew from (and its partner, for a combination of two designs). If it still carries a bug that makes the design it grew from wrong, label it wrong and set inherited_from to that design's key.

Failures to check for by reading the code (the property usually cannot see them):
1. When a buffer, queue or table is full or busy, is the sender told to wait, or is an item dropped or overwritten?
2. Can a handshake lose an item or deliver it twice (valid without ready, ready ignored, an item taken while paused, a push and pop in the same cycle miscounted)?
3. Can any part get stuck forever or starve? A receiver is allowed to wait until it sees valid before it raises ready; a design that freezes with such a receiver is wrong (for example a valid that depends on ready, or a side queue that only empties on another output's ready). Also check that credits or tokens come back after an abort, flush, error or reset.
4. Does it change what the old design did with the same inputs (old outputs different, old inputs ignored, an old setting with a new side effect, a value cut to fewer bits or used against its meaning)?

Two points on checks 3 and 4:
- A clock-enable input (such as en, ce or i_ce) pauses the whole system, the sender and the receiver included: a ready or valid that stays high while it is low is not a lost or repeated item.
- A step may delay something by a cycle, apply new settings when an item leaves rather than when it arrives, keep a log that overwrites its oldest record while counting the loss, or let a clear or flush also drop items still waiting. These are normal design choices, not breaking what the design used to do. Breaking it means a wrong value, a lost or repeated item, an old input ignored, or an old setting with a new side effect.

Labels:
- good: a real hardware feature or block a real chip would have, built sensibly (wired into the old design sensibly, old behaviour intact), with facts that state real relations about its state (counts against pointers, stored entries against what was written, who holds a token, a state machine's phase against its counters).
- weak: little real function: a trivial or decorative design, a small tweak (split a register, add a bound, a pass-through register), counters that only watch, debug words built by XOR, a duplicate of an existing counter or register that the property then compares with the original, one block copied per slot instead of an array, another buffer piled on the same path, a combination that makes no hardware sense, or facts that are only trivial bounds or copies.
- wrong: loses or duplicates data, can freeze or starve, or breaks what the design used to do, even though the property cannot see it. Only say wrong when you can point to exact lines and a short input scenario that shows it.

Be strict and concrete; when torn between good and weak, choose weak. Reasons are plain English, one sentence, for a reader who is not a hardware specialist. Keep write-ups short.

Reply with JSON: {"label": "good" or "weak" or "wrong", "reason": "...", "concern": "for wrong: exact lines and scenario; for good or weak: the closest call you checked", "stored_data_facts": true or false (does any fact relate stored entries' contents to something, not just counts or bounds?), "inherited_from": "the key of the design whose bug it still carries, or empty"}
"""

SCHEMA = {"type": "object",
          "properties": {"label": {"type": "string", "enum": list(LABELS)},
                         "reason": {"type": "string"},
                         "concern": {"type": "string"},
                         "stored_data_facts": {"type": "boolean"}},
          "required": ["label", "reason", "concern", "stored_data_facts"],
          "additionalProperties": False}
GROWN_SCHEMA = {"type": "object",
                "properties": dict(SCHEMA["properties"],
                                   inherited_from={"type": "string"}),
                "required": SCHEMA["required"] + ["inherited_from"],
                "additionalProperties": False}


def to_label(labels):
    """The starting designs nobody labeled, exactly as the biggest
    training set's "unlabeled" group picks them."""
    return ptd.keys_of("unlabeled", labels)


def remaining(labels, labeled_since):
    """Every design still without a label, parents first: by steps from
    a start, then by key."""
    return [r["key"] for r in sorted(labels, key=lambda r: (
        r.get("steps_from_start") or 0, r["key"]))
        if not r["label"] and r["key"] not in labeled_since]


def _numbered(verilog):
    return "\n".join(f"{i:4d}| {line}" for i, line in
                     enumerate(verilog.splitlines(), 1))


def question(d, parent=None, partner=None, parent_label=None):
    """One design's question. With a parent, the grown-design rubric and
    the designs it came from."""
    props = d.get("property") or []
    props = [props] if isinstance(props, str) else props
    q = (RUBRIC_GROWN if parent else RUBRIC) \
        + f"\nWhat the design is meant to be: {d.get('description') or '(no description)'}\n"
    if parent:
        step = " ".join(x for x in (d.get("ext_type"), d.get("move")) if x)
        q += f"The step that made it: {step or '(not saved)'}"
        q += f" - {d['added']}\n" if d.get("added") else "\n"
    q += (f"Top module: {d['top_module']}\n"
          + "Property (what the checker proves):\n"
          + "".join(f"- {p}\n" for p in props)
          + "Helper facts:\n"
          + ("".join(f"- {f}\n" for f in d.get("invariants") or []) or "- (none)\n")
          + f"\nThe design, with line numbers:\n{_numbered(d['verilog'])}\n")
    if parent:
        q += (f"\nThe design it grew from (key {parent['key']}), with line "
              f"numbers:\n{_numbered(parent['verilog'])}\n")
        if parent_label and parent_label.get("label"):
            q += (f"\nThe design it grew from was labeled {parent_label['label']}"
                  f" earlier: {parent_label.get('reason') or ''}")
            if parent_label["label"] == "wrong" and parent_label.get("concern"):
                q += f" ({parent_label['concern']})"
            q += "\n"
    if partner:
        q += (f"\nThe partner design it was combined with (key {partner['key']}),"
              f" with line numbers:\n{_numbered(partner['verilog'])}\n")
    return q


def label_one(d, call=None, effort=EFFORT, parent=None, partner=None,
              parent_label=None):
    """One design, one call. A reply with no usable label is kept with
    its error, and a rerun asks again."""
    call = call or llm_client.call_claude
    grown = parent is not None
    text, usage, stop = call(model=MODEL, max_tokens=MAX_TOKENS,
                             user=question(d, parent, partner, parent_label),
                             schema=GROWN_SCHEMA if grown else SCHEMA,
                             effort=effort)
    row = {"key": d["key"], "label": None,
           "first_pass": f"{llm_client.model_label(MODEL)}, reading only",
           "effort": effort, "usage": usage, "stop": stop}
    if grown:
        row["parent_key"] = parent["key"]
        row["parent_label"] = (parent_label or {}).get("label")
    try:
        got = json.loads(text) if text else None
    except json.JSONDecodeError:
        got = None
    if isinstance(got, dict) and got.get("label") in LABELS:
        fields = ("label", "reason", "concern", "stored_data_facts") + \
            (("inherited_from",) if grown else ())
        row.update({k: got.get(k) for k in fields})
    else:
        row["error"] = f"no usable label (stop: {stop})"
    return row


def label_in_rounds(keys, designs, labels, known, call=None, effort=EFFORT,
                    workers=8, on_row=None):
    """Designs with no parent first, then one step from a start, and so
    on, so every grown design's question carries its parent's label, even
    one made earlier in this run. `labels` holds each design's place in
    its lineage (and any earlier label), `known` the labels made since,
    which this fills as it goes."""
    rounds = {}
    for k in keys:
        rounds.setdefault(labels[k].get("steps_from_start") or 0, []).append(k)
    out = []
    for step in sorted(rounds):
        with ThreadPoolExecutor(max(1, workers)) as pool:
            futs = {pool.submit(_label_key, k, designs, labels, known, call,
                                effort): k for k in rounds[step]}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as exc:        # one design never stops the run
                    row = {"key": futs[fut], "label": None, "error": repr(exc)}
                if row.get("label"):
                    known[row["key"]] = row
                out.append(row)
                if on_row:
                    on_row(row)
    return out


def _label_key(k, designs, labels, known, call, effort):
    meta = labels[k]
    parent = designs.get(meta.get("parent_key") or "")
    if parent is None:
        return label_one(designs[k], call=call, effort=effort)
    earlier = labels.get(parent["key"]) or {}
    parent_label = known.get(parent["key"]) or (earlier if earlier.get("label") else None)
    partner = designs.get(meta.get("partner_key") or "")
    return label_one(designs[k], call=call, effort=effort, parent=parent,
                     partner=partner, parent_label=parent_label)


def call_log_path(out):
    """Every call of a run, saved in the log folder git leaves out (5 Nov
    2026: the 217 and 540 runs had put 9 MB of requests in the repo)."""
    return HERE / "logs" / ("calls_" + Path(out).name)


def already_labeled(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {r["key"] for r in map(json.loads, path.read_text().splitlines())
            if r.get("label")}


def _labeled_rows(path):
    path = Path(path)
    if not path.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, path.read_text().splitlines())
            if r.get("label")}


def calibration_keys(labels, n, seed=CALIBRATION_SEED, grown=False,
                     exclude=()):
    """n designs that already have labels, a third each good, weak and
    wrong, from the same kind of design as the ones to label (starting or
    grown designs, the oldest corpus left out); those whose label was
    checked by simulation first, since those labels are the surest.
    `exclude`: designs an earlier calibration used, for a fresh sample."""
    out = []
    for lab in LABELS:
        pool = [r for r in labels if r.get("label") == lab
                and ((r.get("steps_from_start") or 0) > 0) == grown
                and r["appears_in"][0]["file"] != ptd.OLDEST
                and r["key"] not in exclude]
        simulated = sorted(r["key"] for r in pool if r.get("checked_by_simulation"))
        rest = sorted(r["key"] for r in pool if not r.get("checked_by_simulation"))
        random.Random(seed).shuffle(simulated)
        random.Random(seed).shuffle(rest)
        out += (simulated + rest)[:n // 3]
    return out


def agreement(pairs):
    """(earlier label, model's label) pairs: how often they match, how
    many known-wrong designs the model calls wrong, and how many of its
    "wrong" calls are right - the two numbers that decide whether its
    labels can take designs out of training."""
    known_wrong = [m for e, m in pairs if e == "wrong"]
    called_wrong = [e for e, m in pairs if m == "wrong"]
    return {"total": len(pairs), "agree": sum(e == m for e, m in pairs),
            "wrong_found": (known_wrong.count("wrong"), len(known_wrong)),
            "wrong_right": (called_wrong.count("wrong"), len(called_wrong)),
            "table": Counter(pairs)}


def estimate(items, call=None):
    """What a run would cost, before anything is paid: the questions
    counted (designs, or their questions already built), the replies at
    OUTPUT_GUESS tokens each."""
    texts = [x if isinstance(x, str) else question(x) for x in items]
    tokens_in = sum(len(t) for t in texts) // 3
    tokens_out = OUTPUT_GUESS * len(texts)
    cost = llm_client.dollars(llm_client.model_label(MODEL), tokens_in, tokens_out)
    print(f"{len(texts)} designs with {llm_client.model_label(MODEL)}, effort "
          f"{EFFORT}: about ${cost:.2f} ({tokens_in:,} tokens in; replies "
          f"at about {OUTPUT_GUESS:,} tokens each, thinking included)")
    return cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="label N designs that already have labels (a third "
                         "each good, weak and wrong) and say how often the "
                         "model agrees")
    ap.add_argument("--grown", action="store_true",
                    help="with --calibrate: grown designs, not starting ones")
    ap.add_argument("--fresh", action="store_true",
                    help="with --calibrate --grown: leave out the designs the "
                         "first grown calibration used")
    ap.add_argument("--remaining", action="store_true",
                    help="label every design still without a label "
                         "(September's, the oldest corpus, October's grown "
                         "ones), parents first")
    ap.add_argument("--out", help="one row per design; a rerun carries on")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry", action="store_true",
                    help="price the run; call nothing")
    args = ap.parse_args(argv)

    label_list = [json.loads(l) for l in open(ptd.LABELS)]
    labels = {r["key"]: r for r in label_list}
    designs = {json.loads(l)["key"]: json.loads(l) for l in open(ptd.DESIGNS)}
    known = {} if args.calibrate else _labeled_rows(OUT)
    if args.calibrate:
        used = set(_labeled_rows(CALIBRATION_GROWN_OUT)) if args.fresh else set()
        keys = calibration_keys(label_list, args.calibrate, grown=args.grown,
                                exclude=used)
        default = CALIBRATION_GROWN_FRESH_OUT if args.fresh else \
            CALIBRATION_GROWN_OUT if args.grown else CALIBRATION_OUT
    elif args.remaining:
        keys, default = remaining(label_list, labeled_since=set(known)), OUT_REMAINING
    else:
        keys, default = to_label(label_list), OUT
    out = Path(args.out or default)
    done = _labeled_rows(out)
    known.update(done)
    todo = [k for k in keys if k not in done][:args.limit or None]
    print(f"{len(keys)} designs, {len(keys) - len(todo)} already labeled in "
          f"{out.name}")
    if args.dry:
        def asked(k):
            parent = designs.get(labels[k].get("parent_key") or "")
            partner = designs.get(labels[k].get("partner_key") or "")
            return question(designs[k], parent, partner)
        estimate([asked(k) for k in todo])
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    count = [0]
    with llm_client.call_log(call_log_path(out)), \
            open(out, "a") as fh:
        def write(row):
            with lock:
                count[0] += 1
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"[{count[0]}/{len(todo)}] {row['key']}: "
                      f"{row['label'] or row.get('error')}", flush=True)
        label_in_rounds(todo, designs, labels, known, effort=args.effort,
                        workers=args.workers, on_row=write)
    rows = _labeled_rows(out)
    print("labels:", dict(Counter(rows[k]["label"] for k in keys if k in rows)),
          f"- {sum(k not in rows for k in keys)} still without one (a rerun "
          "asks again)")
    if args.calibrate:
        r = agreement([(labels[k]["label"], rows[k]["label"]) for k in keys
                       if k in rows])
        print(f"agrees with the earlier label on {r['agree']} of {r['total']}; "
              f"finds {r['wrong_found'][0]} of {r['wrong_found'][1]} known wrong "
              f"designs; {r['wrong_right'][0]} of its {r['wrong_right'][1]} "
              "'wrong' calls are right")
        for (e, m), n in sorted(r["table"].items()):
            print(f"  earlier {e:5s} -> now {m:5s}: {n}")


if __name__ == "__main__":
    main()
