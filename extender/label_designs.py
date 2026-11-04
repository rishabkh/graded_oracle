"""Label designs good, weak or wrong with a paid model (4 Nov 2026).

The biggest training set holds 217 starting designs nobody labeled; by
the share of wrong ones in the groups that were labeled, about 20 of them
are wrong, and wrong designs hurt training (seven-set experiment, 30 Oct
2026). This labels them with the rubric of the 27 Oct reading-only pass
(the label-every-design-first-pass workflow), one design a call, so a new
label means what an old one means. Calibrate first: --calibrate N labels
N designs that already have labels and prints how often the model agrees,
above all on the wrong ones.

  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --calibrate 30 --dry
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --calibrate 30
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py --dry
  LLM_PROVIDER=openai venv/bin/python extender/label_designs.py
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
CALIBRATION_OUT = HERE / "logs" / "label_calibration_2026-11-04.jsonl"
CALIBRATION_SEED = 20261104
OUTPUT_GUESS = 3000           # tokens a reply, thinking included: a guess
                              # until the calibration run measures it

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

SCHEMA = {"type": "object",
          "properties": {"label": {"type": "string", "enum": list(LABELS)},
                         "reason": {"type": "string"},
                         "concern": {"type": "string"},
                         "stored_data_facts": {"type": "boolean"}},
          "required": ["label", "reason", "concern", "stored_data_facts"],
          "additionalProperties": False}


def to_label(labels):
    """The starting designs nobody labeled, exactly as the biggest
    training set's "unlabeled" group picks them."""
    return ptd.keys_of("unlabeled", labels)


def question(d):
    props = d.get("property") or []
    props = [props] if isinstance(props, str) else props
    numbered = "\n".join(f"{i:4d}| {line}" for i, line in
                         enumerate(d["verilog"].splitlines(), 1))
    return (RUBRIC
            + f"\nWhat the design is meant to be: {d.get('description') or '(no description)'}\n"
            + f"Top module: {d['top_module']}\n"
            + "Property (what the checker proves):\n"
            + "".join(f"- {p}\n" for p in props)
            + "Helper facts:\n"
            + ("".join(f"- {f}\n" for f in d.get("invariants") or []) or "- (none)\n")
            + f"\nThe design, with line numbers:\n{numbered}\n")


def label_one(d, call=None, effort=EFFORT):
    """One design, one call. A reply with no usable label is kept with
    its error, and a rerun asks again."""
    call = call or llm_client.call_claude
    text, usage, stop = call(model=MODEL, max_tokens=MAX_TOKENS,
                             user=question(d), schema=SCHEMA, effort=effort)
    row = {"key": d["key"], "label": None,
           "first_pass": f"{llm_client.model_label(MODEL)}, reading only",
           "effort": effort, "usage": usage, "stop": stop}
    try:
        got = json.loads(text) if text else None
    except json.JSONDecodeError:
        got = None
    if isinstance(got, dict) and got.get("label") in LABELS:
        row.update({k: got.get(k) for k in
                    ("label", "reason", "concern", "stored_data_facts")})
    else:
        row["error"] = f"no usable label (stop: {stop})"
    return row


def already_labeled(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {r["key"] for r in map(json.loads, path.read_text().splitlines())
            if r.get("label")}


def calibration_keys(labels, n, seed=CALIBRATION_SEED):
    """n designs that already have labels, a third each good, weak and
    wrong, from the same kind of design as the ones to label (starting
    designs, the oldest corpus left out); those whose label was checked
    by simulation first, since those labels are the surest."""
    out = []
    for lab in LABELS:
        pool = [r for r in labels if r.get("label") == lab
                and r.get("steps_from_start") == 0
                and r["appears_in"][0]["file"] != ptd.OLDEST]
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


def estimate(designs, call=None):
    """What a run would cost, before anything is paid: the questions
    counted, the replies guessed at OUTPUT_GUESS tokens each."""
    tokens_in = sum(len(question(d)) for d in designs) // 3
    tokens_out = OUTPUT_GUESS * len(designs)
    cost = llm_client.dollars(llm_client.model_label(MODEL), tokens_in, tokens_out)
    print(f"{len(designs)} designs with {llm_client.model_label(MODEL)}, effort "
          f"{EFFORT}: about ${cost:.2f} ({tokens_in:,} tokens in; replies "
          f"guessed at {OUTPUT_GUESS:,} tokens each, thinking included)")
    return cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="label N designs that already have labels (a third "
                         "each good, weak and wrong) and say how often the "
                         "model agrees")
    ap.add_argument("--out", help="one row per design; a rerun carries on")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry", action="store_true",
                    help="price the run; call nothing")
    args = ap.parse_args(argv)

    labels = [json.loads(l) for l in open(ptd.LABELS)]
    designs = {json.loads(l)["key"]: json.loads(l) for l in open(ptd.DESIGNS)}
    keys = calibration_keys(labels, args.calibrate) if args.calibrate \
        else to_label(labels)
    out = Path(args.out or (CALIBRATION_OUT if args.calibrate else OUT))
    done = already_labeled(out)
    todo = [designs[k] for k in keys if k not in done][:args.limit or None]
    print(f"{len(keys)} designs, {len(keys) - len(todo)} already labeled in "
          f"{out.name}")
    if args.dry:
        estimate(todo)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with llm_client.call_log(out.with_name("calls_" + out.name)), \
            ThreadPoolExecutor(args.workers) as pool, open(out, "a") as fh:
        futs = {pool.submit(label_one, d, None, args.effort): d for d in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as exc:            # one design never stops the run
                row = {"key": futs[fut]["key"], "label": None, "error": repr(exc)}
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(f"[{i}/{len(todo)}] {row['key']}: {row['label'] or row.get('error')}",
                  flush=True)
    rows = {r["key"]: r for r in map(json.loads, out.read_text().splitlines())
            if r.get("label")}
    print("labels:", dict(Counter(r["label"] for r in rows.values())),
          f"- {len(keys) - len(rows)} still without one (a rerun asks again)")
    if args.calibrate:
        earlier = {r["key"]: r["label"] for r in labels}
        r = agreement([(earlier[k], rows[k]["label"]) for k in keys if k in rows])
        print(f"agrees with the earlier label on {r['agree']} of {r['total']}; "
              f"finds {r['wrong_found'][0]} of {r['wrong_found'][1]} known wrong "
              f"designs; {r['wrong_right'][0]} of its {r['wrong_right'][1]} "
              "'wrong' calls are right")
        for (e, m), n in sorted(r["table"].items()):
            print(f"  earlier {e:5s} -> now {m:5s}: {n}")


if __name__ == "__main__":
    main()
