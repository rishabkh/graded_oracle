"""Choose new hardware kinds for the generator mechanically, so the
choice is not ours.

Why: we built a map of what the evaluation set contains (7 Oct 2026).
Having seen it, any kinds we picked by hand could be steered by it, and
training data shaped like the test makes the scores untrustworthy. So
the kinds come from one call to a fresh model, with no system prompt and
no other context, asking for a broad catalog of digital hardware. The
prompt names neither the test set nor any variety from the map (pinned
by tests/test_catalog_kinds.py); it lists our existing 32 kinds only so
the model leaves them out. Repeated names are dropped, then a draw fixed
by a seed written down beforehand picks the kinds used.

Everything needed to repeat or audit the choice is saved in one record:
model, effort, time, the exact prompt, the raw reply, the parsed list,
the seed, the rule and the picked kinds. The picked kinds are written as
generator seed lines, one per line, in the format of constructs.txt.

  venv/bin/python initiator/catalog_kinds.py --dry       # no call
  venv/bin/python initiator/catalog_kinds.py             # one paid call

--use-all RECORD (7 Oct 2026: 150 were drawn, then all 250 wanted) writes
every kind in a saved record as generator lines, the seeded draw first
and the rest in reply order. No call, and still no hand choice, since
every kind returned is used.

  venv/bin/python initiator/catalog_kinds.py --use-all initiator/catalog/kinds_<run>.json
"""
import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "extender"))

import llm_client                                             # noqa: E402

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 32000
ASK = 250
PICK = 150
SEED = 20261007          # the day the choice was made, fixed beforehand
RULE = ("drop repeated names (case-insensitive, first kept), then "
        "random.Random(seed).sample(kinds, pick)")
OUT_DIR = HERE / "catalog"
OURS = HERE / "constructs.txt"

SCHEMA = {
    "type": "object",
    "properties": {"kinds": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "description": {"type": "string"}},
        "required": ["name", "description"],
        "additionalProperties": False}}},
    "required": ["kinds"],
    "additionalProperties": False,
}

PROMPT = """\
List {n} distinct kinds of digital hardware building blocks. Together they
should form a broad, representative catalog of what digital designers
build: the kind of blocks found across digital design textbooks and
open-source hardware libraries.

For each kind give:
- name: a short name for the block;
- description: one sentence saying what the block does and what state it
  keeps (its registers or memories).

Rules:
- Every entry is a different kind of block, not a size or parameter
  variant of another entry.
- Describe the block only. Do not state any correctness property, rule or
  relationship that its state must satisfy.
- Leave out these kinds, which are already covered:
{ours}
"""


CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {"categories": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "description": {"type": "string"}},
        "required": ["name", "description"],
        "additionalProperties": False}}},
    "required": ["categories"],
    "additionalProperties": False,
}

CATEGORY_PROMPT = """\
List {n} distinct categories of digital hardware building blocks. Together
they should cover, broadly and representatively, what digital designers
build: the kinds of blocks found across digital design textbooks and
open-source hardware libraries.

For each category give:
- name: a short name;
- description: one sentence saying what the blocks in it do.

Every category must be clearly different from the others.
"""

KINDS_IN_CATEGORY_PROMPT = """\
List {k} distinct kinds of digital hardware building blocks in the
category "{category}": {description}

For each kind give:
- name: a short name for the block;
- description: one sentence saying what the block does and what state it
  keeps (its registers or memories).

Rules:
- Every entry is a different kind of block, not a size or parameter
  variant of another entry.
- Describe the block only. Do not state any correctness property, rule or
  relationship that its state must satisfy.
- Leave out these kinds, which are already covered:
{known}
"""


def build_category_prompt(n):
    return CATEGORY_PROMPT.format(n=n)


def build_kinds_prompt(k, category, known):
    return KINDS_IN_CATEGORY_PROMPT.format(
        k=k, category=category["name"], description=category["description"],
        known="\n".join(f"  - {x}" for x in known))


def parse_categories(text):
    obj = None
    if text:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = llm_client.extract_json(text)
    items = obj.get("categories") if isinstance(obj, dict) else None
    out, seen = [], set()
    for c in items or []:
        if isinstance(c, dict) and str(c.get("name", "")).strip() and \
                str(c.get("description", "")).strip() and \
                c["name"].strip().lower() not in seen:
            seen.add(c["name"].strip().lower())
            out.append({"name": c["name"].strip(),
                        "description": c["description"].strip()})
    return out


def _call(prompt, schema):
    return llm_client.call_claude(model=MODEL, max_tokens=MAX_TOKENS,
                                  user=prompt, schema=schema, effort=EFFORT)


def by_category(n_categories, per_category, known, call=None, workers=4,
                progress=None):
    """One call for the categories, then one per category, in parallel.
    Kinds whose name repeats one we have, or an earlier one, are dropped
    and listed; near repeats are judged later and applied by --finish."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    call = call or _call
    lock = threading.Lock()
    state = {"done": 0, "total": 1 + n_categories, "input": 0, "output": 0}

    def counted(prompt, schema):
        out = call(prompt, schema)
        with lock:
            state["done"] += 1
            state["input"] += out[1].get("input", 0)
            state["output"] += out[1].get("output", 0)
            if progress:
                progress(state["done"], state["total"],
                         {"input": state["input"], "output": state["output"]})
        return out
    cat_prompt = build_category_prompt(n_categories)
    cat_text, usage, _ = counted(cat_prompt, CATEGORY_SCHEMA)
    categories = parse_categories(cat_text)
    state["total"] = 1 + len(categories)
    prompts = [build_kinds_prompt(per_category, c, known) for c in categories]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(lambda q: counted(q, SCHEMA), prompts))
    total = {"input": usage.get("input", 0), "output": usage.get("output", 0)}
    seen = {k.strip().lower() for k in known}
    candidates, dropped = [], []
    for cat, (text, u, _) in zip(categories, answers):
        total["input"] += u.get("input", 0)
        total["output"] += u.get("output", 0)
        for k in parse(text) or []:
            key = k["name"].strip().lower()
            if key in seen:
                dropped.append(k["name"])
                continue
            seen.add(key)
            candidates.append(dict(k, category=cat["name"]))
    return {"timestamp": datetime.now(timezone.utc).isoformat(),
            "model": llm_client.model_label(MODEL), "effort": EFFORT,
            "max_tokens": MAX_TOKENS, "system_prompt": None,
            "prompts": {"categories": cat_prompt,
                        "kinds": {c["name"]: q for c, q in zip(categories, prompts)}},
            "replies": [cat_text] + [a[0] for a in answers],
            "usage": total, "categories": categories,
            "candidates": candidates, "dropped_repeats": dropped,
            "known_count": len(known)}


def known_kinds(out_dir, ours_file):
    """Every kind we already have: our original lines, and every kind any
    earlier picker run returned (drawn or not)."""
    known = [l.strip() for l in Path(ours_file).read_text().splitlines()
             if l.strip()]
    for rec in sorted(Path(out_dir).glob("kinds_*.json")):
        data = json.loads(rec.read_text())
        for key in ("kinds", "candidates"):
            for k in data.get(key) or []:
                known.append(k["name"])
    return list(dict.fromkeys(known))


def finish(record_path, verdicts_path, k, seed, out_dir):
    """Drop the candidates two judges found to repeat a kind we have or an
    earlier candidate, then draw `k` with the seed."""
    record_path = Path(record_path)
    rec = json.loads(record_path.read_text())
    verdicts = json.loads(Path(verdicts_path).read_text())
    cands = rec["candidates"]
    missing = [c["name"] for c in cands if c["name"] not in verdicts]
    if missing:
        sys.exit(f"no verdict for {len(missing)} candidates, e.g. "
                 f"{missing[:3]}; nothing was written")
    repeats = [c["name"] for c in cands if verdicts[c["name"]].get("repeat_of")]
    kept = [c for c in cands if not verdicts[c["name"]].get("repeat_of")]
    if len(kept) < k:
        sys.exit(f"only {len(kept)} kinds left after removing repeats, fewer "
                 f"than the {k} to keep; nothing was written")
    chosen = pick(kept, k, seed)
    out = Path(out_dir)
    lines = out / f"constructs_{rec['run_id']}_bycat.txt"
    lines.write_text("".join(construct_line(c) + "\n" for c in chosen))
    (out / f"kinds_{rec['run_id']}_bycat_finish.json").write_text(json.dumps({
        "from_record": record_path.name, "verdicts": Path(verdicts_path).name,
        "repeats_dropped": repeats, "kept": len(kept), "picked": k,
        "seed": seed, "chosen": chosen,
        "rule": ("drop candidates judged a repeat of a kind we have or of an "
                 "earlier candidate, then random.Random(seed).sample(kept, pick)"),
        "timestamp": datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f"{len(cands)} candidates, {len(repeats)} repeats dropped, {k} drawn "
          f"with seed {seed} -> {lines}")
    return lines


def build_prompt(n, ours):
    return PROMPT.format(n=n, ours="\n".join(f"  - {k}" for k in ours))


def parse(text):
    """The kinds in reply order, repeated names dropped; None when the
    reply is missing, malformed or empty."""
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = llm_client.extract_json(text)
    items = obj.get("kinds") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None
    out, seen = [], set()
    for k in items:
        if not (isinstance(k, dict) and isinstance(k.get("name"), str)
                and isinstance(k.get("description"), str)
                and k["name"].strip() and k["description"].strip()):
            continue
        key = k["name"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": k["name"].strip(),
                    "description": k["description"].strip()})
    return out or None


def pick(kinds, k, seed):
    return random.Random(seed).sample(kinds, k)


def construct_line(kind):
    """One generator seed line, in the format of constructs.txt."""
    return " ".join(f"{kind['name']}: {kind['description']}".split())


def use_all(record_path, out_dir):
    """Every kind in a saved record, the seeded draw first, then the rest
    in reply order; a small note records how the list was made."""
    record_path = Path(record_path)
    rec = json.loads(record_path.read_text())
    picked = rec["picked"]
    rest = [k for k in rec["kinds"] if k not in picked]
    kinds = picked + rest
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = out / f"constructs_{rec['run_id']}_all.txt"
    lines.write_text("".join(construct_line(k) + "\n" for k in kinds))
    note = {"from_record": record_path.name, "count": len(kinds),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rule": (f"every kind in the record: the {len(picked)} drawn with "
                     f"seed {rec['seed']} first, then the other {len(rest)} "
                     "in reply order; no new call")}
    (out / f"kinds_{rec['run_id']}_all.json").write_text(json.dumps(note, indent=1))
    print(f"{len(kinds)} kinds ({len(picked)} drawn + {len(rest)} more) -> {lines}")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Choose new hardware kinds with one call to a fresh model.")
    p.add_argument("--ask", type=int, default=ASK, help="kinds to ask for")
    p.add_argument("--pick", type=int, default=PICK, help="kinds to keep")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ours", default=str(OURS),
                   help="our existing kinds, one per line, to leave out")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--dry", action="store_true",
                   help="print the prompt; no call")
    p.add_argument("--use-all", default=None, metavar="RECORD",
                   help="write every kind in a saved record; no call")
    p.add_argument("--by-category", action="store_true",
                   help="categories first, then kinds per category (paid)")
    p.add_argument("--categories", type=int, default=40)
    p.add_argument("--per-category", type=int, default=25)
    p.add_argument("--finish", default=None, metavar="RECORD",
                   help="apply near-repeat verdicts to a by-category record "
                        "and draw --pick kinds; no call")
    p.add_argument("--verdicts", default=None)
    args = p.parse_args(argv)
    if args.use_all:
        use_all(args.use_all, args.out_dir)
        return
    if args.finish:
        if not args.verdicts:
            sys.exit("--finish needs --verdicts")
        finish(args.finish, args.verdicts, args.pick, args.seed, args.out_dir)
        return
    if args.by_category:
        known = known_kinds(args.out_dir, args.ours)
        if args.dry:
            print(build_category_prompt(args.categories))
            print(build_kinds_prompt(args.per_category,
                                     {"name": "<category>",
                                      "description": "<from the first call>"},
                                     known)[:1500] + "\n...")
            print(f"{1 + args.categories} calls; {len(known)} known kinds left out")
            return
        need = ("OPENROUTER_API_KEY" if llm_client.provider() == "openrouter"
                else "ANTHROPIC_API_KEY")
        if not os.environ.get(need):
            sys.exit(f"{need} is not set, so no call was made")
        from distractor import Spinner
        spinner = Spinner(f"picking kinds: 0/{1 + args.categories} calls done",
                          always=True)

        def progress(done, total, usage):
            cost = llm_client.dollars(MODEL, usage["input"], usage["output"])
            spinner.label = (f"picking kinds: {done}/{total} calls done, "
                             f"${cost:.2f} so far")
        with spinner:
            rec = by_category(args.categories, args.per_category, known,
                              progress=progress)
        rec["run_id"] = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"kinds_bycat_{rec['run_id']}.json"
        path.write_text(json.dumps(rec, indent=1))
        cost = llm_client.dollars(MODEL, rec["usage"]["input"],
                                  rec["usage"]["output"])
        print(f"{len(rec['categories'])} categories, {len(rec['candidates'])} "
              f"new candidate kinds, {len(rec['dropped_repeats'])} exact "
              f"repeats dropped (${cost:.2f}) -> {path}")
        print("next: two judges check near repeats; then --finish")
        return

    ours = [l.strip() for l in Path(args.ours).read_text().splitlines()
            if l.strip()]
    prompt = build_prompt(args.ask, ours)
    if args.dry:
        print(prompt)
        print(f"one call to {MODEL} (effort {EFFORT}); keeps {args.pick} of "
              f"the {args.ask} asked for, seed {args.seed}")
        return
    need = ("OPENROUTER_API_KEY" if llm_client.provider() == "openrouter"
            else "ANTHROPIC_API_KEY")
    if not os.environ.get(need):
        sys.exit(f"{need} is not set, so no call was made")

    from distractor import Spinner
    with Spinner(f"asking {MODEL} for {args.ask} kinds (one call)", always=True):
        text, usage, stop = llm_client.call_claude(
            model=MODEL, max_tokens=MAX_TOKENS, user=prompt, schema=SCHEMA,
            effort=EFFORT)
    kinds = parse(text)
    stamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    chosen = pick(kinds, args.pick, args.seed) \
        if kinds and len(kinds) >= args.pick else None
    record = {"run_id": stamp,
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "model": llm_client.model_label(MODEL), "effort": EFFORT,
              "max_tokens": MAX_TOKENS, "system_prompt": None,
              "prompt": prompt, "raw_reply": text, "stop": stop,
              "usage": usage, "asked": args.ask, "kinds": kinds,
              "pick": args.pick, "seed": args.seed, "rule": RULE,
              "picked": chosen, "ours_file": Path(args.ours).name}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"kinds_{stamp}.json").write_text(json.dumps(record, indent=1))
    if chosen is None:
        sys.exit(f"the reply gave {len(kinds or [])} usable kinds, fewer than "
                 f"the {args.pick} to keep; the reply is saved in "
                 f"kinds_{stamp}.json and no kind list was written")
    lines = out / f"constructs_{stamp}.txt"
    lines.write_text("".join(construct_line(k) + "\n" for k in chosen))
    dollars = llm_client.dollars(MODEL, usage.get("input", 0),
                                 usage.get("output", 0))
    print(f"{len(kinds)} kinds returned, {args.pick} picked with seed "
          f"{args.seed} (${dollars:.2f})")
    print(f"record: {out / f'kinds_{stamp}.json'}")
    print(f"kinds for the generator: {lines}")


if __name__ == "__main__":
    main()
