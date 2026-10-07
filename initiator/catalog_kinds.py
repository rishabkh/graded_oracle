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

import llm_client                                             # noqa: E402

MODEL = "claude-opus-5"
EFFORT = "medium"
MAX_TOKENS = 32000
ASK = 200
PICK = 50
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
    args = p.parse_args(argv)

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
    dollars = usage.get("input", 0) * 5 / 1e6 + usage.get("output", 0) * 25 / 1e6
    print(f"{len(kinds)} kinds returned, {args.pick} picked with seed "
          f"{args.seed} (${dollars:.2f})")
    print(f"record: {out / f'kinds_{stamp}.json'}")
    print(f"kinds for the generator: {lines}")


if __name__ == "__main__":
    main()
