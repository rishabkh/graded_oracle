"""The generator's training file (14 Nov 2026, Rishab's "let's do it").

Formal Disco's first step for its open model (formal-disco/distill.py,
build_sft_records): before the open model runs as a worker it is trained
on the stronger model's own calls, every call that ended in a program the
checker passed, the question as it was asked and the answer as it was
given. build_sft.generation_records has applied those rules to our saved
calls (distill_log.py) since 8 Oct 2026, and this file uses it as it is.

What this file adds:
- Only calls whose design is labeled good or weak. Our checker cannot see
  a design that does the wrong thing, so the labels decide: 129 of the 701
  saved calls that passed made a design labeled wrong, and October's
  design-writing model, which learned from every passed call, wrote 15
  wrong designs among its own 24 that passed. A design nobody labeled is
  left out and counted: label it first.
- The facts file a solver was trained on goes in first, line for line, so
  the model keeps the skill the test scores. A call that shows one of that
  file's held-back designs is held back too.
- No idea calls: today's initiator writes the design in one step.
- `--test-kinds`: a call for a kind set aside for the generation test is
  left out, so that test stays fair. A growing step records no kind of
  its own; it is kept.

A label file names designs by key (extender/all_designs.key_of: the design
text with its spacing squeezed) or, written LABELS=CORPUS, by the ids of
that corpus's rows, as label_designs.py --corpus writes them. A later
file's label wins.

  venv/bin/python training/build_generator_set.py \\
      --facts extender/sft_train_<the facts set>.jsonl \\
      --calls initiator/logs/distill_<run>.jsonl \\
              extender/logs/<run>/distill_extensions.jsonl \\
      --labels extender/labels_<by key>.jsonl \\
               extender/labels_<by id>.jsonl=extender/corpus_<run>.jsonl \\
      --test-kinds initiator/logs/kinds_unseen_100_even_2026-10-12.txt \\
      --out extender/sft_train_generator_<what it holds>_<date>.jsonl
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))
sys.path.insert(0, str(HERE.parent / "extender"))
sys.path.insert(0, str(HERE))

import build_sft                                              # noqa: E402
import distill_log                                            # noqa: E402
from all_designs import key_of                                # noqa: E402
from distractor import Spinner                                # noqa: E402
from solver_baseline import SOLVER_PROMPT                     # noqa: E402

# the jobs a call is kept for: writing a design, mending one, growing one
JOBS = ("initiate", "generate", "implement", "repair", "extend")
KEPT_LABELS = ("good", "weak")
MAX_LEN = 16384               # train_lora.py's own limit
_DESIGN_KEY = re.compile(r"[0-9a-f]{12}")


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def load_labels(specs):
    """{design key: label} from label files, a later file winning. A spec
    is a label file whose keys are design keys, or LABELS=CORPUS for one
    whose keys are the ids of that corpus's rows. A row with no label says
    nothing. A file of ids given without its corpus, or an id its corpus
    does not have, stops the run: a label that silently matched no design
    would only show up as a design "nobody labeled"."""
    labels = {}
    for spec in specs:
        path, _, corpus = str(spec).partition("=")
        by_id = ({str(r.get("id")): key_of(r["verilog"])
                  for r in _rows(corpus) if r.get("verilog")}
                 if corpus else None)
        unknown = []
        for row in _rows(path):
            if not row.get("label"):
                continue
            key = str(row.get("key"))
            if by_id is None:
                if not _DESIGN_KEY.fullmatch(key):
                    sys.exit(f"{path} names designs by id (such as {key}), "
                             f"not by design key: write it as "
                             f"{path}=<its corpus file>; nothing was written")
            elif key in by_id:
                key = by_id[key]
            else:
                unknown.append(key)
                continue
            labels[key] = row["label"]
        if unknown:
            sys.exit(f"{path}: {corpus} has no row for "
                     f"{', '.join(unknown)}; nothing was written")
    return labels


def design_of_call(example):
    """The design a call made: the answer's own, or for a growing step,
    whose answer is a patch, the grown design saved beside it."""
    if example.get("prompt") == "extend":
        return (example.get("metadata") or {}).get("child_verilog") or None
    return build_sft.design_of(example.get("response"))


def kind_name(line):
    """A kind's name: a kinds list writes `Name: what it does.`"""
    return str(line).split(":")[0].strip().lower()


def kind_names(path):
    return {kind_name(line) for line in Path(path).read_text().splitlines()
            if line.strip()}


def _attempt(example):
    meta = example.get("metadata") or {}
    return meta.get("run_id"), meta.get("attempt")


def _tally(calls, labels):
    return {"kept": collections.Counter(c.get("prompt") for c in calls),
            "labels": collections.Counter(
                labels.get(key_of(design_of_call(c))) for c in calls),
            "writers": collections.Counter(
                str((c.get("metadata") or {}).get("model")) for c in calls)}


def choose_calls(examples, labels, test_kinds=frozenset()):
    """The calls worth learning from, in input order, and a report: how
    many were kept by job, by their design's label and by the model that
    wrote them, and how many were left out and why. A two-step design's
    kind is on its idea call, the one with the same run and attempt."""
    idea_kind = {_attempt(e): e["arguments"]["construct"] for e in examples
                 if e.get("prompt") == "idea" and _attempt(e)[0] is not None
                 and (e.get("arguments") or {}).get("construct")}
    kept, left_out = [], collections.Counter()
    for e in examples:
        job = e.get("prompt")
        if job == "idea":
            left_out["idea"] += 1
        elif job not in JOBS:
            left_out["another job"] += 1
        elif e.get("outcome") != "success":
            left_out["failed"] += 1
        elif not design_of_call(e):
            left_out["no design"] += 1
        else:
            kind = ((e.get("arguments") or {}).get("construct")
                    or idea_kind.get(_attempt(e)))
            label = labels.get(key_of(design_of_call(e)))
            if kind and kind_name(kind) in test_kinds:
                left_out["test kind"] += 1
            elif label in KEPT_LABELS:
                kept.append(e)
            elif label is None:
                left_out["no label"] += 1
            else:
                left_out[str(label)] += 1
    return kept, dict(_tally(kept, labels), left_out=left_out)


def held_designs(facts_lines):
    """The designs the facts file holds back, read out of its held-back
    questions (the solver's question with the design filled in)."""
    rows = [json.loads(line) for line in facts_lines]
    if not all("holdout" in r for r in rows):
        sys.exit("the facts file needs a held-back mark on every line: the "
                 "trainer wants a mark on every line or on none, and every "
                 "call carries one; nothing was written")
    before, after = SOLVER_PROMPT.format(verilog="\0").split("\0")
    designs = set()
    for n, row in enumerate(rows, 1):
        if not row["holdout"]:
            continue
        q = row.get("prompt")
        if not (isinstance(q, str) and q.startswith(before)
                and q.endswith(after) and len(q) > len(before) + len(after)):
            sys.exit(f"line {n} of the facts file is held back but is not "
                     "a solver question, so the design it holds back cannot "
                     "be read; nothing was written")
        designs.add(q[len(before):len(q) - len(after)])
    return designs


def build(facts_lines, examples, labels, test_kinds=frozenset()):
    """The file's lines - the facts lines as they are, then one line for
    each kept call under Formal Disco's rules - and the report."""
    held = held_designs(facts_lines)
    chosen, report = choose_calls(examples, labels, test_kinds)
    fact_report = {}
    records, counts = build_sft.generation_records(chosen, held,
                                                   fact_report=fact_report)
    # generation_records leaves out an answer with a fact the scoring
    # checker cannot read, or a call saved without its question. Each
    # record's question is its call's own message list, so the calls that
    # made it through are found again by walking both lists in order.
    made_it, left = [], iter(records)
    record = next(left, None)
    for call in chosen:
        if record is not None and record["prompt"] is (
                call.get("metadata") or {}).get("messages"):
            made_it.append(call)
            record = next(left, None)
    if record is not None or len(made_it) != len(records):
        raise RuntimeError("a kept record matched no chosen call")
    unreadable = len(fact_report["left_out"])
    skipped = sum(c["skipped"] for c in counts.values())
    left_out = report["left_out"]
    if unreadable:
        left_out["unreadable fact"] += unreadable
    if skipped - unreadable:
        left_out["no saved question"] += skipped - unreadable
    report = dict(_tally(made_it, labels), left_out=left_out,
                  held_back=sum(r["holdout"] for r in records))
    return list(facts_lines) + [json.dumps(r) for r in records], report


def _counts(counter):
    return ", ".join(f"{k} {n}" for k, n in counter.most_common()) or "none"


def main(argv=None):
    p = argparse.ArgumentParser(
        description="The generator's training file: a facts file, line for "
                    "line, plus the saved calls whose design passed the "
                    "checker and is labeled good or weak.")
    p.add_argument("--facts", required=True,
                   help="the facts file a solver was trained on")
    p.add_argument("--calls", nargs="+", required=True,
                   help="saved calls (distill_*.jsonl) of the initiator and "
                        "the extender")
    p.add_argument("--labels", nargs="+", required=True,
                   help="label files, by design key, or LABELS=CORPUS for "
                        "one that names designs by that corpus's ids; a "
                        "later file wins")
    p.add_argument("--test-kinds", default=None,
                   help="kinds list set aside for the generation test; "
                        "calls for these kinds are left out")
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=MAX_LEN,
                   help="the trainer's length limit, for the warning")
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; nothing was written")
    with Spinner("building the generator's training file", always=True):
        facts = [line for line in Path(args.facts).read_text().splitlines()
                 if line.strip()]
        examples = [e for f in args.calls for e in distill_log.read(f)]
        labels = load_labels(args.labels)
        kinds = kind_names(args.test_kinds) if args.test_kinds else frozenset()
        lines, report = build(facts, examples, labels, kinds)
    held = sum(json.loads(line)["holdout"] for line in facts)
    kept = sum(report["kept"].values())
    print(f"facts file: {len(facts)} lines kept as they are ({held} held back)")
    print(f"{len(examples)} saved calls in {len(args.calls)} file(s):")
    print(f"  kept {kept}: {_counts(report['kept'])}")
    print(f"  their designs: {_counts(report['labels'])}")
    print(f"  written by: {_counts(report['writers'])}")
    print(f"  left out: {_counts(report['left_out'])}")
    print(f"  {report['held_back']} kept call(s) held back, because they "
          "show a design the facts file holds back")
    build_sft.report_lengths([json.loads(line) for line in lines],
                             args.max_len)
    out.write_text("".join(line + "\n" for line in lines))
    print(f"{len(lines)} lines -> {out}")


if __name__ == "__main__":
    main()
