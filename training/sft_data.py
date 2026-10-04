"""Data side of the fine-tune, kept free of torch so it can be tested
on a laptop: read the pairs, make chat turns, split, mask the question.

The corpus pairs are (design + property) -> (invariant list). Per Nada's
answer we train on answers alone first; reasoning is a second run, so
nothing here touches the reasoning fields.
"""
import json
import random
from pathlib import Path

IGNORE = -100     # what the loss skips, the HuggingFace convention


def load_pairs(path):
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("prompt") or not row.get("completion"):
            raise ValueError(f"{path} line {n}: prompt and completion "
                             "must both be non-empty")
        pair = {"prompt": row["prompt"], "completion": row["completion"]}
        if "holdout" in row:
            pair["holdout"] = bool(row["holdout"])
        rows.append(pair)
    return rows


def to_messages(pair):
    return [{"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["completion"]}]


def split(rows, holdout=0.1, seed=0):
    """A small held-out slice for watching the loss only. The real scores
    come from riscv-formal and the Technion set, never from here.

    A file whose rows all carry a `holdout` mark keeps its own choice:
    v3 dropped 6 rows from v2's and still holds back v2's rows, which a
    fresh shuffle of the shorter list would not."""
    marked = [("holdout" in r) for r in rows]
    if any(marked):
        if not all(marked):
            raise ValueError("some rows carry a holdout mark and some do "
                             "not; mark every row or none")
        return ([r for r in rows if not r["holdout"]],
                [r for r in rows if r["holdout"]])
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = int(len(shuffled) * holdout)
    return shuffled[n:], shuffled[:n]


def mask_labels(input_ids, prompt_len, ignore=IGNORE):
    """Train on the answer only: the question's tokens are hidden from
    the loss, so the model is never rewarded for reciting the design."""
    if prompt_len > len(input_ids):
        raise ValueError(f"prompt_len {prompt_len} exceeds row length "
                         f"{len(input_ids)}")
    return [ignore] * prompt_len + list(input_ids[prompt_len:])
