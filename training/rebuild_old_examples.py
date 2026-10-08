"""Our earlier design-writing calls as training examples in Formal
Disco's shape.

The earlier generation log (initiator/logs/attempts.jsonl: the runs of
August-September 2026 whose designs started the training set of models
v1 to v4) kept each design's seeds (README, kind, pattern, scope, style,
worked example) but not the messages that were sent. So each question is rebuilt from its seeds with
today's prompt, as Formal Disco rebuilds every example from its stored
arguments (formal-disco/distill.py build_sft_records). The wording may
differ from what the model saw then; every example says "rebuilt" in its
metadata and keeps the original prompt stamp.

Only Opus rows are taken (the Qwen rows were a different, much weaker
generator), and only rows whose seeds, README and worked example can all
be found; the rest are counted, never guessed. READMEs come from today's
pool or an older version of it in git, the newest version winning.

The extender's variations (extender/logs/extensions.jsonl) are not
converted here: their questions depend on the parent row as it was then
and on several revisions of the extender prompts.

  venv/bin/python training/rebuild_old_examples.py \\
      --out initiator/logs/distill_rebuilt_sep2026.jsonl
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "initiator"))

import distill_log                                              # noqa: E402

ATTEMPTS = HERE.parent / "initiator" / "logs" / "attempts.jsonl"
READMES = HERE.parent / "initiator" / "readmes.jsonl"
EXEMPLARS = HERE.parent / "initiator" / "exemplars.json"
SEEDS = ("construct", "style", "pattern", "scope")
NOTE = ("question rebuilt from this row's seeds with the prompt of "
        "8 Oct 2026, as Formal Disco rebuilds its examples; the "
        "original wording was not logged")


def readme_texts(versions):
    """repo -> README text over file versions given oldest first; a later
    version wins."""
    out = {}
    for text in versions:
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["repo"]] = row["readme"]
    return out


def git_versions(path=READMES):
    """Every committed version of the README pool, oldest first, then the
    file as it is now."""
    path = Path(path).resolve()
    # run git from the top of the repo: a path given from inside a
    # subfolder is read relative to that subfolder
    top = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=path.parent, text=True,
        capture_output=True, check=True).stdout.strip()).resolve()
    rel = path.relative_to(top).as_posix()
    shas = subprocess.run(["git", "log", "--format=%H", "--", rel],
                          cwd=top, text=True, capture_output=True,
                          check=True).stdout.split()
    versions = [subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=top,
                               text=True, capture_output=True,
                               check=True).stdout
                for sha in reversed(shas)]
    return versions + [path.read_text()]


def convert(rows, readmes, exemplars):
    """Returns (examples, skipped reasons with counts)."""
    from prompts import SYSTEM_PROMPT
    from run import build_user_msg
    out, skipped = [], {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for r in rows:
        if "opus" not in str(r.get("model", "")):
            skip("not an Opus row")
            continue
        if not r.get("raw_json") or not r.get("verdict"):
            skip("no design")
            continue
        if not all(r.get(k) for k in SEEDS):
            skip("seeds missing")
            continue
        readme = readmes.get(r.get("readme_id"))
        if readme is None:
            skip("README not found")
            continue
        exemplar = exemplars.get(r.get("exemplar_id"))
        if exemplar is None:
            skip("worked example not found")
            continue
        user = build_user_msg({"repo": r["readme_id"], "readme": readme},
                              r["construct"], r["style"], r["pattern"],
                              r["scope"], exemplar)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        arguments = {"repo": r["readme_id"], "readme": readme,
                     "exemplar_id": r["exemplar_id"],
                     **{k: r[k] for k in SEEDS}}
        out.append({"prompt": "initiate", "arguments": arguments,
                    "response": r["raw_json"],
                    "outcome": distill_log.outcome_of(r["verdict"]),
                    "metadata": {"verdict": r["verdict"],
                                 "messages": messages, "rebuilt": True,
                                 "note": NOTE,
                                 "source_run_id": r.get("run_id"),
                                 "source_attempt": r.get("attempt"),
                                 "model": r.get("model"),
                                 "original_prompt_version":
                                     r.get("prompt_version")}})
    return out, skipped


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--attempts", default=str(ATTEMPTS))
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; nothing was written")
    rows = [json.loads(l) for l in Path(args.attempts).read_text()
            .splitlines() if l.strip()]
    readmes = readme_texts(git_versions())
    exemplars = json.loads(EXEMPLARS.read_text())
    examples, skipped = convert(rows, readmes, exemplars)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e) + "\n" for e in examples))
    by = {}
    for e in examples:
        by[e["outcome"]] = by.get(e["outcome"], 0) + 1
    print(f"{len(rows)} rows -> {len(examples)} initiate examples {by} "
          f"-> {out}")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
