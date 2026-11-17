"""Prove the design checker on another machine judges like the laptop's.

Making designs moves to the cluster (16 Nov 2026, Rishab: "alright let's
do it"): 150 growth steps took 4.5 hours of the laptop on 15 Nov, so the
5,000 of the big run would take about six days. The cluster runs another
build of the same tools (yosys, sby, yices, iverilog) on another kind of
processor. Its verdicts count only after designs the laptop judged,
replayed there, all come back with the same verdict, as
ebmc_calibrate.py did for EBMC. Only designs judged well under the time
limit are replayed, so a slower or faster machine cannot flip a verdict
by timing alone: passing designs spread from quick to slow, designs the
freeze check turned away, and designs whose facts were false.

  # laptop, once: choose the designs from a run's log, judge them again
  # with today's code and save what it got to results/
  venv/bin/python initiator/checker_calibrate.py --make
  # any machine: replay them; exits 1 on any difference
  python initiator/checker_calibrate.py --slots 8
  # the same, and one tiny paid question (under a cent) to show that the
  # machine reaches the model
  python initiator/checker_calibrate.py --slots 8 --ask-model
"""
import argparse
import json
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import freeze_check                                          # noqa: E402

CALIBRATION = (HERE.parent / "results"
               / "checker_calibration_pilot_designs_2026-11-16.jsonl")
# the 300 starting designs GPT-6.1 Sol wrote on 15 Nov 2026 (the pilot)
SOURCE_LOG = (HERE / "logs"
              / "initiator_pilot_300_before_big_run_2026-11-14.jsonl")
MAX_TIME_S = 60        # of the checker's 300: far from the limit


def _is_design(text):
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False


def candidates(rows, max_time=MAX_TIME_S):
    """Every judged design of a run's log that can be replayed: a final
    design that passed ("passing"), and from the earlier rounds a design
    the freeze check turned away ("frozen") or one whose proof failed for
    another reason than the time limit ("failing"). Nothing that took more
    than `max_time` seconds, and nothing that ran out of time."""
    out = []
    for r in sorted(rows, key=lambda r: r.get("attempt", 0)):
        if r.get("verdict") == "NECESSARY" and _is_design(r.get("raw_json")) \
                and (r.get("grade_wall_s") or 0) <= max_time:
            out.append({"key": f"attempt_{r['attempt']}_final",
                        "kind": "passing", "raw_json": r["raw_json"],
                        "scope": r["scope"], "log_verdict": "NECESSARY",
                        "log_seconds": r["grade_wall_s"]})
        for n, h in enumerate(r.get("history") or []):
            if not _is_design(h.get("design")):
                continue
            if h.get("verdict") == "FREEZES":
                kind = "frozen"
            elif h.get("verdict") == "NOT_PROVEN" \
                    and "TIMEOUT" not in (h.get("reason") or "") \
                    and (h.get("grade_wall_s") or 0) <= max_time:
                kind = "failing"
            else:
                continue
            out.append({"key": f"attempt_{r['attempt']}_round_{n}",
                        "kind": kind, "raw_json": h["design"],
                        "scope": r["scope"], "log_verdict": h["verdict"],
                        "log_seconds": h.get("grade_wall_s")})
    return out


def _in_order(key):
    """attempt_2 before attempt_10, a final design before its rounds."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]


def select(cands, passing=30, frozen=6, failing=4):
    """The designs to replay, the same whatever order the log is in:
    `passing` passing designs spread evenly from the quickest to the
    slowest, then the first `frozen` and `failing` ones."""
    def first(kind, n):
        return sorted((c for c in cands if c["kind"] == kind),
                      key=lambda c: _in_order(c["key"]))[:n]
    quick_first = sorted((c for c in cands if c["kind"] == "passing"),
                         key=lambda c: (c["log_seconds"], _in_order(c["key"])))
    n, k = len(quick_first), passing
    if n > k > 1:
        quick_first = [quick_first[(i * (n - 1)) // (k - 1)]
                       for i in range(k)]
    chosen = quick_first[:k] + first("frozen", frozen) + first("failing",
                                                               failing)
    return sorted(chosen, key=lambda c: _in_order(c["key"]))


def replay(items, slots=4, judge=None, work=None):
    """Judge every item on this machine as the pilot did, with the freeze
    check on: [{"key", "verdict", "seconds"}] in the items' order. `judge`
    is the initiator's own (run.judge) unless a test hands in another."""
    if judge is None:
        import run
        judge = run.judge

    def one(item, root):
        t0 = time.monotonic()
        j = judge(item["raw_json"], "ok", item["scope"], grade=True,
                  label=item["key"], proof_root=Path(root) / item["key"],
                  spin=False)
        return {"key": item["key"], "verdict": j["verdict"],
                "seconds": round(time.monotonic() - t0, 1)}
    before, freeze_check.ON = freeze_check.ON, True
    try:
        with tempfile.TemporaryDirectory() as tmp, \
                ThreadPoolExecutor(max(1, slots)) as pool:
            root = work or tmp
            return list(pool.map(lambda item: one(item, root), items))
    finally:
        freeze_check.ON = before


def make(source, out, judge=None, slots=4):
    """Choose the designs from the log at `source`, judge them again here
    with today's code, and save each with the verdict it got to `out`."""
    source, out = Path(source), Path(out)
    if out.exists():
        sys.exit(f"{out} is already there; nothing was written (give "
                 "another --file to make a new one)")
    if not source.exists():
        sys.exit(f"no log at {source}; nothing was written")
    rows = [json.loads(l) for l in source.read_text().splitlines()
            if l.strip()]
    chosen = select(candidates(rows))
    got = replay(chosen, slots=slots, judge=judge)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item, g in zip(chosen, got):
            f.write(json.dumps(dict(item, verdict=g["verdict"],
                                    seconds=g["seconds"])) + "\n")
    changed = [(c["key"], c["log_verdict"], g["verdict"])
               for c, g in zip(chosen, got) if c["log_verdict"] != g["verdict"]]
    kinds = {k: sum(1 for c in chosen if c["kind"] == k)
             for k in ("passing", "frozen", "failing")}
    print(f"{len(chosen)} designs saved to {out}: {kinds['passing']} that "
          f"passed, {kinds['frozen']} the freeze check turned away, "
          f"{kinds['failing']} with a false fact or a broken design")
    for key, then, now in changed:
        print(f"  {key}: the log said {then}, today's code says {now}")
    return out


def ask_model():
    """One tiny question to the run's model; its one-word answer. Costs
    under a cent. Raises when the model cannot be reached."""
    import llm_client
    import run
    text, _, stop = llm_client.call_claude(
        model=run.MODEL, max_tokens=2000, system="You answer in one word.",
        user='Reply with the word "ready".', effort="low",
        schema={"type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"], "additionalProperties": False})
    if text is None:
        raise RuntimeError(f"no answer ({stop})")
    return json.loads(text)["word"]


def main(argv=None, judge=None, ask=None):
    p = argparse.ArgumentParser(
        description="Replay judged designs through this machine's checker.")
    p.add_argument("--make", action="store_true",
                   help="laptop: choose the designs, judge them again and "
                        "save them to --file")
    p.add_argument("--source", default=str(SOURCE_LOG),
                   help="with --make: the run's log to choose from")
    p.add_argument("--file", default=str(CALIBRATION),
                   help="the saved designs and their verdicts")
    p.add_argument("--slots", type=int, default=4,
                   help="designs judged at the same time")
    p.add_argument("--ask-model", action="store_true",
                   help="also ask the model one tiny question (paid, under "
                        "a cent) to show this machine reaches it")
    args = p.parse_args(argv)
    if args.make:
        make(args.source, args.file, judge=judge, slots=args.slots)
        return 0
    path = Path(args.file)
    if not path.exists():
        sys.exit(f"no calibration file at {path}; make one on the laptop "
                 "with --make")
    saved = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    got = replay(saved, slots=args.slots, judge=judge)
    wrong = [(s["key"], s["verdict"], g["verdict"])
             for s, g in zip(saved, got) if s["verdict"] != g["verdict"]]
    print(f"{len(saved) - len(wrong)} of {len(saved)} designs judged the "
          "same")
    for key, there, here in wrong:
        print(f"  {key}: {there} on the laptop, {here} here")
    print(f"checking them took {sum(g['seconds'] for g in got):.0f} s here "
          f"and {sum(s.get('seconds') or 0 for s in saved):.0f} s on the "
          "laptop (each design's own time, added up)")
    code = 1 if wrong else 0
    if args.ask_model:
        try:
            print(f"the model answered: {(ask or ask_model)()!r}")
        except Exception as exc:
            print(f"the model was not reached: {type(exc).__name__}: {exc}")
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
