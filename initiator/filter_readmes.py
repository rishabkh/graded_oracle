"""Rebuild the theme README pool so it is strictly digital hardware.

Why (8 Oct 2026): the pool's 1,000 new READMEs came from GitHub hardware
topics, but a topic is a tag the owner picks, not a description. Two
independent judges read every README against the written rubric below,
and a third settled each disagreement (35 of 1,420). 44% of the new
READMEs were not digital hardware: JSON parsers, speech software, Linux
distributions, operating systems for microcontrollers, emulators. Of the
first 133, 23 were placeholders and 48 more were not digital hardware.

The pool is rebuilt from those verdicts by a fixed rule, never by hand:
keep the original READMEs judged digital hardware, as they were; then add
new READMEs judged digital hardware, the higher star tier first (more than
50 stars, then 11 to 50), topics taken in turn in harvest_readmes.TOPICS
order with "risc-v" and "riscv" counted as one topic (one subject spelled
two ways, which had given it double turns), most starred first within a
topic, until --target new ones are in. Placeholders ("example/...") are
never kept. A candidate with no verdict stops the run.

The record file keeps the rubric, the rule and every verdict with its
reasons, so the pool can be audited and rebuilt.

  venv/bin/python initiator/filter_readmes.py --candidates <jsonl ...> \\
      --verdicts <verdicts.json> --target 1000 \\
      --record initiator/catalog/readme_record.json [--dry]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harvest_readmes import OUT as POOL, TOPICS                 # noqa: E402

KEEP = "digital-hardware"
SAME_TOPIC = {"riscv": "risc-v"}
TIER_STARS = 50

RUBRIC = """\
- "digital-hardware": the project designs, implements, verifies, or builds tools for DIGITAL LOGIC hardware. Includes: RTL/HDL code (Verilog, SystemVerilog, VHDL, Chisel, SpinalHDL, Amaranth, Bluespec, MyHDL, Clash, etc.); CPU, SoC, GPU, accelerator, peripheral or IP-core designs; FPGA projects and gateware (including FPGA board support whose substance is the FPGA logic); ASIC/chip design flows and tapeouts; HDL simulators, synthesis, place-and-route, timing and other EDA tools for chips; hardware verification frameworks (UVM, cocotb, formal tools for RTL); HDL language tooling (parsers, linters, language servers, formatters); hardware generators and high-level-synthesis tools; learning material whose substance is digital logic design.
- "other-hardware": about hardware or electronics but NOT digital logic design: PCB/schematic projects, analog or RF circuits, embedded firmware or RTOS for microcontrollers, robotics, drones, 3D printers, IoT devices, operating systems or Linux distributions for boards, emulators of whole consoles/computers written in software, hardware reverse-engineering notes.
- "not-hardware": general software, machine learning, data analysis ("EDA" meaning exploratory data analysis), web, databases, operating systems not about hardware, tutorials unrelated to hardware, anything else.
Judge by what the project IS, not by words it mentions."""

RULE = ("keep the original READMEs judged digital-hardware as they were; add "
        "new READMEs judged digital-hardware, star tier first (more than 50, "
        "then 11 to 50), topics in turn in harvest_readmes.TOPICS order with "
        "risc-v and riscv as one topic, most starred first within a topic, "
        "until the target number of new ones; never a placeholder "
        "(example/...)")


def _placeholder(repo):
    return repo.startswith("example/")


def _order(candidates):
    """Higher star tier first; within a tier, topics in turn, most starred
    first within a topic."""
    topics = [t for t in TOPICS if t not in SAME_TOPIC]
    out = []
    for tier in (True, False):
        by_topic = {}
        for c in candidates:
            if ((c.get("stars") or 0) > TIER_STARS) is not tier:
                continue
            t = SAME_TOPIC.get(c["topic"], c["topic"])
            by_topic.setdefault(t, []).append(c)
        order = topics + [t for t in by_topic if t not in topics]
        queues = [sorted(by_topic.get(t, []), key=lambda c: -(c.get("stars") or 0))
                  for t in order]
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.pop(0))
    return out


def rebuild(old, candidates, verdicts, target):
    """`old`: the original pool rows (no topic); `candidates`: new rows with
    topic and stars. Returns the new pool."""
    missing = [r["repo"] for r in old + candidates
               if not _placeholder(r["repo"]) and r["repo"] not in verdicts]
    if missing:
        raise ValueError(f"{len(missing)} READMEs have no verdict, e.g. "
                         f"{missing[:3]}; nothing was rebuilt")
    kept = [r for r in old if not _placeholder(r["repo"])
            and verdicts[r["repo"]]["category"] == KEEP]
    seen = {r["repo"] for r in kept}
    added = 0
    for c in _order(candidates):
        if added >= target:
            break
        if c["repo"] in seen or _placeholder(c["repo"]) or \
                verdicts[c["repo"]]["category"] != KEEP:
            continue
        seen.add(c["repo"])
        kept.append(c)
        added += 1
    return kept


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool", default=str(POOL))
    p.add_argument("--candidates", nargs="*", default=[],
                   help="extra candidate READMEs (jsonl) besides the pool's")
    p.add_argument("--verdicts", required=True)
    p.add_argument("--target", type=int, default=1000)
    p.add_argument("--record", required=True)
    p.add_argument("--dry", action="store_true",
                   help="print the counts; write nothing")
    args = p.parse_args(argv)

    pool = [json.loads(l) for l in Path(args.pool).read_text().splitlines()
            if l.strip()]
    old = [r for r in pool if "topic" not in r]
    candidates = [r for r in pool if "topic" in r]
    for f in args.candidates:
        candidates += [json.loads(l) for l in Path(f).read_text().splitlines()
                       if l.strip()]
    verdicts = json.loads(Path(args.verdicts).read_text())
    new_pool = rebuild(old, candidates, verdicts, args.target)
    n_old = sum(1 for r in new_pool if "topic" not in r)
    print(f"{len(pool)} READMEs in the pool, {len(candidates) - (len(pool) - len(old))}"
          f" extra candidates; kept {n_old} original + {len(new_pool) - n_old} "
          f"new = {len(new_pool)}")
    if len(new_pool) - n_old < args.target:
        print(f"  only {len(new_pool) - n_old} new READMEs pass, short of "
              f"{args.target}")
    if args.dry:
        return
    Path(args.pool).write_text("".join(json.dumps(r) + "\n" for r in new_pool))
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rubric": RUBRIC,
        "judges": ("two independent judges per README, neither seeing the "
                   "other; a third settles every disagreement"),
        "rule": RULE, "target": args.target, "kept": len(new_pool),
        "kept_original": n_old, "kept_new": len(new_pool) - n_old,
        "verdicts": verdicts}, indent=1))
    print(f"  wrote {args.pool} and {args.record}")


if __name__ == "__main__":
    main()
