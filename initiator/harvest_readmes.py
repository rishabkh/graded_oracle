"""Harvest hardware-project READMEs from the GitHub API into
initiator/readmes.jsonl (same {"repo", "readme"} format the sampler
reads). Formal Disco sampled 100k generic READMEs from BigQuery; we
need ~100 hardware-flavoured ones, so the search API via an
authenticated `gh` is the right size of tool.

  venv/bin/python initiator/harvest_readmes.py --per-topic 14 --dry
  venv/bin/python initiator/harvest_readmes.py --per-topic 14
"""
import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "readmes.jsonl"

# no "rtl": that topic also means right-to-left text and pulls in UI libs
TOPICS = ["verilog", "fpga", "systemverilog", "risc-v", "riscv",
          "hdl", "asic", "soc", "hardware-design", "eda"]
MAX_CHARS = 1600
MIN_CHARS = 200


def clean(text):
    """Strip badges/images/html noise, collapse blank runs, truncate at
    a paragraph boundary near MAX_CHARS."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)     # images
    text = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)                  # html tags
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= MAX_CHARS:
        return text
    cut = text.rfind("\n\n", 0, MAX_CHARS)
    return text[:cut if cut > MIN_CHARS else MAX_CHARS].strip()


def gh_json(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    return json.loads(r.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-topic", type=int, default=14)
    p.add_argument("--dry", action="store_true",
                   help="list repos, fetch nothing else, write nothing")
    args = p.parse_args()

    existing = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            row = json.loads(line)
            existing[row["repo"]] = row
    print(f"{len(existing)} existing readmes")

    seen = set(existing)
    added = 0
    for topic in TOPICS:
        res = gh_json(f"search/repositories?q=topic:{topic}+stars:%3E50"
                      f"&sort=stars&per_page={args.per_topic}")
        for item in res.get("items", []):
            full = item["full_name"]
            if full in seen:
                continue
            seen.add(full)
            if args.dry:
                print(f"  would fetch {full} ({topic})")
                continue
            try:
                blob = gh_json(f"repos/{full}/readme")
                text = base64.b64decode(blob["content"]).decode(
                    "utf-8", errors="replace")
            except Exception as exc:
                print(f"  skip {full}: {exc}")
                continue
            body = clean(text)
            if len(body) < MIN_CHARS:
                print(f"  skip {full}: readme too thin")
                continue
            existing[full] = {"repo": full, "readme": body}
            added += 1
            print(f"  + {full} ({topic}, {len(body)} chars)")
    if args.dry:
        return
    with OUT.open("w") as f:
        for row in existing.values():
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(existing)} readmes (+{added}) -> {OUT.name}")


if __name__ == "__main__":
    main()
