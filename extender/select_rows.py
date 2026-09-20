"""Pick the training slice: keep the corpus rows that are least like the
rest of the corpus.

Ported from formal-disco (`distill.py::_select_top_surprisal_indices`,
`language/__init__.py::surprisal`), which runs this at `fraction=0.33`.
The rule, in three steps:

  1. describe each row by a few feature sets (invariant shapes, property
     shape, assertion form, words used in signal names);
  2. score a row per feature as the self-information of its RAREST value
     under the pooled corpus counts, -log2(count / total) in bits;
  3. rank rows within each feature, take each row's BEST rank across the
     features, and keep the top fraction by that.

Best-rank rather than an average: a row that is the most unusual thing in
the corpus on any single axis earns its place. Without this step, each
self-improvement round re-learns what the pile already has.

  venv/bin/python extender/select_rows.py --fraction 0.33
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import template                              # noqa: E402

CORPUS = HERE / "corpus.jsonl"

# Which features drive the ranking. Size is tracked below for reporting
# but not ranked: it is too coarse to separate rows.
SURPRISAL_METRICS = ("invariant_template", "property_template",
                     "assert_form", "signal_word")

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_KEYWORDS = {"module", "endmodule", "input", "output", "wire", "reg",
             "always", "assert", "begin", "end", "posedge", "negedge",
             "initial", "if", "else", "case", "endcase", "default",
             "localparam", "parameter", "property", "clk", "rst"}


def _words(source):
    out = []
    for ident in _IDENT.findall(source):
        if ident in _KEYWORDS:
            continue
        for part in _CAMEL.findall(ident.replace("_", " ")):
            part = part.lower()
            if len(part) > 2 and not part.isdigit():
                out.append(part)
    return out


def _assert_forms(verilog):
    forms = []
    if re.search(r"always\s*@\s*\(\s*\*\s*\)[^;]*assert", verilog):
        forms.append("plain")
    if re.search(r"always\s*@\s*\(\s*posedge[^)]*\)[^;]*assert\s*\(", verilog):
        forms.append("clocked")
    if "assert property" in verilog:
        forms.append("property")
    for fn in ("past", "rose", "fell", "stable"):
        if f"${fn}" in verilog:
            forms.append(fn)
    if re.search(r"if\s*\([^)]*\)\s*assert", verilog):
        forms.append("guarded")
    if len(re.findall(r"\bassert\b", verilog)) > 1:
        forms.append("multi")
    return forms or ["none"]


def feature_sets(row):
    """One Counter per feature. Everything here is derived from the row
    itself, never from the seed fields, so extended rows and rows made by
    a served model are described the same way."""
    invariants = row.get("invariants") or []
    props = row.get("property") or []
    bits = (row.get("metrics") or {}).get("state_bits")
    return {
        "invariant_template": Counter(template(x) for x in invariants),
        "property_template": Counter(template(p) for p in props),
        "assert_form": Counter(_assert_forms(row.get("verilog", ""))),
        "signal_word": Counter(_words(row.get("verilog", ""))),
        "size_bucket": Counter(
            [str(2 ** int(math.log2(bits))) if bits else "unknown"]),
    }


def pooled_stats(rows, metrics=SURPRISAL_METRICS):
    pooled = defaultdict(Counter)
    for row in rows:
        for metric, counter in feature_sets(row).items():
            if metric in metrics:
                pooled[metric].update(counter)
    return pooled


def surprisals(row, pooled, metrics=SURPRISAL_METRICS):
    """Bits of self-information of this row's rarest value, per feature.
    A value the pool has never seen scores infinity."""
    out = {}
    for metric, counter in feature_sets(row).items():
        if metric not in metrics or not counter:
            continue
        total = sum(pooled[metric].values())
        if not total:
            continue
        worst = 0.0
        for value in counter:
            seen = pooled[metric][value]
            if seen == 0:
                worst = math.inf
                break
            worst = max(worst, math.log2(total / seen))
        out[metric] = worst
    return out


def select_top(rows, fraction=0.33, metrics=SURPRISAL_METRICS):
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not rows:
        return []
    pooled = pooled_stats(rows, metrics)
    scored = [surprisals(row, pooled, metrics) for row in rows]

    best_rank = {i: math.inf for i in range(len(rows))}
    for metric in metrics:
        have = [i for i, s in enumerate(scored) if metric in s]
        have.sort(key=lambda i: (-scored[i][metric], i))
        for rank, i in enumerate(have, start=1):
            best_rank[i] = min(best_rank[i], rank)

    keep_n = max(1, round(len(rows) * fraction))
    order = sorted(range(len(rows)), key=lambda i: (best_rank[i], i))
    return [rows[i] for i in sorted(order[:keep_n])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--fraction", type=float, default=0.33)
    p.add_argument("--out", default=None,
                   help="write the kept rows here; default prints only")
    args = p.parse_args()

    rows = [json.loads(l) for l in
            Path(args.corpus).read_text().splitlines() if l.strip()]
    kept = select_top(rows, args.fraction)
    print(f"{len(rows)} rows -> {len(kept)} kept at fraction {args.fraction}")
    pooled = pooled_stats(rows)
    for metric in SURPRISAL_METRICS:
        print(f"  {metric}: {len(pooled[metric])} distinct values, "
              f"{sum(pooled[metric].values())} total")
    if args.out:
        Path(args.out).write_text(
            "".join(json.dumps(r) + "\n" for r in kept))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
