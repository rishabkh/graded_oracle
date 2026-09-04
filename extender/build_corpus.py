"""Extender step 1: flatten the NECESSARY triples out of the initiator's
run log into one generation-zero corpus file, and establish the four
baseline metrics that later generations will be compared against.

No LLM, no API. Per triple:
  state_bits           flop bits counted by yosys (proc; opt; techmap; stat)
  pdr_wall_s           PDR wall time on the without-run, from the log evidence
  clause_count         number of invariant clauses
  invariant_templates  each clause with identifiers -> id, constants -> N,
                       so family collapse is a number, not an eyeball

  venv/bin/python extender/build_corpus.py            # needs yosys (hwtools)
  venv/bin/python extender/build_corpus.py --no-yosys # skip state_bits

Writes extender/corpus.jsonl (one row per triple: id, design, property,
invariants, generation 0, parent null, metrics) and prints the four
distributions.
"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_LOG = HERE.parent / "initiator" / "logs" / "attempts.jsonl"
CORPUS = HERE / "corpus.jsonl"

# --- invariant template: mask names and constants, keep structure ---

_TOKEN = re.compile(r"""
    \d+'[bodhBODH][0-9a-fA-FxXzZ_?]+     # sized literal  4'b0001
  | \$[A-Za-z_][A-Za-z0-9_]*             # system function $onehot
  | [A-Za-z_][A-Za-z0-9_]*               # identifier
  | \d+                                  # bare number
  | <<<?|>>>?|<=|>=|==|!=|&&|\|\||\|->   # multi-char operators
  | \S                                   # any other single char
""", re.X)


# A concat whose first element is an all-zeros constant is zero-extension —
# width plumbing, not shape. {1'b0, x} collapses to x so that the width-safe
# and plain spellings of one relation count as one family. A NONZERO leading
# constant ({1'b1, x}) or a trailing constant ({cnt, 1'b1}) is a genuine
# concat and is left alone.
_ZERO_EXT = re.compile(r"\{\s*\d+'[bodhBODH][0_]+\s*,\s*([^{},]+?)\s*\}")


def _strip_zero_ext(expr):
    prev = None
    while prev != expr:
        prev, expr = expr, _ZERO_EXT.sub(r"\1", expr)
    return expr


def template(expr):
    """Normalize an invariant expression to its shape: identifiers become
    `id`, numeric constants become `N`, operators and structure remain.
    Zero-extensions are stripped first — see _strip_zero_ext."""
    out = []
    for tok in _TOKEN.findall(_strip_zero_ext(expr)):
        if re.fullmatch(r"\d+'[bodhBODH][0-9a-fA-FxXzZ_?]+|\d+", tok):
            out.append("N")
        elif tok.startswith("$"):
            out.append(tok)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            out.append("id")
        else:
            out.append(tok)
    return " ".join(out)


# --- property extraction: the assertion text inside the verilog ---

def _mask_comments(source):
    source = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), source, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), source)


def extract_asserts(source):
    """Return the expression of every immediate assertion, in order.
    Balanced-paren scan; comments masked first."""
    clean = _mask_comments(source)
    exprs = []
    for m in re.finditer(r"\bassert\b", clean):
        i = m.end()
        while i < len(clean) and clean[i].isspace():
            i += 1
        if i >= len(clean) or clean[i] != "(":
            continue
        depth, j = 0, i
        while j < len(clean):
            if clean[j] == "(":
                depth += 1
            elif clean[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        exprs.append(clean[i + 1:j].strip())
    return exprs


# --- metrics from the log evidence ---

def pdr_wall_s(record):
    """Wall time of the PDR second-opinion run on the without-invariants
    leg — already measured, sitting in the evidence."""
    runs = (record.get("result") or {}).get("without_invariants", {}).get("runs", [])
    for run in runs:
        if run.get("engine") == "abc pdr" and any(
                str(n).startswith("pdr_second_opinion") for n in run.get("notes", [])):
            return run.get("duration_s")
    return None


# stat prints count first, then cell name:  "       26   $_SDFFE_PP0P_"
_DFF_LINE = re.compile(r"^\s+(\d+)\s+(\$_[A-Z]*DFF[A-Z0-9_]*)\s*$", re.M)


def count_dff_bits(stat_output):
    """Sum the single-bit flop cells in yosys `stat` output."""
    return sum(int(n) for n, _ in _DFF_LINE.findall(stat_output)) or None


def state_bits(verilog, top_module):
    """Flop bits per yosys: proc; opt; techmap lowers registers to
    single-bit $_DFF_* cells, so the cell count IS the bit count."""
    if shutil.which("yosys") is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "design.sv"
        f.write_text(verilog)
        script = (f"read_verilog -sv {f}; hierarchy -top {top_module}; "
                  "proc; opt; techmap; opt; stat")
        try:
            out = subprocess.run(["yosys", "-p", script], capture_output=True,
                                 text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
    if out.returncode != 0:
        return None
    return count_dff_bits(out.stdout)


# --- flatten ---

def flatten_record(record, idx):
    t = json.loads(record["raw_json"])
    invariants = t.get("invariants", [])
    return {
        "id": f"g0_{idx:03d}",
        "generation": 0,
        "parent": None,
        "source_run_id": record.get("run_id"),
        "source_attempt": record.get("attempt"),
        "top_module": t["top_module"],
        "clock": t.get("clock", "clk"),
        "antecedents": t.get("antecedents", []),
        "sanity_covers": t.get("sanity_covers", []),
        "verilog": t["verilog"],
        "property": extract_asserts(t["verilog"]),
        "invariants": invariants,
        "metrics": {
            "state_bits": None,          # filled by main() when yosys runs
            "coi_ratio": None,           # filled by main() when yosys runs
            "pdr_wall_s": pdr_wall_s(record),
            "clause_count": len(invariants),
            "invariant_templates": [template(x) for x in invariants],
        },
    }


# --- distributions ---

def _spread(name, values):
    vals = [v for v in values if v is not None]
    if not vals:
        print(f"{name}: no data")
        return
    print(f"{name}: n={len(vals)} min={min(vals):.2f} "
          f"median={statistics.median(vals):.2f} max={max(vals):.2f}")


def merge_corpus(g0_rows, existing_rows):
    """A rebuild regenerates generation zero from the run log; promoted
    later generations live only in corpus.jsonl and must survive."""
    return g0_rows + [r for r in existing_rows
                      if r.get("generation", 0) > 0]


def print_distributions(rows):
    print(f"\n=== generation-zero baseline over {len(rows)} triples ===")
    _spread("state_bits", [r["metrics"]["state_bits"] for r in rows])
    _spread("coi_ratio", [r["metrics"].get("coi_ratio") for r in rows])
    _spread("pdr_wall_s", [r["metrics"]["pdr_wall_s"] for r in rows])

    clauses = Counter(r["metrics"]["clause_count"] for r in rows)
    print("clause_count:", dict(sorted(clauses.items())))

    fams = Counter(t for r in rows for t in r["metrics"]["invariant_templates"])
    total = sum(fams.values())
    print(f"invariant templates: {total} clauses, {len(fams)} distinct shapes")
    for shape, n in fams.most_common(12):
        print(f"  {n:3d}  {shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-yosys", action="store_true",
                   help="skip the state_bits metric")
    args = p.parse_args()

    records = [json.loads(line) for line in SOURCE_LOG.read_text().splitlines()]
    necessary = [r for r in records if r.get("verdict") == "NECESSARY"]
    print(f"{len(necessary)} NECESSARY triples from {SOURCE_LOG.name}")

    rows = [flatten_record(r, i) for i, r in enumerate(necessary)]

    if not args.no_yosys:
        if shutil.which("yosys") is None:
            sys.exit("yosys not on PATH - run `hwtools` first, "
                     "or pass --no-yosys")
        from coi import coi_ratio
        for row in rows:
            row["metrics"]["state_bits"] = state_bits(
                row["verilog"], row["top_module"])
            row["metrics"]["coi_ratio"] = coi_ratio(
                row["verilog"], row["top_module"])
            print(f"  {row['id']} {row['top_module']:24s} "
                  f"state_bits={row['metrics']['state_bits']} "
                  f"coi={row['metrics']['coi_ratio']}")

    existing = ([json.loads(l) for l in CORPUS.read_text().splitlines()]
                if CORPUS.exists() else [])
    rows = merge_corpus(rows, existing)
    with CORPUS.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_later = sum(1 for r in rows if r.get("generation", 0) > 0)
    print(f"\nwrote {len(rows)} rows -> {CORPUS} "
          f"({n_later} promoted row(s) preserved)")

    print_distributions(rows)


if __name__ == "__main__":
    main()
