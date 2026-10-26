"""How real the designs of one run look (21 Oct 2026, Rishab approved: "make
all 1-8 changes", item 8d).

batch.py writes this report at the end of every run with --log-dir, as
<log-dir>/realism_report.txt, and prints it. It reads the run's logs and
the corpus the run grew, and sets the designs the run kept beside real
open-source hardware:

  steps    tried and kept, by verdict and by kind of step
  growth   per lineage, the design the run started from against its
           deepest kept descendant, in honest lines (build_corpus.honest_lines:
           copies counted once), and the factor
  size     lines without comments or blank lines, the way the real modules
           were counted, per design and per module
  writing  the share with more than one module, a valid/ready handshake,
           parameters, generate loops, enum, an asynchronous reset, arrays
  facts    per design, expanded (what the checker proves) and as written
  engine   could abc pdr prove the property with no facts
           (metrics.engine_alone)
  cost     the run's model calls (calls.jsonl) at llm_client.PRICES, half
           price for batch calls

The real numbers (REAL_*) were measured on 15 Oct 2026 over 560 modules of
PULP common_cells, lowRISC Ibex, OpenTitan prims and peripherals,
alexforencich verilog-axis and verilog-axi, and ZipCPU wb2axip. The rules
below give our 252 starting designs of 9 Oct the numbers measured for them
that day (lines median 67 against "about 66", handshake 6%, parameters 38%
against "about 40%", generate 5%, asynchronous reset and enum 0%, arrays
32%), so ours are counted the same way as theirs were.

  venv/bin/python extender/realism_report.py \\
      --log-dir extender/logs/real_systems_trial_2026-10-21 \\
      --corpus extender/corpus_real_systems_trial_2026-10-21.jsonl
  ... --out report.txt     # write it here instead of in the log folder
"""
import argparse
import json
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_corpus import (_mask_comments, engine_alone,          # noqa: E402
                          engine_alone_share, honest_lines)
import extend                                                    # noqa: E402
import llm_client                                                # noqa: E402
import sense_check                                               # noqa: E402

REPORT_NAME = "realism_report.txt"

# --- real hardware, measured 15 Oct 2026 over 560 modules ------------------
REAL_LINES = (76, 223, 504)      # lines per module: median, 75%, 90%
REAL_SHARES = {"more than one module": 0.66,
               "valid/ready handshake": 0.49,
               "parameters": 0.87,
               "generate loops": 0.31,
               "asynchronous reset": 0.21,
               "enum": 0.05,
               "arrays": 0.12}
REAL_FACTS = 18                  # median facts per module, ZipCPU's formal
                                 # sections (59 modules of wb2axip)


# --- how a design is written ------------------------------------------------

_MODULE = re.compile(r"\bmodule\b.*?\bendmodule\b", re.S)
_NAME = re.compile(r"[A-Za-z_]\w*")
_VALID = re.compile(r"valid|(?:^|_)vld(?:_|\d|$)", re.I)
_READY = re.compile(r"ready|(?:^|_)rdy(?:_|\d|$)", re.I)
# two edges in one event list: always @(posedge clk or negedge rst_n),
# always_ff @(posedge clk, posedge reset)
_ASYNC_RESET = re.compile(r"@\s*\(\s*(?:pos|neg)edge\s+[\w.]+\s*(?:\bor\b|,)"
                          r"\s*(?:pos|neg)edge\b")
_PARAMETER = re.compile(r"\b(?:parameter|localparam)\b")
_GENERATE = re.compile(r"\b(?:generate|genvar)\b")
_ENUM = re.compile(r"\benum\b")


def module_texts(verilog):
    """Each module of a design, `module` to `endmodule`, comments masked."""
    return _MODULE.findall(_mask_comments(verilog))


def code_lines(text):
    """Lines that are not blank once comments are taken out: how the real
    modules were counted on 15 Oct 2026."""
    return sum(1 for line in _mask_comments(text).splitlines()
               if line.strip())


def has_handshake(verilog):
    """Does one module have both a valid signal (valid, s_tvalid, in_vld)
    and a ready signal (ready, m_tready, out_rdy)? Names with "invalid" or
    "already" in them are neither."""
    for text in module_texts(verilog):
        names = set(_NAME.findall(text))
        valid = any(_VALID.search(n) and "invalid" not in n.lower()
                    for n in names)
        ready = any(_READY.search(n) and "already" not in n.lower()
                    for n in names)
        if valid and ready:
            return True
    return False


def traits(verilog):
    """{trait: True or False} for one design, the traits of REAL_SHARES.
    Parameters count `parameter` and `localparam`, generate counts
    `generate` and `genvar`, arrays are storage arrays (extend.declared_arrays:
    a constant table is not one)."""
    text = _mask_comments(verilog)
    return {"more than one module": len(module_texts(verilog)) > 1,
            "valid/ready handshake": has_handshake(verilog),
            "parameters": bool(_PARAMETER.search(text)),
            "generate loops": bool(_GENERATE.search(text)),
            "asynchronous reset": bool(_ASYNC_RESET.search(text)),
            "enum": bool(_ENUM.search(text)),
            "arrays": bool(extend.declared_arrays(verilog))}


def percentile(values, p):
    """The p-th percentile of `values`, read on a straight line between
    neighbouring values; None when there are none."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# --- what the run did -------------------------------------------------------

def read_jsonl(path):
    """The rows of a JSONL file, none when it does not exist; a line that
    cannot be read (a run stopped while writing it) is left out."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def step_kind(record):
    """The move of a structural step, the type of any other."""
    return record.get("move") or record.get("ext_type") or "unknown"


def steps(records, kept_ids):
    """(verdict counts, {kind: [tried, kept]}) over a run's log records;
    every answer checked is one step, retries included."""
    verdicts = Counter(r.get("verdict") or "none" for r in records)
    kinds = defaultdict(lambda: [0, 0])
    for r in records:
        kinds[step_kind(r)][0] += 1
        kinds[step_kind(r)][1] += r.get("extension_id") in kept_ids
    return verdicts, dict(kinds)


def _honest(row):
    return honest_lines((row or {}).get("verilog") or "")


def start_of(row, by_id, made):
    """The id of the design a run grew `row`'s lineage from: the nearest
    ancestor through `parent` (the design that was extended; a
    combination's partner is not followed) that the run did not make."""
    seen = set()
    while row["id"] in made and row["id"] not in seen:
        seen.add(row["id"])
        parent = by_id.get(row.get("parent"))
        if parent is None:
            return row.get("parent")
        row = parent
    return row["id"]


def lineage_growth(kept, by_id):
    """One entry per lineage, by start id: the start, its honest lines, the
    deepest kept row (the most lines among the deepest), its honest lines,
    the factor between them, and how many rows the run kept in it. Lines or
    factor are None when the start is not in the corpus."""
    made = {r["id"] for r in kept}
    groups = defaultdict(list)
    for r in kept:
        groups[start_of(r, by_id, made)].append(r)
    out = []
    for start_id in sorted(groups, key=str):
        rows = groups[start_id]
        deepest = max(rows, key=lambda r: (r.get("generation", 0),
                                           _honest(r)))
        start = by_id.get(start_id)
        before = _honest(start) if start else None
        after = _honest(deepest)
        out.append({"start": start_id, "start_lines": before,
                    "deepest": deepest["id"],
                    "generation": deepest.get("generation", 0),
                    "lines": after, "kept": len(rows),
                    "factor": after / before if before else None})
    return out


def facts(row):
    """(facts the checker proves, facts as written): one compact fact for
    every entry of an array counts once as written (21 Oct 2026)."""
    expanded = row.get("invariants") or []
    return len(expanded), len(row.get("invariants_compact") or expanded)


def engine_word(row, record=None):
    """What abc pdr found with no facts: the row's metrics.engine_alone, or
    for a row made before 21 Oct 2026 what its step's own check found
    (build_corpus.engine_alone of the log record), else "unknown"."""
    word = (row.get("metrics") or {}).get("engine_alone")
    if word:
        return word
    if record and record.get("result"):
        return engine_alone(record)
    return "unknown"


def engine_share(words):
    """build_corpus.engine_alone_share over a list of engine words."""
    return engine_alone_share([{"metrics": {"engine_alone": w}}
                               for w in words])


# --- what the model calls cost ----------------------------------------------

def first_line(text):
    text = (text or "").strip()
    return text.splitlines()[0] if text else ""


CALL_KINDS = ("extension questions and partner choices", "reviews",
              "distractor questions", "other")


def call_kind(entry):
    """Who asked, read from the system text: the extension questions and
    partner choices share extend.SYSTEM_PROMPT, the reviewer has
    sense_check.SYSTEM, and the distractor sends none. An OpenAI request
    (26 Oct 2026) carries the system text as its first message."""
    request = entry.get("request") or {}
    system = request.get("system")
    if system is None:
        system = next((m.get("content") for m in request.get("messages") or []
                       if isinstance(m, dict) and m.get("role") == "system"),
                      None)
    system = first_line(system)
    if not system:
        return "distractor questions"
    return {first_line(extend.SYSTEM_PROMPT): CALL_KINDS[0],
            first_line(sense_check.SYSTEM): "reviews"}.get(system, "other")


def call_dollars(entry):
    """What one logged call cost (llm_client.dollars, halved for a batch
    call), or None when its model has no list price."""
    usage = entry.get("usage") or {}
    try:
        return llm_client.dollars(entry.get("model"),
                                  usage.get("input") or 0,
                                  usage.get("output") or 0,
                                  batch=bool(entry.get("batch")))
    except ValueError:
        return None


def call_costs(calls):
    """Calls, batch calls, tokens in and out, and dollars, for all calls
    and by kind (call_kind); calls of a model with no list price are named
    and not counted in the dollars."""
    total = {"calls": 0, "batch": 0, "in": 0, "out": 0, "dollars": 0.0,
             "unpriced": Counter()}
    kinds = {k: {"calls": 0, "dollars": 0.0} for k in CALL_KINDS}
    for entry in calls:
        usage = entry.get("usage") or {}
        cost = call_dollars(entry)
        total["calls"] += 1
        total["batch"] += bool(entry.get("batch"))
        total["in"] += usage.get("input") or 0
        total["out"] += usage.get("output") or 0
        kind = kinds[call_kind(entry)]
        kind["calls"] += 1
        if cost is None:
            total["unpriced"][str(entry.get("model"))] += 1
        else:
            total["dollars"] += cost
            kind["dollars"] += cost
    return total, kinds


# --- measuring and writing ---------------------------------------------------

def measure(log_dir, corpus):
    """Everything the report says about the run in `log_dir` (its
    extensions.jsonl and calls.jsonl) and the corpus it grew. The designs
    the run kept are the corpus rows promoted from its log records
    (source_extension_id)."""
    log_dir = Path(log_dir)
    records = read_jsonl(log_dir / "extensions.jsonl")
    rows = [r for r in read_jsonl(corpus) if "id" in r]
    by_id = {r["id"]: r for r in rows}
    by_extension = {r["extension_id"]: r for r in records
                    if r.get("extension_id")}
    kept = [r for r in rows if r.get("source_extension_id") in by_extension]
    kept_ids = {r["source_extension_id"] for r in kept}
    verdicts, kinds = steps(records, kept_ids)
    lineages = lineage_growth(kept, by_id)
    verilog = [r.get("verilog") or "" for r in kept]
    shown = [traits(v) for v in verilog]
    return {
        "log_dir": str(log_dir), "corpus": str(corpus),
        "tried": len(records), "kept": len(kept),
        "verdicts": verdicts, "kinds": kinds, "lineages": lineages,
        "lines": [code_lines(v) for v in verilog],
        "module_lines": [code_lines(t) for v in verilog
                         for t in module_texts(v)],
        "honest": [honest_lines(v) for v in verilog],
        "traits": {t: sum(s[t] for s in shown) for t in REAL_SHARES},
        "facts": [facts(r) for r in kept],
        "engine": [engine_word(r, by_extension.get(r["source_extension_id"]))
                   for r in kept],
        "engine_starts": [engine_word(by_id[x["start"]]) for x in lineages
                          if x["start"] in by_id],
        "calls": call_costs(read_jsonl(log_dir / "calls.jsonl")),
    }


def _pct(n, total):
    return f"{100 * n / total:.0f}%" if total else "-"


def _num(x):
    return "-" if x is None else f"{x:.0f}"


def _wrap(text, indent="  "):
    return textwrap.wrap(text, width=76, initial_indent=indent,
                         subsequent_indent=indent + "  ")


def _wrap_items(label, items, width=76):
    """`label` and then the items, comma-separated, never breaking a line
    inside one item."""
    lines, cur = [], "  " + label
    for i, item in enumerate(items):
        piece = item + ("," if i < len(items) - 1 else "")
        if len(cur) + 1 + len(piece) > width and cur.strip() != label:
            lines.append(cur)
            cur = "    " + piece
        else:
            cur += " " + piece
    return lines + [cur]


def _steps_text(m):
    out = ["STEPS",
           f"  tried {m['tried']} (every answer checked, retries included), "
           f"kept {m['kept']} ({_pct(m['kept'], m['tried'])})"]
    if m["verdicts"]:
        out += _wrap_items("verdicts:", [f"{v} {n}" for v, n in
                                         m["verdicts"].most_common()])
        out.append(f"  {'kind':24s} {'tried':>6s} {'kept':>6s}")
        for kind, (tried, kept) in sorted(m["kinds"].items(),
                                          key=lambda kv: -kv[1][0]):
            out.append(f"  {kind:24s} {tried:6d} {kept:6d}")
    return out


def _growth_text(m):
    out = ["GROWTH PER LINEAGE, in honest lines: copies counted once, and "
           "comments,",
           "blank lines and lines of only begin or end left out"]
    if not m["lineages"]:
        return out + ["  no designs kept"]
    out.append(f"  {'started from':14s} {'lines':>6s}   "
               f"{'deepest kept':14s} {'gen':>3s} {'lines':>6s} "
               f"{'growth':>7s} {'kept':>5s}")
    for x in m["lineages"]:
        factor = "-" if x["factor"] is None else f"x{x['factor']:.1f}"
        out.append(f"  {str(x['start']):14s} {_num(x['start_lines']):>6s}   "
                   f"{x['deepest']:14s} {x['generation']:3d} "
                   f"{x['lines']:6d} {factor:>7s} {x['kept']:5d}")
    factors = [x["factor"] for x in m["lineages"] if x["factor"]]
    if factors:
        out.append(f"  median growth x{percentile(factors, 50):.1f} over "
                   f"{len(factors)} lineage(s); most x{max(factors):.1f}")
    return out


def _size_text(m):
    def line(name, cells):
        return f"  {name:42s} " + " ".join(f"{c:>6s}" for c in cells)

    def spread(values):
        return [_num(percentile(values, p)) for p in (50, 75, 90)]

    out = ["SIZE (lines without comments or blank lines)",
           line("", ["median", "75%", "90%"])]
    if m["lines"]:
        out += [line("kept designs, whole design", spread(m["lines"])),
                line("kept designs, each module", spread(m["module_lines"]))]
    else:
        out.append("  no designs kept")
    out.append(line("real modules", [str(n) for n in REAL_LINES]))
    if m["honest"]:
        out.append(line("kept designs, in honest lines", spread(m["honest"])))
    out.append("  (75%: three in four are this size or smaller; 90%: nine "
               "in ten)")
    return out


def _traits_text(m):
    n = m["kept"]
    out = [f"HOW THEY ARE WRITTEN {'':20s}{'kept designs':>14s} "
           f"{'real modules':>13s}"]
    for trait, real in REAL_SHARES.items():
        mine = f"{m['traits'][trait]} ({_pct(m['traits'][trait], n)})" \
            if n else "-"
        out.append(f"  {trait:38s} {mine:>14s} {100 * real:12.0f}%")
    return out


def _facts_text(m):
    out = [f"FACTS PER KEPT DESIGN {'':30s}{'median':>6s} {'most':>6s}"]
    if m["facts"]:
        for i, name in enumerate(("as the checker proves them (expanded)",
                                  "as written (one compact fact per array)")):
            values = [f[i] for f in m["facts"]]
            out.append(f"  {name:49s} {_num(percentile(values, 50)):>6s} "
                       f"{max(values):6d}")
    else:
        out.append("  no designs kept")
    out.append(f"  real: ZipCPU's formal sections hold a median of "
               f"{REAL_FACTS} facts per module")
    return out


def _engine_text(m):
    out = ["COULD THE AUTOMATIC ENGINE PROVE THEM WITH NO FACTS? (abc pdr)"]
    if m["engine"]:
        out += _wrap_items("kept designs:",
                           engine_share(m["engine"]).split(", "))
    else:
        out.append("  no designs kept")
    if m["engine_starts"]:
        out += _wrap_items("designs the run started from:",
                           engine_share(m["engine_starts"]).split(", "))
    out += _wrap("A design the engine proves alone needs no hand-written "
                  "facts; where it runs out of time is where engineers "
                  "write them by hand.")
    return out


def _cost_text(m):
    total, kinds = m["calls"]
    out = ["MODEL CALLS (llm_client.PRICES, half price for batch calls)"]
    if not total["calls"]:
        return out + ["  no call log (calls.jsonl) in the log folder"]
    out.append(f"  {total['calls']} calls, {total['batch']} of them at half "
               f"price as a batch: ${total['dollars']:.2f}")
    out.append(f"  tokens: {total['in']:,} in, {total['out']:,} out")
    for kind in CALL_KINDS:
        if kinds[kind]["calls"]:
            cost = f"${kinds[kind]['dollars']:.2f}"
            out.append(f"    {kind:42s} {kinds[kind]['calls']:4d} calls "
                       f"{cost:>9s}")
    if m["kept"]:
        out.append(f"  per kept design: ${total['dollars'] / m['kept']:.2f}")
    for model, n in sorted(total["unpriced"].items()):
        out.append(f"  {n} call(s) of {model}, which has no price in "
                   "llm_client.PRICES, are not in the dollars")
    return out


def render(m):
    """The report as text."""
    out = [f"REALISM REPORT for the run logged in {m['log_dir']}",
           f"designs kept: rows of {m['corpus']} promoted from its log",
           "real numbers: measured 15 Oct 2026 over 560 modules of PULP "
           "common_cells, lowRISC",
           "Ibex, OpenTitan, alexforencich verilog-axis and verilog-axi, "
           "ZipCPU wb2axip", ""]
    for section in (_steps_text, _growth_text, _size_text, _traits_text,
                    _facts_text, _engine_text, _cost_text):
        out += section(m) + [""]
    return "\n".join(out)


def write_report(log_dir, corpus, out=None):
    """Measure the run, write the report to `out` (by default
    realism_report.txt in the log folder) and print it. Returns the
    path written."""
    text = render(measure(log_dir, corpus))
    path = Path(out) if out else Path(log_dir) / REPORT_NAME
    path.write_text(text)
    print(text + f"(saved in {path})")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--log-dir", type=Path, required=True,
                   help="the run's log folder (batch.py --log-dir)")
    p.add_argument("--corpus", type=Path, required=True,
                   help="the corpus the run grew (batch.py --corpus)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"where to write the report; default "
                        f"<log-dir>/{REPORT_NAME}")
    args = p.parse_args(argv)
    if not args.log_dir.is_dir():
        p.error(f"no log folder {args.log_dir}")
    if not args.corpus.exists():
        p.error(f"no corpus {args.corpus}")
    write_report(args.log_dir, args.corpus, args.out)


if __name__ == "__main__":
    main()
