"""Real open-source modules as starting designs for the initiator.

28 Oct 2026, Rishab chose option 1: real designs only as starting points;
our pipeline writes every property and helper fact, and no human-written
proof reaches training. So each module's own formal section (ZipCPU's
`ifdef FORMAL` blocks) is cut out here, before any model sees it.

Sources (shallow clones in ~/Desktop/HarvardResearch/open_source_rtl_2026-10-28,
the repos of the 15 Oct realism study; the test cores and the Technion
designs are not among them): alexforencich/verilog-axis and verilog-axi
(MIT), ZipCPU/wb2axip (Apache 2.0); and, for more designs (28 Oct 2026),
alexforencich's verilog-ethernet, verilog-i2c, verilog-uart and
verilog-wishbone (MIT). A module is kept when yosys reads it with the
sub-modules it uses from the same repo, it has one clock, no clock
crossing, it is 30 to 450 lines in all, and it is not a copy of one kept
before.

  hwtools
  venv/bin/python initiator/open_source_seeds.py      # writes OUT, prints counts
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path.home() / "Desktop/HarvardResearch/open_source_rtl_2026-10-28"
OUT = HERE / "open_source_seeds_2026-10-28.jsonl"
REPOS = {"verilog-axis": "MIT", "verilog-axi": "MIT", "wb2axip": "Apache-2.0",
         # 28 Oct 2026, to reach 250 designs: the same author's other MIT repos
         "verilog-ethernet": "MIT", "verilog-i2c": "MIT", "verilog-uart": "MIT",
         "verilog-wishbone": "MIT",
         # 28 Oct 2026, Rishab: "at least 150 more opensource designs": his PCIe
         # and DSP repos, and Joachim Strömbergson's crypto cores (BSD 2-clause),
         # sources under src/rtl. Ultra-Embedded's cores were left out: LGPL.
         "alexforencich_verilog-pcie": "MIT", "alexforencich_verilog-dsp": "MIT",
         **{f"secworks_{n}": ("BSD-2-Clause", "src/rtl") for n in (
             "aes", "sha256", "sha1", "sha512", "chacha", "siphash", "md5", "blake2s",
             "poly1305", "uart", "trng", "cmac", "keywrap", "xchacha", "prince")},
         # 28 Oct 2026, 70 more: the most-starred permissively licensed Verilog
         # repos on GitHub, minus the RISC-V test cores (VexRiscv, nerv,
         # picorv32, serv) and riscv-formal; "**" reads every folder but tests
         "aolofsson_oh": ("MIT", "**"), "seldridge_verilog": ("Apache-2.0", "**"),
         "ultraembedded_riscv": ("BSD-3-Clause", "**"),
         "ultraembedded_biriscv": ("Apache-2.0", "**"),
         "darklife_darkriscv": ("BSD-3-Clause", "**"), "Wren6991_Hazard3": ("Apache-2.0", "**"),
         "riscv-mcu_e203_hbirdv2": ("Apache-2.0", "**"), "FeSens_openTPU": ("Apache-2.0", "**"),
         "dawsonjon_fpu": ("MIT", "**")}
_TESTS = re.compile(r"/(tb|tbs|test|tests|sim|bench|testbench|example|examples|doc|docs)/|_tb\.v$|/tb_", re.I)
MIN_LINES, MAX_LINES = 30, 450
# two clocks or a clock crossing: our checker steps one clock
SKIP_NAME = re.compile(r"async|xclk|cdc|_clk|clk_|sync_reset|vfifo|migsdram"
                       # pin helpers on both clock edges, and pure wire reordering
                       # (28 Oct 2026, verilog-ethernet)
                       r"|iddr|oddr|ssio_|interleave", re.I)
# all capitals: a vendor part's simulation model (IBUFE3, IDELAYE2, CLKDIV in
# aolofsson/oh), not a design
VENDOR_MODEL = re.compile(r"^[A-Z0-9_]+$")
FORMAL_MACROS = ("FORMAL", "VERIFIC")

_MODULE = re.compile(r"^\s*module\s+(\w+)", re.M)
_INST = re.compile(r"^\s*(\w+)\s*(?:#\s*\(|\w+\s*\()", re.M)


def strip_formal(text):
    """Cut `ifdef FORMAL ... `endif blocks (keeping an `else branch) and keep
    the body of `ifndef FORMAL; other directives are left for yosys."""
    out, stack = [], []          # stack entries: True when lines are kept
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"`(ifdef|ifndef)\s+(\w+)", s)
        if m and m.group(2) in FORMAL_MACROS:
            stack.append(("formal", m.group(1) == "ifndef"))
            continue
        if m:
            stack.append(("other", None))
            out.append(line) if all(k for _, k in stack if _ == "formal") else None
            continue
        if s.startswith("`else") and stack:
            kind, keep = stack[-1]
            if kind == "formal":
                stack[-1] = (kind, not keep)
                continue
        if s.startswith("`endif") and stack:
            kind, _ = stack.pop()
            if kind == "formal":
                continue
        if all(keep for kind, keep in stack if kind == "formal"):
            out.append(line)
    return "\n".join(out) + "\n"


def modules_of(repo_dir, src="rtl"):
    """{module name: (path, text with its formal sections cut)}"""
    found = {}
    paths = (repo_dir.rglob("*.v") if src == "**" else (repo_dir / src).glob("*.v"))
    for path in sorted(p for p in paths if not _TESTS.search(str(p.relative_to(repo_dir.parent)))):
        text = strip_formal(path.read_text(errors="replace"))
        for name in _MODULE.findall(text):
            found.setdefault(name, (path, text))
    return found


def with_submodules(name, mods):
    """The module's file plus the files of every module it uses, deepest first."""
    order, seen = [], set()

    def visit(n):
        if n in seen or n not in mods:
            return
        seen.add(n)
        path, text = mods[n]
        for used in _INST.findall(text):
            if used != n and used in mods:
                visit(used)
        if path not in order:
            order.append(path)
    visit(name)
    return order


def yosys_reads(text, top):
    """(True, "") when yosys builds the module within a minute. Ethernet's
    CRC units unroll into logic yosys took over 2 minutes to build (28 Oct
    2026); a proof on them would be slower still, so they are skipped."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "seed.v").write_text(text)
        try:
            proc = subprocess.run(["yosys", "-p", f"read -formal seed.v; prep -top {top}; stat"],
                                  cwd=d, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "yosys took over 60 s to build it"
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout)[-300:]
        # 28 Oct 2026: 54 of 129 modules in the third run were adders, muxes,
        # clock gates and FPGA pin models with nothing stored; the model rightly
        # wrote no property, and each answer still cost a call
        state = sum(int(n) for n in re.findall(
            r"\$(?:a?l?dff\w*|sdff\w*|dffsr\w*|mem\w*)\s+(\d+)", proc.stdout))
        return (True, "") if state else (False, "no flip-flop or memory: nothing to prove")


def clocks(text):
    """The most clocks any one module in `text` uses, resets left out. Per
    module: a sub-module's own name for the same clock (skidbuffer's i_clk
    on wb2axip's S_AXI_ACLK) is not a second clock."""
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    return max((len({c for c in re.findall(r"posedge\s+(\w+)", body)
                     if not re.search(r"rst|reset", c, re.I)})
                for body in re.split(r"\bendmodule\b", code)), default=0)


def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)).strip()


# 28 Oct 2026: the first list had wb2axip's afifo (two clocks, which the name
# rule missed) and alexforencich's arbiter and priority_encoder twice, once
# from each of his repos; a near-copy is dropped by the module name
def collect():
    rows, seen_text, seen_alex = [], set(), set()
    for repo, licence in REPOS.items():
        licence, src = licence if isinstance(licence, tuple) else (licence, "rtl")
        rdir = SRC / repo
        commit = subprocess.run(["git", "-C", str(rdir), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        mods = modules_of(rdir, src)
        for name, (path, _) in mods.items():
            if SKIP_NAME.search(name) or SKIP_NAME.search(path.name) or VENDOR_MODEL.match(name):
                continue
            files = with_submodules(name, mods)
            text = "".join(strip_formal(p.read_text(errors="replace")) for p in files)
            n = len([l for l in text.splitlines() if l.strip() and not l.strip().startswith("//")])
            if not MIN_LINES <= n <= MAX_LINES or clocks(text) > 1:
                continue
            alex = "verilog-" in repo               # his repos share files
            if _norm(text) in seen_text or (alex and name in seen_alex):
                continue
            seen_text.add(_norm(text))
            if alex:
                seen_alex.add(name)
            ok, err = yosys_reads(text, name)
            rows.append({"name": f"{repo}/{name}", "top_module": name, "repo": repo,
                         "licence": licence, "commit": commit,
                         "files": [str(p.relative_to(rdir)) for p in files],
                         "lines": n, "reads": ok, "error": None if ok else err,
                         "verilog": text})
    return rows


def main():
    rows = collect()
    with open(OUT, "w") as fh:
        for r in rows:
            if r["reads"]:
                fh.write(json.dumps({k: v for k, v in r.items() if k not in ("reads", "error")}) + "\n")
    ok = [r for r in rows if r["reads"]]
    print(f"{len(ok)} modules kept of {len(rows)} in size range -> {OUT.name}")
    for repo in REPOS:
        print(f"  {repo}: {sum(1 for r in ok if r['repo'] == repo)}")
    bad = [r for r in rows if not r["reads"]]
    if bad:
        print("yosys could not read:", ", ".join(r["name"] for r in bad[:12]))


if __name__ == "__main__":
    main()
