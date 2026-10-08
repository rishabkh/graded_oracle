"""The v5 size seeds (8 Oct 2026): sizes real chips use, weighted the way
real chips use them, from general sources only."""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "initiator"))

import run as R                                                   # noqa: E402

FILE = ROOT / "initiator" / "scales_v5.txt"
LINES = R._pool(FILE)


def _width(line):
    return int(re.search(r"(\d+)-bit", line).group(1))


def _entries(line):
    return int(re.search(r"give that storage (\d+) entries", line).group(1))


def test_one_hundred_seeds_weighted_by_repetition():
    assert len(LINES) == 100


def test_widths_lean_to_32_and_64_bits():
    assert Counter(map(_width, LINES)) == {8: 15, 16: 15, 32: 30, 64: 30,
                                           128: 10}


def test_storage_leans_to_8_to_32_slots_with_a_share_of_big_storage():
    n = Counter(map(_entries, LINES))
    assert n[128] == 5 and n[256] == 5
    small = {2: 3, 4: 6, 8: 12, 16: 24, 32: 48, 64: 96}
    got = {k: n[k] + n[v] for k, v in small.items()}
    assert got == {2: 6, 4: 10, 8: 16, 16: 26, 32: 19, 64: 13}
    assert sum(n[v] for v in small.values()) == 18      # about 1 in 5 odd


def test_128_bit_is_for_data_never_counters():
    for line in LINES:
        if _width(line) == 128:
            assert "data words 128-bit" in line and "counters" in line
            assert "counters and data words 128" not in line


def test_the_file_records_where_the_weights_came_from():
    head = FILE.read_text().split("Make the design", 1)[0]
    for word in ("never from the test sets", "Ibex", "OpenTitan",
                 "seed 20261008", "Rishab's call"):
        assert word in head, word
