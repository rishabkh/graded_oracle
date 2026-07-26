from oracle.parse import parse_cover_log, tail

# Verbatim lines (timestamps and all) from a real Stage 1 run:
# vacuity_test/counter_cover/logfile.txt (sby + smtbmc, oss-cad-suite).
MIXED_LOG = """\
SBY 13:58:46 [counter_cover] engine_0: ##   0:00:00  Reached cover statement in step 1 at counter: counter.sv:36.5-36.26 (_witness_.check_cover_counter_sv_36_15)
SBY 13:58:46 [counter_cover] engine_0: ##   0:00:00  Unreached cover statement at counter: counter.sv:35.5-35.26 (_witness_.check_cover_counter_sv_35_13)
SBY 13:58:46 [counter_cover] engine_0: Status returned by engine: FAIL
SBY 13:58:46 [counter_cover] summary: engine_0 (smtbmc) returned FAIL
SBY 13:58:46 [counter_cover] summary: cover trace: counter_cover/engine_0/trace0.vcd
SBY 13:58:46 [counter_cover] summary:   reached cover statement counter._witness_.check_cover_counter_sv_36_15 at counter.sv:36.5-36.26 step 1
SBY 13:58:46 [counter_cover] summary: unreached cover statements:
SBY 13:58:46 [counter_cover] summary:   counter._witness_.check_cover_counter_sv_35_13 at counter.sv:35.5-35.26
SBY 13:58:46 [counter_cover] DONE (FAIL, rc=2)
"""

ALL_REACHED_LOG = """\
SBY 13:58:46 [job] engine_0: ##   0:00:00  Reached cover statement in step 1 at m: m.sv:9.5-9.20 (_witness_.check_cover_m_sv_9_1)
SBY 13:58:46 [job] DONE (PASS, rc=0)
"""


def test_mixed_reached_and_unreached():
    reached, unreached = parse_cover_log(MIXED_LOG)
    assert reached == {36}
    assert unreached == {35}


def test_summary_section_is_not_double_counted():
    # The lowercase summary repeats both facts; only the capitalized
    # engine lines may be counted, and neither line may leak into the
    # other set.
    reached, unreached = parse_cover_log(MIXED_LOG)
    assert 36 not in unreached
    assert 35 not in reached


def test_all_reached():
    reached, unreached = parse_cover_log(ALL_REACHED_LOG)
    assert reached == {9} and unreached == set()


def test_empty_log():
    assert parse_cover_log("") == (set(), set())


def test_tail():
    text = "\n".join(str(i) for i in range(100))
    t = tail(text, 3)
    assert t == "97\n98\n99"
    assert tail("short", 40) == "short"
