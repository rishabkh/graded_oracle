from oracle.parse import parse_assert_failures, parse_cover_log, tail, parse_induction_status

# Verbatim from runs/counter_false_prove_*/job/logfile.txt.
ASSERT_FAIL_LOG = """\
SBY 15:09:12 [job] engine_0.basecase: ##   0:00:00  Assert failed in counter_false: counter_false.sv:15.9-15.31 (_witness_.check_assert_counter_false_sv_15_4)
SBY 15:09:12 [job] engine_0.induction: ##   0:00:00  Assert failed in counter_false: counter_false.sv:15.9-15.31 (_witness_.check_assert_counter_false_sv_15_4)
SBY 15:09:12 [job] engine_0.basecase: ##   0:00:00  Status: failed
SBY 15:09:12 [job] DONE (FAIL, rc=2)
"""

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


def test_assert_failures_extracted_by_line():
    assert parse_assert_failures(ASSERT_FAIL_LOG) == {15}


def test_assert_failures_empty_on_clean_log():
    assert parse_assert_failures(ALL_REACHED_LOG) == set()
    assert parse_assert_failures("") == set()


def test_assert_failures_ignore_cover_lines():
    assert parse_assert_failures(MIXED_LOG) == set()


def test_tail():
    text = "\n".join(str(i) for i in range(100))
    t = tail(text, 3)
    assert t == "97\n98\n99"
    assert tail("short", 40) == "short"


# --- induction status: verbatim lines from the real alu_budget without-run
# (rc=8: induction FAILED in the first second, base case timed out) ---

INDUCTION_FAIL_LOG = """\
SBY 19:16:29 [job] engine_0.induction: ##   0:00:00  Temporal induction failed!
SBY 19:16:29 [job] engine_0.induction: ##   0:00:00  Assert failed in alu_budget: alu_budget.sv:34.22-34.43 (_witness_.check_assert_alu_budget_sv_34_10)
SBY 19:16:29 [job] engine_0.induction: ##   0:00:00  Status: failed
SBY 19:16:29 [job] engine_0.induction: Status returned by engine for induction: FAIL
SBY 19:18:29 [job] DONE (TIMEOUT, rc=8)"""

INDUCTION_PASS_LOG = """\
SBY 22:31:49 [job] engine_0.induction: finished (returncode=0)
SBY 22:31:49 [job] engine_0.induction: Status returned by engine for induction: pass
SBY 22:31:49 [job] DONE (PASS, rc=0)"""


def test_induction_status_fail():
    assert parse_induction_status(INDUCTION_FAIL_LOG) == "FAIL"


def test_induction_status_pass():
    assert parse_induction_status(INDUCTION_PASS_LOG) == "pass"


def test_induction_status_absent():
    assert parse_induction_status("") is None
    assert parse_induction_status("engine_0: Status: failed") is None


def test_induction_status_ignores_summary_wording():
    # the summary section phrases it differently; only the engine status
    # line is definitive and only it may match
    assert parse_induction_status(
        "SBY 19:18:29 [job] summary: engine_0 (smtbmc yices) "
        "returned pass for induction") is None
