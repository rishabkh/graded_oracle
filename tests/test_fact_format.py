"""fact_format: which helper facts the benchmark's checker can read."""
import pytest

from fact_format import all_plain, fact_problems, unreadable_facts


@pytest.mark.parametrize("fact", [
    "count == wr_ptr - rd_ptr",
    "!(v[0] && v[1]) || (age[0] != age[1])",
    "occ == valid[0] + valid[1] + valid[2]",
    "x <= 8'd200",
    "state != 2'b11",
    "a$b == 1'b0",            # a `$` inside a name is part of the name
    "c == 3 * x",
])
def test_plain_facts_pass(fact):
    assert fact_problems(fact) == []


@pytest.mark.parametrize("fact,name", [
    ("occ == $countones(valid)", "$countones"),
    ("$onehot(st)", "$onehot"),
    ("!pv || (b == $past(a))", "$past"),
    ("$stable(d) || !hold", "$stable"),
    ("$signed(acc) >= 0", "$signed"),
])
def test_a_system_function_is_found(fact, name):
    assert fact_problems(fact) == [f"system function {name}"]


def test_the_compact_form_is_found():
    assert fact_problems("forall k in 0..7: !v[k] || (tag[k] != 4'd15)") \
        == ["compact forall form"]


def test_a_dotted_name_is_found():
    assert fact_problems("u_fifo.count <= 4'd8") == ["dotted name"]


def test_every_problem_of_one_fact_is_named():
    probs = fact_problems("forall k in 0..3: sr.mem[k] == $past(d)")
    assert probs == ["system function $past", "compact forall form",
                     "dotted name"]


def test_the_list_helpers_agree():
    facts = ["a == b", "occ == $countones(v)", "x <= 4'd8"]
    assert unreadable_facts(facts) == [
        ("occ == $countones(v)", ["system function $countones"])]
    assert not all_plain(facts)
    assert all_plain(["a == b", "x <= 4'd8"])
