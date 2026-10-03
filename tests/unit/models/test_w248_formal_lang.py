"""Wave-248 formal-language canon tests."""

from quant_fund.models.brzozowski_deriv import bench_brzozowski_deriv, matches
from quant_fund.models.cellular_automata import bench_cellular_automata, eca_step
from quant_fund.models.dfa_equiv import bench_dfa_equiv, equivalent, minimize
from quant_fund.models.mealy_moore import bench_mealy_moore, run_mealy
from quant_fund.models.pda_sim import bench_pda_sim, run_pda
from quant_fund.models.turing_machine import bench_turing_machine, run_tm


def test_tm_basic():
    tr = {("q0", "1"): ("q0", "1", 1), ("q0", "_"): ("qh", "_", 0)}
    out, halted = run_tm(tr, "11")
    assert halted


def test_tm_bench():
    assert bench_turing_machine()["synthetic_increment_exact"] == 1.0


def test_pda_basic():
    tr = {
        ("q0", "a", "Z"): [("q0", "ZA")],
        ("q0", "a", "A"): [("q0", "AA")],
        ("q0", "b", "A"): [("q1", "")],
        ("q1", "b", "A"): [("q1", "")],
        ("q1", None, "Z"): [("qf", "")],
    }
    assert run_pda(tr, "aabb")
    assert not run_pda(tr, "aab")


def test_pda_bench():
    assert bench_pda_sim()["synthetic_palindrome_exact"] == 1.0


def test_brzozowski_basic():
    r = ("cat", ("lit", "a"), ("star", ("lit", "b")))
    assert matches(r, "abbb") and not matches(r, "abba")


def test_brzozowski_bench():
    assert bench_brzozowski_deriv()["synthetic_match_agrees_oracle"] == 1.0


def test_dfa_equiv_basic():
    d = (2, {(0, "a"): 1, (0, "b"): 0, (1, "a"): 1, (1, "b"): 0}, {1}, 0)
    dm = minimize(*d, ("a", "b"))
    assert equivalent(d, dm, ("a", "b"))


def test_dfa_equiv_bench():
    assert bench_dfa_equiv()["synthetic_minimize_equivalent"] == 1.0


def test_mealy_basic():
    m = (1, {(0, "a"): 0, (0, "b"): 0}, {(0, "a"): "1", (0, "b"): "0"}, 0)
    assert run_mealy(m, ["a", "b"]) == ["1", "0"]


def test_mealy_bench():
    assert bench_mealy_moore()["synthetic_conversion_equivalent"] == 1.0


def test_ca_basic():
    assert eca_step([0, 1, 0], 90) == [1, 0, 1]


def test_ca_bench():
    assert bench_cellular_automata()["synthetic_rule90_lucas"] == 1.0
