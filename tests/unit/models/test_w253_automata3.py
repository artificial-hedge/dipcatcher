"""Wave-253 automata-3 canon tests."""

from quant_fund.models.buchi_automata import Buchi, accepts_up, bench_buchi_automata
from quant_fund.models.cfg_pda_equiv import bench_cfg_pda_equiv, cyk, pda_accept
from quant_fund.models.register_automata import bench_register_automata, run_ra
from quant_fund.models.tree_automata import _rules, accepts, bench_tree_automata
from quant_fund.models.two_way_dfa import bench_two_way_dfa, run_2dfa
from quant_fund.models.weighted_fst import WFST, bench_weighted_fst


def test_tree_basic():
    rules, leaf, finals = _rules()
    assert accepts(("and", "1", "1"), rules, leaf, finals)
    assert not accepts(("and", "1", "0"), rules, leaf, finals)


def test_tree_bench():
    assert bench_tree_automata()["synthetic_acceptance_exact"] == 1.0


def test_buchi_basic():
    a = Buchi()
    a.trans = {(0, "a"): [1], (0, "b"): [0], (1, "a"): [1], (1, "b"): [0]}
    a.final = {1}
    assert accepts_up(a, [], ["a", "b"])
    assert not accepts_up(a, [], ["b"])


def test_buchi_bench():
    assert bench_buchi_automata()["synthetic_inf_a"] == 1.0


def test_wfst_basic():
    f = WFST()
    f.trans = {0: {"a": [(1, "A", 1.0), (2, "A", 3.0)]}}
    f.final = {1: 0.0, 2: 0.0}
    assert f.map_weight(["a"]) == 1.0


def test_wfst_bench():
    assert bench_weighted_fst()["synthetic_min_weight_exact"] == 1.0


def test_cfg_pda_basic():
    rules = {"S": [("A", "B"), ("A", "T")], "T": [("S", "B")], "A": [("a",)], "B": [("b",)]}
    assert cyk(rules, "S", "ab")
    assert pda_accept(rules, "S", "ab")
    assert cyk(rules, "S", "aabb")
    assert not cyk(rules, "S", "abba")


def test_cfg_pda_bench():
    assert bench_cfg_pda_equiv()["synthetic_pda_cyk_agree"] == 1.0


def test_2dfa_basic():
    tr = {(0, c): (0, 1) for c in "ab⊢"}
    tr[(0, "a")] = (1, 1)
    tr[(0, "⊣")] = (3, 0)
    tr[(1, "b")] = (2, 0)
    tr[(1, "a")] = (1, 1)
    tr[(1, "⊣")] = (3, 0)
    assert run_2dfa(tr, 0, 2, "ab")
    assert not run_2dfa(tr, 0, 2, "ba")


def test_2dfa_bench():
    assert bench_two_way_dfa()["synthetic_contains_ab"] == 1.0


def test_ra_basic():
    tr = {(0, "fresh"): (1, 0, -1), (1, "fresh"): (1, -1, -1), (1, "eq"): (2, -1, 0)}
    assert run_ra(tr, 0, {2}, [5, 9, 5])
    assert not run_ra(tr, 0, {2}, [5, 9])


def test_ra_bench():
    assert bench_register_automata()["synthetic_first_last_eq"] == 1.0
