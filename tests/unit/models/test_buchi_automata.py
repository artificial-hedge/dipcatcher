from quant_fund.models.buchi_automata import Buchi, accepts_up, bench_buchi_automata


def _a2() -> Buchi:
    a2 = Buchi()
    a2.start = 0
    a2.trans = {
        (0, "a"): [1],
        (0, "b"): [0],
        (1, "a"): [1],
        (1, "b"): [2],
        (2, "a"): [1],
        (2, "b"): [2],
    }
    a2.final = {2}
    return a2


def test_prefix_ab_with_b_free_cycle_rejects():
    """Probe for the oracle defect: 'ab' inside the prefix reaches
    state 2 once, but with no 'b' in the cycle state 2 is never
    revisited — the old truth called this accepted."""
    a2 = _a2()
    assert not accepts_up(a2, ["a", "b"], ["a", "a"])


def test_boundary_ab_with_b_in_cycle_accepts():
    """An 'ab' spanning the prefix->cycle boundary reaches 2 once;
    every later 'b' in the cycle re-enters it — accepted."""
    a2 = _a2()
    assert accepts_up(a2, ["a"], ["b", "b"])


def test_wrap_around_ab_accepts():
    a2 = _a2()
    # 'ab' across the cycle-end -> cycle-start wrap recurs every period
    assert accepts_up(a2, [], ["b", "a"])


def test_internal_ab_accepts():
    a2 = _a2()
    assert accepts_up(a2, ["a", "a"], ["a", "b"])


def test_no_b_in_cycle_rejects():
    a2 = _a2()
    assert not accepts_up(a2, [], ["a"])


def test_bench_oracle_agrees_fully():
    out = bench_buchi_automata()
    assert out["synthetic_inf_a"] == 1.0
    assert out["synthetic_inf_both"] == 1.0
