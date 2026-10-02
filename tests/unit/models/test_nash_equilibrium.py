import numpy as np

from quant_fund.models.nash_equilibrium import (
    bench_nash_equilibrium,
    exploitability,
    fictitious_play,
    regret_matching,
    support_enumeration,
)


def test_fictitious_rps():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    r = fictitious_play(rps, it=6000)
    row = np.asarray(r["row"])
    assert np.abs(row - 1 / 3).max() < 0.1


def test_exploitability_zero_at_ne():
    mp = np.array([[1.0, -1.0], [-1.0, 1.0]])
    e = exploitability(mp, np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert e < 1e-9


def test_support_enum_pure_ne():
    pd_a = np.array([[3.0, 0.0], [5.0, 1.0]])
    pd_b = np.array([[3.0, 5.0], [0.0, 1.0]])
    sols = support_enumeration(pd_a, pd_b)
    # prisoner's dilemma: defect/defect is the unique NE
    assert len(sols) == 1
    assert sols[0][0][1] > 0.99 and sols[0][1][1] > 0.99


def test_regret_matching_uniform_mp():
    mp = np.array([[1.0, -1.0], [-1.0, 1.0]])
    s = regret_matching(mp, it=5000)
    assert np.abs(s - 0.5).max() < 0.15


def test_bench_nash_equilibrium():
    out = bench_nash_equilibrium(seed=569)
    assert out["synthetic_fp_exploit"] < 0.3
    assert out["synthetic_ne_sols"] >= 2
