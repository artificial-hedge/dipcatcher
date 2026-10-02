import numpy as np

from quant_fund.models.mdp_solvers import (
    bench_mdp_solvers,
    policy_eval,
    policy_iteration,
    q_values,
    value_iteration,
)


def _tiny():
    p = np.zeros((3, 2, 3))
    p[0, 0] = [0.9, 0.1, 0.0]
    p[0, 1] = [0.2, 0.8, 0.0]
    p[1, 0] = [0.0, 0.9, 0.1]
    p[1, 1] = [0.0, 0.2, 0.8]
    p[2, :, 2] = 1.0
    r = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    return p, r


def test_policy_eval_exact():
    p, r = _tiny()
    pi = np.full((3, 2), 0.5)
    v = policy_eval(p, r, pi, 0.9)
    # absorbing goal: V(2) = 1/(1-0.9)
    assert abs(v[2] - 10.0) < 1e-8


def test_vi_pi_agree():
    p, r = _tiny()
    v1, pi1 = value_iteration(p, r, 0.9)
    v2, pi2 = policy_iteration(p, r, 0.9)
    assert np.max(np.abs(v1 - v2)) < 1e-6
    assert np.array_equal(pi1.argmax(1), pi2.argmax(1))


def test_q_values_shape():
    p, r = _tiny()
    v, _ = value_iteration(p, r, 0.9)
    q = q_values(p, r, v, 0.9)
    assert q.shape == (3, 2)


def test_bench_keys():
    out = bench_mdp_solvers()
    assert out["synthetic_vi_pi_gap"] < 1e-6
    assert out["synthetic_policy_agree"] == 1.0
