import numpy as np

from quant_fund.models.td_learning import (
    bench_td_learning,
    q_learning,
    sarsa,
    td0_eval,
)


def _chain():
    n_s, n_a = 6, 2
    p = np.zeros((n_s, n_a, n_s))
    for s in range(n_s - 1):
        p[s, :, s] = 1.0
    p[:, 1, :] = 0.0
    for s in range(n_s - 1):
        p[s, 1, s + 1] = 1.0
    p[n_s - 1, :, :] = 0.0
    p[n_s - 1, :, n_s - 1] = 1.0
    r = np.zeros((n_s, n_a))
    r[:, 1] = 1.0
    r[n_s - 1, :] = 2.0
    return p, r


def test_q_learning_finds_policy():
    p, r = _chain()
    q = q_learning(p, r, 0.95, episodes=3000, seed=0)
    assert np.mean(q.argmax(1) == 1) >= 0.8


def test_sarsa_finds_policy():
    p, r = _chain()
    q = sarsa(p, r, 0.95, episodes=3000, seed=1)
    assert np.mean(q.argmax(1) == 1) >= 0.8


def test_td0_tracks():
    p, r = _chain()
    pi = np.zeros((6, 2))
    pi[:, 1] = 1.0
    v = td0_eval(p, r, pi, 0.95, episodes=3000, seed=2)
    assert v[-1] > 20.0  # 2/(1-0.95) = 40-ish discounted
    assert v[0] > 0.0


def test_bench_keys():
    out = bench_td_learning()
    assert out["synthetic_ql_policy_agree"] >= 0.7
    assert out["synthetic_sarsa_policy_agree"] >= 0.7
    assert out["synthetic_td0_eval_err"] < 0.6
