import numpy as np

from quant_fund.models.fuzzy_clustering import (
    bench_fuzzy,
    fuzzy_cmeans,
    partition_coefficient,
    partition_entropy,
    xie_beni,
)


def test_membership_rows_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2))
    r = fuzzy_cmeans(x, 3, seed=0)
    u = np.asarray(r["membership"])
    np.testing.assert_allclose(u.sum(axis=1), 1.0, atol=1e-8)


def test_fcm_finds_centers():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal([0, 0], 0.2, (30, 2)), rng.normal([4, 4], 0.2, (30, 2))])
    r = fuzzy_cmeans(x, 2, seed=1)
    v = np.asarray(r["centers"])
    d = np.min(((v[:, None, :] - np.array([[0, 0], [4, 4]])) ** 2).sum(axis=2), axis=1)
    assert np.all(d < 0.5)


def test_validity_ranges():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 2))
    r = fuzzy_cmeans(x, 2, seed=2)
    u = np.asarray(r["membership"])
    v = np.asarray(r["centers"])
    assert 0.5 <= partition_coefficient(u) <= 1.0
    assert partition_entropy(u) >= 0.0
    assert xie_beni(x, u, v) > 0


def test_bench_fuzzy_runs():
    out = bench_fuzzy(seed=542)
    assert out["synthetic_fcm_acc"] > 0.95
    assert out["synthetic_fcm_pc"] > out["synthetic_fcm_pc_scr"]
