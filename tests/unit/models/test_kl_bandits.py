from quant_fund.models.kl_bandits import bench_kl_bandits, kl_bern, klucb_index


def test_kl_bern_zero():
    assert abs(kl_bern(0.5, 0.5)) < 1e-9


def test_kl_bern_positive():
    assert kl_bern(0.3, 0.7) > 0.3


def test_index_above_mean():
    idx = klucb_index(0.5, n=10, t=100)
    assert 0.5 < idx <= 1.0


def test_index_shrinks_with_n():
    i1 = klucb_index(0.5, n=10, t=1000)
    i2 = klucb_index(0.5, n=200, t=1000)
    assert i2 < i1


def test_klucb_beats_ucb1():
    out = bench_kl_bandits(seed=4)
    assert out["synthetic_klucb_regret"] < out["synthetic_ucb1_regret"]
