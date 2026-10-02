import numpy as np

from quant_fund.models.naive_bayes import (
    bench_naive_bayes,
    bernoulli_nb_fit,
    bernoulli_nb_predict,
    gaussian_nb_fit,
    gaussian_nb_predict,
    multinomial_nb_fit,
    multinomial_nb_predict,
)


def test_gaussian_nb():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0], 0.4, (60, 2)), rng.normal([3, 0], 0.4, (60, 2))])
    y = np.r_[np.zeros(60), np.ones(60)]
    m = gaussian_nb_fit(x, y)
    assert (gaussian_nb_predict(m, x) == y).mean() > 0.95


def test_multinomial_nb():
    rng = np.random.default_rng(1)
    docs = [rng.multinomial(20, np.r_[0.8, np.full(9, 0.0222)]) for _ in range(40)]
    docs += [rng.multinomial(20, np.r_[np.full(9, 0.0222), 0.8]) for _ in range(40)]
    x = np.asarray(docs)
    y = np.r_[np.zeros(40), np.ones(40)]
    m = multinomial_nb_fit(x, y)
    assert (multinomial_nb_predict(m, x) == y).mean() > 0.9


def test_bernoulli_nb():
    rng = np.random.default_rng(2)
    x = (rng.uniform(0, 1, (100, 6)) < np.r_[0.8, np.full(5, 0.1)]).astype(float)
    x[50:] = 1 - x[50:]
    y = np.r_[np.zeros(50), np.ones(50)]
    m = bernoulli_nb_fit(x, y)
    assert (bernoulli_nb_predict(m, x) == y).mean() > 0.95


def test_bench_naive_bayes():
    out = bench_naive_bayes(seed=564)
    assert out["synthetic_gnb_acc"] > 0.95
    assert out["synthetic_mnb_acc"] > 0.9
