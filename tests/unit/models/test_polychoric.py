import numpy as np
import pytest
from scipy import stats

from quant_fund.models.polychoric import (
    bench_polychoric,
    polychoric,
    tetrachoric,
)


def _table(rho, seed=0, n=4000, k=4):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(2), [[1, rho], [rho, 1]], size=n)
    cuts = stats.norm.ppf(np.linspace(0, 1, k + 1))
    ix = np.clip(np.searchsorted(cuts, z[:, 0]) - 1, 0, k - 1)
    iy = np.clip(np.searchsorted(cuts, z[:, 1]) - 1, 0, k - 1)
    t = np.zeros((k, k))
    for i, j in zip(ix, iy, strict=True):
        t[i, j] += 1
    return t


def test_polychoric_positive_rho():
    out = polychoric(_table(0.6))
    assert abs(out["rho"] - 0.6) < 0.08


def test_polychoric_negative_rho():
    out = polychoric(_table(-0.5, seed=1))
    assert abs(out["rho"] + 0.5) < 0.1


def test_polychoric_zero_rho():
    out = polychoric(_table(0.0, seed=2))
    assert abs(out["rho"]) < 0.1


def test_tetrachoric():
    rng = np.random.default_rng(3)
    z = rng.multivariate_normal(np.zeros(2), [[1, 0.5], [0.5, 1]], size=4000)
    mx, my = np.median(z[:, 0]), np.median(z[:, 1])
    t22 = np.array(
        [
            [np.sum((z[:, 0] <= mx) & (z[:, 1] <= my)), np.sum((z[:, 0] <= mx) & (z[:, 1] > my))],
            [np.sum((z[:, 0] > mx) & (z[:, 1] <= my)), np.sum((z[:, 0] > mx) & (z[:, 1] > my))],
        ]
    )
    out = tetrachoric(t22)
    assert abs(out["rho"] - 0.5) < 0.15


def test_polychoric_independence_beats_ind_rho():
    # independent data should fit rho ~ 0
    out = polychoric(_table(0.0, seed=4))
    assert abs(out["rho"]) < 0.1


def test_polychoric_input_validation():
    with pytest.raises(ValueError):
        polychoric(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        polychoric(np.ones((2, 2)) * np.nan)
    with pytest.raises(ValueError):
        polychoric(np.array([1.0, 2.0, 3.0]))


def test_bench_polychoric():
    out = bench_polychoric()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_polychoric_err"] < 0.08
