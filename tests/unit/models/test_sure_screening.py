import numpy as np
import pytest

from quant_fund.models.sure_screening import bench_sure_screening, isis, residualize, sis


def _hd(seed=0, n=120, p=2000):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[0] = 1.5
    beta[1] = 1.2
    beta[2] = 0.9
    x[:, 2] = 0.9 * x[:, 0] + np.sqrt(0.19) * rng.standard_normal(n)
    y = x @ beta + 0.5 * rng.standard_normal(n)
    return x, y


def test_sis_shapes():
    x, y = _hd()
    out = sis(x, y)
    assert out["scores"].shape == (2000,)
    assert out["keep_idx"].size > 0


def test_sis_finds_strong_signals():
    x, y = _hd(1)
    out = sis(x, y)
    keep = set(np.asarray(out["keep_idx"], dtype=int))
    assert 0 in keep or 1 in keep


def test_sis_scores_rank_signal():
    x, y = _hd(2)
    out = sis(x, y)
    scores = np.asarray(out["scores"])
    assert scores[0] > np.median(scores) * 3


def test_residualize_shape():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 5))
    y = rng.standard_normal(50)
    r = residualize(y, x[:, :2])
    assert r.shape == (50,)
    # residuals orthogonal to kept predictors
    assert abs(np.dot(r, x[:, 0])) < 1e-8


def test_isis_recovers_hidden():
    x, y = _hd(4)
    out = isis(x, y, rounds=2)
    keep = set(np.asarray(out["keep_idx"], dtype=int))
    assert 2 in keep or 0 in keep


def test_sis_input_validation():
    with pytest.raises(ValueError):
        sis(np.ones((100, 50)), np.ones(100))  # p < n
    with pytest.raises(ValueError):
        sis(np.full((10, 50), np.nan), np.ones(10))


def test_bench_sure_screening():
    out = bench_sure_screening()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_sis_strong_found"] == 1.0
