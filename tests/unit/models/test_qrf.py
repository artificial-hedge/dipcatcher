"""Tests for quant_fund.models.qrf — quantile regression forests (Meinshausen 2006). SYNTHETIC."""

import numpy as np
import pytest
from scipy.stats import kstest, spearmanr

from quant_fund.metrics.scoring import mean_pinball
from quant_fund.models.qrf import QuantileRegressionForest, weighted_quantiles

TAUS = np.array([0.1, 0.5, 0.9])


def _hetero(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """y = sin(2 pi x1) + (0.2 + x2) * eps, eps ~ N(0,1); scale grows with x2."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + (0.2 + X[:, 1]) * rng.normal(size=n)
    return X, y


def test_weighted_quantiles_matches_empirical_on_uniform_weights() -> None:
    v = np.arange(1.0, 11.0)
    w = np.full((1, 10), 0.1)
    q = weighted_quantiles(v, w, np.array([0.1, 0.5, 1.0 - 1e-9]))
    assert q.shape == (1, 3)
    assert q[0, 0] == 1.0 and q[0, 1] == 5.0 and q[0, 2] == 10.0


def test_weighted_quantiles_point_mass() -> None:
    v = np.array([3.0, 1.0, 2.0])
    w = np.array([[0.0, 1.0, 0.0]])
    assert np.all(weighted_quantiles(v, w, TAUS) == 1.0)


def test_weighted_quantiles_fail_closed() -> None:
    v = np.arange(3.0)
    with pytest.raises(ValueError):
        weighted_quantiles(v, np.ones((1, 3)), np.array([0.0]))
    with pytest.raises(ValueError):
        weighted_quantiles(v, np.ones((1, 2)), TAUS)
    with pytest.raises(ValueError):
        weighted_quantiles(v, np.zeros((1, 3)), TAUS)
    with pytest.raises(ValueError):
        weighted_quantiles(v, -np.ones((1, 3)), TAUS)
    with pytest.raises(ValueError):
        weighted_quantiles(np.array([0.0, np.nan, 1.0]), np.ones((1, 3)), TAUS)
    with pytest.raises(ValueError):
        weighted_quantiles(v, np.ones((1, 3)), np.array([np.nan]))


def test_weights_sum_to_one_and_mean_matches_weights() -> None:
    X, y = _hetero(400, 0)
    qrf = QuantileRegressionForest(n_estimators=50, min_samples_leaf=10, seed=0).fit(X, y)
    Xt, _ = _hetero(20, 1)
    w = qrf.weights(Xt)
    assert w.shape == (20, 400)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(qrf.predict_mean(Xt), w @ y)


def test_quantiles_monotone_and_track_heteroskedastic_scale() -> None:
    X, y = _hetero(1500, 2)
    qrf = QuantileRegressionForest(n_estimators=100, min_samples_leaf=10, seed=1).fit(X, y)
    Xt = np.column_stack([np.full(200, 0.25), np.linspace(0.0, 1.0, 200)])
    q = qrf.predict_quantiles(Xt, TAUS)
    assert np.all(np.diff(q, axis=1) >= 0.0)
    width = q[:, 2] - q[:, 0]
    assert np.mean(width[-50:]) > 1.5 * np.mean(width[:50])


def test_oos_coverage_and_pinball_beat_unconditional() -> None:
    X, y = _hetero(1500, 3)
    Xt, yt = _hetero(600, 4)
    qrf = QuantileRegressionForest(n_estimators=100, min_samples_leaf=10, seed=2).fit(X, y)
    q = qrf.predict_quantiles(Xt, TAUS)
    cov = np.mean((yt >= q[:, 0]) & (yt <= q[:, 2]))
    assert 0.72 <= cov <= 0.88
    for j, tau in enumerate(TAUS):
        uncond = np.full(yt.size, np.quantile(y, tau))
        assert mean_pinball(yt, q[:, j], float(tau)) < mean_pinball(yt, uncond, float(tau))


def test_oob_pit_is_approximately_uniform() -> None:
    X, y = _hetero(1200, 5)
    qrf = QuantileRegressionForest(
        n_estimators=200, min_samples_leaf=10, leaf_mode="oob", seed=3
    ).fit(X, y)
    weights = qrf.weights_oob_train()
    assert np.all(np.diag(weights) == 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    pit = qrf.pit_oob_train()
    assert np.all((pit >= 0.0) & (pit <= 1.0))
    assert kstest(pit, "uniform").pvalue > 0.01


def test_in_sample_all_mode_is_overconfident_relative_to_oob() -> None:
    X, y = _hetero(600, 6)
    q_all = QuantileRegressionForest(n_estimators=60, min_samples_leaf=5, seed=4).fit(X, y)
    q_oob = QuantileRegressionForest(
        n_estimators=60, min_samples_leaf=5, leaf_mode="oob", seed=4
    ).fit(X, y)
    w_all = q_all.predict_quantiles(X, TAUS)
    w_oob = q_oob.predict_quantiles(X, TAUS)
    assert np.mean(w_all[:, 2] - w_all[:, 0]) < np.mean(w_oob[:, 2] - w_oob[:, 0])


def test_predict_cdf_monotone_bounded() -> None:
    X, y = _hetero(300, 7)
    qrf = QuantileRegressionForest(n_estimators=30, seed=0).fit(X, y)
    grid = np.linspace(-3, 3, 25)
    cdf = qrf.predict_cdf(X[:5], grid)
    assert cdf.shape == (5, 25)
    assert np.all(np.diff(cdf, axis=1) >= -1e-12)
    assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0 + 1e-12)


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        QuantileRegressionForest(n_estimators=0)
    with pytest.raises(ValueError):
        QuantileRegressionForest(leaf_mode="bag")
    qrf = QuantileRegressionForest(n_estimators=5)
    with pytest.raises(RuntimeError):
        qrf.predict_quantiles(np.zeros((2, 2)), TAUS)
    X, y = _hetero(100, 0)
    with pytest.raises(ValueError):
        qrf.fit(X, np.r_[np.inf, y[1:]])
    with pytest.raises(ValueError):
        qrf.fit(X[:5], y[:5])
    qrf.fit(X, y)
    with pytest.raises(ValueError):
        qrf.predict_quantiles(np.zeros((2, 3)), TAUS)
    with pytest.raises(ValueError):
        qrf.predict_quantiles(X[:2], np.array([1.5]))
    with pytest.raises(ValueError):
        qrf.predict_quantiles(X[:2], np.array([np.nan]))
    with pytest.raises(ValueError):
        qrf.predict_cdf(X[:2], np.array([]))
    with pytest.raises(ValueError):
        qrf.predict_cdf(X[:2], np.array([np.nan]))
    with pytest.raises(ValueError):
        qrf.pit(X[:2], np.array([np.nan, 0.0]))


# --- Ported from the deleted wave-9 duplicate suite
# --- (tests/unit/models/test_quantile_forest.py) during the wave-9/10
# --- consolidation. Only assertions the canonical suite did not already make:
# --- median tracking, width-vs-sigma rank ordering, sample-size monotonicity,
# --- shape/seed reproducibility, and NaN designs. Not ported (deliberate
# --- canonical behaviour differences): the "levels must be strictly
# --- increasing" contract (canonical maps each tau independently) and the
# --- fixed n >= 30 fit floor (canonical uses n >= 2 * min_samples_leaf).


def _linear_homoskedastic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """y = 2x + N(0, 1) on x ~ U(-3, 3), one feature."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))
    return x, 2.0 * x[:, 0] + rng.normal(0.0, 1.0, size=n)


def test_predict_quantiles_median_tracks_conditional_median() -> None:
    x_tr, y_tr = _linear_homoskedastic(800, seed=1)
    qrf = QuantileRegressionForest(n_estimators=50, seed=20260927).fit(x_tr, y_tr)
    x_te = np.linspace(-3.0, 3.0, 200).reshape(-1, 1)
    med = qrf.predict_quantiles(x_te, TAUS)[:, 1]
    true_med = 2.0 * x_te[:, 0]
    corr = float(np.corrcoef(med, true_med)[0, 1])
    mae = float(np.mean(np.abs(med - true_med)))
    assert corr > 0.8, f"median correlation {corr:.3f}"
    assert mae < 0.6, f"median MAE {mae:.3f}"


def test_predict_quantiles_width_ranks_with_heteroskedastic_sigma() -> None:
    rng = np.random.default_rng(7)
    x_tr = rng.uniform(-2.0, 2.0, size=(800, 1))
    sigma = 0.3 + np.abs(x_tr[:, 0])
    y_tr = x_tr[:, 0] + sigma * rng.standard_normal(800)
    qrf = QuantileRegressionForest(n_estimators=50, seed=20260927).fit(x_tr, y_tr)
    x_te = np.linspace(-2.0, 2.0, 150).reshape(-1, 1)
    grid = qrf.predict_quantiles(x_te, np.array([0.1, 0.9]))
    width = grid[:, 1] - grid[:, 0]
    rho = float(spearmanr(width, 0.3 + np.abs(x_te[:, 0])).statistic)
    assert rho > 0.4, f"width-vs-sigma rank correlation {rho:.3f}"


def test_more_training_data_lowers_oos_median_pinball() -> None:
    small_losses: list[float] = []
    large_losses: list[float] = []
    for split in range(5):
        x_small, y_small = _linear_homoskedastic(120, seed=100 + split)
        x_large, y_large = _linear_homoskedastic(800, seed=100 + split)
        rng = np.random.default_rng(200 + split)
        x_te = rng.uniform(-3.0, 3.0, size=(150, 1))
        y_te = 2.0 * x_te[:, 0] + rng.normal(0.0, 1.0, size=150)
        for x_tr, y_tr, acc in (
            (x_small, y_small, small_losses),
            (x_large, y_large, large_losses),
        ):
            qrf = QuantileRegressionForest(n_estimators=30, seed=20260927 + split).fit(x_tr, y_tr)
            med = qrf.predict_quantiles(x_te, np.array([0.5]))[:, 0]
            acc.append(mean_pinball(y_te, med, 0.5))
    assert float(np.mean(large_losses)) < float(np.mean(small_losses))


def test_predict_quantiles_shape_and_seed_reproducibility() -> None:
    x_tr, y_tr = _linear_homoskedastic(300, seed=71)
    qrf = QuantileRegressionForest(n_estimators=50, seed=20260927).fit(x_tr, y_tr)
    x_te = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    first = qrf.predict_quantiles(x_te, TAUS)
    assert first.shape == (40, TAUS.size)
    np.testing.assert_array_equal(first, qrf.predict_quantiles(x_te, TAUS))
    refit = QuantileRegressionForest(n_estimators=50, seed=20260927).fit(x_tr, y_tr)
    np.testing.assert_array_equal(first, refit.predict_quantiles(x_te, TAUS))


def test_nan_design_raises_at_fit_and_predict() -> None:
    x_tr, y_tr = _linear_homoskedastic(120, seed=41)
    x_bad = x_tr.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        QuantileRegressionForest(n_estimators=30, seed=0).fit(x_bad, y_tr)
    qrf = QuantileRegressionForest(n_estimators=30, seed=0).fit(x_tr, y_tr)
    with pytest.raises(ValueError):
        qrf.predict_quantiles(np.array([[np.nan], [0.0]]), np.array([0.5]))


def test_min_samples_leaf_must_be_positive() -> None:
    with pytest.raises(ValueError):
        QuantileRegressionForest(min_samples_leaf=0)
