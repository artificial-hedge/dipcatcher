"""Tests for quant_fund.models.enbpi_multihorizon — multi-horizon EnbPI.

SYNTHETIC data only: every series below is seeded synthetic AR-type data; the
coverage assertions are correctness tests, never market evidence.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from quant_fund.models.enbpi import EnbPI
from quant_fund.models.enbpi_multihorizon import MultiHorizonEnbPI


def _factory() -> Ridge:
    return Ridge(alpha=1e-3)


def _ar1(n: int, seed: int, sigma: float = 1.0, phi: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def _lag_design(y: np.ndarray, n_lags: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Rows X[t] = [y_{t-1}, ..., y_{t-n_lags}] aligned with target series y[t:]."""
    n = y.size - n_lags
    cols = [y[n_lags - k - 1 : n_lags - k - 1 + n] for k in range(n_lags)]
    return np.column_stack(cols), y[n_lags:]


def _fit(
    n_train: int = 300,
    horizons: int = 3,
    seed: int = 0,
    *,
    sigma_train: float = 1.0,
    sigma_test: float = 1.0,
    n_test: int = 300,
    alpha: float = 0.1,
    aggregate: str = "mean",
    block_size: int = 1,
    n_estimators: int = 25,
    bonferroni_joint: bool = False,
) -> tuple[MultiHorizonEnbPI, np.ndarray, np.ndarray]:
    y_train = _ar1(n_train + 2, seed, sigma_train)
    y_test = _ar1(n_test + 2, seed + 500, sigma_test)
    X_tr, y_tr = _lag_design(y_train)
    X_te, y_te = _lag_design(y_test)
    model = MultiHorizonEnbPI(
        _factory,
        horizons=horizons,
        n_estimators=n_estimators,
        alpha=alpha,
        block_size=block_size,
        aggregate=aggregate,
        seed=seed,
        bonferroni_joint=bonferroni_joint,
    )
    model.fit(X_tr, y_tr)
    return model, X_te, y_te


# ------------------------------------------------------- EnbPI inheritance
def test_h1_residuals_and_bands_match_enbpi_exactly() -> None:
    """SYNTHETIC: H = 1 must reproduce quant_fund.models.enbpi.EnbPI bit-for-bit.

    MH-EnbPI trains on pairs (X[i], y[i+1]) — exactly EnbPI's data when EnbPI
    is handed the pre-shifted design — and draws bootstraps from the same
    seed with the same call order, so the LOO residuals and the fitted bands
    must agree exactly.
    """
    y = _ar1(302, seed=3)
    X, yy = _lag_design(y)
    mh = MultiHorizonEnbPI(_factory, horizons=1, n_estimators=15, alpha=0.1, seed=7)
    mh.fit(X, yy)
    ref = EnbPI(_factory, n_estimators=15, alpha=0.1, seed=7)
    ref.fit(X[:-1], yy[1:])
    np.testing.assert_array_equal(mh.residuals[0], ref.residuals)
    lo_mh, hi_mh, pt_mh = mh.predict_interval(X[-40:])
    lo_ref, hi_ref, pt_ref = ref.predict_interval(X[-40:])
    np.testing.assert_array_equal(lo_mh[:, 0], lo_ref)
    np.testing.assert_array_equal(hi_mh[:, 0], hi_ref)
    np.testing.assert_array_equal(pt_mh[:, 0], pt_ref)


# ------------------------------------------------------------------- shapes
def test_shapes_windows_and_point_forecasts() -> None:
    model, X_te, _ = _fit(n_train=200, horizons=3, seed=1)
    res_windows = model.residuals
    assert len(res_windows) == 3
    # Training design has 200 rows; horizon h drops h targets -> 199, 198, 197.
    assert [r.size for r in res_windows] == [199, 198, 197]
    lo, hi, pt = model.predict_interval(X_te[:12])
    assert lo.shape == hi.shape == pt.shape == (12, 3)
    assert np.all(lo <= pt) and np.all(pt <= hi)
    # residuals returns copies, not views into the live windows
    res_windows[0][0] = np.nan
    assert np.all(np.isfinite(model.residuals[0]))
    assert model.joint_coverage_bound_ == pytest.approx(0.7)


# ------------------------------------------- SYNTHETIC coverage under shift
def test_marginal_coverage_near_nominal_stationary_synthetic() -> None:
    """SYNTHETIC: stationary AR(1); per-horizon marginal coverage near 1-alpha."""
    covs = []
    for seed in range(4):
        model, X_te, y_te = _fit(seed=seed, n_test=300)
        res = model.predict_online(X_te, y_te)
        covs.append(res.marginal_coverage)
    mean_cov = np.mean(np.array(covs), axis=0)
    assert mean_cov.shape == (3,)
    assert np.all(mean_cov >= 0.84), mean_cov
    assert np.all(mean_cov <= 0.96), mean_cov
    # Bands widen with the horizon: further targets carry more innovation.
    assert np.all(np.diff(res.mean_width) > 0.0)


def test_marginal_coverage_tracks_variance_shift_synthetic() -> None:
    """SYNTHETIC: test-window noise inflated 1.7x; sliding windows must track.

    Correctness test only — the shift is a seeded synthetic scale change,
    not market evidence. The sliding per-horizon windows absorb the shift so
    coverage stays in a tolerance band instead of collapsing.
    """
    covs = []
    widths = []
    for seed in range(4):
        model, X_te, y_te = _fit(seed=seed, sigma_test=1.7, n_test=400)
        res = model.predict_online(X_te, y_te)
        covs.append(res.marginal_coverage)
        widths.append(res.mean_width.mean())
    mean_cov = np.mean(np.array(covs), axis=0)
    assert np.all(mean_cov >= 0.78), mean_cov
    assert np.all(mean_cov <= 0.97), mean_cov
    # A static band calibrated on sigma=1 would be far too narrow: the run
    # widths must reflect the inflated test noise.
    model_static, X_st, y_st = _fit(seed=0, sigma_test=1.0, n_test=400)
    static_res = model_static.predict_online(X_st, y_st)
    assert float(np.mean(widths)) > 1.3 * float(static_res.mean_width.mean())


def test_joint_coverage_and_bonferroni_accounting_synthetic() -> None:
    """SYNTHETIC: joint coverage of all H bands vs the Bonferroni union bound."""
    # Plain mode: per-horizon alpha, joint bound 1 - H*alpha.
    model, X_te, y_te = _fit(seed=2, alpha=0.1, horizons=3)
    res = model.predict_online(X_te, y_te)
    assert res.joint_coverage_bound == pytest.approx(0.7)
    assert 0.0 <= res.joint_coverage <= 1.0
    # Bonferroni mode: per-horizon alpha/H -> joint target 1 - alpha.
    joints, marginals = [], []
    for seed in range(4):
        model_b, Xb, yb = _fit(seed=seed, alpha=0.12, horizons=3, bonferroni_joint=True)
        assert model_b.alpha_per_horizon == pytest.approx(0.04)
        res_b = model_b.predict_online(Xb, yb)
        assert res_b.joint_coverage_bound == pytest.approx(0.88)
        joints.append(res_b.joint_coverage)
        marginals.append(res_b.marginal_coverage)
    mean_joint = float(np.mean(joints))
    mean_marg = np.mean(np.array(marginals), axis=0)
    # Realized joint coverage beats the worst-case bound (targets overlap)
    # and per-horizon marginals sit near 1 - alpha/H = 0.96.
    assert mean_joint >= 0.84, mean_joint
    assert mean_joint <= 0.995, mean_joint
    assert np.all(mean_marg >= 0.90), mean_marg


# ------------------------------------------------------------ determinism
def test_determinism_pinned_and_label_causality() -> None:
    """Same seed -> identical arrays; a changed final label cannot leak backwards."""
    model_a, X_te, y_te = _fit(seed=4, n_test=120)
    res_a = model_a.predict_online(X_te, y_te)
    model_b, _, _ = _fit(seed=4, n_test=120)
    res_b = model_b.predict_online(X_te, y_te)
    np.testing.assert_array_equal(res_a.lower, res_b.lower)
    np.testing.assert_array_equal(res_a.upper, res_b.upper)
    np.testing.assert_array_equal(res_a.point, res_b.point)
    np.testing.assert_array_equal(res_a.marginal_coverage, res_b.marginal_coverage)
    assert res_a.joint_coverage == res_b.joint_coverage
    # Causality, last label: corrupting y[T-1] leaves every issued interval
    # untouched (predictions precede label reveal); coverage can only weaken.
    model_c, _, _ = _fit(seed=4, n_test=120)
    y_corrupt = y_te.copy()
    y_corrupt[-1] += 1e6
    res_c = model_c.predict_online(X_te, y_corrupt)
    np.testing.assert_array_equal(res_a.lower, res_c.lower)
    np.testing.assert_array_equal(res_a.upper, res_c.upper)
    assert np.all(res_c.marginal_coverage <= res_a.marginal_coverage + 1e-12)
    assert res_c.joint_coverage <= res_a.joint_coverage + 1e-12
    # Causality, interior label: corrupting y[m] cannot change any interval
    # issued at origins < m, but must change the bands after the reveal.
    m = y_te.size // 2
    model_d, _, _ = _fit(seed=4, n_test=120)
    y_mid = y_te.copy()
    y_mid[m] += 50.0
    res_d = model_d.predict_online(X_te, y_mid)
    np.testing.assert_array_equal(res_a.lower[:m], res_d.lower[:m])
    np.testing.assert_array_equal(res_a.upper[:m], res_d.upper[:m])
    assert not np.allclose(res_a.lower, res_d.lower)


def test_median_aggregate_and_block_bootstrap_run() -> None:
    model, X_te, y_te = _fit(seed=5, horizons=2, aggregate="median", block_size=8, n_test=200)
    res = model.predict_online(X_te, y_te)
    assert np.all(res.marginal_coverage >= 0.78)
    assert np.all(res.marginal_coverage <= 0.99)
    assert np.all(res.mean_width > 0.0)


# --------------------------------------------------------- fail-closed edges
def test_fail_closed_constructor_edges() -> None:
    with pytest.raises(ValueError):
        MultiHorizonEnbPI(_factory, horizons=0)
    with pytest.raises(ValueError):
        MultiHorizonEnbPI(_factory, horizons=-1)
    with pytest.raises(ValueError):
        MultiHorizonEnbPI(_factory, horizons=2, n_estimators=1)
    with pytest.raises(ValueError):
        MultiHorizonEnbPI(_factory, horizons=2, alpha=0.0)
    with pytest.raises(ValueError):
        MultiHorizonEnbPI(_factory, horizons=2, alpha=1.0)
    with pytest.raises(ValueError):
        MultiHorizonEnbPI(_factory, horizons=2, block_size=0)
    with pytest.raises(ValueError):
        MultiHorizonEnbPI(_factory, horizons=2, aggregate="max")


def test_fail_closed_unfitted_and_fit_edges() -> None:
    m = MultiHorizonEnbPI(_factory, horizons=2, n_estimators=5)
    X2 = np.random.default_rng(0).normal(size=(4, 2))
    with pytest.raises(RuntimeError):
        m.predict_point(X2)
    with pytest.raises(RuntimeError):
        m.predict_interval(X2)
    with pytest.raises(RuntimeError):
        m.predict_online(X2, np.zeros(4))
    with pytest.raises(RuntimeError):
        _ = m.residuals
    # Too short: need H + max(10, ceil(1/alpha)+1) = 2 + 12 = 14 rows.
    with pytest.raises(ValueError):
        m.fit(X2[:10], np.zeros(10))
    with pytest.raises(ValueError):
        m.fit(X2[:, 0], np.zeros(4))  # 1-D design
    with pytest.raises(ValueError):
        m.fit(X2, np.zeros(3))  # length mismatch
    with pytest.raises(ValueError):
        m.fit(X2, np.r_[np.nan, np.zeros(3)])
    # Degenerate LOO: full-length circular blocks cover every point.
    X = np.random.default_rng(1).normal(size=(30, 2))
    y = X[:, 0]
    m2 = MultiHorizonEnbPI(_factory, horizons=2, n_estimators=2, block_size=29, alpha=0.2)
    with pytest.raises(ValueError):
        m2.fit(X, y)


def test_fail_closed_predict_edges() -> None:
    model, X_te, y_te = _fit(n_train=60, horizons=2, seed=6, n_test=40)
    with pytest.raises(ValueError):
        model.predict_point(np.zeros((3, X_te.shape[1] + 1)))
    with pytest.raises(ValueError):
        model.predict_point(np.full((3, X_te.shape[1]), np.nan))
    with pytest.raises(ValueError):
        model.predict_point(np.empty((0, X_te.shape[1])))
    with pytest.raises(ValueError):
        model.predict_online(X_te, y_te[:10])  # length mismatch
    with pytest.raises(ValueError):
        model.predict_online(X_te[:2], y_te[:2])  # T < H + 1
    with pytest.raises(ValueError):
        model.predict_online(X_te, np.r_[np.nan, y_te[1:]])
