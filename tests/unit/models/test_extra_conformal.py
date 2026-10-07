"""ExTRA generic tilt-reweighted conformal (arXiv:2609.30886) -- SYNTHETIC tests.

Seeded correctness material only, never market evidence: exact reductions
(theta=0 bitwise-equals unweighted split conformal; degenerate one-sided
tilt bitwise-equals the surviving-subset quantile), weight normalization /
stabilization / quantile clip, the generic marginal-matching fit's recovery
of a planted Gaussian density-ratio coefficient, weighted p-values and the
retention boundary, ESS fail-closed fallback and raise modes, the ESS /
width path along a tilt direction, the harm regime (misspecified label
direction hurts coverage on well-posed weights), determinism, class API,
and fail-closed edges. No Sharpe.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile, cqr_scores, expand_interval
from quant_fund.models.extra_conformal import (
    DEFAULT_ESS_FLOOR,
    ExTRASplitCQR,
    bench_extra_conformal,
    bench_extra_harm,
    extra_conformal_quantile,
    extra_prediction_sets,
    fit_marginal_tilt,
    log_tilt_weights,
    predictive_tilt_scores,
    sign_joint_features,
    synthetic_gaussian_x_shift,
    tilt_diagnostics,
    tilt_path_diagnostics,
    tilt_weights,
    tilted_pvalues,
)
from quant_fund.models.extra_tilt_conformal import synthetic_bimodal_shift
from quant_fund.models.localized_conformal import effective_sample_size
from quant_fund.models.weighted_conformal import weighted_conformal_quantile


def _scores(n: int = 300, seed: int = 4) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).normal(size=n), dtype=float)


# ---------------------------------------------------------------------------
# Weight estimator: normalization, stabilization, clipping, exact reductions
# ---------------------------------------------------------------------------


def test_tilt_weights_theta_zero_returns_exact_ones() -> None:
    feats = np.column_stack([np.linspace(-1, 1, 50), np.ones(50)])
    w = tilt_weights(feats, np.zeros(2))
    assert np.array_equal(w, np.ones(50))


def test_tilt_weights_mean_one_normalized() -> None:
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(200, 3))
    w = tilt_weights(feats, np.array([0.3, -0.7, 1.1]))
    assert w.shape == (200,)
    assert np.all(np.isfinite(w)) and np.all(w >= 0.0)
    assert float(np.mean(w)) == pytest.approx(1.0, abs=1e-12)


def test_tilt_weights_matches_manual_exp() -> None:
    feats = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    theta = np.array([np.log(2.0), np.log(3.0)])
    w = tilt_weights(feats, theta)
    raw = np.array([2.0, 3.0, 6.0])
    assert np.allclose(w, raw / raw.mean())


def test_tilt_weights_extreme_theta_no_nan() -> None:
    feats = np.column_stack([np.linspace(-1, 1, 100)])
    w = tilt_weights(feats, np.array([900.0]))
    assert np.all(np.isfinite(w))
    assert np.any(w == 0.0)  # underflowed rows drop out exactly
    assert float(np.mean(w)) == pytest.approx(1.0)


def test_tilt_weights_clip_quantile_bounds_max_and_lifts_ess() -> None:
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(500, 1))
    theta = np.array([3.0])
    w_plain = tilt_weights(feats, theta)
    w_clip = tilt_weights(feats, theta, clip_quantile=0.9)
    cap = float(np.quantile(w_plain, 0.9))
    # post-renormalization cap; the pre-renorm clip is at the 0.9 quantile.
    assert float(w_clip.max()) <= cap / float(np.mean(np.minimum(w_plain, cap))) + 1e-9
    assert float(w_clip.max()) < float(w_plain.max())
    assert effective_sample_size(w_clip) > effective_sample_size(w_plain)
    assert float(np.mean(w_clip)) == pytest.approx(1.0)


def test_log_tilt_weights_linear_form() -> None:
    feats = np.array([[2.0, -1.0], [0.5, 3.0]])
    theta = np.array([0.25, -0.5])
    assert np.allclose(log_tilt_weights(feats, theta), feats @ theta)


def test_tilt_weights_rejects_bad_inputs() -> None:
    feats = np.ones((10, 2))
    with pytest.raises(ValueError):
        tilt_weights(np.ones(0).reshape(0, 2), np.ones(2))
    with pytest.raises(ValueError):
        tilt_weights(feats, np.ones(3))
    with pytest.raises(ValueError):
        tilt_weights(feats, np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        tilt_weights(feats, np.ones(2), clip_quantile=0.0)
    with pytest.raises(ValueError):
        tilt_weights(feats, np.ones(2), clip_quantile=1.5)
    bad = np.ones((10, 2))
    bad[3, 1] = np.inf
    with pytest.raises(ValueError):
        tilt_weights(bad, np.ones(2))


def test_sign_joint_features_layout() -> None:
    x = np.array([[1.0, -2.0], [0.5, 0.25]])
    y = np.array([3.0, -0.5])
    g = sign_joint_features(x, y)
    assert g.shape == (2, 3)
    assert np.allclose(g[:, :2], x)
    assert np.allclose(g[:, 2], [1.0, -1.0])


def test_sign_joint_features_zero_maps_positive() -> None:
    g = sign_joint_features(np.array([[0.0]]), np.array([0.0]))
    assert g[0, 1] == 1.0


def test_sign_joint_features_rejects_mismatch() -> None:
    with pytest.raises(ValueError):
        sign_joint_features(np.ones((4, 2)), np.ones(3))


# ---------------------------------------------------------------------------
# Exact reductions
# ---------------------------------------------------------------------------


def test_theta_zero_quantile_bitwise_unweighted() -> None:
    s = _scores()
    feats = np.column_stack([s, np.sign(s)])
    w = tilt_weights(feats, np.zeros(2))
    res = extra_conformal_quantile(s, w, 0.10)
    assert res.qhat == conformal_quantile(s, 0.10)
    assert res.qhat == weighted_conformal_quantile(s, np.ones_like(s), 0.10)
    assert res.weights_mode == "tilt"


def test_theta_zero_pvalues_bitwise_unweighted() -> None:
    s = _scores()
    q = np.linspace(-2.0, 2.0, 41)
    p_tilt = tilted_pvalues(s, tilt_weights(np.ones((s.size, 1)), np.zeros(1)), q)
    counts = np.sum(s[None, :] >= q[:, None], axis=1)
    p_unw = (1.0 + counts) / (s.size + 1.0)
    assert np.array_equal(p_tilt, p_unw)


def test_degenerate_negative_tilt_is_subset_quantile() -> None:
    y = _scores(200, 9)
    mask = y > 0.5
    feats = np.column_stack([mask.astype(float)])
    w = tilt_weights(feats, np.array([-800.0]))  # exp underflows to exactly 0
    assert np.all(w[mask] == 0.0)
    res = extra_conformal_quantile(y, w, 0.10)
    assert res.qhat == conformal_quantile(y[~mask], 0.10)


def test_degenerate_positive_tilt_is_subset_quantile() -> None:
    y = _scores(200, 11)
    mask = y < -0.5
    feats = np.column_stack([mask.astype(float)])
    w = tilt_weights(feats, np.array([-800.0]))
    res = extra_conformal_quantile(y, w, 0.10)
    assert res.qhat == conformal_quantile(y[~mask], 0.10)


def test_uniform_any_constant_weights_reduce() -> None:
    s = _scores(120, 3)
    q0 = conformal_quantile(s, 0.05)
    for c in (1.0, 2.5, 1e-4):
        res = extra_conformal_quantile(s, np.full(s.size, c), 0.05, ess_floor=0.0)
        assert res.qhat == q0


# ---------------------------------------------------------------------------
# Diagnostics + ESS guard
# ---------------------------------------------------------------------------


def test_tilt_diagnostics_values() -> None:
    w = np.array([3.0, 1.0, 1.0, 1.0])
    d = tilt_diagnostics(w)
    assert d.n == 4
    assert d.ess == pytest.approx(effective_sample_size(w))
    assert d.ess_fraction == pytest.approx(d.ess / 4.0)
    assert d.max_weight_share == pytest.approx(0.5)


def test_tilt_diagnostics_rejects_degenerate() -> None:
    with pytest.raises(ValueError):
        tilt_diagnostics(np.zeros(5))
    with pytest.raises(ValueError):
        tilt_diagnostics(np.array([1.0, -1.0, 1.0]))


def test_ess_collapse_falls_back_to_unweighted() -> None:
    s = _scores(200, 8)
    w = np.full(200, 1e-12)
    w[0], w[1] = 1.0, 0.5
    res = extra_conformal_quantile(s, w, 0.10, ess_floor=0.05)
    assert res.weights_mode == "uniform_fallback"
    assert res.qhat == conformal_quantile(s, 0.10)


def test_ess_collapse_raise_mode() -> None:
    s = _scores(100, 1)
    w = np.full(100, 1e-12)
    w[0] = 1.0
    with pytest.raises(ValueError, match="ESS"):
        extra_conformal_quantile(s, w, 0.10, ess_floor=0.05, on_collapse="raise")


def test_ess_floor_zero_disables_guard() -> None:
    s = _scores(100, 6)
    w = np.full(100, 1e-12)
    w[0] = 1.0
    res = extra_conformal_quantile(s, w, 0.10, ess_floor=0.0)
    assert res.weights_mode == "tilt"
    assert res.qhat == weighted_conformal_quantile(s, w, 0.10)


def test_extra_conformal_quantile_fail_closed() -> None:
    s = _scores(50)
    with pytest.raises(ValueError):
        extra_conformal_quantile(s, np.ones(49), 0.10)
    with pytest.raises(ValueError):
        extra_conformal_quantile(s, np.ones(50), 1.5)
    with pytest.raises(ValueError):
        extra_conformal_quantile(s, np.ones(50), 0.10, ess_floor=-0.1)
    with pytest.raises(ValueError):
        extra_conformal_quantile(s, np.ones(50), 0.10, on_collapse="bogus")
    with pytest.raises(ValueError):
        extra_conformal_quantile(s, np.zeros(50), 0.10)


# ---------------------------------------------------------------------------
# Weighted p-values
# ---------------------------------------------------------------------------


def test_tilted_pvalues_hand_example() -> None:
    s = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 4.0, 2.0])
    # total = 8; for q=2.5, scores >= 2.5 are s2, s3 -> mass 6.
    p = tilted_pvalues(s, w, np.array([2.5]))
    assert p[0] == pytest.approx((1.0 + 6.0) / 9.0)


def test_tilted_pvalues_bounded_monotone() -> None:
    s = _scores(250, 5)
    rng = np.random.default_rng(5)
    w = np.exp(rng.normal(size=s.size))
    q = np.linspace(s.min() - 1.0, s.max() + 1.0, 80)
    p = tilted_pvalues(s, w, q)
    assert np.all(p > 0.0) and np.all(p <= 1.0)
    assert np.all(np.diff(p) <= 1e-12)


def test_tilted_pvalues_retention_matches_qhat() -> None:
    s = _scores(300, 12)  # a.s. distinct scores
    rng = np.random.default_rng(12)
    w = np.exp(0.8 * rng.normal(size=s.size))
    alpha = 0.10
    res = extra_conformal_quantile(s, w, alpha, ess_floor=0.0)
    p = tilted_pvalues(s, w, s)
    assert np.array_equal(p > alpha, s <= res.qhat)


def test_tilted_pvalues_test_weight_shifts_denominator() -> None:
    s = np.array([0.0, 1.0])
    w = np.ones(2)
    p1 = tilted_pvalues(s, w, np.array([0.5]), test_weight=1.0)
    p4 = tilted_pvalues(s, w, np.array([0.5]), test_weight=4.0)
    assert p1[0] == pytest.approx(2.0 / 3.0)
    assert p4[0] == pytest.approx(5.0 / 6.0)


def test_tilted_pvalues_fail_closed() -> None:
    s = _scores(20)
    with pytest.raises(ValueError):
        tilted_pvalues(s, np.ones(19), np.array([0.0]))
    with pytest.raises(ValueError):
        tilted_pvalues(s, np.ones(20), np.array([np.nan]))
    with pytest.raises(ValueError):
        tilted_pvalues(s, np.ones(20), np.array([0.0]), test_weight=0.0)
    with pytest.raises(ValueError):
        tilted_pvalues(s, np.zeros(20), np.array([0.0]))


# ---------------------------------------------------------------------------
# Marginal-matching tilt fit (generic paper Eq. 6 analog)
# ---------------------------------------------------------------------------


def test_fit_marginal_tilt_recovers_gaussian_shift() -> None:
    xs, xt = synthetic_gaussian_x_shift(4000, 4000, 1.0, 7)
    fit = fit_marginal_tilt(xs[:, None], xt[:, None])
    assert fit.converged
    assert fit.theta[0] == pytest.approx(1.0, abs=0.1)
    assert fit.ess_source > 0.0 and fit.ess_source <= 4000.0


def test_fit_marginal_tilt_recovers_two_dimensional_shift() -> None:
    rng = np.random.default_rng(21)
    xs = rng.normal(0.0, 1.0, (8000, 2))
    xt = rng.normal(0.0, 1.0, (8000, 2)) + np.array([0.8, -0.5])
    fit = fit_marginal_tilt(xs, xt)
    assert np.allclose(fit.theta, [0.8, -0.5], atol=0.1)


def test_fit_marginal_tilt_no_shift_near_zero() -> None:
    rng = np.random.default_rng(13)
    xs = rng.normal(0.0, 1.0, (4000, 1))
    xt = rng.normal(0.0, 1.0, (4000, 1))
    fit = fit_marginal_tilt(xs, xt)
    assert abs(fit.theta[0]) < 0.1


def test_fit_marginal_tilt_deterministic() -> None:
    xs, xt = synthetic_gaussian_x_shift(500, 500, 0.7, 3)
    f1 = fit_marginal_tilt(xs[:, None], xt[:, None])
    f2 = fit_marginal_tilt(xs[:, None], xt[:, None])
    assert np.array_equal(f1.theta, f2.theta)
    assert f1.objective == f2.objective


def test_fit_marginal_tilt_fail_closed() -> None:
    xs, xt = synthetic_gaussian_x_shift(50, 50, 0.5, 1)
    with pytest.raises(ValueError):
        fit_marginal_tilt(xs[:, None], xt[:, None], l2=-1.0)
    with pytest.raises(ValueError):
        fit_marginal_tilt(xs[:, None], xt[:, None], bound=0.0)
    with pytest.raises(ValueError):
        fit_marginal_tilt(xs[:, None], np.ones((50, 2)))
    with pytest.raises(ValueError):
        fit_marginal_tilt(xs[:, None], xt[:, None], theta0=np.array([99.0]))
    with pytest.raises(ValueError):
        fit_marginal_tilt(xs[:, None], xt[:, None], theta0=np.ones(3))
    with pytest.raises(ValueError):
        fit_marginal_tilt(xs[:, None], xt[:, None], maxiter=0)
    with pytest.raises(ValueError):
        synthetic_gaussian_x_shift(0, 10, 1.0, 0)


# ---------------------------------------------------------------------------
# Predictive tilt step + interval constructor
# ---------------------------------------------------------------------------


def test_predictive_tilt_scores_exact_shift() -> None:
    base = np.array([0.5, 1.0, 1.5])
    log_h = np.log(np.array([2.0, 1.0, 0.5]))
    log_z = np.log(3.0)
    out = predictive_tilt_scores(base, log_h, log_z)
    assert np.allclose(out, base - log_h + log_z)


def test_predictive_tilt_scores_zero_tilt_identity() -> None:
    base = _scores(60, 2)
    out = predictive_tilt_scores(base, np.zeros(60))
    assert np.array_equal(out, base)


def test_predictive_tilt_scores_fail_closed() -> None:
    base = _scores(10)
    with pytest.raises(ValueError):
        predictive_tilt_scores(base, np.zeros(9))
    with pytest.raises(ValueError):
        predictive_tilt_scores(base, np.zeros(10), np.full(5, 1.0))


def test_extra_prediction_sets_expand_consistently() -> None:
    rng = np.random.default_rng(17)
    s = np.abs(rng.normal(size=300))
    w = np.exp(0.5 * rng.normal(size=300))
    lo = np.zeros(40)
    hi = np.zeros(40)
    elo, ehi, res = extra_prediction_sets(lo, hi, s, w, 0.10)
    exp_lo, exp_hi = expand_interval(lo, hi, res.qhat)
    assert np.array_equal(elo, exp_lo) and np.array_equal(ehi, exp_hi)
    assert np.all(ehi >= elo)


# ---------------------------------------------------------------------------
# Planted tilt coverage + harm regime (SYNTHETIC)
# ---------------------------------------------------------------------------


def test_bench_correct_direction_beats_unweighted() -> None:
    row = bench_extra_conformal(seed=0)
    assert row["synthetic_coverage"] > row["synthetic_unweighted_coverage"]
    assert row["synthetic_coverage_error"] < row["synthetic_unweighted_coverage_error"]
    assert row["synthetic_coverage"] >= 0.85  # near the 0.90 nominal level
    assert row["synthetic_weights_mode"] == "tilt"
    assert row["synthetic_claim"] == "research_metric_only"
    assert "SYNTHETIC" in str(row["synthetic_dgp"])
    assert all("sharpe" not in str(k).lower() for k in row)


def test_bench_harm_regime_worse_than_unweighted_on_healthy_weights() -> None:
    row = bench_extra_harm(seed=0)
    # Misspecified label direction: high ESS (no collapse, no fallback) yet
    # coverage is worse than doing nothing -- the documented harm regime.
    assert row["synthetic_weights_mode"] == "tilt"
    assert row["synthetic_ess_fraction"] > 0.2
    assert row["synthetic_coverage"] < row["synthetic_unweighted_coverage"]
    assert row["synthetic_harm_coverage_gap"] > 0.1
    assert row["synthetic_harm_mode"] == "misspecified_label_tilt"


def test_bench_deterministic() -> None:
    r1 = bench_extra_conformal(seed=5)
    r2 = bench_extra_conformal(seed=5)
    assert r1 == r2
    h1 = bench_extra_harm(seed=5)
    h2 = bench_extra_harm(seed=5)
    assert h1 == h2


def test_tilt_path_ess_nonincreasing_and_qhat_monotone() -> None:
    rng = np.random.default_rng(30)
    y = rng.normal(size=400)
    feats = np.column_stack([np.sign(y)])
    scales = np.linspace(0.0, 4.0, 9)
    path = tilt_path_diagnostics(np.abs(y), feats, np.array([1.0]), scales, 0.10)
    assert np.all(np.diff(path.ess) <= 1e-9)
    # Upweighting the high-|y| side pushes qhat monotonically upward.
    assert np.all(np.diff(path.qhat) >= -1e-12)
    assert path.modes[0] == "tilt"


def test_tilt_path_reaches_fallback_at_extreme_tilt() -> None:
    rng = np.random.default_rng(31)
    y = rng.normal(size=200)
    feats = rng.normal(size=(200, 1))
    path = tilt_path_diagnostics(y, feats, np.array([1.0]), np.linspace(0.0, 60.0, 7), 0.10)
    assert "uniform_fallback" in path.modes
    assert path.ess[-1] < 0.05 * 200


# ---------------------------------------------------------------------------
# Class API + end-to-end determinism
# ---------------------------------------------------------------------------


def _bimodal_xy(seed: int = 3):
    data = synthetic_bimodal_shift(eta=1.0, a_star=1.0, b_star=1.2, seed=seed)
    m = float(np.median(data.y_train))
    return data, m


def test_class_calibrate_predict_sets() -> None:
    data, m = _bimodal_xy()
    fmap = lambda x, y: np.column_stack([x[:, 0], np.sign(y)])  # noqa: E731
    est = ExTRASplitCQR(0.10, theta=np.array([1.0, 1.2]), feature_map=fmap)
    est.calibrate(
        data.y_cal,
        np.full(data.y_cal.size, m),
        np.full(data.y_cal.size, m),
        x_cal=data.x_cal,
    )
    lo, hi = est.predict_sets(np.full(data.y_test.size, m), np.full(data.y_test.size, m))
    assert lo.shape == hi.shape == (data.y_test.size,)
    cov = float(np.mean((data.y_test >= lo) & (data.y_test <= hi)))
    assert cov > 0.85
    assert est.result_ is not None and est.result_.weights_mode == "tilt"
    assert est.diagnostics_ is not None and 0.0 < est.diagnostics_.ess_fraction <= 1.0


def test_class_theta_requires_feature_map() -> None:
    with pytest.raises(ValueError):
        ExTRASplitCQR(0.10, theta=np.array([1.0]))


def test_class_uniform_path_matches_split_cqr() -> None:
    rng = np.random.default_rng(40)
    y = rng.normal(size=200)
    lo = np.zeros(200)
    hi = np.zeros(200)
    est = ExTRASplitCQR(0.10)
    est.calibrate(y, lo, hi)
    assert est.qhat == conformal_quantile(cqr_scores(y, lo, hi), 0.10)
    elo, ehi = est.predict_sets(np.zeros(10), np.zeros(10))
    assert np.allclose(elo, -est.qhat) and np.allclose(ehi, est.qhat)


def test_class_explicit_weights_and_metadata() -> None:
    rng = np.random.default_rng(41)
    y = rng.normal(size=150)
    w = np.exp(0.3 * rng.normal(size=150))
    est = ExTRASplitCQR(0.10)
    est.calibrate(y, np.zeros(150), np.zeros(150), weights=w)
    meta = est.metadata()
    assert meta.name == "extra_split_cqr"
    assert meta.extra["weights_mode"] == "tilt"
    assert est.result_ is not None and meta.extra["ess"] == est.result_.ess


def test_class_predict_before_calibrate_raises() -> None:
    est = ExTRASplitCQR(0.10)
    with pytest.raises(ValueError):
        est.predict_sets(np.zeros(3), np.zeros(3))


def test_class_fail_closed_constructor() -> None:
    with pytest.raises(ValueError):
        ExTRASplitCQR(1.5)
    with pytest.raises(ValueError):
        ExTRASplitCQR(0.10, ess_floor=2.0)
    with pytest.raises(ValueError):
        ExTRASplitCQR(0.10, on_collapse="bogus")


def test_class_theta_requires_x_cal_at_calibrate() -> None:
    fmap = lambda x, y: np.column_stack([x[:, 0], np.sign(y)])  # noqa: E731
    est = ExTRASplitCQR(0.10, theta=np.array([1.0, 1.2]), feature_map=fmap)
    with pytest.raises(ValueError):
        est.calibrate(np.ones(20), np.zeros(20), np.zeros(20))


def test_end_to_end_determinism_bitwise() -> None:
    def run() -> float:
        data, m = _bimodal_xy(9)
        feats = np.column_stack([data.x_cal[:, 0], np.sign(data.y_cal)])
        w = tilt_weights(feats, np.array([1.0, 1.2]))
        s = cqr_scores(data.y_cal, np.full(data.y_cal.size, m), np.full(data.y_cal.size, m))
        res = extra_conformal_quantile(s, w, 0.10, ess_floor=DEFAULT_ESS_FLOOR)
        return res.qhat

    assert run() == run()
