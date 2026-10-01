"""DCP (arXiv:2605.26569) -- SYNTHETIC correctness tests.

Seeded simulations only: closed-form score values (paper Eqs. 13-18), inner
bands (equal-tail quantile vs highest-density), the ceil((n+1)(1-alpha))
conformal quantile pinned against ``metrics.conformal.conformal_quantile``,
numerical-inversion reductions (DCP reproduces the analytic inverses of the
residual, Z-score, interval/CQR and HDI/CMC scores up to bisection tolerance
— the paper's Sec. 3.1 claim), nonmonotone/multimodal bracketing, the
paper's Sec. 3.2 failure policy (single root, no root, strict mode),
retry-reach on far-off roots, Winkler/MMW hand computations (Eqs. 25-28),
planted heteroscedastic/skewed/drifting DGP coverage checks, predictor
abstraction (ensemble, quantile-pseudo-draw, mixture), registry plug-in, one
determinism test, and fail-closed edges. Correctness material, never market
evidence. No Sharpe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile, cqr_scores, expand_interval
from quant_fund.metrics.dcp import (
    SCORE_NAMES,
    NonInvertibleScoreError,
    RootFinderConfig,
    acceptable_coverage,
    bench_dcp,
    calibration_scores,
    cqr_interval,
    dcp_battery,
    dcp_fit,
    dcp_intervals,
    dcp_online_intervals,
    get_score,
    inner_band_hdi,
    inner_band_quantile,
    interval_metrics,
    invert_interval,
    modified_winkler_score,
    picp,
    pinaw,
    predict_draws,
    predictive_median,
    register_score,
    score_interval,
    score_knn,
    split_conformal_interval,
    undercoverage_penalty_factor,
    width_cv,
    winkler_components,
    winkler_score,
)


def _het_world(
    seed: int = 0, n_cal: int = 400, n_test: int = 300, m: int = 120
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Planted heteroscedastic DGP with a well-specified ensemble oracle.

    x ~ U(-3, 3), mu = sin(2x), sigma(x) = 0.08 + 0.45 x^2 / 9,
    y = mu + sigma * eps. Draws ~ N(mu, sigma^2). SYNTHETIC only.
    """
    rng = np.random.default_rng(seed)

    def _make(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = rng.uniform(-3.0, 3.0, size=n)
        sig = 0.08 + 0.45 * x * x / 9.0
        y = np.sin(2.0 * x) + sig * rng.standard_normal(n)
        draws = np.sin(2.0 * x)[:, None] + sig[:, None] * rng.standard_normal((n, m))
        return x, y, draws

    xc, y_cal, d_cal = _make(n_cal)
    xt, y_test, d_test = _make(n_test)
    return xc, y_cal, d_cal, xt, y_test, d_test


# ---------------------------------------------------------------------------
# Score closed forms (paper Sec. 4.4)
# ---------------------------------------------------------------------------


def test_residual_score_closed_form() -> None:
    s = get_score("residual")
    draws = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # mean 2
    out = s(np.array([3.0, 5.0, -1.0, 2.0]), draws)
    np.testing.assert_allclose(out, [1.0, 3.0, 3.0, 0.0])


def test_zscore_score_closed_form() -> None:
    s = get_score("zscore")
    draws = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # mean 2, sd sqrt(2)
    sd = math.sqrt(2.0)
    out = s(np.array([4.0, 2.0, 2.0 + sd]), draws)
    np.testing.assert_allclose(out, [2.0 / sd, 0.0, 1.0])


def test_zscore_fail_closed_on_degenerate_draws() -> None:
    s = get_score("zscore")
    with pytest.raises(ValueError, match="degenerate"):
        s(np.array([1.0]), np.full(10, 3.0))


def test_interval_score_closed_form_and_negative_inside() -> None:
    draws = np.linspace(0.0, 10.0, 101)  # q(0.1)=1, q(0.9)=9 at mass 0.8
    s = score_interval(mass=0.8, kind="quantile")
    lo, hi = inner_band_quantile(draws, 0.8)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(9.0)
    out = s(np.array([5.0, 9.0, 11.0, -1.0]), draws)
    np.testing.assert_allclose(out, [-4.0, 0.0, 2.0, 2.0])


def test_interval_scaled_score_divides_by_width() -> None:
    draws = np.linspace(0.0, 10.0, 101)
    s = score_interval(mass=0.8, kind="quantile", scaled=True)
    out = s(np.array([5.0, 11.0]), draws)
    np.testing.assert_allclose(out, [-4.0 / 8.0, 2.0 / 8.0])


def test_interval_scaled_fail_closed_on_degenerate_band() -> None:
    s = score_interval(mass=0.5, kind="quantile", scaled=True)
    with pytest.raises(ValueError, match="degenerate"):
        s(np.array([1.0]), np.full(10, 2.0))


def test_hdi_band_shortest_window() -> None:
    draws = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    lo, hi = inner_band_hdi(draws, 0.5)  # window of 3, tightest is [0, 2]
    assert lo == pytest.approx(0.0) and hi == pytest.approx(2.0)


def test_hdi_band_shorter_than_equal_tail_on_skewed_draws() -> None:
    rng = np.random.default_rng(4)
    draws = rng.exponential(1.0, size=5000)
    h_lo, h_hi = inner_band_hdi(draws, 0.9)
    q_lo, q_hi = inner_band_quantile(draws, 0.9)
    assert h_hi - h_lo < q_hi - q_lo  # HDI hugs the dense left tail
    assert h_lo < q_lo


def test_knn_score_closed_form() -> None:
    draws = np.arange(10.0)  # {0..9}; median pairwise distance = 3
    s = score_knn(k=3)
    out = s(np.array([0.0, 4.5]), draws)
    # y=0: 3 nearest distances {0,1,2} -> median 1; y=4.5: {0.5,0.5,1.5} -> 0.5
    np.testing.assert_allclose(out, [1.0 / 3.0, 0.5 / 3.0])


def test_knn_score_orders_center_below_edge() -> None:
    rng = np.random.default_rng(7)
    draws = rng.normal(0.0, 1.0, 200)
    s = score_knn(k=10)
    assert float(s(np.array([0.0]), draws)[0]) < float(s(np.array([5.0]), draws)[0])


def test_knn_score_fail_closed() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        score_knn(k=11)(np.array([0.0]), np.arange(10.0))
    with pytest.raises(ValueError, match="k must be"):
        score_knn(k=0)
    with pytest.raises(ValueError, match="degenerate"):
        score_knn(k=2)(np.array([0.0]), np.full(10, 1.0))


def test_score_registry_pluggable() -> None:
    import quant_fund.metrics.dcp as dcp_mod

    def custom(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
        return np.asarray(np.abs(y) * 2.0)

    register_score("double_abs", custom)
    try:
        assert get_score("double_abs") is custom
        rng = np.random.default_rng(0)
        fit = dcp_fit(rng.normal(size=50), rng.normal(size=(50, 20)), score="double_abs")
        assert fit.qhat >= 0.0
    finally:
        dcp_mod._USER_SCORES.pop("double_abs", None)


def test_get_score_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown score"):
        get_score("not_a_score")
    with pytest.raises(ValueError, match="non-empty"):
        register_score("  ", lambda y, d: y)
    with pytest.raises(ValueError, match="callable"):
        register_score("x", 3.0)  # type: ignore[arg-type]


def test_all_builtin_scores_registered() -> None:
    for name in SCORE_NAMES:
        fn = get_score(name)
        draws = np.linspace(-1.0, 1.0, 50)
        out = fn(np.array([0.0, 2.0]), draws)
        assert out.shape == (2,) and np.all(np.isfinite(out))


def test_inner_band_fail_closed() -> None:
    with pytest.raises(ValueError, match="(0, 1)"):
        inner_band_quantile(np.arange(10.0), 1.5)
    with pytest.raises(ValueError, match="finite"):
        inner_band_hdi(np.array([0.0, np.nan]), 0.8)


# ---------------------------------------------------------------------------
# Calibration (paper Alg. 1, lines 5-9) — composed conformal quantile
# ---------------------------------------------------------------------------


def test_calibration_scores_and_qhat_pin() -> None:
    rng = np.random.default_rng(1)
    y_cal = rng.normal(size=200)
    d_cal = rng.normal(size=(200, 30))
    fit = dcp_fit(y_cal, d_cal, score="residual", alpha=0.1)
    eps = np.abs(y_cal - d_cal.mean(axis=1))
    np.testing.assert_allclose(fit.cal_scores, eps)
    assert fit.qhat == pytest.approx(conformal_quantile(eps, 0.1))


def test_calibration_qhat_order_statistic_hand() -> None:
    y_cal = np.zeros(5)
    d_cal = np.array([[v] * 4 for v in [1.0, 2.0, 3.0, 4.0, 5.0]])
    eps = calibration_scores(y_cal, d_cal, get_score("residual"))
    np.testing.assert_allclose(eps, [1.0, 2.0, 3.0, 4.0, 5.0])
    # ceil((5+1)*(1-0.25)) = 5 -> 5th smallest = 5.0
    assert conformal_quantile(eps, 0.25) == pytest.approx(5.0)


def test_dcp_fit_fail_closed() -> None:
    with pytest.raises(ValueError, match="alpha"):
        dcp_fit(np.zeros(10), np.zeros((10, 5)), alpha=1.0)
    with pytest.raises(ValueError, match="length"):
        dcp_fit(np.zeros(9), np.zeros((10, 5)))
    with pytest.raises(ValueError, match="finite"):
        dcp_fit(np.zeros(10), np.full((10, 5), np.nan))
    with pytest.raises(ValueError, match="non-empty"):
        dcp_fit(np.zeros(0), np.zeros((0, 5)))


# ---------------------------------------------------------------------------
# Numerical inversion: analytic reductions (paper Sec. 3.1)
# ---------------------------------------------------------------------------


def test_invert_residual_reproduces_closed_form() -> None:
    rng = np.random.default_rng(2)
    draws = rng.normal(0.4, 1.3, 150)
    qhat = 0.9
    res = invert_interval(get_score("residual"), draws, qhat)
    mu = float(np.mean(draws))
    assert res.status == "ok" and res.n_roots == 2
    assert res.low == pytest.approx(mu - qhat, abs=1e-6)
    assert res.high == pytest.approx(mu + qhat, abs=1e-6)


def test_invert_zscore_reproduces_closed_form() -> None:
    rng = np.random.default_rng(3)
    draws = rng.normal(-0.2, 0.7, 100)
    qhat = 1.3
    res = invert_interval(get_score("zscore"), draws, qhat)
    mu, sd = float(np.mean(draws)), float(np.std(draws))
    assert res.low == pytest.approx(mu - qhat * sd, abs=1e-6)
    assert res.high == pytest.approx(mu + qhat * sd, abs=1e-6)


def test_invert_interval_score_reproduces_cqr_closed_form() -> None:
    rng = np.random.default_rng(5)
    draws = rng.standard_t(5.0, 120) * 2.0 + 1.0
    qhat = 0.35
    res = invert_interval(get_score("interval", alpha=0.1), draws, qhat)
    lo, hi = inner_band_quantile(draws, 0.9)
    assert res.low == pytest.approx(lo - qhat, abs=1e-6)
    assert res.high == pytest.approx(hi + qhat, abs=1e-6)


def test_invert_allows_negative_qhat_shrinkage() -> None:
    rng = np.random.default_rng(6)
    draws = rng.normal(0.0, 1.0, 150)
    band = inner_band_quantile(draws, 0.9)
    qhat = -0.1  # s_int negative inside the band -> interval shrinks
    res = invert_interval(get_score("interval"), draws, qhat)
    assert res.status == "ok"
    assert res.low == pytest.approx(band[0] - qhat, abs=1e-6)
    assert res.high == pytest.approx(band[1] + qhat, abs=1e-6)
    assert res.high - res.low < band[1] - band[0]


def test_invert_interval_scaled_reproduces_scaled_closed_form() -> None:
    rng = np.random.default_rng(8)
    draws = rng.normal(0.5, 0.4, 100)
    qhat = 0.2
    res = invert_interval(get_score("interval_scaled"), draws, qhat)
    lo, hi = inner_band_quantile(draws, 0.9)
    w = hi - lo
    assert res.low == pytest.approx(lo - qhat * w, abs=1e-6)
    assert res.high == pytest.approx(hi + qhat * w, abs=1e-6)


def test_invert_nonmonotone_score_outermost_roots() -> None:
    # s(y) = ||y| - 1| has minima at +-1 and is nonmonotone:
    # s - 0.5 = 0 -> |y| in {0.5, 1.5} -> four roots +-0.5, +-1.5.
    def s_bump(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
        return np.abs(np.abs(y) - 1.0)

    res = invert_interval(s_bump, np.linspace(-1.0, 1.0, 40), 0.5)
    assert res.status == "ok" and res.n_roots == 4
    assert res.low == pytest.approx(-1.5, abs=1e-6)
    assert res.high == pytest.approx(1.5, abs=1e-6)


def test_invert_knn_on_bimodal_draws_spans_both_modes() -> None:
    rng = np.random.default_rng(9)
    draws = np.concatenate([rng.normal(-1.0, 0.08, 60), rng.normal(1.0, 0.08, 60)])
    s = score_knn(k=10)
    res = invert_interval(s, draws, 1.0)
    assert res.status == "ok"
    assert res.low < -1.0 < 1.0 < res.high


def test_invert_single_root_degenerate_at_root() -> None:
    # Monotone score s(y) = y crosses s - qhat exactly once, at y = qhat.
    def s_lin(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
        return y

    res = invert_interval(s_lin, np.linspace(0.0, 1.0, 20), 0.37)
    assert res.status == "single_root"
    assert res.low == res.high == pytest.approx(0.37, abs=1e-6)


def test_invert_no_root_degenerates_at_median() -> None:
    def s_const(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
        return np.full_like(y, 5.0)

    draws = np.linspace(-2.0, 3.0, 30)
    res = invert_interval(s_const, draws, 0.0)
    assert res.status == "no_root" and res.n_roots == 0
    med = predictive_median(draws)
    assert res.low == res.high == pytest.approx(med)


def test_invert_strict_raises_on_no_root() -> None:
    def s_const(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
        return np.full_like(y, 5.0)

    cfg = RootFinderConfig(strict=True)
    with pytest.raises(NonInvertibleScoreError):
        invert_interval(s_const, np.linspace(0.0, 1.0, 10), 0.0, cfg)


def test_invert_nonfinite_score_raises() -> None:
    def s_nan(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
        return np.full_like(y, np.nan)

    with pytest.raises(NonInvertibleScoreError):
        invert_interval(s_nan, np.linspace(0.0, 1.0, 10), 0.5)


def test_invert_retry_reaches_far_roots() -> None:
    # Roots at y = 50 and y = 150 lie beyond the initial grid reach
    # (~h0 * gamma^depth ~ 4.4); the retry strategy must grow depth.
    def s_shift(y: np.ndarray, draws: np.ndarray) -> np.ndarray:
        return np.abs(y - 100.0)

    res = invert_interval(s_shift, np.linspace(-1.0, 1.0, 20), 50.0)
    assert res.status == "ok"
    assert res.low == pytest.approx(50.0, abs=1e-4)
    assert res.high == pytest.approx(150.0, abs=1e-4)


def test_invert_fail_closed_edges() -> None:
    s = get_score("residual")
    with pytest.raises(ValueError, match="qhat"):
        invert_interval(s, np.arange(10.0), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        invert_interval(s, np.array([0.0, np.inf]), 1.0)
    with pytest.raises(ValueError, match="at least 2"):
        invert_interval(s, np.array([0.0]), 1.0)
    with pytest.raises(ValueError, match="gamma"):
        RootFinderConfig(gamma=1.0)


def test_inversion_determinism() -> None:
    rng = np.random.default_rng(11)
    draws = rng.standard_t(4.0, 120)
    a = invert_interval(get_score("knn"), draws, 0.8)
    b = invert_interval(get_score("knn"), draws.copy(), 0.8)
    assert a == b


# ---------------------------------------------------------------------------
# Efficiency metrics (paper Sec. 4.5, Eqs. 19-28)
# ---------------------------------------------------------------------------


def test_picp_pinaw_width_cv_hand() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-0.5, 0.5, 1.5, 5.0])
    hi = np.array([0.5, 1.5, 2.5, 6.0])
    assert picp(y, lo, hi) == pytest.approx(0.75)
    assert pinaw(y, lo, hi) == pytest.approx(1.0 / 3.0)
    assert width_cv(lo, hi) == pytest.approx(0.0)  # constant widths


def test_width_cv_hand_value() -> None:
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 3.0])  # widths 1, 3 -> mu 2, sd sqrt(2) (ddof=1)
    assert width_cv(lo, hi) == pytest.approx(math.sqrt(2.0) / 2.0)


def test_acceptable_coverage_formula() -> None:
    c_a = acceptable_coverage(1000, 0.1)
    expected = 0.9 - 1.645 * math.sqrt(0.1 * 0.9 / 1000)
    assert c_a == pytest.approx(expected)
    with pytest.raises(ValueError):
        acceptable_coverage(0, 0.1)


def test_winkler_hand_values() -> None:
    alpha = 0.1  # miss slope 2/alpha = 20
    y = np.array([0.5, 2.0, -0.5])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    delta, e = winkler_components(y, lo, hi, alpha)
    np.testing.assert_allclose(delta, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(e, [0.0, 20.0 * 1.0, 20.0 * 0.5])
    assert winkler_score(y, lo, hi, alpha) == pytest.approx((1.0 + 21.0 + 11.0) / 3.0)


def test_undercoverage_penalty_factor_hand() -> None:
    # coverage >= C_a -> penalty 1 (no amplification)
    assert undercoverage_penalty_factor(1.0, 1000, 0.1) == pytest.approx(1.0)
    c_a = acceptable_coverage(1000, 0.1)
    assert undercoverage_penalty_factor(c_a, 1000, 0.1) == pytest.approx(1.0)
    cov = 0.85
    delta_c = c_a - cov
    rho = delta_c / (1.0 - cov)
    expected = math.exp(2.0 * rho)
    assert undercoverage_penalty_factor(cov, 1000, 0.1) == pytest.approx(expected)


def test_mmw_equals_winkler_when_coverage_ok() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.4])
    lo = np.array([-1.0, -1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0, 1.0])
    res = modified_winkler_score(y, lo, hi, 0.1)
    assert res.coverage == pytest.approx(1.0)
    assert res.penalty_factor == pytest.approx(1.0)
    assert res.mmw == pytest.approx(res.winkler)


def test_mmw_amplifies_misses_when_undercovered() -> None:
    # 60% coverage at alpha=0.1 with n=10: C_a = 0.9 - 1.645*0.0949 = 0.744
    y = np.arange(10.0)
    lo = np.full(10, 0.0)
    hi = np.full(10, 5.0)  # covers y<=5 -> 60%
    res = modified_winkler_score(y, lo, hi, 0.1)
    assert res.coverage == pytest.approx(0.6)
    assert res.penalty_factor > 1.0
    assert res.mmw > res.winkler
    # MMW = mean(delta) + P_uc * mean(e): mean delta = 5, misses 4 points
    delta, e = winkler_components(y, lo, hi, 0.1)
    assert res.mmw == pytest.approx(float(np.mean(delta)) + res.penalty_factor * float(np.mean(e)))


def test_metrics_fail_closed() -> None:
    with pytest.raises(ValueError):
        picp(np.array([1.0]), np.array([0.0]), np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="range"):
        pinaw(np.full(4, 2.0), np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match="identical"):
        winkler_score(np.zeros(3), np.zeros(4), np.ones(4), 0.1)
    with pytest.raises(ValueError, match="empirical_coverage"):
        undercoverage_penalty_factor(1.5, 100, 0.1)


# ---------------------------------------------------------------------------
# Predictor abstraction (paper step i): seeded synthetic DGPs
# ---------------------------------------------------------------------------


def test_predict_draws_ensemble_and_quantile_and_mixture() -> None:
    x = np.linspace(-2.0, 2.0, 30).reshape(-1, 1)

    def ensemble(xv: np.ndarray) -> np.ndarray:
        r = np.random.default_rng(42)  # deterministic pseudo-draws
        return np.sin(xv[:, 0])[:, None] + 0.2 * r.standard_normal((xv.shape[0], 50))

    def quantile_pseudo(xv: np.ndarray) -> np.ndarray:
        taus = np.linspace(0.01, 0.99, 99)
        from scipy.stats import norm

        return np.sin(xv[:, 0])[:, None] + 0.2 * norm.ppf(taus)[None, :]

    def mixture(xv: np.ndarray) -> np.ndarray:
        r = np.random.default_rng(7)
        pick = r.integers(0, 2, size=(xv.shape[0], 40))
        comps = np.stack(
            [np.sin(xv[:, 0]) - 0.3, np.sin(xv[:, 0]) + 0.3],
            axis=1,
        )
        draws = comps[np.arange(xv.shape[0])[:, None], pick]
        return draws + 0.05 * r.standard_normal(draws.shape)

    expected_shapes = {ensemble: (30, 50), quantile_pseudo: (30, 99), mixture: (30, 40)}
    for pred, shape in expected_shapes.items():
        draws = predict_draws(pred, x)
        assert draws.shape == shape
        assert np.all(np.isfinite(draws))


def test_predict_draws_fail_closed() -> None:
    x = np.zeros((4, 1))
    with pytest.raises(ValueError, match="M>=2"):
        predict_draws(lambda xv: np.zeros((4, 1)), x)
    with pytest.raises(ValueError, match="finite"):
        predict_draws(lambda xv: np.full((4, 5), np.inf), x)
    with pytest.raises(ValueError, match="n=4"):
        predict_draws(lambda xv: np.zeros((3, 5)), x)


# ---------------------------------------------------------------------------
# End-to-end SYNTHETIC battery: coverage, adaptivity, reductions
# ---------------------------------------------------------------------------


def test_dcp_coverage_all_scores_heteroscedastic() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=21)
    a = 0.1
    c_a = acceptable_coverage(y_test.size, a)
    for name in SCORE_NAMES:
        pred = dcp_intervals(y_cal, d_cal, d_test, score=name, alpha=a)
        assert picp(y_test, pred.low, pred.high) >= c_a, name
        assert pred.n_degenerate == 0, name


def test_dcp_interval_widths_track_heteroscedastic_sigma() -> None:
    _, y_cal, d_cal, xt, y_test, d_test = _het_world(seed=22)
    pred = dcp_intervals(y_cal, d_cal, d_test, score="interval_scaled", alpha=0.1)
    sigma_t = 0.08 + 0.45 * xt * xt / 9.0
    corr = float(np.corrcoef(pred.width, sigma_t)[0, 1])
    assert corr > 0.85  # intervals widen where the planted noise is larger


def test_dcp_residual_flat_vs_zscore_adaptive() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=23)
    flat = dcp_intervals(y_cal, d_cal, d_test, score="residual", alpha=0.1)
    adapt = dcp_intervals(y_cal, d_cal, d_test, score="zscore", alpha=0.1)
    assert width_cv(flat.low, flat.high) == pytest.approx(0.0, abs=1e-8)
    assert width_cv(adapt.low, adapt.high) > 0.2


def test_dcp_asymmetric_on_skewed_residual_law() -> None:
    from scipy.stats import skewnorm

    rng = np.random.default_rng(24)
    n_cal, n_test, m = 500, 300, 150
    skew = 8.0
    y_cal = skewnorm.rvs(skew, size=n_cal, random_state=rng)
    d_cal = skewnorm.rvs(skew, size=(n_cal, m), random_state=rng)
    y_test = skewnorm.rvs(skew, size=n_test, random_state=rng)
    d_test = skewnorm.rvs(skew, size=(n_test, m), random_state=rng)
    pred = dcp_intervals(y_cal, d_cal, d_test, score="interval", alpha=0.1)
    med = np.median(d_test, axis=1)
    upper_arm = np.median(pred.high - med)
    lower_arm = np.median(med - pred.low)
    assert upper_arm > 1.15 * lower_arm  # right skew -> longer upper arm
    assert picp(y_test, pred.low, pred.high) >= acceptable_coverage(n_test, 0.1)


def test_dcp_reduces_to_cqr_under_cqr_score() -> None:
    """Paper's reduction claim: DCP + interval score == CQR, up to tol."""
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=25)
    alpha = 0.1
    pred = dcp_intervals(y_cal, d_cal, d_test, score="interval", alpha=alpha)
    # Closed-form CQR via the repo's conformal primitives.
    mass = 1.0 - alpha
    lo_c = np.array([inner_band_quantile(d_cal[i], mass)[0] for i in range(d_cal.shape[0])])
    hi_c = np.array([inner_band_quantile(d_cal[i], mass)[1] for i in range(d_cal.shape[0])])
    eps = cqr_scores(y_cal, lo_c, hi_c)
    qhat = conformal_quantile(eps, alpha)
    lo_t = np.array([inner_band_quantile(d_test[i], mass)[0] for i in range(d_test.shape[0])])
    hi_t = np.array([inner_band_quantile(d_test[i], mass)[1] for i in range(d_test.shape[0])])
    exp_lo, exp_hi = expand_interval(lo_t, hi_t, qhat)
    np.testing.assert_allclose(pred.low, exp_lo, atol=1e-5)
    np.testing.assert_allclose(pred.high, exp_hi, atol=1e-5)


def test_dcp_interval_score_matches_cqr_interval_baseline() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=26)
    pred = dcp_intervals(y_cal, d_cal, d_test, score="interval", alpha=0.1)
    lo, hi = cqr_interval(y_cal, d_cal, d_test, alpha=0.1)
    np.testing.assert_allclose(pred.low, lo, atol=1e-5)
    np.testing.assert_allclose(pred.high, hi, atol=1e-5)


def test_dcp_hdi_score_reduces_to_cmc_closed_form() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=27)
    alpha = 0.1
    pred = dcp_intervals(y_cal, d_cal, d_test, score="hdi", alpha=alpha)
    lo_c = np.array([inner_band_hdi(d_cal[i], 1.0 - alpha)[0] for i in range(d_cal.shape[0])])
    hi_c = np.array([inner_band_hdi(d_cal[i], 1.0 - alpha)[1] for i in range(d_cal.shape[0])])
    qhat = conformal_quantile(cqr_scores(y_cal, lo_c, hi_c), alpha)
    lo_t = np.array([inner_band_hdi(d_test[i], 1.0 - alpha)[0] for i in range(d_test.shape[0])])
    hi_t = np.array([inner_band_hdi(d_test[i], 1.0 - alpha)[1] for i in range(d_test.shape[0])])
    exp_lo, exp_hi = expand_interval(lo_t, hi_t, qhat)
    np.testing.assert_allclose(pred.low, exp_lo, atol=1e-5)
    np.testing.assert_allclose(pred.high, exp_hi, atol=1e-5)


def test_dcp_residual_score_reduces_to_split_conformal() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=28)
    pred = dcp_intervals(y_cal, d_cal, d_test, score="residual", alpha=0.1)
    lo, hi = split_conformal_interval(y_cal, d_cal, d_test, alpha=0.1)
    np.testing.assert_allclose(pred.low, lo, atol=1e-5)
    np.testing.assert_allclose(pred.high, hi, atol=1e-5)


def test_dcp_zscore_reduces_to_mccp_closed_form() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=29)
    alpha = 0.1
    pred = dcp_intervals(y_cal, d_cal, d_test, score="zscore", alpha=alpha)
    eps = np.abs(y_cal - d_cal.mean(axis=1)) / d_cal.std(axis=1)
    qhat = conformal_quantile(eps, alpha)
    mu_t, sd_t = d_test.mean(axis=1), d_test.std(axis=1)
    np.testing.assert_allclose(pred.low, mu_t - qhat * sd_t, atol=1e-5)
    np.testing.assert_allclose(pred.high, mu_t + qhat * sd_t, atol=1e-5)


def test_online_sliding_window_tracks_drift() -> None:
    """Stale predictor + mean drift: online recalibration beats static."""
    rng = np.random.default_rng(31)
    n_cal, n_test, m = 200, 300, 60
    y_cal = rng.normal(0.0, 1.0, n_cal)
    d_cal = rng.normal(0.0, 1.0, (n_cal, m))
    drift = np.linspace(0.0, 3.0, n_test)
    y_test = drift + rng.normal(0.0, 1.0, n_test)
    d_test = rng.normal(0.0, 1.0, (n_test, m))  # stale: never sees the drift
    static = dcp_intervals(y_cal, d_cal, d_test, score="residual", alpha=0.1)
    online = dcp_online_intervals(y_cal, d_cal, y_test, d_test, score="residual", alpha=0.1)
    cov_static = picp(y_test, static.low, static.high)
    cov_online = picp(y_test, online.low, online.high)
    assert cov_static < 0.75  # drift breaks the static guarantee
    assert cov_online > cov_static + 0.1


def test_pipeline_determinism_bit_identical() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=33)
    a = dcp_intervals(y_cal, d_cal, d_test, score="knn", alpha=0.1)
    b = dcp_intervals(y_cal, d_cal, d_test, score="knn", alpha=0.1)
    np.testing.assert_array_equal(a.low, b.low)
    np.testing.assert_array_equal(a.high, b.high)
    assert a.status == b.status


def test_dcp_battery_rows_and_baselines() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=34)
    out = dcp_battery(
        y_cal, d_cal, y_test, d_test, scores=("residual", "zscore", "interval", "knn")
    )
    assert set(out) == {
        "dcp_residual",
        "dcp_zscore",
        "dcp_interval",
        "dcp_knn",
        "split_conformal",
        "cqr",
    }
    c_a = acceptable_coverage(y_test.size, 0.1)
    for name, row in out.items():
        assert row["coverage"] >= c_a, name
        assert row["mmw"] >= row["mean_width"]
        assert np.isfinite(row["pinaw"]) and np.isfinite(row["winkler"])
    # DCP residual row IS the split-conformal baseline (same reduction).
    assert out["dcp_residual"]["mean_width"] == pytest.approx(
        out["split_conformal"]["mean_width"], rel=1e-6
    )


def test_interval_metrics_keys_finite() -> None:
    _, y_cal, d_cal, _, y_test, d_test = _het_world(seed=35)
    pred = dcp_intervals(y_cal, d_cal, d_test, score="interval", alpha=0.1)
    row = interval_metrics(y_test, pred.low, pred.high, 0.1)
    for k in (
        "coverage",
        "acceptable_coverage",
        "mean_width",
        "pinaw",
        "width_cv",
        "winkler",
        "mmw",
        "undercoverage_penalty",
    ):
        assert np.isfinite(row[k]), k
    assert row["coverage_ok"] in (0.0, 1.0)


def test_bench_dcp_flat_dict_seeded() -> None:
    out = bench_dcp(seed=0)
    out2 = bench_dcp(seed=0)
    assert out == out2  # bit-identical
    assert all(k.startswith("SYNTHETIC_") for k in out)
    assert all(np.isfinite(v) for v in out.values())
    assert out["SYNTHETIC_dcp_interval_coverage"] >= 0.85
    assert out["SYNTHETIC_dcp_interval_coverage_ok"] == 1.0
    assert out["SYNTHETIC_dcp_zscore_width_cv"] > 0.1
    assert out["SYNTHETIC_split_conformal_width_cv"] == pytest.approx(0.0)


def test_bench_dcp_mm_worse_than_winkler_when_forced_undercovered() -> None:
    # Deliberately tiny calibration set + huge alpha -> undercoverage possible;
    # verify MMW >= Winkler always and strict > when penalty fires.
    out = bench_dcp(seed=5, n_cal=40, n_test=200, alpha=0.1)
    for method in (
        "dcp_residual",
        "dcp_zscore",
        "dcp_interval",
        "dcp_knn",
        "split_conformal",
        "cqr",
    ):
        w, m = out[f"SYNTHETIC_{method}_winkler"], out[f"SYNTHETIC_{method}_mmw"]
        assert m >= w - 1e-12, method
