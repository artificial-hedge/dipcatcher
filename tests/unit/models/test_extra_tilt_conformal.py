"""SYNTHETIC correctness tests for ExTRA tilt conformal (ExTRA-WCP / -WCP-T).

Method under test: Choi, "Conformal Prediction under Exponential-Tilt Joint
Shift", arXiv:2609.30886 (fetched and verified) — exponential-tilt joint
shift model (Eq. 1), marginal-matching / KL-projection estimation (Eq. 4-6,
App. B), candidate-dependent weighted calibration (Eq. 12-13, 26), predictive
tilting (Eq. 9-10), the shared surrogate-coverage guarantee (Eq. 15-16), the
identification calculations (Prop. 1 / Eq. 25), and the eta mode-signal
sensitivity ladder (Section 5.4). ExTRA itself: Maity et al., ICLR 2023.

Every fixture here is SYNTHETIC (the paper's Section 5.2 bimodal
truncated-Gaussian DGP, Eq. 22) — a correctness test, never market evidence.
Metrics are proper-score quantities only (coverage, prediction-set length,
ESS, ratio-error diagnostics); no Sharpe/P&L-family keys.

BRIEF CORRECTION (verified against the fetched paper): at zero mode signal
(eta=0) the paper reports ExTRA-WCP coverage 0.727 and ExTRA-WCP-T 0.493
(Fig. 3) — weights-only does NOT stay fully valid on the TRUE target there,
because the estimated joint ratio itself is unidentified (TV error 0.470,
Table 4). What stays valid for BOTH scores is coverage under the surrogate
Q^ = w^.P the fitted weights define (Eq. 15). The uninformative-regime test
therefore asserts (i) surrogate coverage near nominal for both scores, (ii)
weights-only degrading gracefully but under-covering on the true target, and
(iii) full tilting collapsing far below weights-only.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.extra_tilt_conformal import (
    SOURCE_POSITIVE_MODE_PROB,
    ExtraRunResult,
    SourceConditional,
    SyntheticExTRAData,
    TiltFit,
    bench_extra_tilt,
    extra_prediction_intervals,
    extra_tilted_predictive,
    extra_weighted_calibration,
    extra_weighted_rank,
    fit_exponential_tilt,
    fit_source_model,
    regularization_check_ridge,
    run_extra_replication,
    solve_mode_intercept,
    surrogate_coverage,
    surrogate_resample,
    synthetic_bimodal_shift,
    tilt_weights,
    when_to_tilt_diagnostic,
)
from quant_fund.models.weighted_conformal import weighted_conformal_quantile

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA
REPS = 10
SEEDS = tuple(17 + 1000 * r for r in range(REPS))

# Paper Table 2 at (a*,b*)=(1,1.2), eta=1 (full sizes): standard CP
# 0.755/1.488, WCP 0.904/2.190, WCP-T 0.899/1.525, paired length reduction
# 30.34% [28.56, 32.12], coverage diff -0.0052. Tolerances below are sized
# for the shrunk 10-replication fixture (SE ~ 0.01), not the paper's 30.
TOL_INFORMATIVE = 0.05
MIN_REDUCTION_PERCENT = 15.0
# Paper Fig. 3 at eta=0: WCP 0.727, WCP-T 0.493, paired diff -0.234
# [-0.269, -0.199]; 15/30 b-hats wrong sign, 6/30 at the b=3 bound.
UNINFORMATIVE_WCP_T_CEILING = 0.62
UNINFORMATIVE_PAIRED_MARGIN = 0.15
TOL_SURROGATE_MEAN = 0.03
TOL_SURROGATE_REP = 0.06


def _rep(eta: float, seed: int) -> tuple[ExtraRunResult, SyntheticExTRAData]:
    data = synthetic_bimodal_shift(eta, 1.0, 1.2, seed=seed)
    return run_extra_replication(data, alpha=ALPHA), data


@pytest.fixture(scope="module")
def informative() -> list[tuple[ExtraRunResult, SyntheticExTRAData]]:
    """eta=1: target inputs informative about the response shift (Prop. 1)."""
    return [_rep(1.0, s) for s in SEEDS]


@pytest.fixture(scope="module")
def uninformative() -> list[tuple[ExtraRunResult, SyntheticExTRAData]]:
    """eta=0: constant mode probability, Eq. 25 — b* unidentified."""
    return [_rep(0.0, s) for s in SEEDS]


# ---------------------------------------------------------------------------
# Fixture and identification calculations
# ---------------------------------------------------------------------------


def test_mode_intercept_ladder_matches_paper() -> None:
    """App. B: d_eta ~= -2.000, -1.481, -1.331, -1.279 at eta=1,.5,.25,0."""
    expected = {1.0: -2.000, 0.5: -1.481, 0.25: -1.331, 0.0: -1.279}
    for eta, d in expected.items():
        assert solve_mode_intercept(eta) == pytest.approx(d, abs=2e-3)
    # Section 5.2 closed form: p_+ = [log(1+e) - log(1+e^-5)]/6 ~= 0.218
    closed = float((np.log1p(np.e) - np.log1p(np.exp(-5.0))) / 6.0)
    assert closed == pytest.approx(SOURCE_POSITIVE_MODE_PROB, abs=1e-6)
    with pytest.raises(ValueError):
        solve_mode_intercept(-0.5)
    with pytest.raises(ValueError):
        solve_mode_intercept(float("nan"))


def test_fixture_target_marginals_match_the_tilt() -> None:
    """Eval-only oracle check: the fixture target really is h* . P.

    Section 5.2: Q(Z=+1) = p_+ e^{b*} / (p_+ e^{b*} + (1-p_+) e^{-b*}) = 0.754
    at b*=1.2; the a*=1 covariate tilt moves E_Q[U] to E[u e^u]/E[e^u] ~=
    0.313. Source mode frequency stays at p_+ ~= 0.218.
    """
    data = synthetic_bimodal_shift(1.0, 1.0, 1.2, n_test=6000, seed=42)
    frac_pos = float(np.mean(data.y_test > 0.0))
    p_plus = float(SOURCE_POSITIVE_MODE_PROB)
    q_pos = p_plus * np.exp(1.2) / (p_plus * np.exp(1.2) + (1 - p_plus) * np.exp(-1.2))
    assert frac_pos == pytest.approx(float(q_pos), abs=0.03)
    e_u = float(np.mean(data.x_test[:, 0]))
    assert 0.20 < e_u < 0.45
    src_frac = float(np.mean(data.y_cal > 0.0))
    assert src_frac == pytest.approx(p_plus, abs=0.03)
    # truncation: sign(y) always observes the mode (Eq. 22 restricted law)
    assert np.all(np.abs(data.y_train) > 0.0)


def test_synthetic_generator_fail_closed() -> None:
    with pytest.raises(ValueError):
        synthetic_bimodal_shift(1.0, 1.0, 1.2, n_train=0)
    with pytest.raises(ValueError):
        synthetic_bimodal_shift(-0.1, 1.0, 1.2)
    with pytest.raises(ValueError):
        synthetic_bimodal_shift(float("nan"), 1.0, 1.2)
    with pytest.raises(ValueError):
        synthetic_bimodal_shift(1.0, float("inf"), 1.2)
    with pytest.raises(ValueError):
        synthetic_bimodal_shift(1.0, 1.0, 1.2, n_surrogate_pool=0)


def test_synthetic_generator_deterministic() -> None:
    a = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=200,
        n_shift=200,
        m_target=300,
        n_cal=200,
        n_test=300,
        n_surrogate_pool=200,
        seed=5,
    )
    b = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=200,
        n_shift=200,
        m_target=300,
        n_cal=200,
        n_test=300,
        n_surrogate_pool=200,
        seed=5,
    )
    for field in (
        "x_train",
        "y_train",
        "x_shift_target",
        "x_cal",
        "y_cal",
        "x_test",
        "y_test",
        "x_surrogate_pool",
        "y_surrogate_pool",
    ):
        assert np.array_equal(getattr(a, field), getattr(b, field))
    assert a.d_eta == b.d_eta


def test_zero_signal_leaves_response_tilt_unidentified() -> None:
    """Eq. 25: at constant mode probability the induced input ratio is
    invariant to b — target inputs carry NO information about it. With a
    nonconstant mode probability (Prop. 1) the ratio does depend on b."""
    x = np.column_stack([np.linspace(-1, 1, 97), np.linspace(-1, 1, 97)[::-1]])
    mu = np.array([[1.25, 0.0, 0.10], [-1.25, 0.0, 0.10]])
    log_sig = np.array([np.log(0.22), 0.0])
    p = float(SOURCE_POSITIVE_MODE_PROB)
    flat = SourceConditional(
        gate=np.array([np.log(p / (1 - p)), 0.0, 0.0]), mu_coef=mu, log_sigma_coef=log_sig
    )
    varying = SourceConditional(gate=np.array([-2.0, 0.0, 3.0]), mu_coef=mu, log_sigma_coef=log_sig)
    for model, expect_invariant in ((flat, True), (varying, False)):
        r0 = model.moment(x, np.array([1.0, 0.0]))
        r0 = r0 / float(np.mean(r0))
        r12 = model.moment(x, np.array([1.0, 1.2]))
        r12 = r12 / float(np.mean(r12))
        gap = float(np.max(np.abs(r0 - r12)))
        if expect_invariant:
            assert gap < 1e-12
        else:
            assert gap > 0.05


# ---------------------------------------------------------------------------
# Estimation: source fit and ExTRA tilt fit
# ---------------------------------------------------------------------------


def test_fit_source_model_recovers_dgp() -> None:
    """App. B two-stage fit recovers the Eq. 22 generating parameters."""
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=6000,
        n_shift=50,
        m_target=50,
        n_cal=50,
        n_test=50,
        n_surrogate_pool=50,
        seed=99,
    )
    sm = fit_source_model(data.x_train, data.y_train)
    assert sm.gate[0] == pytest.approx(-2.0, abs=0.15)
    assert sm.gate[1] == pytest.approx(0.0, abs=0.15)
    assert sm.gate[2] == pytest.approx(3.0, abs=0.3)
    assert sm.mu_coef[0, 0] == pytest.approx(1.25, abs=0.08)
    assert sm.mu_coef[1, 0] == pytest.approx(-1.25, abs=0.08)
    assert sm.mu_coef[0, 2] == pytest.approx(0.10, abs=0.06)
    assert sm.log_sigma_coef[0] == pytest.approx(np.log(0.22), abs=0.06)
    assert sm.log_sigma_coef[1] == pytest.approx(0.8, abs=0.08)


def test_fit_source_model_fail_closed() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (40, 2))
    y = np.abs(rng.normal(size=40)) + 0.1
    with pytest.raises(ValueError):
        fit_source_model(x[:39], y)  # row mismatch
    with pytest.raises(ValueError):
        fit_source_model(x, y)  # single response sign only
    y2 = y.copy()
    y2[:20] *= -1.0
    y2[0] = 0.0
    with pytest.raises(ValueError):
        fit_source_model(x, y2)  # y = 0: sign undefined
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_source_model(bad, y2)
    with pytest.raises(ValueError):
        fit_source_model(rng.uniform(-1, 1, (40, 3)), y2)  # wrong column count
    with pytest.raises(ValueError):
        fit_source_model(x, y2, c_gate=0.0)


def test_fit_exponential_tilt_recovers_informative_shift(
    informative: list[tuple[ExtraRunResult, SyntheticExTRAData]],
) -> None:
    """Eq. 6 estimation recovers (a*,b*)=(1,1.2) when inputs are informative.

    Paper Sec. 5.4 at eta=1: b-hat averages 1.245 (SD 0.139) at full sizes.
    """
    betas = np.vstack([r.tilt_fit.beta for r, _ in informative])
    assert float(np.mean(np.abs(betas[:, 0] - 1.0))) <= 0.15
    assert float(np.mean(np.abs(betas[:, 1] - 1.2))) <= 0.25
    assert np.all(betas[:, 1] > 0.5)  # no wrong-sign fits at eta=1
    fit = informative[0][0].tilt_fit
    assert fit.ridge == pytest.approx(1.0 / fit.n_shift)  # App. B main penalty
    assert np.all(np.isfinite(fit.start_objectives))
    assert np.all(fit.start_converged)
    assert regularization_check_ridge(fit.n_shift, fit.m_target) > fit.ridge


def test_fit_exponential_tilt_fail_closed() -> None:
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=200,
        n_shift=200,
        m_target=200,
        n_cal=50,
        n_test=50,
        n_surrogate_pool=50,
        seed=3,
    )
    sm = fit_source_model(data.x_train, data.y_train)
    with pytest.raises(TypeError):
        fit_exponential_tilt(object(), data.x_shift_src, data.y_shift_src, data.x_shift_target)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fit_exponential_tilt(
            sm, data.x_shift_src[:199], data.y_shift_src, data.x_shift_target
        )  # row mismatch
    y0 = data.y_shift_src.copy()
    y0[0] = 0.0
    with pytest.raises(ValueError):
        fit_exponential_tilt(sm, data.x_shift_src, y0, data.x_shift_target)
    with pytest.raises(ValueError):
        fit_exponential_tilt(
            sm, data.x_shift_src, data.y_shift_src, np.zeros((50, 3))
        )  # target inputs need 2 columns
    with pytest.raises(ValueError):
        fit_exponential_tilt(
            sm,
            data.x_shift_src,
            data.y_shift_src,
            data.x_shift_target,
            bounds=((1.0, -1.0), (-3.0, 3.0)),
        )  # lo >= hi
    with pytest.raises(ValueError):
        fit_exponential_tilt(
            sm, data.x_shift_src, data.y_shift_src, data.x_shift_target, ridge=-1.0
        )
    with pytest.raises(ValueError):
        fit_exponential_tilt(
            sm, data.x_shift_src, data.y_shift_src, data.x_shift_target, starts=((0.0, 4.0),)
        )  # start outside bounds
    with pytest.raises(ValueError):
        fit_exponential_tilt(
            sm, data.x_shift_src, data.y_shift_src, data.x_shift_target, max_iter=0
        )


# ---------------------------------------------------------------------------
# Weighted calibration algebra (Eq. 12-13, 21, 26)
# ---------------------------------------------------------------------------


def test_extra_weighted_calibration_reduces_to_known_quantiles() -> None:
    """Uniform weights + unit candidate = Vovk; candidate = mean calibration
    weight = the cousin module's Tibshirani quantile (atom conventions agree
    exactly at that candidate weight)."""
    rng = np.random.default_rng(11)
    scores = np.abs(rng.normal(size=500)) + 0.25
    weights = rng.uniform(0.5, 2.0, size=500)
    q_plain = extra_weighted_calibration(scores, np.ones(500), 1.0, ALPHA)
    assert q_plain == pytest.approx(conformal_quantile(scores, ALPHA))
    q_cousin = extra_weighted_calibration(scores, weights, float(np.mean(weights)), ALPHA)
    assert q_cousin == pytest.approx(weighted_conformal_quantile(scores, weights, ALPHA))
    # monotone in the candidate weight
    qs = [extra_weighted_calibration(scores, weights, w, ALPHA) for w in (0.5, 2.0, 8.0)]
    assert qs[0] <= qs[1] <= qs[2]


def test_candidate_atom_forced_inclusion_eq21() -> None:
    """Eq. 21: H_y > alpha*W/(1-alpha) forces inclusion under EVERY score,
    and Eq. 26 returns +inf exactly in that regime (never a silent clip)."""
    rng = np.random.default_rng(12)
    scores = rng.uniform(0.0, 1.0, 300)
    weights = rng.uniform(0.5, 1.5, 300)
    total = float(np.sum(weights))
    boundary = ALPHA * total / (1.0 - ALPHA)
    worst = float(np.max(scores)) + 100.0
    q_inf = extra_weighted_calibration(scores, weights, 1.5 * boundary, ALPHA)
    assert q_inf == float("inf")
    assert extra_weighted_rank(scores, weights, worst, 1.5 * boundary) > ALPHA
    q_fin = extra_weighted_calibration(scores, weights, 0.5 * boundary, ALPHA)
    assert np.isfinite(q_fin)
    assert extra_weighted_rank(scores, weights, worst, 0.5 * boundary) <= ALPHA
    assert extra_weighted_rank(scores, weights, float(np.min(scores)) - 1.0, 0.5 * boundary) > ALPHA


def test_weighted_rank_matches_threshold_rule() -> None:
    """Eq. 12/13 vs Eq. 26: p^(y) > alpha  <=>  S(x,y) <= q on a half-line."""
    rng = np.random.default_rng(13)
    scores = np.sort(rng.uniform(0.0, 5.0, 200)) + np.arange(200) * 1e-4
    weights = rng.uniform(0.2, 3.0, 200)
    total = float(np.sum(weights))
    grid = np.linspace(-1.0, 6.0, 61)
    for cand_w in (0.3, float(np.mean(weights)), 0.9 * ALPHA * total / (1.0 - ALPHA)):
        q = extra_weighted_calibration(scores, weights, cand_w, ALPHA)
        for v in grid:
            included = extra_weighted_rank(scores, weights, float(v), cand_w) > ALPHA
            if np.isinf(q):
                assert included  # forced inclusion everywhere
            else:
                assert included == (v <= q + 1e-12)


def test_extra_weighted_calibration_fail_closed() -> None:
    scores = np.arange(20, dtype=float)
    weights = np.ones(20)
    with pytest.raises(ValueError):
        extra_weighted_calibration(scores, weights[:19], 1.0, ALPHA)
    with pytest.raises(ValueError):
        extra_weighted_calibration(scores, np.zeros(20), 1.0, ALPHA)
    with pytest.raises(ValueError):
        extra_weighted_calibration(scores, weights, 0.0, ALPHA)
    with pytest.raises(ValueError):
        extra_weighted_calibration(scores, weights, float("nan"), ALPHA)
    for bad_alpha in (0.0, 1.0, -0.2, 1.4):
        with pytest.raises(ValueError):
            extra_weighted_calibration(scores, weights, 1.0, bad_alpha)
    nan_scores = scores.copy()
    nan_scores[3] = np.nan
    with pytest.raises(ValueError):
        extra_weighted_calibration(nan_scores, weights, 1.0, ALPHA)
    with pytest.raises(ValueError):
        extra_weighted_rank(scores, weights, float("inf"), 1.0)
    with pytest.raises(ValueError):
        extra_weighted_rank(scores, weights, 1.0, -2.0)


# ---------------------------------------------------------------------------
# Predictive tilting (Eq. 9-10)
# ---------------------------------------------------------------------------


def test_tilted_predictive_gate_shift_cancellation_and_normalization() -> None:
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=800,
        n_shift=100,
        m_target=100,
        n_cal=100,
        n_test=100,
        n_surrogate_pool=100,
        seed=8,
    )
    sm = fit_source_model(data.x_train, data.y_train)
    beta = np.array([0.7, 1.1])
    tilted = extra_tilted_predictive(sm, beta)
    x = np.array([[0.3, 0.4]])
    # Gate logit shifts by exactly 2b (Saerens-style prior adjustment)
    g = float(sm.gate[0] + sm.gate[1] * 0.3 + sm.gate[2] * 0.4)
    assert float(tilted.mode_prob(x)[0]) == pytest.approx(
        float(expit(g + 2.0 * beta[1])), abs=1e-12
    )
    # Input-only tilts cancel from the conditional law (paper Section 3.1)
    input_only = extra_tilted_predictive(sm, np.array([1.3, 0.0]))
    ys = np.linspace(-4.0, 4.0, 121)
    ys = ys[np.abs(ys) > 1e-9]
    xs = np.repeat(x, ys.size, axis=0)
    assert np.allclose(input_only.score(xs, ys), sm.score(xs, ys), atol=1e-12)
    # Within-mode components unchanged: tilted reweights modes only
    y_pos = np.linspace(0.05, 4.0, 41)
    x_pos = np.repeat(x, y_pos.size, axis=0)
    tilt_minus_gate = tilted.log_density(x_pos, y_pos) - np.log(tilted.mode_prob(x)[0])
    src_minus_gate = sm.log_density(x_pos, y_pos) - np.log(sm.mode_prob(x)[0])
    assert np.allclose(tilt_minus_gate, src_minus_gate, atol=1e-12)
    # Eq. 9 normalization: the tilted conditional integrates to one
    grid_p = np.linspace(1e-6, 6.0, 6001)
    grid_n = np.linspace(-6.0, -1e-6, 6001)
    mass_p = float(
        np.trapezoid(np.exp(tilted.log_density(np.repeat(x, grid_p.size, axis=0), grid_p)), grid_p)
    )
    mass_n = float(
        np.trapezoid(np.exp(tilted.log_density(np.repeat(x, grid_n.size, axis=0), grid_n)), grid_n)
    )
    assert mass_p == pytest.approx(float(tilted.mode_prob(x)[0]), abs=1e-3)
    assert mass_p + mass_n == pytest.approx(1.0, abs=1e-3)
    # log normalizer IS the conditional moment (paper Section 3.1, p0 = p^_P)
    assert np.allclose(tilted.log_normalizer(x), sm.log_moment(x, beta), atol=1e-12)


def test_tilted_predictive_fail_closed() -> None:
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=200,
        n_shift=50,
        m_target=50,
        n_cal=50,
        n_test=50,
        n_surrogate_pool=50,
        seed=4,
    )
    sm = fit_source_model(data.x_train, data.y_train)
    with pytest.raises(TypeError):
        extra_tilted_predictive(object(), np.zeros(2))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        extra_tilted_predictive(sm, np.zeros(3))
    with pytest.raises(ValueError):
        extra_tilted_predictive(sm, np.array([0.0, np.nan]))
    tilted = extra_tilted_predictive(sm, np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        tilted.score(np.zeros((2, 2)), np.ones(3))  # row mismatch
    with pytest.raises(ValueError):
        tilted.log_density(np.zeros((2, 2)), np.array([1.0, 0.0]))  # y = 0


def test_tilt_weights_identity() -> None:
    x = np.array([[0.3, -0.4], [-0.9, 0.2]])
    y = np.array([1.5, -0.7])
    beta = np.array([0.8, -1.3])
    expected = np.exp(0.8 * x[:, 0] + (-1.3) * np.sign(y))
    assert np.allclose(tilt_weights(x, y, beta), expected, atol=1e-12)
    with pytest.raises(ValueError):
        tilt_weights(x, np.array([1.5, 0.0]), beta)
    with pytest.raises(ValueError):
        tilt_weights(x, y[:1], beta)


# ---------------------------------------------------------------------------
# Set inversion and the shared coverage theory (Eq. 15-16, App. B)
# ---------------------------------------------------------------------------


def test_standard_cp_reduction_is_vovk_order_statistic() -> None:
    """beta=(0,0): uniform weights + unit candidate reproduce split conformal."""
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=400,
        n_shift=100,
        m_target=100,
        n_cal=400,
        n_test=60,
        n_surrogate_pool=100,
        seed=6,
    )
    sm = fit_source_model(data.x_train, data.y_train)
    iv = extra_prediction_intervals(sm, np.zeros(2), data.x_cal, data.y_cal, data.x_test, ALPHA)
    q_expected = conformal_quantile(sm.score(data.x_cal, data.y_cal), ALPHA)
    assert np.allclose(iv.thresholds_pos, q_expected, atol=1e-12)
    assert np.allclose(iv.thresholds_neg, q_expected, atol=1e-12)
    with pytest.raises(ValueError):
        extra_prediction_intervals(sm, np.zeros(3), data.x_cal, data.y_cal, data.x_test, ALPHA)
    with pytest.raises(ValueError):
        extra_prediction_intervals(
            sm, np.zeros(2), data.x_cal[:399], data.y_cal, data.x_test, ALPHA
        )
    with pytest.raises(ValueError):
        extra_prediction_intervals(sm, np.zeros(2), data.x_cal, data.y_cal, data.x_test, 0.0)
    with pytest.raises(TypeError):
        extra_prediction_intervals(
            object(),
            np.zeros(2),
            data.x_cal,
            data.y_cal,  # type: ignore[arg-type]
            data.x_test,
            ALPHA,
        )


def test_surrogate_coverage_holds_for_both_scores(
    informative: list[tuple[ExtraRunResult, SyntheticExTRAData]],
    uninformative: list[tuple[ExtraRunResult, SyntheticExTRAData]],
) -> None:
    """Eq. 15: under Q^ = w^.P BOTH scores cover at >= 1 - alpha.

    This is the sense in which weighted calibration "stays valid" even when
    the true-target coverage of the same fitted ratio collapses (Eq. 16 pays
    d_TV(Q, Q^)); the eta=0 block below contrasts the two directly.
    """
    for rows in (informative, uninformative):
        cov_w, cov_t = [], []
        for i, (res, data) in enumerate(rows[:4]):
            w, t = surrogate_coverage(data, res, seed=900 + i)
            cov_w.append(w)
            cov_t.append(t)
        assert float(np.mean(cov_w)) >= NOMINAL - TOL_SURROGATE_MEAN
        assert float(np.mean(cov_t)) >= NOMINAL - TOL_SURROGATE_MEAN
        assert min(cov_w) >= NOMINAL - TOL_SURROGATE_REP
        assert min(cov_t) >= NOMINAL - TOL_SURROGATE_REP


def test_surrogate_helpers_fail_closed() -> None:
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=200,
        n_shift=100,
        m_target=100,
        n_cal=100,
        n_test=100,
        n_surrogate_pool=100,
        seed=2,
    )
    with pytest.raises(ValueError):
        surrogate_resample(data.x_surrogate_pool[:99], data.y_surrogate_pool, np.zeros(2), 10, 0)
    with pytest.raises(ValueError):
        surrogate_resample(data.x_surrogate_pool, data.y_surrogate_pool, np.zeros(2), 0, 0)
    x, y = surrogate_resample(
        data.x_surrogate_pool, data.y_surrogate_pool, np.array([1.0, 1.2]), 250, 7
    )
    x2, y2 = surrogate_resample(
        data.x_surrogate_pool, data.y_surrogate_pool, np.array([1.0, 1.2]), 250, 7
    )
    assert np.array_equal(x, x2) and np.array_equal(y, y2)  # seeded determinism
    assert x.shape == (250, 2) and y.shape == (250,)
    # tilted toward the positive mode: the resample favors sign(y) = +1
    assert float(np.mean(y > 0)) > float(np.mean(data.y_surrogate_pool > 0)) + 0.1


# ---------------------------------------------------------------------------
# The paper's headline contrasts (SYNTHETIC, shrunk budgets)
# ---------------------------------------------------------------------------


def test_informative_tilting_narrows_at_matched_coverage(
    informative: list[tuple[ExtraRunResult, SyntheticExTRAData]],
) -> None:
    """Sec. 5.2 / Table 2: matched DGP + informative inputs => WCP-T cuts
    mean set length ~30% (paper: 30.34% [28.56, 32.12]) at coverage near
    nominal for BOTH methods (paper: 0.904 vs 0.899). Direction replicated
    at shrunk sizes; standard CP loses coverage (paper: 0.755)."""
    cov_s = float(np.mean([r.coverage_standard for r, _ in informative]))
    cov_w = float(np.mean([r.coverage_wcp for r, _ in informative]))
    cov_t = float(np.mean([r.coverage_wcp_t for r, _ in informative]))
    len_w = float(np.mean([r.length_wcp for r, _ in informative]))
    len_t = float(np.mean([r.length_wcp_t for r, _ in informative]))
    red = float(np.mean([r.length_reduction_percent for r, _ in informative]))
    diff = float(np.mean([r.coverage_diff_wcp_t_minus_wcp for r, _ in informative]))
    assert cov_w >= NOMINAL - TOL_INFORMATIVE
    assert cov_t >= NOMINAL - TOL_INFORMATIVE
    assert cov_s <= NOMINAL - 0.10
    assert red >= MIN_REDUCTION_PERCENT
    assert len_t < 0.85 * len_w
    assert abs(diff) <= 0.05
    # paper App. B: all evaluated synthetic regression sets have finite length
    for r, _ in informative:
        assert np.all(np.isfinite(r.intervals_wcp_t.total_length()))
        assert np.all(np.isfinite(r.intervals_wcp.total_length()))


def test_uninformative_tilting_undercovers_weights_only_degrades_gracefully(
    uninformative: list[tuple[ExtraRunResult, SyntheticExTRAData]],
) -> None:
    """Sec. 5.4 / Fig. 3 at eta=0: WCP 0.727 vs WCP-T 0.493, paired diff
    -0.234; half the b-hats wrong sign. BRIEF CORRECTION: weights-only also
    under-covers the TRUE target here (the joint ratio is unidentified, TV
    0.470) — but it degrades gracefully while full tilting collapses, and
    both remain valid under the surrogate (separate test above)."""
    cov_w = float(np.mean([r.coverage_wcp for r, _ in uninformative]))
    cov_t = float(np.mean([r.coverage_wcp_t for r, _ in uninformative]))
    diff = float(np.mean([r.coverage_diff_wcp_t_minus_wcp for r, _ in uninformative]))
    assert cov_t <= UNINFORMATIVE_WCP_T_CEILING
    assert cov_t <= NOMINAL - 0.20
    assert cov_w >= cov_t + UNINFORMATIVE_PAIRED_MARGIN
    assert diff <= -UNINFORMATIVE_PAIRED_MARGIN
    assert cov_w >= UNINFORMATIVE_WCP_T_CEILING + 0.05  # graceful, not collapsed
    assert cov_w <= NOMINAL - 0.10  # honest: still under nominal at eta=0
    b_hats = np.array([r.tilt_fit.beta[1] for r, _ in uninformative])
    a_hats = np.array([r.tilt_fit.beta[0] for r, _ in uninformative])
    assert int(np.sum(b_hats < 0)) >= 3  # paper: 15/30 wrong sign at eta=0
    assert float(np.mean(np.abs(a_hats - 1.0))) <= 0.15  # u-marginal still identifies a
    for r, _ in uninformative:
        assert np.all(np.isfinite(r.intervals_wcp_t.total_length()))


def test_when_to_tilt_diagnostic_separates_regimes(
    informative: list[tuple[ExtraRunResult, SyntheticExTRAData]],
    uninformative: list[tuple[ExtraRunResult, SyntheticExTRAData]],
) -> None:
    """The heuristic fires tilt on the informative regime and falls back to
    the conservative weights-only default at zero signal — in every
    replication. The paper proves no such rule can be validated from source
    labels + target inputs alone; this only operationalizes Prop. 1 /
    Eq. 25 (mode-signal variability) and Sec. 5.4 (penalty instability)."""
    for res, data in informative:
        dec = when_to_tilt_diagnostic(
            res.source_model,
            res.tilt_fit,
            data.x_shift_src,
            tilt_weights(data.x_cal, data.y_cal, res.tilt_fit.beta),
            tilt_fit_check=res.tilt_fit_check,
        )
        assert dec.recommend_tilt and dec.decision == "tilt"
        assert dec.mode_signal_sd >= 0.15
        assert dec.tilt_coefficient_spread <= 0.25
    for res, data in uninformative:
        dec = when_to_tilt_diagnostic(
            res.source_model,
            res.tilt_fit,
            data.x_shift_src,
            tilt_weights(data.x_cal, data.y_cal, res.tilt_fit.beta),
            tilt_fit_check=res.tilt_fit_check,
        )
        assert not dec.recommend_tilt and dec.decision == "weights_only"
        assert dec.mode_signal_sd < 0.05
        assert dec.tilt_coefficient_spread >= 0.5
        assert "weights_only" in dec.rationale


def test_when_to_tilt_diagnostic_fail_closed_and_conservative() -> None:
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=300,
        n_shift=200,
        m_target=200,
        n_cal=200,
        n_test=100,
        n_surrogate_pool=100,
        seed=1,
    )
    sm = fit_source_model(data.x_train, data.y_train)
    fit = fit_exponential_tilt(sm, data.x_shift_src, data.y_shift_src, data.x_shift_target)
    weights = tilt_weights(data.x_cal, data.y_cal, fit.beta)
    # fewer than two usable converged coefficients -> infinite spread -> no tilt
    lone = TiltFit(
        beta=fit.beta,
        objective=fit.objective,
        start_betas=np.vstack([fit.beta, [np.nan, np.nan]]),
        start_objectives=np.array([fit.objective, -np.inf]),
        start_converged=np.array([True, False]),
        ridge=fit.ridge,
        n_shift=fit.n_shift,
        m_target=fit.m_target,
    )
    dec = when_to_tilt_diagnostic(sm, lone, data.x_shift_src, weights, min_mode_signal_sd=0.0)
    assert dec.tilt_coefficient_spread == float("inf")
    assert not dec.recommend_tilt and dec.decision == "weights_only"
    with pytest.raises(TypeError):
        when_to_tilt_diagnostic(object(), fit, data.x_shift_src, weights)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        when_to_tilt_diagnostic(sm, object(), data.x_shift_src, weights)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        when_to_tilt_diagnostic(sm, fit, data.x_shift_src, weights, tilt_fit_check=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        when_to_tilt_diagnostic(sm, fit, data.x_shift_src, np.zeros(weights.size))
    with pytest.raises(ValueError):
        when_to_tilt_diagnostic(sm, fit, data.x_shift_src, weights, min_mode_signal_sd=-1.0)
    with pytest.raises(ValueError):
        when_to_tilt_diagnostic(
            sm, fit, data.x_shift_src, weights, max_tilt_coefficient_spread=float("nan")
        )


def test_run_rejects_bad_input() -> None:
    data = synthetic_bimodal_shift(
        1.0,
        1.0,
        1.2,
        n_train=200,
        n_shift=100,
        m_target=100,
        n_cal=100,
        n_test=100,
        n_surrogate_pool=100,
        seed=2,
    )
    with pytest.raises(TypeError):
        run_extra_replication(object())  # type: ignore[arg-type]
    res = run_extra_replication(data)
    with pytest.raises(TypeError):
        surrogate_coverage(data, object(), seed=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_extra_replication(data, alpha=1.5)
    cov_w, cov_t = surrogate_coverage(data, res, seed=0, n_draws=300)
    assert 0.0 <= cov_w <= 1.0 and 0.0 <= cov_t <= 1.0


# ---------------------------------------------------------------------------
# Bench contract (proper scores only, deterministic)
# ---------------------------------------------------------------------------

BENCH_KEYS = (
    "synthetic_coverage_standard_cp",
    "synthetic_mean_length_standard_cp",
    "synthetic_coverage_extra_wcp",
    "synthetic_mean_length_extra_wcp",
    "synthetic_coverage_extra_wcp_t",
    "synthetic_mean_length_extra_wcp_t",
    "synthetic_paired_length_reduction_percent",
    "synthetic_coverage_diff_wcp_t_minus_wcp",
    "synthetic_a_hat_mean",
    "synthetic_b_hat_mean",
    "synthetic_b_hat_spread_mean",
    "synthetic_mode_signal_sd",
    "synthetic_weight_ess_percent",
    "synthetic_replications",
    "synthetic_n",
    "synthetic_alpha",
    "synthetic_eta",
    "synthetic_a_star",
    "synthetic_b_star",
    "synthetic_seed",
)


def test_bench_extra_tilt_keys_proper_scores_only() -> None:
    row = bench_extra_tilt(
        eta=1.0,
        replications=2,
        seed=21,
        n_train=600,
        n_shift=600,
        m_target=800,
        n_cal=500,
        n_test=800,
    )
    for key in BENCH_KEYS:
        assert key in row
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in row:
        assert not (set(key.lower().split("_")) & forbidden)
    assert row["synthetic_dgp"] == "fixture"
    assert row["synthetic_claim"] == "research_metric_only"
    assert float(row["synthetic_n"]) > 0.0
    assert float(row["synthetic_coverage_extra_wcp_t"]) > 0.0
    with pytest.raises(ValueError):
        bench_extra_tilt(replications=0)
    with pytest.raises(ValueError):
        bench_extra_tilt(alpha=0.0)


def test_bench_extra_tilt_deterministic() -> None:
    kwargs = {
        "eta": 0.0,
        "replications": 2,
        "seed": 23,
        "n_train": 500,
        "n_shift": 500,
        "m_target": 600,
        "n_cal": 400,
        "n_test": 600,
    }
    assert bench_extra_tilt(**kwargs) == bench_extra_tilt(**kwargs)
