"""Target alignment, dilution and cautious forecast selection — identity + SYNTHETIC tests.

Covers ``quant_fund.metrics.forecast_selection``, an implementation of
Soleimani (2026), "Target alignment, dilution and forecast selection when
cross-sectional forecasts share a common target", arXiv:2609.26303v1
[econ.EM] (citation verified via fetch of the arXiv abstract page and v1 full
text). Proper-score diagnostics only (relative-score risk under the paper's
estimand, IC-type correlations, admission statistics); never Sharpe/P&L/NAV.

SYNTHETIC correctness tests only (AGENTS.md honesty contract #2): the planted
world (k aligned forecasters + m correlated zero-alignment diluters) is a
fixture, never market evidence. Determinism is pinned (seeded streams,
bit-identical repeats). Fail-closed edges asserted throughout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.forecast_selection import (
    MIN_SUPPORT,
    NO_INFORMATION_RISK,
    AlignmentDecomposition,
    alignment_decomposition,
    attainable_risk,
    calibrated_composite_losses,
    cautious_selection,
    composite_losses,
    dilution_accounting,
    error_correlation_mirror,
    forecast_selection_benchmarks,
    planted_panel,
    relative_score_risk,
    risk_decomposition,
)
from quant_fund.models.ensemble import equal_weights
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False

N_ALIGNED = 6
N_DILUTERS = 16
N_ALL = N_ALIGNED + N_DILUTERS
ALIGNED_POOL = tuple(range(N_ALIGNED))
DILUTERS = tuple(range(N_ALIGNED, N_ALL))
PLANTED_SEED = 20260922


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _small_random_decomp(seed: int = 7, n_dates: int = 64, n_f: int = 5, m: int = 30):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_dates, n_f, m))
    target = rng.normal(size=(n_dates, m))
    return alignment_decomposition(scores, target)


def _zero_alignment_panel(seed: int = 99, n_dates: int = 4, n_f: int = 4, m: int = 30):
    """Raw panel whose scores are exactly orthogonal to the target by construction.

    Rows are pre-standardized (zero mean, unit population norm under the 1/M
    inner product), so ``alignment_decomposition``'s standardization is a
    fixed point and gamma vanishes to machine precision.
    """
    rng = np.random.default_rng(seed)
    targets = []
    panels = []
    for _ in range(n_dates):
        y0 = np.arange(m, dtype=float) + 0.5 * rng.normal(size=m)
        y0 -= y0.mean()
        y0 /= math.sqrt(float(y0 @ y0) / m)
        basis = [y0]
        cols = []
        for _ in range(n_f):
            v = rng.normal(size=m)
            v -= v.mean()
            for b in basis:
                v = v - b * (float(b @ v) / m)
            v -= v.mean()
            v /= math.sqrt(float(v @ v) / m)
            basis.append(v)
            cols.append(v)
        targets.append(y0)
        panels.append(np.stack(cols))
    return np.stack(panels, axis=0), np.stack(targets, axis=0)


def _orthonormal_single_date_panel(
    seed: int = 1234, m: int = 30, n_pool: int = 3, n_batch: int = 2, batch_alignment: float = 0.2
):
    """One-date panel with exact Prop-9 bound side conditions (g_bar_P = 0, q_PA = 0).

    Pool forecasters are orthonormal and target-orthogonal; batch forecasters
    are ``a*y + sqrt(1-a^2)*v`` with ``v`` orthonormal to the pool, giving
    exact alignment ``a`` and ``q_PA = 0`` to machine precision.
    """
    rng = np.random.default_rng(seed)
    y0 = rng.normal(size=m)
    y0 -= y0.mean()
    y0 /= math.sqrt(float(y0 @ y0) / m)

    def _next_orthonormal(basis: list[np.ndarray]) -> np.ndarray:
        v = rng.normal(size=m)
        v -= v.mean()
        for b in basis:
            v = v - b * (float(b @ v) / m)
        v -= v.mean()
        v /= math.sqrt(float(v @ v) / m)
        basis.append(v)
        return v

    basis = [y0]
    vecs = [_next_orthonormal(basis) for _ in range(n_pool + n_batch)]
    pool = vecs[:n_pool]
    a = float(batch_alignment)
    batch = [a * y0 + math.sqrt(1.0 - a * a) * vecs[n_pool + j] for j in range(n_batch)]
    scores = np.stack(pool + batch)[None, :, :]
    return scores, y0[None, :], a


@pytest.fixture(scope="module")
def planted() -> AlignmentDecomposition:
    """Seeded SYNTHETIC planted world: 6 aligned + 16 correlated zero-alignment diluters."""
    scores, y = planted_panel(seed=PLANTED_SEED, alignment=0.06, cluster_corr=0.30)
    return alignment_decomposition(scores, y)


@pytest.fixture(scope="module")
def bench() -> dict[str, float]:
    return forecast_selection_benchmarks(seed=PLANTED_SEED)


def _full_equal_weights(n: int = N_ALL) -> np.ndarray:
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Proposition 1 / Definition 1: exact orthogonal split identities
# ---------------------------------------------------------------------------


def test_decomposition_reconstructs_scores_exactly(planted: AlignmentDecomposition) -> None:
    # s_it = gamma_it * y_t + u_it exactly (brief: residual ~1e-15).
    residual = np.abs(planted.aligned + planted.orthogonal - planted.scores).max()
    assert residual < 1e-13
    assert np.abs(planted.orthogonal - (planted.scores - planted.aligned)).max() < 1e-15


def test_aligned_projection_equals_forecast_target_correlation(
    planted: AlignmentDecomposition,
) -> None:
    # <aligned_it, y_t>_t == gamma_it == corr(s_it, y_t) == <s_it, y_t>_t.
    # Note: the aligned component is gamma_it * y_t, a scalar multiple of the
    # target, so its *Pearson correlation* with y_t is sign(gamma); the exact
    # identity is at the projection-coefficient / inner-product level, and
    # gamma_it is itself the forecast's own correlation with the target.
    m = planted.n_units
    inner_aligned = np.einsum("tnm,tm->tn", planted.aligned, planted.target) / m
    inner_scores = np.einsum("tnm,tm->tn", planted.scores, planted.target) / m
    assert np.abs(inner_aligned - planted.gamma).max() < 1e-14
    assert np.abs(inner_scores - planted.gamma).max() < 1e-14
    # gamma equals the per-date Pearson correlation of the raw score rows.
    for t in range(0, planted.n_dates, 97):
        for i in range(planted.n_forecasters):
            pearson = np.corrcoef(planted.scores[t, i], planted.target[t])[0, 1]
            assert pearson == pytest.approx(float(planted.gamma[t, i]), abs=1e-12)
    # standardized rows are zero-mean and unit-norm under <.,.>_t (Eq. (1)).
    assert np.abs(planted.scores.mean(axis=2)).max() < 1e-13
    assert np.abs(np.einsum("tnm,tnm->tn", planted.scores, planted.scores) / m - 1.0).max() < 1e-12
    assert np.abs(np.einsum("tm,tm->t", planted.target, planted.target) / m - 1.0).max() < 1e-12


def test_orthogonal_component_orthogonal_by_construction(
    planted: AlignmentDecomposition,
) -> None:
    m = planted.n_units
    inner = np.einsum("tnm,tm->tn", planted.orthogonal, planted.target) / m
    assert np.abs(inner).max() < 1e-14
    # ||u_it||_t^2 = 1 - gamma_it^2 (Proposition 1).
    norm2 = np.einsum("tnm,tnm->tn", planted.orthogonal, planted.orthogonal) / m
    assert np.abs(norm2 - (1.0 - planted.gamma**2)).max() < 1e-12
    # u_it inherits zero mean (s and y are both centered).
    assert np.abs(planted.orthogonal.mean(axis=2)).max() < 1e-13


def test_gram_split_and_psd(planted: AlignmentDecomposition) -> None:
    # C_t = gamma_t gamma_t' + K_t with K_t PSD (Proposition 1).
    outer = planted.gamma[:, :, None] * planted.gamma[:, None, :]
    assert np.abs(planted.forecast_corr - (outer + planted.orthogonal_cov)).max() < 1e-13
    assert np.abs(planted.orthogonal_cov - (planted.forecast_corr - outer)).max() < 1e-13
    n = planted.n_forecasters
    diag = np.arange(n)
    assert np.abs(planted.forecast_corr[:, diag, diag] - 1.0).max() < 1e-12
    k_diag = planted.orthogonal_cov[:, diag, diag]
    assert np.abs(k_diag - (1.0 - planted.gamma**2)).max() < 1e-12
    eig_min = np.linalg.eigvalsh(planted.orthogonal_cov).min()
    assert eig_min >= -1e-12


# ---------------------------------------------------------------------------
# Propositions 2, 3, 6 + Corollaries 1, 2: risk geometry
# ---------------------------------------------------------------------------


def test_risk_decomposition_identities(planted: AlignmentDecomposition) -> None:
    rng = np.random.default_rng(3)
    for _ in range(4):
        w = rng.normal(size=N_ALL)  # any real weights: affine, convex, rescaled
        rd = risk_decomposition(planted, w)
        risk = rd["risk"]
        g, q = rd["g_w"], rd["q_w"]
        # Proposition 2: R(w) = 1 - 2 g_w + q_w = alignment + orthogonal terms.
        assert risk == pytest.approx(1.0 - 2.0 * g + q, abs=1e-10)
        assert risk == pytest.approx(
            rd["alignment_term_within"] + rd["orthogonal_term_within"], abs=1e-10
        )
        # Proposition 6: pooled form (1 - g_w)^2 + w' K_pool w.
        assert risk == pytest.approx(
            rd["alignment_term_pooled"] + rd["orthogonal_term_pooled"], abs=1e-10
        )
        # Proposition 3: scale split R = (1 - rho^2) + q (1 - c)^2.
        assert risk == pytest.approx(
            rd["scale_free_risk"] + rd["scale_mismatch_penalty"], abs=1e-10
        )
        assert rd["scale_free_risk"] == pytest.approx(1.0 - rd["rho2_w"], abs=1e-14)
        # Corollary 1: R - 1 = -2 m with margin m = g_w - q_w / 2.
        assert (risk - 1.0) == pytest.approx(-2.0 * rd["margin_vs_no_information"], abs=1e-12)
        assert (risk < 1.0) == (rd["margin_vs_no_information"] > 0.0)


def test_no_information_forecast_has_unit_risk(planted: AlignmentDecomposition) -> None:
    w0 = np.zeros(N_ALL)
    assert relative_score_risk(planted, w0) == pytest.approx(NO_INFORMATION_RISK, abs=1e-14)
    # A composite that cancels exactly (duplicate forecasters, w = e0 - e1).
    scores = np.zeros((40, 2, MIN_SUPPORT))
    rng = np.random.default_rng(11)
    scores[:, 0] = rng.normal(size=(40, MIN_SUPPORT))
    scores[:, 1] = scores[:, 0]
    dup = alignment_decomposition(scores, rng.normal(size=(40, MIN_SUPPORT)))
    assert relative_score_risk(dup, np.array([1.0, -1.0])) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="nonpositive squared norm"):
        calibrated_composite_losses(dup, np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="nonpositive squared norm"):
        risk_decomposition(dup, np.array([1.0, -1.0]))


def test_equal_weight_margin_corollary1(planted: AlignmentDecomposition) -> None:
    n = N_ALL
    w = equal_weights(n)
    rd = risk_decomposition(planted, w)
    cbar = planted.mean_forecast_corr
    gbar = planted.mean_gamma
    ones = np.ones(n)
    margin = float(gbar.mean()) - 0.5 * float(ones @ cbar @ ones) / n**2
    # Corollary 1: R(1/N) - 1 = -2 m exactly.
    assert (rd["risk"] - 1.0) == pytest.approx(-2.0 * margin, abs=1e-12)
    # q_EW = rho_bar + (1 - rho_bar)/N with rho_bar the mean off-diagonal corr.
    off = ~np.eye(n, dtype=bool)
    rho_bar = float(cbar[off].mean())
    assert rd["q_w"] == pytest.approx(rho_bar + (1.0 - rho_bar) / n, abs=1e-12)
    # Beating no-information iff mean alignment exceeds half the squared norm.
    assert (rd["risk"] < 1.0) == (float(gbar.mean()) > 0.5 * (rho_bar + (1.0 - rho_bar) / n))


def test_pooled_orthogonal_covariance_prop6(planted: AlignmentDecomposition) -> None:
    gbar = planted.gamma.mean(axis=0)
    gram = np.einsum("ti,tj->ij", planted.gamma, planted.gamma) / planted.n_dates
    cov_a = gram - np.outer(gbar, gbar)  # Cov_a(gamma_t), uniform weights
    k_pool = planted.pooled_orthogonal_cov
    assert np.abs(k_pool - (planted.mean_forecast_corr - np.outer(gbar, gbar))).max() < 1e-13
    assert np.abs(k_pool - (planted.mean_orthogonal_cov + cov_a)).max() < 1e-13


def test_attainable_risk_corollary2(planted: AlignmentDecomposition) -> None:
    att = attainable_risk(planted)
    value = float(att["attainable_risk"])
    v_star = np.asarray(att["optimal_weights"])
    assert 0.0 <= value <= 1.0
    assert relative_score_risk(planted, v_star) == pytest.approx(value, abs=1e-6)
    # v* minimizes the pooled quadratic form: no random or structured w beats it.
    rng = np.random.default_rng(5)
    trials = [rng.normal(size=N_ALL) for _ in range(5)]
    trials.append(equal_weights(N_ALL))
    trials.extend(np.eye(N_ALL)[i] for i in range(N_ALL))
    for w in trials:
        assert relative_score_risk(planted, v_star) <= relative_score_risk(planted, w) + 1e-9


def test_calibrated_losses_average_to_scale_free_risk(planted: AlignmentDecomposition) -> None:
    # In-sample plug-in scale: mean_t ||c_w s_w,t - y_t||^2 = 1 - rho_w^2 exactly.
    rng = np.random.default_rng(9)
    w = rng.normal(size=N_ALL)
    rd = risk_decomposition(planted, w)
    ell = calibrated_composite_losses(planted, w)
    assert float(ell.mean()) == pytest.approx(rd["scale_free_risk"], abs=1e-12)


# ---------------------------------------------------------------------------
# Proposition 5: deviation correlation is a translation of forecast correlation
# ---------------------------------------------------------------------------


def test_deviation_correlation_formula_prop5(planted: AlignmentDecomposition) -> None:
    gamma = planted.gamma
    num = 1.0 + planted.forecast_corr - gamma[:, :, None] - gamma[:, None, :]
    den = 2.0 * np.sqrt((1.0 - gamma[:, :, None]) * (1.0 - gamma[:, None, :]))
    assert np.abs(planted.deviation_corr - num / den).max() < 1e-13
    n = planted.n_forecasters
    diag = np.arange(n)
    assert np.abs(planted.deviation_corr[:, diag, diag] - 1.0).max() < 1e-12


def test_deviation_correlation_zero_alignment_translation() -> None:
    scores, target = _zero_alignment_panel()
    decomp = alignment_decomposition(scores, target)
    # gamma vanishes to machine precision by construction...
    assert np.abs(decomp.gamma).max() < 1e-12
    # ...so rho^e = (1 + rho^s)/2 exactly (Proposition 5 at zero alignment).
    translation = (1.0 + decomp.forecast_corr) / 2.0
    assert np.abs(decomp.deviation_corr - translation).max() < 1e-10


# ---------------------------------------------------------------------------
# Propositions 8-9: admission algebra and the dilution split
# ---------------------------------------------------------------------------


def test_admission_algebra_prop8(planted: AlignmentDecomposition) -> None:
    pool = ALIGNED_POOL
    batch = (6, 7, 8)
    n, m = len(pool), len(batch)
    acc = dilution_accounting(planted, pool, batch)
    # Proposition 8 union identity: V_{P u A} = [n^2 V_P + m^2 V_A + 2nm C^e]/(n+m)^2.
    closed = (n * n * acc.v_pool + m * m * acc.v_batch + 2.0 * n * m * acc.deviation_cov) / (
        (n + m) ** 2
    )
    assert acc.v_union == pytest.approx(closed, abs=1e-10)
    assert acc.delta == pytest.approx(acc.v_union - acc.v_pool, abs=1e-14)
    # V_P agrees with the direct risk of the pool's equal-weight composite.
    w_p = np.zeros(N_ALL)
    w_p[list(pool)] = equal_weights(n)
    assert acc.v_pool == pytest.approx(relative_score_risk(planted, w_p), abs=1e-12)
    # Single-candidate form and Lambda^EW equivalence.
    for j in (6, 11, 21):
        one = dilution_accounting(planted, pool, (j,))
        lam = one.lambda_ew
        assert math.isfinite(lam)
        assert one.delta == pytest.approx(
            (2.0 * n + 1.0) * one.v_pool * (lam - 1.0) / (n + 1.0) ** 2, abs=1e-12
        )
        assert (lam < 1.0) == (one.delta < 0.0)
        assert one.admitted_by_equal_weight == (one.delta < 0.0)


def test_dilution_split_prop9(planted: AlignmentDecomposition) -> None:
    pool = ALIGNED_POOL
    batch = (6, 7, 8)
    n, m = len(pool), len(batch)
    acc = dilution_accounting(planted, pool, batch)
    # Proposition 9: Delta = Delta^sf + Delta^scale exactly.
    assert acc.delta == pytest.approx(acc.delta_scale_free + acc.delta_scale_mismatch, abs=1e-10)
    # Equation (6): rho^2_{P u A} from block moments.
    gbar = planted.mean_gamma
    cbar = planted.mean_forecast_corr
    w_p = np.zeros(N_ALL)
    w_p[list(pool)] = equal_weights(n)
    w_a = np.zeros(N_ALL)
    w_a[list(batch)] = equal_weights(m)
    g_p = float(w_p @ gbar)
    q_p = float(w_p @ cbar @ w_p)
    q_a = float(w_a @ cbar @ w_a)
    q_pa = float(w_p @ cbar @ w_a)
    g_bar_a = float(gbar[list(batch)].mean())
    rho2_manual = (n * g_p + m * g_bar_a) ** 2 / (n * n * q_p + m * m * q_a + 2.0 * n * m * q_pa)
    assert acc.rho2_union == pytest.approx(rho2_manual, abs=1e-9)
    # Proposition 3 scale split holds for both composites.
    rd_p = risk_decomposition(planted, w_p)
    w_u = np.zeros(N_ALL)
    w_u[list(pool) + list(batch)] = 1.0 / (n + m)
    rd_u = risk_decomposition(planted, w_u)
    assert acc.v_pool == pytest.approx(
        (1.0 - acc.rho2_pool) + rd_p["scale_mismatch_penalty"], abs=1e-10
    )
    assert acc.v_union == pytest.approx(
        (1.0 - acc.rho2_union) + rd_u["scale_mismatch_penalty"], abs=1e-10
    )
    assert acc.scale_pool == pytest.approx(rd_p["c_w"], abs=1e-12)
    assert acc.scale_union == pytest.approx(rd_u["c_w"], abs=1e-12)
    assert acc.delta_scale_free == pytest.approx(rd_p["rho2_w"] - rd_u["rho2_w"], abs=1e-12)


def test_dilution_bound_prop9_exact_side_conditions() -> None:
    # Hand-built one-date panel with g_bar_P = 0 and q_PA = 0 exactly, so the
    # Proposition 9 bound -Delta^sf <= (m g_bar_A)^2 / (n^2 q_P) applies.
    scores, target, a = _orthonormal_single_date_panel(n_pool=3, n_batch=2, batch_alignment=0.2)
    decomp = alignment_decomposition(scores, target)
    acc = dilution_accounting(decomp, (0, 1, 2), (3, 4))
    n, m = 3, 2
    assert np.abs(decomp.mean_gamma[[0, 1, 2]]).max() < 1e-12  # g_bar_P = 0
    # Pool is orthonormal: q_P = 1/n exactly; batch pairwise corr = a^2.
    assert acc.rho2_pool == pytest.approx(0.0, abs=1e-18)
    expected_gain = (m * a) ** 2 / (n * n / n + m * m * (1.0 + a * a) / 2.0)
    assert -acc.delta_scale_free == pytest.approx(expected_gain, abs=1e-10)
    assert acc.scale_free_bound == pytest.approx((m * a) ** 2 / (n * n * (1.0 / n)), abs=1e-12)
    assert -acc.delta_scale_free <= acc.scale_free_bound + 1e-12
    # Genuine improvement is detected here (aligned batch): sf term negative.
    assert acc.genuine_improvement
    assert not acc.pure_dilution


# ---------------------------------------------------------------------------
# Fail-closed edges
# ---------------------------------------------------------------------------


def test_alignment_decomposition_fail_closed() -> None:
    rng = np.random.default_rng(21)
    good_s = rng.normal(size=(8, 3, MIN_SUPPORT))
    good_y = rng.normal(size=(8, MIN_SUPPORT))
    with pytest.raises(ValueError, match="T x N x M"):
        alignment_decomposition(rng.normal(size=(8, MIN_SUPPORT)), good_y)
    with pytest.raises(ValueError, match="T x M"):
        alignment_decomposition(good_s, rng.normal(size=MIN_SUPPORT))
    with pytest.raises(ValueError, match="shape"):
        alignment_decomposition(good_s, rng.normal(size=(8, MIN_SUPPORT + 1)))
    nan_s = good_s.copy()
    nan_s[2, 1, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        alignment_decomposition(nan_s, good_y)
    inf_y = good_y.copy()
    inf_y[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        alignment_decomposition(good_s, inf_y)
    with pytest.raises(ValueError, match="common support"):
        alignment_decomposition(
            rng.normal(size=(8, 3, MIN_SUPPORT - 1)), rng.normal(size=(8, MIN_SUPPORT - 1))
        )
    const_y = np.ones((8, MIN_SUPPORT))
    with pytest.raises(ValueError, match="no valid dates"):
        alignment_decomposition(good_s, const_y)
    # A forecaster that reproduces the target has |gamma| = 1: Prop 5 undefined.
    perfect = good_s.copy()
    perfect[5, 0, :] = good_y[5, :] * 3.0 + 7.0
    with pytest.raises(ValueError, match="perfectly aligned"):
        alignment_decomposition(perfect, good_y)
    with pytest.raises(ValueError, match="eps"):
        alignment_decomposition(good_s, good_y, eps=0.0)


def test_degenerate_dates_are_excluded_fixed_membership() -> None:
    rng = np.random.default_rng(33)
    t, n, m = 12, 3, MIN_SUPPORT + 4
    scores = rng.normal(size=(t, n, m))
    target = rng.normal(size=(t, m))
    target[3, :] = 5.0  # degenerate target date
    scores[10, 2, :] = -3.0  # degenerate forecaster date (fixed membership drops it)
    decomp = alignment_decomposition(scores, target)
    assert decomp.n_dates == t - 2
    assert decomp.n_excluded_dates == 2
    assert not decomp.valid_mask[3] and not decomp.valid_mask[10]
    assert decomp.valid_mask.sum() == t - 2


def test_index_and_parameter_validation_fail_closed(planted: AlignmentDecomposition) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        dilution_accounting(planted, (0, 1), (1, 2))
    with pytest.raises(ValueError, match="nonempty"):
        dilution_accounting(planted, (), (1,))
    with pytest.raises(ValueError, match="repeat"):
        dilution_accounting(planted, (0, 0), (1,))
    with pytest.raises(ValueError, match="out of range"):
        dilution_accounting(planted, (0,), (N_ALL,))
    with pytest.raises(ValueError, match="integer"):
        dilution_accounting(planted, (True,), (1,))
    with pytest.raises(ValueError, match="finite vector"):
        relative_score_risk(planted, np.full(N_ALL, np.nan))
    with pytest.raises(ValueError, match="length"):
        relative_score_risk(planted, np.ones(N_ALL - 1))
    with pytest.raises(ValueError, match="basis"):
        cautious_selection(planted, basis="bogus")
    with pytest.raises(ValueError, match="decision_rule"):
        cautious_selection(planted, decision_rule="bogus")
    with pytest.raises(ValueError, match="alpha"):
        cautious_selection(planted, alpha=0.7)
    with pytest.raises(ValueError, match="delta"):
        cautious_selection(planted, delta=-0.01)
    with pytest.raises(ValueError, match="n_draws"):
        cautious_selection(planted, n_draws=10)
    with pytest.raises(ValueError, match="seed"):
        cautious_selection(planted, seed=-1)
    with pytest.raises(ValueError, match="integer"):
        cautious_selection(planted, seed=True)
    with pytest.raises(ValueError, match="eigen_rel_tol"):
        cautious_selection(planted, eigen_rel_tol=0.0)
    with pytest.raises(ValueError, match="nw_lag"):
        cautious_selection(planted, nw_lag=-1)
    with pytest.raises(ValueError, match="nw_lag"):
        cautious_selection(planted, nw_lag=planted.n_dates)
    with pytest.raises(ValueError, match="max_steps"):
        cautious_selection(planted, max_steps=-1)
    with pytest.raises(ValueError, match="at least two forecasters"):
        error_correlation_mirror(_small_random_decomp(n_f=1))


def test_planted_panel_fail_closed() -> None:
    with pytest.raises(ValueError, match="seed"):
        planted_panel(seed=-1)
    with pytest.raises(ValueError, match="planted world requires"):
        planted_panel(n_units=MIN_SUPPORT - 1)
    with pytest.raises(ValueError, match="planted world requires"):
        planted_panel(n_dates=5)
    with pytest.raises(ValueError, match="planted world requires"):
        planted_panel(n_aligned=1)
    with pytest.raises(ValueError, match="alignment"):
        planted_panel(alignment=1.5)
    with pytest.raises(ValueError, match="cluster_corr"):
        planted_panel(cluster_corr=1.0)


def test_single_forecaster_selection_and_weights(planted: AlignmentDecomposition) -> None:
    single = _small_random_decomp(n_f=1)
    res = cautious_selection(single, seed=1)
    assert res.selected == (0,)
    assert res.steps == ()
    w0 = np.zeros(N_ALL)
    w0[3] = 1.0
    assert relative_score_risk(planted, w0) == pytest.approx(
        float(composite_losses(planted, w0).mean()), abs=1e-15
    )


# ---------------------------------------------------------------------------
# SYNTHETIC validation: the paper's failure mode and headline
# ---------------------------------------------------------------------------


def test_equal_weight_admission_rewards_diluters(planted: AlignmentDecomposition) -> None:
    """Failure mode (Prop. 9 + Sec. 6/9.4): zero-alignment high-variance diluters
    clear equal-weight admission on scale mismatch alone, with no genuine
    improvement — Delta < 0 and Lambda^EW < 1 while Delta^sf > 0."""
    for j in DILUTERS:
        acc = dilution_accounting(planted, ALIGNED_POOL, (j,))
        assert acc.delta < 0.0, f"diluter {j} should clear equal-weight admission"
        assert acc.lambda_ew < 1.0
        assert acc.admitted_by_equal_weight
        assert acc.delta_scale_free > 0.0, f"diluter {j}: pooled correlation must not rise"
        assert acc.delta_scale_mismatch < 0.0
        assert acc.pure_dilution
        assert not acc.genuine_improvement
        assert acc.rho2_union < acc.rho2_pool
        # Diluters carry maximal individual loss 2(1 - gamma) ~= 2 (high variance,
        # zero alignment) yet look "risk-reducing" to equal-weight admission.
        assert acc.v_batch == pytest.approx(2.0, abs=0.05)
    # Aligned members, by contrast, are genuine improvements.
    for j in ALIGNED_POOL[1:]:
        pool_wo_j = tuple(i for i in ALIGNED_POOL if i != j)
        acc = dilution_accounting(planted, pool_wo_j, (j,))
        assert acc.delta_scale_free < 0.0
        assert acc.genuine_improvement
        assert not acc.pure_dilution


def test_error_correlation_mirrors_forecast_correlation(planted: AlignmentDecomposition) -> None:
    """Consequence (a) / Prop. 5 / Sec. 9.2: deviation correlation is an affine
    translation of forecast correlation, hence a poor diversity measure."""
    mirror = error_correlation_mirror(planted)
    # Pair-level (time-averaged) mirror: near-perfect affine relation (paper: R^2 = 0.9999).
    assert mirror["pair_mirror_affine_r2"] > 0.99
    assert mirror["pair_mirror_pearson"] > 0.99
    # Mean deviation correlation sits within O(gamma) of the zero-alignment
    # benchmark (1 + rho^s)/2 (paper Table 8: within 0.005 at alignment 0.006).
    assert abs(mirror["pair_translation_gap"]) < 0.03
    assert abs(mirror["translation_gap"]) < 0.03
    # A quarter of forecast correlations are negative, yet no deviation
    # correlation is: the translation compresses [-1, 1] into [0, 1].
    assert mirror["negative_forecast_corr_share"] > 0.05
    assert mirror["negative_deviation_corr_share"] == 0.0
    assert mirror["min_deviation_corr"] > 0.0
    # Date-level mirror also strongly positive (within-date sampling noise only).
    assert mirror["mirror_pearson"] > 0.8


def test_cautious_scale_free_selection_removes_dilution_loss(
    planted: AlignmentDecomposition,
) -> None:
    """Paper headline: selection removes most of the dilution loss carried by
    the full equal-weight pool — and the scale-free basis never rewards the
    planted diluters (their Delta^sf > 0)."""
    risk_full = relative_score_risk(planted, _full_equal_weights())
    assert risk_full > NO_INFORMATION_RISK  # the diluted pool sits above no-information
    sf = cautious_selection(planted, basis="scale_free", seed=PLANTED_SEED + 1)
    assert set(sf.selected) == set(ALIGNED_POOL)  # recovers exactly the aligned pool
    assert all(j not in sf.selected for j in DILUTERS)
    removal = (risk_full - sf.final_risk) / (risk_full - NO_INFORMATION_RISK)
    assert removal >= 0.6, f"scale-free selection should remove most dilution loss, got {removal}"
    assert sf.final_risk >= NO_INFORMATION_RISK  # nothing beats no-information here
    # Every step's diluter decisions are non-admissions.
    for step in sf.steps:
        for pos, j in enumerate(step.candidates):
            if j in DILUTERS:
                assert step.decision[pos] in ("reject", "undecided")


def test_three_way_equal_weight_and_point_estimate_admit_diluters(
    planted: AlignmentDecomposition,
) -> None:
    """On the equal-weight basis even the three-way rule admits pure dilution
    (paper Table 10: admitted 6/6 with Delta^sf ~= 0); the point-estimate foil
    does the same without bands. Dilution lowers risk toward no-information
    but strictly lowers the pooled squared correlation (skill)."""
    ew = cautious_selection(planted, seed=PLANTED_SEED + 1)
    naive = cautious_selection(planted, decision_rule="point_estimate", seed=PLANTED_SEED + 1)
    ew_diluters = [j for j in ew.selected if j in DILUTERS]
    naive_diluters = [j for j in naive.selected if j in DILUTERS]
    assert len(ew_diluters) >= 1, "equal-weight three-way rule should reward the diluters"
    assert len(naive_diluters) >= 1
    assert all(j in ALIGNED_POOL for j in ew.selected[:N_ALIGNED])
    sf = cautious_selection(planted, basis="scale_free", seed=PLANTED_SEED + 1)
    # Dilution trades skill for norm shrinkage: rho^2 falls, risk drifts down
    # toward (never below) the no-information benchmark.
    assert ew.final_rho2 < sf.final_rho2
    assert naive.final_rho2 < sf.final_rho2
    assert ew.final_risk < sf.final_risk
    assert ew.final_risk >= NO_INFORMATION_RISK
    assert naive.final_risk >= NO_INFORMATION_RISK
    # The equal-weight three-way pool still removes most of the full pool's loss.
    risk_full = relative_score_risk(planted, _full_equal_weights())
    assert (risk_full - ew.final_risk) / (risk_full - NO_INFORMATION_RISK) >= 0.6
    # Admitted diluters carried no genuine improvement at their admission step.
    for j in ew_diluters:
        pool_before = tuple(
            i for i in ew.selected if i != j and ew.selected.index(i) < ew.selected.index(j)
        )
        acc = dilution_accounting(planted, pool_before, (j,))
        assert acc.delta_scale_free > 0.0
        assert acc.pure_dilution


def test_cautious_selection_reported_guarantee_and_min_detectable(
    planted: AlignmentDecomposition,
) -> None:
    res = cautious_selection(planted, seed=PLANTED_SEED + 1, alpha=0.05, delta=0.005)
    assert "Remark 1" in res.guarantee
    assert "0.950" in res.guarantee
    assert "0.0250" in res.guarantee
    z80 = 0.8416212335729143
    for step in res.steps:
        np.testing.assert_allclose(
            step.min_detectable, res.delta + (step.crit_value + z80) * step.se, rtol=1e-12
        )
        for pos, dec in enumerate(step.decision):
            band_hi = step.delta_hat[pos] + step.crit_value * step.se[pos]
            band_lo = step.delta_hat[pos] - step.crit_value * step.se[pos]
            if dec == "admit":
                assert band_hi < -res.delta
            elif dec == "reject":
                assert band_lo > 0.0
            else:
                assert band_hi >= -res.delta and band_lo <= 0.0
    assert res.nw_lag == int(math.floor(4.0 * (planted.n_dates / 100.0) ** (2.0 / 9.0)))


# ---------------------------------------------------------------------------
# Determinism + bench battery
# ---------------------------------------------------------------------------


def test_selection_and_panel_are_deterministic(planted: AlignmentDecomposition) -> None:
    a = cautious_selection(planted, seed=17)
    b = cautious_selection(planted, seed=17)
    assert a.selected == b.selected
    assert a.final_risk == b.final_risk
    assert len(a.steps) == len(b.steps)
    for sa, sb in zip(a.steps, b.steps, strict=True):
        assert sa.candidates == sb.candidates
        assert sa.decision == sb.decision
        assert sa.admitted == sb.admitted
        assert sa.crit_value == sb.crit_value
        assert sa.delta_hat.tobytes() == sb.delta_hat.tobytes()
        assert sa.se.tobytes() == sb.se.tobytes()
    s1, y1 = planted_panel(seed=5)
    s2, y2 = planted_panel(seed=5)
    assert np.array_equal(s1, s2) and np.array_equal(y1, y2)


def test_bench_battery_values_and_determinism(bench: dict[str, float]) -> None:
    again = forecast_selection_benchmarks(seed=PLANTED_SEED)
    assert bench == again  # bit-identical values on repeat
    assert bench["diluter_rewarded_share"] == 1.0
    assert bench["three_way_sf_diluters_admitted"] == 0.0
    assert bench["three_way_ew_diluters_admitted"] >= 1.0
    assert bench["point_estimate_diluters_admitted"] >= 1.0
    assert bench["three_way_sf_removal_fraction"] >= 0.6
    assert bench["three_way_ew_removal_fraction"] >= 0.6
    assert bench["diluter_delta_mean"] < 0.0
    assert bench["diluter_delta_scale_free_mean"] > 0.0
    assert bench["diluter_delta_scale_mismatch_mean"] < 0.0
    assert bench["diluter_lambda_ew_mean"] < 1.0
    assert bench["pair_deviation_mirror_affine_r2"] > 0.99
    assert bench["negative_deviation_corr_share"] == 0.0
    assert bench["negative_forecast_corr_share"] > 0.05
    # Paper headline on weak-alignment worlds: no combination beats no-information.
    for key in (
        "equal_weight_full_risk",
        "aligned_pool_risk",
        "three_way_ew_pool_risk",
        "three_way_sf_pool_risk",
        "point_estimate_pool_risk",
    ):
        assert bench[key] >= NO_INFORMATION_RISK
    assert bench["three_way_sf_pool_rho2"] > bench["three_way_ew_pool_rho2"]
    fast = forecast_selection_benchmarks(seed=PLANTED_SEED, fast=True)
    assert fast["n_dates"] == 800.0
    assert fast["diluter_rewarded_share"] == 1.0
    assert fast["three_way_sf_diluters_admitted"] == 0.0


def test_bench_blob_is_proper_scores_only(bench: dict[str, float]) -> None:
    # AGENTS.md honesty contract #1: no Sharpe/Sortino/Calmar/P&L/NAV keys.
    assert family_blob_forbidden_metrics_absent({"forecast_selection": bench})
    assert all(isinstance(v, float) and np.isfinite(v) for v in bench.values())
    assert bench["no_information_risk"] == 1.0
