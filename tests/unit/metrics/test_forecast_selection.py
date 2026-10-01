"""Wave-17 spec-matrix tests for ``metrics.forecast_selection`` (Soleimani 2026).

Covers ``quant_fund.metrics.forecast_selection``, an implementation of
Soleimani (2026), "Target alignment, dilution and forecast selection when
cross-sectional forecasts share a common target", arXiv:2609.26303v1
[econ.EM] (citation verified by fetching the arXiv abstract page). Proper-score
diagnostics only (relative-score risk under the paper's estimand, IC-type
correlations, admission statistics); never Sharpe/P&L/NAV.

This file complements ``tests/unit/core/test_forecast_selection.py`` with the
wave-17 lane's asserted matrix:

- standardization utilities exercised directly (per-date z-scoring under the
  paper's 1/M inner product, degenerate-dispersion date exclusion, and the
  already-standardized fixed point);
- the alignment/dilution decomposition's exact identities on *constructed*
  panels (graded planted alignments, AR(1)-across-dates panels) at 1e-12
  residuals — including the per-forecaster form
  ``mean_t ||s_kt - y_t||_t^2 = 2 (1 - gamma_bar_k)``;
- planted skill ordering recovered and deterministic;
- a paired circular block-bootstrap CI on *alignment differences* built on the
  composed primitive ``metrics.inference.bootstrap_mean_ci`` — the pairing is
  exact because the resampled series is the per-date difference
  ``gamma_i,t - gamma_j,t`` itself — with seeded Monte-Carlo coverage,
  detection, and a first-class ``no_significant_difference`` verdict;
- the no-skill cautious no-rank verdict: under zero planted skill the
  three-way rule almost never admits beyond the anchor forecaster (seeded MC
  share bounded), and ``undecided`` appears as a first-class decision;
- fail-closed edges and bit-identical determinism.

SYNTHETIC correctness tests only (AGENTS.md honesty contract #2): every panel
below is a seeded fixture, never market evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sstats

from quant_fund.metrics.forecast_selection import (
    MIN_SUPPORT,
    NO_INFORMATION_RISK,
    AlignmentDecomposition,
    alignment_decomposition,
    attainable_risk,
    calibrated_composite_losses,
    cautious_selection,
    composite_losses,
    composite_scores,
    dilution_accounting,
    error_correlation_mirror,
    forecast_selection_benchmarks,
    planted_panel,
    relative_score_risk,
    risk_decomposition,
)
from quant_fund.metrics.inference import bootstrap_mean_ci
from quant_fund.models.ensemble import equal_weights
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False

_SEED = 20260930
_IDENTITY_TOL = 1e-12


# ---------------------------------------------------------------------------
# SYNTHETIC panel builders (constructions disjoint from the core suite's)
# ---------------------------------------------------------------------------


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score under the paper's 1/M inner product (Eq. (1))."""
    z = np.asarray(x, dtype=float) - x.mean(axis=-1, keepdims=True)
    return np.asarray(z / z.std(axis=-1, keepdims=True), dtype=float)


def _ar1_along_dates(x: np.ndarray, phi: float) -> np.ndarray:
    """Stationary AR(1) filter along the date axis; preserves unit variance."""
    out = np.empty_like(x)
    out[0] = x[0]
    gain = math.sqrt(1.0 - phi * phi)
    for t in range(1, x.shape[0]):
        out[t] = phi * out[t - 1] + gain * x[t]
    return out


def _graded_panel(
    *,
    seed: int,
    n_dates: int,
    n_units: int,
    alignments: tuple[float, ...],
    phi: float = 0.0,
    n_diluters: int = 0,
    diluter_corr: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """SYNTHETIC panel where forecaster ``k`` has planted alignment ``alignments[k]``.

    ``s_kt = a_k y_t + sqrt(1 - a_k^2) u_kt`` with ``u`` iid across forecasters
    and target-orthogonal in expectation, so ``gamma_kt ~= a_k`` and
    ``E||s_kt - y_t||_t^2 = 2 (1 - a_k)``. ``phi > 0`` makes the per-date
    alignments autocorrelated across dates (exercises the block bootstrap).
    ``n_diluters`` appends zero-alignment forecasters sharing one cluster
    factor at pairwise correlation ``diluter_corr``.
    """
    rng = np.random.default_rng(seed)
    t, m = int(n_dates), int(n_units)
    y = rng.normal(size=(t, m))
    if phi > 0.0:
        y = _ar1_along_dates(y, phi)
    y_std = _zscore_rows(y)
    cols: list[np.ndarray] = []
    for a in alignments:
        u = rng.normal(size=(t, m))
        if phi > 0.0:
            u = _ar1_along_dates(u, phi)
        cols.append(float(a) * y_std + math.sqrt(1.0 - float(a) * float(a)) * u)
    if n_diluters > 0:
        z = rng.normal(size=(t, m))
        eps = rng.normal(size=(t, n_diluters, m))
        c = float(diluter_corr)
        for j in range(n_diluters):
            cols.append(math.sqrt(c) * z + math.sqrt(1.0 - c) * eps[:, j, :])
    return np.stack(cols, axis=1), np.ascontiguousarray(y)


def _graded_decomp(
    alignments: tuple[float, ...],
    *,
    seed: int = _SEED,
    n_dates: int = 400,
    n_units: int = 30,
    phi: float = 0.0,
    n_diluters: int = 0,
    diluter_corr: float = 0.4,
) -> AlignmentDecomposition:
    scores, y = _graded_panel(
        seed=seed,
        n_dates=n_dates,
        n_units=n_units,
        alignments=alignments,
        phi=phi,
        n_diluters=n_diluters,
        diluter_corr=diluter_corr,
    )
    return alignment_decomposition(scores, y)


def _pure_noise_decomp(
    seed: int, n_dates: int = 320, n_f: int = 6, m: int = 30
) -> AlignmentDecomposition:
    """Zero-skill panel: every forecaster is iid noise orthogonal in expectation."""
    scores, y = _graded_panel(
        seed=seed,
        n_dates=n_dates,
        n_units=m,
        alignments=tuple(0.0 for _ in range(n_f)),
    )
    return alignment_decomposition(scores, y)


def _paired_alignment_diff_ci(
    decomp: AlignmentDecomposition,
    i: int,
    j: int,
    *,
    n_boot: int = 500,
    alpha: float = 0.10,
    seed: int = 17,
    block: int | None = None,
) -> tuple[float, float, float]:
    """Paired circular block-bootstrap CI for ``E[gamma_i,t - gamma_j,t]``.

    Resampling dates of the *difference series* is exactly the paired
    comparison (each draw resamples the two forecasters' alignments at the
    same dates, preserving their cross-sectional dependence); the circular
    blocks preserve within-forecaster date autocorrelation. Composes
    ``metrics.inference.bootstrap_mean_ci``.
    """
    d = decomp.gamma[:, i] - decomp.gamma[:, j]
    return bootstrap_mean_ci(d, n_boot=n_boot, block=block, alpha=alpha, seed=seed)


def _cautious_pair_verdict(
    decomp: AlignmentDecomposition,
    i: int,
    j: int,
    *,
    alpha: float = 0.10,
    n_boot: int = 500,
    seed: int = 17,
) -> str:
    """Two-sided refusal verdict on an alignment difference.

    Returns ``'no_significant_difference'`` when the paired block-bootstrap CI
    covers zero — a first-class refusal rather than a forced ranking.
    """
    lo, hi, _ = _paired_alignment_diff_ci(decomp, i, j, alpha=alpha, n_boot=n_boot, seed=seed)
    if lo <= 0.0 <= hi:
        return "no_significant_difference"
    return "higher_alignment" if lo > 0.0 else "lower_alignment"


# ---------------------------------------------------------------------------
# Standardization utilities (spec item 1)
# ---------------------------------------------------------------------------


def test_standardization_centers_and_unit_norm() -> None:
    """Each date's scores and target are zero-mean, unit-norm under <.,.>_t."""
    decomp = _graded_decomp((0.12, 0.05, 0.0), n_dates=96)
    m = decomp.n_units
    assert np.abs(decomp.scores.mean(axis=2)).max() < _IDENTITY_TOL
    assert np.abs(decomp.target.mean(axis=1)).max() < _IDENTITY_TOL
    norm_s = np.einsum("tnm,tnm->tn", decomp.scores, decomp.scores) / m
    norm_y = np.einsum("tm,tm->t", decomp.target, decomp.target) / m
    assert np.abs(norm_s - 1.0).max() < _IDENTITY_TOL
    assert np.abs(norm_y - 1.0).max() < _IDENTITY_TOL
    # Population std (the 1/M convention, not ddof=1) is exactly one per row.
    assert np.abs(decomp.scores.std(axis=2) - 1.0).max() < _IDENTITY_TOL


def test_standardization_fixed_point_on_prestandardized_input() -> None:
    """Already-standardized rows are a fixed point of the Eq. (1) step."""
    scores, y = _graded_panel(seed=11, n_dates=64, n_units=30, alignments=(0.1, 0.0))
    s_pre = _zscore_rows(scores)
    y_pre = _zscore_rows(y)
    decomp = alignment_decomposition(s_pre, y_pre)
    assert np.abs(decomp.scores - s_pre).max() < _IDENTITY_TOL
    assert np.abs(decomp.target - y_pre).max() < _IDENTITY_TOL
    # gamma equals the per-date Pearson correlation of the raw rows.
    manual = np.einsum("tnm,tm->tn", s_pre, y_pre) / s_pre.shape[2]
    assert np.abs(decomp.gamma - manual).max() < _IDENTITY_TOL


def test_degenerate_forecaster_dispersion_date_excluded() -> None:
    """A date with a zero-variance forecaster row is dropped, not imputed."""
    scores, y = _graded_panel(seed=13, n_dates=60, n_units=30, alignments=(0.2, 0.1))
    scores[5, 1, :] = 3.0  # constant row at date 5
    decomp = alignment_decomposition(scores, y)
    assert decomp.n_excluded_dates == 1
    assert decomp.valid_mask.sum() == 59
    assert not decomp.valid_mask[5]
    assert decomp.n_dates == 59


def test_degenerate_target_dispersion_date_excluded() -> None:
    """A date with a constant target row is dropped under fixed membership."""
    scores, y = _graded_panel(seed=14, n_dates=60, n_units=30, alignments=(0.2, 0.1))
    y[2, :] = -1.5
    decomp = alignment_decomposition(scores, y)
    assert decomp.n_excluded_dates == 1
    assert decomp.n_dates == 59


# ---------------------------------------------------------------------------
# Decomposition identities at ~1e-12 on constructed panels (spec item 2)
# ---------------------------------------------------------------------------


def test_reconstruction_and_orthogonality_machine_precision() -> None:
    """s_it = gamma_it y_t + u_it, <u_it, y_t>_t = 0 — residuals ~1e-14."""
    decomp = _graded_decomp((0.3, 0.1, 0.0, 0.0), n_dates=80)
    residual = np.abs(decomp.aligned + decomp.orthogonal - decomp.scores).max()
    assert residual < 1e-13
    inner = np.einsum("tnm,tm->tn", decomp.orthogonal, decomp.target) / decomp.n_units
    assert np.abs(inner).max() < 1e-13
    # aligned_it is exactly the projection gamma_it * y_t.
    proj = decomp.gamma[:, :, None] * decomp.target[:, None, :]
    assert np.abs(decomp.aligned - proj).max() < 1e-15


def test_orthogonal_norm_and_gram_identities() -> None:
    """||u||_t^2 = 1 - gamma^2 and C_t = gamma gamma' + K_t with K_t PSD."""
    decomp = _graded_decomp((0.25, 0.1, 0.0), n_dates=80)
    norm_u = np.einsum("tnm,tnm->tn", decomp.orthogonal, decomp.orthogonal) / decomp.n_units
    assert np.abs(norm_u - (1.0 - decomp.gamma**2)).max() < _IDENTITY_TOL
    outer = decomp.gamma[:, :, None] * decomp.gamma[:, None, :]
    assert np.abs(decomp.forecast_corr - (outer + decomp.orthogonal_cov)).max() < _IDENTITY_TOL
    assert np.linalg.eigvalsh(decomp.mean_orthogonal_cov).min() >= -1e-10


def test_per_forecaster_mse_alignment_identity() -> None:
    """Spec identity: mean_t ||s_kt - y_t||^2 = 2 (1 - gamma_bar_k) exactly."""
    decomp = _graded_decomp((0.4, 0.15, 0.05), n_dates=120)
    gbar = decomp.mean_gamma
    for k in range(decomp.n_forecasters):
        w = np.zeros(decomp.n_forecasters)
        w[k] = 1.0
        mse_k = float(composite_losses(decomp, w).mean())
        assert mse_k == pytest.approx(2.0 * (1.0 - float(gbar[k])), abs=_IDENTITY_TOL)
        # The same quantity shows up as V_k in dilution accounting.
        acc = dilution_accounting(decomp, (k,), tuple(j for j in range(3) if j != k)[:1])
        assert acc.v_pool == pytest.approx(mse_k, abs=_IDENTITY_TOL)


def test_risk_decomposition_prop2_prop3_prop6_exact() -> None:
    """R = 1 - 2g + q; R = (1 - rho^2) + q(1 - c)^2; R = (1-g)^2 + w'K_pool w."""
    decomp = _graded_decomp((0.2, 0.1, 0.05, 0.0), n_dates=100)
    w = np.array([0.5, 0.3, 0.15, 0.05])
    rd = risk_decomposition(decomp, w)
    risk, g, q = rd["risk"], rd["g_w"], rd["q_w"]
    assert risk == pytest.approx(1.0 - 2.0 * g + q, abs=_IDENTITY_TOL)
    assert risk == pytest.approx(
        rd["scale_free_risk"] + rd["scale_mismatch_penalty"], abs=_IDENTITY_TOL
    )
    assert risk == pytest.approx(
        rd["alignment_term_pooled"] + rd["orthogonal_term_pooled"], abs=_IDENTITY_TOL
    )
    assert risk == pytest.approx(
        rd["alignment_term_within"] + rd["orthogonal_term_within"], abs=_IDENTITY_TOL
    )
    # K_pool = K_bar + Cov_a(gamma_t) (Proposition 6 aggregation).
    gamma_centered = decomp.gamma - decomp.gamma.mean(axis=0)
    cov_gamma = np.einsum("ti,tj->ij", gamma_centered, gamma_centered) / decomp.n_dates
    k_pool_manual = decomp.mean_orthogonal_cov + cov_gamma
    assert np.abs(decomp.pooled_orthogonal_cov - k_pool_manual).max() < _IDENTITY_TOL


def test_prop5_deviation_correlation_closed_form() -> None:
    """rho^e_ij = (1 + rho^s_ij - gamma_i - gamma_j) / (2 sqrt((1-g_i)(1-g_j)))."""
    decomp = _graded_decomp((0.2, 0.1, 0.0), n_dates=60)
    g = decomp.gamma
    denom = 2.0 * np.sqrt((1.0 - g[:, :, None]) * (1.0 - g[:, None, :]))
    manual = (1.0 + decomp.forecast_corr - g[:, :, None] - g[:, None, :]) / denom
    n = decomp.n_forecasters
    off = ~np.eye(n, dtype=bool)
    assert np.abs(decomp.deviation_corr[:, off] - manual[:, off]).max() < _IDENTITY_TOL


def test_deviation_corr_affine_at_zero_alignment() -> None:
    """At zero alignment rho^e = (1 + rho^s) / 2: error corr mirrors forecast corr."""
    decomp = _pure_noise_decomp(seed=21, n_dates=160, n_f=4)
    n = decomp.n_forecasters
    off = ~np.eye(n, dtype=bool)
    rho_s = decomp.mean_forecast_corr[off]
    rho_e = decomp.mean_deviation_corr[off]
    assert np.abs(rho_e - (1.0 + rho_s) / 2.0).max() < 0.02  # O(gamma_bar) noise
    assert (rho_e > 0.0).all()  # the translation compresses to (0, 1)


def test_prop8_admission_algebra_closed_form() -> None:
    """V_{P u A} = [n^2 V_P + m^2 V_A + 2nm C^e_PA] / (n+m)^2; Lambda^EW < 1 iff Delta < 0."""
    decomp = _graded_decomp((0.15, 0.08, 0.0, 0.0, 0.0), n_dates=120)
    acc = dilution_accounting(decomp, (0, 1), (2, 3))
    n, m_batch = acc.n_pool, acc.n_batch
    closed = (
        n * n * acc.v_pool + m_batch * m_batch * acc.v_batch + 2.0 * n * m_batch * acc.deviation_cov
    ) / ((n + m_batch) ** 2)
    assert acc.v_union == pytest.approx(closed, abs=_IDENTITY_TOL)
    assert acc.delta == pytest.approx(acc.v_union - acc.v_pool, abs=_IDENTITY_TOL)
    single = dilution_accounting(decomp, (0, 1, 2), (3,))
    closed_single = (
        single.v_batch + 2.0 * 3 * single.deviation_cov - (2.0 * 3 + 1.0) * single.v_pool
    ) / (4.0**2)
    assert single.delta == pytest.approx(closed_single, abs=_IDENTITY_TOL)
    assert (single.lambda_ew < 1.0) == (single.delta < 0.0)


def test_prop9_dilution_split_and_bound() -> None:
    """Delta = Delta^sf + Delta^scale; -Delta^sf <= (m g_bar_A)^2/(n^2 q_P)."""
    decomp = _graded_decomp((0.12, 0.08, 0.0, 0.0), n_dates=120)
    acc = dilution_accounting(decomp, (0, 1), (2,))
    assert acc.delta == pytest.approx(
        acc.delta_scale_free + acc.delta_scale_mismatch, abs=_IDENTITY_TOL
    )
    # Construct the bound's side conditions exactly: a zero-alignment
    # orthonormal pool (g_bar_P = 0) and a batch target-orthogonal to it
    # (q_PA = 0). Then -Delta^sf = (m g_bar_A)^2 / (n^2 q_P + m^2 q_A), strictly
    # below the bound (m g_bar_A)^2 / (n^2 q_P): admission into a large pool is
    # unresponsive to aligned candidates.
    rng = np.random.default_rng(31)
    m_units = 30
    y0 = rng.normal(size=m_units)
    y0 -= y0.mean()
    y0 /= math.sqrt(float(y0 @ y0) / m_units)
    basis = [y0]

    def _orth() -> np.ndarray:
        v = rng.normal(size=m_units)
        v -= v.mean()
        for b in basis:
            v = v - b * (float(b @ v) / m_units)
        v -= v.mean()
        v /= math.sqrt(float(v @ v) / m_units)
        basis.append(v)
        return v

    pool = [_orth(), _orth()]  # zero-alignment orthonormal pool: g_bar_P = 0
    a = 0.35
    batch = [a * y0 + math.sqrt(1.0 - a * a) * _orth() for _ in range(2)]
    scores1 = np.stack(pool + batch)[None, :, :]
    decomp1 = alignment_decomposition(scores1, y0[None, :], min_support=MIN_SUPPORT)
    acc1 = dilution_accounting(decomp1, (0, 1), (2, 3))
    cbar = decomp1.mean_forecast_corr
    n, m_batch = acc1.n_pool, acc1.n_batch
    g_bar_a = float(decomp1.mean_gamma[[2, 3]].mean())
    q_p = float(np.mean(cbar[np.ix_([0, 1], [0, 1])]))
    q_a = float(np.mean(cbar[np.ix_([2, 3], [2, 3])]))
    q_pa = float(np.mean(cbar[np.ix_([0, 1], [2, 3])]))
    assert abs(q_pa) < 1e-10 and abs(float(decomp1.mean_gamma[[0, 1]].mean())) < 1e-10
    assert -acc1.delta_scale_free == pytest.approx(
        (m_batch * g_bar_a) ** 2 / (n * n * q_p + m_batch * m_batch * q_a),
        rel=1e-8,
        abs=1e-10,
    )
    assert -acc1.delta_scale_free < acc1.scale_free_bound + 1e-12


def test_corollary1_margin_and_no_information_benchmark() -> None:
    """R(0) = 1; R(w) - 1 = -2 margin; margin sign separates aligned vs diluted pools."""
    decomp = _graded_decomp((0.3, 0.2, 0.0, 0.0), n_dates=120)
    zero = np.zeros(decomp.n_forecasters)
    assert np.abs(composite_losses(decomp, zero) - NO_INFORMATION_RISK).max() < _IDENTITY_TOL
    rd = risk_decomposition(decomp, equal_weights(4))
    assert rd["risk"] - 1.0 == pytest.approx(
        -2.0 * rd["margin_vs_no_information"], abs=_IDENTITY_TOL
    )
    # Strong alignment (a > 2 - sqrt(3) for an equal-weight pair) beats the
    # no-information forecast; diluting it with a correlated zero-skill
    # cluster flips the margin negative (the paper's weak-alignment regime).
    hi = _graded_decomp((0.6, 0.5), seed=43, n_dates=200)
    rd_aligned = risk_decomposition(hi, equal_weights(2))
    assert rd_aligned["margin_vs_no_information"] > 0.0
    assert rd_aligned["risk"] < NO_INFORMATION_RISK
    rng = np.random.default_rng(45)
    t, m = 300, 30
    y_d = rng.normal(size=(t, m))
    y_std = _zscore_rows(y_d)
    z_d = rng.normal(size=(t, m))
    cols_d = [
        0.6 * y_std + math.sqrt(0.64) * rng.normal(size=(t, m)),
        0.5 * y_std + math.sqrt(0.75) * rng.normal(size=(t, m)),
    ]
    for _j in range(6):
        cols_d.append(math.sqrt(0.4) * z_d + math.sqrt(0.6) * rng.normal(size=(t, m)))
    diluted = alignment_decomposition(np.stack(cols_d, axis=1), y_d)
    rd_full = risk_decomposition(diluted, equal_weights(8))
    assert rd_full["margin_vs_no_information"] < 0.0
    assert rd_full["risk"] > NO_INFORMATION_RISK


def test_attainable_risk_bounds_equal_weight_risks() -> None:
    """Corollary 2 bound <= every realized composite risk, in [0, 1]."""
    decomp = _graded_decomp((0.25, 0.15, 0.05), n_dates=150)
    att = attainable_risk(decomp)
    bound = float(att["attainable_risk"])
    assert 0.0 <= bound <= 1.0
    for w in (equal_weights(3), np.array([1.0, 0.0, 0.0]), np.array([0.2, 0.5, 0.3])):
        assert relative_score_risk(decomp, w) >= bound - 1e-9


# ---------------------------------------------------------------------------
# Planted skill ordering recovered + deterministic (spec items 2, 4)
# ---------------------------------------------------------------------------


def test_planted_alignment_ordering_recovered() -> None:
    """Graded planted alignments are recovered in the right order by gamma_bar."""
    align = (0.30, 0.20, 0.10, 0.04, 0.0, 0.0)
    decomp = _graded_decomp(align, n_dates=900, n_units=40)
    gbar = decomp.mean_gamma
    order = np.argsort(-gbar)
    # The four skilled forecasters occupy the top-4 in planted order; the two
    # zero-alignment forecasters trail.
    assert list(order[:4]) == [0, 1, 2, 3]
    assert set(order[4:]) == {4, 5}
    assert gbar[0] > gbar[1] > gbar[2] > gbar[3] > gbar[4]
    assert gbar[3] > gbar[5]


def test_mean_gamma_recovers_planted_levels() -> None:
    """Estimated alignment is within Monte-Carlo error of the planted level."""
    align = (0.25, 0.12, 0.0)
    decomp = _graded_decomp(align, seed=41, n_dates=1200, n_units=40)
    gbar = decomp.mean_gamma
    for k, a in enumerate(align):
        assert abs(float(gbar[k]) - a) < 0.03


def test_ordering_deterministic_across_seeds() -> None:
    """The skill ordering is stable across independent SYNTHETIC draws."""
    align = (0.28, 0.18, 0.08, 0.0)
    orders = set()
    for seed in range(9):
        decomp = _graded_decomp(align, seed=200 + seed, n_dates=600, n_units=30)
        orders.add(tuple(np.argsort(-decomp.mean_gamma)))
    assert orders == {(0, 1, 2, 3)}


def test_zero_skill_panel_alignment_near_zero() -> None:
    """With no planted skill, gamma_bar concentrates at zero for every forecaster."""
    decomp = _pure_noise_decomp(seed=23, n_dates=400, n_f=5)
    assert np.abs(decomp.mean_gamma).max() < 0.06
    # Every individual MSE is statistically indistinguishable from 2(1 - 0) = 2.
    for k in range(decomp.n_forecasters):
        w = np.zeros(decomp.n_forecasters)
        w[k] = 1.0
        assert relative_score_risk(decomp, w) == pytest.approx(2.0, abs=0.15)


# ---------------------------------------------------------------------------
# Cautious selection: no-rank verdict under no skill (spec item 3, seeded MC)
# ---------------------------------------------------------------------------


def test_no_skill_selection_refuses_to_grow_past_anchor() -> None:
    """Seeded MC: under zero planted skill the scale-free rule refuses to rank.

    On the scale-free basis the decision statistic is Delta^sf (genuine
    improvement only): with no planted skill every rep's pool stays at the
    argmax-alignment anchor — a no-rank verdict rather than a forced ordering.
    Paper Remark 1 bounds the conditional admission probability by alpha/2;
    empirically it is far below the generous bound asserted here. The
    equal-weight basis shows the documented foil: it admits purely on the
    dilution term (Delta ~= Delta^scale < 0 shrinks the composite's norm).
    """
    n_reps, admissions, undecided_seen, ew_admissions = 40, 0, 0, 0
    for rep in range(n_reps):
        decomp = _pure_noise_decomp(seed=1000 + rep, n_dates=300, n_f=6)
        res = cautious_selection(decomp, basis="scale_free", seed=_SEED + rep, n_draws=1000)
        if len(res.selected) > 1:
            admissions += 1
        else:
            # A refusal: the pool is just the argmax-alignment anchor.
            assert res.selected == (res.start,)
        for step in res.steps:
            assert set(step.decision) <= {"admit", "reject", "undecided"}
            if "undecided" in step.decision:
                undecided_seen += 1
                break
        ew = cautious_selection(decomp, basis="equal_weight", seed=_SEED + rep, n_draws=1000)
        ew_admissions += int(len(ew.selected) > 1)
    assert admissions / n_reps <= 0.10, f"admission rate {admissions / n_reps} too high"
    assert undecided_seen / n_reps >= 0.80  # 'undecided' is a first-class verdict
    # Dilution foil: raw equal-weight admission rewards zero-skill shrinkage.
    assert ew_admissions / n_reps >= 0.5


def test_three_way_decisions_recompute_from_bands() -> None:
    """Each step's admit/reject/undecided labels follow the documented inequalities."""
    decomp = _graded_decomp((0.25, 0.05, 0.0, 0.0, 0.0), n_dates=400, n_diluters=4)
    res = cautious_selection(decomp, basis="scale_free", seed=_SEED + 3, n_draws=1500)
    assert res.steps
    for step in res.steps:
        for pos in range(len(step.candidates)):
            d_hat, se, c = float(step.delta_hat[pos]), float(step.se[pos]), step.crit_value
            admit = d_hat + c * se < -res.delta
            reject = d_hat - c * se > 0.0
            expected = "admit" if admit else ("reject" if reject else "undecided")
            assert step.decision[pos] == expected
        if step.admitted is not None:
            # The admitted candidate is the smallest delta_hat among the cleared.
            cleared = [p for p in range(len(step.candidates)) if step.decision[p] == "admit"]
            best = min(cleared, key=lambda p: float(step.delta_hat[p]))
            assert step.candidates[best] == step.admitted
        # Reported minimum detectable effect = delta + (c + z_0.8) * SE.
        z80 = float(sstats.norm.ppf(0.8))
        expected_mde = res.delta + (step.crit_value + z80) * step.se
        assert np.abs(step.min_detectable - expected_mde).max() < 1e-10


def test_scale_free_selection_skips_planted_diluters() -> None:
    """On a graded world the scale-free basis admits aligned forecasters only."""
    align = (0.22, 0.15, 0.10)
    decomp = _graded_decomp(align, n_dates=1200, n_units=40, n_diluters=8, diluter_corr=0.35)
    res = cautious_selection(decomp, basis="scale_free", seed=_SEED + 4, n_draws=2000)
    assert set(res.selected) <= set(range(len(align)))
    assert len(res.selected) >= 2  # the clearly-skilled forecasters are admitted
    for step in res.steps:
        for pos, j in enumerate(step.candidates):
            if j >= len(align):
                assert step.decision[pos] != "admit"


def test_selection_preserves_history_scales_via_calibrated_losses() -> None:
    """The scale-free basis decision statistic aggregates to Delta^sf exactly."""
    decomp = _graded_decomp((0.18, 0.06, 0.0, 0.0), n_dates=200)
    pool, cand = (0,), 1
    n = len(pool)
    w_p = np.zeros(decomp.n_forecasters)
    w_p[list(pool)] = equal_weights(n)
    w_u = np.zeros(decomp.n_forecasters)
    w_u[list(pool) + [cand]] = 1.0 / (n + 1)
    # Mean calibrated loss difference equals rho_P^2 - rho_{P u {j}}^2.
    d = calibrated_composite_losses(decomp, w_u) - calibrated_composite_losses(decomp, w_p)
    acc = dilution_accounting(decomp, pool, (cand,))
    assert float(d.mean()) == pytest.approx(acc.delta_scale_free, abs=1e-8)


# ---------------------------------------------------------------------------
# Paired block-bootstrap CI on alignment differences (spec item 3 core piece)
# ---------------------------------------------------------------------------


def test_alignment_diff_ci_coverage_under_null() -> None:
    """Equal-skill pair: the paired block-bootstrap CI covers zero at ~nominal."""
    covered = 0
    n_reps = 50
    for rep in range(n_reps):
        decomp = _graded_decomp(
            (0.10, 0.10, 0.0), seed=3000 + rep, n_dates=400, n_units=30, phi=0.6
        )
        lo, hi, _ = _paired_alignment_diff_ci(decomp, 0, 1, alpha=0.10, n_boot=400, seed=rep)
        covered += int(lo <= 0.0 <= hi)
    assert covered / n_reps >= 0.76, f"coverage {covered / n_reps} below nominal band"


def test_alignment_diff_ci_detects_planted_gap() -> None:
    """A planted 0.12 alignment gap is detected at high rate under autocorrelation."""
    detected = 0
    n_reps = 40
    for rep in range(n_reps):
        decomp = _graded_decomp(
            (0.16, 0.04, 0.0), seed=4000 + rep, n_dates=500, n_units=30, phi=0.6
        )
        lo, hi, _ = _paired_alignment_diff_ci(decomp, 0, 1, alpha=0.10, n_boot=400, seed=rep)
        detected += int(lo > 0.0)
    assert detected / n_reps >= 0.85


def test_cautious_pair_verdict_refuses_under_equal_skill() -> None:
    """The refusal verdict 'no_significant_difference' fires at ~nominal rate."""
    refused = 0
    n_reps = 40
    for rep in range(n_reps):
        decomp = _graded_decomp((0.08, 0.08), seed=5000 + rep, n_dates=350, n_units=30)
        verdict = _cautious_pair_verdict(decomp, 0, 1, alpha=0.10, n_boot=300, seed=rep)
        refused += int(verdict == "no_significant_difference")
    assert refused / n_reps >= 0.76


def test_cautious_pair_verdict_ranks_planted_gap() -> None:
    """With a clear planted gap the verdict ranks instead of refusing."""
    decomp = _graded_decomp((0.28, 0.05, 0.0), n_dates=500, n_units=40)
    assert _cautious_pair_verdict(decomp, 0, 1, n_boot=400, seed=1) == "higher_alignment"
    assert _cautious_pair_verdict(decomp, 1, 0, n_boot=400, seed=1) == "lower_alignment"
    # A diluter versus a skilled forecaster also ranks.
    decomp2 = _graded_decomp((0.25, 0.0), n_dates=500, n_units=40)
    assert _cautious_pair_verdict(decomp2, 0, 1, n_boot=400, seed=2) == "higher_alignment"


def test_alignment_diff_ci_width_shrinks_with_dates() -> None:
    """More dates -> tighter CI on the same planted difference."""
    align = (0.15, 0.15)
    lo_s, hi_s, _ = _paired_alignment_diff_ci(
        _graded_decomp(align, seed=61, n_dates=200), 0, 1, n_boot=400, seed=9
    )
    lo_l, hi_l, _ = _paired_alignment_diff_ci(
        _graded_decomp(align, seed=61, n_dates=1200), 0, 1, n_boot=400, seed=9
    )
    assert (hi_l - lo_l) < (hi_s - lo_s)


# ---------------------------------------------------------------------------
# Fail-closed edges (spec item 5)
# ---------------------------------------------------------------------------


def test_alignment_decomposition_shape_and_finiteness_edges() -> None:
    scores, y = _graded_panel(seed=71, n_dates=40, n_units=30, alignments=(0.1, 0.0))
    with pytest.raises(ValueError, match="T x N x M"):
        alignment_decomposition(scores[:, 0, :], y)
    with pytest.raises(ValueError, match="T x M"):
        alignment_decomposition(scores, y[:, :, None])
    with pytest.raises(ValueError, match="shape must match"):
        alignment_decomposition(scores, y[:30])
    bad = scores.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        alignment_decomposition(bad, y)
    bad_y = y.copy()
    bad_y[1, 2] = np.inf
    with pytest.raises(ValueError, match="finite"):
        alignment_decomposition(scores, bad_y)
    with pytest.raises(ValueError, match="common support"):
        alignment_decomposition(scores[:, :, :10], y[:, :10])
    with pytest.raises(ValueError, match="at least one date"):
        alignment_decomposition(scores[:0], y[:0])


def test_alignment_decomposition_all_degenerate_and_perfect_alignment() -> None:
    """All-degenerate dispersion raises; a |gamma| = 1 forecaster raises."""
    scores = np.zeros((40, 2, 30))
    y = np.ones((40, 30))
    with pytest.raises(ValueError, match="no valid dates"):
        alignment_decomposition(scores, y)
    # Perfectly aligned forecaster: deviation correlation undefined (Prop. 5).
    scores2, y2 = _graded_panel(seed=72, n_dates=40, n_units=30, alignments=(0.2, 0.0))
    z = _zscore_rows(y2)
    scores2[:, 1, :] = z  # s_1 = y exactly
    with pytest.raises(ValueError, match="perfect|aligned|deviation"):
        alignment_decomposition(scores2, y2)


def test_weight_and_index_validation_edges() -> None:
    decomp = _graded_decomp((0.2, 0.1, 0.0), n_dates=60)
    with pytest.raises(ValueError, match="length"):
        composite_scores(decomp, np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        composite_scores(decomp, np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError, match="disjoint"):
        dilution_accounting(decomp, (0, 1), (1, 2))
    with pytest.raises(ValueError, match="nonempty"):
        dilution_accounting(decomp, (0,), ())
    with pytest.raises(ValueError, match="repeat"):
        dilution_accounting(decomp, (0, 0), (2,))
    with pytest.raises(ValueError, match="out of range"):
        dilution_accounting(decomp, (0,), (7,))
    with pytest.raises(ValueError, match="integer"):
        dilution_accounting(decomp, (True,), (2,))  # bools are not indices


def test_cautious_selection_parameter_edges() -> None:
    decomp = _graded_decomp((0.2, 0.1, 0.0), n_dates=80)
    with pytest.raises(ValueError, match="basis"):
        cautious_selection(decomp, basis="bad")
    with pytest.raises(ValueError, match="decision_rule"):
        cautious_selection(decomp, decision_rule="bad")
    with pytest.raises(ValueError, match="alpha"):
        cautious_selection(decomp, alpha=0.6)
    with pytest.raises(ValueError, match="alpha"):
        cautious_selection(decomp, alpha=0.0)
    with pytest.raises(ValueError, match="delta"):
        cautious_selection(decomp, delta=-0.1)
    with pytest.raises(ValueError, match="n_draws"):
        cautious_selection(decomp, n_draws=50)
    with pytest.raises(ValueError, match="integer"):
        cautious_selection(decomp, n_draws=True)
    with pytest.raises(ValueError, match="nw_lag"):
        cautious_selection(decomp, nw_lag=10_000)
    with pytest.raises(ValueError, match="eigen_rel_tol"):
        cautious_selection(decomp, eigen_rel_tol=1.5)
    with pytest.raises(ValueError, match="integer"):
        cautious_selection(decomp, max_steps=False)


def test_cautious_selection_rejects_degenerate_loss_differences() -> None:
    """Identical forecasters give a zero-variance difference series -> raise."""
    rng = np.random.default_rng(77)
    t, m = 80, 30
    y = rng.normal(size=(t, m))
    y_std = _zscore_rows(y)
    s = 0.1 * y_std + math.sqrt(0.99) * rng.normal(size=(t, m))
    scores = np.stack([s, s, s], axis=1)  # three identical forecasters
    decomp = alignment_decomposition(scores, y)
    with pytest.raises(ValueError, match="degenerate loss-difference"):
        cautious_selection(decomp, seed=7)


def test_planted_panel_parameter_edges() -> None:
    with pytest.raises(ValueError, match="n_dates"):
        planted_panel(n_dates=10)
    with pytest.raises(ValueError, match="n_units"):
        planted_panel(n_units=5)
    with pytest.raises(ValueError, match="alignment"):
        planted_panel(alignment=1.2)
    with pytest.raises(ValueError, match="cluster_corr"):
        planted_panel(cluster_corr=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        planted_panel(seed=-1)


def test_mirror_and_calibration_edges() -> None:
    one = _graded_decomp((0.2,), n_dates=60)
    with pytest.raises(ValueError, match="at least two"):
        error_correlation_mirror(one)
    decomp = _graded_decomp((0.2, 0.1), n_dates=60)
    with pytest.raises(ValueError, match="squared norm"):
        calibrated_composite_losses(decomp, np.zeros(2))  # zero composite


# ---------------------------------------------------------------------------
# Determinism + honesty (spec items 4, 6)
# ---------------------------------------------------------------------------


def test_decomposition_bit_identical() -> None:
    d1 = _graded_decomp((0.2, 0.1, 0.0), seed=81, n_dates=80)
    d2 = _graded_decomp((0.2, 0.1, 0.0), seed=81, n_dates=80)
    for f in ("gamma", "scores", "aligned", "orthogonal", "forecast_corr", "orthogonal_cov"):
        assert np.array_equal(getattr(d1, f), getattr(d2, f))
    assert np.array_equal(d1.valid_mask, d2.valid_mask)


def test_cautious_selection_bit_identical() -> None:
    decomp = _graded_decomp((0.2, 0.1, 0.05, 0.0, 0.0), seed=83, n_dates=300)
    r1 = cautious_selection(decomp, seed=5, n_draws=800)
    r2 = cautious_selection(decomp, seed=5, n_draws=800)
    assert r1.selected == r2.selected
    assert r1.final_risk == r2.final_risk
    for s1, s2 in zip(r1.steps, r2.steps, strict=True):
        assert np.array_equal(s1.delta_hat, s2.delta_hat)
        assert np.array_equal(s1.se, s2.se)
        assert s1.crit_value == s2.crit_value
        assert s1.decision == s2.decision
        assert np.array_equal(s1.min_detectable, s2.min_detectable)


def test_planted_panel_and_bootstrap_ci_deterministic() -> None:
    s1, y1 = planted_panel(seed=91, n_dates=100, n_units=30, n_aligned=3, n_diluters=3)
    s2, y2 = planted_panel(seed=91, n_dates=100, n_units=30, n_aligned=3, n_diluters=3)
    assert np.array_equal(s1, s2) and np.array_equal(y1, y2)
    s3, _ = planted_panel(seed=92, n_dates=100, n_units=30, n_aligned=3, n_diluters=3)
    assert not np.array_equal(s1, s3)
    decomp = _graded_decomp((0.2, 0.05), seed=95, n_dates=200)
    ci1 = _paired_alignment_diff_ci(decomp, 0, 1, n_boot=200, seed=3)
    ci2 = _paired_alignment_diff_ci(decomp, 0, 1, n_boot=200, seed=3)
    assert ci1 == ci2


def test_bench_blob_proper_scores_only_and_deterministic() -> None:
    """The bench blob carries proper diagnostics only — no headline ratios."""
    b1 = forecast_selection_benchmarks(seed=_SEED, fast=True)
    b2 = forecast_selection_benchmarks(seed=_SEED, fast=True)
    assert b1 == b2
    assert family_blob_forbidden_metrics_absent(b1)
    # Planted diluters are flagged by equal-weight accounting (delta < 0 with
    # no scale-free improvement) yet the scale-free rule admits none of them.
    assert b1["diluter_rewarded_share"] > 0.5
    assert b1["three_way_sf_diluters_admitted"] == 0.0
    assert b1["three_way_sf_removal_fraction"] > 0.5
